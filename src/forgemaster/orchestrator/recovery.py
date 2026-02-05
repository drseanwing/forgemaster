"""Crash recovery system for Forgemaster orchestrator.

Implements orphan session detection, session cleanup, task retry scheduling,
and startup recovery routines. When the orchestrator restarts after a crash,
the RecoveryManager detects orphaned sessions and tasks, cleans up stale
resources, and schedules retries for recoverable failures.

Components:
    - OrphanDetector: Identifies sessions that outlived their process.
    - SessionCleaner: Terminates orphan sessions and resets associated tasks.
    - RetryScheduler: Evaluates and schedules task retries with backoff.
    - RecoveryManager: Orchestrates the full recovery lifecycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import select

from forgemaster.database.models.file_lock import FileLock
from forgemaster.database.models.session import AgentSession, SessionStatus
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.database.queries.session import end_session
from forgemaster.database.queries.task import get_task, increment_retry_count, update_task_status

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from forgemaster.config import AgentConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrphanReason(str, Enum):
    """Reason a session has been classified as orphaned.

    Values:
        PROCESS_CRASH: The owning process terminated unexpectedly.
        SESSION_TIMEOUT: The session exceeded the maximum allowed duration.
        STALE_HEARTBEAT: The session's heartbeat stopped updating.
        UNKNOWN: The reason could not be determined.
    """

    PROCESS_CRASH = "process_crash"
    SESSION_TIMEOUT = "session_timeout"
    STALE_HEARTBEAT = "stale_heartbeat"
    UNKNOWN = "unknown"


class CleanupAction(str, Enum):
    """Actions taken during orphan session cleanup.

    Values:
        SESSION_TERMINATED: The orphan session was marked as failed/ended.
        TASK_RESET: The associated task was reset to READY for retry.
        WORKTREE_CLEANED: The session's worktree was cleaned up.
        LOCK_RELEASED: File locks held by the session were released.
    """

    SESSION_TERMINATED = "session_terminated"
    TASK_RESET = "task_reset"
    WORKTREE_CLEANED = "worktree_cleaned"
    LOCK_RELEASED = "lock_released"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class OrphanSession(BaseModel):
    """An orphaned agent session detected by the recovery system.

    Attributes:
        session_id: UUID string of the orphaned session.
        task_id: UUID string of the associated task.
        status: Current session status string.
        started_at: ISO-8601 timestamp when the session started, or None.
        last_activity: ISO-8601 timestamp of last heartbeat, or None.
        reason: Classification of why the session is orphaned.
    """

    session_id: str
    task_id: str
    status: str
    started_at: str | None = Field(default=None)
    last_activity: str | None = Field(default=None)
    reason: OrphanReason = Field(default=OrphanReason.UNKNOWN)


class CleanupResult(BaseModel):
    """Result of a single cleanup action during orphan recovery.

    Attributes:
        session_id: UUID string of the session being cleaned.
        task_id: UUID string of the associated task, if any.
        action: The cleanup action that was taken.
        success: Whether the action completed successfully.
        error: Error message if the action failed.
    """

    session_id: str
    task_id: str | None = Field(default=None)
    action: CleanupAction
    success: bool = Field(default=True)
    error: str | None = Field(default=None)


class RetryDecision(BaseModel):
    """Decision on whether to retry a failed task.

    Attributes:
        task_id: UUID string of the task being evaluated.
        should_retry: Whether the task should be retried.
        current_retries: Number of retries already attempted.
        max_retries: Maximum allowed retry attempts.
        delay_seconds: Exponential backoff delay before retry.
        reason: Human-readable explanation of the decision.
    """

    task_id: str
    should_retry: bool = Field(default=False)
    current_retries: int = Field(default=0)
    max_retries: int = Field(default=3)
    delay_seconds: float = Field(default=0.0)
    reason: str = Field(default="")


class RecoveryReport(BaseModel):
    """Summary report of a recovery run.

    Attributes:
        orphan_sessions_found: Number of orphan sessions detected.
        sessions_cleaned: Number of sessions successfully cleaned up.
        tasks_retried: Number of tasks scheduled for retry.
        tasks_failed: Number of tasks marked as permanently failed.
        cleanup_results: Detailed list of cleanup actions taken.
        retry_decisions: Detailed list of retry decisions made.
        started_at: ISO-8601 timestamp when recovery started.
        completed_at: ISO-8601 timestamp when recovery completed.
        duration_seconds: Total recovery duration in seconds.
    """

    orphan_sessions_found: int = Field(default=0)
    sessions_cleaned: int = Field(default=0)
    tasks_retried: int = Field(default=0)
    tasks_failed: int = Field(default=0)
    cleanup_results: list[CleanupResult] = Field(default_factory=list)
    retry_decisions: list[RetryDecision] = Field(default_factory=list)
    started_at: str = Field(default="")
    completed_at: str = Field(default="")
    duration_seconds: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Orphan Detector
# ---------------------------------------------------------------------------


class OrphanDetector:
    """Detects orphaned agent sessions that have outlived their process.

    Queries the database for sessions in active states (RUNNING, ASSIGNED)
    that have not received a heartbeat within the configured timeout period.

    Attributes:
        session_factory: Callable that produces async database sessions.
        config: Agent configuration with timeout thresholds.
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        config: AgentConfig,
    ) -> None:
        """Initialize the orphan detector.

        Args:
            session_factory: Callable returning new AsyncSession instances.
            config: Agent configuration with session_timeout_seconds and
                idle_timeout_seconds thresholds.
        """
        self.session_factory = session_factory
        self.config = config
        self._logger = logger.bind(component="OrphanDetector")

    async def detect_orphans(self) -> list[OrphanSession]:
        """Detect all orphaned agent sessions.

        Identifies sessions in two categories:
        1. RUNNING/active sessions with no heartbeat beyond session_timeout_seconds
           (classified as SESSION_TIMEOUT or STALE_HEARTBEAT).
        2. ASSIGNED/initialising sessions that were never started and are stale
           beyond idle_timeout_seconds (classified as PROCESS_CRASH).

        Returns:
            List of OrphanSession instances describing each orphan.
        """
        orphans: list[OrphanSession] = []
        now = datetime.now(timezone.utc)

        async with self.session_factory() as db_session:
            # 1. Detect running sessions that have timed out
            session_timeout_cutoff = now - timedelta(seconds=self.config.session_timeout_seconds)
            running_stmt = (
                select(AgentSession)
                .where(
                    AgentSession.status.in_(
                        [
                            SessionStatus.active,
                            SessionStatus.idle,
                        ]
                    )
                )
                .where(AgentSession.last_activity_at < session_timeout_cutoff)
            )
            running_result = await db_session.execute(running_stmt)
            timed_out_sessions = list(running_result.scalars().all())

            for session in timed_out_sessions:
                reason = OrphanReason.SESSION_TIMEOUT
                # If last_activity is much older than session start, likely stale heartbeat
                if session.started_at and session.last_activity_at:
                    activity_gap = (now - session.last_activity_at).total_seconds()
                    session_age = (now - session.started_at).total_seconds()
                    if activity_gap > session_age * 0.5:
                        reason = OrphanReason.STALE_HEARTBEAT

                orphans.append(
                    OrphanSession(
                        session_id=str(session.id),
                        task_id=str(session.task_id) if session.task_id else "",
                        status=session.status.value,
                        started_at=(session.started_at.isoformat() if session.started_at else None),
                        last_activity=(
                            session.last_activity_at.isoformat()
                            if session.last_activity_at
                            else None
                        ),
                        reason=reason,
                    )
                )

            # 2. Detect assigned/initialising sessions that were never started
            idle_timeout_cutoff = now - timedelta(seconds=self.config.idle_timeout_seconds)
            assigned_stmt = (
                select(AgentSession)
                .where(
                    AgentSession.status.in_(
                        [
                            SessionStatus.initialising,
                        ]
                    )
                )
                .where(AgentSession.last_activity_at < idle_timeout_cutoff)
            )
            assigned_result = await db_session.execute(assigned_stmt)
            stale_assigned = list(assigned_result.scalars().all())

            for session in stale_assigned:
                orphans.append(
                    OrphanSession(
                        session_id=str(session.id),
                        task_id=str(session.task_id) if session.task_id else "",
                        status=session.status.value,
                        started_at=(session.started_at.isoformat() if session.started_at else None),
                        last_activity=(
                            session.last_activity_at.isoformat()
                            if session.last_activity_at
                            else None
                        ),
                        reason=OrphanReason.PROCESS_CRASH,
                    )
                )

        self._logger.info(
            "orphan_detection_complete",
            orphan_count=len(orphans),
            timed_out=len(timed_out_sessions),
            stale_assigned=len(stale_assigned),
        )

        return orphans

    async def detect_orphaned_tasks(self) -> list[str]:
        """Detect tasks in RUNNING status with no corresponding active session.

        A task is considered orphaned if it is in RUNNING or ASSIGNED status
        but has no session in an active state (active, idle, initialising).

        Returns:
            List of task ID strings for orphaned tasks.
        """
        async with self.session_factory() as db_session:
            # Get all tasks in RUNNING or ASSIGNED status
            task_stmt = select(Task).where(
                Task.status.in_([TaskStatus.running, TaskStatus.assigned])
            )
            task_result = await db_session.execute(task_stmt)
            active_tasks = list(task_result.scalars().all())

            if not active_tasks:
                return []

            # Get all active sessions
            session_stmt = select(AgentSession).where(
                AgentSession.status.in_(
                    [
                        SessionStatus.active,
                        SessionStatus.idle,
                        SessionStatus.initialising,
                        SessionStatus.completing,
                    ]
                )
            )
            session_result = await db_session.execute(session_stmt)
            active_sessions = list(session_result.scalars().all())

            # Build set of task IDs that have active sessions
            task_ids_with_sessions = {
                session.task_id for session in active_sessions if session.task_id
            }

            # Find tasks without active sessions
            orphaned_task_ids = [
                str(task.id) for task in active_tasks if task.id not in task_ids_with_sessions
            ]

        self._logger.info(
            "orphaned_task_detection_complete",
            orphaned_count=len(orphaned_task_ids),
            total_active_tasks=len(active_tasks),
        )

        return orphaned_task_ids


# ---------------------------------------------------------------------------
# Session Cleaner
# ---------------------------------------------------------------------------


class SessionCleaner:
    """Cleans up orphaned sessions and their associated resources.

    Terminates orphan sessions, resets their associated tasks for retry
    or marks them as failed, and releases any file locks held by the session.

    Attributes:
        session_factory: Callable that produces async database sessions.
        config: Agent configuration with retry thresholds.
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        config: AgentConfig,
    ) -> None:
        """Initialize the session cleaner.

        Args:
            session_factory: Callable returning new AsyncSession instances.
            config: Agent configuration with max_retries threshold.
        """
        self.session_factory = session_factory
        self.config = config
        self._logger = logger.bind(component="SessionCleaner")

    async def cleanup_orphan(self, orphan: OrphanSession) -> list[CleanupResult]:
        """Clean up a single orphaned session.

        Performs the following cleanup steps:
        1. End the orphan session (mark as failed in database).
        2. Release any file locks held by the session's task.
        3. Reset the associated task to READY if retries remain, FAILED if exhausted.

        Args:
            orphan: The orphan session to clean up.

        Returns:
            List of CleanupResult instances describing each action taken.
        """
        results: list[CleanupResult] = []

        # Step 1: Terminate the session
        try:
            async with self.session_factory() as db_session:
                await end_session(
                    db_session,
                    session_id=_to_uuid(orphan.session_id),
                    status=SessionStatus.failed,
                    error_message=f"Orphan recovery: {orphan.reason.value}",
                )
            results.append(
                CleanupResult(
                    session_id=orphan.session_id,
                    task_id=orphan.task_id,
                    action=CleanupAction.SESSION_TERMINATED,
                    success=True,
                )
            )
            self._logger.info(
                "orphan_session_terminated",
                session_id=orphan.session_id,
                reason=orphan.reason.value,
            )
        except Exception as e:
            results.append(
                CleanupResult(
                    session_id=orphan.session_id,
                    task_id=orphan.task_id,
                    action=CleanupAction.SESSION_TERMINATED,
                    success=False,
                    error=str(e),
                )
            )
            self._logger.error(
                "orphan_session_termination_failed",
                session_id=orphan.session_id,
                error=str(e),
            )

        # Step 2: Release file locks for the associated task
        if orphan.task_id:
            try:
                released = await self._release_task_locks(orphan.task_id)
                if released > 0:
                    results.append(
                        CleanupResult(
                            session_id=orphan.session_id,
                            task_id=orphan.task_id,
                            action=CleanupAction.LOCK_RELEASED,
                            success=True,
                        )
                    )
            except Exception as e:
                results.append(
                    CleanupResult(
                        session_id=orphan.session_id,
                        task_id=orphan.task_id,
                        action=CleanupAction.LOCK_RELEASED,
                        success=False,
                        error=str(e),
                    )
                )
                self._logger.error(
                    "lock_release_failed",
                    session_id=orphan.session_id,
                    task_id=orphan.task_id,
                    error=str(e),
                )

        # Step 3: Reset the associated task
        if orphan.task_id:
            try:
                task_result = await self._reset_task(orphan.task_id)
                results.append(
                    CleanupResult(
                        session_id=orphan.session_id,
                        task_id=orphan.task_id,
                        action=CleanupAction.TASK_RESET,
                        success=True,
                    )
                )
                self._logger.info(
                    "orphan_task_reset",
                    task_id=orphan.task_id,
                    new_status=task_result,
                )
            except Exception as e:
                results.append(
                    CleanupResult(
                        session_id=orphan.session_id,
                        task_id=orphan.task_id,
                        action=CleanupAction.TASK_RESET,
                        success=False,
                        error=str(e),
                    )
                )
                self._logger.error(
                    "orphan_task_reset_failed",
                    task_id=orphan.task_id,
                    error=str(e),
                )

        return results

    async def cleanup_all_orphans(self, orphans: list[OrphanSession]) -> list[CleanupResult]:
        """Clean up all orphaned sessions.

        Iterates through each orphan and performs cleanup, collecting
        all results.

        Args:
            orphans: List of orphan sessions to clean up.

        Returns:
            Combined list of CleanupResult instances from all cleanups.
        """
        all_results: list[CleanupResult] = []

        for orphan in orphans:
            results = await self.cleanup_orphan(orphan)
            all_results.extend(results)

        self._logger.info(
            "all_orphans_cleaned",
            orphan_count=len(orphans),
            total_actions=len(all_results),
            successful=sum(1 for r in all_results if r.success),
            failed=sum(1 for r in all_results if not r.success),
        )

        return all_results

    async def _release_task_locks(self, task_id: str) -> int:
        """Release all active file locks held by a task.

        Args:
            task_id: UUID string of the task whose locks to release.

        Returns:
            Number of locks released.
        """
        from sqlalchemy import update as sa_update

        async with self.session_factory() as db_session:
            now = datetime.now(timezone.utc)
            stmt = (
                sa_update(FileLock)
                .where(
                    FileLock.task_id == uuid.UUID(task_id),
                    FileLock.released_at.is_(None),
                )
                .values(released_at=now)
            )
            result = await db_session.execute(stmt)
            await db_session.commit()

            count: int = result.rowcount  # type: ignore[assignment]

            if count > 0:
                self._logger.info(
                    "task_locks_released",
                    task_id=task_id,
                    count=count,
                )

            return count

    async def _reset_task(self, task_id: str) -> str:
        """Reset a task based on its retry count.

        If the task has retries remaining (retry_count < max_retries),
        it is transitioned to READY. Otherwise, it is marked as FAILED.

        Args:
            task_id: UUID string of the task to reset.

        Returns:
            The new task status as a string.
        """
        task_uuid = uuid.UUID(task_id)

        async with self.session_factory() as db_session:
            task = await get_task(db_session, task_uuid)
            if task is None:
                raise ValueError(f"Task {task_id} not found")

            if task.retry_count < task.max_retries:
                new_status = TaskStatus.ready
            else:
                new_status = TaskStatus.failed

            await update_task_status(db_session, task_uuid, new_status)

        return new_status.value


# ---------------------------------------------------------------------------
# Retry Scheduler
# ---------------------------------------------------------------------------


class RetryScheduler:
    """Evaluates and schedules task retries with exponential backoff.

    Uses the task's current retry count and the configured max_retries
    to determine whether a retry is appropriate, and calculates the
    backoff delay.

    Attributes:
        session_factory: Callable that produces async database sessions.
        config: Agent configuration with max_retries threshold.
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        config: AgentConfig,
        base_delay: float = 30.0,
        max_delay: float = 600.0,
    ) -> None:
        """Initialize the retry scheduler.

        Args:
            session_factory: Callable returning new AsyncSession instances.
            config: Agent configuration with max_retries.
            base_delay: Base delay for exponential backoff (default 30s).
            max_delay: Maximum delay cap (default 600s / 10 minutes).
        """
        self.session_factory = session_factory
        self.config = config
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._logger = logger.bind(component="RetryScheduler")

    async def evaluate_retry(self, task_id: str) -> RetryDecision:
        """Evaluate whether a task should be retried.

        Checks the task's current retry count against max_retries and
        calculates the exponential backoff delay if a retry is warranted.

        The delay formula is: min(base_delay * 2^retries, max_delay)

        Args:
            task_id: UUID string of the task to evaluate.

        Returns:
            RetryDecision with the evaluation result.
        """
        task_uuid = uuid.UUID(task_id)

        async with self.session_factory() as db_session:
            task = await get_task(db_session, task_uuid)
            if task is None:
                return RetryDecision(
                    task_id=task_id,
                    should_retry=False,
                    reason=f"Task {task_id} not found",
                )

            current_retries = task.retry_count
            max_retries = task.max_retries

            if current_retries >= max_retries:
                return RetryDecision(
                    task_id=task_id,
                    should_retry=False,
                    current_retries=current_retries,
                    max_retries=max_retries,
                    delay_seconds=0.0,
                    reason=(f"Max retries exhausted ({current_retries}/{max_retries})"),
                )

            delay = min(self.base_delay * (2**current_retries), self.max_delay)

            return RetryDecision(
                task_id=task_id,
                should_retry=True,
                current_retries=current_retries,
                max_retries=max_retries,
                delay_seconds=delay,
                reason=(
                    f"Retry {current_retries + 1}/{max_retries} " f"after {delay:.1f}s backoff"
                ),
            )

    async def schedule_retry(self, task_id: str) -> RetryDecision:
        """Evaluate and schedule a retry for a task.

        First evaluates whether a retry should occur. If yes, increments
        the retry count and transitions the task to READY status after
        the computed backoff delay. If not, marks the task as FAILED.

        Args:
            task_id: UUID string of the task to retry.

        Returns:
            RetryDecision with the scheduling result.
        """
        decision = await self.evaluate_retry(task_id)
        task_uuid = uuid.UUID(task_id)

        if decision.should_retry:
            # Apply backoff delay
            if decision.delay_seconds > 0:
                self._logger.info(
                    "retry_backoff_waiting",
                    task_id=task_id,
                    delay_seconds=decision.delay_seconds,
                )
                await asyncio.sleep(decision.delay_seconds)

            # Increment retry count and set to READY
            async with self.session_factory() as db_session:
                await increment_retry_count(db_session, task_uuid)
                await update_task_status(db_session, task_uuid, TaskStatus.ready)

            self._logger.info(
                "task_retry_scheduled",
                task_id=task_id,
                retry_count=decision.current_retries + 1,
                max_retries=decision.max_retries,
            )
        else:
            # Mark as failed - retries exhausted
            async with self.session_factory() as db_session:
                await update_task_status(db_session, task_uuid, TaskStatus.failed)

            self._logger.warning(
                "task_retries_exhausted",
                task_id=task_id,
                retry_count=decision.current_retries,
                max_retries=decision.max_retries,
            )

        return decision


# ---------------------------------------------------------------------------
# Recovery Manager
# ---------------------------------------------------------------------------


class RecoveryManager:
    """Main entry point for crash recovery operations.

    Orchestrates the full recovery lifecycle: orphan detection, session
    cleanup, orphaned task detection, retry scheduling, and periodic
    background recovery.

    Attributes:
        config: Agent configuration.
        session_factory: Callable that produces async database sessions.
        detector: OrphanDetector instance.
        cleaner: SessionCleaner instance.
        scheduler: RetryScheduler instance.
    """

    def __init__(
        self,
        config: AgentConfig,
        session_factory: Callable[[], AsyncSession],
    ) -> None:
        """Initialize the recovery manager.

        Args:
            config: Agent configuration with timeout and retry thresholds.
            session_factory: Callable returning new AsyncSession instances.
        """
        self.config = config
        self.session_factory = session_factory
        self.detector = OrphanDetector(session_factory, config)
        self.cleaner = SessionCleaner(session_factory, config)
        self.scheduler = RetryScheduler(session_factory, config)
        self._running = False
        self._periodic_task: asyncio.Task[None] | None = None
        self._logger = logger.bind(component="RecoveryManager")

    async def run_startup_recovery(self) -> RecoveryReport:
        """Execute the full startup recovery routine.

        Performs the following steps in order:
        1. Detect orphan sessions (timed-out, stale assigned).
        2. Clean up each orphan (terminate session, release locks, reset task).
        3. Detect orphaned tasks (running with no active session).
        4. Evaluate retry for each orphaned task.
        5. Schedule retries where appropriate.
        6. Log recovery summary.

        Returns:
            RecoveryReport summarising all recovery actions taken.
        """
        started_at = datetime.now(timezone.utc)
        self._logger.info("startup_recovery_started")

        report = RecoveryReport(started_at=started_at.isoformat())

        # Step 1: Detect orphan sessions
        orphans = await self.detector.detect_orphans()
        report.orphan_sessions_found = len(orphans)

        # Step 2: Clean up orphan sessions
        if orphans:
            cleanup_results = await self.cleaner.cleanup_all_orphans(orphans)
            report.cleanup_results = cleanup_results
            report.sessions_cleaned = sum(
                1
                for r in cleanup_results
                if r.action == CleanupAction.SESSION_TERMINATED and r.success
            )

        # Step 3: Detect orphaned tasks
        orphaned_task_ids = await self.detector.detect_orphaned_tasks()

        # Step 4 & 5: Evaluate and schedule retries for orphaned tasks
        for task_id in orphaned_task_ids:
            decision = await self.scheduler.schedule_retry(task_id)
            report.retry_decisions.append(decision)

            if decision.should_retry:
                report.tasks_retried += 1
            else:
                report.tasks_failed += 1

        # Step 6: Complete report
        completed_at = datetime.now(timezone.utc)
        report.completed_at = completed_at.isoformat()
        report.duration_seconds = (completed_at - started_at).total_seconds()

        self._logger.info(
            "startup_recovery_completed",
            orphan_sessions=report.orphan_sessions_found,
            sessions_cleaned=report.sessions_cleaned,
            tasks_retried=report.tasks_retried,
            tasks_failed=report.tasks_failed,
            duration_seconds=report.duration_seconds,
        )

        return report

    async def run_periodic_recovery(self, interval_seconds: float = 300.0) -> None:
        """Run recovery at regular intervals as a background task.

        Starts a loop that executes run_startup_recovery at the specified
        interval. Errors in individual recovery runs are logged but do not
        stop the loop.

        Args:
            interval_seconds: Seconds between recovery runs (default 300 = 5 min).
        """
        if self._running:
            self._logger.warning("periodic_recovery_already_running")
            return

        self._running = True
        self._logger.info(
            "periodic_recovery_started",
            interval_seconds=interval_seconds,
        )

        while self._running:
            try:
                await self.run_startup_recovery()
            except asyncio.CancelledError:
                self._logger.info("periodic_recovery_cancelled")
                break
            except Exception as e:
                self._logger.error(
                    "periodic_recovery_error",
                    error=str(e),
                    exc_info=True,
                )

            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                self._logger.info("periodic_recovery_sleep_cancelled")
                break

    async def stop(self) -> None:
        """Stop the periodic recovery background task.

        Cancels the background task and waits for clean shutdown.
        """
        if not self._running:
            self._logger.warning("periodic_recovery_not_running")
            return

        self._running = False

        if self._periodic_task is not None:
            self._periodic_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._periodic_task

        self._logger.info("periodic_recovery_stopped")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_uuid(value: str) -> uuid.UUID:
    """Convert a string to a UUID object.

    Args:
        value: String representation of a UUID.

    Returns:
        UUID object.
    """
    return uuid.UUID(value)
