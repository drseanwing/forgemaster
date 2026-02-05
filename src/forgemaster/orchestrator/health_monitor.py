"""Session health monitoring service for Forgemaster.

This module provides continuous health monitoring of agent sessions, detecting
idle or stuck sessions and implementing automated recovery through session killing
and task retry scheduling. The health monitor runs as a background service that
periodically checks all active sessions against configured timeout thresholds.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable
from uuid import UUID

import structlog

from forgemaster.config import AgentConfig
from forgemaster.database.models.session import SessionStatus
from forgemaster.database.models.task import TaskStatus
from forgemaster.database.queries.session import end_session, get_idle_sessions
from forgemaster.database.queries.task import get_task, increment_retry_count, update_task_status

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from forgemaster.agents.session import AgentSessionManager

logger = structlog.get_logger(__name__)


class HealthMonitor:
    """Continuous health monitoring service for agent sessions.

    The HealthMonitor runs a background monitoring loop that periodically checks
    all active sessions for health issues such as:
    - Idle timeout: Sessions with no activity beyond idle_timeout_seconds
    - Stuck timeout: Sessions running longer than session_timeout_seconds

    When an unhealthy session is detected, the monitor:
    1. Kills the session and updates its status
    2. Updates the associated task status
    3. Schedules a retry if within retry limits, or marks as FAILED

    Attributes:
        config: Agent configuration with timeout thresholds
        session_factory: Async session factory for database operations
        session_manager: AgentSessionManager for session lifecycle operations
        monitor_interval: Seconds between health checks (default: 30)
    """

    def __init__(
        self,
        config: AgentConfig,
        session_factory: async_sessionmaker,
        session_manager: AgentSessionManager,
        monitor_interval: int = 30,
    ):
        """Initialize health monitor.

        Args:
            config: Agent configuration for timeouts and thresholds
            session_factory: SQLAlchemy async session factory
            session_manager: Session manager for ending sessions
            monitor_interval: Seconds between health checks (default: 30)
        """
        self.config = config
        self.session_factory = session_factory
        self.session_manager = session_manager
        self.monitor_interval = monitor_interval
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._logger = structlog.get_logger(__name__)

    async def start(self) -> None:
        """Start the health monitoring loop.

        Creates and starts the background monitoring task. If already running,
        this is a no-op.
        """
        if self._running:
            self._logger.warning("health_monitor_already_running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._logger.info(
            "health_monitor_started",
            interval_seconds=self.monitor_interval,
        )

    async def stop(self) -> None:
        """Gracefully stop the health monitoring loop.

        Cancels the background task and waits for clean shutdown.
        """
        if not self._running:
            self._logger.warning("health_monitor_not_running")
            return

        self._running = False

        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self._logger.info("health_monitor_stopped")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop that periodically checks session health.

        This loop runs until stop() is called, checking all active sessions
        at each interval and taking corrective action as needed.
        """
        self._logger.info("health_monitor_loop_started")

        while self._running:
            try:
                await self._check_idle_sessions()
                await asyncio.sleep(self.monitor_interval)
            except asyncio.CancelledError:
                self._logger.info("health_monitor_loop_cancelled")
                break
            except Exception as e:
                self._logger.error(
                    "health_monitor_loop_error",
                    error=str(e),
                    exc_info=True,
                )
                # Continue monitoring despite errors
                await asyncio.sleep(self.monitor_interval)

    async def _check_idle_sessions(self) -> list[UUID]:
        """Check for idle sessions and kill them if beyond timeout threshold.

        Queries the database for sessions that have been idle longer than
        config.idle_timeout_seconds, then kills each one and schedules retries.

        Returns:
            List of session UUIDs that were killed
        """
        async with self.session_factory() as db_session:
            idle_sessions = await get_idle_sessions(
                db_session,
                idle_threshold_seconds=self.config.idle_timeout_seconds,
            )

        killed_session_ids = []

        for session in idle_sessions:
            self._logger.warning(
                "idle_session_detected",
                session_id=str(session.id),
                task_id=str(session.task_id),
                idle_seconds=(
                    session.last_activity_at
                    and (
                        (
                            session.last_activity_at.timestamp()
                            if hasattr(session.last_activity_at, "timestamp")
                            else session.last_activity_at
                        )
                    )
                ),
            )

            try:
                await self._kill_session(
                    session.id,
                    reason=f"Idle timeout exceeded ({self.config.idle_timeout_seconds}s)",
                )
                killed_session_ids.append(session.id)

                # Schedule retry for the associated task
                if session.task_id:
                    await self._schedule_retry(session.task_id)

            except Exception as e:
                self._logger.error(
                    "failed_to_kill_idle_session",
                    session_id=str(session.id),
                    error=str(e),
                    exc_info=True,
                )

        if killed_session_ids:
            self._logger.info(
                "idle_sessions_killed",
                count=len(killed_session_ids),
                session_ids=[str(sid) for sid in killed_session_ids],
            )

        return killed_session_ids

    async def _kill_session(self, session_id: UUID, reason: str) -> None:
        """Kill a session and update its status.

        Ends the agent session via the session_manager, updates the database
        session status to 'killed', and logs the kill action with reason.

        Args:
            session_id: UUID of the session to kill
            reason: Human-readable reason for killing the session
        """
        self._logger.info(
            "killing_session",
            session_id=str(session_id),
            reason=reason,
        )

        # End the session via session manager (if it exists in memory)
        try:
            # Convert UUID to string for session manager lookup
            await self.session_manager.end_session(
                session_id=str(session_id),
                status="failed",
            )
        except ValueError:
            # Session not found in memory - may have already been cleaned up
            self._logger.debug(
                "session_not_in_manager",
                session_id=str(session_id),
            )

        # Update database session status to killed
        async with self.session_factory() as db_session:
            await end_session(
                db_session,
                session_id=session_id,
                status=SessionStatus.killed,
                error_message=reason,
            )

        self._logger.info(
            "session_killed",
            session_id=str(session_id),
            reason=reason,
        )

    async def _schedule_retry(self, task_id: UUID) -> None:
        """Schedule a retry for a task or mark it as failed if max retries exceeded.

        Increments the task's retry count and transitions it based on retry limits:
        - If retries < max_retries: transition task back to READY for retry
        - If retries >= max_retries: transition task to FAILED

        Args:
            task_id: UUID of the task to schedule retry for
        """
        async with self.session_factory() as db_session:
            # Get current task
            task = await get_task(db_session, task_id)
            if task is None:
                self._logger.error(
                    "task_not_found_for_retry",
                    task_id=str(task_id),
                )
                return

            # Increment retry count
            task = await increment_retry_count(db_session, task_id)

            # Determine new status based on retry limits
            if task.retry_count < task.max_retries:
                # Still have retries left - transition to READY
                new_status = TaskStatus.ready
                await update_task_status(db_session, task_id, new_status)

                self._logger.info(
                    "task_retry_scheduled",
                    task_id=str(task_id),
                    retry_count=task.retry_count,
                    max_retries=task.max_retries,
                )
            else:
                # Max retries exceeded - mark as FAILED
                new_status = TaskStatus.failed
                await update_task_status(db_session, task_id, new_status)

                self._logger.error(
                    "task_failed_max_retries",
                    task_id=str(task_id),
                    retry_count=task.retry_count,
                    max_retries=task.max_retries,
                )
