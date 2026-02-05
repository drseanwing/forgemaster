"""Integration tests for crash recovery system.

Tests cover the full recovery lifecycle: orphan session detection, session
cleanup, task retry scheduling with exponential backoff, startup recovery,
periodic recovery, and edge cases. All database operations are mocked to
isolate the recovery logic from the database layer.

Test coverage:
- P6-006: Orphan detection (timed-out sessions, stale assigned, no orphans)
- P6-006: Orphaned task detection (running tasks with no session)
- P6-007: Session cleanup (single orphan, multiple orphans, cleanup failure)
- P6-007: Task reset (retries remaining, retries exhausted)
- P6-007: Lock release during cleanup
- P6-008: Retry evaluation (should retry, exhausted retries, delay calculation)
- P6-008: Retry scheduling (success, increment count, set to READY)
- P6-009: Full startup recovery (with orphans, with retries, clean startup)
- P6-009: Recovery report generation
- P6-009: Periodic recovery (start, stop, interval check)
- P6-010: Edge cases (empty database, concurrent recovery, partial failures)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import pytest

from forgemaster.config import AgentConfig
from forgemaster.database.models.session import SessionStatus
from forgemaster.database.models.task import TaskStatus
from forgemaster.orchestrator.recovery import (
    CleanupAction,
    CleanupResult,
    OrphanDetector,
    OrphanReason,
    OrphanSession,
    RecoveryManager,
    RecoveryReport,
    RetryDecision,
    RetryScheduler,
    SessionCleaner,
)

# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentSession:
    """Minimal fake of AgentSession ORM model for testing."""

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    task_id: uuid.UUID | None = field(default_factory=uuid.uuid4)
    status: SessionStatus = SessionStatus.active
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC) - timedelta(hours=2))
    last_activity_at: datetime = field(
        default_factory=lambda: datetime.now(UTC) - timedelta(hours=2)
    )
    model: str = "sonnet"
    worktree_path: str | None = None


@dataclass
class FakeTask:
    """Minimal fake of Task ORM model for testing."""

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    status: TaskStatus = TaskStatus.running
    retry_count: int = 0
    max_retries: int = 3
    title: str = "Test task"


class FakeScalarsResult:
    """Mimics SQLAlchemy scalars result."""

    def __init__(self, items: list[Any]) -> None:
        """Initialize with items."""
        self._items = items

    def all(self) -> list[Any]:
        """Return all items."""
        return self._items


class FakeExecuteResult:
    """Mimics SQLAlchemy execute result."""

    def __init__(self, items: list[Any] | None = None, rowcount: int = 0) -> None:
        """Initialize with items and rowcount."""
        self._items = items or []
        self.rowcount = rowcount

    def scalars(self) -> FakeScalarsResult:
        """Return scalars result."""
        return FakeScalarsResult(self._items)


def make_session_factory(execute_results: list[FakeExecuteResult] | None = None) -> Any:
    """Create a mock session factory that returns mock database sessions.

    Args:
        execute_results: Optional list of execute results to return in order.

    Returns:
        Callable that produces mock async sessions.
    """
    call_count = 0

    @asynccontextmanager
    async def factory() -> AsyncGenerator[AsyncMock, None]:
        nonlocal call_count
        mock_session = AsyncMock()

        if execute_results:
            side_effects = list(execute_results)
            mock_session.execute = AsyncMock(side_effect=side_effects)
        else:
            mock_session.execute = AsyncMock(return_value=FakeExecuteResult([], 0))

        mock_session.commit = AsyncMock()
        mock_session.begin = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(),
                __aexit__=AsyncMock(),
            )
        )
        call_count += 1
        yield mock_session

    factory.call_count = lambda: call_count  # type: ignore[attr-defined]
    return factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create an AgentConfig with shorter timeouts for testing.

    Returns:
        AgentConfig with 60s session timeout and 30s idle timeout.
    """
    return AgentConfig(
        session_timeout_seconds=60,
        idle_timeout_seconds=30,
        max_retries=3,
    )


@pytest.fixture
def default_config() -> AgentConfig:
    """Create an AgentConfig with default values.

    Returns:
        AgentConfig with all defaults.
    """
    return AgentConfig()


# ---------------------------------------------------------------------------
# P6-006: OrphanReason Enum
# ---------------------------------------------------------------------------


class TestOrphanReasonEnum:
    """Tests for OrphanReason enum values."""

    def test_process_crash_value(self) -> None:
        """Test PROCESS_CRASH has correct string value."""
        assert OrphanReason.PROCESS_CRASH == "process_crash"

    def test_session_timeout_value(self) -> None:
        """Test SESSION_TIMEOUT has correct string value."""
        assert OrphanReason.SESSION_TIMEOUT == "session_timeout"

    def test_stale_heartbeat_value(self) -> None:
        """Test STALE_HEARTBEAT has correct string value."""
        assert OrphanReason.STALE_HEARTBEAT == "stale_heartbeat"

    def test_unknown_value(self) -> None:
        """Test UNKNOWN has correct string value."""
        assert OrphanReason.UNKNOWN == "unknown"

    def test_all_reasons_are_str(self) -> None:
        """Test all OrphanReason values are strings."""
        for reason in OrphanReason:
            assert isinstance(reason.value, str)
            assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# P6-007: CleanupAction Enum
# ---------------------------------------------------------------------------


class TestCleanupActionEnum:
    """Tests for CleanupAction enum values."""

    def test_session_terminated_value(self) -> None:
        """Test SESSION_TERMINATED has correct string value."""
        assert CleanupAction.SESSION_TERMINATED == "session_terminated"

    def test_task_reset_value(self) -> None:
        """Test TASK_RESET has correct string value."""
        assert CleanupAction.TASK_RESET == "task_reset"

    def test_worktree_cleaned_value(self) -> None:
        """Test WORKTREE_CLEANED has correct string value."""
        assert CleanupAction.WORKTREE_CLEANED == "worktree_cleaned"

    def test_lock_released_value(self) -> None:
        """Test LOCK_RELEASED has correct string value."""
        assert CleanupAction.LOCK_RELEASED == "lock_released"


# ---------------------------------------------------------------------------
# Pydantic Model Tests
# ---------------------------------------------------------------------------


class TestPydanticModels:
    """Tests for Pydantic model creation and defaults."""

    def test_orphan_session_creation(self) -> None:
        """Test OrphanSession with all fields."""
        orphan = OrphanSession(
            session_id="sess-1",
            task_id="task-1",
            status="active",
            started_at="2025-01-01T00:00:00Z",
            last_activity="2025-01-01T00:00:00Z",
            reason=OrphanReason.SESSION_TIMEOUT,
        )
        assert orphan.session_id == "sess-1"
        assert orphan.reason == OrphanReason.SESSION_TIMEOUT

    def test_orphan_session_defaults(self) -> None:
        """Test OrphanSession default values."""
        orphan = OrphanSession(
            session_id="sess-1",
            task_id="task-1",
            status="active",
        )
        assert orphan.started_at is None
        assert orphan.last_activity is None
        assert orphan.reason == OrphanReason.UNKNOWN

    def test_cleanup_result_creation(self) -> None:
        """Test CleanupResult with all fields."""
        result = CleanupResult(
            session_id="sess-1",
            task_id="task-1",
            action=CleanupAction.SESSION_TERMINATED,
            success=True,
        )
        assert result.success is True
        assert result.error is None

    def test_cleanup_result_failure(self) -> None:
        """Test CleanupResult representing a failure."""
        result = CleanupResult(
            session_id="sess-1",
            action=CleanupAction.LOCK_RELEASED,
            success=False,
            error="Permission denied",
        )
        assert result.success is False
        assert result.error == "Permission denied"
        assert result.task_id is None

    def test_retry_decision_should_retry(self) -> None:
        """Test RetryDecision when retry is warranted."""
        decision = RetryDecision(
            task_id="task-1",
            should_retry=True,
            current_retries=1,
            max_retries=3,
            delay_seconds=60.0,
            reason="Retry 2/3 after 60.0s backoff",
        )
        assert decision.should_retry is True
        assert decision.delay_seconds == 60.0

    def test_retry_decision_exhausted(self) -> None:
        """Test RetryDecision when retries are exhausted."""
        decision = RetryDecision(
            task_id="task-1",
            should_retry=False,
            current_retries=3,
            max_retries=3,
            delay_seconds=0.0,
            reason="Max retries exhausted (3/3)",
        )
        assert decision.should_retry is False

    def test_retry_decision_defaults(self) -> None:
        """Test RetryDecision default values."""
        decision = RetryDecision(task_id="task-1")
        assert decision.should_retry is False
        assert decision.current_retries == 0
        assert decision.max_retries == 3
        assert decision.delay_seconds == 0.0
        assert decision.reason == ""

    def test_recovery_report_defaults(self) -> None:
        """Test RecoveryReport default values."""
        report = RecoveryReport()
        assert report.orphan_sessions_found == 0
        assert report.sessions_cleaned == 0
        assert report.tasks_retried == 0
        assert report.tasks_failed == 0
        assert report.cleanup_results == []
        assert report.retry_decisions == []
        assert report.duration_seconds == 0.0

    def test_recovery_report_with_data(self) -> None:
        """Test RecoveryReport with populated data."""
        report = RecoveryReport(
            orphan_sessions_found=3,
            sessions_cleaned=2,
            tasks_retried=1,
            tasks_failed=1,
            started_at="2025-01-01T00:00:00Z",
            completed_at="2025-01-01T00:00:05Z",
            duration_seconds=5.0,
        )
        assert report.orphan_sessions_found == 3
        assert report.sessions_cleaned == 2


# ---------------------------------------------------------------------------
# P6-006: Orphan Detection
# ---------------------------------------------------------------------------


class TestOrphanDetector:
    """Tests for OrphanDetector."""

    @pytest.mark.asyncio
    async def test_detect_timed_out_sessions(self, agent_config: AgentConfig) -> None:
        """Test detection of sessions that have exceeded session timeout."""
        old_time = datetime.now(UTC) - timedelta(hours=2)
        fake_session = FakeAgentSession(
            status=SessionStatus.active,
            started_at=old_time,
            last_activity_at=old_time,
        )

        factory = make_session_factory(
            [
                FakeExecuteResult([fake_session]),  # Running sessions query
                FakeExecuteResult([]),  # Assigned sessions query
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphans = await detector.detect_orphans()

        assert len(orphans) == 1
        assert orphans[0].session_id == str(fake_session.id)
        assert orphans[0].reason in (OrphanReason.SESSION_TIMEOUT, OrphanReason.STALE_HEARTBEAT)

    @pytest.mark.asyncio
    async def test_detect_stale_assigned_sessions(self, agent_config: AgentConfig) -> None:
        """Test detection of assigned sessions that were never started."""
        old_time = datetime.now(UTC) - timedelta(minutes=10)
        fake_session = FakeAgentSession(
            status=SessionStatus.initialising,
            started_at=old_time,
            last_activity_at=old_time,
        )

        factory = make_session_factory(
            [
                FakeExecuteResult([]),  # Running sessions query (empty)
                FakeExecuteResult([fake_session]),  # Assigned sessions query
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphans = await detector.detect_orphans()

        assert len(orphans) == 1
        assert orphans[0].reason == OrphanReason.PROCESS_CRASH

    @pytest.mark.asyncio
    async def test_detect_no_orphans(self, agent_config: AgentConfig) -> None:
        """Test detection returns empty when no orphans exist."""
        factory = make_session_factory(
            [
                FakeExecuteResult([]),  # No running orphans
                FakeExecuteResult([]),  # No assigned orphans
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphans = await detector.detect_orphans()

        assert len(orphans) == 0

    @pytest.mark.asyncio
    async def test_detect_multiple_orphans(self, agent_config: AgentConfig) -> None:
        """Test detection of multiple orphans across categories."""
        old_time = datetime.now(UTC) - timedelta(hours=2)
        running_orphan = FakeAgentSession(
            status=SessionStatus.active,
            started_at=old_time,
            last_activity_at=old_time,
        )
        assigned_orphan = FakeAgentSession(
            status=SessionStatus.initialising,
            started_at=old_time,
            last_activity_at=old_time,
        )

        factory = make_session_factory(
            [
                FakeExecuteResult([running_orphan]),
                FakeExecuteResult([assigned_orphan]),
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphans = await detector.detect_orphans()

        assert len(orphans) == 2

    @pytest.mark.asyncio
    async def test_detect_stale_heartbeat_classification(self, agent_config: AgentConfig) -> None:
        """Test that sessions with very stale heartbeats are classified correctly."""
        # Session started 2h ago but last activity was 2h ago (100% gap)
        start_time = datetime.now(UTC) - timedelta(hours=2)
        fake_session = FakeAgentSession(
            status=SessionStatus.active,
            started_at=start_time,
            last_activity_at=start_time,
        )

        factory = make_session_factory(
            [
                FakeExecuteResult([fake_session]),
                FakeExecuteResult([]),
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphans = await detector.detect_orphans()

        assert len(orphans) == 1
        assert orphans[0].reason == OrphanReason.STALE_HEARTBEAT

    @pytest.mark.asyncio
    async def test_orphan_session_fields_populated(self, agent_config: AgentConfig) -> None:
        """Test that orphan session fields are properly populated."""
        now = datetime.now(UTC)
        old_time = now - timedelta(hours=2)
        task_id = uuid.uuid4()
        fake_session = FakeAgentSession(
            status=SessionStatus.active,
            task_id=task_id,
            started_at=old_time,
            last_activity_at=old_time,
        )

        factory = make_session_factory(
            [
                FakeExecuteResult([fake_session]),
                FakeExecuteResult([]),
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphans = await detector.detect_orphans()

        assert len(orphans) == 1
        orphan = orphans[0]
        assert orphan.task_id == str(task_id)
        assert orphan.status == "active"
        assert orphan.started_at is not None
        assert orphan.last_activity is not None


# ---------------------------------------------------------------------------
# P6-006: Orphaned Task Detection
# ---------------------------------------------------------------------------


class TestOrphanedTaskDetection:
    """Tests for detecting tasks with no active session."""

    @pytest.mark.asyncio
    async def test_detect_orphaned_tasks(self, agent_config: AgentConfig) -> None:
        """Test detection of running tasks with no active session."""
        task = FakeTask(status=TaskStatus.running)

        factory = make_session_factory(
            [
                FakeExecuteResult([task]),  # Active tasks
                FakeExecuteResult([]),  # No active sessions
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphaned_ids = await detector.detect_orphaned_tasks()

        assert len(orphaned_ids) == 1
        assert orphaned_ids[0] == str(task.id)

    @pytest.mark.asyncio
    async def test_no_orphaned_tasks_when_sessions_exist(self, agent_config: AgentConfig) -> None:
        """Test no orphaned tasks when all tasks have active sessions."""
        task = FakeTask(status=TaskStatus.running)
        session = FakeAgentSession(
            task_id=task.id,
            status=SessionStatus.active,
        )

        factory = make_session_factory(
            [
                FakeExecuteResult([task]),  # Active tasks
                FakeExecuteResult([session]),  # Active session for this task
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphaned_ids = await detector.detect_orphaned_tasks()

        assert len(orphaned_ids) == 0

    @pytest.mark.asyncio
    async def test_no_orphaned_tasks_empty_database(self, agent_config: AgentConfig) -> None:
        """Test no orphaned tasks when database is empty."""
        factory = make_session_factory(
            [
                FakeExecuteResult([]),  # No active tasks
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphaned_ids = await detector.detect_orphaned_tasks()

        assert len(orphaned_ids) == 0

    @pytest.mark.asyncio
    async def test_detect_assigned_orphaned_tasks(self, agent_config: AgentConfig) -> None:
        """Test detection of assigned tasks with no active session."""
        task = FakeTask(status=TaskStatus.assigned)

        factory = make_session_factory(
            [
                FakeExecuteResult([task]),  # Active tasks
                FakeExecuteResult([]),  # No active sessions
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphaned_ids = await detector.detect_orphaned_tasks()

        assert len(orphaned_ids) == 1


# ---------------------------------------------------------------------------
# P6-007: Session Cleanup
# ---------------------------------------------------------------------------


class TestSessionCleaner:
    """Tests for SessionCleaner."""

    @pytest.mark.asyncio
    async def test_cleanup_single_orphan(self, agent_config: AgentConfig) -> None:
        """Test cleanup of a single orphan session."""
        task_id = str(uuid.uuid4())
        orphan = OrphanSession(
            session_id=str(uuid.uuid4()),
            task_id=task_id,
            status="active",
            reason=OrphanReason.SESSION_TIMEOUT,
        )

        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=0,
            max_retries=3,
        )

        with (
            patch(
                "forgemaster.orchestrator.recovery.end_session",
                new_callable=AsyncMock,
            ),
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=fake_task,
            ),
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ),
        ):
            factory = make_session_factory(
                [
                    FakeExecuteResult([], rowcount=0),  # lock release query
                ]
            )
            cleaner = SessionCleaner(factory, agent_config)
            results = await cleaner.cleanup_orphan(orphan)

        assert len(results) >= 1
        # Should have SESSION_TERMINATED
        terminated = [r for r in results if r.action == CleanupAction.SESSION_TERMINATED]
        assert len(terminated) == 1
        assert terminated[0].success is True

    @pytest.mark.asyncio
    async def test_cleanup_multiple_orphans(self, agent_config: AgentConfig) -> None:
        """Test cleanup of multiple orphan sessions."""
        orphans = [
            OrphanSession(
                session_id=str(uuid.uuid4()),
                task_id=str(uuid.uuid4()),
                status="active",
                reason=OrphanReason.SESSION_TIMEOUT,
            )
            for _ in range(3)
        ]

        with (
            patch(
                "forgemaster.orchestrator.recovery.end_session",
                new_callable=AsyncMock,
            ),
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=FakeTask(retry_count=0, max_retries=3),
            ),
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ),
        ):
            factory = make_session_factory(
                [
                    FakeExecuteResult([], rowcount=0),
                ]
            )
            # Re-create factory for each call - use a simpler mock approach
            cleaner = SessionCleaner(factory, agent_config)
            results = await cleaner.cleanup_all_orphans(orphans)

        # At least 3 terminated actions
        terminated = [r for r in results if r.action == CleanupAction.SESSION_TERMINATED]
        assert len(terminated) == 3

    @pytest.mark.asyncio
    async def test_cleanup_handles_termination_failure(self, agent_config: AgentConfig) -> None:
        """Test cleanup continues when session termination fails."""
        orphan = OrphanSession(
            session_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            status="active",
            reason=OrphanReason.PROCESS_CRASH,
        )

        with patch(
            "forgemaster.orchestrator.recovery.end_session",
            new_callable=AsyncMock,
            side_effect=ValueError("Session not found"),
        ):
            factory = make_session_factory(
                [
                    FakeExecuteResult([], rowcount=0),
                ]
            )
            cleaner = SessionCleaner(factory, agent_config)

            # Should also patch get_task and update_task_status for task reset
            with (
                patch(
                    "forgemaster.orchestrator.recovery.get_task",
                    new_callable=AsyncMock,
                    return_value=FakeTask(retry_count=0, max_retries=3),
                ),
                patch(
                    "forgemaster.orchestrator.recovery.update_task_status",
                    new_callable=AsyncMock,
                ),
            ):
                results = await cleaner.cleanup_orphan(orphan)

        # Termination should have failed
        terminated = [r for r in results if r.action == CleanupAction.SESSION_TERMINATED]
        assert len(terminated) == 1
        assert terminated[0].success is False
        assert "Session not found" in (terminated[0].error or "")

    @pytest.mark.asyncio
    async def test_cleanup_orphan_without_task(self, agent_config: AgentConfig) -> None:
        """Test cleanup of orphan session with no associated task."""
        orphan = OrphanSession(
            session_id=str(uuid.uuid4()),
            task_id="",  # No task
            status="active",
            reason=OrphanReason.UNKNOWN,
        )

        with patch(
            "forgemaster.orchestrator.recovery.end_session",
            new_callable=AsyncMock,
        ):
            factory = make_session_factory()
            cleaner = SessionCleaner(factory, agent_config)
            results = await cleaner.cleanup_orphan(orphan)

        # Should only have SESSION_TERMINATED (no lock release or task reset)
        assert len(results) == 1
        assert results[0].action == CleanupAction.SESSION_TERMINATED


# ---------------------------------------------------------------------------
# P6-007: Task Reset
# ---------------------------------------------------------------------------


class TestTaskReset:
    """Tests for task reset logic during cleanup."""

    @pytest.mark.asyncio
    async def test_task_reset_with_retries_remaining(self, agent_config: AgentConfig) -> None:
        """Test task is reset to READY when retries remain."""
        task_id = str(uuid.uuid4())
        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=1,
            max_retries=3,
        )

        with (
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=fake_task,
            ),
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            factory = make_session_factory()
            cleaner = SessionCleaner(factory, agent_config)
            result = await cleaner._reset_task(task_id)

        assert result == "ready"
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][2] == TaskStatus.ready

    @pytest.mark.asyncio
    async def test_task_reset_with_retries_exhausted(self, agent_config: AgentConfig) -> None:
        """Test task is marked FAILED when retries are exhausted."""
        task_id = str(uuid.uuid4())
        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=3,
            max_retries=3,
        )

        with (
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=fake_task,
            ),
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            factory = make_session_factory()
            cleaner = SessionCleaner(factory, agent_config)
            result = await cleaner._reset_task(task_id)

        assert result == "failed"
        call_args = mock_update.call_args
        assert call_args[0][2] == TaskStatus.failed

    @pytest.mark.asyncio
    async def test_task_reset_not_found(self, agent_config: AgentConfig) -> None:
        """Test task reset raises when task not found."""
        task_id = str(uuid.uuid4())

        with patch(
            "forgemaster.orchestrator.recovery.get_task",
            new_callable=AsyncMock,
            return_value=None,
        ):
            factory = make_session_factory()
            cleaner = SessionCleaner(factory, agent_config)

            with pytest.raises(ValueError, match="not found"):
                await cleaner._reset_task(task_id)


# ---------------------------------------------------------------------------
# P6-007: Lock Release
# ---------------------------------------------------------------------------


class TestLockRelease:
    """Tests for file lock release during cleanup."""

    @pytest.mark.asyncio
    async def test_lock_release_during_cleanup(self, agent_config: AgentConfig) -> None:
        """Test that file locks are released during orphan cleanup."""
        task_id = str(uuid.uuid4())
        orphan = OrphanSession(
            session_id=str(uuid.uuid4()),
            task_id=task_id,
            status="active",
            reason=OrphanReason.SESSION_TIMEOUT,
        )

        with (
            patch(
                "forgemaster.orchestrator.recovery.end_session",
                new_callable=AsyncMock,
            ),
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=FakeTask(
                    id=uuid.UUID(task_id),
                    retry_count=0,
                    max_retries=3,
                ),
            ),
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ),
        ):
            # Session factory that returns rowcount=2 for lock release
            factory = make_session_factory(
                [
                    FakeExecuteResult([], rowcount=2),
                ]
            )
            cleaner = SessionCleaner(factory, agent_config)
            results = await cleaner.cleanup_orphan(orphan)

        lock_results = [r for r in results if r.action == CleanupAction.LOCK_RELEASED]
        assert len(lock_results) == 1
        assert lock_results[0].success is True

    @pytest.mark.asyncio
    async def test_no_lock_release_when_zero_locks(self, agent_config: AgentConfig) -> None:
        """Test no LOCK_RELEASED action when no locks exist."""
        task_id = str(uuid.uuid4())
        orphan = OrphanSession(
            session_id=str(uuid.uuid4()),
            task_id=task_id,
            status="active",
            reason=OrphanReason.SESSION_TIMEOUT,
        )

        with (
            patch(
                "forgemaster.orchestrator.recovery.end_session",
                new_callable=AsyncMock,
            ),
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=FakeTask(
                    id=uuid.UUID(task_id),
                    retry_count=0,
                    max_retries=3,
                ),
            ),
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ),
        ):
            factory = make_session_factory(
                [
                    FakeExecuteResult([], rowcount=0),
                ]
            )
            cleaner = SessionCleaner(factory, agent_config)
            results = await cleaner.cleanup_orphan(orphan)

        lock_results = [r for r in results if r.action == CleanupAction.LOCK_RELEASED]
        assert len(lock_results) == 0


# ---------------------------------------------------------------------------
# P6-008: Retry Evaluation
# ---------------------------------------------------------------------------


class TestRetrySchedulerEvaluation:
    """Tests for RetryScheduler.evaluate_retry."""

    @pytest.mark.asyncio
    async def test_should_retry_when_retries_remaining(self, agent_config: AgentConfig) -> None:
        """Test retry decision is positive when retries remain."""
        task_id = str(uuid.uuid4())
        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=1,
            max_retries=3,
        )

        with patch(
            "forgemaster.orchestrator.recovery.get_task",
            new_callable=AsyncMock,
            return_value=fake_task,
        ):
            factory = make_session_factory()
            scheduler = RetryScheduler(factory, agent_config)
            decision = await scheduler.evaluate_retry(task_id)

        assert decision.should_retry is True
        assert decision.current_retries == 1
        assert decision.max_retries == 3
        assert decision.delay_seconds > 0

    @pytest.mark.asyncio
    async def test_should_not_retry_when_exhausted(self, agent_config: AgentConfig) -> None:
        """Test retry decision is negative when retries exhausted."""
        task_id = str(uuid.uuid4())
        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=3,
            max_retries=3,
        )

        with patch(
            "forgemaster.orchestrator.recovery.get_task",
            new_callable=AsyncMock,
            return_value=fake_task,
        ):
            factory = make_session_factory()
            scheduler = RetryScheduler(factory, agent_config)
            decision = await scheduler.evaluate_retry(task_id)

        assert decision.should_retry is False
        assert "exhausted" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_retry_not_found_task(self, agent_config: AgentConfig) -> None:
        """Test retry decision when task not found."""
        task_id = str(uuid.uuid4())

        with patch(
            "forgemaster.orchestrator.recovery.get_task",
            new_callable=AsyncMock,
            return_value=None,
        ):
            factory = make_session_factory()
            scheduler = RetryScheduler(factory, agent_config)
            decision = await scheduler.evaluate_retry(task_id)

        assert decision.should_retry is False
        assert "not found" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay(self, agent_config: AgentConfig) -> None:
        """Test exponential backoff delay calculation."""
        factory = make_session_factory()
        scheduler = RetryScheduler(factory, agent_config, base_delay=10.0, max_delay=1000.0)

        delays = []
        for retry_count in range(4):
            fake_task = FakeTask(
                retry_count=retry_count,
                max_retries=5,
            )
            with patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=fake_task,
            ):
                decision = await scheduler.evaluate_retry(str(fake_task.id))
                delays.append(decision.delay_seconds)

        # Verify exponential growth: 10, 20, 40, 80
        assert delays[0] == 10.0
        assert delays[1] == 20.0
        assert delays[2] == 40.0
        assert delays[3] == 80.0

    @pytest.mark.asyncio
    async def test_backoff_delay_capped_at_max(self, agent_config: AgentConfig) -> None:
        """Test that backoff delay is capped at max_delay."""
        factory = make_session_factory()
        scheduler = RetryScheduler(factory, agent_config, base_delay=100.0, max_delay=200.0)

        fake_task = FakeTask(retry_count=5, max_retries=10)

        with patch(
            "forgemaster.orchestrator.recovery.get_task",
            new_callable=AsyncMock,
            return_value=fake_task,
        ):
            decision = await scheduler.evaluate_retry(str(fake_task.id))

        # 100 * 2^5 = 3200, but capped at 200
        assert decision.delay_seconds == 200.0

    @pytest.mark.asyncio
    async def test_zero_retries_base_delay(self, agent_config: AgentConfig) -> None:
        """Test delay at zero retries is exactly base_delay."""
        factory = make_session_factory()
        scheduler = RetryScheduler(factory, agent_config, base_delay=30.0, max_delay=600.0)

        fake_task = FakeTask(retry_count=0, max_retries=3)

        with patch(
            "forgemaster.orchestrator.recovery.get_task",
            new_callable=AsyncMock,
            return_value=fake_task,
        ):
            decision = await scheduler.evaluate_retry(str(fake_task.id))

        assert decision.delay_seconds == 30.0


# ---------------------------------------------------------------------------
# P6-008: Retry Scheduling
# ---------------------------------------------------------------------------


class TestRetryScheduling:
    """Tests for RetryScheduler.schedule_retry."""

    @pytest.mark.asyncio
    async def test_schedule_retry_success(self, agent_config: AgentConfig) -> None:
        """Test scheduling a retry increments count and sets READY."""
        task_id = str(uuid.uuid4())
        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=0,
            max_retries=3,
        )

        with (
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=fake_task,
            ),
            patch(
                "forgemaster.orchestrator.recovery.increment_retry_count",
                new_callable=AsyncMock,
            ) as mock_inc,
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ) as mock_update,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            factory = make_session_factory()
            scheduler = RetryScheduler(factory, agent_config, base_delay=0.0, max_delay=0.0)
            decision = await scheduler.schedule_retry(task_id)

        assert decision.should_retry is True
        mock_inc.assert_called_once()
        # update_task_status should set to READY
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][2] == TaskStatus.ready

    @pytest.mark.asyncio
    async def test_schedule_retry_exhausted_marks_failed(self, agent_config: AgentConfig) -> None:
        """Test scheduling with exhausted retries marks task as FAILED."""
        task_id = str(uuid.uuid4())
        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=3,
            max_retries=3,
        )

        with (
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=fake_task,
            ),
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            factory = make_session_factory()
            scheduler = RetryScheduler(factory, agent_config)
            decision = await scheduler.schedule_retry(task_id)

        assert decision.should_retry is False
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][2] == TaskStatus.failed

    @pytest.mark.asyncio
    async def test_schedule_retry_applies_backoff(self, agent_config: AgentConfig) -> None:
        """Test that schedule_retry waits for the backoff delay."""
        task_id = str(uuid.uuid4())
        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=1,
            max_retries=3,
        )

        with (
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                return_value=fake_task,
            ),
            patch(
                "forgemaster.orchestrator.recovery.increment_retry_count",
                new_callable=AsyncMock,
            ),
            patch(
                "forgemaster.orchestrator.recovery.update_task_status",
                new_callable=AsyncMock,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            factory = make_session_factory()
            scheduler = RetryScheduler(factory, agent_config, base_delay=10.0, max_delay=600.0)
            decision = await scheduler.schedule_retry(task_id)

        assert decision.should_retry is True
        # Should sleep for 10 * 2^1 = 20 seconds
        mock_sleep.assert_called_once_with(20.0)


# ---------------------------------------------------------------------------
# P6-009: Startup Recovery
# ---------------------------------------------------------------------------


class TestRecoveryManager:
    """Tests for RecoveryManager startup recovery."""

    @pytest.mark.asyncio
    async def test_startup_recovery_with_orphans(self, agent_config: AgentConfig) -> None:
        """Test startup recovery detects and cleans orphans."""
        orphan = OrphanSession(
            session_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            status="active",
            reason=OrphanReason.SESSION_TIMEOUT,
        )

        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        # Mock the sub-components
        manager.detector.detect_orphans = AsyncMock(return_value=[orphan])
        manager.detector.detect_orphaned_tasks = AsyncMock(return_value=[])
        manager.cleaner.cleanup_all_orphans = AsyncMock(
            return_value=[
                CleanupResult(
                    session_id=orphan.session_id,
                    task_id=orphan.task_id,
                    action=CleanupAction.SESSION_TERMINATED,
                    success=True,
                )
            ]
        )

        report = await manager.run_startup_recovery()

        assert report.orphan_sessions_found == 1
        assert report.sessions_cleaned == 1
        assert report.tasks_retried == 0
        assert report.tasks_failed == 0

    @pytest.mark.asyncio
    async def test_startup_recovery_with_retries(self, agent_config: AgentConfig) -> None:
        """Test startup recovery schedules retries for orphaned tasks."""
        task_id = str(uuid.uuid4())

        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        manager.detector.detect_orphans = AsyncMock(return_value=[])
        manager.detector.detect_orphaned_tasks = AsyncMock(return_value=[task_id])
        manager.scheduler.schedule_retry = AsyncMock(
            return_value=RetryDecision(
                task_id=task_id,
                should_retry=True,
                current_retries=0,
                max_retries=3,
                delay_seconds=30.0,
                reason="Retry 1/3",
            )
        )

        report = await manager.run_startup_recovery()

        assert report.orphan_sessions_found == 0
        assert report.tasks_retried == 1
        assert report.tasks_failed == 0
        assert len(report.retry_decisions) == 1

    @pytest.mark.asyncio
    async def test_startup_recovery_clean_state(self, agent_config: AgentConfig) -> None:
        """Test startup recovery with no orphans or orphaned tasks."""
        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        manager.detector.detect_orphans = AsyncMock(return_value=[])
        manager.detector.detect_orphaned_tasks = AsyncMock(return_value=[])

        report = await manager.run_startup_recovery()

        assert report.orphan_sessions_found == 0
        assert report.sessions_cleaned == 0
        assert report.tasks_retried == 0
        assert report.tasks_failed == 0

    @pytest.mark.asyncio
    async def test_startup_recovery_with_failed_tasks(self, agent_config: AgentConfig) -> None:
        """Test startup recovery with tasks that have exhausted retries."""
        task_id = str(uuid.uuid4())

        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        manager.detector.detect_orphans = AsyncMock(return_value=[])
        manager.detector.detect_orphaned_tasks = AsyncMock(return_value=[task_id])
        manager.scheduler.schedule_retry = AsyncMock(
            return_value=RetryDecision(
                task_id=task_id,
                should_retry=False,
                current_retries=3,
                max_retries=3,
                delay_seconds=0.0,
                reason="Max retries exhausted",
            )
        )

        report = await manager.run_startup_recovery()

        assert report.tasks_retried == 0
        assert report.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_startup_recovery_mixed_outcomes(self, agent_config: AgentConfig) -> None:
        """Test startup recovery with mix of retried and failed tasks."""
        task_id_1 = str(uuid.uuid4())
        task_id_2 = str(uuid.uuid4())

        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        manager.detector.detect_orphans = AsyncMock(return_value=[])
        manager.detector.detect_orphaned_tasks = AsyncMock(return_value=[task_id_1, task_id_2])

        async def mock_schedule(tid: str) -> RetryDecision:
            if tid == task_id_1:
                return RetryDecision(
                    task_id=tid,
                    should_retry=True,
                    current_retries=0,
                    max_retries=3,
                    delay_seconds=30.0,
                    reason="Retry 1/3",
                )
            else:
                return RetryDecision(
                    task_id=tid,
                    should_retry=False,
                    current_retries=3,
                    max_retries=3,
                    reason="Max retries exhausted",
                )

        manager.scheduler.schedule_retry = AsyncMock(side_effect=mock_schedule)

        report = await manager.run_startup_recovery()

        assert report.tasks_retried == 1
        assert report.tasks_failed == 1
        assert len(report.retry_decisions) == 2


# ---------------------------------------------------------------------------
# P6-009: Recovery Report
# ---------------------------------------------------------------------------


class TestRecoveryReport:
    """Tests for recovery report generation."""

    @pytest.mark.asyncio
    async def test_report_timestamps(self, agent_config: AgentConfig) -> None:
        """Test recovery report has correct timestamps."""
        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        manager.detector.detect_orphans = AsyncMock(return_value=[])
        manager.detector.detect_orphaned_tasks = AsyncMock(return_value=[])

        report = await manager.run_startup_recovery()

        assert report.started_at != ""
        assert report.completed_at != ""
        assert report.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_report_cleanup_results_included(self, agent_config: AgentConfig) -> None:
        """Test cleanup results are included in the report."""
        orphan = OrphanSession(
            session_id="sess-1",
            task_id="task-1",
            status="active",
            reason=OrphanReason.SESSION_TIMEOUT,
        )

        cleanup_results = [
            CleanupResult(
                session_id="sess-1",
                task_id="task-1",
                action=CleanupAction.SESSION_TERMINATED,
                success=True,
            ),
            CleanupResult(
                session_id="sess-1",
                task_id="task-1",
                action=CleanupAction.LOCK_RELEASED,
                success=True,
            ),
        ]

        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        manager.detector.detect_orphans = AsyncMock(return_value=[orphan])
        manager.detector.detect_orphaned_tasks = AsyncMock(return_value=[])
        manager.cleaner.cleanup_all_orphans = AsyncMock(return_value=cleanup_results)

        report = await manager.run_startup_recovery()

        assert len(report.cleanup_results) == 2
        assert report.sessions_cleaned == 1


# ---------------------------------------------------------------------------
# P6-009: Periodic Recovery
# ---------------------------------------------------------------------------


class TestPeriodicRecovery:
    """Tests for periodic recovery background task."""

    @pytest.mark.asyncio
    async def test_periodic_recovery_runs(self, agent_config: AgentConfig) -> None:
        """Test that periodic recovery calls run_startup_recovery."""
        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        call_count = 0

        async def mock_startup_recovery() -> RecoveryReport:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                manager._running = False
            return RecoveryReport()

        manager.run_startup_recovery = AsyncMock(side_effect=mock_startup_recovery)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await manager.run_periodic_recovery(interval_seconds=0.01)

        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_periodic_recovery_stop(self, agent_config: AgentConfig) -> None:
        """Test stopping periodic recovery."""
        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        call_count = 0

        async def mock_startup_recovery() -> RecoveryReport:
            nonlocal call_count
            call_count += 1
            # After first call, stop the manager
            if call_count >= 1:
                manager._running = False
            return RecoveryReport()

        manager.run_startup_recovery = AsyncMock(side_effect=mock_startup_recovery)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await manager.run_periodic_recovery(interval_seconds=0.01)

        assert manager._running is False
        assert call_count >= 1

        # Test that stop on an already-stopped manager is safe
        await manager.stop()

    @pytest.mark.asyncio
    async def test_periodic_recovery_handles_errors(self, agent_config: AgentConfig) -> None:
        """Test periodic recovery continues after errors."""
        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        call_count = 0

        async def mock_recovery_with_error() -> RecoveryReport:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Database connection lost")
            manager._running = False
            return RecoveryReport()

        manager.run_startup_recovery = AsyncMock(side_effect=mock_recovery_with_error)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await manager.run_periodic_recovery(interval_seconds=0.01)

        # Should have continued past the error
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_periodic_recovery_already_running(self, agent_config: AgentConfig) -> None:
        """Test that starting periodic recovery twice is a no-op."""
        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)
        manager._running = True

        # Should return immediately without error
        await manager.run_periodic_recovery(interval_seconds=1.0)

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, agent_config: AgentConfig) -> None:
        """Test stopping when not running is a no-op."""
        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        # Should not raise
        await manager.stop()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_orphan_with_none_task_id(self, agent_config: AgentConfig) -> None:
        """Test handling orphan session with None task_id."""
        fake_session = FakeAgentSession(
            status=SessionStatus.active,
            task_id=None,
            started_at=datetime.now(UTC) - timedelta(hours=2),
            last_activity_at=datetime.now(UTC) - timedelta(hours=2),
        )

        factory = make_session_factory(
            [
                FakeExecuteResult([fake_session]),
                FakeExecuteResult([]),
            ]
        )

        detector = OrphanDetector(factory, agent_config)
        orphans = await detector.detect_orphans()

        assert len(orphans) == 1
        assert orphans[0].task_id == ""

    @pytest.mark.asyncio
    async def test_recovery_manager_initialization(self, agent_config: AgentConfig) -> None:
        """Test RecoveryManager properly initializes sub-components."""
        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        assert isinstance(manager.detector, OrphanDetector)
        assert isinstance(manager.cleaner, SessionCleaner)
        assert isinstance(manager.scheduler, RetryScheduler)
        assert manager._running is False
        assert manager._periodic_task is None

    @pytest.mark.asyncio
    async def test_cleanup_partial_failure(self, agent_config: AgentConfig) -> None:
        """Test cleanup continues even when some actions fail."""
        task_id = str(uuid.uuid4())
        orphan = OrphanSession(
            session_id=str(uuid.uuid4()),
            task_id=task_id,
            status="active",
            reason=OrphanReason.PROCESS_CRASH,
        )

        async def mock_end(*args: Any, **kwargs: Any) -> None:
            """Succeed on first call."""
            pass

        with (
            patch(
                "forgemaster.orchestrator.recovery.end_session",
                new_callable=AsyncMock,
                side_effect=mock_end,
            ),
            patch(
                "forgemaster.orchestrator.recovery.get_task",
                new_callable=AsyncMock,
                side_effect=ValueError("Task not found"),
            ),
        ):
            factory = make_session_factory(
                [
                    FakeExecuteResult([], rowcount=0),
                ]
            )
            cleaner = SessionCleaner(factory, agent_config)
            results = await cleaner.cleanup_orphan(orphan)

        # Session termination should succeed, task reset should fail
        terminated = [r for r in results if r.action == CleanupAction.SESSION_TERMINATED]
        task_reset = [r for r in results if r.action == CleanupAction.TASK_RESET]
        assert len(terminated) == 1
        assert terminated[0].success is True
        assert len(task_reset) == 1
        assert task_reset[0].success is False

    @pytest.mark.asyncio
    async def test_retry_decision_reason_messages(self, agent_config: AgentConfig) -> None:
        """Test that retry decision reasons are meaningful."""
        task_id = str(uuid.uuid4())
        fake_task = FakeTask(
            id=uuid.UUID(task_id),
            retry_count=2,
            max_retries=3,
        )

        with patch(
            "forgemaster.orchestrator.recovery.get_task",
            new_callable=AsyncMock,
            return_value=fake_task,
        ):
            factory = make_session_factory()
            scheduler = RetryScheduler(factory, agent_config)
            decision = await scheduler.evaluate_retry(task_id)

        assert "3/3" in decision.reason
        assert "backoff" in decision.reason.lower()

    def test_orphan_reason_is_str_enum(self) -> None:
        """Test OrphanReason is a proper str enum."""
        assert isinstance(OrphanReason.PROCESS_CRASH, str)
        assert OrphanReason.PROCESS_CRASH == "process_crash"

    def test_cleanup_action_is_str_enum(self) -> None:
        """Test CleanupAction is a proper str enum."""
        assert isinstance(CleanupAction.SESSION_TERMINATED, str)
        assert CleanupAction.SESSION_TERMINATED == "session_terminated"

    @pytest.mark.asyncio
    async def test_full_recovery_lifecycle(self, agent_config: AgentConfig) -> None:
        """Test complete recovery lifecycle from detection through cleanup."""
        orphan_session_id = str(uuid.uuid4())
        orphan_task_id = str(uuid.uuid4())
        orphaned_task_id = str(uuid.uuid4())

        orphan = OrphanSession(
            session_id=orphan_session_id,
            task_id=orphan_task_id,
            status="active",
            reason=OrphanReason.SESSION_TIMEOUT,
        )

        factory = make_session_factory()
        manager = RecoveryManager(agent_config, factory)

        manager.detector.detect_orphans = AsyncMock(return_value=[orphan])
        manager.detector.detect_orphaned_tasks = AsyncMock(return_value=[orphaned_task_id])
        manager.cleaner.cleanup_all_orphans = AsyncMock(
            return_value=[
                CleanupResult(
                    session_id=orphan_session_id,
                    task_id=orphan_task_id,
                    action=CleanupAction.SESSION_TERMINATED,
                    success=True,
                ),
                CleanupResult(
                    session_id=orphan_session_id,
                    task_id=orphan_task_id,
                    action=CleanupAction.TASK_RESET,
                    success=True,
                ),
            ]
        )
        manager.scheduler.schedule_retry = AsyncMock(
            return_value=RetryDecision(
                task_id=orphaned_task_id,
                should_retry=True,
                current_retries=0,
                max_retries=3,
                delay_seconds=30.0,
                reason="Retry 1/3 after 30.0s backoff",
            )
        )

        report = await manager.run_startup_recovery()

        assert report.orphan_sessions_found == 1
        assert report.sessions_cleaned == 1
        assert report.tasks_retried == 1
        assert report.tasks_failed == 0
        assert len(report.cleanup_results) == 2
        assert len(report.retry_decisions) == 1
        assert report.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_concurrent_orphan_detection(self, agent_config: AgentConfig) -> None:
        """Test that multiple orphans are handled sequentially."""
        orphans = [
            OrphanSession(
                session_id=str(uuid.uuid4()),
                task_id=str(uuid.uuid4()),
                status="active",
                reason=OrphanReason.SESSION_TIMEOUT,
            )
            for _ in range(5)
        ]

        cleanup_calls: list[OrphanSession] = []

        async def track_cleanup(o: OrphanSession) -> list[CleanupResult]:
            cleanup_calls.append(o)
            return [
                CleanupResult(
                    session_id=o.session_id,
                    task_id=o.task_id,
                    action=CleanupAction.SESSION_TERMINATED,
                    success=True,
                )
            ]

        factory = make_session_factory()
        cleaner = SessionCleaner(factory, agent_config)
        cleaner.cleanup_orphan = AsyncMock(side_effect=track_cleanup)

        results = await cleaner.cleanup_all_orphans(orphans)

        assert len(cleanup_calls) == 5
        assert len(results) == 5
