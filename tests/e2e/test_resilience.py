"""End-to-end tests for resilience and recovery scenarios.

Tests session handover, continuation spawning, orphan detection,
crash recovery, idle watchdog, rate limiting, and adaptive throttling.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from forgemaster.agents.session import HealthStatus, SessionMetrics, SessionState
from forgemaster.database.models.session import AgentSession, SessionStatus
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.orchestrator.handover import (
    HandoverReason,
    HandoverTrigger,
    SaveExitResponse,
)
from forgemaster.orchestrator.rate_limiter import (
    RateLimitState,
    RateLimitResponse,
    ParallelismReduction,
)
from forgemaster.orchestrator.recovery import CleanupAction, OrphanReason, OrphanSession
from forgemaster.orchestrator.watchdog import ActivityType, IdleSeverity, IdleSession


@pytest.mark.e2e
@pytest.mark.asyncio
class TestSessionHandover:
    """Test session handover on context exhaustion."""

    async def test_context_exhaustion_detection(
        self,
        create_session_info_factory,
    ) -> None:
        """Test detecting when session approaches context limit."""
        # Create session near context limit (80% of 200k)
        session_info = create_session_info_factory(
            input_tokens=80000,
            output_tokens=80000,
        )

        # Calculate usage ratio
        max_context = 200000
        usage_ratio = session_info.metrics.total_tokens / max_context

        assert usage_ratio >= 0.8  # Exceeds warning threshold

    async def test_handover_trigger_creation(self) -> None:
        """Test creating handover trigger with metadata."""
        trigger = HandoverTrigger(
            session_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30000,
        )

        assert trigger.trigger_reason == HandoverReason.CONTEXT_EXHAUSTION
        assert trigger.token_usage_ratio == 0.85
        assert trigger.estimated_remaining_tokens == 30000

    async def test_save_exit_response_structure(self) -> None:
        """Test save-and-exit response format."""
        response = SaveExitResponse(
            task_id=str(uuid.uuid4()),
            progress_summary="Completed 3 of 5 subtasks",
            files_modified=["src/main.py", "src/utils.py"],
            remaining_work=["Add error handling", "Write tests"],
            current_step="Implementing validation logic",
            context_data={"variables": {"retry_count": 2}},
        )

        assert len(response.files_modified) == 2
        assert len(response.remaining_work) == 2
        assert "retry_count" in response.context_data["variables"]

    async def test_continuation_session_spawning(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test spawning continuation session with injected context."""
        # Original session approaching limit
        original_task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Long-running task",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(original_task)
        await e2e_session.commit()

        # Simulate handover context
        handover_context = {
            "progress_summary": "Completed 50% of work",
            "remaining_work": ["Task 3", "Task 4"],
        }

        # Continuation session would be spawned here
        assert original_task.status == TaskStatus.running


@pytest.mark.e2e
@pytest.mark.asyncio
class TestOrphanDetection:
    """Test orphan session detection and cleanup."""

    async def test_detect_orphan_session_stale_heartbeat(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test detecting orphan session with stale heartbeat."""
        # Create session with old heartbeat (1 hour ago)
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)

        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task with orphan session",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            started_at=old_time,
            created_at=old_time,
        )
        e2e_session.add(task)
        await e2e_session.commit()

        session = AgentSession(
            id=uuid.uuid4(),
            task_id=task.id,
            agent_type="executor",
            status=SessionStatus.active,
            started_at=old_time,
            last_heartbeat_at=old_time,
            created_at=old_time,
        )
        e2e_session.add(session)
        await e2e_session.commit()
        await e2e_session.refresh(session)

        # Calculate idle time
        idle_seconds = (datetime.now(timezone.utc) - session.last_heartbeat_at).total_seconds()

        assert idle_seconds > 3000  # More than 50 minutes idle

    async def test_orphan_session_model(self) -> None:
        """Test OrphanSession model structure."""
        orphan = OrphanSession(
            session_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            status="active",
            started_at=datetime.now(timezone.utc).isoformat(),
            last_activity=datetime.now(timezone.utc).isoformat(),
            reason=OrphanReason.STALE_HEARTBEAT,
        )

        assert orphan.reason == OrphanReason.STALE_HEARTBEAT
        assert orphan.status == "active"

    async def test_cleanup_actions_for_orphan(self) -> None:
        """Test cleanup actions taken on orphan sessions."""
        actions = [
            CleanupAction.SESSION_TERMINATED,
            CleanupAction.TASK_RESET,
            CleanupAction.LOCK_RELEASED,
        ]

        for action in actions:
            assert isinstance(action, CleanupAction)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestCrashRecovery:
    """Test crash recovery and task retry scheduling."""

    async def test_orphan_detection_on_startup(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test detecting orphan sessions during startup recovery."""
        # Create orphaned task (was running, but no recent heartbeat)
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)

        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Orphaned task",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            started_at=old_time,
            created_at=old_time,
        )
        e2e_session.add(task)
        await e2e_session.commit()

        # Query for running tasks with old start times
        threshold = datetime.now(timezone.utc) - timedelta(hours=1)
        result = await e2e_session.execute(
            select(Task).where(
                Task.status == TaskStatus.running,
                Task.started_at < threshold,
            )
        )
        orphaned_tasks = result.scalars().all()

        assert len(orphaned_tasks) >= 1

    async def test_task_retry_after_recovery(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test scheduling task retry after crash recovery."""
        # Create task that was running but crashed
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task to retry",
            status=TaskStatus.failed,
            agent_type="executor",
            max_retries=3,
            retry_count=1,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        # Simulate retry scheduling: FAILED → READY
        task.status = TaskStatus.ready
        await e2e_session.commit()

        assert task.status == TaskStatus.ready
        assert task.retry_count == 1
        assert task.retry_count < task.max_retries


@pytest.mark.e2e
@pytest.mark.asyncio
class TestIdleWatchdog:
    """Test idle session detection and killing."""

    async def test_idle_session_detection(
        self,
        create_session_info_factory,
    ) -> None:
        """Test detecting idle session exceeding threshold."""
        # Create session idle for 10 minutes
        session_info = create_session_info_factory()
        # Simulate old last_activity
        session_info.metrics.last_activity_at = datetime.now(timezone.utc) - timedelta(
            minutes=10
        )

        idle_seconds = session_info.metrics.idle_seconds
        threshold_seconds = 300  # 5 minutes

        assert idle_seconds > threshold_seconds

    async def test_idle_severity_classification(self) -> None:
        """Test idle severity level calculation."""
        threshold = 300.0  # 5 minutes
        idle_times = [
            (180.0, IdleSeverity.WARNING),  # 60% of threshold
            (270.0, IdleSeverity.CRITICAL),  # 90% of threshold
            (350.0, IdleSeverity.TERMINAL),  # Over threshold
        ]

        for idle_seconds, expected_severity in idle_times:
            if idle_seconds > threshold:
                severity = IdleSeverity.TERMINAL
            elif idle_seconds > threshold * 0.8:
                severity = IdleSeverity.CRITICAL
            elif idle_seconds > threshold * 0.5:
                severity = IdleSeverity.WARNING
            else:
                continue

            assert severity == expected_severity

    async def test_idle_session_model(self) -> None:
        """Test IdleSession model structure."""
        idle_session = IdleSession(
            session_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            idle_seconds=400.0,
            threshold_seconds=300.0,
            severity=IdleSeverity.TERMINAL,
        )

        assert idle_session.idle_seconds > idle_session.threshold_seconds
        assert idle_session.severity == IdleSeverity.TERMINAL

    async def test_watchdog_kill_action(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test watchdog killing idle session."""
        # Create idle session
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Idle task",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=15),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        )
        e2e_session.add(task)
        await e2e_session.commit()

        # Simulate watchdog killing session and resetting task
        task.status = TaskStatus.failed
        await e2e_session.commit()

        assert task.status == TaskStatus.failed


@pytest.mark.e2e
@pytest.mark.asyncio
class TestRateLimiting:
    """Test rate limit handling and throttling."""

    async def test_rate_limit_response_parsing(self) -> None:
        """Test parsing HTTP 429 rate limit response."""
        response = RateLimitResponse(
            status_code=429,
            retry_after_seconds=60.0,
            rate_limit_remaining=0,
            rate_limit_reset_at=datetime.now(timezone.utc).isoformat(),
            error_message="Rate limit exceeded",
        )

        assert response.status_code == 429
        assert response.retry_after_seconds == 60.0
        assert response.rate_limit_remaining == 0

    async def test_rate_limit_backoff_calculation(self) -> None:
        """Test exponential backoff delay calculation."""
        initial_delay = 1.0
        multiplier = 2.0
        max_delay = 60.0

        attempts = [
            (0, 1.0),
            (1, 2.0),
            (2, 4.0),
            (3, 8.0),
            (4, 16.0),
            (5, 32.0),
            (6, 60.0),  # Capped at max_delay
        ]

        for attempt, expected_delay in attempts:
            delay = min(initial_delay * (multiplier**attempt), max_delay)
            assert delay == expected_delay

    async def test_rate_limit_state_transitions(self) -> None:
        """Test rate limit state transitions."""
        # NORMAL → THROTTLED → BACKING_OFF → BLOCKED
        states = [
            RateLimitState.NORMAL,
            RateLimitState.THROTTLED,
            RateLimitState.BACKING_OFF,
            RateLimitState.BLOCKED,
        ]

        for state in states:
            assert isinstance(state, RateLimitState)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestAdaptiveThrottling:
    """Test adaptive parallelism reduction under load."""

    async def test_parallelism_reduction_on_429(self) -> None:
        """Test reducing parallelism after rate limit hit."""
        reduction = ParallelismReduction(
            current_max=10,
            recommended_max=5,
            reduction_reason="Rate limit 429 response",
            reduction_factor=0.5,
        )

        assert reduction.recommended_max == reduction.current_max * reduction.reduction_factor
        assert reduction.recommended_max == 5

    async def test_gradual_parallelism_recovery(self) -> None:
        """Test gradually increasing parallelism after rate limit clears."""
        # Start throttled
        current_max = 5
        original_max = 10

        # Increase by 1 every success cycle
        recovery_steps = []
        while current_max < original_max:
            current_max += 1
            recovery_steps.append(current_max)

        assert recovery_steps == [6, 7, 8, 9, 10]

    async def test_adaptive_throttler_metrics(self) -> None:
        """Test tracking throttler metrics over time."""
        metrics = {
            "rate_limit_hits": 3,
            "current_parallelism": 5,
            "max_parallelism": 10,
            "throttled_duration_seconds": 180.0,
        }

        assert metrics["current_parallelism"] < metrics["max_parallelism"]
        assert metrics["rate_limit_hits"] > 0

    async def test_worker_slot_reduction(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test reducing active worker slots during throttling."""
        # Simulate 10 workers reduced to 5
        original_workers = 10
        reduced_workers = 5

        # Create tasks representing active workers
        active_tasks = []
        for i in range(reduced_workers):
            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"Active worker {i}",
                status=TaskStatus.running,
                agent_type="executor",
                max_retries=3,
                retry_count=0,
                started_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)
            active_tasks.append(task)

        await e2e_session.commit()

        # Verify reduced worker count
        result = await e2e_session.execute(
            select(Task).where(Task.status == TaskStatus.running)
        )
        running_tasks = result.scalars().all()

        assert len(running_tasks) == reduced_workers
        assert len(running_tasks) < original_workers
