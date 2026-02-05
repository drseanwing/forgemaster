"""Unit tests for the multi-worker parallel dispatcher.

Tests cover:
- MultiWorkerDispatcher lifecycle (start/stop)
- Concurrent task dispatch to multiple workers
- Worker slot allocation and deallocation
- Concurrency limit enforcement via asyncio.Semaphore
- Worker health tracking and stuck worker detection
- Graceful shutdown with active workers
- Worker statistics and metrics
- Error resilience during dispatch and execution
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forgemaster.agents.result_schema import AgentResult
from forgemaster.agents.session import AgentSessionManager
from forgemaster.config import AgentConfig
from forgemaster.context.generator import ContextGenerator
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.orchestrator.dispatcher import (
    MultiWorkerDispatcher,
    WorkerSlot,
    WorkerState,
)
from forgemaster.orchestrator.state_machine import (
    InvalidTransitionError,
    TaskStateMachine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create an AgentConfig with 3 max concurrent workers."""
    return AgentConfig(
        max_concurrent_workers=3,
        session_timeout_seconds=1800,
        idle_timeout_seconds=300,
        max_retries=3,
        context_warning_threshold=0.8,
    )


@pytest.fixture
def mock_state_machine() -> TaskStateMachine:
    """Create a mock TaskStateMachine."""
    sm = AsyncMock(spec=TaskStateMachine)
    sm.update_pending_tasks = AsyncMock(return_value=[])
    sm.transition = AsyncMock()
    return sm


@pytest.fixture
def mock_session_manager() -> AgentSessionManager:
    """Create a mock AgentSessionManager."""
    mgr = AsyncMock(spec=AgentSessionManager)
    mgr.start_session = AsyncMock(return_value="session-001")
    mgr.send_message = AsyncMock(
        return_value='{"status": "success", "summary": "Done", "details": "All good"}'
    )
    mgr.end_session = AsyncMock(return_value={"session_id": "session-001"})
    return mgr


@pytest.fixture
def mock_context_generator() -> ContextGenerator:
    """Create a mock ContextGenerator."""
    gen = MagicMock(spec=ContextGenerator)
    gen.generate_agent_context = MagicMock(return_value="System prompt for agent")
    return gen


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create a mock async database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def session_factory(mock_db_session):
    """Create a session factory that yields the mock db session."""

    @asynccontextmanager
    async def _factory():
        yield mock_db_session

    return _factory


@pytest.fixture
def mock_worktree_pool():
    """Create a mock WorktreePool."""
    pool = AsyncMock()

    # Track worktree counter for unique names
    counter = {"value": 0}

    async def mock_acquire():
        counter["value"] += 1
        wt = MagicMock()
        wt.name = f"wt-{counter['value']}"
        wt.path = Path(f"/workspace/worker-wt-{counter['value']}")
        wt.branch = f"worktree/wt-{counter['value']}"
        wt.status = "active"
        return wt

    pool.acquire = AsyncMock(side_effect=mock_acquire)
    pool.release = AsyncMock()
    pool.assign_worktree = MagicMock()

    return pool


@pytest.fixture
def sample_project_id() -> str:
    """Return a stable project UUID string."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_task(sample_project_id) -> Task:
    """Create a sample Task in READY status."""
    return Task(
        id=uuid.uuid4(),
        project_id=uuid.UUID(sample_project_id),
        title="Implement feature X",
        description="Add the X feature to the codebase",
        status=TaskStatus.ready,
        agent_type="executor",
        model_tier="sonnet",
        priority=10,
        retry_count=0,
        max_retries=3,
        files_touched=["src/feature_x.py"],
    )


def make_task(project_id: str, title: str = "Task", priority: int = 100) -> Task:
    """Helper to create tasks with unique IDs."""
    return Task(
        id=uuid.uuid4(),
        project_id=uuid.UUID(project_id),
        title=title,
        description=f"Description for {title}",
        status=TaskStatus.ready,
        agent_type="executor",
        model_tier="sonnet",
        priority=priority,
        retry_count=0,
        max_retries=3,
        files_touched=[f"src/{title.lower().replace(' ', '_')}.py"],
    )


@pytest.fixture
def dispatcher(
    agent_config,
    session_factory,
    mock_state_machine,
    mock_session_manager,
    mock_context_generator,
    mock_worktree_pool,
) -> MultiWorkerDispatcher:
    """Create a MultiWorkerDispatcher instance with all mocked dependencies."""
    return MultiWorkerDispatcher(
        config=agent_config,
        session_factory=session_factory,
        state_machine=mock_state_machine,
        session_manager=mock_session_manager,
        context_generator=mock_context_generator,
        worktree_pool=mock_worktree_pool,
        poll_interval=0.01,  # Very short for fast tests
    )


# ---------------------------------------------------------------------------
# WorkerState and WorkerSlot Tests
# ---------------------------------------------------------------------------


class TestWorkerState:
    """Test WorkerState enum."""

    def test_states_are_strings(self):
        """WorkerState values should be lowercase strings."""
        assert WorkerState.IDLE == "idle"
        assert WorkerState.ASSIGNED == "assigned"
        assert WorkerState.RUNNING == "running"
        assert WorkerState.COMPLETING == "completing"
        assert WorkerState.FAILED == "failed"

    def test_state_string_conversion(self):
        """WorkerState should support string conversion."""
        assert str(WorkerState.IDLE) == "WorkerState.IDLE"
        assert WorkerState.IDLE.value == "idle"


class TestWorkerSlot:
    """Test WorkerSlot Pydantic model."""

    def test_create_minimal_slot(self):
        """Should create a worker slot with minimal required fields."""
        slot = WorkerSlot(
            worker_id="worker-1",
            worktree_name="wt-1",
            worktree_path="/workspace/worker-wt-1",
        )
        assert slot.worker_id == "worker-1"
        assert slot.worktree_name == "wt-1"
        assert slot.task_id is None
        assert slot.session_id is None
        assert slot.state == WorkerState.IDLE
        assert slot.started_at is None
        assert slot.tasks_completed == 0
        assert slot.tasks_failed == 0

    def test_create_full_slot(self):
        """Should create a worker slot with all fields populated."""
        now = datetime.now(timezone.utc)
        slot = WorkerSlot(
            worker_id="worker-2",
            worktree_name="wt-2",
            worktree_path="/workspace/worker-wt-2",
            task_id="task-123",
            session_id="session-456",
            state=WorkerState.RUNNING,
            started_at=now,
            tasks_completed=5,
            tasks_failed=1,
            last_health_check=now,
        )
        assert slot.task_id == "task-123"
        assert slot.state == WorkerState.RUNNING
        assert slot.tasks_completed == 5
        assert slot.tasks_failed == 1

    def test_default_timestamps(self):
        """Default timestamps should be UTC now."""
        slot = WorkerSlot(
            worker_id="worker-1",
            worktree_name="wt-1",
            worktree_path="/workspace/worker-wt-1",
        )
        now = datetime.now(timezone.utc)
        assert (now - slot.last_health_check).total_seconds() < 2
        assert (now - slot.created_at).total_seconds() < 2


# ---------------------------------------------------------------------------
# MultiWorkerDispatcher Lifecycle Tests
# ---------------------------------------------------------------------------


class TestMultiWorkerDispatcherLifecycle:
    """Test dispatcher start and stop mechanics."""

    def test_initial_state(self, dispatcher):
        """Dispatcher should start in non-running state."""
        assert dispatcher.is_running is False
        assert dispatcher.active_workers == 0
        assert dispatcher.available_slots == 3
        assert dispatcher._poll_task is None
        assert dispatcher._health_task is None

    @pytest.mark.asyncio
    async def test_start_sets_running(self, dispatcher, sample_project_id):
        """Start should set running flag and create background tasks."""
        await dispatcher.start(sample_project_id)

        assert dispatcher.is_running is True
        assert dispatcher._poll_task is not None
        assert dispatcher._health_task is not None

        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_start_raises_if_already_running(self, dispatcher, sample_project_id):
        """Start should raise RuntimeError if already running."""
        await dispatcher.start(sample_project_id)

        with pytest.raises(RuntimeError, match="already running"):
            await dispatcher.start(sample_project_id)

        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_state(self, dispatcher, sample_project_id):
        """Stop should clear running flag and all background tasks."""
        await dispatcher.start(sample_project_id)
        await dispatcher.stop()

        assert dispatcher.is_running is False
        assert dispatcher._poll_task is None
        assert dispatcher._health_task is None

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_running(self, dispatcher):
        """Stop should be safe to call when not running."""
        await dispatcher.stop()
        assert dispatcher.is_running is False

    @pytest.mark.asyncio
    async def test_max_workers_from_config(self, dispatcher, agent_config):
        """Should use max_concurrent_workers from config."""
        assert dispatcher._max_workers == agent_config.max_concurrent_workers
        assert dispatcher._max_workers == 3


# ---------------------------------------------------------------------------
# Worker Slot Allocation Tests
# ---------------------------------------------------------------------------


class TestWorkerSlotAllocation:
    """Test worker slot allocation and deallocation."""

    @pytest.mark.asyncio
    async def test_dispatch_acquires_worktree(
        self, dispatcher, mock_worktree_pool, sample_task
    ):
        """Dispatch should acquire a worktree from the pool."""
        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        assert worker_id is not None
        mock_worktree_pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_assigns_worktree_to_task(
        self, dispatcher, mock_worktree_pool, sample_task
    ):
        """Dispatch should assign the worktree to the task in the pool."""
        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        mock_worktree_pool.assign_worktree.assert_called_once()
        call_args = mock_worktree_pool.assign_worktree.call_args
        assert call_args[0][1] == str(sample_task.id)  # task_id

    @pytest.mark.asyncio
    async def test_dispatch_creates_worker_slot(
        self, dispatcher, sample_task
    ):
        """Dispatch should create a WorkerSlot in the registry."""
        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        assert worker_id is not None
        assert worker_id in dispatcher._workers
        slot = dispatcher._workers[worker_id]
        assert slot.task_id == str(sample_task.id)
        assert slot.state == WorkerState.ASSIGNED

    @pytest.mark.asyncio
    async def test_dispatch_increments_worker_counter(
        self, dispatcher, sample_project_id
    ):
        """Each dispatch should use an incrementing worker ID."""
        task1 = make_task(sample_project_id, "Task 1", priority=1)
        task2 = make_task(sample_project_id, "Task 2", priority=2)

        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            wid1 = await dispatcher._dispatch_to_worker(task1)
            wid2 = await dispatcher._dispatch_to_worker(task2)

        assert wid1 == "worker-1"
        assert wid2 == "worker-2"
        assert dispatcher._worker_counter == 2

    @pytest.mark.asyncio
    async def test_release_worker_returns_worktree(
        self, dispatcher, mock_worktree_pool, sample_task
    ):
        """Release should return the worktree to the pool."""
        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        # Cancel the background task to prevent it from running
        if worker_id in dispatcher._worker_tasks:
            dispatcher._worker_tasks[worker_id].cancel()
            try:
                await dispatcher._worker_tasks[worker_id]
            except (asyncio.CancelledError, Exception):
                pass

        await dispatcher._release_worker(worker_id)

        mock_worktree_pool.release.assert_called_once()
        assert worker_id not in dispatcher._workers

    @pytest.mark.asyncio
    async def test_release_nonexistent_worker_is_safe(self, dispatcher):
        """Releasing a nonexistent worker should not raise."""
        await dispatcher._release_worker("nonexistent-worker")
        # Should not raise


# ---------------------------------------------------------------------------
# Concurrent Task Limit Enforcement Tests
# ---------------------------------------------------------------------------


class TestConcurrencyLimits:
    """Test concurrent task limit enforcement with asyncio.Semaphore."""

    @pytest.mark.asyncio
    async def test_semaphore_initialized_from_config(self, dispatcher, agent_config):
        """Semaphore should be initialized with max_concurrent_workers."""
        assert dispatcher._semaphore._value == agent_config.max_concurrent_workers

    @pytest.mark.asyncio
    async def test_dispatch_reduces_semaphore(self, dispatcher, sample_task):
        """Each dispatch should consume one semaphore slot."""
        initial_value = dispatcher._semaphore._value

        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            await dispatcher._dispatch_to_worker(sample_task)

        # Semaphore should have been acquired (value decreased by 1)
        assert dispatcher._semaphore._value == initial_value - 1

    @pytest.mark.asyncio
    async def test_release_restores_semaphore(
        self, dispatcher, sample_task
    ):
        """Releasing a worker should restore the semaphore slot."""
        initial_value = dispatcher._semaphore._value

        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        # Cancel the background task
        if worker_id in dispatcher._worker_tasks:
            dispatcher._worker_tasks[worker_id].cancel()
            try:
                await dispatcher._worker_tasks[worker_id]
            except (asyncio.CancelledError, Exception):
                pass

        await dispatcher._release_worker(worker_id)

        assert dispatcher._semaphore._value == initial_value

    @pytest.mark.asyncio
    async def test_available_slots_tracks_active_workers(
        self, dispatcher, sample_project_id
    ):
        """available_slots should decrease as workers are dispatched."""
        assert dispatcher.available_slots == 3

        task1 = make_task(sample_project_id, "Task 1")
        task2 = make_task(sample_project_id, "Task 2")

        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            await dispatcher._dispatch_to_worker(task1)
            assert dispatcher.available_slots == 2
            assert dispatcher.active_workers == 1

            await dispatcher._dispatch_to_worker(task2)
            assert dispatcher.available_slots == 1
            assert dispatcher.active_workers == 2

    @pytest.mark.asyncio
    async def test_dispatch_blocked_when_at_capacity(
        self, dispatcher, sample_project_id
    ):
        """Dispatch should return None when all semaphore slots are taken."""
        # Exhaust the semaphore manually
        for _ in range(dispatcher._max_workers):
            await dispatcher._semaphore.acquire()

        task = make_task(sample_project_id, "Overflow task")
        worker_id = await dispatcher._dispatch_to_worker(task)

        assert worker_id is None

        # Restore semaphore
        for _ in range(dispatcher._max_workers):
            dispatcher._semaphore.release()

    @pytest.mark.asyncio
    async def test_dispatch_fails_on_worktree_acquire_error(
        self, dispatcher, mock_worktree_pool, sample_task
    ):
        """Should return None and release semaphore if worktree acquire fails."""
        mock_worktree_pool.acquire.side_effect = RuntimeError("No worktrees")
        initial_value = dispatcher._semaphore._value

        worker_id = await dispatcher._dispatch_to_worker(sample_task)

        assert worker_id is None
        # Semaphore should be restored on failure
        assert dispatcher._semaphore._value == initial_value


# ---------------------------------------------------------------------------
# Worker Health Tracking Tests
# ---------------------------------------------------------------------------


class TestWorkerHealthTracking:
    """Test worker health monitoring and stuck worker detection."""

    @pytest.mark.asyncio
    async def test_health_check_updates_timestamp(self, dispatcher, sample_task):
        """Health check should update last_health_check on workers."""
        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        # Set old health check time
        slot = dispatcher._workers[worker_id]
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        slot.last_health_check = old_time

        await dispatcher._check_worker_health()

        assert slot.last_health_check > old_time

        # Clean up
        if worker_id in dispatcher._worker_tasks:
            dispatcher._worker_tasks[worker_id].cancel()
            try:
                await dispatcher._worker_tasks[worker_id]
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_stuck_worker_detected(self, dispatcher):
        """Workers exceeding idle timeout should be flagged as FAILED."""
        # Create a worker that started long ago
        slot = WorkerSlot(
            worker_id="worker-stuck",
            worktree_name="wt-stuck",
            worktree_path="/workspace/worker-wt-stuck",
            task_id="task-stuck",
            state=WorkerState.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(seconds=500),
        )
        dispatcher._workers["worker-stuck"] = slot

        unhealthy = await dispatcher._check_worker_health()

        assert len(unhealthy) == 1
        assert unhealthy[0].worker_id == "worker-stuck"
        assert slot.state == WorkerState.FAILED

    @pytest.mark.asyncio
    async def test_healthy_worker_not_flagged(self, dispatcher):
        """Workers within timeout should not be flagged."""
        slot = WorkerSlot(
            worker_id="worker-healthy",
            worktree_name="wt-healthy",
            worktree_path="/workspace/worker-wt-healthy",
            task_id="task-healthy",
            state=WorkerState.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        dispatcher._workers["worker-healthy"] = slot

        unhealthy = await dispatcher._check_worker_health()

        assert len(unhealthy) == 0
        assert slot.state == WorkerState.RUNNING

    @pytest.mark.asyncio
    async def test_failed_worker_in_unhealthy_list(self, dispatcher):
        """Workers already in FAILED state should appear in unhealthy list."""
        slot = WorkerSlot(
            worker_id="worker-failed",
            worktree_name="wt-failed",
            worktree_path="/workspace/worker-wt-failed",
            state=WorkerState.FAILED,
        )
        dispatcher._workers["worker-failed"] = slot

        unhealthy = await dispatcher._check_worker_health()

        assert len(unhealthy) == 1
        assert unhealthy[0].worker_id == "worker-failed"

    @pytest.mark.asyncio
    async def test_idle_worker_not_flagged_as_stuck(self, dispatcher):
        """IDLE workers should not be flagged as stuck."""
        slot = WorkerSlot(
            worker_id="worker-idle",
            worktree_name="wt-idle",
            worktree_path="/workspace/worker-wt-idle",
            state=WorkerState.IDLE,
        )
        dispatcher._workers["worker-idle"] = slot

        unhealthy = await dispatcher._check_worker_health()

        assert len(unhealthy) == 0
        assert slot.state == WorkerState.IDLE


# ---------------------------------------------------------------------------
# Worker Statistics Tests
# ---------------------------------------------------------------------------


class TestWorkerStats:
    """Test worker statistics and metrics reporting."""

    def test_empty_stats(self, dispatcher):
        """Stats should be correct with no workers."""
        stats = dispatcher.get_worker_stats()

        assert stats["max_workers"] == 3
        assert stats["active_workers"] == 0
        assert stats["available_slots"] == 3
        assert stats["total_workers_created"] == 0
        assert stats["total_tasks_completed"] == 0
        assert stats["total_tasks_failed"] == 0
        assert stats["workers"] == []
        assert stats["state_distribution"] == {}

    def test_stats_with_workers(self, dispatcher):
        """Stats should reflect worker state accurately."""
        dispatcher._workers["w1"] = WorkerSlot(
            worker_id="w1",
            worktree_name="wt-1",
            worktree_path="/workspace/wt-1",
            state=WorkerState.RUNNING,
            task_id="task-1",
            tasks_completed=3,
            tasks_failed=1,
        )
        dispatcher._workers["w2"] = WorkerSlot(
            worker_id="w2",
            worktree_name="wt-2",
            worktree_path="/workspace/wt-2",
            state=WorkerState.IDLE,
            tasks_completed=5,
            tasks_failed=0,
        )
        dispatcher._worker_counter = 2

        stats = dispatcher.get_worker_stats()

        assert stats["active_workers"] == 1  # Only RUNNING counts
        assert stats["available_slots"] == 2
        assert stats["total_workers_created"] == 2
        assert stats["total_tasks_completed"] == 8
        assert stats["total_tasks_failed"] == 1
        assert stats["state_distribution"] == {"running": 1, "idle": 1}
        assert len(stats["workers"]) == 2

    def test_stats_worker_details(self, dispatcher):
        """Worker detail entries should contain expected fields."""
        dispatcher._workers["w1"] = WorkerSlot(
            worker_id="w1",
            worktree_name="wt-1",
            worktree_path="/workspace/wt-1",
            state=WorkerState.RUNNING,
            task_id="task-1",
            session_id="session-1",
            tasks_completed=2,
            tasks_failed=0,
        )

        stats = dispatcher.get_worker_stats()
        detail = stats["workers"][0]

        assert detail["worker_id"] == "w1"
        assert detail["state"] == "running"
        assert detail["task_id"] == "task-1"
        assert detail["session_id"] == "session-1"
        assert detail["tasks_completed"] == 2
        assert detail["tasks_failed"] == 0
        assert detail["worktree_name"] == "wt-1"
        assert "uptime_seconds" in detail


# ---------------------------------------------------------------------------
# Task Selection Tests
# ---------------------------------------------------------------------------


class TestMultiWorkerTaskSelection:
    """Test multi-task selection for parallel dispatch."""

    @pytest.mark.asyncio
    async def test_select_ready_tasks_returns_multiple(
        self, dispatcher, mock_db_session, sample_project_id
    ):
        """Should return multiple ready tasks up to max_count."""
        tasks = [
            make_task(sample_project_id, f"Task {i}", priority=i)
            for i in range(5)
        ]

        with patch(
            "forgemaster.orchestrator.dispatcher.get_ready_tasks",
            new_callable=AsyncMock,
            return_value=tasks,
        ):
            selected = await dispatcher._select_ready_tasks(
                mock_db_session, sample_project_id, max_count=3
            )

        assert len(selected) == 3

    @pytest.mark.asyncio
    async def test_select_ready_tasks_filters_active(
        self, dispatcher, mock_db_session, sample_project_id
    ):
        """Should not select tasks already assigned to workers."""
        tasks = [
            make_task(sample_project_id, f"Task {i}", priority=i)
            for i in range(3)
        ]

        # Mark first task as already in a worker
        dispatcher._workers["w1"] = WorkerSlot(
            worker_id="w1",
            worktree_name="wt-1",
            worktree_path="/workspace/wt-1",
            task_id=str(tasks[0].id),
            state=WorkerState.RUNNING,
        )

        with patch(
            "forgemaster.orchestrator.dispatcher.get_ready_tasks",
            new_callable=AsyncMock,
            return_value=tasks,
        ):
            selected = await dispatcher._select_ready_tasks(
                mock_db_session, sample_project_id, max_count=3
            )

        assert len(selected) == 2
        selected_ids = {str(t.id) for t in selected}
        assert str(tasks[0].id) not in selected_ids

    @pytest.mark.asyncio
    async def test_select_ready_tasks_empty(
        self, dispatcher, mock_db_session, sample_project_id
    ):
        """Should return empty list when no tasks are ready."""
        with patch(
            "forgemaster.orchestrator.dispatcher.get_ready_tasks",
            new_callable=AsyncMock,
            return_value=[],
        ):
            selected = await dispatcher._select_ready_tasks(
                mock_db_session, sample_project_id, max_count=3
            )

        assert selected == []


# ---------------------------------------------------------------------------
# Task Assignment Tests (within multi-worker context)
# ---------------------------------------------------------------------------


class TestMultiWorkerTaskAssignment:
    """Test task assignment within the multi-worker dispatcher."""

    @pytest.mark.asyncio
    async def test_assign_task_transitions(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task
    ):
        """Assignment should transition READY -> ASSIGNED -> RUNNING."""
        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            session_id = await dispatcher._assign_task(sample_task, mock_db_session)

        assert session_id is not None

        transition_calls = mock_state_machine.transition.call_args_list
        assert len(transition_calls) == 2
        assert transition_calls[0][0][1] == TaskStatus.assigned
        assert transition_calls[1][0][1] == TaskStatus.running

    @pytest.mark.asyncio
    async def test_assign_task_returns_none_on_transition_error(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task
    ):
        """Should return None if state transition fails."""
        mock_state_machine.transition.side_effect = InvalidTransitionError(
            TaskStatus.ready, TaskStatus.assigned, str(sample_task.id)
        )

        session_id = await dispatcher._assign_task(sample_task, mock_db_session)

        assert session_id is None

    @pytest.mark.asyncio
    async def test_assign_task_starts_session(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Assignment should start an agent session."""
        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            session_id = await dispatcher._assign_task(sample_task, mock_db_session)

        assert session_id == "session-001"
        mock_session_manager.start_session.assert_called_once()


# ---------------------------------------------------------------------------
# Task Execution Tests
# ---------------------------------------------------------------------------


class TestMultiWorkerTaskExecution:
    """Test task execution within a worker."""

    @pytest.mark.asyncio
    async def test_execute_task_sends_message(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Should send task message to the agent session."""
        result = await dispatcher._execute_task(
            sample_task, "session-001", mock_db_session
        )

        mock_session_manager.send_message.assert_called_once()
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_execute_task_returns_failure_on_error(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Should return failure result if execution throws."""
        mock_session_manager.send_message.side_effect = RuntimeError("Connection lost")

        result = await dispatcher._execute_task(
            sample_task, "session-001", mock_db_session
        )

        assert result.status == "failed"
        assert result.confidence_score == 0.0


# ---------------------------------------------------------------------------
# Result Handling Tests
# ---------------------------------------------------------------------------


class TestMultiWorkerResultHandling:
    """Test result handling within multi-worker dispatcher."""

    @pytest.mark.asyncio
    async def test_success_transitions_to_review(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task
    ):
        """Success should transition task to REVIEW."""
        result = AgentResult(
            status="success",
            summary="Done",
            details="All good",
            confidence_score=0.95,
        )

        await dispatcher._default_handle_result(sample_task, result, mock_db_session)

        mock_state_machine.transition.assert_called_once_with(
            str(sample_task.id), TaskStatus.review, mock_db_session
        )

    @pytest.mark.asyncio
    async def test_failure_transitions_to_failed(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task
    ):
        """Failure should transition task to FAILED."""
        result = AgentResult(
            status="failed",
            summary="Broken",
            details="Error",
            confidence_score=0.0,
        )

        await dispatcher._default_handle_result(sample_task, result, mock_db_session)

        mock_state_machine.transition.assert_called_once_with(
            str(sample_task.id), TaskStatus.failed, mock_db_session
        )

    @pytest.mark.asyncio
    async def test_partial_does_not_transition(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task
    ):
        """Partial result should not trigger any transition."""
        result = AgentResult(
            status="partial",
            summary="Halfway",
            details="In progress",
            confidence_score=0.5,
        )

        await dispatcher._default_handle_result(sample_task, result, mock_db_session)

        mock_state_machine.transition.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_result_delegates_to_handler(
        self,
        agent_config,
        session_factory,
        mock_state_machine,
        mock_session_manager,
        mock_context_generator,
        mock_worktree_pool,
        mock_db_session,
        sample_task,
    ):
        """Should delegate to result_handler when configured."""
        mock_handler = AsyncMock()

        disp = MultiWorkerDispatcher(
            config=agent_config,
            session_factory=session_factory,
            state_machine=mock_state_machine,
            session_manager=mock_session_manager,
            context_generator=mock_context_generator,
            worktree_pool=mock_worktree_pool,
            result_handler=mock_handler,
            poll_interval=0.01,
        )

        result = AgentResult(
            status="success",
            summary="Done",
            details="All good",
            confidence_score=0.95,
        )

        await disp._handle_result(sample_task, "session-001", result, mock_db_session)

        mock_handler.handle_result.assert_called_once_with(
            sample_task, mock_db_session, result
        )


# ---------------------------------------------------------------------------
# Worker Run Tests
# ---------------------------------------------------------------------------


class TestWorkerRun:
    """Test the _run_worker coroutine."""

    @pytest.mark.asyncio
    async def test_worker_completes_successfully(
        self, dispatcher, sample_task
    ):
        """Worker should complete task and release itself."""
        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        # Wait for the worker to complete
        if worker_id in dispatcher._worker_tasks:
            await dispatcher._worker_tasks[worker_id]

        # Worker should have been released
        assert worker_id not in dispatcher._workers

    @pytest.mark.asyncio
    async def test_worker_handles_assignment_failure(
        self, dispatcher, mock_state_machine, sample_task
    ):
        """Worker should handle assignment failure gracefully."""
        mock_state_machine.transition.side_effect = InvalidTransitionError(
            TaskStatus.ready, TaskStatus.assigned, str(sample_task.id)
        )

        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        # Wait for the worker to complete
        if worker_id and worker_id in dispatcher._worker_tasks:
            await dispatcher._worker_tasks[worker_id]

        # Worker should have been released even on failure
        assert worker_id not in dispatcher._workers

    @pytest.mark.asyncio
    async def test_worker_handles_execution_error(
        self, dispatcher, mock_session_manager, sample_task
    ):
        """Worker should handle execution errors and still release."""
        mock_session_manager.send_message.side_effect = RuntimeError("Agent crashed")

        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            worker_id = await dispatcher._dispatch_to_worker(sample_task)

        # Wait for the worker to complete
        if worker_id and worker_id in dispatcher._worker_tasks:
            await dispatcher._worker_tasks[worker_id]

        # Worker should have been released even on error
        assert worker_id not in dispatcher._workers


# ---------------------------------------------------------------------------
# Graceful Shutdown Tests
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Test graceful shutdown with active workers."""

    @pytest.mark.asyncio
    async def test_stop_waits_for_active_workers(
        self, dispatcher, sample_project_id, mock_session_manager
    ):
        """Stop should wait for all running workers to complete."""
        # Make execution take a bit longer
        async def slow_response(session_id, message):
            await asyncio.sleep(0.05)
            return '{"status": "success", "summary": "Done", "details": "ok"}'

        mock_session_manager.send_message.side_effect = slow_response

        task = make_task(sample_project_id, "Slow task")

        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            await dispatcher.start(sample_project_id)
            await dispatcher._dispatch_to_worker(task)

        # Give the worker a moment to start
        await asyncio.sleep(0.01)

        # Stop should wait for the worker
        await dispatcher.stop()

        # After stop, no workers should remain
        assert len(dispatcher._workers) == 0
        assert len(dispatcher._worker_tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_handles_worker_errors_during_shutdown(
        self, dispatcher, sample_project_id, mock_session_manager
    ):
        """Stop should handle errors from workers during shutdown."""
        async def error_response(session_id, message):
            raise RuntimeError("Worker crashed during shutdown")

        mock_session_manager.send_message.side_effect = error_response

        task = make_task(sample_project_id, "Error task")

        with patch(
            "forgemaster.database.queries.session.create_session",
            new_callable=AsyncMock,
        ):
            await dispatcher.start(sample_project_id)
            await dispatcher._dispatch_to_worker(task)

        await asyncio.sleep(0.01)

        # Should not raise even with worker errors
        await dispatcher.stop()

        assert dispatcher.is_running is False


# ---------------------------------------------------------------------------
# Poll Loop Integration Tests
# ---------------------------------------------------------------------------


class TestMultiWorkerPollLoop:
    """Test the multi-worker poll loop behavior."""

    @pytest.mark.asyncio
    async def test_poll_loop_exits_when_stopped(
        self, dispatcher, sample_project_id
    ):
        """Poll loop should exit when _running is set to False."""
        dispatcher._running = True

        async def stop_after_one(*args, **kwargs):
            dispatcher._running = False
            return []

        with patch.object(
            dispatcher, "_select_ready_tasks", side_effect=stop_after_one
        ):
            await dispatcher._poll_loop(sample_project_id)

        assert dispatcher.is_running is False

    @pytest.mark.asyncio
    async def test_poll_loop_handles_exceptions(
        self, dispatcher, sample_project_id
    ):
        """Poll loop should survive exceptions and continue."""
        call_count = 0
        dispatcher._running = True

        async def error_then_stop(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient DB error")
            dispatcher._running = False
            return []

        with patch.object(
            dispatcher, "_select_ready_tasks", side_effect=error_then_stop
        ):
            await dispatcher._poll_loop(sample_project_id)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_poll_loop_dispatches_multiple_tasks(
        self, dispatcher, sample_project_id
    ):
        """Poll loop should dispatch multiple tasks when slots are available."""
        call_count = 0
        tasks = [
            make_task(sample_project_id, f"Task {i}", priority=i)
            for i in range(2)
        ]

        async def return_tasks_then_stop(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tasks
            dispatcher._running = False
            return []

        dispatch_calls = []

        original_dispatch = dispatcher._dispatch_to_worker

        async def track_dispatch(task):
            dispatch_calls.append(str(task.id))
            return await original_dispatch(task)

        dispatcher._running = True

        with (
            patch.object(
                dispatcher, "_select_ready_tasks", side_effect=return_tasks_then_stop
            ),
            patch.object(
                dispatcher, "_dispatch_to_worker", side_effect=track_dispatch
            ),
        ):
            await dispatcher._poll_loop(sample_project_id)

        assert len(dispatch_calls) == 2


# ---------------------------------------------------------------------------
# Model Tier and Message Building Tests
# ---------------------------------------------------------------------------


class TestMultiWorkerHelpers:
    """Test shared helper methods on MultiWorkerDispatcher."""

    def test_resolve_model_tier_sonnet(self, dispatcher):
        """Should resolve 'sonnet' tier."""
        assert "sonnet" in dispatcher._resolve_model_tier("sonnet")

    def test_resolve_model_tier_haiku(self, dispatcher):
        """Should resolve 'haiku' tier."""
        assert "haiku" in dispatcher._resolve_model_tier("haiku")

    def test_resolve_model_tier_opus(self, dispatcher):
        """Should resolve 'opus' tier."""
        assert "opus" in dispatcher._resolve_model_tier("opus")

    def test_resolve_model_tier_auto(self, dispatcher):
        """Should default to sonnet for 'auto'."""
        assert "sonnet" in dispatcher._resolve_model_tier("auto")

    def test_resolve_model_tier_none(self, dispatcher):
        """Should default to sonnet for None."""
        assert "sonnet" in dispatcher._resolve_model_tier(None)

    def test_build_task_message_includes_title(self, dispatcher, sample_task):
        """Message should include the task title."""
        message = dispatcher._build_task_message(sample_task)
        assert sample_task.title in message

    def test_build_task_message_includes_description(self, dispatcher, sample_task):
        """Message should include the task description."""
        message = dispatcher._build_task_message(sample_task)
        assert sample_task.description in message

    def test_build_task_message_includes_files(self, dispatcher, sample_task):
        """Message should include files to modify."""
        message = dispatcher._build_task_message(sample_task)
        assert "src/feature_x.py" in message

    def test_build_project_info_without_project(self, dispatcher, sample_task):
        """Should return defaults when task has no project."""
        info = dispatcher._build_project_info(sample_task)
        assert info["name"] == "Unknown Project"
