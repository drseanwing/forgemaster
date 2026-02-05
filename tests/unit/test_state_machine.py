"""Unit tests for task state machine.

Tests cover:
- Valid state transitions
- Invalid state transition handling
- Timestamp setting on transitions
- Dependency resolution logic
- Batch pending-to-ready updates
- Transition logging
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.orchestrator.state_machine import (
    InvalidTransitionError,
    TaskStateMachine,
    VALID_TRANSITIONS,
    validate_transition,
)


class TestValidTransitions:
    """Test the VALID_TRANSITIONS mapping and validation."""

    def test_valid_transitions_definition(self):
        """Verify VALID_TRANSITIONS includes all TaskStatus values."""
        all_statuses = set(TaskStatus)
        defined_statuses = set(VALID_TRANSITIONS.keys())
        assert all_statuses == defined_statuses, "Missing status in VALID_TRANSITIONS"

    @pytest.mark.parametrize(
        "current,target,expected",
        [
            # Valid transitions
            (TaskStatus.pending, TaskStatus.ready, True),
            (TaskStatus.ready, TaskStatus.assigned, True),
            (TaskStatus.assigned, TaskStatus.running, True),
            (TaskStatus.running, TaskStatus.review, True),
            (TaskStatus.running, TaskStatus.failed, True),
            (TaskStatus.running, TaskStatus.blocked, True),
            (TaskStatus.review, TaskStatus.done, True),
            (TaskStatus.review, TaskStatus.running, True),
            (TaskStatus.failed, TaskStatus.ready, True),
            (TaskStatus.blocked, TaskStatus.ready, True),
            # Invalid transitions
            (TaskStatus.pending, TaskStatus.running, False),
            (TaskStatus.ready, TaskStatus.done, False),
            (TaskStatus.assigned, TaskStatus.done, False),
            (TaskStatus.done, TaskStatus.running, False),
            (TaskStatus.done, TaskStatus.pending, False),
            (TaskStatus.failed, TaskStatus.done, False),
        ],
    )
    def test_validate_transition(self, current, target, expected):
        """Test validate_transition for various state combinations."""
        result = validate_transition(current, target)
        assert result == expected


class TestInvalidTransitionError:
    """Test the InvalidTransitionError exception."""

    def test_error_without_task_id(self):
        """Test error message without task ID."""
        error = InvalidTransitionError(TaskStatus.pending, TaskStatus.done)
        assert "pending" in str(error)
        assert "done" in str(error)
        assert error.current == TaskStatus.pending
        assert error.target == TaskStatus.done
        assert error.task_id is None

    def test_error_with_task_id(self):
        """Test error message with task ID."""
        task_id = str(uuid.uuid4())
        error = InvalidTransitionError(
            TaskStatus.pending,
            TaskStatus.done,
            task_id
        )
        assert task_id in str(error)
        assert error.task_id == task_id


class TestTaskStateMachine:
    """Test the TaskStateMachine class."""

    @pytest.fixture
    def state_machine(self):
        """Create a TaskStateMachine instance."""
        return TaskStateMachine()

    @pytest.fixture
    def mock_session(self):
        """Create a mock AsyncSession."""
        session = AsyncMock(spec=AsyncSession)
        session.flush = AsyncMock()
        return session

    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        task = Task(
            id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            title="Test Task",
            description="Test description",
            status=TaskStatus.pending,
            agent_type="executor",
        )
        return task

    @pytest.mark.asyncio
    async def test_transition_valid(self, state_machine, mock_session, sample_task):
        """Test a valid state transition."""
        # Setup mock to return the task
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Transition from pending to ready
        updated_task = await state_machine.transition(
            str(sample_task.id),
            TaskStatus.ready,
            mock_session
        )

        assert updated_task.status == TaskStatus.ready
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_transition_invalid(self, state_machine, mock_session, sample_task):
        """Test an invalid state transition raises error."""
        # Setup mock to return the task
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Attempt invalid transition from pending to done
        with pytest.raises(InvalidTransitionError) as exc_info:
            await state_machine.transition(
                str(sample_task.id),
                TaskStatus.done,
                mock_session
            )

        assert exc_info.value.current == TaskStatus.pending
        assert exc_info.value.target == TaskStatus.done

    @pytest.mark.asyncio
    async def test_transition_task_not_found(self, state_machine, mock_session):
        """Test transition raises ValueError for non-existent task."""
        # Setup mock to return None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="not found"):
            await state_machine.transition(
                str(uuid.uuid4()),
                TaskStatus.ready,
                mock_session
            )

    @pytest.mark.asyncio
    async def test_transition_sets_started_at(self, state_machine, mock_session, sample_task):
        """Test transition to RUNNING sets started_at timestamp."""
        sample_task.status = TaskStatus.assigned
        sample_task.started_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute = AsyncMock(return_value=mock_result)

        updated_task = await state_machine.transition(
            str(sample_task.id),
            TaskStatus.running,
            mock_session
        )

        assert updated_task.started_at is not None
        assert isinstance(updated_task.started_at, datetime)

    @pytest.mark.asyncio
    async def test_transition_preserves_started_at(self, state_machine, mock_session, sample_task):
        """Test transition to RUNNING preserves existing started_at."""
        sample_task.status = TaskStatus.review
        original_time = datetime.now(timezone.utc)
        sample_task.started_at = original_time

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute = AsyncMock(return_value=mock_result)

        updated_task = await state_machine.transition(
            str(sample_task.id),
            TaskStatus.running,
            mock_session
        )

        assert updated_task.started_at == original_time

    @pytest.mark.asyncio
    async def test_transition_sets_completed_at(self, state_machine, mock_session, sample_task):
        """Test transition to DONE sets completed_at timestamp."""
        sample_task.status = TaskStatus.review
        sample_task.completed_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute = AsyncMock(return_value=mock_result)

        updated_task = await state_machine.transition(
            str(sample_task.id),
            TaskStatus.done,
            mock_session
        )

        assert updated_task.completed_at is not None
        assert isinstance(updated_task.completed_at, datetime)

    @pytest.mark.asyncio
    async def test_transition_logging(self, state_machine, mock_session, sample_task):
        """Test that transitions are logged."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(state_machine.logger, "info") as mock_log:
            await state_machine.transition(
                str(sample_task.id),
                TaskStatus.ready,
                mock_session
            )

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == "task_transition"
            assert call_args[1]["from_status"] == "pending"
            assert call_args[1]["to_status"] == "ready"


class TestDependencyResolution:
    """Test dependency resolution logic."""

    @pytest.fixture
    def state_machine(self):
        """Create a TaskStateMachine instance."""
        return TaskStateMachine()

    @pytest.fixture
    def mock_session(self):
        """Create a mock AsyncSession."""
        return AsyncMock(spec=AsyncSession)

    @pytest.mark.asyncio
    async def test_resolve_dependencies_no_deps(self, state_machine, mock_session):
        """Test task with no dependencies is immediately resolved."""
        task = Task(
            id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            title="Test",
            status=TaskStatus.pending,
            agent_type="executor",
            dependencies=None,
        )

        result = await state_machine.resolve_dependencies(task, mock_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_resolve_dependencies_empty_list(self, state_machine, mock_session):
        """Test task with empty dependencies list is immediately resolved."""
        task = Task(
            id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            title="Test",
            status=TaskStatus.pending,
            agent_type="executor",
            dependencies=[],
        )

        result = await state_machine.resolve_dependencies(task, mock_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_resolve_dependencies_all_done(self, state_machine, mock_session):
        """Test task with all dependencies DONE resolves successfully."""
        dep1_id = uuid.uuid4()
        dep2_id = uuid.uuid4()

        task = Task(
            id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            title="Test",
            status=TaskStatus.pending,
            agent_type="executor",
            dependencies=[dep1_id, dep2_id],
        )

        dep1 = Task(
            id=dep1_id,
            project_id=task.project_id,
            title="Dep 1",
            status=TaskStatus.done,
            agent_type="executor",
        )
        dep2 = Task(
            id=dep2_id,
            project_id=task.project_id,
            title="Dep 2",
            status=TaskStatus.done,
            agent_type="executor",
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [dep1, dep2]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await state_machine.resolve_dependencies(task, mock_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_resolve_dependencies_some_pending(self, state_machine, mock_session):
        """Test task with some pending dependencies does not resolve."""
        dep1_id = uuid.uuid4()
        dep2_id = uuid.uuid4()

        task = Task(
            id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            title="Test",
            status=TaskStatus.pending,
            agent_type="executor",
            dependencies=[dep1_id, dep2_id],
        )

        dep1 = Task(
            id=dep1_id,
            project_id=task.project_id,
            title="Dep 1",
            status=TaskStatus.done,
            agent_type="executor",
        )
        dep2 = Task(
            id=dep2_id,
            project_id=task.project_id,
            title="Dep 2",
            status=TaskStatus.running,
            agent_type="executor",
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [dep1, dep2]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await state_machine.resolve_dependencies(task, mock_session)
        assert result is False


class TestUpdatePendingTasks:
    """Test batch pending-to-ready update logic."""

    @pytest.fixture
    def state_machine(self):
        """Create a TaskStateMachine instance."""
        return TaskStateMachine()

    @pytest.fixture
    def mock_session(self):
        """Create a mock AsyncSession."""
        return AsyncMock(spec=AsyncSession)

    @pytest.mark.asyncio
    async def test_update_pending_tasks_no_pending(self, state_machine, mock_session):
        """Test with no pending tasks returns empty list."""
        project_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        newly_ready = await state_machine.update_pending_tasks(
            str(project_id),
            mock_session
        )

        assert newly_ready == []

    @pytest.mark.asyncio
    async def test_update_pending_tasks_with_resolved_deps(
        self,
        state_machine,
        mock_session
    ):
        """Test pending tasks with resolved dependencies transition to ready."""
        project_id = uuid.uuid4()
        dep_id = uuid.uuid4()
        pending_id = uuid.uuid4()

        pending_task = Task(
            id=pending_id,
            project_id=project_id,
            title="Pending",
            status=TaskStatus.pending,
            agent_type="executor",
            dependencies=[dep_id],
        )

        dep_task = Task(
            id=dep_id,
            project_id=project_id,
            title="Dependency",
            status=TaskStatus.done,
            agent_type="executor",
        )

        # Mock the pending tasks query
        pending_result = MagicMock()
        pending_result.scalars.return_value.all.return_value = [pending_task]

        # Mock the dependency query
        dep_result = MagicMock()
        dep_result.scalars.return_value.all.return_value = [dep_task]

        # Mock the task fetch for transition
        task_fetch_result = MagicMock()
        task_fetch_result.scalar_one_or_none.return_value = pending_task

        async def execute_side_effect(*args, **kwargs):
            """Return different results based on query type."""
            # Simple heuristic - first call is pending tasks, second is dependencies
            if mock_session.execute.call_count == 1:
                return pending_result
            elif mock_session.execute.call_count == 2:
                return dep_result
            else:
                return task_fetch_result

        mock_session.execute = AsyncMock(side_effect=execute_side_effect)

        newly_ready = await state_machine.update_pending_tasks(
            str(project_id),
            mock_session
        )

        assert len(newly_ready) == 1
        assert newly_ready[0].status == TaskStatus.ready

    @pytest.mark.asyncio
    async def test_update_pending_tasks_with_unresolved_deps(
        self,
        state_machine,
        mock_session
    ):
        """Test pending tasks with unresolved dependencies stay pending."""
        project_id = uuid.uuid4()
        dep_id = uuid.uuid4()
        pending_id = uuid.uuid4()

        pending_task = Task(
            id=pending_id,
            project_id=project_id,
            title="Pending",
            status=TaskStatus.pending,
            agent_type="executor",
            dependencies=[dep_id],
        )

        dep_task = Task(
            id=dep_id,
            project_id=project_id,
            title="Dependency",
            status=TaskStatus.running,  # Not done yet
            agent_type="executor",
        )

        # Mock the pending tasks query
        pending_result = MagicMock()
        pending_result.scalars.return_value.all.return_value = [pending_task]

        # Mock the dependency query
        dep_result = MagicMock()
        dep_result.scalars.return_value.all.return_value = [dep_task]

        async def execute_side_effect(*args, **kwargs):
            """Return different results based on query type."""
            if mock_session.execute.call_count == 1:
                return pending_result
            else:
                return dep_result

        mock_session.execute = AsyncMock(side_effect=execute_side_effect)

        newly_ready = await state_machine.update_pending_tasks(
            str(project_id),
            mock_session
        )

        assert newly_ready == []
        assert pending_task.status == TaskStatus.pending
