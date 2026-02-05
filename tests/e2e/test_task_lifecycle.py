"""End-to-end tests for task lifecycle flows.

Tests the complete task lifecycle from creation through all states
to completion, including state transitions, retries, blocking, and
lesson extraction.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.orchestrator.state_machine import InvalidTransitionError, TaskStateMachine


@pytest.mark.e2e
@pytest.mark.asyncio
class TestTaskCreationAndAssignment:
    """Test task creation and assignment flows."""

    async def test_task_created_in_pending_state(
        self,
        sample_task_pending: Task,
    ) -> None:
        """Test that newly created tasks start in PENDING state."""
        assert sample_task_pending.status == TaskStatus.pending
        assert sample_task_pending.started_at is None
        assert sample_task_pending.completed_at is None
        assert sample_task_pending.retry_count == 0

    async def test_task_ready_for_assignment(
        self,
        sample_task_ready: Task,
    ) -> None:
        """Test that tasks can transition to READY state."""
        assert sample_task_ready.status == TaskStatus.ready
        assert sample_task_ready.started_at is None
        assert sample_task_ready.completed_at is None


@pytest.mark.e2e
@pytest.mark.asyncio
class TestTaskStateTransitions:
    """Test valid and invalid task state transitions."""

    async def test_valid_transition_pending_to_ready(
        self,
        sample_task_pending: Task,
        e2e_session,
    ) -> None:
        """Test valid transition from PENDING to READY."""
        state_machine = TaskStateMachine()

        updated_task = await state_machine.transition(
            task_id=str(sample_task_pending.id),
            target_status=TaskStatus.ready,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.ready

    async def test_valid_transition_ready_to_assigned(
        self,
        sample_task_ready: Task,
        e2e_session,
    ) -> None:
        """Test valid transition from READY to ASSIGNED."""
        state_machine = TaskStateMachine()

        updated_task = await state_machine.transition(
            task_id=str(sample_task_ready.id),
            target_status=TaskStatus.assigned,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.assigned

    async def test_valid_transition_assigned_to_running(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test valid transition from ASSIGNED to RUNNING with timestamp."""
        # Create task in ASSIGNED state
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Assigned task",
            status=TaskStatus.assigned,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        state_machine = TaskStateMachine()
        updated_task = await state_machine.transition(
            task_id=str(task.id),
            target_status=TaskStatus.running,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.running
        assert updated_task.started_at is not None

    async def test_valid_transition_running_to_review(
        self,
        sample_task_running: Task,
        e2e_session,
    ) -> None:
        """Test valid transition from RUNNING to REVIEW."""
        state_machine = TaskStateMachine()

        updated_task = await state_machine.transition(
            task_id=str(sample_task_running.id),
            target_status=TaskStatus.review,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.review

    async def test_valid_transition_review_to_done(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test valid transition from REVIEW to DONE with completion timestamp."""
        # Create task in REVIEW state
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task in review",
            status=TaskStatus.review,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        state_machine = TaskStateMachine()
        updated_task = await state_machine.transition(
            task_id=str(task.id),
            target_status=TaskStatus.done,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.done
        assert updated_task.completed_at is not None

    async def test_invalid_transition_pending_to_running(
        self,
        sample_task_pending: Task,
        e2e_session,
    ) -> None:
        """Test that invalid transition from PENDING to RUNNING raises error."""
        state_machine = TaskStateMachine()

        with pytest.raises(InvalidTransitionError) as exc_info:
            await state_machine.transition(
                task_id=str(sample_task_pending.id),
                target_status=TaskStatus.running,
                session=e2e_session,
            )

        assert exc_info.value.current == TaskStatus.pending
        assert exc_info.value.target == TaskStatus.running

    async def test_done_state_is_terminal(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that DONE state cannot transition to any other state."""
        # Create task in DONE state
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Completed task",
            status=TaskStatus.done,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        state_machine = TaskStateMachine()

        with pytest.raises(InvalidTransitionError):
            await state_machine.transition(
                task_id=str(task.id),
                target_status=TaskStatus.running,
                session=e2e_session,
            )


@pytest.mark.e2e
@pytest.mark.asyncio
class TestTaskFailureAndRetry:
    """Test task failure and retry flows."""

    async def test_task_failure_transition(
        self,
        sample_task_running: Task,
        e2e_session,
    ) -> None:
        """Test transition from RUNNING to FAILED."""
        state_machine = TaskStateMachine()

        updated_task = await state_machine.transition(
            task_id=str(sample_task_running.id),
            target_status=TaskStatus.failed,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.failed

    async def test_failed_task_retry_transition(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that failed task can transition back to READY for retry."""
        # Create task in FAILED state
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Failed task",
            status=TaskStatus.failed,
            agent_type="executor",
            max_retries=3,
            retry_count=1,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        state_machine = TaskStateMachine()
        updated_task = await state_machine.transition(
            task_id=str(task.id),
            target_status=TaskStatus.ready,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.ready

    async def test_task_exhausts_retries(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that task with max retries stays in FAILED state."""
        # Create task that has exhausted retries
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task with exhausted retries",
            status=TaskStatus.failed,
            agent_type="executor",
            max_retries=3,
            retry_count=3,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        # Verify retry count equals max retries
        assert task.retry_count == task.max_retries
        assert task.status == TaskStatus.failed


@pytest.mark.e2e
@pytest.mark.asyncio
class TestTaskBlockingAndUnblocking:
    """Test task blocking and dependency flows."""

    async def test_task_blocked_transition(
        self,
        sample_task_running: Task,
        e2e_session,
    ) -> None:
        """Test transition from RUNNING to BLOCKED."""
        state_machine = TaskStateMachine()

        updated_task = await state_machine.transition(
            task_id=str(sample_task_running.id),
            target_status=TaskStatus.blocked,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.blocked

    async def test_blocked_task_unblock_transition(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that blocked task can transition back to READY."""
        # Create task in BLOCKED state
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Blocked task",
            status=TaskStatus.blocked,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        state_machine = TaskStateMachine()
        updated_task = await state_machine.transition(
            task_id=str(task.id),
            target_status=TaskStatus.ready,
            session=e2e_session,
        )

        assert updated_task.status == TaskStatus.ready

    async def test_task_with_dependencies_stays_pending(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that task with unmet dependencies stays in PENDING."""
        # Create dependency task
        dep_task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Dependency task",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(dep_task)
        await e2e_session.commit()

        # Create task that depends on it
        blocked_task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task with dependencies",
            status=TaskStatus.pending,
            agent_type="executor",
            dependencies=[dep_task.id],
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(blocked_task)
        await e2e_session.commit()
        await e2e_session.refresh(blocked_task)

        assert blocked_task.status == TaskStatus.pending
        assert blocked_task.dependencies == [dep_task.id]


@pytest.mark.e2e
@pytest.mark.asyncio
class TestFullTaskLifecycle:
    """Test complete task lifecycle flows."""

    async def test_complete_happy_path_lifecycle(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test complete lifecycle: PENDING → READY → ASSIGNED → RUNNING → REVIEW → DONE."""
        # Create task in PENDING
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Full lifecycle task",
            description="Test complete lifecycle",
            status=TaskStatus.pending,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        state_machine = TaskStateMachine()

        # PENDING → READY
        task = await state_machine.transition(str(task.id), TaskStatus.ready, e2e_session)
        assert task.status == TaskStatus.ready

        # READY → ASSIGNED
        task = await state_machine.transition(str(task.id), TaskStatus.assigned, e2e_session)
        assert task.status == TaskStatus.assigned

        # ASSIGNED → RUNNING
        task = await state_machine.transition(str(task.id), TaskStatus.running, e2e_session)
        assert task.status == TaskStatus.running
        assert task.started_at is not None

        # RUNNING → REVIEW
        task = await state_machine.transition(str(task.id), TaskStatus.review, e2e_session)
        assert task.status == TaskStatus.review

        # REVIEW → DONE
        task = await state_machine.transition(str(task.id), TaskStatus.done, e2e_session)
        assert task.status == TaskStatus.done
        assert task.completed_at is not None

    async def test_lifecycle_with_single_retry(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test lifecycle with failure and retry: RUNNING → FAILED → READY → RUNNING → DONE."""
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task with retry",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        state_machine = TaskStateMachine()

        # RUNNING → FAILED
        task = await state_machine.transition(str(task.id), TaskStatus.failed, e2e_session)
        assert task.status == TaskStatus.failed

        # FAILED → READY (retry)
        task = await state_machine.transition(str(task.id), TaskStatus.ready, e2e_session)
        assert task.status == TaskStatus.ready

        # READY → ASSIGNED → RUNNING
        task = await state_machine.transition(str(task.id), TaskStatus.assigned, e2e_session)
        task = await state_machine.transition(str(task.id), TaskStatus.running, e2e_session)
        assert task.status == TaskStatus.running

        # RUNNING → REVIEW → DONE
        task = await state_machine.transition(str(task.id), TaskStatus.review, e2e_session)
        task = await state_machine.transition(str(task.id), TaskStatus.done, e2e_session)
        assert task.status == TaskStatus.done
        assert task.completed_at is not None
