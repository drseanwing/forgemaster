"""Task state machine for Forgemaster orchestrator.

This module implements the task lifecycle state machine, handling state transitions,
validation, and dependency resolution for tasks in the orchestrator.

The state machine enforces valid transitions between task states and ensures that
tasks only become ready when all their dependencies are complete.
"""

from __future__ import annotations

import structlog
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from forgemaster.database.models.task import Task, TaskStatus

logger = structlog.get_logger(__name__)


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted.

    Attributes:
        current: The current task status.
        target: The attempted target status.
        task_id: The ID of the task that failed to transition.
    """

    def __init__(self, current: TaskStatus, target: TaskStatus, task_id: str | None = None):
        self.current = current
        self.target = target
        self.task_id = task_id
        msg = f"Invalid transition from {current.value} to {target.value}"
        if task_id:
            msg += f" for task {task_id}"
        super().__init__(msg)


# Authoritative state machine definition
VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.pending: {TaskStatus.ready},
    TaskStatus.ready: {TaskStatus.assigned},
    TaskStatus.assigned: {TaskStatus.running},
    TaskStatus.running: {TaskStatus.review, TaskStatus.failed, TaskStatus.blocked},
    TaskStatus.review: {TaskStatus.done, TaskStatus.running},
    TaskStatus.done: set(),  # Terminal state - no transitions allowed
    TaskStatus.failed: {TaskStatus.ready},
    TaskStatus.blocked: {TaskStatus.ready},
}


def validate_transition(current: TaskStatus, target: TaskStatus) -> bool:
    """Validate if a state transition is allowed.

    Args:
        current: Current task status.
        target: Target task status.

    Returns:
        True if the transition is valid according to VALID_TRANSITIONS.
    """
    return target in VALID_TRANSITIONS.get(current, set())


class TaskStateMachine:
    """Manages task state transitions with validation and side effects.

    This class handles:
    - Validation of state transitions
    - Updating task status in the database
    - Setting timestamps (started_at, completed_at)
    - Logging all transitions
    - Dependency resolution
    """

    def __init__(self):
        """Initialize the task state machine."""
        self.logger = logger.bind(component="TaskStateMachine")

    async def transition(
        self,
        task_id: str,
        target_status: TaskStatus,
        session: AsyncSession
    ) -> Task:
        """Transition a task to a new status.

        Args:
            task_id: UUID of the task to transition.
            target_status: Target status for the task.
            session: Database session for the transaction.

        Returns:
            The updated Task object.

        Raises:
            InvalidTransitionError: If the transition is not valid.
            ValueError: If the task does not exist.
        """
        # Fetch the task
        result = await session.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()

        if task is None:
            raise ValueError(f"Task {task_id} not found")

        current_status = task.status

        # Validate transition
        if not validate_transition(current_status, target_status):
            raise InvalidTransitionError(current_status, target_status, str(task_id))

        # Update status
        task.status = target_status

        # Set timestamps based on target status
        if target_status == TaskStatus.running and task.started_at is None:
            task.started_at = datetime.now(timezone.utc)

        if target_status == TaskStatus.done:
            task.completed_at = datetime.now(timezone.utc)

        # Log the transition
        self.logger.info(
            "task_transition",
            task_id=str(task_id),
            from_status=current_status.value,
            to_status=target_status.value,
            started_at=task.started_at.isoformat() if task.started_at else None,
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
        )

        # Commit is handled by caller
        await session.flush()

        return task

    async def resolve_dependencies(self, task: Task, session: AsyncSession) -> bool:
        """Check if all dependencies for a task are resolved.

        Args:
            task: The task to check dependencies for.
            session: Database session for queries.

        Returns:
            True if all dependencies are DONE or if there are no dependencies.
        """
        if not task.dependencies or len(task.dependencies) == 0:
            return True

        # Query all dependency tasks
        result = await session.execute(
            select(Task).where(Task.id.in_(task.dependencies))
        )
        dependency_tasks = result.scalars().all()

        # Check if all dependencies are DONE
        all_done = all(dep.status == TaskStatus.done for dep in dependency_tasks)

        if all_done:
            self.logger.debug(
                "dependencies_resolved",
                task_id=str(task.id),
                dependency_count=len(task.dependencies),
            )
        else:
            pending_deps = [
                str(dep.id) for dep in dependency_tasks
                if dep.status != TaskStatus.done
            ]
            self.logger.debug(
                "dependencies_unresolved",
                task_id=str(task.id),
                pending_dependencies=pending_deps,
            )

        return all_done

    async def update_pending_tasks(
        self,
        project_id: str,
        session: AsyncSession
    ) -> list[Task]:
        """Find and transition PENDING tasks whose dependencies are now complete.

        This should be called after any task transitions to DONE to check if
        downstream tasks can now proceed.

        Args:
            project_id: UUID of the project to check.
            session: Database session for the transaction.

        Returns:
            List of tasks that were transitioned from PENDING to READY.
        """
        # Find all PENDING tasks in the project
        result = await session.execute(
            select(Task).where(
                Task.project_id == project_id,
                Task.status == TaskStatus.pending,
            )
        )
        pending_tasks = result.scalars().all()

        newly_ready: list[Task] = []

        for task in pending_tasks:
            if await self.resolve_dependencies(task, session):
                # Transition to READY
                try:
                    updated_task = await self.transition(
                        str(task.id),
                        TaskStatus.ready,
                        session
                    )
                    newly_ready.append(updated_task)
                except InvalidTransitionError as e:
                    # Log but don't fail - might be a race condition
                    self.logger.warning(
                        "failed_to_transition_pending_task",
                        task_id=str(task.id),
                        error=str(e),
                    )

        if newly_ready:
            self.logger.info(
                "pending_tasks_resolved",
                project_id=project_id,
                count=len(newly_ready),
                task_ids=[str(t.id) for t in newly_ready],
            )

        return newly_ready
