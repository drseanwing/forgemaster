"""Task CRUD query functions for Forgemaster.

Provides async functions for creating, reading, updating, and managing
Task records, including dependency-aware task selection and status updates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.task import Task, TaskStatus

logger = structlog.get_logger(__name__)


async def create_task(
    session: AsyncSession,
    project_id: UUID,
    title: str,
    agent_type: str,
    description: str | None = None,
    model_tier: str | None = "auto",
    priority: int = 100,
    estimated_minutes: int | None = None,
    files_touched: list[str] | None = None,
    dependencies: list[UUID] | None = None,
    parallel_group: str | None = None,
    max_retries: int = 3,
) -> Task:
    """Create a new task.

    Args:
        session: Active async database session.
        project_id: UUID of the parent project.
        title: Short task description.
        agent_type: Type of agent required (e.g., 'executor', 'architect').
        description: Detailed task description and instructions.
        model_tier: Model tier preference ('auto', 'haiku', 'sonnet', 'opus').
        priority: Numeric priority (lower = higher priority).
        estimated_minutes: Estimated completion time in minutes.
        files_touched: List of file paths this task will modify.
        dependencies: List of task UUIDs this task depends on.
        parallel_group: Optional group identifier for parallel execution.
        max_retries: Maximum allowed retry attempts.

    Returns:
        The newly created Task instance.
    """
    task = Task(
        project_id=project_id,
        title=title,
        description=description,
        agent_type=agent_type,
        model_tier=model_tier,
        priority=priority,
        estimated_minutes=estimated_minutes,
        files_touched=files_touched,
        dependencies=dependencies,
        parallel_group=parallel_group,
        max_retries=max_retries,
        status=TaskStatus.pending,
        retry_count=0,
    )

    async with session.begin():
        session.add(task)
        await session.flush()
        await session.refresh(task)

    logger.info(
        "task_created",
        task_id=str(task.id),
        project_id=str(project_id),
        title=title,
        agent_type=agent_type,
        status=task.status.value,
    )

    return task


async def get_task(
    session: AsyncSession,
    task_id: UUID,
) -> Task | None:
    """Retrieve a task by ID.

    Args:
        session: Active async database session.
        task_id: UUID of the task to retrieve.

    Returns:
        The Task instance if found, None otherwise.
    """
    stmt = select(Task).where(Task.id == task_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_tasks(
    session: AsyncSession,
    project_id: UUID | None = None,
    status_filter: TaskStatus | None = None,
    priority_sort: bool = True,
) -> list[Task]:
    """List tasks with optional filters and sorting.

    Args:
        session: Active async database session.
        project_id: Optional project UUID to filter by.
        status_filter: Optional status to filter by.
        priority_sort: If True, sort by priority (lowest first). If False, sort by created_at.

    Returns:
        List of matching Task instances.
    """
    stmt = select(Task)

    if project_id is not None:
        stmt = stmt.where(Task.project_id == project_id)

    if status_filter is not None:
        stmt = stmt.where(Task.status == status_filter)

    if priority_sort:
        stmt = stmt.order_by(Task.priority.asc(), Task.created_at.asc())
    else:
        stmt = stmt.order_by(Task.created_at.desc())

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_task_status(
    session: AsyncSession,
    task_id: UUID,
    new_status: TaskStatus,
) -> Task:
    """Update a task's status and set appropriate timestamps.

    Args:
        session: Active async database session.
        task_id: UUID of the task to update.
        new_status: The new TaskStatus to set.

    Returns:
        The updated Task instance.

    Raises:
        ValueError: If task not found.
    """
    task = await get_task(session, task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")

    updates: dict[str, Any] = {"status": new_status}

    # Set started_at when transitioning to running
    if new_status == TaskStatus.running and task.started_at is None:
        updates["started_at"] = datetime.now(timezone.utc)

    # Set completed_at when reaching terminal states
    if new_status in {TaskStatus.done, TaskStatus.failed} and task.completed_at is None:
        updates["completed_at"] = datetime.now(timezone.utc)

    async with session.begin():
        stmt = update(Task).where(Task.id == task_id).values(**updates)
        await session.execute(stmt)
        await session.refresh(task)

    logger.info(
        "task_status_updated",
        task_id=str(task_id),
        old_status=task.status.value,
        new_status=new_status.value,
    )

    return task


async def get_ready_tasks(
    session: AsyncSession,
    project_id: UUID,
) -> list[Task]:
    """Get tasks that are ready to execute (all dependencies DONE).

    A task is ready if:
    - It is currently in 'pending' or 'ready' status
    - All of its dependencies (if any) are in 'done' status

    Args:
        session: Active async database session.
        project_id: UUID of the project to search within.

    Returns:
        List of ready Task instances, sorted by priority.
    """
    # Fetch all tasks in pending/ready status for this project
    stmt = (
        select(Task)
        .where(Task.project_id == project_id)
        .where(Task.status.in_([TaskStatus.pending, TaskStatus.ready]))
    )
    result = await session.execute(stmt)
    candidate_tasks = list(result.scalars().all())

    # Fetch all tasks from this project to check dependency status
    all_tasks_stmt = select(Task).where(Task.project_id == project_id)
    all_tasks_result = await session.execute(all_tasks_stmt)
    all_tasks = {task.id: task for task in all_tasks_result.scalars().all()}

    ready_tasks = []
    for task in candidate_tasks:
        # If task has no dependencies, it's ready
        if not task.dependencies:
            ready_tasks.append(task)
            continue

        # Check if all dependencies are done
        all_done = all(
            all_tasks.get(dep_id) and all_tasks[dep_id].status == TaskStatus.done
            for dep_id in task.dependencies
        )

        if all_done:
            ready_tasks.append(task)

    # Sort by priority
    ready_tasks.sort(key=lambda t: (t.priority, t.created_at))

    return ready_tasks


async def get_next_task(
    session: AsyncSession,
    project_id: UUID,
) -> Task | None:
    """Get the next highest priority task that is ready to execute.

    Args:
        session: Active async database session.
        project_id: UUID of the project to search within.

    Returns:
        The highest priority ready Task, or None if no tasks are ready.
    """
    ready_tasks = await get_ready_tasks(session, project_id)
    return ready_tasks[0] if ready_tasks else None


async def increment_retry_count(
    session: AsyncSession,
    task_id: UUID,
) -> Task:
    """Increment a task's retry count.

    Args:
        session: Active async database session.
        task_id: UUID of the task to update.

    Returns:
        The updated Task instance.

    Raises:
        ValueError: If task not found.
    """
    task = await get_task(session, task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")

    new_count = task.retry_count + 1

    async with session.begin():
        stmt = (
            update(Task)
            .where(Task.id == task_id)
            .values(retry_count=new_count)
        )
        await session.execute(stmt)
        await session.refresh(task)

    logger.info(
        "task_retry_incremented",
        task_id=str(task_id),
        retry_count=new_count,
        max_retries=task.max_retries,
    )

    return task
