"""Integration tests for task query functions.

Tests all task CRUD operations including creation with dependencies,
status updates with timestamp tracking, dependency resolution for ready tasks,
priority-based ordering, and retry count management.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.task import TaskStatus
from forgemaster.database.queries.project import create_project
from forgemaster.database.queries.task import (
    create_task,
    get_next_task,
    get_ready_tasks,
    get_task,
    increment_retry_count,
    list_tasks,
    update_task_status,
)


@pytest.mark.asyncio
async def test_create_task_with_all_fields(db_session: AsyncSession) -> None:
    """Test creating a task with all optional fields populated."""
    project = await create_project(db_session, name="Test Project", config={})
    dependency_ids = [uuid4(), uuid4()]

    task = await create_task(
        db_session,
        project_id=project.id,
        title="Implement authentication service",
        agent_type="executor",
        description="Build JWT-based authentication with refresh tokens",
        model_tier="sonnet",
        priority=50,
        estimated_minutes=120,
        files_touched=["src/auth/jwt.py", "src/auth/middleware.py"],
        dependencies=dependency_ids,
        parallel_group="auth-services",
        max_retries=5,
    )

    assert task.id is not None
    assert isinstance(task.id, UUID)
    assert task.project_id == project.id
    assert task.title == "Implement authentication service"
    assert task.description == "Build JWT-based authentication with refresh tokens"
    assert task.agent_type == "executor"
    assert task.model_tier == "sonnet"
    assert task.priority == 50
    assert task.estimated_minutes == 120
    assert task.files_touched == ["src/auth/jwt.py", "src/auth/middleware.py"]
    assert task.dependencies == dependency_ids
    assert task.parallel_group == "auth-services"
    assert task.max_retries == 5
    assert task.status == TaskStatus.pending
    assert task.retry_count == 0
    assert task.started_at is None
    assert task.completed_at is None


@pytest.mark.asyncio
async def test_create_task_minimal(db_session: AsyncSession) -> None:
    """Test creating a task with only required fields."""
    project = await create_project(db_session, name="Minimal Test", config={})

    task = await create_task(
        db_session,
        project_id=project.id,
        title="Simple task",
        agent_type="executor",
    )

    assert task.title == "Simple task"
    assert task.agent_type == "executor"
    assert task.model_tier == "auto"
    assert task.priority == 100
    assert task.max_retries == 3
    assert task.dependencies is None
    assert task.files_touched is None


@pytest.mark.asyncio
async def test_get_task_found(db_session: AsyncSession) -> None:
    """Test retrieving an existing task by ID."""
    project = await create_project(db_session, name="Test Project", config={})
    created = await create_task(
        db_session,
        project_id=project.id,
        title="Find Me",
        agent_type="architect",
    )

    retrieved = await get_task(db_session, created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.title == "Find Me"


@pytest.mark.asyncio
async def test_get_task_not_found(db_session: AsyncSession) -> None:
    """Test retrieving a non-existent task returns None."""
    non_existent_id = uuid4()
    result = await get_task(db_session, non_existent_id)

    assert result is None


@pytest.mark.asyncio
async def test_list_tasks_no_filters(db_session: AsyncSession) -> None:
    """Test listing all tasks without filters."""
    project = await create_project(db_session, name="Multi-Task Project", config={})

    await create_task(db_session, project_id=project.id, title="Task 1", agent_type="executor")
    await create_task(db_session, project_id=project.id, title="Task 2", agent_type="architect")

    tasks = await list_tasks(db_session)

    assert len(tasks) >= 2


@pytest.mark.asyncio
async def test_list_tasks_filter_by_project(db_session: AsyncSession) -> None:
    """Test listing tasks filtered by project ID."""
    project_a = await create_project(db_session, name="Project A", config={})
    project_b = await create_project(db_session, name="Project B", config={})

    await create_task(db_session, project_id=project_a.id, title="A1", agent_type="executor")
    await create_task(db_session, project_id=project_a.id, title="A2", agent_type="executor")
    await create_task(db_session, project_id=project_b.id, title="B1", agent_type="executor")

    tasks_a = await list_tasks(db_session, project_id=project_a.id)

    assert len(tasks_a) == 2
    assert all(t.project_id == project_a.id for t in tasks_a)
    assert {t.title for t in tasks_a} == {"A1", "A2"}


@pytest.mark.asyncio
async def test_list_tasks_filter_by_status(db_session: AsyncSession) -> None:
    """Test listing tasks filtered by status."""
    project = await create_project(db_session, name="Status Test", config={})

    pending = await create_task(
        db_session, project_id=project.id, title="Pending", agent_type="executor"
    )
    running = await create_task(
        db_session, project_id=project.id, title="Running", agent_type="executor"
    )

    await update_task_status(db_session, running.id, TaskStatus.running)

    running_tasks = await list_tasks(
        db_session, project_id=project.id, status_filter=TaskStatus.running
    )

    assert len(running_tasks) == 1
    assert running_tasks[0].title == "Running"
    assert running_tasks[0].status == TaskStatus.running


@pytest.mark.asyncio
async def test_list_tasks_priority_sorting(db_session: AsyncSession) -> None:
    """Test that tasks are sorted by priority (lowest number first)."""
    project = await create_project(db_session, name="Priority Test", config={})

    high = await create_task(
        db_session, project_id=project.id, title="High", agent_type="executor", priority=10
    )
    low = await create_task(
        db_session, project_id=project.id, title="Low", agent_type="executor", priority=100
    )
    medium = await create_task(
        db_session, project_id=project.id, title="Medium", agent_type="executor", priority=50
    )

    tasks = await list_tasks(db_session, project_id=project.id, priority_sort=True)

    assert len(tasks) == 3
    assert tasks[0].title == "High"
    assert tasks[1].title == "Medium"
    assert tasks[2].title == "Low"


@pytest.mark.asyncio
async def test_list_tasks_created_at_sorting(db_session: AsyncSession) -> None:
    """Test that tasks can be sorted by created_at descending."""
    project = await create_project(db_session, name="Time Test", config={})

    first = await create_task(
        db_session, project_id=project.id, title="First", agent_type="executor"
    )
    second = await create_task(
        db_session, project_id=project.id, title="Second", agent_type="executor"
    )

    tasks = await list_tasks(db_session, project_id=project.id, priority_sort=False)

    # Most recent first
    task_ids = [t.id for t in tasks]
    assert task_ids.index(second.id) < task_ids.index(first.id)


@pytest.mark.asyncio
async def test_update_task_status_to_running(db_session: AsyncSession) -> None:
    """Test that updating to running status sets started_at timestamp."""
    project = await create_project(db_session, name="Timestamp Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Start Me", agent_type="executor"
    )

    assert task.started_at is None

    updated = await update_task_status(db_session, task.id, TaskStatus.running)

    assert updated.status == TaskStatus.running
    assert updated.started_at is not None
    assert isinstance(updated.started_at, datetime)
    assert updated.completed_at is None


@pytest.mark.asyncio
async def test_update_task_status_to_done(db_session: AsyncSession) -> None:
    """Test that updating to done status sets completed_at timestamp."""
    project = await create_project(db_session, name="Completion Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Complete Me", agent_type="executor"
    )

    # First set to running to get started_at
    await update_task_status(db_session, task.id, TaskStatus.running)

    # Then complete
    completed = await update_task_status(db_session, task.id, TaskStatus.done)

    assert completed.status == TaskStatus.done
    assert completed.started_at is not None
    assert completed.completed_at is not None
    assert completed.completed_at >= completed.started_at


@pytest.mark.asyncio
async def test_update_task_status_to_failed(db_session: AsyncSession) -> None:
    """Test that updating to failed status sets completed_at timestamp."""
    project = await create_project(db_session, name="Failure Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Fail Me", agent_type="executor"
    )

    await update_task_status(db_session, task.id, TaskStatus.running)
    failed = await update_task_status(db_session, task.id, TaskStatus.failed)

    assert failed.status == TaskStatus.failed
    assert failed.completed_at is not None


@pytest.mark.asyncio
async def test_update_task_status_not_found(db_session: AsyncSession) -> None:
    """Test that updating a non-existent task raises ValueError."""
    non_existent_id = uuid4()

    with pytest.raises(ValueError, match=f"Task {non_existent_id} not found"):
        await update_task_status(db_session, non_existent_id, TaskStatus.running)


@pytest.mark.asyncio
async def test_get_ready_tasks_no_dependencies(db_session: AsyncSession) -> None:
    """Test that tasks with no dependencies are immediately ready."""
    project = await create_project(db_session, name="Ready Test", config={})

    task1 = await create_task(
        db_session, project_id=project.id, title="Independent 1", agent_type="executor"
    )
    task2 = await create_task(
        db_session, project_id=project.id, title="Independent 2", agent_type="executor"
    )

    ready = await get_ready_tasks(db_session, project.id)

    assert len(ready) == 2
    assert {t.id for t in ready} == {task1.id, task2.id}


@pytest.mark.asyncio
async def test_get_ready_tasks_with_dependencies_not_done(db_session: AsyncSession) -> None:
    """Test that tasks with incomplete dependencies are not ready."""
    project = await create_project(db_session, name="Dependency Test", config={})

    dep_task = await create_task(
        db_session, project_id=project.id, title="Dependency", agent_type="executor"
    )
    blocked_task = await create_task(
        db_session,
        project_id=project.id,
        title="Blocked",
        agent_type="executor",
        dependencies=[dep_task.id],
    )

    ready = await get_ready_tasks(db_session, project.id)

    # Only the dependency should be ready
    assert len(ready) == 1
    assert ready[0].id == dep_task.id


@pytest.mark.asyncio
async def test_get_ready_tasks_with_dependencies_done(db_session: AsyncSession) -> None:
    """Test that tasks become ready when all dependencies are done."""
    project = await create_project(db_session, name="Unblock Test", config={})

    dep_task = await create_task(
        db_session, project_id=project.id, title="Dependency", agent_type="executor"
    )
    dependent_task = await create_task(
        db_session,
        project_id=project.id,
        title="Dependent",
        agent_type="executor",
        dependencies=[dep_task.id],
    )

    # Complete the dependency
    await update_task_status(db_session, dep_task.id, TaskStatus.running)
    await update_task_status(db_session, dep_task.id, TaskStatus.done)

    ready = await get_ready_tasks(db_session, project.id)

    # Now the dependent task should be ready
    assert len(ready) == 1
    assert ready[0].id == dependent_task.id


@pytest.mark.asyncio
async def test_get_ready_tasks_priority_ordering(db_session: AsyncSession) -> None:
    """Test that ready tasks are ordered by priority."""
    project = await create_project(db_session, name="Priority Ready Test", config={})

    await create_task(
        db_session, project_id=project.id, title="Low", agent_type="executor", priority=100
    )
    await create_task(
        db_session, project_id=project.id, title="High", agent_type="executor", priority=10
    )
    await create_task(
        db_session, project_id=project.id, title="Medium", agent_type="executor", priority=50
    )

    ready = await get_ready_tasks(db_session, project.id)

    assert len(ready) == 3
    assert ready[0].title == "High"
    assert ready[1].title == "Medium"
    assert ready[2].title == "Low"


@pytest.mark.asyncio
async def test_get_ready_tasks_excludes_running_tasks(db_session: AsyncSession) -> None:
    """Test that running tasks are not included in ready tasks."""
    project = await create_project(db_session, name="Running Test", config={})

    pending = await create_task(
        db_session, project_id=project.id, title="Pending", agent_type="executor"
    )
    running = await create_task(
        db_session, project_id=project.id, title="Running", agent_type="executor"
    )

    await update_task_status(db_session, running.id, TaskStatus.running)

    ready = await get_ready_tasks(db_session, project.id)

    assert len(ready) == 1
    assert ready[0].id == pending.id


@pytest.mark.asyncio
async def test_get_next_task_returns_highest_priority(db_session: AsyncSession) -> None:
    """Test that get_next_task returns the highest priority ready task."""
    project = await create_project(db_session, name="Next Task Test", config={})

    await create_task(
        db_session, project_id=project.id, title="Low", agent_type="executor", priority=100
    )
    high = await create_task(
        db_session, project_id=project.id, title="High", agent_type="executor", priority=10
    )

    next_task = await get_next_task(db_session, project.id)

    assert next_task is not None
    assert next_task.id == high.id


@pytest.mark.asyncio
async def test_get_next_task_none_available(db_session: AsyncSession) -> None:
    """Test that get_next_task returns None when no tasks are ready."""
    project = await create_project(db_session, name="Empty Test", config={})

    next_task = await get_next_task(db_session, project.id)

    assert next_task is None


@pytest.mark.asyncio
async def test_increment_retry_count(db_session: AsyncSession) -> None:
    """Test incrementing a task's retry count."""
    project = await create_project(db_session, name="Retry Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Retry Me", agent_type="executor"
    )

    assert task.retry_count == 0

    updated = await increment_retry_count(db_session, task.id)
    assert updated.retry_count == 1

    updated = await increment_retry_count(db_session, task.id)
    assert updated.retry_count == 2


@pytest.mark.asyncio
async def test_increment_retry_count_not_found(db_session: AsyncSession) -> None:
    """Test that incrementing retry count on non-existent task raises ValueError."""
    non_existent_id = uuid4()

    with pytest.raises(ValueError, match=f"Task {non_existent_id} not found"):
        await increment_retry_count(db_session, non_existent_id)


@pytest.mark.asyncio
async def test_get_ready_tasks_with_multiple_dependencies(db_session: AsyncSession) -> None:
    """Test dependency resolution with multiple dependencies."""
    project = await create_project(db_session, name="Multi-Dep Test", config={})

    dep1 = await create_task(
        db_session, project_id=project.id, title="Dep 1", agent_type="executor"
    )
    dep2 = await create_task(
        db_session, project_id=project.id, title="Dep 2", agent_type="executor"
    )
    dependent = await create_task(
        db_session,
        project_id=project.id,
        title="Multi-Dependent",
        agent_type="executor",
        dependencies=[dep1.id, dep2.id],
    )

    # Complete only one dependency
    await update_task_status(db_session, dep1.id, TaskStatus.done)

    ready = await get_ready_tasks(db_session, project.id)
    ready_ids = {t.id for t in ready}

    # Dependent should NOT be ready yet
    assert dependent.id not in ready_ids
    assert dep2.id in ready_ids  # But dep2 should be ready

    # Complete second dependency
    await update_task_status(db_session, dep2.id, TaskStatus.done)

    ready = await get_ready_tasks(db_session, project.id)
    ready_ids = {t.id for t in ready}

    # Now dependent should be ready
    assert dependent.id in ready_ids
