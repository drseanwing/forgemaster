"""Task CRUD REST API endpoints for Forgemaster.

Provides FastAPI routes for creating, reading, updating, and listing
Task records, with filtering by project and status, and dependency-aware
ready task retrieval.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.task import TaskStatus
from forgemaster.database.queries.task import (
    create_task,
    get_ready_tasks,
    get_task,
    list_tasks,
    update_task_status,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


# --- Pydantic Schemas ---


class TaskCreate(BaseModel):
    """Request schema for creating a new task."""

    project_id: UUID
    title: str = Field(..., min_length=1, max_length=255)
    agent_type: str
    description: str | None = None
    model_tier: str | None = "auto"
    priority: int = Field(default=100, ge=1, le=1000)
    estimated_minutes: int | None = None
    files_touched: list[str] | None = None
    dependencies: list[UUID] | None = None
    parallel_group: str | None = None
    max_retries: int = Field(default=3, ge=0, le=10)


class TaskStatusUpdate(BaseModel):
    """Request schema for updating a task's status."""

    status: str  # "pending", "ready", "assigned", "running", "review", "done", "failed", "blocked"


class TaskResponse(BaseModel):
    """Response schema for task data."""

    id: UUID
    project_id: UUID | None
    title: str
    description: str | None
    status: str
    agent_type: str
    model_tier: str | None
    priority: int
    estimated_minutes: int | None
    files_touched: list[str] | None
    dependencies: list[UUID] | None
    parallel_group: str | None
    retry_count: int
    max_retries: int
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Dependency Injection ---


def get_session_factory(request: Request) -> Callable[[], AsyncSession]:
    """Extract session factory from FastAPI app state.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Callable that produces AsyncSession instances.
    """
    return request.app.state.session_factory


# --- Route Handlers ---


@router.get("/", response_model=list[TaskResponse])
async def list_tasks_endpoint(
    project_id: UUID | None = None,
    status: str | None = None,
    session_factory: Callable[[], AsyncSession] = Depends(get_session_factory),
) -> list[TaskResponse]:
    """List tasks with optional project_id and status filters.

    Args:
        project_id: Optional project UUID to filter by.
        status: Optional status string to filter by.
        session_factory: Injected session factory.

    Returns:
        List of TaskResponse objects.

    Raises:
        HTTPException: 400 if status is invalid.
    """
    # Validate status if provided
    status_filter = None
    if status is not None:
        try:
            status_filter = TaskStatus[status]
        except KeyError:
            logger.warning("invalid_status_filter", status=status)
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    async with session_factory() as session:
        tasks = await list_tasks(
            session=session,
            project_id=project_id,
            status_filter=status_filter,
        )

    logger.info(
        "tasks_listed",
        count=len(tasks),
        project_id=str(project_id) if project_id else None,
        status=status,
    )

    return [TaskResponse.model_validate(task) for task in tasks]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_endpoint(
    task_id: UUID,
    session_factory: Callable[[], AsyncSession] = Depends(get_session_factory),
) -> TaskResponse:
    """Get a single task by ID.

    Args:
        task_id: UUID of the task to retrieve.
        session_factory: Injected session factory.

    Returns:
        TaskResponse object.

    Raises:
        HTTPException: 404 if task not found.
    """
    async with session_factory() as session:
        task = await get_task(session, task_id)

    if task is None:
        logger.warning("task_not_found", task_id=str(task_id))
        raise HTTPException(status_code=404, detail="Task not found")

    logger.info("task_retrieved", task_id=str(task_id))
    return TaskResponse.model_validate(task)


@router.post("/", response_model=TaskResponse, status_code=201)
async def create_task_endpoint(
    task_data: TaskCreate,
    session_factory: Callable[[], AsyncSession] = Depends(get_session_factory),
) -> TaskResponse:
    """Create a new task.

    Args:
        task_data: Task creation data.
        session_factory: Injected session factory.

    Returns:
        Newly created TaskResponse object.
    """
    async with session_factory() as session:
        task = await create_task(
            session=session,
            project_id=task_data.project_id,
            title=task_data.title,
            agent_type=task_data.agent_type,
            description=task_data.description,
            model_tier=task_data.model_tier,
            priority=task_data.priority,
            estimated_minutes=task_data.estimated_minutes,
            files_touched=task_data.files_touched,
            dependencies=task_data.dependencies,
            parallel_group=task_data.parallel_group,
            max_retries=task_data.max_retries,
        )

    logger.info("task_created_via_api", task_id=str(task.id), title=task.title)
    return TaskResponse.model_validate(task)


@router.put("/{task_id}/status", response_model=TaskResponse)
async def update_task_status_endpoint(
    task_id: UUID,
    status_update: TaskStatusUpdate,
    session_factory: Callable[[], AsyncSession] = Depends(get_session_factory),
) -> TaskResponse:
    """Update a task's status.

    Args:
        task_id: UUID of the task to update.
        status_update: New status data.
        session_factory: Injected session factory.

    Returns:
        Updated TaskResponse object.

    Raises:
        HTTPException: 400 if status is invalid, 404 if task not found.
    """
    # Validate status
    try:
        new_status = TaskStatus[status_update.status]
    except KeyError:
        logger.warning("invalid_status_update", status=status_update.status)
        raise HTTPException(status_code=400, detail=f"Invalid status: {status_update.status}")

    async with session_factory() as session:
        try:
            task = await update_task_status(session, task_id, new_status)
        except ValueError as e:
            logger.warning("task_status_update_failed", task_id=str(task_id), error=str(e))
            raise HTTPException(status_code=404, detail=str(e))

    logger.info("task_status_updated_via_api", task_id=str(task_id), status=new_status.value)
    return TaskResponse.model_validate(task)


@router.get("/ready/{project_id}", response_model=list[TaskResponse])
async def get_ready_tasks_endpoint(
    project_id: UUID,
    session_factory: Callable[[], AsyncSession] = Depends(get_session_factory),
) -> list[TaskResponse]:
    """Get tasks ready for execution (all dependencies satisfied).

    Args:
        project_id: UUID of the project to search within.
        session_factory: Injected session factory.

    Returns:
        List of ready TaskResponse objects, sorted by priority.
    """
    async with session_factory() as session:
        tasks = await get_ready_tasks(session, project_id)

    logger.info(
        "ready_tasks_retrieved",
        project_id=str(project_id),
        count=len(tasks),
    )

    return [TaskResponse.model_validate(task) for task in tasks]
