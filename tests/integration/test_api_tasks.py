"""Integration tests for task API endpoints.

This module provides integration tests for the task CRUD endpoints
in the Forgemaster web API. Tests use mocked database sessions to verify
request handling, response formatting, and error conditions.

All tests use httpx.AsyncClient with ASGITransport for async API testing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from forgemaster.database.models.task import TaskStatus
from forgemaster.web.app import create_app


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock database session.

    Returns:
        AsyncMock configured to work as async context manager.
    """
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
def mock_session_factory(mock_session: AsyncMock) -> MagicMock:
    """Create a mock session factory.

    Args:
        mock_session: The mock session fixture.

    Returns:
        MagicMock that returns the mock session when called.
    """
    factory = MagicMock(return_value=mock_session)
    return factory


@pytest.fixture
def app(mock_session_factory: MagicMock) -> Any:
    """Create test app with mocked database.

    Args:
        mock_session_factory: The mock session factory fixture.

    Returns:
        FastAPI application configured for testing.
    """
    test_app = create_app()
    test_app.state.session_factory = mock_session_factory
    return test_app


@pytest.fixture
async def client(app: Any) -> AsyncClient:
    """Create test client.

    Args:
        app: The test app fixture.

    Yields:
        AsyncClient configured for the test app.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def make_mock_task(
    id: UUID | None = None,
    project_id: UUID | None = None,
    title: str = "Test Task",
    status: TaskStatus = TaskStatus.pending,
    agent_type: str = "executor",
) -> MagicMock:
    """Create a mock task instance.

    Args:
        id: UUID for the task (generated if None).
        project_id: UUID of the parent project (generated if None).
        title: Short task description.
        status: Task lifecycle status.
        agent_type: Type of agent required.

    Returns:
        MagicMock configured with task attributes.
    """
    task = MagicMock()
    task.id = id or uuid4()
    task.project_id = project_id or uuid4()
    task.title = title
    task.description = "Test description"
    task.status = status
    task.agent_type = agent_type
    task.model_tier = "auto"
    task.priority = 100
    task.estimated_minutes = None
    task.files_touched = None
    task.dependencies = None
    task.parallel_group = None
    task.retry_count = 0
    task.max_retries = 3
    task.started_at = None
    task.completed_at = None
    task.created_at = datetime.now(timezone.utc)
    task.updated_at = datetime.now(timezone.utc)
    return task


@pytest.mark.asyncio
async def test_list_tasks_empty(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test listing tasks when database is empty.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    with patch("forgemaster.web.routes.tasks.list_tasks") as mock_list:
        mock_list.return_value = []
        response = await client.get("/tasks/")
        assert response.status_code == 200
        assert response.json() == []
        mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_list_tasks_with_data(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test listing tasks when database contains data.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task1 = make_mock_task(title="Task 1")
    task2 = make_mock_task(title="Task 2", status=TaskStatus.running)

    with patch("forgemaster.web.routes.tasks.list_tasks") as mock_list:
        mock_list.return_value = [task1, task2]
        response = await client.get("/tasks/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["title"] == "Task 1"
        assert data[0]["status"] == "pending"
        assert data[1]["title"] == "Task 2"
        assert data[1]["status"] == "running"


@pytest.mark.asyncio
async def test_list_tasks_with_project_filter(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test listing tasks with project_id filter.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()
    task = make_mock_task(project_id=project_id)

    with patch("forgemaster.web.routes.tasks.list_tasks") as mock_list:
        mock_list.return_value = [task]
        response = await client.get(f"/tasks/?project_id={project_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["project_id"] == str(project_id)
        # Verify the query was called with the correct project_id
        mock_list.assert_called_once()
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["project_id"] == project_id


@pytest.mark.asyncio
async def test_list_tasks_with_status_filter(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test listing tasks with status filter.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task = make_mock_task(status=TaskStatus.done)

    with patch("forgemaster.web.routes.tasks.list_tasks") as mock_list:
        mock_list.return_value = [task]
        response = await client.get("/tasks/?status=done")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "done"
        # Verify the query was called with the correct status enum
        mock_list.assert_called_once()
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["status_filter"] == TaskStatus.done


@pytest.mark.asyncio
async def test_list_tasks_invalid_status(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test listing tasks with invalid status filter.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    with patch("forgemaster.web.routes.tasks.list_tasks") as mock_list:
        response = await client.get("/tasks/?status=invalid_status")
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]
        mock_list.assert_not_called()


@pytest.mark.asyncio
async def test_get_task_success(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting a task by ID when it exists.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task_id = uuid4()
    task = make_mock_task(id=task_id, title="Test Task")

    with patch("forgemaster.web.routes.tasks.get_task") as mock_get:
        mock_get.return_value = task
        response = await client.get(f"/tasks/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(task_id)
        assert data["title"] == "Test Task"
        assert data["status"] == "pending"
        mock_get.assert_called_once_with(mock_session, task_id)


@pytest.mark.asyncio
async def test_get_task_not_found(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting a task by ID when it does not exist.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task_id = uuid4()

    with patch("forgemaster.web.routes.tasks.get_task") as mock_get:
        mock_get.return_value = None
        response = await client.get(f"/tasks/{task_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        mock_get.assert_called_once_with(mock_session, task_id)


@pytest.mark.asyncio
async def test_create_task_success(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test creating a new task successfully.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task_id = uuid4()
    project_id = uuid4()
    task = make_mock_task(
        id=task_id,
        project_id=project_id,
        title="New Task",
        agent_type="executor",
    )

    with patch("forgemaster.web.routes.tasks.create_task") as mock_create:
        mock_create.return_value = task
        response = await client.post(
            "/tasks/",
            json={
                "project_id": str(project_id),
                "title": "New Task",
                "agent_type": "executor",
                "description": "Test description",
                "priority": 100,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == str(task_id)
        assert data["title"] == "New Task"
        assert data["agent_type"] == "executor"
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_create_task_validation_error(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test creating a task with invalid data.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()

    # Empty title should fail validation
    response = await client.post(
        "/tasks/",
        json={
            "project_id": str(project_id),
            "title": "",
            "agent_type": "executor",
        },
    )
    assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_update_task_status_success(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test updating a task's status successfully.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task_id = uuid4()
    updated_task = make_mock_task(id=task_id, status=TaskStatus.running)

    with patch("forgemaster.web.routes.tasks.update_task_status") as mock_update:
        mock_update.return_value = updated_task
        response = await client.put(
            f"/tasks/{task_id}/status",
            json={"status": "running"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(task_id)
        assert data["status"] == "running"
        mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_update_task_status_not_found(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test updating a task's status when task does not exist.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task_id = uuid4()

    with patch("forgemaster.web.routes.tasks.update_task_status") as mock_update:
        mock_update.side_effect = ValueError(f"Task {task_id} not found")
        response = await client.put(
            f"/tasks/{task_id}/status",
            json={"status": "running"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_update_task_status_invalid(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test updating a task's status with invalid status value.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task_id = uuid4()

    with patch("forgemaster.web.routes.tasks.update_task_status") as mock_update:
        response = await client.put(
            f"/tasks/{task_id}/status",
            json={"status": "invalid_status"},
        )
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]
        mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_get_ready_tasks(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting ready tasks for a project.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()
    task1 = make_mock_task(project_id=project_id, status=TaskStatus.ready, priority=50)
    task2 = make_mock_task(project_id=project_id, status=TaskStatus.ready, priority=100)

    with patch("forgemaster.web.routes.tasks.get_ready_tasks") as mock_get_ready:
        mock_get_ready.return_value = [task1, task2]
        response = await client.get(f"/tasks/ready/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["status"] == "ready"
        assert data[1]["status"] == "ready"
        mock_get_ready.assert_called_once_with(mock_session, project_id)
