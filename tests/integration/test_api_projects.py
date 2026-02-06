"""Integration tests for project API endpoints.

This module provides integration tests for the project CRUD endpoints
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

from forgemaster.database.models.project import ProjectStatus
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


def make_mock_project(
    id: UUID | None = None,
    name: str = "Test Project",
    status: ProjectStatus = ProjectStatus.draft,
    config: dict[str, Any] | None = None,
    spec_document: dict[str, Any] | None = None,
    architecture_document: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock project instance.

    Args:
        id: UUID for the project (generated if None).
        name: Human-readable project name.
        status: Project lifecycle status.
        config: Project-specific configuration.
        spec_document: Optional project specification.
        architecture_document: Optional architecture description.

    Returns:
        MagicMock configured with project attributes.
    """
    project = MagicMock()
    project.id = id or uuid4()
    project.name = name
    project.status = status
    project.config = config or {}
    project.spec_document = spec_document
    project.architecture_document = architecture_document
    project.created_at = datetime.now(timezone.utc)
    project.updated_at = datetime.now(timezone.utc)
    return project


@pytest.mark.asyncio
async def test_list_projects_empty(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test listing projects when database is empty.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    with patch("forgemaster.web.routes.projects.project_queries.list_projects") as mock_list:
        mock_list.return_value = []
        response = await client.get("/projects/")
        assert response.status_code == 200
        assert response.json() == []
        mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_list_projects_with_data(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test listing projects when database contains data.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project1 = make_mock_project(name="Project 1")
    project2 = make_mock_project(name="Project 2", status=ProjectStatus.active)

    with patch("forgemaster.web.routes.projects.project_queries.list_projects") as mock_list:
        mock_list.return_value = [project1, project2]
        response = await client.get("/projects/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Project 1"
        assert data[0]["status"] == "draft"
        assert data[1]["name"] == "Project 2"
        assert data[1]["status"] == "active"


@pytest.mark.asyncio
async def test_list_projects_with_status_filter(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test listing projects with status filter.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project = make_mock_project(status=ProjectStatus.active)

    with patch("forgemaster.web.routes.projects.project_queries.list_projects") as mock_list:
        mock_list.return_value = [project]
        response = await client.get("/projects/?status=active")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "active"
        # Verify the query was called with the correct status enum
        mock_list.assert_called_once()
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["status_filter"] == ProjectStatus.active


@pytest.mark.asyncio
async def test_list_projects_invalid_status(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test listing projects with invalid status filter.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    with patch("forgemaster.web.routes.projects.project_queries.list_projects") as mock_list:
        response = await client.get("/projects/?status=invalid_status")
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]
        mock_list.assert_not_called()


@pytest.mark.asyncio
async def test_get_project_success(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting a project by ID when it exists.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()
    project = make_mock_project(id=project_id, name="Test Project")

    with patch("forgemaster.web.routes.projects.project_queries.get_project") as mock_get:
        mock_get.return_value = project
        response = await client.get(f"/projects/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(project_id)
        assert data["name"] == "Test Project"
        assert data["status"] == "draft"
        mock_get.assert_called_once_with(session=mock_session, project_id=project_id)


@pytest.mark.asyncio
async def test_get_project_not_found(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting a project by ID when it does not exist.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()

    with patch("forgemaster.web.routes.projects.project_queries.get_project") as mock_get:
        mock_get.return_value = None
        response = await client.get(f"/projects/{project_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        mock_get.assert_called_once_with(session=mock_session, project_id=project_id)


@pytest.mark.asyncio
async def test_create_project_success(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test creating a new project successfully.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()
    project = make_mock_project(
        id=project_id,
        name="New Project",
        config={"key": "value"},
    )

    with patch("forgemaster.web.routes.projects.project_queries.create_project") as mock_create:
        mock_create.return_value = project
        response = await client.post(
            "/projects/",
            json={
                "name": "New Project",
                "config": {"key": "value"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == str(project_id)
        assert data["name"] == "New Project"
        assert data["config"] == {"key": "value"}
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_create_project_validation_error(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test creating a project with invalid data.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    # Empty name should fail validation
    response = await client.post(
        "/projects/",
        json={
            "name": "",
            "config": {},
        },
    )
    assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_update_project_success(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test updating an existing project successfully.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()
    updated_project = make_mock_project(
        id=project_id,
        name="Updated Name",
        status=ProjectStatus.active,
    )

    with patch("forgemaster.web.routes.projects.project_queries.update_project") as mock_update:
        mock_update.return_value = updated_project
        response = await client.put(
            f"/projects/{project_id}",
            json={
                "name": "Updated Name",
                "status": "active",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(project_id)
        assert data["name"] == "Updated Name"
        assert data["status"] == "active"
        mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_update_project_not_found(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test updating a project that does not exist.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()

    with patch("forgemaster.web.routes.projects.project_queries.update_project") as mock_update:
        mock_update.side_effect = ValueError(f"Project {project_id} not found")
        response = await client.put(
            f"/projects/{project_id}",
            json={"name": "Updated Name"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_delete_project_success(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test deleting a project successfully.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()

    with patch("forgemaster.web.routes.projects.project_queries.delete_project") as mock_delete:
        mock_delete.return_value = True
        response = await client.delete(f"/projects/{project_id}")
        assert response.status_code == 204
        assert response.content == b""
        mock_delete.assert_called_once_with(session=mock_session, project_id=project_id)


@pytest.mark.asyncio
async def test_delete_project_not_found(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test deleting a project that does not exist.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    project_id = uuid4()

    with patch("forgemaster.web.routes.projects.project_queries.delete_project") as mock_delete:
        mock_delete.return_value = False
        response = await client.delete(f"/projects/{project_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        mock_delete.assert_called_once_with(session=mock_session, project_id=project_id)
