"""Integration tests for session API endpoints.

This module provides integration tests for the agent session query endpoints
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

from forgemaster.database.models.session import SessionStatus
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


def make_mock_agent_session(
    id: UUID | None = None,
    task_id: UUID | None = None,
    model: str = "claude-3-sonnet",
    status: SessionStatus = SessionStatus.active,
    worktree_path: str | None = None,
    token_count: int = 0,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
    started_at: datetime | None = None,
    last_activity_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> MagicMock:
    """Create a mock agent session instance.

    Args:
        id: Session UUID (generated if None).
        task_id: Associated task UUID (generated if None).
        model: Claude model identifier.
        status: Current session status.
        worktree_path: Git worktree path.
        token_count: Total tokens consumed.
        result: Session output/result data.
        error_message: Error description if failed.
        started_at: Session start timestamp.
        last_activity_at: Last activity timestamp.
        ended_at: Session termination timestamp.

    Returns:
        MagicMock configured with session attributes.
    """
    agent_session = MagicMock()
    agent_session.id = id or uuid4()
    agent_session.task_id = task_id or uuid4()
    agent_session.model = model
    agent_session.status = status
    agent_session.worktree_path = worktree_path
    agent_session.token_count = token_count
    agent_session.result = result
    agent_session.error_message = error_message
    agent_session.started_at = started_at or datetime.now(timezone.utc)
    agent_session.last_activity_at = last_activity_at or datetime.now(timezone.utc)
    agent_session.ended_at = ended_at
    return agent_session


@pytest.mark.asyncio
async def test_list_sessions_empty(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test listing sessions when database is empty.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    with patch("forgemaster.web.routes.sessions.list_sessions") as mock_list:
        mock_list.return_value = []
        response = await client.get("/sessions/")
        assert response.status_code == 200
        assert response.json() == []
        mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_list_sessions_with_data(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test listing sessions when database contains data.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    session1 = make_mock_agent_session(model="claude-3-sonnet")
    session2 = make_mock_agent_session(model="claude-3-opus", status=SessionStatus.completed)

    with patch("forgemaster.web.routes.sessions.list_sessions") as mock_list:
        mock_list.return_value = [session1, session2]
        response = await client.get("/sessions/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["model"] == "claude-3-sonnet"
        assert data[0]["status"] == "active"
        assert data[1]["model"] == "claude-3-opus"
        assert data[1]["status"] == "completed"


@pytest.mark.asyncio
async def test_list_sessions_with_task_filter(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test listing sessions with task_id filter.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    task_id = uuid4()
    session = make_mock_agent_session(task_id=task_id)

    with patch("forgemaster.web.routes.sessions.list_sessions") as mock_list:
        mock_list.return_value = [session]
        response = await client.get(f"/sessions/?task_id={task_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["task_id"] == str(task_id)
        # Verify the query was called with the correct task_id
        mock_list.assert_called_once()
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["task_id"] == task_id


@pytest.mark.asyncio
async def test_list_sessions_with_status_filter(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test listing sessions with status filter.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    session = make_mock_agent_session(status=SessionStatus.completed)

    with patch("forgemaster.web.routes.sessions.list_sessions") as mock_list:
        mock_list.return_value = [session]
        response = await client.get("/sessions/?status=completed")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "completed"
        # Verify the query was called with the correct status enum
        mock_list.assert_called_once()
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["status_filter"] == SessionStatus.completed


@pytest.mark.asyncio
async def test_get_session_success(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting a session by ID when it exists.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    session_id = uuid4()
    agent_session = make_mock_agent_session(
        id=session_id,
        model="claude-3-sonnet",
        status=SessionStatus.active,
    )

    with patch("forgemaster.web.routes.sessions.get_session") as mock_get:
        mock_get.return_value = agent_session
        response = await client.get(f"/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(session_id)
        assert data["model"] == "claude-3-sonnet"
        assert data["status"] == "active"
        mock_get.assert_called_once_with(session=mock_session, session_id=session_id)


@pytest.mark.asyncio
async def test_get_session_not_found(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting a session by ID when it does not exist.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    session_id = uuid4()

    with patch("forgemaster.web.routes.sessions.get_session") as mock_get:
        mock_get.return_value = None
        response = await client.get(f"/sessions/{session_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        mock_get.assert_called_once_with(session=mock_session, session_id=session_id)


@pytest.mark.asyncio
async def test_get_active_sessions_empty(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting active sessions when there are none.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    with patch("forgemaster.web.routes.sessions.get_active_sessions") as mock_get_active:
        mock_get_active.return_value = []
        response = await client.get("/sessions/active")
        assert response.status_code == 200
        assert response.json() == []
        mock_get_active.assert_called_once()


@pytest.mark.asyncio
async def test_get_active_sessions_with_data(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test getting active sessions when they exist.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    session1 = make_mock_agent_session(status=SessionStatus.active)
    session2 = make_mock_agent_session(status=SessionStatus.idle)

    with patch("forgemaster.web.routes.sessions.get_active_sessions") as mock_get_active:
        mock_get_active.return_value = [session1, session2]
        response = await client.get("/sessions/active")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["status"] == "active"
        assert data[1]["status"] == "idle"


@pytest.mark.asyncio
async def test_get_idle_sessions_empty(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test getting idle sessions when there are none.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    with patch("forgemaster.web.routes.sessions.get_idle_sessions") as mock_get_idle:
        mock_get_idle.return_value = []
        response = await client.get("/sessions/idle")
        assert response.status_code == 200
        assert response.json() == []
        mock_get_idle.assert_called_once()


@pytest.mark.asyncio
async def test_get_idle_sessions_with_threshold(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test getting idle sessions with custom threshold parameter.

    Args:
        client: The test client fixture.
        mock_session: The mock session fixture.
    """
    session = make_mock_agent_session(status=SessionStatus.active)

    with patch("forgemaster.web.routes.sessions.get_idle_sessions") as mock_get_idle:
        mock_get_idle.return_value = [session]
        response = await client.get("/sessions/idle?threshold_seconds=600")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        # Verify the threshold parameter was passed
        mock_get_idle.assert_called_once()
        call_args = mock_get_idle.call_args
        # Second argument should be the threshold
        assert call_args.args[1] == 600
