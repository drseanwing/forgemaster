"""Integration tests for lesson learned API endpoints.

Tests all lesson REST API endpoints including list with filters, get by ID,
full-text search, and file-based search operations.

Uses httpx.AsyncClient with ASGITransport for testing FastAPI app without
actual HTTP server. Mocks database layer via session_factory injection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from forgemaster.web.app import create_app


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session with proper context manager support.

    Returns:
        AsyncMock configured for async context manager usage
    """
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
def mock_session_factory(mock_session: AsyncMock) -> MagicMock:
    """Create a mock session factory that returns the mock session.

    Args:
        mock_session: The mock session to be returned by the factory

    Returns:
        MagicMock callable that returns mock_session
    """
    factory = MagicMock(return_value=mock_session)
    return factory


@pytest.fixture
def app(mock_session_factory: MagicMock) -> Any:
    """Create FastAPI app with mocked session factory.

    Args:
        mock_session_factory: Mock session factory to inject into app state

    Returns:
        FastAPI application instance with mocked database
    """
    test_app = create_app()
    test_app.state.session_factory = mock_session_factory
    return test_app


@pytest.fixture
async def client(app: Any) -> Any:
    """Create async HTTP client for testing API endpoints.

    Args:
        app: FastAPI application instance

    Yields:
        AsyncClient configured with ASGITransport
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def make_mock_lesson(
    id: Any = None,
    project_id: Any = None,
    task_id: Any = None,
    symptom: str = "Test symptom",
    root_cause: str = "Test root cause",
    fix_applied: str = "Test fix",
    files_affected: list[str] | None = None,
    pattern_tags: list[str] | None = None,
    verification_status: str = "unverified",
    confidence_score: float = 0.5,
) -> MagicMock:
    """Create a mock LessonLearned object with specified attributes.

    Args:
        id: Lesson UUID (generates new if None)
        project_id: Project UUID (generates new if None)
        task_id: Task UUID (generates new if None)
        symptom: Problem description
        root_cause: Identified root cause
        fix_applied: Fix description
        files_affected: List of affected files
        pattern_tags: Classification tags
        verification_status: Verification status
        confidence_score: Confidence score (0.0 to 1.0)

    Returns:
        MagicMock configured with all LessonLearned attributes
    """
    lesson = MagicMock()
    lesson.id = id or uuid4()
    lesson.project_id = project_id or uuid4()
    lesson.task_id = task_id or uuid4()
    lesson.symptom = symptom
    lesson.root_cause = root_cause
    lesson.fix_applied = fix_applied
    lesson.files_affected = files_affected or ["src/test.py"]
    lesson.pattern_tags = pattern_tags or ["error-handling"]
    lesson.verification_status = verification_status
    lesson.confidence_score = confidence_score
    lesson.created_at = datetime.now(timezone.utc)
    lesson.updated_at = datetime.now(timezone.utc)
    return lesson


@pytest.mark.asyncio
async def test_list_lessons_empty(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test list lessons endpoint returns empty list when no lessons exist."""
    with patch("forgemaster.web.routes.lessons.list_lessons") as mock_list:
        mock_list.return_value = []

        response = await client.get("/lessons/")

        assert response.status_code == 200
        assert response.json() == []
        mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_list_lessons_with_data(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test list lessons endpoint returns list of lessons."""
    lesson1 = make_mock_lesson(symptom="Error 1", root_cause="Cause 1")
    lesson2 = make_mock_lesson(symptom="Error 2", root_cause="Cause 2")

    with patch("forgemaster.web.routes.lessons.list_lessons") as mock_list:
        mock_list.return_value = [lesson1, lesson2]

        response = await client.get("/lessons/")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["symptom"] == "Error 1"
        assert data[1]["symptom"] == "Error 2"


@pytest.mark.asyncio
async def test_list_lessons_with_project_filter(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test list lessons endpoint filters by project_id."""
    project_id = uuid4()
    lesson = make_mock_lesson(project_id=project_id)

    with patch("forgemaster.web.routes.lessons.list_lessons") as mock_list:
        mock_list.return_value = [lesson]

        response = await client.get(f"/lessons/?project_id={project_id}")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["project_id"] == str(project_id)
        mock_list.assert_called_once()
        # Verify project_id was passed to query function
        call_kwargs = mock_list.call_args[1]
        assert call_kwargs["project_id"] == project_id


@pytest.mark.asyncio
async def test_list_lessons_with_status_filter(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test list lessons endpoint filters by verification_status."""
    lesson = make_mock_lesson(verification_status="verified")

    with patch("forgemaster.web.routes.lessons.list_lessons") as mock_list:
        mock_list.return_value = [lesson]

        response = await client.get("/lessons/?verification_status=verified")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["verification_status"] == "verified"
        mock_list.assert_called_once()
        # Verify status filter was passed
        call_kwargs = mock_list.call_args[1]
        assert call_kwargs["verification_status"] == "verified"


@pytest.mark.asyncio
async def test_list_lessons_with_tags_filter(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test list lessons endpoint filters by pattern_tags."""
    lesson = make_mock_lesson(pattern_tags=["python", "type-error"])

    with patch("forgemaster.web.routes.lessons.list_lessons") as mock_list:
        mock_list.return_value = [lesson]

        response = await client.get("/lessons/?pattern_tags=python&pattern_tags=type-error")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "python" in data[0]["pattern_tags"]
        assert "type-error" in data[0]["pattern_tags"]
        mock_list.assert_called_once()
        # Verify tags filter was passed
        call_kwargs = mock_list.call_args[1]
        assert call_kwargs["pattern_tags"] == ["python", "type-error"]


@pytest.mark.asyncio
async def test_get_lesson_success(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test get lesson endpoint returns lesson by ID."""
    lesson_id = uuid4()
    lesson = make_mock_lesson(
        id=lesson_id, symptom="Test symptom", root_cause="Test root cause"
    )

    with patch("forgemaster.web.routes.lessons.get_lesson") as mock_get:
        mock_get.return_value = lesson

        response = await client.get(f"/lessons/{lesson_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(lesson_id)
        assert data["symptom"] == "Test symptom"
        assert data["root_cause"] == "Test root cause"
        mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_get_lesson_not_found(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test get lesson endpoint returns 404 when lesson not found."""
    lesson_id = uuid4()

    with patch("forgemaster.web.routes.lessons.get_lesson") as mock_get:
        mock_get.return_value = None

        response = await client.get(f"/lessons/{lesson_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_search_lessons_by_text(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test full-text search endpoint returns matching lessons."""
    project_id = uuid4()
    lesson = make_mock_lesson(
        project_id=project_id,
        symptom="Authentication token expired",
        root_cause="JWT token TTL too short",
    )

    with patch("forgemaster.web.routes.lessons.search_lessons_by_text") as mock_search:
        mock_search.return_value = [lesson]

        response = await client.post(
            "/lessons/search/text",
            json={"project_id": str(project_id), "query": "authentication token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "authentication" in data[0]["symptom"].lower()
        mock_search.assert_called_once()
        # Verify search parameters
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["project_id"] == project_id
        assert call_kwargs["query"] == "authentication token"


@pytest.mark.asyncio
async def test_search_lessons_by_text_empty(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test full-text search returns empty list for no matches."""
    project_id = uuid4()

    with patch("forgemaster.web.routes.lessons.search_lessons_by_text") as mock_search:
        mock_search.return_value = []

        response = await client.post(
            "/lessons/search/text",
            json={"project_id": str(project_id), "query": "nonexistent"},
        )

        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.asyncio
async def test_search_lessons_by_files(client: AsyncClient, mock_session: AsyncMock) -> None:
    """Test file overlap search endpoint returns matching lessons."""
    project_id = uuid4()
    lesson = make_mock_lesson(
        project_id=project_id, files_affected=["src/auth/jwt.py", "src/auth/middleware.py"]
    )

    with patch("forgemaster.web.routes.lessons.search_lessons_by_files") as mock_search:
        mock_search.return_value = [lesson]

        response = await client.post(
            "/lessons/search/files",
            json={"project_id": str(project_id), "files": ["src/auth/jwt.py"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "src/auth/jwt.py" in data[0]["files_affected"]
        mock_search.assert_called_once()
        # Verify search parameters
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["project_id"] == project_id
        assert call_kwargs["files"] == ["src/auth/jwt.py"]


@pytest.mark.asyncio
async def test_search_lessons_by_files_empty(
    client: AsyncClient, mock_session: AsyncMock
) -> None:
    """Test file overlap search returns empty list for no matches."""
    project_id = uuid4()

    with patch("forgemaster.web.routes.lessons.search_lessons_by_files") as mock_search:
        mock_search.return_value = []

        response = await client.post(
            "/lessons/search/files",
            json={"project_id": str(project_id), "files": ["src/unrelated/file.py"]},
        )

        assert response.status_code == 200
        assert response.json() == []
