"""Pytest fixtures for E2E tests.

Provides fixtures for end-to-end testing of FORGEMASTER orchestration workflows,
including mock database sessions, agent session managers, git repositories,
and sample task objects.

These fixtures enable testing of full orchestration flows without requiring
external dependencies like actual agent SDK connections or live git repositories.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from forgemaster.agents.result_schema import AgentResult
from forgemaster.agents.session import (
    AgentSessionManager,
    HealthStatus,
    SessionInfo,
    SessionMetrics,
    SessionState,
)
from forgemaster.config import AgentConfig
from forgemaster.database.models.base import Base
from forgemaster.database.models.project import Project
from forgemaster.database.models.task import Task, TaskStatus


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Use asyncio as the async backend for pytest-asyncio.

    Returns:
        The name of the async backend to use.
    """
    return "asyncio"


@pytest_asyncio.fixture
async def e2e_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create an in-memory SQLite async engine for E2E testing.

    Yields:
        Configured AsyncEngine instance using in-memory SQLite.
    """
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield test_engine

    await test_engine.dispose()


@pytest_asyncio.fixture
async def e2e_session_factory(
    e2e_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the test engine.

    Args:
        e2e_engine: The test database engine.

    Returns:
        Configured async_sessionmaker for creating test sessions.
    """
    return async_sessionmaker(
        bind=e2e_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture
async def e2e_session(
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Create a new async database session for each E2E test.

    Args:
        e2e_session_factory: The session factory fixture.

    Yields:
        AsyncSession instance for the test.
    """
    async with e2e_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    """Create a mock AgentConfig for testing.

    Returns:
        AgentConfig with test-appropriate defaults.
    """
    return AgentConfig(
        model_tier="sonnet",
        timeout_seconds=300,
        retry_limit=3,
        retry_delay_seconds=1,
        max_context_tokens=200000,
        context_warning_threshold=0.8,
    )


@pytest.fixture
def mock_agent_session_manager() -> MagicMock:
    """Create a mock AgentSessionManager with configurable responses.

    Returns:
        Mock AgentSessionManager that can be configured per test.
    """
    manager = MagicMock(spec=AgentSessionManager)

    # Default success behavior
    async def default_start_session(task_id: str, agent_type: str, prompt: str) -> SessionInfo:
        return SessionInfo(
            session_id=str(uuid.uuid4()),
            task_id=task_id,
            agent_type=agent_type,
            state=SessionState.ACTIVE,
            health_status=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        )

    async def default_wait_for_completion(session_id: str) -> AgentResult:
        return AgentResult(
            status="success",
            summary="Task completed successfully",
            details="Task completed successfully",
            files_modified=["test.py"],
        )

    async def default_terminate_session(session_id: str) -> None:
        pass

    manager.start_session = AsyncMock(side_effect=default_start_session)
    manager.wait_for_completion = AsyncMock(side_effect=default_wait_for_completion)
    manager.terminate_session = AsyncMock(side_effect=default_terminate_session)
    manager.get_session_info = AsyncMock(
        return_value=SessionInfo(
            session_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            agent_type="executor",
            state=SessionState.ACTIVE,
            health_status=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        )
    )

    return manager


@pytest.fixture
def mock_git_manager() -> MagicMock:
    """Create a mock GitManager for testing.

    Returns:
        Mock GitManager with temporary directory setup.
    """
    manager = MagicMock()
    temp_dir = tempfile.mkdtemp()

    manager.repo_path = Path(temp_dir)
    manager.create_branch = AsyncMock(return_value=None)
    manager.commit_changes = AsyncMock(return_value="abc123")
    manager.merge_branch = AsyncMock(return_value=None)
    manager.get_current_branch = AsyncMock(return_value="main")
    manager.has_uncommitted_changes = AsyncMock(return_value=False)

    return manager


@pytest_asyncio.fixture
async def sample_project(e2e_session: AsyncSession) -> Project:
    """Create a sample Project for testing.

    Args:
        e2e_session: Database session for the test.

    Returns:
        Persisted Project instance.
    """
    project = Project(
        id=uuid.uuid4(),
        name="test-project",
        repository_url="https://github.com/test/test-project",
        branch="main",
        project_type="python",
        created_at=datetime.now(timezone.utc),
    )
    e2e_session.add(project)
    await e2e_session.commit()
    await e2e_session.refresh(project)
    return project


@pytest_asyncio.fixture
async def sample_task_pending(e2e_session: AsyncSession, sample_project: Project) -> Task:
    """Create a sample Task in PENDING state.

    Args:
        e2e_session: Database session for the test.
        sample_project: Parent project for the task.

    Returns:
        Persisted Task instance in PENDING state.
    """
    task = Task(
        id=uuid.uuid4(),
        project_id=sample_project.id,
        title="Test task in pending state",
        description="This is a test task",
        status=TaskStatus.pending,
        agent_type="executor",
        priority=100,
        max_retries=3,
        retry_count=0,
        created_at=datetime.now(timezone.utc),
    )
    e2e_session.add(task)
    await e2e_session.commit()
    await e2e_session.refresh(task)
    return task


@pytest_asyncio.fixture
async def sample_task_ready(e2e_session: AsyncSession, sample_project: Project) -> Task:
    """Create a sample Task in READY state.

    Args:
        e2e_session: Database session for the test.
        sample_project: Parent project for the task.

    Returns:
        Persisted Task instance in READY state.
    """
    task = Task(
        id=uuid.uuid4(),
        project_id=sample_project.id,
        title="Test task in ready state",
        description="This task is ready for assignment",
        status=TaskStatus.ready,
        agent_type="executor",
        priority=100,
        max_retries=3,
        retry_count=0,
        created_at=datetime.now(timezone.utc),
    )
    e2e_session.add(task)
    await e2e_session.commit()
    await e2e_session.refresh(task)
    return task


@pytest_asyncio.fixture
async def sample_task_running(e2e_session: AsyncSession, sample_project: Project) -> Task:
    """Create a sample Task in RUNNING state.

    Args:
        e2e_session: Database session for the test.
        sample_project: Parent project for the task.

    Returns:
        Persisted Task instance in RUNNING state.
    """
    task = Task(
        id=uuid.uuid4(),
        project_id=sample_project.id,
        title="Test task in running state",
        description="This task is currently running",
        status=TaskStatus.running,
        agent_type="executor",
        priority=100,
        max_retries=3,
        retry_count=0,
        started_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )
    e2e_session.add(task)
    await e2e_session.commit()
    await e2e_session.refresh(task)
    return task


@pytest.fixture
def create_task_factory(e2e_session_factory: async_sessionmaker[AsyncSession]) -> Callable:
    """Factory fixture for creating tasks with custom attributes.

    Args:
        e2e_session_factory: Session factory for database access.

    Returns:
        Async function that creates and persists tasks.
    """

    async def _create_task(
        project_id: uuid.UUID,
        title: str = "Test task",
        description: str = "Test description",
        status: TaskStatus = TaskStatus.pending,
        agent_type: str = "executor",
        priority: int = 100,
        dependencies: list[uuid.UUID] | None = None,
        parallel_group: str | None = None,
        files_touched: list[str] | None = None,
    ) -> Task:
        async with e2e_session_factory() as session:
            task = Task(
                id=uuid.uuid4(),
                project_id=project_id,
                title=title,
                description=description,
                status=status,
                agent_type=agent_type,
                priority=priority,
                dependencies=dependencies,
                parallel_group=parallel_group,
                files_touched=files_touched,
                max_retries=3,
                retry_count=0,
                created_at=datetime.now(timezone.utc),
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return task

    return _create_task


@pytest.fixture
def create_session_info_factory() -> Callable:
    """Factory fixture for creating SessionInfo objects.

    Returns:
        Function that creates SessionInfo instances with custom attributes.
    """

    def _create_session_info(
        session_id: str | None = None,
        task_id: str | None = None,
        agent_type: str = "executor",
        state: SessionState = SessionState.ACTIVE,
        health_status: HealthStatus = HealthStatus.HEALTHY,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> SessionInfo:
        metrics = SessionMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return SessionInfo(
            session_id=session_id or str(uuid.uuid4()),
            task_id=task_id or str(uuid.uuid4()),
            agent_type=agent_type,
            state=state,
            health_status=health_status,
            metrics=metrics,
        )

    return _create_session_info
