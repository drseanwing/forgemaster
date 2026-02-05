"""Pytest fixtures for integration tests.

Provides async database fixtures for testing query functions against
an in-memory SQLite database. While the production system uses PostgreSQL
with pgvector extensions, these tests use SQLite for fast, isolated testing
of query logic.

Tests that require PostgreSQL-specific features (pgvector, tsvector) should
be marked with @pytest.mark.postgres to skip them in SQLite environments.
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from forgemaster.database.models.base import Base


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Use asyncio as the async backend for pytest-asyncio.

    Returns:
        The name of the async backend to use.
    """
    return "asyncio"


@pytest_asyncio.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create an in-memory SQLite async engine for testing.

    Yields:
        Configured AsyncEngine instance using in-memory SQLite.
    """
    # Use aiosqlite for async SQLite support
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield test_engine

    # Cleanup
    await test_engine.dispose()


@pytest_asyncio.fixture
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the test engine.

    Args:
        engine: The test database engine.

    Returns:
        Configured async_sessionmaker for creating test sessions.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture
async def db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Create a new async database session for each test.

    The session is automatically rolled back after the test completes
    to ensure test isolation.

    Args:
        session_factory: The session factory fixture.

    Yields:
        AsyncSession instance for the test.
    """
    async with session_factory() as session:
        yield session
        await session.rollback()
