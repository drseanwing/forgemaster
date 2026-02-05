"""Database connection management for Forgemaster.

This module provides factory functions for creating SQLAlchemy async engines
and session factories, configured from the application's DatabaseConfig.

The connection manager uses asyncpg as the PostgreSQL driver and supports
connection pooling with configurable pool size and overflow limits.

Example usage:
    >>> from forgemaster.config import DatabaseConfig
    >>> from forgemaster.database.connection import get_engine, get_session_factory
    >>>
    >>> config = DatabaseConfig(url="postgresql+asyncpg://localhost/forgemaster")
    >>> engine = get_engine(config)
    >>> SessionFactory = get_session_factory(engine)
    >>>
    >>> async with SessionFactory() as session:
    ...     result = await session.execute(select(Project))
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from forgemaster.config import DatabaseConfig


def get_engine(config: DatabaseConfig) -> AsyncEngine:
    """Create an async SQLAlchemy engine from database configuration.

    Configures connection pooling using the pool_size and max_overflow
    settings from DatabaseConfig. The engine uses asyncpg as the
    underlying PostgreSQL driver.

    Args:
        config: Database configuration containing URL, pool settings,
                and SQL echo preference.

    Returns:
        Configured AsyncEngine instance with connection pooling.
    """
    return create_async_engine(
        config.url,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        echo=config.echo,
    )


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the given engine.

    The returned factory produces AsyncSession instances configured with:
    - expire_on_commit=False to allow accessing attributes after commit
      without triggering lazy loads (important for async contexts)

    Args:
        engine: AsyncEngine to bind sessions to.

    Returns:
        Configured async_sessionmaker that produces AsyncSession instances.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
