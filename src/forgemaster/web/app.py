"""FastAPI application factory for Forgemaster.

This module provides the main application factory function that creates
and configures a FastAPI application with:
- CORS middleware for cross-origin requests
- Request logging middleware with correlation IDs
- Database connection lifecycle management
- Health and readiness endpoints

Example usage:
    >>> from forgemaster.config import ForgemasterConfig
    >>> from forgemaster.web.app import create_app
    >>>
    >>> config = ForgemasterConfig()
    >>> app = create_app(config)
    >>>
    >>> # Run with uvicorn
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from forgemaster.config import ForgemasterConfig
from forgemaster.database.connection import get_engine, get_session_factory
from forgemaster.logging import get_logger
from forgemaster.web.middleware import RequestLoggingMiddleware
from forgemaster.web.routes.dashboard import create_dashboard_router
from forgemaster.web.routes.events import create_events_router
from forgemaster.web.routes.health import create_health_router
from forgemaster.web.routes.lessons import create_lessons_router
from forgemaster.web.routes.projects import create_projects_router
from forgemaster.web.routes.sessions import create_sessions_router
from forgemaster.web.routes.tasks import create_tasks_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

logger = get_logger(__name__)

# Version from package (fallback to dev if not installed)
APP_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle with database connections.

    This context manager handles:
    - Creating database engine and session factory on startup
    - Storing references in app.state for dependency injection
    - Disposing of database connections on shutdown

    Args:
        app: FastAPI application instance

    Yields:
        None after startup, cleans up on context exit
    """
    # Access config from app.state (set in create_app)
    config: ForgemasterConfig = app.state.config

    logger.info("app_startup_begin", host=config.web.host, port=config.web.port)

    # Initialize database connection pool
    engine: AsyncEngine = get_engine(config.database)
    session_factory: async_sessionmaker[AsyncSession] = get_session_factory(engine)

    # Store in app.state for dependency injection
    app.state.engine = engine
    app.state.session_factory = session_factory

    logger.info(
        "database_pool_initialized",
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
    )

    yield

    # Shutdown: dispose of database connections
    logger.info("app_shutdown_begin")
    await engine.dispose()
    logger.info("database_pool_disposed")


def create_app(config: ForgemasterConfig | None = None) -> FastAPI:
    """Create and configure a FastAPI application.

    Creates a fully configured FastAPI application with:
    - CORS middleware based on config.web.cors_origins
    - Request logging middleware with correlation IDs
    - Health and readiness endpoints at /health/
    - Database connection lifecycle management

    Args:
        config: Optional ForgemasterConfig. If None, creates default config.

    Returns:
        Configured FastAPI application instance.

    Example:
        >>> from forgemaster.config import ForgemasterConfig, WebConfig
        >>>
        >>> # Default configuration
        >>> app = create_app()
        >>>
        >>> # Custom configuration
        >>> config = ForgemasterConfig(
        ...     web=WebConfig(cors_origins=["https://example.com"])
        ... )
        >>> app = create_app(config)
    """
    if config is None:
        config = ForgemasterConfig()

    # Create FastAPI app with lifespan manager
    app = FastAPI(
        title="Forgemaster",
        version=APP_VERSION,
        description="Autonomous development orchestration system",
        lifespan=lifespan,
    )

    # Store config in app.state for lifespan access
    app.state.config = config

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.web.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Register routers
    health_router = create_health_router()
    app.include_router(health_router)

    projects_router = create_projects_router()
    app.include_router(projects_router)

    tasks_router = create_tasks_router()
    app.include_router(tasks_router)

    sessions_router = create_sessions_router()
    app.include_router(sessions_router)

    lessons_router = create_lessons_router()
    app.include_router(lessons_router)

    events_router = create_events_router()
    app.include_router(events_router)

    dashboard_router = create_dashboard_router()
    app.include_router(dashboard_router)

    logger.info(
        "app_created",
        cors_origins=config.web.cors_origins,
        version=APP_VERSION,
    )

    return app
