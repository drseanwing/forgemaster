"""Health check endpoints for Forgemaster.

This module provides health and readiness endpoints for:
- Kubernetes liveness probes (/health/)
- Kubernetes readiness probes (/health/ready)
- Load balancer health checks

The readiness endpoint verifies database connectivity to ensure
the application is ready to serve traffic.

Example:
    >>> from fastapi import FastAPI
    >>> from forgemaster.web.routes.health import create_health_router
    >>>
    >>> app = FastAPI()
    >>> app.include_router(create_health_router())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import text

from forgemaster.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Current health status ("ok", "degraded", "unhealthy")
    """

    status: str


class ReadinessResponse(BaseModel):
    """Readiness check response model.

    Attributes:
        status: Current readiness status ("ok", "unhealthy")
        database: Database connectivity status ("connected", "disconnected")
    """

    status: str
    database: str


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Dependency that retrieves session factory from app state.

    Args:
        request: FastAPI request object

    Returns:
        Session factory from app.state
    """
    return request.app.state.session_factory  # type: ignore[return-value]


def create_health_router() -> APIRouter:
    """Create health check router with endpoints.

    Returns:
        Configured APIRouter with health endpoints.

    Routes:
        GET /health/ - Basic liveness check
        GET /health/ready - Readiness check with database verification
    """
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("/", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        """Basic liveness check.

        Returns:
            Simple status response indicating the service is alive.
        """
        return {"status": "ok"}

    @router.get("/ready", response_model=ReadinessResponse)
    async def readiness(
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> dict[str, Any]:
        """Readiness check with database connectivity verification.

        Verifies that:
        - Database connection pool is available
        - Database accepts queries

        Args:
            session_factory: Injected session factory from app state

        Returns:
            Status response with database connectivity information.
        """
        try:
            async with session_factory() as session:
                # Execute simple query to verify connectivity
                await session.execute(text("SELECT 1"))

            logger.debug("readiness_check_passed", database="connected")
            return {
                "status": "ok",
                "database": "connected",
            }

        except Exception as exc:
            logger.warning(
                "readiness_check_failed",
                database="disconnected",
                error=str(exc),
            )
            return {
                "status": "unhealthy",
                "database": "disconnected",
            }

    return router
