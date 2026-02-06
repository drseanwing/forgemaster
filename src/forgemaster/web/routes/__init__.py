"""FastAPI route definitions for Forgemaster web interface.

This module contains API and dashboard route handlers for projects, tasks,
agents, health checks, and webhook endpoints.
"""

from __future__ import annotations

from forgemaster.web.routes.health import (
    HealthResponse,
    ReadinessResponse,
    create_health_router,
)

__all__ = [
    "HealthResponse",
    "ReadinessResponse",
    "create_health_router",
]
