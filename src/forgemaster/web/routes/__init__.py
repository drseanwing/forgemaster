"""FastAPI route definitions for Forgemaster web interface.

This module contains API and dashboard route handlers for projects, tasks,
sessions, lessons, health checks, SSE events, and dashboard views.
"""

from __future__ import annotations

from forgemaster.web.routes.dashboard import create_dashboard_router
from forgemaster.web.routes.events import (
    EventBroadcaster,
    SSEEvent,
    SSEEventType,
    create_events_router,
    get_broadcaster,
)
from forgemaster.web.routes.health import (
    HealthResponse,
    ReadinessResponse,
    create_health_router,
)
from forgemaster.web.routes.lessons import (
    LessonResponse,
    LessonSearchFiles,
    LessonSearchText,
    create_lessons_router,
)
from forgemaster.web.routes.projects import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    create_projects_router,
)
from forgemaster.web.routes.sessions import (
    SessionResponse,
    create_sessions_router,
)
from forgemaster.web.routes.tasks import (
    TaskCreate,
    TaskResponse,
    TaskStatusUpdate,
    create_tasks_router,
)

__all__ = [
    # Dashboard
    "create_dashboard_router",
    # Events / SSE
    "EventBroadcaster",
    "SSEEvent",
    "SSEEventType",
    "create_events_router",
    "get_broadcaster",
    # Health
    "HealthResponse",
    "ReadinessResponse",
    "create_health_router",
    # Lessons
    "LessonResponse",
    "LessonSearchFiles",
    "LessonSearchText",
    "create_lessons_router",
    # Projects
    "ProjectCreate",
    "ProjectResponse",
    "ProjectUpdate",
    "create_projects_router",
    # Sessions
    "SessionResponse",
    "create_sessions_router",
    # Tasks
    "TaskCreate",
    "TaskResponse",
    "TaskStatusUpdate",
    "create_tasks_router",
]
