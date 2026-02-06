"""Web interface for Forgemaster.

This module provides the FastAPI-based dashboard and API endpoints for
monitoring agent sessions, task queues, and system health via htmx and SSE.
"""

from __future__ import annotations

from forgemaster.web.app import create_app
from forgemaster.web.middleware import RequestLoggingMiddleware

__all__ = [
    "create_app",
    "RequestLoggingMiddleware",
]
