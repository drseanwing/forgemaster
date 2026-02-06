"""Web interface for Forgemaster.

This module provides the FastAPI-based dashboard and API endpoints for
monitoring agent sessions, task queues, and system health via htmx and SSE.

Also includes webhook dispatcher for sending event notifications to external
services with HMAC signature verification and retry support.
"""

from __future__ import annotations

from forgemaster.web.app import create_app
from forgemaster.web.middleware import RequestLoggingMiddleware
from forgemaster.web.routes.events import (
    EventBroadcaster,
    SSEEvent,
    SSEEventType,
    get_broadcaster,
)
from forgemaster.web.webhooks import (
    WebhookConfig,
    WebhookDispatcher,
    WebhookEndpoint,
    WebhookEvent,
    WebhookPayload,
)

__all__ = [
    # Application
    "create_app",
    "RequestLoggingMiddleware",
    # SSE Events
    "EventBroadcaster",
    "SSEEvent",
    "SSEEventType",
    "get_broadcaster",
    # Webhooks
    "WebhookConfig",
    "WebhookDispatcher",
    "WebhookEndpoint",
    "WebhookEvent",
    "WebhookPayload",
]
