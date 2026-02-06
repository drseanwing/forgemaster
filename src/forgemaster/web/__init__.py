"""Web interface for Forgemaster.

This module provides the FastAPI-based dashboard and API endpoints for
monitoring agent sessions, task queues, and system health via htmx and SSE.

Also includes webhook dispatcher for sending event notifications to external
services with HMAC signature verification and retry support.
"""

from __future__ import annotations

from forgemaster.web.app import create_app
from forgemaster.web.middleware import RequestLoggingMiddleware
from forgemaster.web.webhooks import (
    WebhookConfig,
    WebhookDispatcher,
    WebhookEndpoint,
    WebhookEvent,
    WebhookPayload,
)

__all__ = [
    "create_app",
    "RequestLoggingMiddleware",
    "WebhookConfig",
    "WebhookDispatcher",
    "WebhookEndpoint",
    "WebhookEvent",
    "WebhookPayload",
]
