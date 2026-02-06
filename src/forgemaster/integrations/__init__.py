"""Integration modules for external systems."""

from __future__ import annotations

from forgemaster.integrations.n8n import (
    N8nClient,
    N8nConfig,
    N8nEventType,
    N8nPayload,
)

__all__ = [
    "N8nClient",
    "N8nConfig",
    "N8nEventType",
    "N8nPayload",
]
