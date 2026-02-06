"""Server-Sent Events (SSE) endpoint for real-time updates.

Provides a streaming endpoint that broadcasts task status changes and session
activity updates to connected web clients.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from forgemaster.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class SSEEventType(str, Enum):
    """Types of SSE events."""

    TASK_STATUS = "task_status"
    SESSION_ACTIVITY = "session_activity"
    SYSTEM = "system"


@dataclass
class SSEEvent:
    """Server-Sent Event data structure."""

    event: SSEEventType
    data: dict
    id: str | None = None
    retry: int | None = None

    def to_dict(self) -> dict:
        """Convert event to dictionary format for SSE transmission."""
        result = {
            "event": self.event.value,
            "data": json.dumps(self.data),
        }
        if self.id is not None:
            result["id"] = self.id
        if self.retry is not None:
            result["retry"] = self.retry
        return result


class EventBroadcaster:
    """Manages SSE client connections and event broadcasting.

    Maintains a list of connected client queues and broadcasts events to all
    subscribers. Handles client connection/disconnection lifecycle.
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[SSEEvent | None]] = []
        self._lock = asyncio.Lock()
        self.logger = get_logger(__name__)

    async def subscribe(self) -> AsyncIterator[SSEEvent]:
        """Subscribe to events, yields events as they arrive.

        Yields:
            SSEEvent objects as they are broadcast to subscribers.
        """
        queue: asyncio.Queue[SSEEvent | None] = asyncio.Queue()
        async with self._lock:
            self._queues.append(queue)
        self.logger.info("sse_client_connected", total_clients=len(self._queues))
        try:
            while True:
                event = await queue.get()
                if event is None:  # Shutdown signal
                    break
                yield event
        finally:
            async with self._lock:
                self._queues.remove(queue)
            self.logger.info("sse_client_disconnected", total_clients=len(self._queues))

    async def broadcast(self, event: SSEEvent) -> None:
        """Broadcast event to all connected clients.

        Args:
            event: The SSEEvent to broadcast to all subscribers.
        """
        async with self._lock:
            for queue in self._queues:
                await queue.put(event)
        self.logger.debug(
            "sse_event_broadcast",
            event_type=event.event.value,
            client_count=len(self._queues),
        )

    async def broadcast_task_status(
        self, task_id: UUID, status: str, details: dict | None = None
    ) -> None:
        """Broadcast task status change.

        Args:
            task_id: ID of the task that changed status.
            status: New status of the task.
            details: Optional additional details about the status change.
        """
        data = {
            "task_id": str(task_id),
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if details:
            data["details"] = details
        await self.broadcast(SSEEvent(event=SSEEventType.TASK_STATUS, data=data))

    async def broadcast_session_activity(
        self, session_id: UUID, activity_type: str, details: dict | None = None
    ) -> None:
        """Broadcast session activity.

        Args:
            session_id: ID of the session with activity.
            activity_type: Type of activity that occurred.
            details: Optional additional details about the activity.
        """
        data = {
            "session_id": str(session_id),
            "activity_type": activity_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if details:
            data["details"] = details
        await self.broadcast(SSEEvent(event=SSEEventType.SESSION_ACTIVITY, data=data))


# Global broadcaster instance
_broadcaster: EventBroadcaster | None = None


def get_broadcaster() -> EventBroadcaster:
    """Get or create the global event broadcaster.

    Returns:
        The singleton EventBroadcaster instance.
    """
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = EventBroadcaster()
    return _broadcaster


def create_events_router() -> APIRouter:
    """Create the events router with SSE streaming endpoint.

    Returns:
        FastAPI router configured with the /events/stream endpoint.
    """
    router = APIRouter(prefix="/events", tags=["events"])

    @router.get("/stream")
    async def stream_events(request: Request) -> EventSourceResponse:
        """Stream server-sent events to connected clients.

        Establishes a long-lived connection and streams events as they occur.
        Connection is automatically cleaned up when the client disconnects.

        Args:
            request: FastAPI request object for disconnection detection.

        Returns:
            EventSourceResponse that streams SSE events.
        """
        broadcaster = get_broadcaster()

        async def event_generator() -> AsyncIterator[dict]:
            async for event in broadcaster.subscribe():
                if await request.is_disconnected():
                    break
                yield event.to_dict()

        return EventSourceResponse(event_generator())

    return router
