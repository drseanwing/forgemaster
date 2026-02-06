"""Integration tests for Server-Sent Events (SSE) endpoint."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from httpx import AsyncClient

if TYPE_CHECKING:
    from forgemaster.web.routes.events import EventBroadcaster


@pytest.mark.asyncio
async def test_sse_endpoint_returns_event_source_response(
    async_client: AsyncClient,
) -> None:
    """Test that SSE endpoint returns EventSourceResponse."""
    # Use a timeout to prevent hanging if the endpoint doesn't stream
    async with asyncio.timeout(2):
        async with async_client.stream("GET", "/events/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            assert response.headers["cache-control"] == "no-cache"
            # Close the connection immediately
            break


@pytest.mark.asyncio
async def test_event_format(
    async_client: AsyncClient,
    event_broadcaster: EventBroadcaster,
) -> None:
    """Test that events are formatted correctly as SSE."""
    task_id = uuid4()
    status = "running"

    # Start streaming in background
    async def stream_events() -> list[str]:
        events = []
        async with async_client.stream("GET", "/events/stream") as response:
            async for line in response.aiter_lines():
                events.append(line)
                if len(events) >= 4:  # event, data, id (optional), empty line
                    break
        return events

    stream_task = asyncio.create_task(stream_events())
    await asyncio.sleep(0.1)  # Give time for connection to establish

    # Broadcast a task status event
    await event_broadcaster.broadcast_task_status(task_id, status)

    events = await asyncio.wait_for(stream_task, timeout=2)

    # Parse the SSE format
    event_lines = [line for line in events if line]
    assert len(event_lines) >= 2

    # Check event type
    event_type_line = next((line for line in event_lines if line.startswith("event:")), None)
    assert event_type_line is not None
    assert "task_status" in event_type_line

    # Check data payload
    data_line = next((line for line in event_lines if line.startswith("data:")), None)
    assert data_line is not None
    data_json = data_line.replace("data:", "", 1).strip()
    data = json.loads(data_json)

    assert data["task_id"] == str(task_id)
    assert data["status"] == status
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_task_status_event(
    event_broadcaster: EventBroadcaster,
) -> None:
    """Test task status event broadcasting."""
    task_id = uuid4()
    status = "completed"
    details = {"result": "success"}

    events_received = []

    async def subscriber() -> None:
        async for event in event_broadcaster.subscribe():
            events_received.append(event)
            break  # Get first event only

    subscriber_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.1)  # Give time for subscription

    await event_broadcaster.broadcast_task_status(task_id, status, details)
    await asyncio.wait_for(subscriber_task, timeout=2)

    assert len(events_received) == 1
    event = events_received[0]
    assert event.event.value == "task_status"
    assert event.data["task_id"] == str(task_id)
    assert event.data["status"] == status
    assert event.data["details"] == details
    assert "timestamp" in event.data


@pytest.mark.asyncio
async def test_session_activity_event(
    event_broadcaster: EventBroadcaster,
) -> None:
    """Test session activity event broadcasting."""
    session_id = uuid4()
    activity_type = "heartbeat"
    details = {"cpu_usage": 45.2}

    events_received = []

    async def subscriber() -> None:
        async for event in event_broadcaster.subscribe():
            events_received.append(event)
            break

    subscriber_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.1)

    await event_broadcaster.broadcast_session_activity(session_id, activity_type, details)
    await asyncio.wait_for(subscriber_task, timeout=2)

    assert len(events_received) == 1
    event = events_received[0]
    assert event.event.value == "session_activity"
    assert event.data["session_id"] == str(session_id)
    assert event.data["activity_type"] == activity_type
    assert event.data["details"] == details
    assert "timestamp" in event.data


@pytest.mark.asyncio
async def test_client_connection_and_disconnection(
    event_broadcaster: EventBroadcaster,
) -> None:
    """Test that clients can connect and disconnect cleanly."""
    initial_client_count = len(event_broadcaster._queues)

    # Start subscriber
    async def subscriber() -> None:
        async for event in event_broadcaster.subscribe():
            pass  # Just keep connection alive

    subscriber_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.1)

    # Client should be connected
    assert len(event_broadcaster._queues) == initial_client_count + 1

    # Cancel subscriber (simulates client disconnect)
    subscriber_task.cancel()
    try:
        await subscriber_task
    except asyncio.CancelledError:
        pass

    await asyncio.sleep(0.1)

    # Client should be disconnected
    assert len(event_broadcaster._queues) == initial_client_count


@pytest.mark.asyncio
async def test_multiple_clients_receive_same_event(
    event_broadcaster: EventBroadcaster,
) -> None:
    """Test that all connected clients receive broadcast events."""
    task_id = uuid4()
    status = "running"

    clients_received = []

    async def subscriber(client_id: int) -> None:
        async for event in event_broadcaster.subscribe():
            clients_received.append(client_id)
            break

    # Start 3 subscribers
    tasks = [asyncio.create_task(subscriber(i)) for i in range(3)]
    await asyncio.sleep(0.1)

    # Broadcast event
    await event_broadcaster.broadcast_task_status(task_id, status)
    await asyncio.gather(*tasks, return_exceptions=True)

    # All 3 clients should have received the event
    assert len(clients_received) == 3
    assert set(clients_received) == {0, 1, 2}


@pytest.mark.asyncio
async def test_event_timestamp_format(
    event_broadcaster: EventBroadcaster,
) -> None:
    """Test that event timestamps are in ISO format with UTC timezone."""
    task_id = uuid4()
    status = "pending"

    events_received = []

    async def subscriber() -> None:
        async for event in event_broadcaster.subscribe():
            events_received.append(event)
            break

    subscriber_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.1)

    before_time = datetime.now(timezone.utc)
    await event_broadcaster.broadcast_task_status(task_id, status)
    after_time = datetime.now(timezone.utc)
    await asyncio.wait_for(subscriber_task, timeout=2)

    assert len(events_received) == 1
    event = events_received[0]

    # Parse timestamp
    timestamp_str = event.data["timestamp"]
    timestamp = datetime.fromisoformat(timestamp_str)

    # Verify it's between before and after times
    assert before_time <= timestamp <= after_time
    # Verify it has timezone info
    assert timestamp.tzinfo is not None
