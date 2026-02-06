"""Integration tests for n8n webhook client."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
import respx

from forgemaster.integrations.n8n import N8nClient, N8nConfig, N8nEventType, N8nPayload


def test_n8n_payload_to_dict() -> None:
    """Verify payload serialization."""
    project_id = uuid4()
    task_id = uuid4()
    session_id = uuid4()
    timestamp = datetime.now(timezone.utc)

    payload = N8nPayload(
        event_type=N8nEventType.TASK_COMPLETED,
        timestamp=timestamp,
        project_id=project_id,
        task_id=task_id,
        session_id=session_id,
        data={"status": "done"},
    )

    result = payload.to_dict()

    assert result["event_type"] == "task_completed"
    assert result["timestamp"] == timestamp.isoformat()
    assert result["project_id"] == str(project_id)
    assert result["task_id"] == str(task_id)
    assert result["session_id"] == str(session_id)
    assert result["data"] == {"status": "done"}


def test_n8n_payload_to_dict_with_nulls() -> None:
    """Verify payload serialization with null UUIDs."""
    timestamp = datetime.now(timezone.utc)

    payload = N8nPayload(
        event_type=N8nEventType.BUILD_SUCCESS,
        timestamp=timestamp,
        project_id=None,
        task_id=None,
        session_id=None,
        data={},
    )

    result = payload.to_dict()

    assert result["event_type"] == "build_success"
    assert result["timestamp"] == timestamp.isoformat()
    assert result["project_id"] is None
    assert result["task_id"] is None
    assert result["session_id"] is None
    assert result["data"] == {}


@respx.mock
@pytest.mark.asyncio
async def test_n8n_client_send_success() -> None:
    """Mock successful webhook."""
    webhook_url = "https://n8n.example.com/webhook/test"
    respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    payload = N8nPayload(
        event_type=N8nEventType.TASK_COMPLETED,
        timestamp=datetime.now(timezone.utc),
        project_id=uuid4(),
        task_id=uuid4(),
        data={"status": "done"},
    )

    result = await client.send(payload)
    await client.close()

    assert result is True


@respx.mock
@pytest.mark.asyncio
async def test_n8n_client_send_failure() -> None:
    """Mock failed webhook (non-2xx)."""
    webhook_url = "https://n8n.example.com/webhook/test"
    respx.post(webhook_url).mock(return_value=httpx.Response(500, text="Internal Error"))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    payload = N8nPayload(
        event_type=N8nEventType.TASK_FAILED,
        timestamp=datetime.now(timezone.utc),
        project_id=uuid4(),
        task_id=uuid4(),
        data={"error": "oops"},
    )

    result = await client.send(payload)
    await client.close()

    assert result is False


@respx.mock
@pytest.mark.asyncio
async def test_n8n_client_send_error() -> None:
    """Mock network error."""
    webhook_url = "https://n8n.example.com/webhook/test"
    respx.post(webhook_url).mock(side_effect=httpx.ConnectError("Connection failed"))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    payload = N8nPayload(
        event_type=N8nEventType.BUILD_SUCCESS,
        timestamp=datetime.now(timezone.utc),
        project_id=uuid4(),
        data={},
    )

    result = await client.send(payload)
    await client.close()

    assert result is False


@pytest.mark.asyncio
async def test_n8n_client_disabled() -> None:
    """Verify disabled client returns True without sending."""
    config = N8nConfig(
        webhook_url="https://n8n.example.com/webhook/test",
        enabled=False,
    )
    client = N8nClient(config)

    payload = N8nPayload(
        event_type=N8nEventType.TASK_COMPLETED,
        timestamp=datetime.now(timezone.utc),
        project_id=uuid4(),
        task_id=uuid4(),
        data={},
    )

    # Should return True without making any HTTP call
    result = await client.send(payload)
    await client.close()

    assert result is True


@respx.mock
@pytest.mark.asyncio
async def test_n8n_client_with_auth_header() -> None:
    """Verify auth header is sent."""
    webhook_url = "https://n8n.example.com/webhook/test"
    auth_token = "Bearer secret-token"

    route = respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url, auth_header=auth_token)
    client = N8nClient(config)

    payload = N8nPayload(
        event_type=N8nEventType.TASK_COMPLETED,
        timestamp=datetime.now(timezone.utc),
        project_id=uuid4(),
        task_id=uuid4(),
        data={},
    )

    await client.send(payload)
    await client.close()

    # Verify the Authorization header was sent
    assert route.called
    assert route.calls.last.request.headers["Authorization"] == auth_token


@respx.mock
@pytest.mark.asyncio
async def test_notify_task_completed() -> None:
    """Test convenience method."""
    webhook_url = "https://n8n.example.com/webhook/test"
    respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    project_id = uuid4()
    task_id = uuid4()

    result = await client.notify_task_completed(
        project_id=project_id,
        task_id=task_id,
        title="Test task",
        status="completed",
        duration_seconds=120,
    )
    await client.close()

    assert result is True


@respx.mock
@pytest.mark.asyncio
async def test_notify_task_failed() -> None:
    """Test convenience method."""
    webhook_url = "https://n8n.example.com/webhook/test"
    respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    project_id = uuid4()
    task_id = uuid4()

    result = await client.notify_task_failed(
        project_id=project_id,
        task_id=task_id,
        title="Test task",
        error="Something went wrong",
        retry_count=3,
    )
    await client.close()

    assert result is True


@respx.mock
@pytest.mark.asyncio
async def test_notify_build_result_success() -> None:
    """Test build success."""
    webhook_url = "https://n8n.example.com/webhook/test"
    route = respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    project_id = uuid4()

    result = await client.notify_build_result(
        project_id=project_id,
        success=True,
        version="1.0.0",
        error=None,
    )
    await client.close()

    assert result is True

    # Verify event_type is BUILD_SUCCESS
    sent_json = route.calls.last.request.content
    import json

    sent_data = json.loads(sent_json)
    assert sent_data["event_type"] == "build_success"
    assert sent_data["data"]["success"] is True
    assert sent_data["data"]["version"] == "1.0.0"


@respx.mock
@pytest.mark.asyncio
async def test_notify_build_result_failure() -> None:
    """Test build failure."""
    webhook_url = "https://n8n.example.com/webhook/test"
    route = respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    project_id = uuid4()

    result = await client.notify_build_result(
        project_id=project_id,
        success=False,
        version=None,
        error="Build failed",
    )
    await client.close()

    assert result is True

    # Verify event_type is BUILD_FAILURE
    sent_json = route.calls.last.request.content
    import json

    sent_data = json.loads(sent_json)
    assert sent_data["event_type"] == "build_failure"
    assert sent_data["data"]["success"] is False
    assert sent_data["data"]["error"] == "Build failed"


@respx.mock
@pytest.mark.asyncio
async def test_notify_deploy_success() -> None:
    """Test deployment notification."""
    webhook_url = "https://n8n.example.com/webhook/test"
    respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    project_id = uuid4()

    result = await client.notify_deploy_success(
        project_id=project_id,
        version="1.0.0",
        environment="production",
    )
    await client.close()

    assert result is True


@respx.mock
@pytest.mark.asyncio
async def test_notify_review_requested() -> None:
    """Test review request."""
    webhook_url = "https://n8n.example.com/webhook/test"
    route = respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    project_id = uuid4()
    task_id = uuid4()

    result = await client.notify_review(
        project_id=project_id,
        task_id=task_id,
        reviewer="architect",
        approved=None,  # Not yet reviewed
        findings=None,
    )
    await client.close()

    assert result is True

    # Verify event_type is REVIEW_REQUESTED
    sent_json = route.calls.last.request.content
    import json

    sent_data = json.loads(sent_json)
    assert sent_data["event_type"] == "review_requested"
    assert sent_data["data"]["reviewer"] == "architect"
    assert sent_data["data"]["approved"] is None


@respx.mock
@pytest.mark.asyncio
async def test_notify_review_completed() -> None:
    """Test review completion."""
    webhook_url = "https://n8n.example.com/webhook/test"
    route = respx.post(webhook_url).mock(return_value=httpx.Response(200))

    config = N8nConfig(webhook_url=webhook_url)
    client = N8nClient(config)

    project_id = uuid4()
    task_id = uuid4()

    result = await client.notify_review(
        project_id=project_id,
        task_id=task_id,
        reviewer="architect",
        approved=True,
        findings=["Minor: Add docstring", "Good error handling"],
    )
    await client.close()

    assert result is True

    # Verify event_type is REVIEW_COMPLETED
    sent_json = route.calls.last.request.content
    import json

    sent_data = json.loads(sent_json)
    assert sent_data["event_type"] == "review_completed"
    assert sent_data["data"]["reviewer"] == "architect"
    assert sent_data["data"]["approved"] is True
    assert len(sent_data["data"]["findings"]) == 2
