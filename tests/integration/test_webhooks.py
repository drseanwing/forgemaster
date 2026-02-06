"""Integration tests for webhook dispatcher.

This module provides integration tests for the webhook system including
payload formatting, HMAC signature generation, retry logic, and event delivery.

All tests use mocked httpx.AsyncClient to avoid actual HTTP requests while
verifying the webhook dispatcher's behavior.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from forgemaster.web.webhooks import (
    WebhookDispatcher,
    WebhookEndpoint,
    WebhookEvent,
    WebhookPayload,
)


@pytest.fixture
def sample_endpoint() -> WebhookEndpoint:
    """Create a sample webhook endpoint for testing.

    Returns:
        WebhookEndpoint configured for testing.
    """
    return WebhookEndpoint(
        url="https://api.example.com/webhook",
        secret="test-secret-key",
        enabled=True,
        retry_count=3,
        timeout_seconds=30,
    )


@pytest.fixture
def sample_endpoint_no_secret() -> WebhookEndpoint:
    """Create a sample webhook endpoint without a secret.

    Returns:
        WebhookEndpoint configured without signature verification.
    """
    return WebhookEndpoint(
        url="https://api.example.com/webhook",
        secret=None,
        enabled=True,
        retry_count=2,
        timeout_seconds=10,
    )


@pytest.fixture
def disabled_endpoint() -> WebhookEndpoint:
    """Create a disabled webhook endpoint.

    Returns:
        WebhookEndpoint that is disabled.
    """
    return WebhookEndpoint(
        url="https://api.example.com/disabled",
        secret="secret",
        enabled=False,
    )


@pytest.fixture
def dispatcher(sample_endpoint: WebhookEndpoint) -> WebhookDispatcher:
    """Create a webhook dispatcher with a sample endpoint.

    Args:
        sample_endpoint: The sample endpoint fixture.

    Returns:
        WebhookDispatcher configured with the sample endpoint.
    """
    return WebhookDispatcher(endpoints=[sample_endpoint])


@pytest.fixture
def mock_response_success() -> MagicMock:
    """Create a mock successful HTTP response.

    Returns:
        MagicMock configured as a successful response.
    """
    response = MagicMock()
    response.is_success = True
    response.status_code = 200
    response.request = MagicMock()
    return response


@pytest.fixture
def mock_response_failure() -> MagicMock:
    """Create a mock failed HTTP response.

    Returns:
        MagicMock configured as a failed response.
    """
    response = MagicMock()
    response.is_success = False
    response.status_code = 500
    response.request = MagicMock()
    return response


class TestWebhookPayload:
    """Tests for WebhookPayload model."""

    def test_payload_creation(self) -> None:
        """Test creating a webhook payload with required fields."""
        payload = WebhookPayload(
            event=WebhookEvent.TASK_COMPLETED,
            timestamp="2024-02-05T12:00:00Z",
            data={"task_id": "123", "status": "done"},
        )

        assert payload.event == WebhookEvent.TASK_COMPLETED
        assert payload.timestamp == "2024-02-05T12:00:00Z"
        assert payload.data["task_id"] == "123"
        assert payload.data["status"] == "done"

    def test_payload_serialization(self) -> None:
        """Test payload JSON serialization."""
        payload = WebhookPayload(
            event=WebhookEvent.BUILD_FAILURE,
            timestamp="2024-02-05T12:00:00Z",
            data={"error": "Build failed"},
        )

        json_str = payload.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["event"] == "build.failure"
        assert parsed["timestamp"] == "2024-02-05T12:00:00Z"
        assert parsed["data"]["error"] == "Build failed"

    def test_payload_with_empty_data(self) -> None:
        """Test payload creation with empty data."""
        payload = WebhookPayload(
            event=WebhookEvent.DEPLOY_SUCCESS,
            timestamp="2024-02-05T12:00:00Z",
            data={},
        )

        assert payload.data == {}


class TestWebhookEndpoint:
    """Tests for WebhookEndpoint configuration."""

    def test_endpoint_defaults(self) -> None:
        """Test endpoint default values."""
        endpoint = WebhookEndpoint(url="https://api.example.com/hook")

        assert endpoint.url == "https://api.example.com/hook"
        assert endpoint.secret is None
        assert endpoint.enabled is True
        assert endpoint.retry_count == 3
        assert endpoint.timeout_seconds == 30

    def test_endpoint_custom_values(self) -> None:
        """Test endpoint with custom configuration."""
        endpoint = WebhookEndpoint(
            url="https://custom.example.com/hook",
            secret="my-secret",
            enabled=False,
            retry_count=5,
            timeout_seconds=60,
        )

        assert endpoint.url == "https://custom.example.com/hook"
        assert endpoint.secret == "my-secret"
        assert endpoint.enabled is False
        assert endpoint.retry_count == 5
        assert endpoint.timeout_seconds == 60


class TestSignatureGeneration:
    """Tests for HMAC signature generation."""

    def test_sign_payload(self, dispatcher: WebhookDispatcher) -> None:
        """Test HMAC-SHA256 signature generation.

        Args:
            dispatcher: The dispatcher fixture.
        """
        payload = '{"event": "test", "data": {}}'
        secret = "test-secret"

        signature = dispatcher._sign_payload(payload, secret)

        # Verify manually
        expected = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        assert signature == expected
        assert len(signature) == 64  # SHA256 hex digest length

    def test_sign_payload_different_secrets(self, dispatcher: WebhookDispatcher) -> None:
        """Test that different secrets produce different signatures.

        Args:
            dispatcher: The dispatcher fixture.
        """
        payload = '{"event": "test"}'

        sig1 = dispatcher._sign_payload(payload, "secret1")
        sig2 = dispatcher._sign_payload(payload, "secret2")

        assert sig1 != sig2

    def test_sign_payload_different_payloads(self, dispatcher: WebhookDispatcher) -> None:
        """Test that different payloads produce different signatures.

        Args:
            dispatcher: The dispatcher fixture.
        """
        secret = "same-secret"

        sig1 = dispatcher._sign_payload('{"event": "event1"}', secret)
        sig2 = dispatcher._sign_payload('{"event": "event2"}', secret)

        assert sig1 != sig2


class TestHeaderBuilding:
    """Tests for HTTP header construction."""

    def test_build_headers_with_secret(self, dispatcher: WebhookDispatcher) -> None:
        """Test header building with signature.

        Args:
            dispatcher: The dispatcher fixture.
        """
        headers = dispatcher._build_headers(
            event=WebhookEvent.TASK_COMPLETED,
            payload_str='{"test": "data"}',
            secret="my-secret",
        )

        assert headers["Content-Type"] == "application/json"
        assert headers["X-Forgemaster-Event"] == "task.completed"
        assert "X-Forgemaster-Timestamp" in headers
        assert "X-Forgemaster-Signature" in headers
        assert len(headers["X-Forgemaster-Signature"]) == 64

    def test_build_headers_without_secret(self, dispatcher: WebhookDispatcher) -> None:
        """Test header building without signature.

        Args:
            dispatcher: The dispatcher fixture.
        """
        headers = dispatcher._build_headers(
            event=WebhookEvent.BUILD_FAILURE,
            payload_str='{"test": "data"}',
            secret=None,
        )

        assert headers["Content-Type"] == "application/json"
        assert headers["X-Forgemaster-Event"] == "build.failure"
        assert "X-Forgemaster-Timestamp" in headers
        assert "X-Forgemaster-Signature" not in headers

    def test_timestamp_is_unix_epoch(self, dispatcher: WebhookDispatcher) -> None:
        """Test that timestamp header is a Unix epoch timestamp.

        Args:
            dispatcher: The dispatcher fixture.
        """
        headers = dispatcher._build_headers(
            event=WebhookEvent.DEPLOY_SUCCESS,
            payload_str="{}",
            secret=None,
        )

        timestamp = int(headers["X-Forgemaster-Timestamp"])
        now = int(datetime.now(timezone.utc).timestamp())

        # Should be within 5 seconds of now
        assert abs(timestamp - now) < 5


class TestWebhookDispatcher:
    """Tests for WebhookDispatcher functionality."""

    def test_dispatcher_no_endpoints(self) -> None:
        """Test dispatcher initialization with no endpoints."""
        dispatcher = WebhookDispatcher()
        assert dispatcher.endpoints == []

    def test_dispatcher_with_endpoints(self, sample_endpoint: WebhookEndpoint) -> None:
        """Test dispatcher initialization with endpoints.

        Args:
            sample_endpoint: The sample endpoint fixture.
        """
        dispatcher = WebhookDispatcher(endpoints=[sample_endpoint])
        assert len(dispatcher.endpoints) == 1
        assert dispatcher.endpoints[0].url == sample_endpoint.url

    @pytest.mark.asyncio
    async def test_send_no_endpoints(self) -> None:
        """Test sending event with no endpoints configured."""
        dispatcher = WebhookDispatcher()

        result = await dispatcher.send(WebhookEvent.TASK_COMPLETED, {"task_id": "123"})

        assert result is True  # No endpoints means success

    @pytest.mark.asyncio
    async def test_send_disabled_endpoint(self, disabled_endpoint: WebhookEndpoint) -> None:
        """Test that disabled endpoints are skipped.

        Args:
            disabled_endpoint: The disabled endpoint fixture.
        """
        dispatcher = WebhookDispatcher(endpoints=[disabled_endpoint])

        with patch.object(dispatcher, "_get_client") as mock_get_client:
            result = await dispatcher.send(WebhookEvent.TASK_COMPLETED, {"task_id": "123"})

            assert result is True
            # Client should not be used for disabled endpoint
            mock_get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_success(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test successful webhook delivery.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_success)
        mock_client.is_closed = False

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send(
                WebhookEvent.TASK_COMPLETED,
                {"task_id": "123", "status": "done"},
            )

        assert result is True
        mock_client.post.assert_called_once()

        # Verify the call arguments
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["timeout"] == 30
        assert "X-Forgemaster-Event" in call_kwargs["headers"]
        assert "X-Forgemaster-Signature" in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_send_retry_on_failure(
        self,
        sample_endpoint: WebhookEndpoint,
        mock_response_failure: MagicMock,
        mock_response_success: MagicMock,
    ) -> None:
        """Test retry logic on initial failure.

        Args:
            sample_endpoint: The sample endpoint fixture.
            mock_response_failure: The mock failure response fixture.
            mock_response_success: The mock success response fixture.
        """
        # Reduce retry count for faster test
        sample_endpoint.retry_count = 2
        dispatcher = WebhookDispatcher(endpoints=[sample_endpoint])

        mock_client = AsyncMock()
        mock_client.is_closed = False
        # Fail twice, then succeed
        mock_client.post = AsyncMock(
            side_effect=[mock_response_failure, mock_response_failure, mock_response_success]
        )

        with (
            patch.object(dispatcher, "_get_client", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await dispatcher.send(
                WebhookEvent.BUILD_FAILURE,
                {"error": "Build failed"},
            )

        assert result is True
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_send_all_retries_failed(
        self,
        sample_endpoint: WebhookEndpoint,
        mock_response_failure: MagicMock,
    ) -> None:
        """Test behavior when all retries are exhausted.

        Args:
            sample_endpoint: The sample endpoint fixture.
            mock_response_failure: The mock failure response fixture.
        """
        sample_endpoint.retry_count = 2
        dispatcher = WebhookDispatcher(endpoints=[sample_endpoint])

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_failure)

        with (
            patch.object(dispatcher, "_get_client", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await dispatcher.send(
                WebhookEvent.TASK_COMPLETED,
                {"task_id": "123"},
            )

        assert result is False
        # Initial attempt + retries
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_send_timeout_error(
        self,
        sample_endpoint: WebhookEndpoint,
        mock_response_success: MagicMock,
    ) -> None:
        """Test handling of timeout errors.

        Args:
            sample_endpoint: The sample endpoint fixture.
            mock_response_success: The mock success response fixture.
        """
        sample_endpoint.retry_count = 1
        dispatcher = WebhookDispatcher(endpoints=[sample_endpoint])

        mock_client = AsyncMock()
        mock_client.is_closed = False
        # Timeout on first attempt, succeed on retry
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.TimeoutException("Connection timed out"),
                mock_response_success,
            ]
        )

        with (
            patch.object(dispatcher, "_get_client", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await dispatcher.send(
                WebhookEvent.DEPLOY_SUCCESS,
                {"version": "1.0.0"},
            )

        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_request_error(
        self,
        sample_endpoint: WebhookEndpoint,
    ) -> None:
        """Test handling of request errors.

        Args:
            sample_endpoint: The sample endpoint fixture.
        """
        sample_endpoint.retry_count = 0
        dispatcher = WebhookDispatcher(endpoints=[sample_endpoint])

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send(
                WebhookEvent.TASK_COMPLETED,
                {"task_id": "123"},
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_multiple_endpoints(
        self,
        sample_endpoint: WebhookEndpoint,
        sample_endpoint_no_secret: WebhookEndpoint,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending to multiple endpoints concurrently.

        Args:
            sample_endpoint: The sample endpoint fixture.
            sample_endpoint_no_secret: The endpoint without secret fixture.
            mock_response_success: The mock success response fixture.
        """
        sample_endpoint_no_secret.url = "https://api.other.com/webhook"
        dispatcher = WebhookDispatcher(endpoints=[sample_endpoint, sample_endpoint_no_secret])

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send(
                WebhookEvent.REVIEW_CYCLE,
                {"cycle": 1},
            )

        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing the dispatcher client."""
        dispatcher = WebhookDispatcher()
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        dispatcher._client = mock_client

        await dispatcher.close()

        mock_client.aclose.assert_called_once()
        assert dispatcher._client is None


class TestTaskCompletedWebhook:
    """Tests for task completion webhook."""

    @pytest.mark.asyncio
    async def test_send_task_completed(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending task completed webhook.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        task_id = uuid4()
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send_task_completed(
                task_id=task_id,
                status="done",
                result={"output": "Success!"},
            )

        assert result is True

        # Verify payload structure
        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["event"] == "task.completed"
        assert payload["data"]["task_id"] == str(task_id)
        assert payload["data"]["status"] == "done"
        assert payload["data"]["result"]["output"] == "Success!"
        assert "completed_at" in payload["data"]

    @pytest.mark.asyncio
    async def test_send_task_completed_no_result(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending task completed webhook without result.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        task_id = uuid4()
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send_task_completed(
                task_id=task_id,
                status="failed",
            )

        assert result is True

        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["data"]["result"] == {}


class TestReviewCycleWebhook:
    """Tests for review cycle webhook."""

    @pytest.mark.asyncio
    async def test_send_review_cycle(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending review cycle webhook.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        task_id = uuid4()
        findings = [
            {"severity": "warning", "message": "Consider refactoring"},
            {"severity": "info", "message": "Good test coverage"},
        ]

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send_review_cycle(
                task_id=task_id,
                cycle_number=2,
                reviewer="code-reviewer",
                approved=True,
                findings=findings,
            )

        assert result is True

        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["event"] == "review.cycle"
        assert payload["data"]["task_id"] == str(task_id)
        assert payload["data"]["cycle_number"] == 2
        assert payload["data"]["reviewer"] == "code-reviewer"
        assert payload["data"]["approved"] is True
        assert len(payload["data"]["findings"]) == 2
        assert "reviewed_at" in payload["data"]

    @pytest.mark.asyncio
    async def test_send_review_cycle_rejected(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending review cycle webhook for rejected review.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        task_id = uuid4()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send_review_cycle(
                task_id=task_id,
                cycle_number=1,
                reviewer="security-reviewer",
                approved=False,
                findings=[{"severity": "critical", "message": "SQL injection vulnerability"}],
            )

        assert result is True

        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["data"]["approved"] is False


class TestBuildFailureWebhook:
    """Tests for build failure webhook."""

    @pytest.mark.asyncio
    async def test_send_build_failure(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending build failure webhook.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        task_id = uuid4()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send_build_failure(
                task_id=task_id,
                error="Compilation failed: undefined reference to 'main'",
                logs="gcc -o app main.c\n/usr/bin/ld: error...",
            )

        assert result is True

        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["event"] == "build.failure"
        assert payload["data"]["task_id"] == str(task_id)
        assert "Compilation failed" in payload["data"]["error"]
        assert "gcc" in payload["data"]["logs"]
        assert "failed_at" in payload["data"]

    @pytest.mark.asyncio
    async def test_send_build_failure_no_logs(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending build failure webhook without logs.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        task_id = uuid4()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send_build_failure(
                task_id=task_id,
                error="Docker build failed",
            )

        assert result is True

        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["data"]["logs"] is None


class TestDeploySuccessWebhook:
    """Tests for deploy success webhook."""

    @pytest.mark.asyncio
    async def test_send_deploy_success(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending deploy success webhook.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        project_id = uuid4()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send_deploy_success(
                project_id=project_id,
                version="v1.2.3",
                environment="production",
            )

        assert result is True

        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["event"] == "deploy.success"
        assert payload["data"]["project_id"] == str(project_id)
        assert payload["data"]["version"] == "v1.2.3"
        assert payload["data"]["environment"] == "production"
        assert "deployed_at" in payload["data"]

    @pytest.mark.asyncio
    async def test_send_deploy_success_staging(
        self,
        dispatcher: WebhookDispatcher,
        mock_response_success: MagicMock,
    ) -> None:
        """Test sending deploy success webhook for staging environment.

        Args:
            dispatcher: The dispatcher fixture.
            mock_response_success: The mock success response fixture.
        """
        project_id = uuid4()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_success)

        with patch.object(dispatcher, "_get_client", return_value=mock_client):
            result = await dispatcher.send_deploy_success(
                project_id=project_id,
                version="v2.0.0-beta.1",
                environment="staging",
            )

        assert result is True

        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["data"]["environment"] == "staging"


class TestExponentialBackoff:
    """Tests for exponential backoff retry behavior."""

    @pytest.mark.asyncio
    async def test_backoff_delays(
        self,
        sample_endpoint: WebhookEndpoint,
        mock_response_failure: MagicMock,
    ) -> None:
        """Test that backoff uses exponential delays.

        Args:
            sample_endpoint: The sample endpoint fixture.
            mock_response_failure: The mock failure response fixture.
        """
        sample_endpoint.retry_count = 3
        dispatcher = WebhookDispatcher(endpoints=[sample_endpoint])

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response_failure)

        sleep_delays: list[int | float] = []

        async def mock_sleep(delay: int | float) -> None:
            sleep_delays.append(delay)

        with (
            patch.object(dispatcher, "_get_client", return_value=mock_client),
            patch("asyncio.sleep", mock_sleep),
        ):
            await dispatcher.send(WebhookEvent.TASK_COMPLETED, {"task_id": "123"})

        # Should have 3 sleeps (between attempts 1-2, 2-3, 3-4)
        assert len(sleep_delays) == 3
        assert sleep_delays[0] == 1  # 2^0
        assert sleep_delays[1] == 2  # 2^1
        assert sleep_delays[2] == 4  # 2^2
