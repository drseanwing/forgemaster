"""Webhook dispatcher for Forgemaster event notifications.

This module provides a webhook system for sending event notifications to
external services. It supports:
- Multiple webhook endpoints with per-endpoint configuration
- HMAC-SHA256 signature verification for payload authenticity
- Retry logic with exponential backoff on delivery failures
- Structured event payloads with timestamps

Supported event types:
- TASK_COMPLETED: Sent when a task finishes (success or failure)
- REVIEW_CYCLE: Sent after each review cycle completes
- BUILD_FAILURE: Sent when a build operation fails
- DEPLOY_SUCCESS: Sent when a deployment completes successfully

Example webhook configuration:
    endpoints = [
        WebhookEndpoint(
            url="https://api.example.com/hooks",
            secret="your-secret-key",
            retry_count=3,
        ),
    ]
    dispatcher = WebhookDispatcher(endpoints=endpoints)
    await dispatcher.send_task_completed(task_id, "done", {"output": "..."})
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from uuid import UUID

logger = structlog.get_logger(__name__)


class WebhookEvent(str, Enum):
    """Types of webhook events that can be dispatched.

    Attributes:
        TASK_COMPLETED: A task has completed (success or failure).
        REVIEW_CYCLE: A review cycle has completed.
        BUILD_FAILURE: A build operation has failed.
        DEPLOY_SUCCESS: A deployment has completed successfully.
    """

    TASK_COMPLETED = "task.completed"
    REVIEW_CYCLE = "review.cycle"
    BUILD_FAILURE = "build.failure"
    DEPLOY_SUCCESS = "deploy.success"


@dataclass
class WebhookEndpoint:
    """Configuration for a single webhook endpoint.

    Attributes:
        url: The URL to send webhook payloads to.
        secret: Optional secret for HMAC-SHA256 signature generation.
        enabled: Whether this endpoint is active.
        retry_count: Number of retry attempts on failure.
        timeout_seconds: Request timeout in seconds.
    """

    url: str
    secret: str | None = None
    enabled: bool = True
    retry_count: int = 3
    timeout_seconds: int = 30


@dataclass
class WebhookConfig:
    """Configuration for the webhook system.

    Attributes:
        endpoints: List of webhook endpoints to send events to.
        default_retry_count: Default retry count for endpoints.
        default_timeout_seconds: Default timeout for requests.
    """

    endpoints: list[WebhookEndpoint] = field(default_factory=list)
    default_retry_count: int = 3
    default_timeout_seconds: int = 30


class WebhookPayload(BaseModel):
    """Structured payload for webhook delivery.

    Attributes:
        event: The type of event being delivered.
        timestamp: ISO 8601 timestamp when the event occurred.
        data: Event-specific data payload.
    """

    event: WebhookEvent = Field(..., description="The webhook event type")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the event")
    data: dict[str, Any] = Field(default_factory=dict, description="Event-specific data")


class WebhookDispatcher:
    """Dispatches webhook events to configured endpoints.

    This class manages the delivery of webhook events to multiple endpoints
    with support for retries, signatures, and structured logging.

    Attributes:
        endpoints: List of configured webhook endpoints.
        logger: Structured logger for this dispatcher.
    """

    def __init__(self, endpoints: list[WebhookEndpoint] | None = None) -> None:
        """Initialize the webhook dispatcher.

        Args:
            endpoints: List of webhook endpoints to send events to.
                      If None, no webhooks will be sent.
        """
        self.endpoints = endpoints or []
        self.logger = logger.bind(component="webhook_dispatcher")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Returns:
            An httpx.AsyncClient instance for making requests.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _sign_payload(self, payload: str, secret: str) -> str:
        """Generate HMAC-SHA256 signature for a payload.

        Args:
            payload: The JSON payload string to sign.
            secret: The secret key for HMAC generation.

        Returns:
            Hexadecimal digest of the HMAC-SHA256 signature.
        """
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _build_headers(
        self,
        event: WebhookEvent,
        payload_str: str,
        secret: str | None,
    ) -> dict[str, str]:
        """Build HTTP headers for a webhook request.

        Args:
            event: The webhook event type.
            payload_str: The JSON payload string.
            secret: Optional secret for signature generation.

        Returns:
            Dictionary of HTTP headers.
        """
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        headers = {
            "Content-Type": "application/json",
            "X-Forgemaster-Event": event.value,
            "X-Forgemaster-Timestamp": timestamp,
        }

        if secret:
            signature = self._sign_payload(payload_str, secret)
            headers["X-Forgemaster-Signature"] = signature

        return headers

    async def _send_to_endpoint(
        self,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
        payload: WebhookPayload,
    ) -> bool:
        """Send a webhook payload to a single endpoint with retries.

        Args:
            endpoint: The endpoint configuration.
            event: The webhook event type.
            payload: The payload to send.

        Returns:
            True if delivery was successful, False otherwise.
        """
        if not endpoint.enabled:
            self.logger.debug(
                "Skipping disabled endpoint",
                url=endpoint.url,
                event_type=event.value,
            )
            return True

        payload_str = payload.model_dump_json()
        headers = self._build_headers(event, payload_str, endpoint.secret)

        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(endpoint.retry_count + 1):
            try:
                response = await client.post(
                    endpoint.url,
                    content=payload_str,
                    headers=headers,
                    timeout=endpoint.timeout_seconds,
                )

                if response.is_success:
                    self.logger.info(
                        "Webhook delivered successfully",
                        url=endpoint.url,
                        event_type=event.value,
                        status_code=response.status_code,
                        attempt=attempt + 1,
                    )
                    return True

                self.logger.warning(
                    "Webhook delivery returned non-success status",
                    url=endpoint.url,
                    event_type=event.value,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                )
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )

            except httpx.TimeoutException as e:
                last_error = e
                self.logger.warning(
                    "Webhook delivery timed out",
                    url=endpoint.url,
                    event_type=event.value,
                    attempt=attempt + 1,
                    error=str(e),
                )

            except httpx.RequestError as e:
                last_error = e
                self.logger.warning(
                    "Webhook delivery failed",
                    url=endpoint.url,
                    event_type=event.value,
                    attempt=attempt + 1,
                    error=str(e),
                )

            # Exponential backoff before retry (1s, 2s, 4s, ...)
            if attempt < endpoint.retry_count:
                backoff = 2**attempt
                await asyncio.sleep(backoff)

        self.logger.error(
            "Webhook delivery failed after all retries",
            url=endpoint.url,
            event_type=event.value,
            retry_count=endpoint.retry_count,
            error=str(last_error),
        )
        return False

    async def send(self, event: WebhookEvent, data: dict[str, Any]) -> bool:
        """Send a webhook event to all configured endpoints.

        Args:
            event: The type of event to send.
            data: Event-specific data payload.

        Returns:
            True if delivery to all enabled endpoints succeeded, False otherwise.
        """
        if not self.endpoints:
            self.logger.debug("No webhook endpoints configured", event_type=event.value)
            return True

        payload = WebhookPayload(
            event=event,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )

        self.logger.info(
            "Sending webhook event",
            event_type=event.value,
            endpoint_count=len(self.endpoints),
        )

        # Send to all endpoints concurrently
        tasks = [self._send_to_endpoint(endpoint, event, payload) for endpoint in self.endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check if all deliveries succeeded
        all_success = all(result is True for result in results if not isinstance(result, Exception))

        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "Unexpected error during webhook delivery",
                    url=self.endpoints[i].url,
                    error=str(result),
                )
                all_success = False

        return all_success

    async def send_task_completed(
        self,
        task_id: UUID,
        status: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Send a task completion webhook event.

        Args:
            task_id: The unique identifier of the completed task.
            status: The final status of the task (e.g., "done", "failed").
            result: Optional result data from the task execution.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        data = {
            "task_id": str(task_id),
            "status": status,
            "result": result or {},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        self.logger.info(
            "Sending task completed webhook",
            task_id=str(task_id),
            status=status,
        )

        return await self.send(WebhookEvent.TASK_COMPLETED, data)

    async def send_review_cycle(
        self,
        task_id: UUID,
        cycle_number: int,
        reviewer: str,
        approved: bool,
        findings: list[dict[str, Any]],
    ) -> bool:
        """Send a review cycle completion webhook event.

        Args:
            task_id: The unique identifier of the task under review.
            cycle_number: The review cycle number (1-indexed).
            reviewer: The name/type of the reviewer agent.
            approved: Whether the review was approved.
            findings: List of review findings/comments.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        data = {
            "task_id": str(task_id),
            "cycle_number": cycle_number,
            "reviewer": reviewer,
            "approved": approved,
            "findings": findings,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

        self.logger.info(
            "Sending review cycle webhook",
            task_id=str(task_id),
            cycle_number=cycle_number,
            reviewer=reviewer,
            approved=approved,
        )

        return await self.send(WebhookEvent.REVIEW_CYCLE, data)

    async def send_build_failure(
        self,
        task_id: UUID,
        error: str,
        logs: str | None = None,
    ) -> bool:
        """Send a build failure webhook event.

        Args:
            task_id: The unique identifier of the task that failed to build.
            error: The error message describing the failure.
            logs: Optional build logs for debugging.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        data = {
            "task_id": str(task_id),
            "error": error,
            "logs": logs,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

        self.logger.info(
            "Sending build failure webhook",
            task_id=str(task_id),
            error=error[:100] if error else None,  # Truncate for logging
        )

        return await self.send(WebhookEvent.BUILD_FAILURE, data)

    async def send_deploy_success(
        self,
        project_id: UUID,
        version: str,
        environment: str,
    ) -> bool:
        """Send a deploy success webhook event.

        Args:
            project_id: The unique identifier of the deployed project.
            version: The version/tag that was deployed.
            environment: The deployment environment (e.g., "staging", "production").

        Returns:
            True if delivery succeeded, False otherwise.
        """
        data = {
            "project_id": str(project_id),
            "version": version,
            "environment": environment,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }

        self.logger.info(
            "Sending deploy success webhook",
            project_id=str(project_id),
            version=version,
            environment=environment,
        )

        return await self.send(WebhookEvent.DEPLOY_SUCCESS, data)
