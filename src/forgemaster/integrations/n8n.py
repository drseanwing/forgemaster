"""n8n webhook client for external workflow integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

import httpx

from forgemaster.logging import get_logger

logger = get_logger(__name__)


class N8nEventType(str, Enum):
    """Types of events sent to n8n workflows."""

    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    BUILD_SUCCESS = "build_success"
    BUILD_FAILURE = "build_failure"
    DEPLOY_SUCCESS = "deploy_success"
    REVIEW_REQUESTED = "review_requested"
    REVIEW_COMPLETED = "review_completed"


@dataclass
class N8nConfig:
    """Configuration for n8n webhook integration."""

    webhook_url: str
    auth_header: str | None = None  # Optional Authorization header
    timeout_seconds: int = 30
    enabled: bool = True


@dataclass
class N8nPayload:
    """Standard payload format for n8n webhooks."""

    event_type: N8nEventType
    timestamp: datetime
    project_id: UUID | None = None
    task_id: UUID | None = None
    session_id: UUID | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "project_id": str(self.project_id) if self.project_id else None,
            "task_id": str(self.task_id) if self.task_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "data": self.data,
        }


class N8nClient:
    """Client for sending webhook notifications to n8n workflows."""

    def __init__(self, config: N8nConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(self, payload: N8nPayload) -> bool:
        """Send payload to n8n webhook.

        Returns True if successful, False otherwise.
        """
        if not self.config.enabled:
            self.logger.debug("n8n_disabled", event_type=payload.event_type.value)
            return True

        try:
            client = await self._get_client()
            headers = {"Content-Type": "application/json"}
            if self.config.auth_header:
                headers["Authorization"] = self.config.auth_header

            response = await client.post(
                self.config.webhook_url,
                json=payload.to_dict(),
                headers=headers,
            )

            if response.is_success:
                self.logger.info(
                    "n8n_webhook_sent",
                    event_type=payload.event_type.value,
                    status_code=response.status_code,
                )
                return True
            else:
                self.logger.warning(
                    "n8n_webhook_failed",
                    event_type=payload.event_type.value,
                    status_code=response.status_code,
                    response_text=response.text[:200],
                )
                return False

        except httpx.RequestError as e:
            self.logger.error(
                "n8n_webhook_error",
                event_type=payload.event_type.value,
                error=str(e),
            )
            return False

    async def notify_task_completed(
        self,
        project_id: UUID,
        task_id: UUID,
        title: str,
        status: str,
        duration_seconds: int | None = None,
    ) -> bool:
        """Notify n8n that a task completed."""
        payload = N8nPayload(
            event_type=N8nEventType.TASK_COMPLETED,
            timestamp=datetime.now(timezone.utc),
            project_id=project_id,
            task_id=task_id,
            data={
                "title": title,
                "status": status,
                "duration_seconds": duration_seconds,
            },
        )
        return await self.send(payload)

    async def notify_task_failed(
        self,
        project_id: UUID,
        task_id: UUID,
        title: str,
        error: str,
        retry_count: int,
    ) -> bool:
        """Notify n8n that a task failed."""
        payload = N8nPayload(
            event_type=N8nEventType.TASK_FAILED,
            timestamp=datetime.now(timezone.utc),
            project_id=project_id,
            task_id=task_id,
            data={
                "title": title,
                "error": error,
                "retry_count": retry_count,
            },
        )
        return await self.send(payload)

    async def notify_build_result(
        self,
        project_id: UUID,
        success: bool,
        version: str | None = None,
        error: str | None = None,
    ) -> bool:
        """Notify n8n of build result."""
        event_type = N8nEventType.BUILD_SUCCESS if success else N8nEventType.BUILD_FAILURE
        payload = N8nPayload(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            project_id=project_id,
            data={
                "success": success,
                "version": version,
                "error": error,
            },
        )
        return await self.send(payload)

    async def notify_deploy_success(
        self,
        project_id: UUID,
        version: str,
        environment: str,
    ) -> bool:
        """Notify n8n of successful deployment."""
        payload = N8nPayload(
            event_type=N8nEventType.DEPLOY_SUCCESS,
            timestamp=datetime.now(timezone.utc),
            project_id=project_id,
            data={
                "version": version,
                "environment": environment,
            },
        )
        return await self.send(payload)

    async def notify_review(
        self,
        project_id: UUID,
        task_id: UUID,
        reviewer: str,
        approved: bool | None = None,
        findings: list[str] | None = None,
    ) -> bool:
        """Notify n8n of review status."""
        event_type = (
            N8nEventType.REVIEW_COMPLETED
            if approved is not None
            else N8nEventType.REVIEW_REQUESTED
        )
        payload = N8nPayload(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            project_id=project_id,
            task_id=task_id,
            data={
                "reviewer": reviewer,
                "approved": approved,
                "findings": findings or [],
            },
        )
        return await self.send(payload)
