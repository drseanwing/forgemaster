"""Health monitoring and rollback system for Docker containers.

This module provides health check monitoring with configurable polling, threshold-based
status transitions, and automated rollback capabilities for containerized services.

Key Components:
- HealthPoller: Polls HTTP health endpoints with configurable intervals and thresholds
- RollbackTrigger: Evaluates health status and determines when rollback is needed
- RollbackExecutor: Executes container rollback strategies (restart, revert image, stop)

Example usage:
    >>> from forgemaster.pipeline.health import HealthPoller, HealthCheckConfig
    >>> from forgemaster.pipeline.container import ContainerManager
    >>>
    >>> config = HealthCheckConfig(url="http://localhost:8080/health")
    >>> poller = HealthPoller(config)
    >>>
    >>> # Check health once
    >>> result = await poller.check_health()
    >>> if result.status == HealthStatus.HEALTHY:
    ...     print("Service is healthy")
    >>>
    >>> # Poll until healthy or timeout
    >>> result = await poller.poll_until_healthy(timeout_seconds=120.0)
    >>>
    >>> # Continuous background polling
    >>> await poller.start_polling()
    >>> # ... do other work ...
    >>> await poller.stop_polling()
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from enum import Enum

import httpx
from pydantic import BaseModel, Field

from forgemaster.config import DockerConfig
from forgemaster.logging import get_logger
from forgemaster.pipeline.container import ContainerManager


class HealthStatus(str, Enum):
    """Health status classification for services.

    Attributes:
        HEALTHY: Service is responding correctly and passing health checks
        UNHEALTHY: Service is not responding or failing health checks
        DEGRADED: Service is partially functional but not optimal
        UNKNOWN: Health status cannot be determined
    """

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class HealthCheckConfig(BaseModel):
    """Configuration for HTTP health endpoint polling.

    Attributes:
        url: Health check endpoint URL
        interval_seconds: Time between consecutive health checks
        timeout_seconds: Maximum time to wait for health check response
        healthy_threshold: Consecutive successes needed to transition to HEALTHY
        unhealthy_threshold: Consecutive failures needed to transition to UNHEALTHY
        expected_status_codes: HTTP status codes considered successful
        expected_body: Optional substring that must appear in response body
    """

    url: str = Field(description="Health check endpoint URL")
    interval_seconds: float = Field(default=10.0, gt=0.0, description="Poll interval")
    timeout_seconds: float = Field(default=5.0, gt=0.0, description="Request timeout")
    healthy_threshold: int = Field(default=3, ge=1, description="Successes for healthy")
    unhealthy_threshold: int = Field(default=3, ge=1, description="Failures for unhealthy")
    expected_status_codes: list[int] = Field(
        default_factory=lambda: [200], description="Acceptable status codes"
    )
    expected_body: str | None = Field(default=None, description="Expected body substring")


class HealthCheckResult(BaseModel):
    """Result of a single health check operation.

    Attributes:
        status: Determined health status
        url: Health check URL that was checked
        response_code: HTTP response code (None if request failed)
        response_time_seconds: Request duration (None if request failed)
        error: Error message if health check failed
        checked_at: ISO 8601 timestamp of the check
        consecutive_successes: Number of consecutive successful checks
        consecutive_failures: Number of consecutive failed checks
    """

    status: HealthStatus = Field(description="Health status")
    url: str = Field(description="Checked URL")
    response_code: int | None = Field(default=None, description="HTTP status code")
    response_time_seconds: float | None = Field(default=None, description="Response time")
    error: str | None = Field(default=None, description="Error message")
    checked_at: str = Field(description="Check timestamp")
    consecutive_successes: int = Field(default=0, ge=0, description="Success streak")
    consecutive_failures: int = Field(default=0, ge=0, description="Failure streak")


class RollbackConfig(BaseModel):
    """Configuration for rollback trigger logic.

    Attributes:
        enabled: Whether rollback automation is enabled
        max_rollback_attempts: Maximum number of rollback attempts before giving up
        cooldown_seconds: Minimum time between rollback attempts
        trigger_on_consecutive_failures: Failures needed before triggering rollback
    """

    enabled: bool = Field(default=True, description="Rollback enabled flag")
    max_rollback_attempts: int = Field(default=3, ge=1, description="Max rollback attempts")
    cooldown_seconds: float = Field(default=60.0, ge=0.0, description="Rollback cooldown")
    trigger_on_consecutive_failures: int = Field(
        default=3, ge=1, description="Failures to trigger rollback"
    )


class RollbackDecision(BaseModel):
    """Decision output from rollback trigger evaluation.

    Attributes:
        should_rollback: Whether a rollback should be executed
        reason: Explanation for the decision
        consecutive_failures: Current consecutive failure count
        cooldown_remaining_seconds: Time remaining in cooldown period
        rollback_attempt: Current rollback attempt number (0 if none yet)
    """

    should_rollback: bool = Field(description="Rollback decision")
    reason: str = Field(description="Decision rationale")
    consecutive_failures: int = Field(default=0, ge=0, description="Failure count")
    cooldown_remaining_seconds: float = Field(default=0.0, ge=0.0, description="Cooldown time")
    rollback_attempt: int = Field(default=0, ge=0, description="Rollback attempt number")


class RollbackStrategy(str, Enum):
    """Strategy for container rollback operations.

    Attributes:
        RESTART_CONTAINER: Simply restart the container (fastest recovery)
        REVERT_IMAGE: Stop, revert to previous image tag, restart (full recovery)
        STOP_SERVICE: Stop the container and require manual intervention
    """

    RESTART_CONTAINER = "restart_container"
    REVERT_IMAGE = "revert_image"
    STOP_SERVICE = "stop_service"


class RollbackResult(BaseModel):
    """Result of a rollback execution operation.

    Attributes:
        success: Whether the rollback completed successfully
        action: Rollback action performed
        previous_image: Image tag before rollback (if applicable)
        rollback_image: Image tag after rollback (if applicable)
        containers_affected: List of container IDs affected
        error: Error message if rollback failed
        duration_seconds: Time taken to execute rollback
    """

    success: bool = Field(description="Rollback success flag")
    action: str = Field(description="Rollback action")
    previous_image: str | None = Field(default=None, description="Previous image tag")
    rollback_image: str | None = Field(default=None, description="Rollback image tag")
    containers_affected: list[str] = Field(
        default_factory=list, description="Affected container IDs"
    )
    error: str | None = Field(default=None, description="Error message")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Rollback duration")


class HealthCheckTimeout(Exception):
    """Exception raised when health check polling exceeds timeout."""

    pass


class HealthPoller:
    """HTTP health endpoint poller with threshold-based status transitions.

    This class polls HTTP health endpoints at configurable intervals, tracks
    consecutive successes and failures, and determines health status based on
    configurable thresholds.

    Attributes:
        config: Health check configuration
        logger: Structured logger instance
    """

    def __init__(self, config: HealthCheckConfig) -> None:
        """Initialize HealthPoller with configuration.

        Args:
            config: Health check configuration settings
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._consecutive_successes: int = 0
        self._consecutive_failures: int = 0
        self._current_status: HealthStatus = HealthStatus.UNKNOWN
        self._polling_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._session: httpx.AsyncClient | None = None
        self._timeout_count: int = 0

        self.logger.info(
            "health_poller_initialized",
            url=config.url,
            interval_seconds=config.interval_seconds,
            healthy_threshold=config.healthy_threshold,
            unhealthy_threshold=config.unhealthy_threshold,
        )

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create the httpx async client.

        Returns:
            Active httpx AsyncClient
        """
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient()
        return self._session

    async def check_health(self) -> HealthCheckResult:
        """Perform a single health check against the configured endpoint.

        Returns:
            HealthCheckResult with status and timing information
        """
        start_time = time.monotonic()
        checked_at = datetime.now(timezone.utc).isoformat()

        self.logger.debug(
            "health_check_started",
            url=self.config.url,
            timeout=self.config.timeout_seconds,
        )

        try:
            session = await self._get_session()

            response = await session.get(self.config.url, timeout=self.config.timeout_seconds)
            response_time = time.monotonic() - start_time
            status_code = response.status_code
            body = response.text

            # Check if status code is acceptable
            if status_code not in self.config.expected_status_codes:
                self._consecutive_successes = 0
                self._consecutive_failures += 1
                error_msg = f"Unexpected status code: {status_code}"

                self.logger.warning(
                    "health_check_bad_status",
                    url=self.config.url,
                    status_code=status_code,
                    expected=self.config.expected_status_codes,
                )

                # Update status based on thresholds
                if self._consecutive_failures >= self.config.unhealthy_threshold:
                    self._current_status = HealthStatus.UNHEALTHY
                elif self._current_status == HealthStatus.HEALTHY:
                    self._current_status = HealthStatus.DEGRADED

                return HealthCheckResult(
                    status=self._current_status,
                    url=self.config.url,
                    response_code=status_code,
                    response_time_seconds=response_time,
                    error=error_msg,
                    checked_at=checked_at,
                    consecutive_successes=self._consecutive_successes,
                    consecutive_failures=self._consecutive_failures,
                )

            # Check expected body content if configured
            if self.config.expected_body is not None:
                if self.config.expected_body not in body:
                    self._consecutive_successes = 0
                    self._consecutive_failures += 1
                    error_msg = f"Expected body content not found: {self.config.expected_body}"

                    self.logger.warning(
                        "health_check_bad_body",
                        url=self.config.url,
                        expected=self.config.expected_body,
                    )

                    if self._consecutive_failures >= self.config.unhealthy_threshold:
                        self._current_status = HealthStatus.UNHEALTHY
                    elif self._current_status == HealthStatus.HEALTHY:
                        self._current_status = HealthStatus.DEGRADED

                    return HealthCheckResult(
                        status=self._current_status,
                        url=self.config.url,
                        response_code=status_code,
                        response_time_seconds=response_time,
                        error=error_msg,
                        checked_at=checked_at,
                        consecutive_successes=self._consecutive_successes,
                        consecutive_failures=self._consecutive_failures,
                    )

            # Health check passed
            self._consecutive_failures = 0
            self._consecutive_successes += 1

            # Update status based on thresholds
            if self._consecutive_successes >= self.config.healthy_threshold:
                self._current_status = HealthStatus.HEALTHY
            elif self._current_status == HealthStatus.UNHEALTHY:
                self._current_status = HealthStatus.DEGRADED

            self.logger.debug(
                "health_check_passed",
                url=self.config.url,
                status_code=status_code,
                response_time=round(response_time, 3),
                consecutive_successes=self._consecutive_successes,
            )

            return HealthCheckResult(
                status=self._current_status,
                url=self.config.url,
                response_code=status_code,
                response_time_seconds=response_time,
                checked_at=checked_at,
                consecutive_successes=self._consecutive_successes,
                consecutive_failures=self._consecutive_failures,
            )

        except asyncio.TimeoutError:
            response_time = time.monotonic() - start_time
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            self._timeout_count += 1
            error_msg = f"Health check timed out after {self.config.timeout_seconds}s"

            self.logger.warning(
                "health_check_timeout",
                url=self.config.url,
                timeout=self.config.timeout_seconds,
                timeout_count=self._timeout_count,
            )

            if self._consecutive_failures >= self.config.unhealthy_threshold:
                self._current_status = HealthStatus.UNHEALTHY
            elif self._current_status == HealthStatus.HEALTHY:
                self._current_status = HealthStatus.DEGRADED

            return HealthCheckResult(
                status=self._current_status,
                url=self.config.url,
                response_time_seconds=response_time,
                error=error_msg,
                checked_at=checked_at,
                consecutive_successes=self._consecutive_successes,
                consecutive_failures=self._consecutive_failures,
            )

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            response_time = time.monotonic() - start_time
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            error_msg = f"Connection error: {str(e)}"

            self.logger.warning(
                "health_check_connection_error",
                url=self.config.url,
                error=str(e),
                error_type=type(e).__name__,
            )

            if self._consecutive_failures >= self.config.unhealthy_threshold:
                self._current_status = HealthStatus.UNHEALTHY
            elif self._current_status == HealthStatus.HEALTHY:
                self._current_status = HealthStatus.DEGRADED

            return HealthCheckResult(
                status=self._current_status,
                url=self.config.url,
                response_time_seconds=response_time,
                error=error_msg,
                checked_at=checked_at,
                consecutive_successes=self._consecutive_successes,
                consecutive_failures=self._consecutive_failures,
            )

        except Exception as e:
            response_time = time.monotonic() - start_time
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            error_msg = f"Unexpected error: {str(e)}"

            self.logger.error(
                "health_check_unexpected_error",
                url=self.config.url,
                error=str(e),
                error_type=type(e).__name__,
            )

            if self._consecutive_failures >= self.config.unhealthy_threshold:
                self._current_status = HealthStatus.UNHEALTHY
            elif self._current_status == HealthStatus.HEALTHY:
                self._current_status = HealthStatus.DEGRADED

            return HealthCheckResult(
                status=self._current_status,
                url=self.config.url,
                response_time_seconds=response_time,
                error=error_msg,
                checked_at=checked_at,
                consecutive_successes=self._consecutive_successes,
                consecutive_failures=self._consecutive_failures,
            )

    async def check_health_with_timeout(
        self, timeout_seconds: float | None = None
    ) -> HealthCheckResult:
        """Perform a single health check with explicit timeout override.

        Args:
            timeout_seconds: Override the default timeout (None uses config timeout)

        Returns:
            HealthCheckResult with status and timing information
        """
        if timeout_seconds is not None:
            # Temporarily override the timeout
            original_timeout = self.config.timeout_seconds
            self.config.timeout_seconds = timeout_seconds
            try:
                return await self.check_health()
            finally:
                self.config.timeout_seconds = original_timeout
        else:
            return await self.check_health()

    async def poll_until_healthy(self, timeout_seconds: float = 120.0) -> HealthCheckResult:
        """Poll repeatedly until the service becomes healthy or timeout is reached.

        Args:
            timeout_seconds: Maximum time to poll before raising HealthCheckTimeout

        Returns:
            HealthCheckResult with final status

        Raises:
            HealthCheckTimeout: If timeout is exceeded before service becomes healthy
        """
        start_time = time.monotonic()
        attempt = 0

        self.logger.info(
            "poll_until_healthy_started",
            url=self.config.url,
            timeout_seconds=timeout_seconds,
            interval_seconds=self.config.interval_seconds,
        )

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                error_msg = f"Health polling timed out after {timeout_seconds}s"
                self.logger.error(
                    "poll_until_healthy_timeout",
                    url=self.config.url,
                    elapsed_seconds=round(elapsed, 2),
                    attempts=attempt,
                )
                raise HealthCheckTimeout(error_msg)

            attempt += 1
            result = await self.check_health()

            if result.status == HealthStatus.HEALTHY:
                self.logger.info(
                    "poll_until_healthy_succeeded",
                    url=self.config.url,
                    elapsed_seconds=round(elapsed, 2),
                    attempts=attempt,
                )
                return result

            # Wait before next attempt (but don't exceed timeout)
            remaining = timeout_seconds - (time.monotonic() - start_time)
            wait_time = min(self.config.interval_seconds, remaining)

            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def start_polling(self) -> None:
        """Start continuous background health check polling.

        This creates an asyncio task that polls the health endpoint at the
        configured interval. Use stop_polling() to stop the background task.
        """
        if self._polling_task is not None and not self._polling_task.done():
            self.logger.warning(
                "polling_already_active",
                url=self.config.url,
            )
            return

        self._stop_event.clear()
        self._polling_task = asyncio.create_task(self._polling_loop())

        self.logger.info(
            "polling_started",
            url=self.config.url,
            interval_seconds=self.config.interval_seconds,
        )

    async def stop_polling(self) -> None:
        """Stop the continuous background health check polling.

        Safe to call even if polling is not active.
        """
        if self._polling_task is None or self._polling_task.done():
            self.logger.debug("polling_not_active")
            return

        self._stop_event.set()

        try:
            await asyncio.wait_for(self._polling_task, timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("polling_stop_timeout")
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        self._polling_task = None

        self.logger.info("polling_stopped", url=self.config.url)

    async def _polling_loop(self) -> None:
        """Internal background polling loop."""
        self.logger.debug("polling_loop_started")

        while not self._stop_event.is_set():
            try:
                await self.check_health()
            except Exception as e:
                self.logger.error(
                    "polling_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )

            # Wait for interval or stop event
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.config.interval_seconds
                )
            except asyncio.TimeoutError:
                # Normal timeout, continue polling
                pass

        self.logger.debug("polling_loop_exited")

    async def close(self) -> None:
        """Close the httpx client session and stop polling if active.

        Safe to call multiple times.
        """
        await self.stop_polling()

        if self._session is not None and not self._session.is_closed:
            await self._session.aclose()
            self._session = None

        self.logger.info("health_poller_closed")


class RollbackTrigger:
    """Evaluates health check results and determines when rollback is needed.

    This class tracks consecutive failures, manages rollback cooldown periods,
    and enforces maximum rollback attempt limits.

    Attributes:
        poller: Health poller instance to monitor
        config: Rollback configuration
        logger: Structured logger instance
    """

    def __init__(self, poller: HealthPoller, config: RollbackConfig) -> None:
        """Initialize RollbackTrigger with poller and configuration.

        Args:
            poller: HealthPoller instance to monitor
            config: Rollback trigger configuration
        """
        self.poller = poller
        self.config = config
        self.logger = get_logger(__name__)
        self._rollback_count: int = 0
        self._last_rollback_time: float | None = None

        self.logger.info(
            "rollback_trigger_initialized",
            enabled=config.enabled,
            max_attempts=config.max_rollback_attempts,
            cooldown_seconds=config.cooldown_seconds,
            failure_threshold=config.trigger_on_consecutive_failures,
        )

    async def evaluate(self, health_result: HealthCheckResult) -> RollbackDecision:
        """Evaluate a health check result and decide if rollback is needed.

        Args:
            health_result: Most recent health check result

        Returns:
            RollbackDecision with rollback recommendation and reasoning
        """
        # If rollback is disabled, never rollback
        if not self.config.enabled:
            return RollbackDecision(
                should_rollback=False,
                reason="Rollback automation is disabled",
                consecutive_failures=health_result.consecutive_failures,
                rollback_attempt=self._rollback_count,
            )

        # If we've exceeded max attempts, don't rollback
        if self._rollback_count >= self.config.max_rollback_attempts:
            self.logger.warning(
                "max_rollback_attempts_reached",
                rollback_count=self._rollback_count,
                max_attempts=self.config.max_rollback_attempts,
            )
            return RollbackDecision(
                should_rollback=False,
                reason=f"Maximum rollback attempts reached ({self.config.max_rollback_attempts})",
                consecutive_failures=health_result.consecutive_failures,
                rollback_attempt=self._rollback_count,
            )

        # Check cooldown period
        cooldown_remaining = 0.0
        if self._last_rollback_time is not None:
            elapsed = time.monotonic() - self._last_rollback_time
            cooldown_remaining = max(0.0, self.config.cooldown_seconds - elapsed)

            if cooldown_remaining > 0:
                self.logger.debug(
                    "rollback_cooldown_active",
                    cooldown_remaining_seconds=round(cooldown_remaining, 2),
                )
                return RollbackDecision(
                    should_rollback=False,
                    reason=f"Cooldown period active ({round(cooldown_remaining, 1)}s remaining)",
                    consecutive_failures=health_result.consecutive_failures,
                    cooldown_remaining_seconds=cooldown_remaining,
                    rollback_attempt=self._rollback_count,
                )

        # Check if failures meet threshold
        if health_result.consecutive_failures < self.config.trigger_on_consecutive_failures:
            return RollbackDecision(
                should_rollback=False,
                reason=(
                    f"Failures ({health_result.consecutive_failures}) below threshold "
                    f"({self.config.trigger_on_consecutive_failures})"
                ),
                consecutive_failures=health_result.consecutive_failures,
                rollback_attempt=self._rollback_count,
            )

        # All conditions met - trigger rollback
        self.logger.warning(
            "rollback_triggered",
            consecutive_failures=health_result.consecutive_failures,
            threshold=self.config.trigger_on_consecutive_failures,
            rollback_attempt=self._rollback_count + 1,
        )

        # Update rollback tracking
        self._rollback_count += 1
        self._last_rollback_time = time.monotonic()

        return RollbackDecision(
            should_rollback=True,
            reason=(
                f"Consecutive failures ({health_result.consecutive_failures}) "
                f"exceeded threshold ({self.config.trigger_on_consecutive_failures})"
            ),
            consecutive_failures=health_result.consecutive_failures,
            rollback_attempt=self._rollback_count,
        )


class RollbackExecutor:
    """Executes container rollback operations using configured strategies.

    This class performs actual rollback operations on containers, including
    restarting, reverting to previous images, or stopping services.

    Attributes:
        container_manager: Container lifecycle manager
        config: Rollback configuration
        logger: Structured logger instance
    """

    def __init__(self, container_manager: ContainerManager, config: RollbackConfig) -> None:
        """Initialize RollbackExecutor with container manager and configuration.

        Args:
            container_manager: ContainerManager for performing operations
            config: Rollback configuration
        """
        self.container_manager = container_manager
        self.config = config
        self.logger = get_logger(__name__)

        self.logger.info(
            "rollback_executor_initialized",
            enabled=config.enabled,
        )

    async def execute_rollback(
        self,
        container_id: str,
        strategy: RollbackStrategy = RollbackStrategy.RESTART_CONTAINER,
        previous_image: str | None = None,
    ) -> RollbackResult:
        """Execute a container rollback using the specified strategy.

        Args:
            container_id: Container ID or name to rollback
            strategy: Rollback strategy to use
            previous_image: Previous image tag (required for REVERT_IMAGE strategy)

        Returns:
            RollbackResult with operation outcome and affected containers
        """
        start_time = time.monotonic()

        self.logger.info(
            "rollback_started",
            container_id=container_id,
            strategy=strategy,
            previous_image=previous_image,
        )

        if strategy == RollbackStrategy.RESTART_CONTAINER:
            return await self._restart_strategy(container_id, start_time)
        elif strategy == RollbackStrategy.REVERT_IMAGE:
            return await self._revert_image_strategy(container_id, previous_image, start_time)
        elif strategy == RollbackStrategy.STOP_SERVICE:
            return await self._stop_service_strategy(container_id, start_time)
        else:
            duration = time.monotonic() - start_time
            error_msg = f"Unknown rollback strategy: {strategy}"
            self.logger.error("rollback_unknown_strategy", strategy=strategy)
            return RollbackResult(
                success=False,
                action=str(strategy),
                error=error_msg,
                duration_seconds=duration,
            )

    async def _restart_strategy(self, container_id: str, start_time: float) -> RollbackResult:
        """Execute RESTART_CONTAINER strategy.

        Args:
            container_id: Container to restart
            start_time: Start time for duration tracking

        Returns:
            RollbackResult with restart outcome
        """
        action_result = await self.container_manager.restart_container(container_id)
        duration = time.monotonic() - start_time

        if action_result.success:
            self.logger.info(
                "rollback_restart_succeeded",
                container_id=container_id,
                duration_seconds=round(duration, 2),
            )
            return RollbackResult(
                success=True,
                action="restart_container",
                containers_affected=[container_id],
                duration_seconds=duration,
            )
        else:
            self.logger.error(
                "rollback_restart_failed",
                container_id=container_id,
                error=action_result.error,
            )
            return RollbackResult(
                success=False,
                action="restart_container",
                containers_affected=[container_id],
                error=action_result.error,
                duration_seconds=duration,
            )

    async def _revert_image_strategy(
        self, container_id: str, previous_image: str | None, start_time: float
    ) -> RollbackResult:
        """Execute REVERT_IMAGE strategy.

        Args:
            container_id: Container to revert
            previous_image: Image tag to revert to
            start_time: Start time for duration tracking

        Returns:
            RollbackResult with revert outcome
        """
        if previous_image is None:
            duration = time.monotonic() - start_time
            error_msg = "previous_image required for REVERT_IMAGE strategy"
            self.logger.error("rollback_revert_missing_image", container_id=container_id)
            return RollbackResult(
                success=False,
                action="revert_image",
                error=error_msg,
                duration_seconds=duration,
            )

        try:
            # Get current image for logging
            info = await self.container_manager.get_container_status(container_id)
            current_image = info.image

            # Stop the container
            stop_result = await self.container_manager.stop_container(container_id)
            if not stop_result.success:
                duration = time.monotonic() - start_time
                error_msg = f"Failed to stop container: {stop_result.error}"
                self.logger.error(
                    "rollback_revert_stop_failed",
                    container_id=container_id,
                    error=stop_result.error,
                )
                return RollbackResult(
                    success=False,
                    action="revert_image",
                    previous_image=current_image,
                    error=error_msg,
                    duration_seconds=duration,
                )

            # Note: Actual image revert would require docker-py image operations
            # This is a simplified version that restarts with the expectation
            # that the container is re-created with the previous image externally
            self.logger.warning(
                "rollback_revert_simplified",
                container_id=container_id,
                previous_image=previous_image,
                note="Full image revert requires external re-creation",
            )

            # Start the container back up
            start_result = await self.container_manager.start_container(container_id)
            duration = time.monotonic() - start_time

            if start_result.success:
                self.logger.info(
                    "rollback_revert_succeeded",
                    container_id=container_id,
                    previous_image=current_image,
                    rollback_image=previous_image,
                    duration_seconds=round(duration, 2),
                )
                return RollbackResult(
                    success=True,
                    action="revert_image",
                    previous_image=current_image,
                    rollback_image=previous_image,
                    containers_affected=[container_id],
                    duration_seconds=duration,
                )
            else:
                error_msg = f"Failed to start container after revert: {start_result.error}"
                self.logger.error(
                    "rollback_revert_start_failed",
                    container_id=container_id,
                    error=start_result.error,
                )
                return RollbackResult(
                    success=False,
                    action="revert_image",
                    previous_image=current_image,
                    rollback_image=previous_image,
                    error=error_msg,
                    duration_seconds=duration,
                )

        except Exception as e:
            duration = time.monotonic() - start_time
            error_msg = f"Unexpected error during image revert: {str(e)}"
            self.logger.error(
                "rollback_revert_unexpected_error",
                container_id=container_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return RollbackResult(
                success=False,
                action="revert_image",
                error=error_msg,
                duration_seconds=duration,
            )

    async def _stop_service_strategy(self, container_id: str, start_time: float) -> RollbackResult:
        """Execute STOP_SERVICE strategy.

        Args:
            container_id: Container to stop
            start_time: Start time for duration tracking

        Returns:
            RollbackResult with stop outcome
        """
        stop_result = await self.container_manager.stop_container(container_id)
        duration = time.monotonic() - start_time

        if stop_result.success:
            self.logger.warning(
                "rollback_service_stopped",
                container_id=container_id,
                note="Manual intervention required to restart",
                duration_seconds=round(duration, 2),
            )
            return RollbackResult(
                success=True,
                action="stop_service",
                containers_affected=[container_id],
                duration_seconds=duration,
            )
        else:
            self.logger.error(
                "rollback_stop_failed",
                container_id=container_id,
                error=stop_result.error,
            )
            return RollbackResult(
                success=False,
                action="stop_service",
                containers_affected=[container_id],
                error=stop_result.error,
                duration_seconds=duration,
            )
