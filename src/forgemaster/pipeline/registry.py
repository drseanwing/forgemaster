"""Docker registry operations with authentication and retry logic for FORGEMASTER.

This module provides high-level async operations for pushing Docker images to registries
with comprehensive authentication support (token-based and password-based), exponential
backoff retry logic, and structured logging. Designed to work with ghcr.io, Docker Hub,
and any OCI-compliant registry.

Example usage:
    >>> from forgemaster.config import DockerConfig
    >>> from forgemaster.pipeline.registry import RegistryClient, RegistryAuth
    >>>
    >>> config = DockerConfig(registry="ghcr.io")
    >>> client = RegistryClient(config)
    >>>
    >>> auth = RegistryAuth(registry="ghcr.io", username="user", password="token")
    >>> await client.authenticate(auth)
    >>>
    >>> result = await client.push_with_retry("ghcr.io/org/repo:v1.0.0")
    >>> if result.success:
    ...     print(f"Pushed with digest: {result.digest}")
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from enum import Enum

from docker.errors import APIError, DockerException  # type: ignore[import-not-found]
from pydantic import BaseModel, Field

import docker
from forgemaster.config import DockerConfig
from forgemaster.logging import get_logger

# Regex pattern for extracting digest from push output
_DIGEST_PATTERN = re.compile(r"digest:\s*(sha256:[a-f0-9]{64})")


class PushStatus(str, Enum):
    """Status of a Docker image push operation.

    Attributes:
        PENDING: Push has been queued but not started
        PUSHING: Push is currently in progress
        SUCCEEDED: Push completed successfully
        FAILED: Push encountered an error
        RETRYING: Push is being retried after failure
    """

    PENDING = "pending"
    PUSHING = "pushing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"


class RegistryAuth(BaseModel):
    """Authentication credentials for a Docker registry.

    Supports both password-based and token-based authentication.

    Attributes:
        registry: Registry URL (e.g., 'ghcr.io', 'docker.io')
        username: Registry username or organization
        password: Registry password or access token
        token: Explicit authentication token (optional, overrides password)
    """

    registry: str = Field(description="Registry URL")
    username: str = Field(description="Registry username")
    password: str = Field(description="Registry password or token")
    token: str | None = Field(default=None, description="Explicit auth token")


class RetryConfig(BaseModel):
    """Configuration for retry logic with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (0 means no retries)
        initial_delay_seconds: Initial delay before first retry
        max_delay_seconds: Maximum delay between retries (caps exponential growth)
        backoff_multiplier: Multiplier for exponential backoff (delay *= multiplier^attempt)
    """

    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    initial_delay_seconds: float = Field(
        default=5.0, ge=0.0, le=300.0, description="Initial retry delay"
    )
    max_delay_seconds: float = Field(
        default=60.0, ge=0.0, le=600.0, description="Max retry delay"
    )
    backoff_multiplier: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff multiplier"
    )


class PushResult(BaseModel):
    """Result of a Docker image push operation.

    Attributes:
        success: Whether the push completed successfully
        image_tag: Full image tag that was pushed
        digest: Image digest (sha256 hash) from the registry
        push_log: Collected push log lines
        duration_seconds: Total push time in seconds
        error: Error message if push failed, None otherwise
        status: Current push status
        retry_attempts: Number of retry attempts made
    """

    success: bool = Field(default=False, description="Push success flag")
    image_tag: str = Field(description="Image tag pushed")
    digest: str | None = Field(default=None, description="Image digest from registry")
    push_log: list[str] = Field(default_factory=list, description="Push log lines")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Push duration")
    error: str | None = Field(default=None, description="Error message if failed")
    status: PushStatus = Field(default=PushStatus.PENDING, description="Push status")
    retry_attempts: int = Field(default=0, ge=0, description="Number of retry attempts")


class RegistryClient:
    """High-level async Docker registry client using docker-py.

    This class wraps the docker-py SDK with async compatibility, authentication
    management, retry logic with exponential backoff, and structured logging.

    Attributes:
        config: Docker configuration from ForgemasterConfig
        logger: Structured logger instance
    """

    def __init__(self, config: DockerConfig, auth: RegistryAuth | None = None) -> None:
        """Initialize RegistryClient with configuration.

        Args:
            config: Docker configuration settings
            auth: Optional default authentication credentials

        The Docker client connection is deferred until first use to allow
        graceful handling of connection failures.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._client: docker.DockerClient | None = None  # type: ignore[name-defined]
        self._auth: RegistryAuth | None = auth

        self.logger.info(
            "registry_client_initialized",
            registry=config.registry,
            has_auth=auth is not None,
        )

    def _get_client(self) -> docker.DockerClient:  # type: ignore[name-defined]
        """Get or create the Docker client connection.

        Returns:
            Active Docker client instance

        Raises:
            DockerException: If unable to connect to Docker daemon
        """
        if self._client is None:
            try:
                # Determine Docker socket based on rootless mode
                docker_host = os.environ.get("DOCKER_HOST")
                if docker_host:
                    self._client = docker.DockerClient(base_url=docker_host)  # type: ignore[attr-defined]
                elif self.config.rootless:
                    # Try rootless socket paths (Linux/Unix only)
                    try:
                        import pwd
                        uid = pwd.getpwuid(os.getuid()).pw_uid  # type: ignore[attr-defined]
                        xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")
                    except (ImportError, KeyError, AttributeError):
                        # Windows or missing pwd module - use default
                        xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000")
                    rootless_socket = f"unix://{xdg_runtime}/docker.sock"
                    try:
                        self._client = docker.DockerClient(base_url=rootless_socket)  # type: ignore[attr-defined]
                    except DockerException:
                        # Fall back to default
                        self._client = docker.DockerClient.from_env()  # type: ignore[attr-defined]
                else:
                    self._client = docker.DockerClient.from_env()  # type: ignore[attr-defined]

                self.logger.info(
                    "docker_client_connected",
                    rootless=self.config.rootless,
                )
            except DockerException as e:
                self.logger.error(
                    "docker_client_connection_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    rootless=self.config.rootless,
                )
                raise

        return self._client

    def _get_auth_from_env(self) -> RegistryAuth | None:
        """Get authentication credentials from environment variables.

        Checks for DOCKER_REGISTRY_USERNAME, DOCKER_REGISTRY_PASSWORD, and
        DOCKER_REGISTRY_TOKEN environment variables.

        Returns:
            RegistryAuth if credentials found, None otherwise
        """
        username = os.environ.get("DOCKER_REGISTRY_USERNAME")
        password = os.environ.get("DOCKER_REGISTRY_PASSWORD")
        token = os.environ.get("DOCKER_REGISTRY_TOKEN")

        if username and (password or token):
            return RegistryAuth(
                registry=self.config.registry,
                username=username,
                password=password or token or "",
                token=token,
            )

        return None

    async def authenticate(self, auth: RegistryAuth | None = None) -> bool:
        """Authenticate with the Docker registry.

        Authentication priority:
        1. Explicit auth parameter
        2. Default auth from constructor
        3. Auth from environment variables

        Args:
            auth: Authentication credentials to use

        Returns:
            True if authentication succeeded, False otherwise
        """
        # Determine which auth to use
        effective_auth = auth or self._auth or self._get_auth_from_env()

        if effective_auth is None:
            self.logger.warning(
                "no_registry_auth_provided",
                registry=self.config.registry,
            )
            return False

        try:
            client = await asyncio.to_thread(self._get_client)

            # Use token if provided, otherwise use password
            auth_password = effective_auth.token or effective_auth.password

            # Authenticate with registry
            login_result: dict[str, str] = await asyncio.to_thread(
                client.login,
                username=effective_auth.username,
                password=auth_password,
                registry=effective_auth.registry,
            )

            success = login_result.get("Status") == "Login Succeeded"

            if success:
                # Store auth for reuse
                self._auth = effective_auth
                self.logger.info(
                    "registry_authentication_succeeded",
                    registry=effective_auth.registry,
                    username=effective_auth.username,
                    using_token=effective_auth.token is not None,
                )
            else:
                self.logger.error(
                    "registry_authentication_failed",
                    registry=effective_auth.registry,
                    username=effective_auth.username,
                    status=login_result.get("Status"),
                )

            return success

        except APIError as e:
            self.logger.error(
                "registry_authentication_api_error",
                registry=effective_auth.registry if effective_auth else "unknown",
                error=str(e),
                status_code=e.status_code,
            )
            return False

        except Exception as e:
            self.logger.error(
                "registry_authentication_error",
                registry=effective_auth.registry if effective_auth else "unknown",
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def push_image(
        self,
        image_tag: str,
        auth: RegistryAuth | None = None,
    ) -> PushResult:
        """Push a Docker image to the registry.

        Args:
            image_tag: Full image tag to push (e.g., 'ghcr.io/org/repo:v1.0.0')
            auth: Optional authentication credentials (uses default if not provided)

        Returns:
            PushResult with success status, digest, logs, and timing
        """
        start_time = time.monotonic()

        self.logger.info(
            "docker_push_started",
            image_tag=image_tag,
        )

        # Authenticate if auth provided
        if auth is not None or self._auth is None:
            auth_success = await self.authenticate(auth)
            if not auth_success:
                return PushResult(
                    success=False,
                    image_tag=image_tag,
                    error="Authentication failed",
                    status=PushStatus.FAILED,
                    duration_seconds=time.monotonic() - start_time,
                )

        try:
            client = await asyncio.to_thread(self._get_client)

            # Push the image
            push_response = await asyncio.to_thread(
                client.images.push,
                repository=image_tag.rsplit(":", 1)[0] if ":" in image_tag else image_tag,
                tag=image_tag.rsplit(":", 1)[1] if ":" in image_tag else "latest",
                stream=True,
                decode=True,
            )

            # Collect push logs and extract digest
            push_log: list[str] = []
            digest: str | None = None

            for log_entry in push_response:
                if isinstance(log_entry, dict):
                    # Extract status and progress
                    status_msg = log_entry.get("status", "")
                    progress_msg = log_entry.get("progress", "")

                    if status_msg:
                        log_line = f"{status_msg}"
                        if progress_msg:
                            log_line += f" {progress_msg}"
                        push_log.append(log_line)

                    # Extract digest from aux field or status message
                    if "aux" in log_entry and isinstance(log_entry["aux"], dict):
                        digest = log_entry["aux"].get("Digest") or log_entry["aux"].get("digest")
                    elif "status" in log_entry:
                        # Try to extract digest from status message
                        match = _DIGEST_PATTERN.search(log_entry["status"])
                        if match:
                            digest = match.group(1)

                    # Check for errors
                    if "error" in log_entry:
                        error_msg = log_entry["error"]
                        push_log.append(f"ERROR: {error_msg}")
                        raise APIError(error_msg)

            duration = time.monotonic() - start_time

            self.logger.info(
                "docker_push_succeeded",
                image_tag=image_tag,
                digest=digest,
                duration_seconds=round(duration, 2),
                log_lines=len(push_log),
            )

            return PushResult(
                success=True,
                image_tag=image_tag,
                digest=digest,
                push_log=push_log,
                duration_seconds=duration,
                status=PushStatus.SUCCEEDED,
            )

        except APIError as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "docker_push_api_error",
                image_tag=image_tag,
                error=str(e),
                status_code=e.status_code if hasattr(e, "status_code") else None,
                duration_seconds=round(duration, 2),
            )
            return PushResult(
                success=False,
                image_tag=image_tag,
                duration_seconds=duration,
                error=str(e),
                status=PushStatus.FAILED,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "docker_push_error",
                image_tag=image_tag,
                error=str(e),
                error_type=type(e).__name__,
                duration_seconds=round(duration, 2),
            )
            return PushResult(
                success=False,
                image_tag=image_tag,
                duration_seconds=duration,
                error=str(e),
                status=PushStatus.FAILED,
            )

    async def push_with_retry(
        self,
        image_tag: str,
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
        auth: RegistryAuth | None = None,
        retry_config: RetryConfig | None = None,
    ) -> PushResult:
        """Push a Docker image with exponential backoff retry logic.

        Args:
            image_tag: Full image tag to push
            max_retries: Maximum number of retry attempts (deprecated, use retry_config)
            retry_delay_seconds: Initial retry delay (deprecated, use retry_config)
            auth: Optional authentication credentials
            retry_config: Retry configuration (overrides max_retries and retry_delay_seconds)

        Returns:
            PushResult with success status, digest, logs, and retry count
        """
        # Use retry_config if provided, otherwise build from parameters
        if retry_config is None:
            retry_config = RetryConfig(
                max_retries=max_retries,
                initial_delay_seconds=retry_delay_seconds,
            )

        self.logger.info(
            "docker_push_with_retry_started",
            image_tag=image_tag,
            max_retries=retry_config.max_retries,
            initial_delay=retry_config.initial_delay_seconds,
        )

        last_result: PushResult | None = None

        for attempt in range(retry_config.max_retries + 1):
            if attempt > 0:
                # Calculate exponential backoff delay
                delay = min(
                    retry_config.initial_delay_seconds * (retry_config.backoff_multiplier ** (attempt - 1)),
                    retry_config.max_delay_seconds,
                )

                self.logger.info(
                    "docker_push_retry_attempt",
                    image_tag=image_tag,
                    attempt=attempt,
                    max_retries=retry_config.max_retries,
                    delay_seconds=round(delay, 2),
                    previous_error=last_result.error if last_result else None,
                )

                await asyncio.sleep(delay)

            # Attempt push
            result = await self.push_image(image_tag, auth)
            result.retry_attempts = attempt

            if result.success:
                if attempt > 0:
                    self.logger.info(
                        "docker_push_retry_succeeded",
                        image_tag=image_tag,
                        attempts=attempt + 1,
                    )
                return result

            # Update status to retrying if we have retries left
            if attempt < retry_config.max_retries:
                result.status = PushStatus.RETRYING

            last_result = result

        # All retries exhausted
        if last_result is not None:
            last_result.status = PushStatus.FAILED
            self.logger.error(
                "docker_push_retry_exhausted",
                image_tag=image_tag,
                attempts=retry_config.max_retries + 1,
                final_error=last_result.error,
            )
            return last_result

        # Should never reach here, but provide a fallback
        return PushResult(
            success=False,
            image_tag=image_tag,
            error="Push failed with no result",
            status=PushStatus.FAILED,
            retry_attempts=retry_config.max_retries,
        )

    async def close(self) -> None:
        """Close the Docker client connection.

        Safe to call multiple times or if the client was never connected.
        """
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
                self.logger.info("registry_client_closed")
            except Exception as e:
                self.logger.warning(
                    "registry_client_close_error",
                    error=str(e),
                )
            finally:
                self._client = None
