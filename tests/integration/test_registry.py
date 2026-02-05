"""Integration tests for Docker registry operations.

Tests cover authentication, image pushing, retry logic with exponential backoff,
and error handling scenarios using mocked docker-py client.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docker.errors import APIError, DockerException

from forgemaster.config import DockerConfig
from forgemaster.pipeline.registry import (
    PushResult,
    PushStatus,
    RegistryAuth,
    RegistryClient,
    RetryConfig,
)


@pytest.fixture
def docker_config() -> DockerConfig:
    """Create a test Docker configuration."""
    return DockerConfig(
        registry="ghcr.io",
        rootless=True,
        build_timeout_seconds=600,
    )


@pytest.fixture
def registry_auth() -> RegistryAuth:
    """Create test registry authentication credentials."""
    return RegistryAuth(
        registry="ghcr.io",
        username="testuser",
        password="testtoken",
    )


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """Create a mock Docker client."""
    client = MagicMock()
    client.login.return_value = {"Status": "Login Succeeded"}
    client.images.push.return_value = [
        {"status": "Pushing", "progress": "1/10"},
        {"status": "Pushed", "aux": {"Digest": "sha256:abc123"}},
    ]
    return client


class TestRegistryClientInitialization:
    """Tests for RegistryClient initialization."""

    def test_init_without_auth(self, docker_config: DockerConfig) -> None:
        """Test initialization without authentication."""
        client = RegistryClient(docker_config)
        assert client.config == docker_config
        assert client._auth is None

    def test_init_with_auth(self, docker_config: DockerConfig, registry_auth: RegistryAuth) -> None:
        """Test initialization with authentication credentials."""
        client = RegistryClient(docker_config, auth=registry_auth)
        assert client.config == docker_config
        assert client._auth == registry_auth

    def test_init_stores_config_registry(self, docker_config: DockerConfig) -> None:
        """Test that registry from config is stored."""
        client = RegistryClient(docker_config)
        assert client.config.registry == "ghcr.io"


class TestRegistryAuth:
    """Tests for RegistryAuth model."""

    def test_registry_auth_required_fields(self) -> None:
        """Test that required fields are present."""
        auth = RegistryAuth(
            registry="ghcr.io",
            username="user",
            password="pass",
        )
        assert auth.registry == "ghcr.io"
        assert auth.username == "user"
        assert auth.password == "pass"
        assert auth.token is None

    def test_registry_auth_with_token(self) -> None:
        """Test token-based authentication."""
        auth = RegistryAuth(
            registry="docker.io",
            username="user",
            password="pass",
            token="explicit_token",
        )
        assert auth.token == "explicit_token"

    def test_registry_auth_pydantic_validation(self) -> None:
        """Test that Pydantic validation works."""
        with pytest.raises(Exception):
            # Missing required field
            RegistryAuth(registry="ghcr.io", username="user")  # type: ignore


class TestRegistryAuthentication:
    """Tests for Docker registry authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_success_with_password(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test successful password-based authentication."""
        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config)
            result = await client.authenticate(registry_auth)

            assert result is True
            assert client._auth == registry_auth
            mock_docker_client.login.assert_called_once_with(
                username="testuser",
                password="testtoken",
                registry="ghcr.io",
            )

    @pytest.mark.asyncio
    async def test_authenticate_success_with_token(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test successful token-based authentication."""
        auth = RegistryAuth(
            registry="ghcr.io",
            username="user",
            password="pass",
            token="explicit_token",
        )

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config)
            result = await client.authenticate(auth)

            assert result is True
            # Should use token, not password
            mock_docker_client.login.assert_called_once_with(
                username="user",
                password="explicit_token",
                registry="ghcr.io",
            )

    @pytest.mark.asyncio
    async def test_authenticate_failure(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test authentication failure."""
        mock_docker_client.login.return_value = {"Status": "Login Failed"}

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config)
            result = await client.authenticate(registry_auth)

            assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_api_error(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test authentication with API error."""
        mock_docker_client.login.side_effect = APIError("Unauthorized")

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config)
            result = await client.authenticate(registry_auth)

            assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_from_env(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test authentication from environment variables."""
        with patch.dict(
            os.environ,
            {
                "DOCKER_REGISTRY_USERNAME": "envuser",
                "DOCKER_REGISTRY_PASSWORD": "envpass",
            },
        ):
            with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
                client = RegistryClient(docker_config)
                result = await client.authenticate()

                assert result is True
                mock_docker_client.login.assert_called_once_with(
                    username="envuser",
                    password="envpass",
                    registry="ghcr.io",
                )

    @pytest.mark.asyncio
    async def test_authenticate_from_env_with_token(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test authentication from environment with token."""
        with patch.dict(
            os.environ,
            {
                "DOCKER_REGISTRY_USERNAME": "envuser",
                "DOCKER_REGISTRY_TOKEN": "envtoken",
            },
        ):
            with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
                client = RegistryClient(docker_config)
                result = await client.authenticate()

                assert result is True
                # Should use token from env
                mock_docker_client.login.assert_called_once_with(
                    username="envuser",
                    password="envtoken",
                    registry="ghcr.io",
                )

    @pytest.mark.asyncio
    async def test_authenticate_no_credentials(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test authentication with no credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
                client = RegistryClient(docker_config)
                result = await client.authenticate()

                assert result is False
                mock_docker_client.login.assert_not_called()

    @pytest.mark.asyncio
    async def test_authenticate_uses_default_auth(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test that authenticate uses default auth from constructor."""
        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.authenticate()

            assert result is True
            mock_docker_client.login.assert_called_once()


class TestPushImage:
    """Tests for push_image method."""

    @pytest.mark.asyncio
    async def test_push_success(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test successful image push."""
        mock_docker_client.images.push.return_value = [
            {"status": "Pushing", "progress": "1/10"},
            {"status": "Pushed", "aux": {"Digest": "sha256:abcdef1234567890"}},
        ]

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_image("ghcr.io/org/repo:v1.0.0")

            assert result.success is True
            assert result.status == PushStatus.SUCCEEDED
            assert result.image_tag == "ghcr.io/org/repo:v1.0.0"
            assert result.digest == "sha256:abcdef1234567890"
            assert len(result.push_log) > 0

    @pytest.mark.asyncio
    async def test_push_with_digest_in_status(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test push with digest in status message."""
        mock_docker_client.images.push.return_value = [
            {"status": "Pushing"},
            {"status": "digest: sha256:fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321 size: 1234"},
        ]

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_image("ghcr.io/org/repo:latest")

            assert result.success is True
            assert result.digest == "sha256:fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"

    @pytest.mark.asyncio
    async def test_push_without_tag(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test push with no tag defaults to latest."""
        mock_docker_client.images.push.return_value = [
            {"status": "Pushed", "aux": {"Digest": "sha256:abc"}},
        ]

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_image("ghcr.io/org/repo")

            assert result.success is True
            mock_docker_client.images.push.assert_called_once_with(
                repository="ghcr.io/org/repo",
                tag="latest",
                stream=True,
                decode=True,
            )

    @pytest.mark.asyncio
    async def test_push_api_error(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test push with API error."""
        mock_docker_client.images.push.side_effect = APIError("Push failed")

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_image("ghcr.io/org/repo:v1.0.0")

            assert result.success is False
            assert result.status == PushStatus.FAILED
            assert "Push failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_push_with_error_in_stream(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test push with error in stream."""
        mock_docker_client.images.push.return_value = [
            {"status": "Pushing"},
            {"error": "Image not found"},
        ]

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_image("ghcr.io/org/repo:v1.0.0")

            assert result.success is False
            assert result.status == PushStatus.FAILED

    @pytest.mark.asyncio
    async def test_push_auth_failure(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test push with authentication failure."""
        mock_docker_client.login.return_value = {"Status": "Login Failed"}

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config)
            auth = RegistryAuth(registry="ghcr.io", username="user", password="wrong")
            result = await client.push_image("ghcr.io/org/repo:v1.0.0", auth=auth)

            assert result.success is False
            assert result.status == PushStatus.FAILED
            assert "Authentication failed" in (result.error or "")


class TestPushWithRetry:
    """Tests for push_with_retry method."""

    @pytest.mark.asyncio
    async def test_push_retry_success_first_try(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test retry logic with success on first try."""
        mock_docker_client.images.push.return_value = [
            {"status": "Pushed", "aux": {"Digest": "sha256:abc"}},
        ]

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_with_retry("ghcr.io/org/repo:v1.0.0", max_retries=3)

            assert result.success is True
            assert result.retry_attempts == 0
            assert result.status == PushStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_push_retry_success_after_retry(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test retry logic with success after one retry."""
        call_count = 0

        def push_side_effect(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIError("Network error")
            return [{"status": "Pushed", "aux": {"Digest": "sha256:abc"}}]

        mock_docker_client.images.push.side_effect = push_side_effect

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_with_retry(
                "ghcr.io/org/repo:v1.0.0",
                max_retries=3,
                retry_delay_seconds=0.1,
            )

            assert result.success is True
            assert result.retry_attempts == 1
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_push_retry_all_exhausted(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test retry logic with all retries exhausted."""
        mock_docker_client.images.push.side_effect = APIError("Persistent error")

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_with_retry(
                "ghcr.io/org/repo:v1.0.0",
                max_retries=2,
                retry_delay_seconds=0.1,
            )

            assert result.success is False
            assert result.status == PushStatus.FAILED
            assert result.retry_attempts == 2
            assert "Persistent error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_push_retry_with_retry_config(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test retry logic with explicit RetryConfig."""
        retry_config = RetryConfig(
            max_retries=5,
            initial_delay_seconds=0.1,
            max_delay_seconds=1.0,
            backoff_multiplier=2.0,
        )

        mock_docker_client.images.push.return_value = [
            {"status": "Pushed", "aux": {"Digest": "sha256:abc"}},
        ]

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_with_retry(
                "ghcr.io/org/repo:v1.0.0",
                retry_config=retry_config,
            )

            assert result.success is True


class TestExponentialBackoff:
    """Tests for exponential backoff calculation."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test that exponential backoff increases delay."""
        import time
        call_times: list[float] = []

        def push_side_effect(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            call_times.append(time.monotonic())
            if len(call_times) < 3:
                raise APIError("Error")
            return [{"status": "Pushed", "aux": {"Digest": "sha256:abc"}}]

        mock_docker_client.images.push.side_effect = push_side_effect

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_with_retry(
                "ghcr.io/org/repo:v1.0.0",
                max_retries=3,
                retry_delay_seconds=0.1,
            )

            assert result.success is True
            assert len(call_times) == 3

            # Verify delays are increasing (allowing small tolerance)
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay2 > delay1

    @pytest.mark.asyncio
    async def test_max_delay_cap(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test that delay is capped at max_delay_seconds."""
        import time
        retry_config = RetryConfig(
            max_retries=5,
            initial_delay_seconds=1.0,
            max_delay_seconds=2.0,
            backoff_multiplier=10.0,  # Very high multiplier
        )

        call_times: list[float] = []

        def push_side_effect(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            call_times.append(time.monotonic())
            if len(call_times) < 4:
                raise APIError("Error")
            return [{"status": "Pushed", "aux": {"Digest": "sha256:abc"}}]

        mock_docker_client.images.push.side_effect = push_side_effect

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            await client.push_with_retry(
                "ghcr.io/org/repo:v1.0.0",
                retry_config=retry_config,
            )

            # Verify delays don't exceed max_delay
            for i in range(1, len(call_times)):
                delay = call_times[i] - call_times[i - 1]
                # Allow small tolerance for timing
                assert delay <= retry_config.max_delay_seconds + 0.5


class TestRetryConfig:
    """Tests for RetryConfig model."""

    def test_retry_config_defaults(self) -> None:
        """Test default values for RetryConfig."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay_seconds == 5.0
        assert config.max_delay_seconds == 60.0
        assert config.backoff_multiplier == 2.0

    def test_retry_config_custom_values(self) -> None:
        """Test custom values for RetryConfig."""
        config = RetryConfig(
            max_retries=5,
            initial_delay_seconds=10.0,
            max_delay_seconds=120.0,
            backoff_multiplier=3.0,
        )
        assert config.max_retries == 5
        assert config.initial_delay_seconds == 10.0
        assert config.max_delay_seconds == 120.0
        assert config.backoff_multiplier == 3.0

    def test_retry_config_validation(self) -> None:
        """Test that invalid values are rejected."""
        with pytest.raises(Exception):
            RetryConfig(max_retries=-1)

        with pytest.raises(Exception):
            RetryConfig(initial_delay_seconds=-1.0)

        with pytest.raises(Exception):
            RetryConfig(backoff_multiplier=0.5)


class TestPushResult:
    """Tests for PushResult model."""

    def test_push_result_defaults(self) -> None:
        """Test default values for PushResult."""
        result = PushResult(image_tag="test:latest")
        assert result.success is False
        assert result.digest is None
        assert result.push_log == []
        assert result.duration_seconds == 0.0
        assert result.error is None
        assert result.status == PushStatus.PENDING
        assert result.retry_attempts == 0

    def test_push_result_success(self) -> None:
        """Test successful PushResult."""
        result = PushResult(
            success=True,
            image_tag="ghcr.io/org/repo:v1.0.0",
            digest="sha256:abc123",
            push_log=["Pushing", "Pushed"],
            duration_seconds=10.5,
            status=PushStatus.SUCCEEDED,
        )
        assert result.success is True
        assert result.digest == "sha256:abc123"
        assert len(result.push_log) == 2

    def test_push_result_failure(self) -> None:
        """Test failed PushResult."""
        result = PushResult(
            success=False,
            image_tag="ghcr.io/org/repo:v1.0.0",
            error="Push failed",
            status=PushStatus.FAILED,
            retry_attempts=3,
        )
        assert result.success is False
        assert result.error == "Push failed"
        assert result.retry_attempts == 3


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_push_empty_tag(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test push with empty tag."""
        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_image("")

            # Should still attempt push
            assert mock_docker_client.images.push.called

    @pytest.mark.asyncio
    async def test_connection_timeout(
        self, docker_config: DockerConfig, registry_auth: RegistryAuth, mock_docker_client: MagicMock
    ) -> None:
        """Test handling of connection timeout."""
        mock_docker_client.images.push.side_effect = DockerException("Connection timeout")

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config, auth=registry_auth)
            result = await client.push_image("ghcr.io/org/repo:v1.0.0")

            assert result.success is False
            assert "timeout" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_close_client(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test closing the client."""
        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config)
            # Force client creation
            client._get_client()
            await client.close()

            assert client._client is None
            mock_docker_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_client_not_connected(
        self, docker_config: DockerConfig
    ) -> None:
        """Test closing client that was never connected."""
        client = RegistryClient(docker_config)
        # Should not raise
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_invalid_registry_url(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test authentication with invalid registry."""
        auth = RegistryAuth(
            registry="invalid://registry",
            username="user",
            password="pass",
        )

        mock_docker_client.login.side_effect = APIError("Invalid registry")

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = RegistryClient(docker_config)
            result = await client.authenticate(auth)

            assert result is False
