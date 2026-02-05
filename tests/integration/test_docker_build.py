"""Integration tests for the Docker build system and image tagging.

These tests mock the docker-py client to verify Docker build operations,
streaming logs, rootless compatibility, and image tagging strategies
without requiring a running Docker daemon.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError, BuildError, DockerException, ImageNotFound

from forgemaster.config import DockerConfig
from forgemaster.pipeline.docker_ops import (
    BuildLogEntry,
    BuildResult,
    BuildStatus,
    DockerBuildClient,
    DockerHealth,
    RootlessConfig,
    SemverTag,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docker_config() -> DockerConfig:
    """Create a DockerConfig for testing.

    Returns:
        DockerConfig with test defaults.
    """
    return DockerConfig(
        registry="ghcr.io",
        rootless=False,
        build_timeout_seconds=120,
    )


@pytest.fixture
def rootless_config() -> DockerConfig:
    """Create a DockerConfig with rootless mode enabled.

    Returns:
        DockerConfig with rootless=True.
    """
    return DockerConfig(
        registry="ghcr.io",
        rootless=True,
        build_timeout_seconds=120,
    )


@pytest.fixture
def client(docker_config: DockerConfig) -> DockerBuildClient:
    """Create a DockerBuildClient with a mocked docker-py client.

    Args:
        docker_config: Docker configuration fixture.

    Returns:
        DockerBuildClient with injected mock.
    """
    build_client = DockerBuildClient(docker_config)
    mock_docker = MagicMock()
    mock_docker.version.return_value = {
        "Version": "24.0.7",
        "ApiVersion": "1.43",
    }
    build_client._client = mock_docker
    return build_client


@pytest.fixture
def rootless_client(rootless_config: DockerConfig) -> DockerBuildClient:
    """Create a DockerBuildClient with rootless mode enabled.

    Args:
        rootless_config: Rootless Docker configuration fixture.

    Returns:
        DockerBuildClient configured for rootless mode.
    """
    build_client = DockerBuildClient(rootless_config)
    mock_docker = MagicMock()
    mock_docker.version.return_value = {
        "Version": "24.0.7",
        "ApiVersion": "1.43",
    }
    build_client._client = mock_docker
    return build_client


@pytest.fixture
def build_context(tmp_path: Path) -> Path:
    """Create a temporary build context with a Dockerfile.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Path to the build context directory.
    """
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nCOPY . /app\n")
    return tmp_path


# ---------------------------------------------------------------------------
# DockerBuildClient Initialization Tests
# ---------------------------------------------------------------------------


class TestDockerClientInit:
    """Tests for DockerBuildClient initialization."""

    def test_init_with_defaults(self, docker_config: DockerConfig) -> None:
        """Test client initializes with config values."""
        build_client = DockerBuildClient(docker_config)
        assert build_client.config is docker_config
        assert build_client._client is None

    def test_init_rootless_config(self, rootless_config: DockerConfig) -> None:
        """Test client initializes with rootless configuration."""
        build_client = DockerBuildClient(rootless_config)
        assert build_client.config.rootless is True
        assert build_client.rootless_config.userns_mode == "keep-id"

    @patch("forgemaster.pipeline.docker_ops.docker.DockerClient.from_env")
    def test_get_client_success(
        self,
        mock_from_env: MagicMock,
        docker_config: DockerConfig,
    ) -> None:
        """Test successful Docker client connection."""
        mock_docker = MagicMock()
        mock_from_env.return_value = mock_docker

        build_client = DockerBuildClient(docker_config)
        result = build_client._get_client()

        assert result is mock_docker
        mock_from_env.assert_called_once()

    @patch("forgemaster.pipeline.docker_ops.docker.DockerClient.from_env")
    def test_get_client_connection_error(
        self,
        mock_from_env: MagicMock,
        docker_config: DockerConfig,
    ) -> None:
        """Test Docker client raises on connection failure."""
        mock_from_env.side_effect = DockerException("Connection refused")

        build_client = DockerBuildClient(docker_config)
        with pytest.raises(DockerException, match="Connection refused"):
            build_client._get_client()

    @patch("forgemaster.pipeline.docker_ops.docker.DockerClient")
    def test_get_client_with_docker_host_env(
        self,
        mock_client_cls: MagicMock,
        docker_config: DockerConfig,
    ) -> None:
        """Test client uses DOCKER_HOST environment variable."""
        mock_docker = MagicMock()
        mock_client_cls.return_value = mock_docker

        build_client = DockerBuildClient(docker_config)

        with patch.dict("os.environ", {"DOCKER_HOST": "tcp://localhost:2375"}):
            result = build_client._get_client()

        assert result is mock_docker
        mock_client_cls.assert_called_once_with(base_url="tcp://localhost:2375")

    def test_get_client_reuses_connection(self, client: DockerBuildClient) -> None:
        """Test that subsequent calls return the same client."""
        first = client._get_client()
        second = client._get_client()
        assert first is second


# ---------------------------------------------------------------------------
# Docker Health Check Tests
# ---------------------------------------------------------------------------


class TestDockerHealthCheck:
    """Tests for Docker daemon health checking."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: DockerBuildClient) -> None:
        """Test successful health check returns version info."""
        health = await client.check_docker_health()

        assert health.available is True
        assert health.version == "24.0.7"
        assert health.api_version == "1.43"
        assert health.error is None

    @pytest.mark.asyncio
    async def test_health_check_rootless_detection(
        self, rootless_client: DockerBuildClient
    ) -> None:
        """Test health check detects rootless mode from config."""
        health = await rootless_client.check_docker_health()

        assert health.available is True
        assert health.rootless is True

    @pytest.mark.asyncio
    async def test_health_check_daemon_unavailable(self, docker_config: DockerConfig) -> None:
        """Test health check when Docker daemon is not running."""
        build_client = DockerBuildClient(docker_config)

        with patch.object(
            build_client,
            "_get_client",
            side_effect=DockerException("Cannot connect"),
        ):
            health = await build_client.check_docker_health()

        assert health.available is False
        assert health.error is not None
        assert "Cannot connect" in health.error

    @pytest.mark.asyncio
    async def test_health_check_rootless_env_detection(self, client: DockerBuildClient) -> None:
        """Test rootless detection via DOCKER_HOST environment variable."""
        with patch.dict(
            "os.environ",
            {"DOCKER_HOST": "unix:///run/user/1000/docker.sock"},
        ):
            health = await client.check_docker_health()

        assert health.available is True
        assert health.rootless is True


# ---------------------------------------------------------------------------
# Image Build Tests
# ---------------------------------------------------------------------------


class TestImageBuild:
    """Tests for Docker image building."""

    @pytest.mark.asyncio
    async def test_build_success(self, client: DockerBuildClient, build_context: Path) -> None:
        """Test successful image build returns correct result."""
        mock_image = MagicMock()
        mock_image.id = "sha256:abc123def456"
        mock_image.tags = ["myapp:latest"]

        build_logs = [
            {"stream": "Step 1/2 : FROM python:3.12-slim"},
            {"stream": "Step 2/2 : COPY . /app"},
            {"stream": "Successfully built abc123def456"},
        ]

        client._client.images.build.return_value = (mock_image, build_logs)

        result = await client.build_image(
            path=build_context,
            tag="myapp:latest",
        )

        assert result.success is True
        assert result.image_id == "sha256:abc123def456"
        assert "myapp:latest" in result.tags
        assert result.status == BuildStatus.SUCCEEDED
        assert result.duration_seconds > 0
        assert len(result.build_log) == 3

    @pytest.mark.asyncio
    async def test_build_with_build_args(
        self, client: DockerBuildClient, build_context: Path
    ) -> None:
        """Test build passes build args to docker-py."""
        mock_image = MagicMock()
        mock_image.id = "sha256:abc"
        mock_image.tags = []

        client._client.images.build.return_value = (mock_image, [])

        build_args = {"PYTHON_VERSION": "3.12", "ENV": "production"}
        await client.build_image(
            path=build_context,
            build_args=build_args,
        )

        call_kwargs = client._client.images.build.call_args[1]
        assert call_kwargs["buildargs"] == build_args

    @pytest.mark.asyncio
    async def test_build_no_cache(self, client: DockerBuildClient, build_context: Path) -> None:
        """Test build passes no_cache flag."""
        mock_image = MagicMock()
        mock_image.id = "sha256:abc"
        mock_image.tags = []

        client._client.images.build.return_value = (mock_image, [])

        await client.build_image(path=build_context, no_cache=True)

        call_kwargs = client._client.images.build.call_args[1]
        assert call_kwargs["nocache"] is True

    @pytest.mark.asyncio
    async def test_build_failure(self, client: DockerBuildClient, build_context: Path) -> None:
        """Test build failure returns error result."""
        build_log = [
            {"stream": "Step 1/2 : FROM nonexistent:image"},
            {"error": "pull access denied"},
        ]
        client._client.images.build.side_effect = BuildError("Build failed", build_log)

        result = await client.build_image(path=build_context)

        assert result.success is False
        assert result.status == BuildStatus.FAILED
        assert result.error is not None
        assert len(result.build_log) > 0

    @pytest.mark.asyncio
    async def test_build_api_error(self, client: DockerBuildClient, build_context: Path) -> None:
        """Test build API error is handled gracefully."""
        response = MagicMock()
        response.status_code = 500
        response.reason = "Internal Server Error"
        client._client.images.build.side_effect = APIError("Server error", response=response)

        result = await client.build_image(path=build_context)

        assert result.success is False
        assert result.status == BuildStatus.FAILED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_build_path_not_found(self, client: DockerBuildClient) -> None:
        """Test build with nonexistent path returns error."""
        result = await client.build_image(path=Path("/nonexistent/path"))

        assert result.success is False
        assert result.status == BuildStatus.FAILED
        assert "does not exist" in (result.error or "")

    @pytest.mark.asyncio
    async def test_build_missing_dockerfile(
        self, client: DockerBuildClient, tmp_path: Path
    ) -> None:
        """Test build with missing Dockerfile returns error."""
        # tmp_path exists but has no Dockerfile
        result = await client.build_image(path=tmp_path)

        assert result.success is False
        assert result.status == BuildStatus.FAILED
        assert "Dockerfile not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_build_custom_dockerfile(self, client: DockerBuildClient, tmp_path: Path) -> None:
        """Test build with custom Dockerfile name."""
        custom_df = tmp_path / "Dockerfile.prod"
        custom_df.write_text("FROM python:3.12-slim\n")

        mock_image = MagicMock()
        mock_image.id = "sha256:custom"
        mock_image.tags = []
        client._client.images.build.return_value = (mock_image, [])

        result = await client.build_image(path=tmp_path, dockerfile="Dockerfile.prod")

        assert result.success is True
        call_kwargs = client._client.images.build.call_args[1]
        assert call_kwargs["dockerfile"] == "Dockerfile.prod"

    @pytest.mark.asyncio
    async def test_build_empty_log(self, client: DockerBuildClient, build_context: Path) -> None:
        """Test build with empty log output."""
        mock_image = MagicMock()
        mock_image.id = "sha256:empty"
        mock_image.tags = ["empty:latest"]
        client._client.images.build.return_value = (mock_image, [])

        result = await client.build_image(path=build_context)

        assert result.success is True
        assert result.build_log == []

    @pytest.mark.asyncio
    async def test_build_no_tag_on_image(
        self, client: DockerBuildClient, build_context: Path
    ) -> None:
        """Test build result when image has no tags."""
        mock_image = MagicMock()
        mock_image.id = "sha256:notags"
        mock_image.tags = None
        client._client.images.build.return_value = (mock_image, [])

        result = await client.build_image(path=build_context)

        assert result.success is True
        assert result.tags == []


# ---------------------------------------------------------------------------
# Build Log Streaming Tests
# ---------------------------------------------------------------------------


class TestBuildLogStreaming:
    """Tests for streaming build log output."""

    @pytest.mark.asyncio
    async def test_streaming_build_multiple_entries(
        self, client: DockerBuildClient, build_context: Path
    ) -> None:
        """Test streaming build yields multiple log entries."""
        chunks = [
            {"stream": "Step 1/3 : FROM python:3.12\n"},
            {"stream": "Step 2/3 : COPY . /app\n"},
            {"status": "Downloading", "progress": "[=====>    ] 50%"},
            {"stream": "Step 3/3 : CMD python app.py\n"},
            {"aux": {"ID": "sha256:final123"}},
        ]
        client._client.api.build.return_value = iter(chunks)

        entries: list[BuildLogEntry] = []
        async for entry in client.build_image_streaming(path=build_context):
            entries.append(entry)

        assert len(entries) == 5
        assert entries[0].stream is not None
        assert "Step 1/3" in (entries[0].stream or "")
        assert entries[2].status == "Downloading"
        assert entries[2].progress is not None
        assert entries[4].aux is not None
        assert entries[4].aux.get("ID") == "sha256:final123"

    @pytest.mark.asyncio
    async def test_streaming_build_error_entry(
        self, client: DockerBuildClient, build_context: Path
    ) -> None:
        """Test streaming build yields error entries."""
        chunks = [
            {"stream": "Step 1/2 : FROM invalid\n"},
            {"error": "pull access denied for invalid"},
        ]
        client._client.api.build.return_value = iter(chunks)

        entries: list[BuildLogEntry] = []
        async for entry in client.build_image_streaming(path=build_context):
            entries.append(entry)

        assert len(entries) == 2
        assert entries[1].error is not None
        assert "pull access denied" in entries[1].error

    @pytest.mark.asyncio
    async def test_streaming_build_path_not_found(self, client: DockerBuildClient) -> None:
        """Test streaming build with invalid path yields error."""
        entries: list[BuildLogEntry] = []
        async for entry in client.build_image_streaming(
            path=Path("/nonexistent"),
        ):
            entries.append(entry)

        assert len(entries) == 1
        assert entries[0].error is not None
        assert "does not exist" in entries[0].error

    @pytest.mark.asyncio
    async def test_streaming_build_missing_dockerfile(
        self, client: DockerBuildClient, tmp_path: Path
    ) -> None:
        """Test streaming build with missing Dockerfile yields error."""
        entries: list[BuildLogEntry] = []
        async for entry in client.build_image_streaming(path=tmp_path):
            entries.append(entry)

        assert len(entries) == 1
        assert entries[0].error is not None
        assert "Dockerfile not found" in entries[0].error

    @pytest.mark.asyncio
    async def test_streaming_build_exception(
        self, client: DockerBuildClient, build_context: Path
    ) -> None:
        """Test streaming build handles unexpected exceptions."""
        client._client.api.build.side_effect = DockerException("Connection lost")

        entries: list[BuildLogEntry] = []
        async for entry in client.build_image_streaming(path=build_context):
            entries.append(entry)

        assert len(entries) == 1
        assert entries[0].error is not None
        assert "Connection lost" in entries[0].error


# ---------------------------------------------------------------------------
# Rootless Compatibility Tests
# ---------------------------------------------------------------------------


class TestRootlessCompatibility:
    """Tests for rootless Docker mode handling."""

    def test_rootless_config_defaults(self) -> None:
        """Test RootlessConfig has correct defaults."""
        config = RootlessConfig()
        assert config.userns_mode == "keep-id"
        assert "label=disable" in config.security_opt

    def test_rootless_config_custom(self) -> None:
        """Test RootlessConfig with custom values."""
        config = RootlessConfig(
            userns_mode="host",
            security_opt=["seccomp=unconfined"],
        )
        assert config.userns_mode == "host"
        assert config.security_opt == ["seccomp=unconfined"]

    @pytest.mark.asyncio
    async def test_rootless_health_check(self, rootless_client: DockerBuildClient) -> None:
        """Test health check identifies rootless mode."""
        health = await rootless_client.check_docker_health()
        assert health.rootless is True

    def test_client_stores_rootless_config(self, rootless_client: DockerBuildClient) -> None:
        """Test client stores rootless configuration."""
        assert rootless_client.rootless_config is not None
        assert rootless_client.rootless_config.userns_mode == "keep-id"


# ---------------------------------------------------------------------------
# Git SHA Tagging Tests
# ---------------------------------------------------------------------------


class TestGitShaTagging:
    """Tests for git SHA-based image tagging."""

    def test_sha_tag_valid(self, client: DockerBuildClient) -> None:
        """Test generating tag from valid git SHA."""
        sha = "a" * 40
        tag = client.generate_sha_tag(sha, repository="forgemaster")

        assert tag == f"ghcr.io/forgemaster:{sha[:12]}"

    def test_sha_tag_with_custom_registry(self, client: DockerBuildClient) -> None:
        """Test SHA tag with custom registry."""
        sha = "b" * 40
        tag = client.generate_sha_tag(sha, registry="docker.io", repository="myorg/myapp")

        assert tag == f"docker.io/myorg/myapp:{sha[:12]}"

    def test_sha_tag_no_repository(self, client: DockerBuildClient) -> None:
        """Test SHA tag without repository."""
        sha = "c" * 40
        tag = client.generate_sha_tag(sha)

        assert tag == f"ghcr.io:{sha[:12]}"

    def test_sha_tag_invalid_sha_too_short(self, client: DockerBuildClient) -> None:
        """Test SHA tag with too-short SHA raises ValueError."""
        with pytest.raises(ValueError, match="Invalid git SHA"):
            client.generate_sha_tag("abc123")

    def test_sha_tag_invalid_sha_not_hex(self, client: DockerBuildClient) -> None:
        """Test SHA tag with non-hex characters raises ValueError."""
        with pytest.raises(ValueError, match="Invalid git SHA"):
            client.generate_sha_tag("g" * 40)

    def test_sha_tag_invalid_sha_empty(self, client: DockerBuildClient) -> None:
        """Test SHA tag with empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid git SHA"):
            client.generate_sha_tag("")

    def test_sha_tag_invalid_sha_uppercase(self, client: DockerBuildClient) -> None:
        """Test SHA tag with uppercase hex raises ValueError."""
        with pytest.raises(ValueError, match="Invalid git SHA"):
            client.generate_sha_tag("A" * 40)

    def test_sha_tag_uses_first_12_chars(self, client: DockerBuildClient) -> None:
        """Test that SHA tag uses exactly the first 12 characters."""
        sha = "0123456789abcdef" * 2 + "01234567"
        tag = client.generate_sha_tag(sha, repository="test")

        assert ":0123456789ab" in tag


# ---------------------------------------------------------------------------
# Semantic Version Tagging Tests
# ---------------------------------------------------------------------------


class TestSemverTagging:
    """Tests for semantic version-based image tagging."""

    def test_parse_semver_basic(self, client: DockerBuildClient) -> None:
        """Test parsing basic semver string."""
        result = client.parse_semver("1.2.3")

        assert result.major == 1
        assert result.minor == 2
        assert result.patch == 3
        assert result.prerelease is None
        assert result.build is None

    def test_parse_semver_with_prerelease(self, client: DockerBuildClient) -> None:
        """Test parsing semver with prerelease identifier."""
        result = client.parse_semver("1.0.0-alpha.1")

        assert result.major == 1
        assert result.minor == 0
        assert result.patch == 0
        assert result.prerelease == "alpha.1"
        assert result.build is None

    def test_parse_semver_with_build(self, client: DockerBuildClient) -> None:
        """Test parsing semver with build metadata."""
        result = client.parse_semver("1.0.0+build.123")

        assert result.major == 1
        assert result.build == "build.123"
        assert result.prerelease is None

    def test_parse_semver_full(self, client: DockerBuildClient) -> None:
        """Test parsing semver with prerelease and build."""
        result = client.parse_semver("2.1.0-rc.2+20240101")

        assert result.major == 2
        assert result.minor == 1
        assert result.patch == 0
        assert result.prerelease == "rc.2"
        assert result.build == "20240101"

    def test_parse_semver_zero_version(self, client: DockerBuildClient) -> None:
        """Test parsing zero version."""
        result = client.parse_semver("0.0.0")

        assert result.major == 0
        assert result.minor == 0
        assert result.patch == 0

    def test_parse_semver_invalid_format(self, client: DockerBuildClient) -> None:
        """Test parsing invalid semver raises ValueError."""
        with pytest.raises(ValueError, match="Invalid semver"):
            client.parse_semver("not-a-version")

    def test_parse_semver_missing_patch(self, client: DockerBuildClient) -> None:
        """Test parsing semver missing patch number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid semver"):
            client.parse_semver("1.2")

    def test_parse_semver_leading_zero(self, client: DockerBuildClient) -> None:
        """Test parsing semver with leading zero in major raises ValueError."""
        with pytest.raises(ValueError, match="Invalid semver"):
            client.parse_semver("01.2.3")

    def test_semver_tag_to_string(self) -> None:
        """Test SemverTag to_string conversion."""
        tag = SemverTag(major=1, minor=2, patch=3, prerelease="beta.1", build="001")
        assert tag.to_string() == "1.2.3-beta.1+001"

    def test_semver_tag_to_string_simple(self) -> None:
        """Test SemverTag to_string without prerelease/build."""
        tag = SemverTag(major=3, minor=0, patch=1)
        assert tag.to_string() == "3.0.1"

    def test_generate_semver_tag(self, client: DockerBuildClient) -> None:
        """Test generating Docker tag from semver string."""
        tag = client.generate_semver_tag("1.2.3", repository="myapp")

        assert tag == "ghcr.io/myapp:1.2.3"

    def test_generate_semver_tag_with_prerelease(self, client: DockerBuildClient) -> None:
        """Test generating Docker tag from semver with prerelease."""
        tag = client.generate_semver_tag("1.0.0-alpha", repository="myapp")

        assert tag == "ghcr.io/myapp:1.0.0-alpha"

    def test_generate_semver_tag_custom_registry(self, client: DockerBuildClient) -> None:
        """Test generating semver tag with custom registry."""
        tag = client.generate_semver_tag("2.0.0", registry="docker.io", repository="org/app")

        assert tag == "docker.io/org/app:2.0.0"

    def test_generate_semver_tag_no_repository(self, client: DockerBuildClient) -> None:
        """Test generating semver tag without repository."""
        tag = client.generate_semver_tag("1.0.0")

        assert tag == "ghcr.io:1.0.0"

    def test_generate_semver_tag_invalid(self, client: DockerBuildClient) -> None:
        """Test generating tag with invalid semver raises ValueError."""
        with pytest.raises(ValueError, match="Invalid semver"):
            client.generate_semver_tag("not-a-version", repository="myapp")


# ---------------------------------------------------------------------------
# Image Tagging Tests
# ---------------------------------------------------------------------------


class TestImageTagging:
    """Tests for applying tags to Docker images."""

    @pytest.mark.asyncio
    async def test_tag_image_success(self, client: DockerBuildClient) -> None:
        """Test successful image tagging."""
        mock_image = MagicMock()
        mock_image.tag.return_value = True
        client._client.images.get.return_value = mock_image

        result = await client.tag_image("sha256:abc123", "ghcr.io/myapp:v1.0")

        assert result is True
        mock_image.tag.assert_called_once_with(repository="ghcr.io/myapp", tag="v1.0")

    @pytest.mark.asyncio
    async def test_tag_image_not_found(self, client: DockerBuildClient) -> None:
        """Test tagging a nonexistent image returns False."""
        client._client.images.get.side_effect = ImageNotFound("Not found")

        result = await client.tag_image("sha256:nonexistent", "myapp:latest")

        assert result is False

    @pytest.mark.asyncio
    async def test_tag_image_api_error(self, client: DockerBuildClient) -> None:
        """Test tagging with API error returns False."""
        response = MagicMock()
        response.status_code = 500
        client._client.images.get.side_effect = APIError("Server error", response=response)

        result = await client.tag_image("sha256:abc", "myapp:latest")

        assert result is False

    @pytest.mark.asyncio
    async def test_tag_image_no_tag_part(self, client: DockerBuildClient) -> None:
        """Test tagging without explicit tag defaults to latest."""
        mock_image = MagicMock()
        mock_image.tag.return_value = True
        client._client.images.get.return_value = mock_image

        result = await client.tag_image("sha256:abc123", "ghcr.io/myapp")

        assert result is True
        mock_image.tag.assert_called_once_with(repository="ghcr.io/myapp", tag="latest")


# ---------------------------------------------------------------------------
# Latest Tag Management Tests
# ---------------------------------------------------------------------------


class TestLatestTagManagement:
    """Tests for latest tag operations."""

    @pytest.mark.asyncio
    async def test_update_latest_tag_success(self, client: DockerBuildClient) -> None:
        """Test updating latest tag succeeds."""
        mock_image = MagicMock()
        mock_image.tag.return_value = True
        client._client.images.get.return_value = mock_image

        result = await client.update_latest_tag("sha256:abc123", repository="myapp")

        assert result is True
        mock_image.tag.assert_called_once_with(repository="ghcr.io/myapp", tag="latest")

    @pytest.mark.asyncio
    async def test_update_latest_tag_custom_registry(self, client: DockerBuildClient) -> None:
        """Test updating latest tag with custom registry."""
        mock_image = MagicMock()
        mock_image.tag.return_value = True
        client._client.images.get.return_value = mock_image

        result = await client.update_latest_tag(
            "sha256:abc123",
            registry="docker.io",
            repository="org/app",
        )

        assert result is True
        mock_image.tag.assert_called_once_with(repository="docker.io/org/app", tag="latest")

    @pytest.mark.asyncio
    async def test_update_latest_tag_no_repository(self, client: DockerBuildClient) -> None:
        """Test updating latest tag without repository."""
        mock_image = MagicMock()
        mock_image.tag.return_value = True
        client._client.images.get.return_value = mock_image

        result = await client.update_latest_tag("sha256:abc123")

        assert result is True
        mock_image.tag.assert_called_once_with(repository="ghcr.io", tag="latest")

    @pytest.mark.asyncio
    async def test_update_latest_tag_image_not_found(self, client: DockerBuildClient) -> None:
        """Test updating latest tag for missing image returns False."""
        client._client.images.get.side_effect = ImageNotFound("Not found")

        result = await client.update_latest_tag("sha256:missing", repository="myapp")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_image_tags_success(self, client: DockerBuildClient) -> None:
        """Test retrieving tags for an image."""
        mock_image = MagicMock()
        mock_image.tags = ["myapp:latest", "myapp:v1.0.0", "myapp:abc123"]
        client._client.images.get.return_value = mock_image

        tags = await client.get_image_tags("sha256:abc123")

        assert len(tags) == 3
        assert "myapp:latest" in tags
        assert "myapp:v1.0.0" in tags

    @pytest.mark.asyncio
    async def test_get_image_tags_none(self, client: DockerBuildClient) -> None:
        """Test retrieving tags when image has None tags."""
        mock_image = MagicMock()
        mock_image.tags = None
        client._client.images.get.return_value = mock_image

        tags = await client.get_image_tags("sha256:notags")

        assert tags == []

    @pytest.mark.asyncio
    async def test_get_image_tags_not_found(self, client: DockerBuildClient) -> None:
        """Test retrieving tags for missing image returns empty list."""
        client._client.images.get.side_effect = ImageNotFound("Not found")

        tags = await client.get_image_tags("sha256:missing")

        assert tags == []

    @pytest.mark.asyncio
    async def test_get_image_tags_error(self, client: DockerBuildClient) -> None:
        """Test retrieving tags with unexpected error returns empty list."""
        client._client.images.get.side_effect = Exception("Unexpected")

        tags = await client.get_image_tags("sha256:error")

        assert tags == []


# ---------------------------------------------------------------------------
# Model Validation Tests
# ---------------------------------------------------------------------------


class TestModelValidation:
    """Tests for Pydantic model validation."""

    def test_build_result_defaults(self) -> None:
        """Test BuildResult default values."""
        result = BuildResult()

        assert result.image_id == ""
        assert result.tags == []
        assert result.build_log == []
        assert result.duration_seconds == 0.0
        assert result.success is False
        assert result.error is None
        assert result.status == BuildStatus.PENDING

    def test_build_result_full(self) -> None:
        """Test BuildResult with all fields populated."""
        result = BuildResult(
            image_id="sha256:abc",
            tags=["v1", "latest"],
            build_log=["Step 1", "Step 2"],
            duration_seconds=12.5,
            success=True,
            error=None,
            status=BuildStatus.SUCCEEDED,
        )

        assert result.image_id == "sha256:abc"
        assert len(result.tags) == 2
        assert result.duration_seconds == 12.5

    def test_docker_health_defaults(self) -> None:
        """Test DockerHealth default values."""
        health = DockerHealth()

        assert health.available is False
        assert health.rootless is False
        assert health.version is None
        assert health.api_version is None
        assert health.error is None

    def test_build_log_entry_all_none(self) -> None:
        """Test BuildLogEntry with all None fields."""
        entry = BuildLogEntry()

        assert entry.stream is None
        assert entry.error is None
        assert entry.status is None
        assert entry.progress is None
        assert entry.aux is None

    def test_build_log_entry_stream(self) -> None:
        """Test BuildLogEntry with stream data."""
        entry = BuildLogEntry(stream="Step 1/2 : FROM python:3.12\n")

        assert entry.stream is not None
        assert "FROM python" in entry.stream

    def test_build_status_enum_values(self) -> None:
        """Test BuildStatus enum has all expected values."""
        assert BuildStatus.PENDING == "pending"
        assert BuildStatus.BUILDING == "building"
        assert BuildStatus.SUCCEEDED == "succeeded"
        assert BuildStatus.FAILED == "failed"
        assert BuildStatus.CANCELLED == "cancelled"

    def test_build_status_is_string(self) -> None:
        """Test BuildStatus values are strings."""
        assert isinstance(BuildStatus.PENDING, str)
        assert isinstance(BuildStatus.SUCCEEDED, str)

    def test_semver_tag_validation(self) -> None:
        """Test SemverTag field validation."""
        tag = SemverTag(major=0, minor=0, patch=0)
        assert tag.major == 0

    def test_rootless_config_defaults(self) -> None:
        """Test RootlessConfig default values."""
        config = RootlessConfig()
        assert config.userns_mode == "keep-id"
        assert len(config.security_opt) == 1


# ---------------------------------------------------------------------------
# Client Close / Cleanup Tests
# ---------------------------------------------------------------------------


class TestClientCleanup:
    """Tests for client close and resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_connected_client(self, client: DockerBuildClient) -> None:
        """Test closing a connected client."""
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_unconnected_client(self, docker_config: DockerConfig) -> None:
        """Test closing a client that was never connected."""
        build_client = DockerBuildClient(docker_config)
        await build_client.close()  # Should not raise
        assert build_client._client is None

    @pytest.mark.asyncio
    async def test_close_with_error(self, client: DockerBuildClient) -> None:
        """Test close handles errors gracefully."""
        client._client.close.side_effect = Exception("Close failed")

        await client.close()  # Should not raise
        assert client._client is None
