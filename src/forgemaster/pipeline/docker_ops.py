"""Docker build system and image tagging for FORGEMASTER.

This module provides a high-level async interface to Docker operations using docker-py,
with comprehensive error handling and structured logging. Supports image building with
streaming logs, rootless Docker compatibility, and flexible image tagging strategies
including git SHA, semantic versioning, and latest tag management.

Example usage:
    >>> from forgemaster.config import DockerConfig
    >>> from forgemaster.pipeline.docker_ops import DockerBuildClient
    >>>
    >>> config = DockerConfig(registry="ghcr.io", rootless=True)
    >>> client = DockerBuildClient(config)
    >>>
    >>> health = await client.check_docker_health()
    >>> if health.available:
    ...     result = await client.build_image(Path("./project"), tag="myapp:latest")
    ...     if result.success:
    ...         print(f"Built image: {result.image_id}")
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import AsyncIterator
from enum import Enum
from pathlib import Path
from typing import Any

from docker.errors import APIError, BuildError, DockerException, ImageNotFound
from pydantic import BaseModel, Field

import docker
from forgemaster.config import DockerConfig
from forgemaster.logging import get_logger

# Regex patterns for validation
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z\-]+(?:\.[0-9A-Za-z\-]+)*))?"
    r"(?:\+(?P<build>[0-9A-Za-z\-]+(?:\.[0-9A-Za-z\-]+)*))?$"
)


class BuildStatus(str, Enum):
    """Status of a Docker image build operation.

    Attributes:
        PENDING: Build has been queued but not started
        BUILDING: Build is currently in progress
        SUCCEEDED: Build completed successfully
        FAILED: Build encountered an error
        CANCELLED: Build was cancelled before completion
    """

    PENDING = "pending"
    BUILDING = "building"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BuildResult(BaseModel):
    """Result of a Docker image build operation.

    Attributes:
        image_id: Docker image ID (sha256 hash)
        tags: List of tags applied to the built image
        build_log: Collected build log lines
        duration_seconds: Total build time in seconds
        success: Whether the build completed successfully
        error: Error message if build failed, None otherwise
        status: Current build status
    """

    image_id: str = Field(default="", description="Docker image ID")
    tags: list[str] = Field(default_factory=list, description="Image tags")
    build_log: list[str] = Field(default_factory=list, description="Build log lines")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Build duration")
    success: bool = Field(default=False, description="Build success flag")
    error: str | None = Field(default=None, description="Error message if failed")
    status: BuildStatus = Field(default=BuildStatus.PENDING, description="Build status")


class BuildLogEntry(BaseModel):
    """A single entry from the Docker build log stream.

    Attributes:
        stream: Standard output line from the build process
        error: Error message if this entry represents an error
        status: Status message (e.g., downloading, extracting)
        progress: Progress indicator string
        aux: Auxiliary data (e.g., image ID after successful build)
    """

    stream: str | None = Field(default=None, description="Build output line")
    error: str | None = Field(default=None, description="Error detail")
    status: str | None = Field(default=None, description="Status message")
    progress: str | None = Field(default=None, description="Progress indicator")
    aux: dict[str, str] | None = Field(default=None, description="Auxiliary data")


class DockerHealth(BaseModel):
    """Health status of the Docker daemon.

    Attributes:
        available: Whether Docker daemon is reachable and responding
        rootless: Whether the daemon is running in rootless mode
        version: Docker engine version string
        api_version: Docker API version string
        error: Error message if health check failed
    """

    available: bool = Field(default=False, description="Daemon availability")
    rootless: bool = Field(default=False, description="Rootless mode detected")
    version: str | None = Field(default=None, description="Docker version")
    api_version: str | None = Field(default=None, description="API version")
    error: str | None = Field(default=None, description="Health check error")


class RootlessConfig(BaseModel):
    """Configuration for rootless Docker compatibility.

    Attributes:
        userns_mode: User namespace mode for rootless Docker
        security_opt: Security options for rootless containers
    """

    userns_mode: str = Field(default="keep-id", description="User namespace mode")
    security_opt: list[str] = Field(
        default_factory=lambda: ["label=disable"],
        description="Security options for rootless mode",
    )


class SemverTag(BaseModel):
    """Parsed semantic version tag components.

    Attributes:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        prerelease: Pre-release identifier (e.g., 'alpha.1', 'rc.2')
        build: Build metadata identifier
    """

    major: int = Field(ge=0, description="Major version")
    minor: int = Field(ge=0, description="Minor version")
    patch: int = Field(ge=0, description="Patch version")
    prerelease: str | None = Field(default=None, description="Pre-release identifier")
    build: str | None = Field(default=None, description="Build metadata")

    def to_string(self) -> str:
        """Convert back to a semver string.

        Returns:
            Semantic version string (e.g., '1.2.3-alpha+build.1')
        """
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease is not None:
            version += f"-{self.prerelease}"
        if self.build is not None:
            version += f"+{self.build}"
        return version


class DockerBuildClient:
    """High-level async Docker build client using docker-py.

    This class wraps the docker-py SDK with async compatibility, error handling,
    structured logging, and FORGEMASTER-specific conventions for image building
    and tagging.

    Attributes:
        config: Docker configuration from ForgemasterConfig
        logger: Structured logger instance
        rootless_config: Rootless Docker configuration
    """

    def __init__(self, config: DockerConfig) -> None:
        """Initialize DockerBuildClient with configuration.

        Args:
            config: Docker configuration settings

        The Docker client connection is deferred until first use to allow
        graceful handling of connection failures.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.rootless_config = RootlessConfig()
        self._client: docker.DockerClient | None = None

        self.logger.info(
            "docker_build_client_initialized",
            registry=config.registry,
            rootless=config.rootless,
            build_timeout=config.build_timeout_seconds,
        )

    def _get_client(self) -> docker.DockerClient:
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
                    self._client = docker.DockerClient(base_url=docker_host)
                elif self.config.rootless:
                    # Try rootless socket paths
                    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
                    rootless_socket = f"unix://{xdg_runtime}/docker.sock"
                    try:
                        self._client = docker.DockerClient(base_url=rootless_socket)
                    except DockerException:
                        # Fall back to default
                        self._client = docker.DockerClient.from_env()
                else:
                    self._client = docker.DockerClient.from_env()

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

    async def check_docker_health(self) -> DockerHealth:
        """Check Docker daemon health and capabilities.

        Returns:
            DockerHealth instance with daemon status information
        """
        try:
            client = await asyncio.to_thread(self._get_client)
            version_info: dict[str, Any] = await asyncio.to_thread(client.version)

            # Detect rootless mode from security options or Docker context
            docker_host = os.environ.get("DOCKER_HOST", "")
            is_rootless = (
                "rootless" in docker_host or "/run/user/" in docker_host or self.config.rootless
            )

            health = DockerHealth(
                available=True,
                rootless=is_rootless,
                version=str(version_info.get("Version", "")),
                api_version=str(version_info.get("ApiVersion", "")),
            )

            self.logger.info(
                "docker_health_check_passed",
                version=health.version,
                api_version=health.api_version,
                rootless=health.rootless,
            )

            return health

        except Exception as e:
            error_msg = str(e)
            self.logger.warning(
                "docker_health_check_failed",
                error=error_msg,
                error_type=type(e).__name__,
            )
            return DockerHealth(
                available=False,
                rootless=False,
                error=error_msg,
            )

    async def build_image(
        self,
        path: Path,
        dockerfile: str = "Dockerfile",
        tag: str | None = None,
        build_args: dict[str, str] | None = None,
        no_cache: bool = False,
    ) -> BuildResult:
        """Build a Docker image from a Dockerfile.

        Args:
            path: Path to the build context directory
            dockerfile: Name of the Dockerfile (relative to path)
            tag: Tag for the built image (e.g., 'myapp:latest')
            build_args: Build-time variables to pass to Docker
            no_cache: If True, do not use cache when building
            build_timeout: Optional timeout override in seconds

        Returns:
            BuildResult with image ID, tags, logs, and status

        Raises:
            FileNotFoundError: If path or Dockerfile does not exist
        """
        start_time = time.monotonic()

        self.logger.info(
            "docker_build_started",
            path=str(path),
            dockerfile=dockerfile,
            tag=tag,
            no_cache=no_cache,
            build_args=list(build_args.keys()) if build_args else [],
        )

        # Validate build context exists
        if not path.exists():
            error_msg = f"Build context path does not exist: {path}"
            self.logger.error("docker_build_path_not_found", path=str(path))
            return BuildResult(
                success=False,
                error=error_msg,
                status=BuildStatus.FAILED,
                duration_seconds=time.monotonic() - start_time,
            )

        # Validate Dockerfile exists
        dockerfile_path = path / dockerfile
        if not dockerfile_path.exists():
            error_msg = f"Dockerfile not found: {dockerfile_path}"
            self.logger.error(
                "docker_build_dockerfile_not_found",
                dockerfile=str(dockerfile_path),
            )
            return BuildResult(
                success=False,
                error=error_msg,
                status=BuildStatus.FAILED,
                duration_seconds=time.monotonic() - start_time,
            )

        try:
            client = await asyncio.to_thread(self._get_client)

            # Build the image
            build_kwargs: dict[str, Any] = {
                "path": str(path),
                "dockerfile": dockerfile,
                "nocache": no_cache,
                "rm": True,
                "timeout": self.config.build_timeout_seconds,
            }
            if tag is not None:
                build_kwargs["tag"] = tag
            if build_args is not None:
                build_kwargs["buildargs"] = build_args

            image, build_logs_raw = await asyncio.to_thread(client.images.build, **build_kwargs)

            # Collect build logs
            build_log: list[str] = []
            for log_entry in build_logs_raw:
                if isinstance(log_entry, dict):
                    line = log_entry.get("stream", "")
                    if line and line.strip():
                        build_log.append(line.strip())

            duration = time.monotonic() - start_time
            tags = image.tags if image.tags else []
            image_id = image.id or ""

            self.logger.info(
                "docker_build_succeeded",
                image_id=image_id[:20],
                tags=tags,
                duration_seconds=round(duration, 2),
                log_lines=len(build_log),
            )

            return BuildResult(
                image_id=image_id,
                tags=tags,
                build_log=build_log,
                duration_seconds=duration,
                success=True,
                status=BuildStatus.SUCCEEDED,
            )

        except BuildError as e:
            duration = time.monotonic() - start_time
            build_log = []
            for log_entry in e.build_log:
                if isinstance(log_entry, dict):
                    line = log_entry.get("stream", "") or log_entry.get("error", "")
                    if line and line.strip():
                        build_log.append(line.strip())

            self.logger.error(
                "docker_build_failed",
                error=str(e),
                duration_seconds=round(duration, 2),
                log_lines=len(build_log),
            )

            return BuildResult(
                build_log=build_log,
                duration_seconds=duration,
                success=False,
                error=str(e),
                status=BuildStatus.FAILED,
            )

        except APIError as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "docker_build_api_error",
                error=str(e),
                status_code=e.status_code,
                duration_seconds=round(duration, 2),
            )
            return BuildResult(
                duration_seconds=duration,
                success=False,
                error=str(e),
                status=BuildStatus.FAILED,
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            error_msg = f"Build timed out after {self.config.build_timeout_seconds} seconds"
            self.logger.error(
                "docker_build_timeout",
                timeout_seconds=self.config.build_timeout_seconds,
                duration_seconds=round(duration, 2),
            )
            return BuildResult(
                duration_seconds=duration,
                success=False,
                error=error_msg,
                status=BuildStatus.CANCELLED,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "docker_build_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                duration_seconds=round(duration, 2),
            )
            return BuildResult(
                duration_seconds=duration,
                success=False,
                error=str(e),
                status=BuildStatus.FAILED,
            )

    async def build_image_streaming(
        self,
        path: Path,
        dockerfile: str = "Dockerfile",
        tag: str | None = None,
        build_args: dict[str, str] | None = None,
        no_cache: bool = False,
    ) -> AsyncIterator[BuildLogEntry]:
        """Build a Docker image with streaming log output.

        This method yields build log entries as they arrive, enabling real-time
        progress monitoring of the build process.

        Args:
            path: Path to the build context directory
            dockerfile: Name of the Dockerfile (relative to path)
            tag: Tag for the built image
            build_args: Build-time variables to pass to Docker
            no_cache: If True, do not use cache when building

        Yields:
            BuildLogEntry instances for each build event

        Raises:
            FileNotFoundError: If path or Dockerfile does not exist
        """
        self.logger.info(
            "docker_build_streaming_started",
            path=str(path),
            dockerfile=dockerfile,
            tag=tag,
        )

        # Validate paths
        if not path.exists():
            yield BuildLogEntry(error=f"Build context path does not exist: {path}")
            return

        dockerfile_path = path / dockerfile
        if not dockerfile_path.exists():
            yield BuildLogEntry(error=f"Dockerfile not found: {dockerfile_path}")
            return

        try:
            client = await asyncio.to_thread(self._get_client)

            build_kwargs: dict[str, Any] = {
                "path": str(path),
                "dockerfile": dockerfile,
                "nocache": no_cache,
                "rm": True,
                "decode": True,
            }
            if tag is not None:
                build_kwargs["tag"] = tag
            if build_args is not None:
                build_kwargs["buildargs"] = build_args

            # Use low-level API for streaming
            response = await asyncio.to_thread(client.api.build, **build_kwargs)

            entry_count = 0
            for chunk in response:
                if isinstance(chunk, dict):
                    entry = BuildLogEntry(
                        stream=chunk.get("stream"),
                        error=chunk.get("error"),
                        status=chunk.get("status"),
                        progress=chunk.get("progress"),
                        aux=(
                            {str(k): str(v) for k, v in chunk["aux"].items()}
                            if "aux" in chunk and isinstance(chunk["aux"], dict)
                            else None
                        ),
                    )
                    entry_count += 1
                    yield entry

                    if entry.error:
                        self.logger.error(
                            "docker_build_stream_error",
                            error=entry.error,
                        )

            self.logger.info(
                "docker_build_streaming_complete",
                entries=entry_count,
            )

        except Exception as e:
            self.logger.error(
                "docker_build_streaming_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            yield BuildLogEntry(error=str(e))

    def generate_sha_tag(
        self,
        git_sha: str,
        registry: str | None = None,
        repository: str = "",
    ) -> str:
        """Generate a Docker image tag from a git commit SHA.

        The tag format is: {registry}/{repository}:{sha[:12]}

        Args:
            git_sha: Full 40-character git commit SHA hex string
            registry: Docker registry URL (defaults to config registry)
            repository: Repository name/path within the registry

        Returns:
            Formatted Docker image tag string

        Raises:
            ValueError: If git_sha is not a valid 40-character hex string
        """
        if not _SHA_PATTERN.match(git_sha):
            raise ValueError(f"Invalid git SHA: '{git_sha}'. Expected 40-character hex string.")

        effective_registry = registry if registry is not None else self.config.registry
        short_sha = git_sha[:12]

        if repository:
            tag = f"{effective_registry}/{repository}:{short_sha}"
        else:
            tag = f"{effective_registry}:{short_sha}"

        self.logger.debug(
            "sha_tag_generated",
            git_sha=git_sha[:12],
            tag=tag,
        )

        return tag

    def parse_semver(self, version: str) -> SemverTag:
        """Parse a semantic version string into components.

        Args:
            version: Semantic version string (e.g., '1.2.3', '1.0.0-alpha+build.1')

        Returns:
            SemverTag with parsed version components

        Raises:
            ValueError: If version string does not match semver format
        """
        match = _SEMVER_PATTERN.match(version)
        if not match:
            raise ValueError(
                f"Invalid semver format: '{version}'. "
                f"Expected format: major.minor.patch[-prerelease][+build]"
            )

        return SemverTag(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
            build=match.group("build"),
        )

    def generate_semver_tag(
        self,
        version: str,
        registry: str | None = None,
        repository: str = "",
    ) -> str:
        """Generate a Docker image tag from a semantic version string.

        The tag format is: {registry}/{repository}:{version}

        Args:
            version: Semantic version string (e.g., '1.2.3')
            registry: Docker registry URL (defaults to config registry)
            repository: Repository name/path within the registry

        Returns:
            Formatted Docker image tag string

        Raises:
            ValueError: If version is not a valid semver string
        """
        # Validate semver format by parsing
        self.parse_semver(version)

        effective_registry = registry if registry is not None else self.config.registry

        if repository:
            tag = f"{effective_registry}/{repository}:{version}"
        else:
            tag = f"{effective_registry}:{version}"

        self.logger.debug(
            "semver_tag_generated",
            version=version,
            tag=tag,
        )

        return tag

    async def tag_image(self, image_id: str, tag: str) -> bool:
        """Apply a tag to an existing Docker image.

        Args:
            image_id: Docker image ID or existing tag
            tag: New tag to apply (format: repository:tag or registry/repo:tag)

        Returns:
            True if tagging succeeded, False otherwise
        """
        try:
            client = await asyncio.to_thread(self._get_client)

            # Parse tag into repository and tag components
            if ":" in tag:
                repo_part, tag_part = tag.rsplit(":", 1)
            else:
                repo_part = tag
                tag_part = "latest"

            image = await asyncio.to_thread(client.images.get, image_id)
            result: bool = await asyncio.to_thread(image.tag, repository=repo_part, tag=tag_part)

            self.logger.info(
                "image_tagged",
                image_id=image_id[:20] if len(image_id) > 20 else image_id,
                tag=tag,
                success=result,
            )

            return result

        except ImageNotFound:
            self.logger.error(
                "image_not_found_for_tagging",
                image_id=image_id,
                tag=tag,
            )
            return False

        except APIError as e:
            self.logger.error(
                "image_tag_api_error",
                image_id=image_id,
                tag=tag,
                error=str(e),
            )
            return False

        except Exception as e:
            self.logger.error(
                "image_tag_failed",
                image_id=image_id,
                tag=tag,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def update_latest_tag(
        self,
        image_id: str,
        registry: str | None = None,
        repository: str = "",
    ) -> bool:
        """Tag an image as 'latest' in addition to any existing tags.

        Args:
            image_id: Docker image ID or existing tag
            registry: Docker registry URL (defaults to config registry)
            repository: Repository name/path within the registry

        Returns:
            True if the latest tag was applied successfully
        """
        effective_registry = registry if registry is not None else self.config.registry

        if repository:
            latest_tag = f"{effective_registry}/{repository}:latest"
        else:
            latest_tag = f"{effective_registry}:latest"

        self.logger.info(
            "updating_latest_tag",
            image_id=image_id[:20] if len(image_id) > 20 else image_id,
            latest_tag=latest_tag,
        )

        return await self.tag_image(image_id, latest_tag)

    async def get_image_tags(self, image_id: str) -> list[str]:
        """Get all tags associated with a Docker image.

        Args:
            image_id: Docker image ID or existing tag

        Returns:
            List of tag strings for the image, empty list if image not found
        """
        try:
            client = await asyncio.to_thread(self._get_client)
            image = await asyncio.to_thread(client.images.get, image_id)
            tags: list[str] = image.tags if image.tags else []

            self.logger.debug(
                "image_tags_retrieved",
                image_id=image_id[:20] if len(image_id) > 20 else image_id,
                tag_count=len(tags),
                tags=tags,
            )

            return tags

        except ImageNotFound:
            self.logger.warning(
                "image_not_found_for_tag_listing",
                image_id=image_id,
            )
            return []

        except Exception as e:
            self.logger.error(
                "get_image_tags_failed",
                image_id=image_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

    async def close(self) -> None:
        """Close the Docker client connection.

        Safe to call multiple times or if the client was never connected.
        """
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
                self.logger.info("docker_client_closed")
            except Exception as e:
                self.logger.warning(
                    "docker_client_close_error",
                    error=str(e),
                )
            finally:
                self._client = None
