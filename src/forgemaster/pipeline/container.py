"""Container lifecycle management for FORGEMASTER.

This module provides high-level async interfaces for Docker container operations and
Docker Compose service management. Wraps docker-py and docker-compose CLI with async
compatibility, comprehensive error handling, and structured logging.

ContainerManager handles individual container operations (start, stop, restart, status).
ComposeManager handles Docker Compose service operations (restart services, get status).

Example usage:
    >>> from forgemaster.config import DockerConfig
    >>> from forgemaster.pipeline.container import ContainerManager, ComposeManager
    >>>
    >>> config = DockerConfig(registry="ghcr.io", rootless=True)
    >>> manager = ContainerManager(config)
    >>>
    >>> # Stop a container
    >>> action = await manager.stop_container("my-container-id")
    >>> if action.success:
    ...     print(f"Stopped: {action.container_id}")
    >>>
    >>> # Manage compose services
    >>> compose = ComposeManager(config, Path("docker-compose.yml"))
    >>> result = await compose.restart_service("web")
"""

from __future__ import annotations

import asyncio
import os
import time
from enum import Enum
from pathlib import Path

from docker.errors import APIError, DockerException, NotFound
from pydantic import BaseModel, Field

import docker
from forgemaster.config import DockerConfig
from forgemaster.logging import get_logger


class ContainerStatus(str, Enum):
    """Status of a Docker container.

    Attributes:
        RUNNING: Container is running
        STOPPED: Container has been stopped
        PAUSED: Container is paused
        RESTARTING: Container is restarting
        EXITED: Container has exited
        DEAD: Container is dead (non-recoverable error state)
        CREATED: Container has been created but not started
        REMOVING: Container is being removed
    """

    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"
    RESTARTING = "restarting"
    EXITED = "exited"
    DEAD = "dead"
    CREATED = "created"
    REMOVING = "removing"


class ContainerAction(BaseModel):
    """Result of a container lifecycle operation.

    Attributes:
        success: Whether the operation completed successfully
        container_id: Docker container ID or name
        action: Action performed (stop, start, restart)
        previous_status: Container status before the operation
        current_status: Container status after the operation
        error: Error message if operation failed
        duration_seconds: Time taken for the operation
    """

    success: bool = Field(description="Operation success flag")
    container_id: str = Field(description="Container ID or name")
    action: str = Field(description="Action performed")
    previous_status: str | None = Field(default=None, description="Status before action")
    current_status: str | None = Field(default=None, description="Status after action")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Operation duration")


class ContainerInfo(BaseModel):
    """Information about a Docker container.

    Attributes:
        container_id: Docker container ID
        name: Container name
        image: Image name and tag
        status: Current container status
        health: Health check status (if configured)
        ports: Port mappings (container_port -> host_port)
        created_at: Container creation timestamp
        started_at: Container start timestamp
    """

    container_id: str = Field(description="Container ID")
    name: str = Field(description="Container name")
    image: str = Field(description="Image name")
    status: ContainerStatus = Field(description="Container status")
    health: str | None = Field(default=None, description="Health status")
    ports: dict[str, str | None] = Field(default_factory=dict, description="Port mappings")
    created_at: str | None = Field(default=None, description="Creation timestamp")
    started_at: str | None = Field(default=None, description="Start timestamp")


class ComposeAction(BaseModel):
    """Result of a Docker Compose service operation.

    Attributes:
        success: Whether the operation completed successfully
        service_name: Name of the compose service
        action: Action performed (restart_service, restart_all)
        containers_affected: List of container IDs/names affected
        error: Error message if operation failed
        duration_seconds: Time taken for the operation
    """

    success: bool = Field(description="Operation success flag")
    service_name: str = Field(description="Service name")
    action: str = Field(description="Action performed")
    containers_affected: list[str] = Field(default_factory=list, description="Affected containers")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Operation duration")


class ContainerManager:
    """High-level async Docker container lifecycle manager.

    This class wraps docker-py container operations with async compatibility,
    error handling, structured logging, and FORGEMASTER conventions.

    Attributes:
        config: Docker configuration from ForgemasterConfig
        logger: Structured logger instance
    """

    def __init__(self, config: DockerConfig) -> None:
        """Initialize ContainerManager with configuration.

        Args:
            config: Docker configuration settings

        The Docker client connection is deferred until first use.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._client: docker.DockerClient | None = None

        self.logger.info(
            "container_manager_initialized",
            rootless=config.rootless,
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
                docker_host = os.environ.get("DOCKER_HOST")
                if docker_host:
                    self._client = docker.DockerClient(base_url=docker_host)
                elif self.config.rootless:
                    # Try rootless socket paths (Unix-like systems only)
                    try:
                        if hasattr(os, "getuid"):
                            uid = os.getuid()
                            xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")
                            rootless_socket = f"unix://{xdg_runtime}/docker.sock"
                            self._client = docker.DockerClient(base_url=rootless_socket)
                        else:
                            # Windows - fall back to default
                            self._client = docker.DockerClient.from_env()
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
                )
                raise

        return self._client

    async def stop_container(self, container_id: str, timeout: int = 30) -> ContainerAction:
        """Stop a running Docker container.

        Args:
            container_id: Container ID or name to stop
            timeout: Seconds to wait before forcefully killing the container

        Returns:
            ContainerAction with operation result and status information
        """
        start_time = time.monotonic()

        self.logger.info(
            "stopping_container",
            container_id=container_id,
            timeout=timeout,
        )

        try:
            client = await asyncio.to_thread(self._get_client)
            container = await asyncio.to_thread(client.containers.get, container_id)

            # Get previous status
            await asyncio.to_thread(container.reload)
            previous_status = container.status

            # Already stopped?
            if previous_status in (ContainerStatus.STOPPED.value, ContainerStatus.EXITED.value):
                duration = time.monotonic() - start_time
                self.logger.info(
                    "container_already_stopped",
                    container_id=container_id,
                    status=previous_status,
                )
                return ContainerAction(
                    success=True,
                    container_id=container_id,
                    action="stop",
                    previous_status=previous_status,
                    current_status=previous_status,
                    duration_seconds=duration,
                )

            # Stop the container
            await asyncio.to_thread(container.stop, timeout=timeout)

            # Get new status
            await asyncio.to_thread(container.reload)
            current_status = container.status

            duration = time.monotonic() - start_time

            self.logger.info(
                "container_stopped",
                container_id=container_id,
                previous_status=previous_status,
                current_status=current_status,
                duration_seconds=round(duration, 2),
            )

            return ContainerAction(
                success=True,
                container_id=container_id,
                action="stop",
                previous_status=previous_status,
                current_status=current_status,
                duration_seconds=duration,
            )

        except NotFound:
            duration = time.monotonic() - start_time
            error_msg = f"Container not found: {container_id}"
            self.logger.error(
                "container_not_found",
                container_id=container_id,
                action="stop",
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="stop",
                error=error_msg,
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            error_msg = f"Stop operation timed out after {timeout} seconds"
            self.logger.error(
                "container_stop_timeout",
                container_id=container_id,
                timeout=timeout,
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="stop",
                error=error_msg,
                duration_seconds=duration,
            )

        except APIError as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "container_stop_api_error",
                container_id=container_id,
                error=str(e),
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="stop",
                error=str(e),
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "container_stop_failed",
                container_id=container_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="stop",
                error=str(e),
                duration_seconds=duration,
            )

    async def start_container(self, container_id: str) -> ContainerAction:
        """Start a stopped Docker container.

        Args:
            container_id: Container ID or name to start

        Returns:
            ContainerAction with operation result and status information
        """
        start_time = time.monotonic()

        self.logger.info(
            "starting_container",
            container_id=container_id,
        )

        try:
            client = await asyncio.to_thread(self._get_client)
            container = await asyncio.to_thread(client.containers.get, container_id)

            # Get previous status
            await asyncio.to_thread(container.reload)
            previous_status = container.status

            # Already running?
            if previous_status == ContainerStatus.RUNNING.value:
                duration = time.monotonic() - start_time
                self.logger.info(
                    "container_already_running",
                    container_id=container_id,
                )
                return ContainerAction(
                    success=True,
                    container_id=container_id,
                    action="start",
                    previous_status=previous_status,
                    current_status=previous_status,
                    duration_seconds=duration,
                )

            # Start the container
            await asyncio.to_thread(container.start)

            # Get new status
            await asyncio.to_thread(container.reload)
            current_status = container.status

            duration = time.monotonic() - start_time

            self.logger.info(
                "container_started",
                container_id=container_id,
                previous_status=previous_status,
                current_status=current_status,
                duration_seconds=round(duration, 2),
            )

            return ContainerAction(
                success=True,
                container_id=container_id,
                action="start",
                previous_status=previous_status,
                current_status=current_status,
                duration_seconds=duration,
            )

        except NotFound:
            duration = time.monotonic() - start_time
            error_msg = f"Container not found: {container_id}"
            self.logger.error(
                "container_not_found",
                container_id=container_id,
                action="start",
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="start",
                error=error_msg,
                duration_seconds=duration,
            )

        except APIError as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "container_start_api_error",
                container_id=container_id,
                error=str(e),
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="start",
                error=str(e),
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "container_start_failed",
                container_id=container_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="start",
                error=str(e),
                duration_seconds=duration,
            )

    async def restart_container(self, container_id: str, timeout: int = 30) -> ContainerAction:
        """Restart a Docker container.

        Args:
            container_id: Container ID or name to restart
            timeout: Seconds to wait before forcefully killing during restart

        Returns:
            ContainerAction with operation result and status information
        """
        start_time = time.monotonic()

        self.logger.info(
            "restarting_container",
            container_id=container_id,
            timeout=timeout,
        )

        try:
            client = await asyncio.to_thread(self._get_client)
            container = await asyncio.to_thread(client.containers.get, container_id)

            # Get previous status
            await asyncio.to_thread(container.reload)
            previous_status = container.status

            # Restart the container
            await asyncio.to_thread(container.restart, timeout=timeout)

            # Get new status
            await asyncio.to_thread(container.reload)
            current_status = container.status

            duration = time.monotonic() - start_time

            self.logger.info(
                "container_restarted",
                container_id=container_id,
                previous_status=previous_status,
                current_status=current_status,
                duration_seconds=round(duration, 2),
            )

            return ContainerAction(
                success=True,
                container_id=container_id,
                action="restart",
                previous_status=previous_status,
                current_status=current_status,
                duration_seconds=duration,
            )

        except NotFound:
            duration = time.monotonic() - start_time
            error_msg = f"Container not found: {container_id}"
            self.logger.error(
                "container_not_found",
                container_id=container_id,
                action="restart",
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="restart",
                error=error_msg,
                duration_seconds=duration,
            )

        except APIError as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "container_restart_api_error",
                container_id=container_id,
                error=str(e),
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="restart",
                error=str(e),
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            self.logger.error(
                "container_restart_failed",
                container_id=container_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return ContainerAction(
                success=False,
                container_id=container_id,
                action="restart",
                error=str(e),
                duration_seconds=duration,
            )

    async def get_container_status(self, container_id: str) -> ContainerInfo:
        """Get detailed status information for a container.

        Args:
            container_id: Container ID or name

        Returns:
            ContainerInfo with detailed container information

        Raises:
            NotFound: If container does not exist
        """
        self.logger.debug(
            "getting_container_status",
            container_id=container_id,
        )

        client = await asyncio.to_thread(self._get_client)
        container = await asyncio.to_thread(client.containers.get, container_id)
        await asyncio.to_thread(container.reload)

        attrs = container.attrs

        # Parse status
        status_str = attrs.get("State", {}).get("Status", "unknown")
        try:
            status = ContainerStatus(status_str.lower())
        except ValueError:
            status = ContainerStatus.EXITED

        # Parse ports
        ports: dict[str, str | None] = {}
        port_bindings = attrs.get("NetworkSettings", {}).get("Ports", {})
        if port_bindings:
            for container_port, host_bindings in port_bindings.items():
                if host_bindings:
                    # Take first binding
                    host_port = host_bindings[0].get("HostPort")
                    ports[container_port] = host_port
                else:
                    ports[container_port] = None

        # Parse timestamps
        created_at = attrs.get("Created")
        started_at = attrs.get("State", {}).get("StartedAt")

        # Parse health
        health_status = attrs.get("State", {}).get("Health", {}).get("Status")

        info = ContainerInfo(
            container_id=container.id[:12],
            name=container.name,
            image=attrs.get("Config", {}).get("Image", "unknown"),
            status=status,
            health=health_status,
            ports=ports,
            created_at=created_at,
            started_at=started_at,
        )

        self.logger.debug(
            "container_status_retrieved",
            container_id=container_id,
            status=info.status,
            health=info.health,
        )

        return info

    async def close(self) -> None:
        """Close the Docker client connection.

        Safe to call multiple times or if the client was never connected.
        """
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
                self.logger.info("container_manager_closed")
            except Exception as e:
                self.logger.warning(
                    "container_manager_close_error",
                    error=str(e),
                )
            finally:
                self._client = None


class ComposeManager:
    """High-level async Docker Compose service manager.

    This class manages Docker Compose services using the docker-compose CLI
    for reliability. Provides async wrappers around subprocess calls with
    comprehensive error handling and structured logging.

    Attributes:
        config: Docker configuration from ForgemasterConfig
        compose_file: Path to docker-compose.yml file
        logger: Structured logger instance
    """

    def __init__(self, config: DockerConfig, compose_file: Path) -> None:
        """Initialize ComposeManager with configuration.

        Args:
            config: Docker configuration settings
            compose_file: Path to docker-compose.yml file
        """
        self.config = config
        self.compose_file = compose_file
        self.logger = get_logger(__name__)

        self.logger.info(
            "compose_manager_initialized",
            compose_file=str(compose_file),
            rootless=config.rootless,
        )

    async def _run_compose_command(self, *args: str, timeout: int = 60) -> tuple[bool, str, str]:
        """Run a docker-compose command via subprocess.

        Args:
            *args: Command arguments to pass to docker compose
            timeout: Command timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        # Build command
        cmd = [
            "docker",
            "compose",
            "-f",
            str(self.compose_file),
            *args,
        ]

        self.logger.debug(
            "running_compose_command",
            command=" ".join(cmd),
            timeout=timeout,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            success = proc.returncode == 0

            if not success:
                self.logger.error(
                    "compose_command_failed",
                    command=" ".join(cmd),
                    returncode=proc.returncode,
                    stderr=stderr[:500],
                )
            else:
                self.logger.debug(
                    "compose_command_succeeded",
                    command=" ".join(cmd),
                )

            return success, stdout, stderr

        except asyncio.TimeoutError:
            self.logger.error(
                "compose_command_timeout",
                command=" ".join(cmd),
                timeout=timeout,
            )
            return False, "", f"Command timed out after {timeout} seconds"

        except FileNotFoundError:
            error_msg = "docker compose command not found. Is Docker Compose installed?"
            self.logger.error("compose_command_not_found")
            return False, "", error_msg

        except Exception as e:
            self.logger.error(
                "compose_command_error",
                command=" ".join(cmd),
                error=str(e),
                error_type=type(e).__name__,
            )
            return False, "", str(e)

    async def restart_service(self, service_name: str, timeout: int = 30) -> ComposeAction:
        """Restart a Docker Compose service.

        Args:
            service_name: Name of the service to restart
            timeout: Seconds to wait for restart to complete

        Returns:
            ComposeAction with operation result and affected containers
        """
        start_time = time.monotonic()

        self.logger.info(
            "restarting_compose_service",
            service_name=service_name,
            compose_file=str(self.compose_file),
            timeout=timeout,
        )

        # Check if compose file exists
        if not self.compose_file.exists():
            duration = time.monotonic() - start_time
            error_msg = f"Compose file not found: {self.compose_file}"
            self.logger.error(
                "compose_file_not_found",
                compose_file=str(self.compose_file),
            )
            return ComposeAction(
                success=False,
                service_name=service_name,
                action="restart_service",
                error=error_msg,
                duration_seconds=duration,
            )

        # Get container IDs before restart
        success, stdout, stderr = await self._run_compose_command(
            "ps", "-q", service_name, timeout=timeout
        )

        containers_before = [line.strip() for line in stdout.splitlines() if line.strip()]

        # Restart the service
        success, stdout, stderr = await self._run_compose_command(
            "restart", "-t", str(timeout), service_name, timeout=timeout + 10
        )

        duration = time.monotonic() - start_time

        if not success:
            # Check if service doesn't exist
            if "no such service" in stderr.lower():
                error_msg = f"Service not found: {service_name}"
            else:
                error_msg = stderr or "Restart command failed"

            self.logger.error(
                "compose_service_restart_failed",
                service_name=service_name,
                error=error_msg,
            )

            return ComposeAction(
                success=False,
                service_name=service_name,
                action="restart_service",
                error=error_msg,
                duration_seconds=duration,
            )

        # Get container IDs after restart
        success_after, stdout_after, _ = await self._run_compose_command(
            "ps", "-q", service_name, timeout=timeout
        )

        containers_after = [line.strip() for line in stdout_after.splitlines() if line.strip()]

        containers_affected = list(set(containers_before + containers_after))

        self.logger.info(
            "compose_service_restarted",
            service_name=service_name,
            containers_affected=len(containers_affected),
            duration_seconds=round(duration, 2),
        )

        return ComposeAction(
            success=True,
            service_name=service_name,
            action="restart_service",
            containers_affected=containers_affected,
            duration_seconds=duration,
        )

    async def get_service_status(self, service_name: str) -> list[ContainerInfo]:
        """Get status information for all containers in a service.

        Args:
            service_name: Name of the service

        Returns:
            List of ContainerInfo for each container in the service

        Raises:
            FileNotFoundError: If compose file doesn't exist
        """
        self.logger.debug(
            "getting_compose_service_status",
            service_name=service_name,
        )

        if not self.compose_file.exists():
            raise FileNotFoundError(f"Compose file not found: {self.compose_file}")

        # Get container IDs for the service
        success, stdout, stderr = await self._run_compose_command("ps", "-q", service_name)

        if not success:
            self.logger.warning(
                "compose_service_status_failed",
                service_name=service_name,
                error=stderr,
            )
            return []

        container_ids = [line.strip() for line in stdout.splitlines() if line.strip()]

        if not container_ids:
            self.logger.debug(
                "no_containers_for_service",
                service_name=service_name,
            )
            return []

        # Get detailed info for each container
        container_manager = ContainerManager(self.config)
        results: list[ContainerInfo] = []

        for container_id in container_ids:
            try:
                info = await container_manager.get_container_status(container_id)
                results.append(info)
            except NotFound:
                self.logger.warning(
                    "compose_container_not_found",
                    container_id=container_id,
                    service_name=service_name,
                )
                continue
            except Exception as e:
                self.logger.error(
                    "compose_container_status_error",
                    container_id=container_id,
                    service_name=service_name,
                    error=str(e),
                )
                continue

        await container_manager.close()

        self.logger.debug(
            "compose_service_status_retrieved",
            service_name=service_name,
            container_count=len(results),
        )

        return results

    async def restart_all_services(self, timeout: int = 30) -> list[ComposeAction]:
        """Restart all services defined in the compose file.

        Args:
            timeout: Seconds to wait for each service restart

        Returns:
            List of ComposeAction results, one per service
        """
        start_time = time.monotonic()

        self.logger.info(
            "restarting_all_compose_services",
            compose_file=str(self.compose_file),
            timeout=timeout,
        )

        if not self.compose_file.exists():
            duration = time.monotonic() - start_time
            error_msg = f"Compose file not found: {self.compose_file}"
            self.logger.error(
                "compose_file_not_found",
                compose_file=str(self.compose_file),
            )
            return [
                ComposeAction(
                    success=False,
                    service_name="all",
                    action="restart_all",
                    error=error_msg,
                    duration_seconds=duration,
                )
            ]

        # Get all service names
        success, stdout, stderr = await self._run_compose_command(
            "config", "--services", timeout=10
        )

        if not success:
            duration = time.monotonic() - start_time
            error_msg = f"Failed to list services: {stderr}"
            self.logger.error(
                "compose_list_services_failed",
                error=error_msg,
            )
            return [
                ComposeAction(
                    success=False,
                    service_name="all",
                    action="restart_all",
                    error=error_msg,
                    duration_seconds=duration,
                )
            ]

        service_names = [line.strip() for line in stdout.splitlines() if line.strip()]

        if not service_names:
            duration = time.monotonic() - start_time
            self.logger.info(
                "no_services_to_restart",
                compose_file=str(self.compose_file),
            )
            return []

        # Restart each service
        results: list[ComposeAction] = []
        for service_name in service_names:
            result = await self.restart_service(service_name, timeout=timeout)
            results.append(result)

        duration = time.monotonic() - start_time

        success_count = sum(1 for r in results if r.success)
        self.logger.info(
            "all_compose_services_restarted",
            total_services=len(results),
            successful=success_count,
            failed=len(results) - success_count,
            duration_seconds=round(duration, 2),
        )

        return results
