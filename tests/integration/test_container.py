"""Integration tests for container lifecycle management.

These tests verify ContainerManager and ComposeManager functionality using
mocked docker-py and subprocess calls. Tests cover container operations
(start, stop, restart, status) and Docker Compose service management.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from docker.errors import APIError, NotFound

from forgemaster.config import DockerConfig
from forgemaster.pipeline.container import (
    ComposeAction,
    ComposeManager,
    ContainerAction,
    ContainerInfo,
    ContainerManager,
    ContainerStatus,
)


@pytest.fixture
def docker_config() -> DockerConfig:
    """Create a test Docker configuration."""
    return DockerConfig(
        registry="test.registry.io",
        rootless=True,
        build_timeout_seconds=300,
    )


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """Create a mock Docker client."""
    client = MagicMock()
    client.version.return_value = {"Version": "24.0.0", "ApiVersion": "1.43"}
    return client


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock Docker container."""
    container = MagicMock()
    container.id = "abc123def456"
    container.name = "test-container"
    container.status = "running"
    container.tags = ["test:latest"]
    container.attrs = {
        "State": {
            "Status": "running",
            "StartedAt": "2026-02-05T12:00:00Z",
            "Health": {"Status": "healthy"},
        },
        "Created": "2026-02-05T11:00:00Z",
        "Config": {"Image": "test-image:latest"},
        "NetworkSettings": {
            "Ports": {
                "8080/tcp": [{"HostPort": "8080"}],
                "9000/tcp": None,
            }
        },
    }
    return container


class TestContainerManagerInitialization:
    """Test ContainerManager initialization."""

    def test_initialization(self, docker_config: DockerConfig) -> None:
        """Test ContainerManager initializes correctly."""
        manager = ContainerManager(docker_config)
        assert manager.config == docker_config
        assert manager._client is None

    @pytest.mark.asyncio
    async def test_client_connection_lazy(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test Docker client connection is lazy."""
        manager = ContainerManager(docker_config)
        assert manager._client is None

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            client = await asyncio.to_thread(manager._get_client)
            assert client is not None
            assert manager._client is not None


class TestContainerStopOperation:
    """Test container stop operations."""

    @pytest.mark.asyncio
    async def test_stop_container_success(
        self,
        docker_config: DockerConfig,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test successfully stopping a running container."""
        mock_container.status = "running"
        mock_docker_client.containers.get.return_value = mock_container

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.stop_container("test-container", timeout=10)

        assert result.success is True
        assert result.container_id == "test-container"
        assert result.action == "stop"
        assert result.previous_status == "running"
        assert result.error is None
        mock_container.stop.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_stop_container_not_found(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test stopping a non-existent container."""
        mock_docker_client.containers.get.side_effect = NotFound("Container not found")

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.stop_container("missing-container")

        assert result.success is False
        assert result.container_id == "missing-container"
        assert result.action == "stop"
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stop_container_already_stopped(
        self,
        docker_config: DockerConfig,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test stopping a container that is already stopped."""
        mock_container.status = "exited"
        mock_docker_client.containers.get.return_value = mock_container

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.stop_container("test-container")

        assert result.success is True
        assert result.previous_status == "exited"
        assert result.current_status == "exited"
        mock_container.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_container_api_error(
        self,
        docker_config: DockerConfig,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test stop operation with API error."""
        mock_container.status = "running"
        mock_container.stop.side_effect = APIError("API error occurred")
        mock_docker_client.containers.get.return_value = mock_container

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.stop_container("test-container")

        assert result.success is False
        assert result.action == "stop"
        assert "API error" in result.error


class TestContainerStartOperation:
    """Test container start operations."""

    @pytest.mark.asyncio
    async def test_start_container_success(
        self,
        docker_config: DockerConfig,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test successfully starting a stopped container."""
        mock_container.status = "exited"
        mock_docker_client.containers.get.return_value = mock_container

        # Simulate status change after start
        def start_side_effect() -> None:
            mock_container.status = "running"

        mock_container.start.side_effect = start_side_effect

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.start_container("test-container")

        assert result.success is True
        assert result.container_id == "test-container"
        assert result.action == "start"
        assert result.previous_status == "exited"
        assert result.error is None
        mock_container.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_container_not_found(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test starting a non-existent container."""
        mock_docker_client.containers.get.side_effect = NotFound("Container not found")

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.start_container("missing-container")

        assert result.success is False
        assert result.container_id == "missing-container"
        assert result.action == "start"
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_start_container_already_running(
        self,
        docker_config: DockerConfig,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test starting a container that is already running."""
        mock_container.status = "running"
        mock_docker_client.containers.get.return_value = mock_container

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.start_container("test-container")

        assert result.success is True
        assert result.previous_status == "running"
        assert result.current_status == "running"
        mock_container.start.assert_not_called()


class TestContainerRestartOperation:
    """Test container restart operations."""

    @pytest.mark.asyncio
    async def test_restart_container_success(
        self,
        docker_config: DockerConfig,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test successfully restarting a container."""
        mock_container.status = "running"
        mock_docker_client.containers.get.return_value = mock_container

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.restart_container("test-container", timeout=15)

        assert result.success is True
        assert result.container_id == "test-container"
        assert result.action == "restart"
        assert result.previous_status == "running"
        assert result.error is None
        mock_container.restart.assert_called_once_with(timeout=15)

    @pytest.mark.asyncio
    async def test_restart_container_not_found(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test restarting a non-existent container."""
        mock_docker_client.containers.get.side_effect = NotFound("Container not found")

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.restart_container("missing-container")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_restart_container_api_error(
        self,
        docker_config: DockerConfig,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test restart operation with API error."""
        mock_container.status = "running"
        mock_container.restart.side_effect = APIError("Restart failed")
        mock_docker_client.containers.get.return_value = mock_container

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            result = await manager.restart_container("test-container")

        assert result.success is False
        assert "Restart failed" in result.error


class TestContainerStatusOperation:
    """Test container status retrieval."""

    @pytest.mark.asyncio
    async def test_get_container_status_success(
        self,
        docker_config: DockerConfig,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test successfully getting container status."""
        mock_docker_client.containers.get.return_value = mock_container

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            info = await manager.get_container_status("test-container")

        assert isinstance(info, ContainerInfo)
        assert info.container_id == "abc123def456"
        assert info.name == "test-container"
        assert info.image == "test-image:latest"
        assert info.status == ContainerStatus.RUNNING
        assert info.health == "healthy"
        assert "8080/tcp" in info.ports
        assert info.ports["8080/tcp"] == "8080"

    @pytest.mark.asyncio
    async def test_get_container_status_not_found(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test getting status for non-existent container."""
        mock_docker_client.containers.get.side_effect = NotFound("Container not found")

        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            with pytest.raises(NotFound):
                await manager.get_container_status("missing-container")


class TestContainerActionModel:
    """Test ContainerAction Pydantic model."""

    def test_container_action_validation(self) -> None:
        """Test ContainerAction model validation."""
        action = ContainerAction(
            success=True,
            container_id="abc123",
            action="stop",
            previous_status="running",
            current_status="exited",
            duration_seconds=1.5,
        )

        assert action.success is True
        assert action.container_id == "abc123"
        assert action.action == "stop"
        assert action.error is None

    def test_container_action_with_error(self) -> None:
        """Test ContainerAction with error."""
        action = ContainerAction(
            success=False,
            container_id="abc123",
            action="start",
            error="Container failed to start",
            duration_seconds=0.5,
        )

        assert action.success is False
        assert action.error == "Container failed to start"


class TestContainerInfoModel:
    """Test ContainerInfo Pydantic model."""

    def test_container_info_validation(self) -> None:
        """Test ContainerInfo model validation."""
        info = ContainerInfo(
            container_id="abc123",
            name="test-container",
            image="nginx:latest",
            status=ContainerStatus.RUNNING,
            health="healthy",
            ports={"80/tcp": "8080", "443/tcp": "8443"},
            created_at="2026-02-05T10:00:00Z",
            started_at="2026-02-05T10:01:00Z",
        )

        assert info.container_id == "abc123"
        assert info.status == ContainerStatus.RUNNING
        assert len(info.ports) == 2


class TestContainerStatusEnum:
    """Test ContainerStatus enum."""

    def test_enum_values(self) -> None:
        """Test all enum values are defined."""
        assert ContainerStatus.RUNNING.value == "running"
        assert ContainerStatus.STOPPED.value == "stopped"
        assert ContainerStatus.PAUSED.value == "paused"
        assert ContainerStatus.RESTARTING.value == "restarting"
        assert ContainerStatus.EXITED.value == "exited"
        assert ContainerStatus.DEAD.value == "dead"
        assert ContainerStatus.CREATED.value == "created"
        assert ContainerStatus.REMOVING.value == "removing"

    def test_enum_from_string(self) -> None:
        """Test creating enum from string."""
        status = ContainerStatus("running")
        assert status == ContainerStatus.RUNNING


class TestComposeManagerInitialization:
    """Test ComposeManager initialization."""

    def test_initialization(self, docker_config: DockerConfig, tmp_path: Path) -> None:
        """Test ComposeManager initializes correctly."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\nservices:\n  web:\n    image: nginx")

        manager = ComposeManager(docker_config, compose_file)
        assert manager.config == docker_config
        assert manager.compose_file == compose_file


class TestComposeRestartService:
    """Test Docker Compose service restart."""

    @pytest.mark.asyncio
    async def test_restart_service_success(
        self, docker_config: DockerConfig, tmp_path: Path
    ) -> None:
        """Test successfully restarting a compose service."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\nservices:\n  web:\n    image: nginx")

        manager = ComposeManager(docker_config, compose_file)

        # Mock subprocess calls
        async def mock_communicate() -> tuple[bytes, bytes]:
            return b"abc123\ndef456\n", b""

        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(side_effect=mock_communicate)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await manager.restart_service("web", timeout=30)

        assert result.success is True
        assert result.service_name == "web"
        assert result.action == "restart_service"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_restart_service_not_found(
        self, docker_config: DockerConfig, tmp_path: Path
    ) -> None:
        """Test restarting a non-existent service."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\nservices:\n  web:\n    image: nginx")

        manager = ComposeManager(docker_config, compose_file)

        # Mock subprocess calls - first ps succeeds, restart fails
        call_count = [0]

        async def mock_communicate() -> tuple[bytes, bytes]:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: ps -q (list containers)
                return b"", b""
            else:
                # Second call: restart (fails)
                return b"", b"no such service: nonexistent"

        mock_proc = Mock()
        mock_proc.communicate = AsyncMock(side_effect=mock_communicate)

        def set_returncode() -> Mock:
            call_count[0] += 1
            if call_count[0] <= 2:
                mock_proc.returncode = 0
            else:
                mock_proc.returncode = 1
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=lambda *args, **kwargs: set_returncode()):
            result = await manager.restart_service("nonexistent", timeout=30)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_restart_service_compose_file_not_found(
        self, docker_config: DockerConfig
    ) -> None:
        """Test restarting service when compose file doesn't exist."""
        compose_file = Path("/nonexistent/docker-compose.yml")

        manager = ComposeManager(docker_config, compose_file)

        result = await manager.restart_service("web")

        assert result.success is False
        assert "not found" in result.error.lower()


class TestComposeGetServiceStatus:
    """Test getting compose service status."""

    @pytest.mark.asyncio
    async def test_get_service_status_success(
        self,
        docker_config: DockerConfig,
        tmp_path: Path,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test successfully getting service status."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\nservices:\n  web:\n    image: nginx")

        manager = ComposeManager(docker_config, compose_file)

        # Mock subprocess for ps command
        async def mock_communicate() -> tuple[bytes, bytes]:
            return b"abc123\n", b""

        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(side_effect=mock_communicate)

        mock_docker_client.containers.get.return_value = mock_container

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
                results = await manager.get_service_status("web")

        assert len(results) == 1
        assert isinstance(results[0], ContainerInfo)

    @pytest.mark.asyncio
    async def test_get_service_status_no_containers(
        self, docker_config: DockerConfig, tmp_path: Path
    ) -> None:
        """Test getting status for service with no containers."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\nservices:\n  web:\n    image: nginx")

        manager = ComposeManager(docker_config, compose_file)

        # Mock subprocess returning empty container list
        async def mock_communicate() -> tuple[bytes, bytes]:
            return b"", b""

        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(side_effect=mock_communicate)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            results = await manager.get_service_status("web")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_service_status_compose_file_not_found(
        self, docker_config: DockerConfig
    ) -> None:
        """Test getting status when compose file doesn't exist."""
        compose_file = Path("/nonexistent/docker-compose.yml")

        manager = ComposeManager(docker_config, compose_file)

        with pytest.raises(FileNotFoundError):
            await manager.get_service_status("web")


class TestComposeRestartAllServices:
    """Test restarting all compose services."""

    @pytest.mark.asyncio
    async def test_restart_all_services_success(
        self, docker_config: DockerConfig, tmp_path: Path
    ) -> None:
        """Test successfully restarting all services."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text(
            "version: '3'\nservices:\n  web:\n    image: nginx\n  db:\n    image: postgres"
        )

        manager = ComposeManager(docker_config, compose_file)

        call_count = [0]

        async def mock_communicate() -> tuple[bytes, bytes]:
            call_count[0] += 1
            # First call: config --services
            if call_count[0] == 1:
                return b"web\ndb\n", b""
            # Subsequent calls: ps and restart for each service
            return b"container123\n", b""

        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(side_effect=mock_communicate)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            results = await manager.restart_all_services(timeout=30)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert {r.service_name for r in results} == {"web", "db"}

    @pytest.mark.asyncio
    async def test_restart_all_services_no_services(
        self, docker_config: DockerConfig, tmp_path: Path
    ) -> None:
        """Test restarting all when no services defined."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\nservices: {}")

        manager = ComposeManager(docker_config, compose_file)

        # Mock subprocess returning empty service list
        async def mock_communicate() -> tuple[bytes, bytes]:
            return b"", b""

        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(side_effect=mock_communicate)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            results = await manager.restart_all_services()

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_restart_all_services_compose_file_not_found(
        self, docker_config: DockerConfig
    ) -> None:
        """Test restarting all when compose file doesn't exist."""
        compose_file = Path("/nonexistent/docker-compose.yml")

        manager = ComposeManager(docker_config, compose_file)

        results = await manager.restart_all_services()

        assert len(results) == 1
        assert results[0].success is False
        assert "not found" in results[0].error.lower()


class TestComposeActionModel:
    """Test ComposeAction Pydantic model."""

    def test_compose_action_validation(self) -> None:
        """Test ComposeAction model validation."""
        action = ComposeAction(
            success=True,
            service_name="web",
            action="restart_service",
            containers_affected=["abc123", "def456"],
            duration_seconds=2.5,
        )

        assert action.success is True
        assert action.service_name == "web"
        assert len(action.containers_affected) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_container_id(self, docker_config: DockerConfig) -> None:
        """Test handling empty container ID."""
        manager = ContainerManager(docker_config)

        mock_client = MagicMock()
        mock_client.containers.get.side_effect = NotFound("Container not found")

        with patch("docker.DockerClient.from_env", return_value=mock_client):
            result = await manager.stop_container("")

        assert result.success is False
        assert result.container_id == ""

    @pytest.mark.asyncio
    async def test_compose_invalid_file(
        self, docker_config: DockerConfig, tmp_path: Path
    ) -> None:
        """Test handling invalid compose file."""
        compose_file = tmp_path / "invalid.yml"
        compose_file.write_text("invalid: yaml: content:")

        manager = ComposeManager(docker_config, compose_file)

        # Mock subprocess returning error
        async def mock_communicate() -> tuple[bytes, bytes]:
            return b"", b"yaml parse error"

        mock_proc = Mock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(side_effect=mock_communicate)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await manager.restart_service("web")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_container_manager_close(
        self, docker_config: DockerConfig, mock_docker_client: MagicMock
    ) -> None:
        """Test closing container manager."""
        manager = ContainerManager(docker_config)

        with patch("docker.DockerClient.from_env", return_value=mock_docker_client):
            # Trigger client creation
            await asyncio.to_thread(manager._get_client)
            assert manager._client is not None

            # Close
            await manager.close()
            assert manager._client is None

    @pytest.mark.asyncio
    async def test_container_manager_close_idempotent(
        self, docker_config: DockerConfig
    ) -> None:
        """Test closing manager multiple times is safe."""
        manager = ContainerManager(docker_config)

        # Close without ever connecting
        await manager.close()
        assert manager._client is None

        # Close again
        await manager.close()
        assert manager._client is None
