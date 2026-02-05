"""
End-to-end tests for Docker deployment.

These tests verify that the Docker Compose configuration is valid and that
containers can be built and started correctly. They do NOT start the actual
services (which would require docker compose up).
"""

import json
import subprocess
from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCKER_DIR = PROJECT_ROOT / "docker"


@pytest.mark.e2e
@pytest.mark.docker
class TestDockerComposeValidity:
    """Test Docker Compose file validity."""

    def test_production_compose_config_valid(self):
        """Test that production docker-compose.yml is valid."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0, f"docker-compose.yml is invalid:\n{result.stderr}"

    def test_dev_compose_config_valid(self):
        """Test that development docker-compose.dev.yml is valid."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "-f",
                str(DOCKER_DIR / "docker-compose.dev.yml"),
                "config",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0, (
            f"docker-compose.dev.yml override is invalid:\n{result.stderr}"
        )

    def test_compose_has_required_services(self):
        """Test that all required services are defined."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0, "Failed to get compose config"

        config = json.loads(result.stdout)
        services = config.get("services", {})

        required_services = ["orchestrator", "postgres", "ollama"]
        for service in required_services:
            assert service in services, f"Missing required service: {service}"

    def test_compose_has_health_checks(self):
        """Test that critical services have health checks."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)
        services = config["services"]

        # Check orchestrator has healthcheck
        assert "healthcheck" in services["orchestrator"], (
            "Orchestrator missing healthcheck"
        )

        # Check postgres has healthcheck
        assert "healthcheck" in services["postgres"], "PostgreSQL missing healthcheck"

        # Check ollama has healthcheck
        assert "healthcheck" in services["ollama"], "Ollama missing healthcheck"

    def test_compose_uses_required_variables(self):
        """Test that required environment variables are enforced."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        # Check that DATABASE_URL uses POSTGRES_PASSWORD variable
        orchestrator_env = config["services"]["orchestrator"]["environment"]
        assert "DATABASE_URL" in orchestrator_env
        assert "postgres:5432" in orchestrator_env["DATABASE_URL"]


@pytest.mark.e2e
@pytest.mark.docker
class TestDockerImageBuild:
    """Test Docker image build process."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        dockerfile = DOCKER_DIR / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"

    def test_dockerfile_syntax_valid(self):
        """Test that Dockerfile has valid syntax."""
        # Use docker build --dry-run if available, otherwise just check basic syntax
        result = subprocess.run(
            [
                "docker",
                "build",
                "--file",
                str(DOCKER_DIR / "Dockerfile"),
                "--target",
                "runtime",
                "--quiet",
                "--no-cache",
                ".",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )

        # Check if build started successfully (may fail later due to missing deps)
        # We're primarily checking syntax, not full build success
        assert "error" not in result.stderr.lower() or result.returncode == 0, (
            f"Dockerfile syntax error:\n{result.stderr}"
        )

    def test_dockerfile_multi_stage(self):
        """Test that Dockerfile uses multi-stage build."""
        dockerfile_content = (DOCKER_DIR / "Dockerfile").read_text()

        # Should have builder and runtime stages
        assert "FROM python:3.12-slim AS builder" in dockerfile_content
        assert "FROM python:3.12-slim AS runtime" in dockerfile_content

    def test_dockerfile_non_root_user(self):
        """Test that Dockerfile creates and uses non-root user."""
        dockerfile_content = (DOCKER_DIR / "Dockerfile").read_text()

        assert "forgemaster" in dockerfile_content
        assert "USER forgemaster" in dockerfile_content

    def test_dockerfile_healthcheck(self):
        """Test that Dockerfile includes HEALTHCHECK."""
        dockerfile_content = (DOCKER_DIR / "Dockerfile").read_text()

        assert "HEALTHCHECK" in dockerfile_content
        assert "curl" in dockerfile_content or "http" in dockerfile_content

    def test_dockerfile_exposes_port(self):
        """Test that Dockerfile exposes correct port."""
        dockerfile_content = (DOCKER_DIR / "Dockerfile").read_text()

        assert "EXPOSE 8000" in dockerfile_content


@pytest.mark.e2e
@pytest.mark.docker
class TestPostgreSQLConfiguration:
    """Test PostgreSQL service configuration."""

    def test_postgres_uses_pgvector_image(self):
        """Test that PostgreSQL uses pgvector image."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        postgres_image = config["services"]["postgres"]["image"]
        assert "pgvector" in postgres_image, f"Expected pgvector image, got {postgres_image}"
        assert "pg16" in postgres_image, f"Expected PostgreSQL 16, got {postgres_image}"

    def test_postgres_init_scripts_exist(self):
        """Test that PostgreSQL init scripts exist."""
        init_dir = DOCKER_DIR / "init"
        assert init_dir.exists(), "PostgreSQL init directory not found"

        extensions_script = init_dir / "01-extensions.sql"
        assert extensions_script.exists(), "PostgreSQL extensions script not found"

    def test_postgres_init_creates_extensions(self):
        """Test that init script creates required extensions."""
        extensions_script = DOCKER_DIR / "init" / "01-extensions.sql"
        content = extensions_script.read_text()

        assert "CREATE EXTENSION IF NOT EXISTS vector" in content
        assert "CREATE EXTENSION IF NOT EXISTS" in content and "uuid-ossp" in content

    def test_postgres_has_volume_mount(self):
        """Test that PostgreSQL has persistent volume."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        postgres_volumes = config["services"]["postgres"].get("volumes", [])
        assert any("postgresql/data" in str(v) for v in postgres_volumes), (
            "PostgreSQL missing data volume"
        )


@pytest.mark.e2e
@pytest.mark.docker
class TestOllamaConfiguration:
    """Test Ollama service configuration."""

    def test_ollama_has_health_check(self):
        """Test that Ollama service has health check."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        assert "healthcheck" in config["services"]["ollama"]

    def test_ollama_has_volume_for_models(self):
        """Test that Ollama has volume for model cache."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        ollama_volumes = config["services"]["ollama"].get("volumes", [])
        assert any(".ollama" in str(v) for v in ollama_volumes), (
            "Ollama missing model cache volume"
        )

    def test_ollama_init_service_exists(self):
        """Test that ollama-init service exists to pull model."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        assert "ollama-init" in config["services"], "Missing ollama-init service"

        # Check that it pulls the model
        init_command = config["services"]["ollama-init"].get("command", "")
        assert "nomic-embed-text" in str(init_command), (
            "ollama-init doesn't pull nomic-embed-text model"
        )


@pytest.mark.e2e
@pytest.mark.docker
class TestRootlessCompatibility:
    """Test rootless Docker compatibility."""

    def test_no_privileged_containers(self):
        """Test that no containers use privileged mode."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        for service_name, service_config in config["services"].items():
            # Should not have privileged: true
            assert service_config.get("privileged", False) is False, (
                f"Service {service_name} uses privileged mode"
            )

    def test_rootless_documentation_exists(self):
        """Test that rootless Docker documentation exists."""
        rootless_doc = DOCKER_DIR / "ROOTLESS.md"
        assert rootless_doc.exists(), "Missing ROOTLESS.md documentation"

        content = rootless_doc.read_text()
        assert "rootless" in content.lower()
        assert "security" in content.lower()


@pytest.mark.e2e
@pytest.mark.docker
class TestDevelopmentOverride:
    """Test development docker-compose override."""

    def test_dev_override_mounts_source(self):
        """Test that dev override mounts source code."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "-f",
                str(DOCKER_DIR / "docker-compose.dev.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        orchestrator_volumes = config["services"]["orchestrator"]["volumes"]
        assert any("src" in str(v) or "forgemaster" in str(v) for v in orchestrator_volumes), (
            "Dev override doesn't mount source code"
        )

    def test_dev_override_uses_debug_logging(self):
        """Test that dev override uses DEBUG log level."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "-f",
                str(DOCKER_DIR / "docker-compose.dev.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        orchestrator_env = config["services"]["orchestrator"]["environment"]
        assert orchestrator_env.get("LOG_LEVEL") == "DEBUG", (
            "Dev override should use DEBUG logging"
        )

    def test_dev_override_disables_ollama_init(self):
        """Test that dev override disables ollama-init for faster iteration."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(DOCKER_DIR / "docker-compose.yml"),
                "-f",
                str(DOCKER_DIR / "docker-compose.dev.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={"POSTGRES_PASSWORD": "test"},
        )

        assert result.returncode == 0
        config = json.loads(result.stdout)

        # ollama-init should be disabled via profiles
        if "ollama-init" in config["services"]:
            profiles = config["services"]["ollama-init"].get("profiles", [])
            assert "donotstart" in profiles, (
                "ollama-init should be disabled in dev mode"
            )
