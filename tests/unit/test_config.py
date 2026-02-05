"""Unit tests for configuration management.

Tests cover:
- Default configuration values
- TOML file loading
- Environment variable overrides
- Nested configuration resolution
- Validation errors for invalid configurations
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from forgemaster.config import (
    AgentConfig,
    DatabaseConfig,
    DockerConfig,
    ForgemasterConfig,
    GitConfig,
    LoggingConfig,
    OllamaConfig,
    WebConfig,
    load_config,
)


class TestDatabaseConfig:
    """Test DatabaseConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Test that default database configuration values are correct."""
        config = DatabaseConfig()
        assert config.url == "postgresql+asyncpg://forgemaster:password@localhost:5432/forgemaster"
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.echo is False
        assert config.pgvector_enabled is True

    def test_custom_values(self) -> None:
        """Test that custom values override defaults."""
        config = DatabaseConfig(
            url="postgresql+asyncpg://custom/db",
            pool_size=20,
            echo=True,
        )
        assert config.url == "postgresql+asyncpg://custom/db"
        assert config.pool_size == 20
        assert config.echo is True

    def test_pool_size_validation(self) -> None:
        """Test that pool_size is validated within range."""
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=0)  # Below minimum
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=101)  # Above maximum


class TestOllamaConfig:
    """Test OllamaConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Test that default Ollama configuration values are correct."""
        config = OllamaConfig()
        assert config.url == "http://localhost:11434"
        assert config.model == "nomic-embed-text"
        assert config.timeout_seconds == 30

    def test_timeout_validation(self) -> None:
        """Test that timeout is validated within range."""
        with pytest.raises(ValidationError):
            OllamaConfig(timeout_seconds=0)  # Below minimum
        with pytest.raises(ValidationError):
            OllamaConfig(timeout_seconds=301)  # Above maximum


class TestLoggingConfig:
    """Test LoggingConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Test that default logging configuration values are correct."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.file is None
        assert config.rotation_size_mb == 50
        assert config.retention_count == 10

    def test_level_validation(self) -> None:
        """Test that log level is validated."""
        # Valid levels (case insensitive)
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"

        config = LoggingConfig(level="INFO")
        assert config.level == "INFO"

        # Invalid level
        with pytest.raises(ValidationError, match="Invalid log level"):
            LoggingConfig(level="INVALID")

    def test_format_validation(self) -> None:
        """Test that log format is validated."""
        # Valid formats (case insensitive)
        config = LoggingConfig(format="JSON")
        assert config.format == "json"

        config = LoggingConfig(format="console")
        assert config.format == "console"

        # Invalid format
        with pytest.raises(ValidationError, match="Invalid log format"):
            LoggingConfig(format="xml")


class TestAgentConfig:
    """Test AgentConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Test that default agent configuration values are correct."""
        config = AgentConfig()
        assert config.max_concurrent_workers == 3
        assert config.session_timeout_seconds == 1800
        assert config.idle_timeout_seconds == 300
        assert config.max_retries == 3
        assert config.context_warning_threshold == 0.8

    def test_threshold_validation(self) -> None:
        """Test that context warning threshold is validated within range."""
        with pytest.raises(ValidationError):
            AgentConfig(context_warning_threshold=-0.1)  # Below minimum
        with pytest.raises(ValidationError):
            AgentConfig(context_warning_threshold=1.1)  # Above maximum

        # Valid edge cases
        config = AgentConfig(context_warning_threshold=0.0)
        assert config.context_warning_threshold == 0.0

        config = AgentConfig(context_warning_threshold=1.0)
        assert config.context_warning_threshold == 1.0


class TestGitConfig:
    """Test GitConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Test that default Git configuration values are correct."""
        config = GitConfig()
        assert config.worktree_base_path == Path("/workspace")
        assert config.main_branch == "main"
        assert config.auto_push is True


class TestDockerConfig:
    """Test DockerConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Test that default Docker configuration values are correct."""
        config = DockerConfig()
        assert config.registry == "ghcr.io"
        assert config.rootless is True
        assert config.build_timeout_seconds == 600


class TestWebConfig:
    """Test WebConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Test that default web configuration values are correct."""
        config = WebConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.cors_origins == ["http://localhost:3000"]

    def test_port_validation(self) -> None:
        """Test that port is validated within range."""
        with pytest.raises(ValidationError):
            WebConfig(port=0)  # Below minimum
        with pytest.raises(ValidationError):
            WebConfig(port=65536)  # Above maximum


class TestForgemasterConfig:
    """Test ForgemasterConfig integration."""

    def test_default_values(self) -> None:
        """Test that default root configuration creates all subsections."""
        config = ForgemasterConfig()
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.git, GitConfig)
        assert isinstance(config.docker, DockerConfig)
        assert isinstance(config.web, WebConfig)

    def test_nested_override(self) -> None:
        """Test that nested configuration can be overridden."""
        config = ForgemasterConfig(
            database=DatabaseConfig(pool_size=20),
            agent=AgentConfig(max_concurrent_workers=10),
        )
        assert config.database.pool_size == 20
        assert config.agent.max_concurrent_workers == 10
        # Other values should remain default
        assert config.database.url == "postgresql+asyncpg://forgemaster:password@localhost:5432/forgemaster"


class TestLoadConfig:
    """Test TOML configuration loading."""

    def test_load_defaults_when_no_file(self) -> None:
        """Test that defaults are loaded when no config file exists."""
        config = load_config(Path("/nonexistent/config.toml") if False else None)
        assert isinstance(config, ForgemasterConfig)
        assert config.database.pool_size == 5

    def test_explicit_path_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing explicit path."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(Path("/nonexistent/config.toml"))

    def test_load_from_toml_file(self, tmp_path: Path) -> None:
        """Test loading configuration from a TOML file."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("""
[database]
pool_size = 15
echo = true

[agent]
max_concurrent_workers = 7

[logging]
level = "DEBUG"
format = "console"
""")

        config = load_config(config_file)
        assert config.database.pool_size == 15
        assert config.database.echo is True
        assert config.agent.max_concurrent_workers == 7
        assert config.logging.level == "DEBUG"
        assert config.logging.format == "console"
        # Unspecified values should be defaults
        assert config.database.max_overflow == 10

    def test_load_partial_config(self, tmp_path: Path) -> None:
        """Test loading partial configuration merges with defaults."""
        config_file = tmp_path / "partial.toml"
        config_file.write_text("""
[database]
pool_size = 25
""")

        config = load_config(config_file)
        assert config.database.pool_size == 25
        # All other values should be defaults
        assert config.database.url == "postgresql+asyncpg://forgemaster:password@localhost:5432/forgemaster"
        assert config.agent.max_concurrent_workers == 3

    def test_invalid_toml_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid TOML raises a ValueError."""
        config_file = tmp_path / "invalid.toml"
        config_file.write_text("""
[database]
pool_size = "not a number"
""")

        with pytest.raises(ValueError, match="Invalid configuration"):
            load_config(config_file)

    def test_search_current_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that load_config searches current directory."""
        config_file = tmp_path / "forgemaster.toml"
        config_file.write_text("""
[database]
pool_size = 99
""")

        # Change to tmp_path so it finds forgemaster.toml
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.database.pool_size == 99

    def test_environment_variable_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables override TOML values."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("""
[database]
pool_size = 10

[agent]
max_concurrent_workers = 5
""")

        # Set environment variables
        monkeypatch.setenv("FORGEMASTER_DATABASE__POOL_SIZE", "30")
        monkeypatch.setenv("FORGEMASTER_AGENT__MAX_CONCURRENT_WORKERS", "15")

        config = load_config(config_file)
        # Environment variables should override TOML
        assert config.database.pool_size == 30
        assert config.agent.max_concurrent_workers == 15

    def test_environment_variables_without_toml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables work without a TOML file."""
        monkeypatch.setenv("FORGEMASTER_DATABASE__URL", "postgresql://testdb")
        monkeypatch.setenv("FORGEMASTER_WEB__PORT", "9000")
        monkeypatch.setenv("FORGEMASTER_LOGGING__LEVEL", "WARNING")

        config = load_config()
        assert config.database.url == "postgresql://testdb"
        assert config.web.port == 9000
        assert config.logging.level == "WARNING"

    def test_complex_nested_config(self, tmp_path: Path) -> None:
        """Test loading complex nested configuration."""
        config_file = tmp_path / "complex.toml"
        config_file.write_text("""
[database]
url = "postgresql+asyncpg://prod:secret@db.example.com:5432/forgemaster_prod"
pool_size = 50
max_overflow = 20
echo = false
pgvector_enabled = true

[ollama]
url = "http://ollama.example.com:11434"
model = "mxbai-embed-large"
timeout_seconds = 60

[logging]
level = "WARNING"
format = "json"
rotation_size_mb = 100
retention_count = 20

[agent]
max_concurrent_workers = 10
session_timeout_seconds = 3600
idle_timeout_seconds = 600
max_retries = 5
context_warning_threshold = 0.9

[git]
worktree_base_path = "/data/worktrees"
main_branch = "master"
auto_push = false

[docker]
registry = "registry.example.com"
rootless = false
build_timeout_seconds = 1200

[web]
host = "127.0.0.1"
port = 9000
cors_origins = ["https://app.example.com", "https://admin.example.com"]
""")

        config = load_config(config_file)

        # Verify all sections loaded correctly
        assert config.database.url == "postgresql+asyncpg://prod:secret@db.example.com:5432/forgemaster_prod"
        assert config.database.pool_size == 50

        assert config.ollama.model == "mxbai-embed-large"
        assert config.ollama.timeout_seconds == 60

        assert config.logging.level == "WARNING"
        assert config.logging.rotation_size_mb == 100

        assert config.agent.max_concurrent_workers == 10
        assert config.agent.context_warning_threshold == 0.9

        assert config.git.worktree_base_path == Path("/data/worktrees")
        assert config.git.main_branch == "master"
        assert config.git.auto_push is False

        assert config.docker.registry == "registry.example.com"
        assert config.docker.rootless is False

        assert config.web.port == 9000
        assert "https://app.example.com" in config.web.cors_origins
