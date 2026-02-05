"""Configuration management for Forgemaster.

This module defines the configuration schema using Pydantic settings,
supporting TOML files, environment variables, and programmatic overrides.

Configuration loading priority (highest to lowest):
1. Programmatic overrides (passed to ForgemasterConfig constructor)
2. Environment variables (FORGEMASTER_* prefix)
3. TOML configuration file
4. Default values defined in this module

Example TOML configuration:
    [database]
    url = "postgresql+asyncpg://localhost/forgemaster"
    pool_size = 10

Example environment variable override:
    FORGEMASTER_DATABASE__URL="postgresql+asyncpg://prod/forgemaster"
    FORGEMASTER_AGENT__MAX_CONCURRENT_WORKERS=5
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database connection configuration.

    Attributes:
        url: SQLAlchemy database URL with asyncpg driver
        pool_size: Number of connections to maintain in the pool
        max_overflow: Maximum number of connections beyond pool_size
        echo: Enable SQL query logging
        pgvector_enabled: Enable pgvector extension for embeddings
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGEMASTER_DATABASE__",
        extra="forbid",
    )

    url: str = Field(
        default="postgresql+asyncpg://forgemaster:password@localhost:5432/forgemaster",
        description="PostgreSQL connection URL",
    )
    pool_size: int = Field(default=5, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    echo: bool = Field(default=False)
    pgvector_enabled: bool = Field(default=True)


class OllamaConfig(BaseSettings):
    """Ollama embedding service configuration.

    Attributes:
        url: Base URL of Ollama API server
        model: Embedding model name to use
        timeout_seconds: Request timeout in seconds
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGEMASTER_OLLAMA__",
        extra="forbid",
    )

    url: str = Field(default="http://localhost:11434")
    model: str = Field(default="nomic-embed-text")
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class LoggingConfig(BaseSettings):
    """Logging configuration.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format (json or console)
        file: Optional log file path (None for stdout only)
        rotation_size_mb: Log file rotation size in megabytes
        retention_count: Number of rotated log files to keep
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGEMASTER_LOGGING__",
        extra="forbid",
    )

    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file: Path | None = Field(default=None)
    rotation_size_mb: int = Field(default=50, ge=1, le=1000)
    retention_count: int = Field(default=10, ge=1, le=100)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level is recognized."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate log format is recognized."""
        valid_formats = {"json", "console"}
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"Invalid log format: {v}. Must be one of {valid_formats}")
        return v_lower


class AgentConfig(BaseSettings):
    """Agent execution configuration.

    Attributes:
        max_concurrent_workers: Maximum number of concurrent agent workers
        session_timeout_seconds: Maximum session duration in seconds
        idle_timeout_seconds: Idle timeout before session cleanup
        max_retries: Maximum retry attempts for failed operations
        context_warning_threshold: Context window usage threshold (0.0-1.0)
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGEMASTER_AGENT__",
        extra="forbid",
    )

    max_concurrent_workers: int = Field(default=3, ge=1, le=20)
    session_timeout_seconds: int = Field(default=1800, ge=60, le=86400)  # 30 min
    idle_timeout_seconds: int = Field(default=300, ge=30, le=3600)  # 5 min
    max_retries: int = Field(default=3, ge=0, le=10)
    context_warning_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class GitConfig(BaseSettings):
    """Git operations configuration.

    Attributes:
        worktree_base_path: Base path for git worktrees
        main_branch: Name of the main branch
        auto_push: Automatically push commits to remote
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGEMASTER_GIT__",
        extra="forbid",
    )

    worktree_base_path: Path = Field(default=Path("/workspace"))
    main_branch: str = Field(default="main")
    auto_push: bool = Field(default=True)


class DockerConfig(BaseSettings):
    """Docker operations configuration.

    Attributes:
        registry: Docker registry URL
        rootless: Use rootless Docker daemon
        build_timeout_seconds: Build operation timeout in seconds
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGEMASTER_DOCKER__",
        extra="forbid",
    )

    registry: str = Field(default="ghcr.io")
    rootless: bool = Field(default=True)
    build_timeout_seconds: int = Field(default=600, ge=60, le=3600)


class WebConfig(BaseSettings):
    """Web dashboard configuration.

    Attributes:
        host: Bind host address
        port: Bind port number
        cors_origins: Allowed CORS origins
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGEMASTER_WEB__",
        extra="forbid",
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class ForgemasterConfig(BaseSettings):
    """Root configuration for Forgemaster.

    This is the main configuration class that aggregates all subsystem configurations.
    Configuration can be loaded from:
    1. TOML files (using load_config function)
    2. Environment variables (FORGEMASTER_* prefix)
    3. Direct instantiation with keyword arguments

    Environment variable format for nested config:
        FORGEMASTER_<SECTION>__<KEY>=value

    Example:
        FORGEMASTER_DATABASE__URL="postgresql://localhost/db"
        FORGEMASTER_AGENT__MAX_CONCURRENT_WORKERS=5
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGEMASTER_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    web: WebConfig = Field(default_factory=WebConfig)


def load_config(config_path: Path | None = None) -> ForgemasterConfig:
    """Load configuration from TOML file with environment variable overrides.

    Configuration search order (first found is used):
    1. config_path if explicitly provided
    2. ./forgemaster.toml (current directory)
    3. ~/.config/forgemaster/config.toml (user config directory)

    After loading TOML, environment variables are applied as overrides
    using the FORGEMASTER_* prefix scheme.

    Args:
        config_path: Explicit path to TOML config file. If None, searches
                    default locations.

    Returns:
        ForgemasterConfig: Fully resolved configuration instance.

    Raises:
        FileNotFoundError: If config_path is explicitly provided but doesn't exist.
        ValueError: If TOML file contains invalid configuration.

    Example:
        >>> config = load_config()  # Load from default locations
        >>> config = load_config(Path("custom.toml"))  # Load from specific file
    """
    toml_data: dict[str, Any] = {}

    # Determine which TOML file to load
    if config_path is not None:
        # Explicit path provided - must exist
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        selected_path = config_path
    else:
        # Search default locations
        search_paths = [
            Path.cwd() / "forgemaster.toml",
            Path.home() / ".config" / "forgemaster" / "config.toml",
        ]
        selected_path = None
        for path in search_paths:
            if path.exists():
                selected_path = path
                break

    # Load TOML file if found
    if selected_path is not None:
        with open(selected_path, "rb") as f:
            toml_data = tomli.load(f)

    # Create config from TOML data
    # Pydantic will automatically overlay environment variables
    try:
        return ForgemasterConfig(**toml_data)
    except Exception as e:
        if selected_path:
            raise ValueError(
                f"Invalid configuration in {selected_path}: {e}"
            ) from e
        else:
            raise ValueError(f"Invalid configuration: {e}") from e
