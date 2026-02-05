"""Structured logging configuration for Forgemaster.

This module configures structlog with support for:
- JSON and console output formats
- File rotation based on size
- Correlation IDs for distributed tracing
- Task and session context binding

The logging system integrates structlog with Python's stdlib logging
for handlers (file rotation), while using structlog exclusively for
actual log emission.

Example usage:
    >>> from forgemaster.config import LoggingConfig
    >>> from forgemaster.logging import setup_logging, get_logger, bind_task_context
    >>>
    >>> config = LoggingConfig(level="INFO", format="json", file=Path("app.log"))
    >>> setup_logging(config)
    >>>
    >>> logger = get_logger(__name__)
    >>> bind_task_context(task_id="T-123", session_id="S-456")
    >>> logger.info("task_started", status="running")
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog

from forgemaster.config import LoggingConfig

# Context variable for correlation ID
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def add_correlation_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add correlation_id to log event if set in context.

    Args:
        logger: Logger instance (unused, required by structlog protocol)
        method_name: Log method name (unused, required by structlog protocol)
        event_dict: Current event dictionary to augment

    Returns:
        Event dictionary with correlation_id added if available
    """
    correlation_id = _correlation_id.get()
    if correlation_id is not None:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def set_correlation_id(correlation_id: str | None) -> None:
    """Set correlation ID for current context.

    Args:
        correlation_id: Correlation ID string or None to clear
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str | None:
    """Get current correlation ID from context.

    Returns:
        Current correlation ID or None if not set
    """
    return _correlation_id.get()


def bind_task_context(task_id: str, session_id: str) -> None:
    """Bind task and session context to all subsequent logs.

    This function binds both task_id and session_id to the structlog
    context, ensuring they appear in all log entries within the current
    async context.

    Args:
        task_id: Task identifier to bind
        session_id: Session identifier to bind
    """
    structlog.contextvars.bind_contextvars(task_id=task_id, session_id=session_id)


def setup_logging(config: LoggingConfig) -> None:
    """Configure structlog with the given configuration.

    This function sets up the complete logging pipeline including:
    - JSON or console rendering based on config.format
    - File rotation if config.file is specified
    - Timestamp, log level, and logger name processors
    - Correlation ID processor

    Args:
        config: Logging configuration from ForgemasterConfig

    Example:
        >>> from forgemaster.config import LoggingConfig
        >>> from pathlib import Path
        >>>
        >>> # JSON logging to file with rotation
        >>> config = LoggingConfig(
        ...     level="INFO",
        ...     format="json",
        ...     file=Path("/var/log/forgemaster.log"),
        ...     rotation_size_mb=100,
        ...     retention_count=5
        ... )
        >>> setup_logging(config)
        >>>
        >>> # Console logging for development
        >>> dev_config = LoggingConfig(level="DEBUG", format="console")
        >>> setup_logging(dev_config)
    """
    # Convert log level string to logging constant
    log_level = getattr(logging, config.level)

    # Configure stdlib logging root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Create appropriate handler
    if config.file is not None:
        # Ensure parent directory exists
        config.file.parent.mkdir(parents=True, exist_ok=True)

        # File handler with rotation
        handler = logging.handlers.RotatingFileHandler(
            filename=config.file,
            maxBytes=config.rotation_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=config.retention_count,
            encoding="utf-8",
        )
    else:
        # Stream handler to stdout
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(log_level)
    root_logger.addHandler(handler)

    # Choose renderer based on format
    if config.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:  # console
        renderer = structlog.dev.ConsoleRenderer()

    # Configure structlog with complete processor chain
    structlog.configure(
        processors=[
            # Add log level to event dict
            structlog.stdlib.add_log_level,
            # Add logger name to event dict
            structlog.stdlib.add_logger_name,
            # Add timestamp in ISO format
            structlog.processors.TimeStamper(fmt="iso"),
            # Add contextvars (for task_id, session_id from bind_task_context)
            structlog.contextvars.merge_contextvars,
            # Add correlation ID if present
            add_correlation_id,
            # Stack info and exception formatting
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Final rendering
            renderer,
        ],
        # Wrap stdlib logger for compatibility
        wrapper_class=structlog.stdlib.BoundLogger,
        # Cache logger instances
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Enable contextvars for async safety
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured structlog BoundLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("operation_complete", duration=1.23, status="success")
    """
    return structlog.get_logger(name)
