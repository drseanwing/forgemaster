"""Unit tests for logging configuration."""

from __future__ import annotations

import json
import logging
import logging.handlers
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import structlog

from forgemaster.config import LoggingConfig
from forgemaster.logging import (
    add_correlation_id,
    bind_task_context,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging() -> None:
    """Reset logging configuration before each test."""
    # Clear stdlib logging handlers
    root = logging.getLogger()
    root.handlers.clear()

    # Reset structlog
    structlog.reset_defaults()

    # Clear contextvars
    structlog.contextvars.clear_contextvars()
    set_correlation_id(None)


@pytest.fixture
def capture_stream() -> StringIO:
    """Create a StringIO stream for capturing log output."""
    return StringIO()


@pytest.fixture
def json_config() -> LoggingConfig:
    """Create a LoggingConfig for JSON output to stdout."""
    return LoggingConfig(level="INFO", format="json", file=None)


@pytest.fixture
def console_config() -> LoggingConfig:
    """Create a LoggingConfig for console output to stdout."""
    return LoggingConfig(level="DEBUG", format="console", file=None)


def test_json_output_format(json_config: LoggingConfig, capture_stream: StringIO) -> None:
    """Test that JSON format produces valid JSON output."""
    # Setup logging
    setup_logging(json_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    # Emit a log
    logger = get_logger("test.module")
    logger.info("test_event", key1="value1", key2=42)

    # Parse output as JSON
    output = capture_stream.getvalue()
    log_entry = json.loads(output.strip())

    # Verify JSON structure
    assert log_entry["event"] == "test_event"
    assert log_entry["key1"] == "value1"
    assert log_entry["key2"] == 42
    assert log_entry["level"] == "info"
    assert log_entry["logger"] == "test.module"
    assert "timestamp" in log_entry


def test_console_output_format(console_config: LoggingConfig, capture_stream: StringIO) -> None:
    """Test that console format produces human-readable output."""
    # Setup logging
    setup_logging(console_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    # Emit a log
    logger = get_logger("test.module")
    logger.debug("test_event", status="active")

    # Verify console output is human-readable (not JSON)
    output = capture_stream.getvalue()
    assert "test_event" in output
    assert "status" in output
    assert "active" in output
    # Console format should not be valid JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.strip())


def test_log_level_filtering(json_config: LoggingConfig, capture_stream: StringIO) -> None:
    """Test that log level filtering works correctly."""
    # INFO level should filter out DEBUG
    setup_logging(json_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    logger = get_logger("test.module")

    # DEBUG should be filtered
    logger.debug("debug_message")
    assert capture_stream.getvalue() == ""

    # INFO should pass
    logger.info("info_message")
    output = capture_stream.getvalue()
    assert "info_message" in output

    # WARNING should pass
    capture_stream.truncate(0)
    capture_stream.seek(0)
    logger.warning("warning_message")
    output = capture_stream.getvalue()
    assert "warning_message" in output


def test_correlation_id_binding(json_config: LoggingConfig, capture_stream: StringIO) -> None:
    """Test that correlation ID is added to log entries."""
    setup_logging(json_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    logger = get_logger("test.module")

    # Set correlation ID
    test_corr_id = "corr-12345"
    set_correlation_id(test_corr_id)

    # Verify getter returns correct value
    assert get_correlation_id() == test_corr_id

    # Emit log
    logger.info("test_with_correlation")
    output = capture_stream.getvalue()
    log_entry = json.loads(output.strip())

    # Verify correlation ID in output
    assert log_entry["correlation_id"] == test_corr_id

    # Clear correlation ID
    set_correlation_id(None)
    assert get_correlation_id() is None

    # Emit another log
    capture_stream.truncate(0)
    capture_stream.seek(0)
    logger.info("test_without_correlation")
    output = capture_stream.getvalue()
    log_entry = json.loads(output.strip())

    # Verify correlation ID not in output
    assert "correlation_id" not in log_entry


def test_correlation_id_processor() -> None:
    """Test the correlation ID processor directly."""
    event_dict: dict[str, Any] = {"event": "test"}

    # No correlation ID set
    result = add_correlation_id(None, "", event_dict.copy())
    assert "correlation_id" not in result

    # Correlation ID set
    set_correlation_id("test-id")
    result = add_correlation_id(None, "", event_dict.copy())
    assert result["correlation_id"] == "test-id"

    # Clean up
    set_correlation_id(None)


def test_task_context_binding(json_config: LoggingConfig, capture_stream: StringIO) -> None:
    """Test that task and session context is bound correctly."""
    setup_logging(json_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    logger = get_logger("test.module")

    # Bind task context
    bind_task_context(task_id="T-999", session_id="S-888")

    # Emit log
    logger.info("task_event")
    output = capture_stream.getvalue()
    log_entry = json.loads(output.strip())

    # Verify context in output
    assert log_entry["task_id"] == "T-999"
    assert log_entry["session_id"] == "S-888"


def test_file_rotation_handler_configuration(tmp_path: Path) -> None:
    """Test that file rotation handler is configured correctly."""
    log_file = tmp_path / "test.log"
    config = LoggingConfig(
        level="INFO",
        format="json",
        file=log_file,
        rotation_size_mb=10,
        retention_count=3,
    )

    # Setup logging
    setup_logging(config)

    # Verify file was created
    assert log_file.exists()

    # Verify handler configuration
    root = logging.getLogger()
    assert len(root.handlers) == 1

    handler = root.handlers[0]
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.maxBytes == 10 * 1024 * 1024  # 10 MB
    assert handler.backupCount == 3

    # Emit a log to verify writing works
    logger = get_logger("test.module")
    logger.info("test_file_write", data="test")

    # Force flush
    handler.flush()

    # Verify log was written
    log_content = log_file.read_text()
    log_entry = json.loads(log_content.strip())
    assert log_entry["event"] == "test_file_write"
    assert log_entry["data"] == "test"


def test_file_rotation_creates_parent_directories(tmp_path: Path) -> None:
    """Test that parent directories are created for log file."""
    log_file = tmp_path / "subdir" / "nested" / "test.log"
    config = LoggingConfig(level="INFO", format="json", file=log_file)

    # Setup logging (should create parent dirs)
    setup_logging(config)

    # Verify directory structure
    assert log_file.parent.exists()
    assert log_file.exists()


def test_logger_name_in_output(json_config: LoggingConfig, capture_stream: StringIO) -> None:
    """Test that logger name is included in output."""
    setup_logging(json_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    # Create logger with specific name
    logger = get_logger("forgemaster.agents.executor")
    logger.info("test_event")

    output = capture_stream.getvalue()
    log_entry = json.loads(output.strip())

    assert log_entry["logger"] == "forgemaster.agents.executor"


def test_timestamp_in_output(json_config: LoggingConfig, capture_stream: StringIO) -> None:
    """Test that ISO timestamp is included in output."""
    setup_logging(json_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    logger = get_logger("test.module")
    logger.info("test_event")

    output = capture_stream.getvalue()
    log_entry = json.loads(output.strip())

    # Verify timestamp exists and is ISO format
    assert "timestamp" in log_entry
    # ISO format should parse (basic validation)
    from datetime import datetime

    datetime.fromisoformat(log_entry["timestamp"].replace("Z", "+00:00"))


def test_exception_formatting(json_config: LoggingConfig, capture_stream: StringIO) -> None:
    """Test that exceptions are formatted correctly in logs."""
    setup_logging(json_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    logger = get_logger("test.module")

    # Log an exception
    try:
        raise ValueError("Test exception")
    except ValueError:
        logger.exception("error_occurred")

    output = capture_stream.getvalue()
    log_entry = json.loads(output.strip())

    # Verify exception info in output
    assert log_entry["event"] == "error_occurred"
    assert log_entry["level"] == "error"
    assert "exception" in log_entry
    assert "ValueError: Test exception" in log_entry["exception"]


def test_multiple_loggers_share_context(
    json_config: LoggingConfig, capture_stream: StringIO
) -> None:
    """Test that context is shared across different logger instances."""
    setup_logging(json_config)

    # Replace stdout handler with our capture stream
    root = logging.getLogger()
    root.handlers[0].stream = capture_stream

    # Bind context
    bind_task_context(task_id="T-111", session_id="S-222")

    # Create multiple loggers
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    # Both should have context
    logger1.info("event1")
    output1 = capture_stream.getvalue()
    log1 = json.loads(output1.strip())

    capture_stream.truncate(0)
    capture_stream.seek(0)

    logger2.info("event2")
    output2 = capture_stream.getvalue()
    log2 = json.loads(output2.strip())

    # Verify both have task context
    assert log1["task_id"] == "T-111"
    assert log1["session_id"] == "S-222"
    assert log2["task_id"] == "T-111"
    assert log2["session_id"] == "S-222"


def test_config_validation() -> None:
    """Test that LoggingConfig validates inputs correctly."""
    # Valid configs
    LoggingConfig(level="DEBUG")
    LoggingConfig(level="info")  # Case insensitive
    LoggingConfig(format="json")
    LoggingConfig(format="CONSOLE")  # Case insensitive

    # Invalid log level
    with pytest.raises(ValueError, match="Invalid log level"):
        LoggingConfig(level="INVALID")

    # Invalid format
    with pytest.raises(ValueError, match="Invalid log format"):
        LoggingConfig(format="xml")
