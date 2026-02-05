"""Unit tests for agent session management.

Tests cover:
- Session lifecycle (start → message → end)
- Health monitoring (idle, stuck detection)
- Token counting and threshold warnings
- Result schema validation
- Result parsing from raw output
- Malformed result handling
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forgemaster.agents.result_schema import (
    AgentResult,
    IssueDiscovered,
    LessonLearned,
    TestResult,
    create_minimal_result,
    parse_agent_result,
    parse_agent_result_safe,
)
from forgemaster.agents.sdk_wrapper import AgentClient, AgentSession
from forgemaster.agents.session import (
    AgentSessionManager,
    HealthStatus,
    SessionState,
)
from forgemaster.config import AgentConfig


# Mock implementations of SDK protocols for testing


class MockAgentSession:
    """Mock implementation of AgentSession protocol."""

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    async def send_message(self, message: str) -> str:
        if self._closed:
            raise RuntimeError("Session is closed")
        return f"Mock response to: {message[:50]}"

    async def close(self) -> None:
        self._closed = True


class MockClaudeSDK:
    """Mock implementation of ClaudeSDK protocol."""

    def __init__(self):
        self._closed = False
        self._session_counter = 0

    async def create_session(
        self,
        model: str,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AgentSession:
        if self._closed:
            raise RuntimeError("SDK is closed")
        self._session_counter += 1
        return MockAgentSession(f"mock-session-{self._session_counter}")

    async def close(self) -> None:
        self._closed = True


@pytest.fixture
def agent_config():
    """Create agent configuration for testing."""
    return AgentConfig(
        max_concurrent_workers=3,
        session_timeout_seconds=1800,  # 30 minutes
        idle_timeout_seconds=300,  # 5 minutes
        max_retries=3,
        context_warning_threshold=0.8,
    )


@pytest.fixture
def agent_client():
    """Create mock agent client for testing."""
    client = AgentClient(api_key="test-key")
    client._sdk = MockClaudeSDK()
    return client


@pytest.fixture
def session_manager(agent_config, agent_client):
    """Create session manager for testing."""
    return AgentSessionManager(config=agent_config, agent_client=agent_client)


# Session Lifecycle Tests


@pytest.mark.asyncio
async def test_start_session_creates_session(session_manager):
    """Test that start_session creates and initializes a session."""
    session_id = await session_manager.start_session(
        task_id="T-001",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    assert session_id is not None
    assert session_id.startswith("T-001-executor-")

    # Verify session info
    session_info = session_manager.get_session_info(session_id)
    assert session_info.task_id == "T-001"
    assert session_info.agent_type == "executor"
    assert session_info.model == "claude-3-5-sonnet-20241022"
    assert session_info.state == SessionState.ACTIVE
    assert session_info.health == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_send_message_updates_metrics(session_manager):
    """Test that send_message updates session metrics."""
    session_id = await session_manager.start_session(
        task_id="T-002",
        agent_type="architect",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    # Send a message
    response = await session_manager.send_message(session_id, "Hello agent!")

    assert response is not None
    assert response.startswith("Mock response to:")

    # Verify metrics updated
    session_info = session_manager.get_session_info(session_id)
    assert session_info.metrics.messages_sent == 1
    assert session_info.metrics.last_activity_at is not None


@pytest.mark.asyncio
async def test_end_session_returns_metrics(session_manager):
    """Test that end_session returns final metrics."""
    session_id = await session_manager.start_session(
        task_id="T-003",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    # Send some messages
    await session_manager.send_message(session_id, "Message 1")
    await session_manager.send_message(session_id, "Message 2")

    # Update token counts
    session_manager.update_token_count(session_id, input_tokens=100, output_tokens=200)

    # End session
    result = await session_manager.end_session(session_id, status="completed")

    assert result["session_id"] == session_id
    assert result["state"] == "completed"
    assert result["metrics"]["messages_sent"] == 2
    assert result["metrics"]["input_tokens"] == 100
    assert result["metrics"]["output_tokens"] == 200
    assert result["metrics"]["total_tokens"] == 300


@pytest.mark.asyncio
async def test_send_message_to_nonexistent_session_raises_error(session_manager):
    """Test that sending message to non-existent session raises ValueError."""
    with pytest.raises(ValueError, match="Session not found"):
        await session_manager.send_message("invalid-session-id", "Hello")


@pytest.mark.asyncio
async def test_end_session_transitions_through_completing(session_manager):
    """Test that end_session transitions through COMPLETING state."""
    session_id = await session_manager.start_session(
        task_id="T-004",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    result = await session_manager.end_session(session_id)

    # Should end in COMPLETED state
    session_info = session_manager.get_session_info(session_id)
    assert session_info.state == SessionState.COMPLETED


# Health Monitoring Tests


@pytest.mark.asyncio
async def test_check_health_detects_idle_session(session_manager, agent_config):
    """Test that check_health detects idle sessions."""
    session_id = await session_manager.start_session(
        task_id="T-005",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    # Manually set last_activity_at to simulate idle session
    session_info = session_manager.get_session_info(session_id)
    session_info.metrics.last_activity_at = datetime.now(timezone.utc) - timedelta(
        seconds=agent_config.idle_timeout_seconds + 10
    )

    # Check health
    health = session_manager.check_health(session_id)

    assert health == HealthStatus.IDLE


@pytest.mark.asyncio
async def test_check_health_detects_stuck_session(session_manager, agent_config):
    """Test that check_health detects stuck sessions."""
    session_id = await session_manager.start_session(
        task_id="T-006",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    # Manually set created_at to simulate stuck session
    session_info = session_manager.get_session_info(session_id)
    session_info.metrics.created_at = datetime.now(timezone.utc) - timedelta(
        seconds=agent_config.session_timeout_seconds + 10
    )

    # Check health
    health = session_manager.check_health(session_id)

    assert health == HealthStatus.STUCK


@pytest.mark.asyncio
async def test_check_health_detects_healthy_session(session_manager):
    """Test that check_health correctly identifies healthy sessions."""
    session_id = await session_manager.start_session(
        task_id="T-007",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    # Immediately check health - should be healthy
    health = session_manager.check_health(session_id)

    assert health == HealthStatus.HEALTHY


# Token Counting Tests


@pytest.mark.asyncio
async def test_update_token_count_accumulates_tokens(session_manager):
    """Test that update_token_count accumulates tokens correctly."""
    session_id = await session_manager.start_session(
        task_id="T-008",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    # Update tokens multiple times
    result1 = session_manager.update_token_count(session_id, input_tokens=100, output_tokens=150)
    result2 = session_manager.update_token_count(session_id, input_tokens=200, output_tokens=250)

    # Verify cumulative counts
    assert result2["input_tokens"] == 300
    assert result2["output_tokens"] == 400
    assert result2["total_tokens"] == 700


@pytest.mark.asyncio
async def test_update_token_count_warns_at_threshold(session_manager, agent_config):
    """Test that update_token_count warns when approaching context limit."""
    session_id = await session_manager.start_session(
        task_id="T-009",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a test agent.",
    )

    # Use tokens exceeding warning threshold (80% of 200k = 160k)
    result = session_manager.update_token_count(
        session_id, input_tokens=80000, output_tokens=85000
    )

    assert result["context_warning"] is True
    assert result["usage_ratio"] >= agent_config.context_warning_threshold

    # Verify health status changed
    session_info = session_manager.get_session_info(session_id)
    assert session_info.health == HealthStatus.CONTEXT_WARNING


@pytest.mark.asyncio
async def test_update_token_count_for_nonexistent_session_raises_error(session_manager):
    """Test that updating tokens for non-existent session raises ValueError."""
    with pytest.raises(ValueError, match="Session not found"):
        session_manager.update_token_count("invalid-session-id", 100, 200)


# Result Schema Validation Tests


def test_test_result_validates_status():
    """Test that TestResult validates status field."""
    # Valid statuses
    result = TestResult(name="test1", status="passed")
    assert result.status == "passed"

    result = TestResult(name="test2", status="FAILED")  # Should normalize to lowercase
    assert result.status == "failed"

    # Invalid status
    with pytest.raises(ValueError, match="Invalid test status"):
        TestResult(name="test3", status="unknown")


def test_issue_discovered_validates_severity():
    """Test that IssueDiscovered validates severity field."""
    # Valid severities
    issue = IssueDiscovered(description="Test issue", severity="high")
    assert issue.severity == "high"

    issue = IssueDiscovered(description="Test issue", severity="CRITICAL")
    assert issue.severity == "critical"

    # Invalid severity
    with pytest.raises(ValueError, match="Invalid severity"):
        IssueDiscovered(description="Test issue", severity="urgent")


def test_agent_result_validates_status():
    """Test that AgentResult validates status field."""
    # Valid statuses
    result = AgentResult(status="success", summary="Done", details="All good")
    assert result.status == "success"

    result = AgentResult(status="PARTIAL", summary="Incomplete", details="Some issues")
    assert result.status == "partial"

    # Invalid status
    with pytest.raises(ValueError, match="Invalid status"):
        AgentResult(status="in_progress", summary="Test", details="Test")


def test_agent_result_with_all_fields():
    """Test AgentResult with all optional fields populated."""
    result = AgentResult(
        status="success",
        summary="Task completed",
        details="All files processed successfully",
        tests_run=[
            TestResult(name="test1", status="passed", duration_seconds=1.5),
            TestResult(name="test2", status="passed", duration_seconds=2.3),
        ],
        issues_discovered=[
            IssueDiscovered(
                description="Deprecated API",
                severity="medium",
                location="src/api.py:42",
                suggested_fix="Use new API endpoint",
            )
        ],
        lessons_learned=[
            LessonLearned(
                context="API migration",
                observation="Old API still works but is deprecated",
                recommendation="Update to new API in next sprint",
            )
        ],
        files_modified=["src/api.py", "tests/test_api.py"],
        confidence_score=0.95,
    )

    assert result.status == "success"
    assert len(result.tests_run) == 2
    assert len(result.issues_discovered) == 1
    assert len(result.lessons_learned) == 1
    assert len(result.files_modified) == 2
    assert result.confidence_score == 0.95


# Result Parsing Tests


def test_parse_agent_result_from_json_code_block():
    """Test parsing agent result from markdown JSON code block."""
    raw_output = """
    Here's my result:

    ```json
    {
        "status": "success",
        "summary": "Task completed",
        "details": "All work finished",
        "confidence_score": 1.0
    }
    ```

    Everything went well!
    """

    result = parse_agent_result(raw_output)

    assert result.status == "success"
    assert result.summary == "Task completed"
    assert result.details == "All work finished"
    assert result.confidence_score == 1.0


def test_parse_agent_result_from_plain_json():
    """Test parsing agent result from plain JSON without markdown."""
    raw_output = """
    {"status": "success", "summary": "Done", "details": "All good"}
    """

    result = parse_agent_result(raw_output)

    assert result.status == "success"
    assert result.summary == "Done"


def test_parse_agent_result_with_nested_objects():
    """Test parsing agent result with nested test results and issues."""
    raw_output = """
    ```json
    {
        "status": "success",
        "summary": "Tests passed",
        "details": "All tests completed successfully",
        "tests_run": [
            {"name": "test_feature_a", "status": "passed", "duration_seconds": 1.2},
            {"name": "test_feature_b", "status": "passed", "duration_seconds": 2.5}
        ],
        "issues_discovered": [
            {
                "description": "Potential memory leak",
                "severity": "medium",
                "location": "src/cache.py:89"
            }
        ]
    }
    ```
    """

    result = parse_agent_result(raw_output)

    assert result.status == "success"
    assert len(result.tests_run) == 2
    assert result.tests_run[0].name == "test_feature_a"
    assert result.tests_run[0].duration_seconds == 1.2
    assert len(result.issues_discovered) == 1
    assert result.issues_discovered[0].severity == "medium"


def test_parse_agent_result_raises_error_for_no_json():
    """Test that parse_agent_result raises error when no JSON found."""
    raw_output = "This is just plain text without any JSON"

    with pytest.raises(ValueError, match="No JSON found in agent output"):
        parse_agent_result(raw_output)


def test_parse_agent_result_raises_error_for_invalid_json():
    """Test that parse_agent_result raises error for malformed JSON."""
    raw_output = """
    ```json
    {"status": "success", "summary": "Done",
    ```
    """

    with pytest.raises(ValueError, match="Invalid JSON in agent output"):
        parse_agent_result(raw_output)


def test_parse_agent_result_safe_returns_fallback():
    """Test that parse_agent_result_safe returns fallback for malformed output."""
    raw_output = "No JSON here at all"

    result = parse_agent_result_safe(raw_output)

    assert result.status == "partial"
    assert result.summary == "Agent output could not be parsed"
    assert "No JSON here at all" in result.details
    assert result.confidence_score == 0.0


def test_parse_agent_result_safe_succeeds_with_valid_json():
    """Test that parse_agent_result_safe works with valid JSON."""
    raw_output = """
    ```json
    {"status": "success", "summary": "Done", "details": "All good"}
    ```
    """

    result = parse_agent_result_safe(raw_output)

    assert result.status == "success"
    assert result.summary == "Done"
    assert result.confidence_score == 1.0  # Default value


def test_create_minimal_result():
    """Test create_minimal_result helper function."""
    result = create_minimal_result(
        status="success",
        summary="Test completed",
        details="All tests passed",
        confidence=0.9,
    )

    assert result.status == "success"
    assert result.summary == "Test completed"
    assert result.details == "All tests passed"
    assert result.confidence_score == 0.9
    assert len(result.tests_run) == 0
    assert len(result.issues_discovered) == 0


def test_create_minimal_result_defaults_details():
    """Test that create_minimal_result defaults details to summary."""
    result = create_minimal_result(status="success", summary="Done")

    assert result.details == "Done"


# List Sessions Tests


@pytest.mark.asyncio
async def test_list_sessions_returns_all_sessions(session_manager):
    """Test that list_sessions returns all tracked sessions."""
    # Create multiple sessions
    session_id1 = await session_manager.start_session(
        task_id="T-010",
        agent_type="executor",
        model="claude-3-5-sonnet-20241022",
        system_prompt="Agent 1",
    )
    session_id2 = await session_manager.start_session(
        task_id="T-011",
        agent_type="architect",
        model="claude-3-5-sonnet-20241022",
        system_prompt="Agent 2",
    )

    sessions = session_manager.list_sessions()

    assert len(sessions) >= 2
    session_ids = [s.session_id for s in sessions]
    assert session_id1 in session_ids
    assert session_id2 in session_ids
