"""Integration tests for session handover protocol.

Tests cover the full handover lifecycle: context exhaustion detection,
save-and-exit prompt generation, handover context persistence, continuation
session spawning, and error handling. All external dependencies
(AgentSessionManager, database) are mocked.

Test coverage:
- P6-001: Context exhaustion detection (threshold, below, above, timeout)
- P6-001: Token estimation (various usage levels, different max tokens)
- P6-002: Save-and-exit prompt generation (structure, task context)
- P6-002: Save exit response parsing (valid JSON, markdown fences, fallback)
- P6-003: Handover context persistence (save, load, list pending)
- P6-004: Full handover initiation flow (detection -> prompt -> save -> end)
- P6-004: Continuation session spawning (new session, context injection)
- P6-004: Check and handle exhaustion (triggers, does not trigger)
- P6-001: HandoverReason enum values
- Error handling (session not found, failed save, failed spawn)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from forgemaster.config import AgentConfig
from forgemaster.orchestrator.handover import (
    ContextExhaustionDetector,
    HandoverContext,
    HandoverPromptGenerator,
    HandoverReason,
    HandoverStore,
    HandoverTrigger,
    SaveExitResponse,
    SessionHandoverManager,
)

# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeSessionMetrics:
    """Minimal fake of SessionMetrics for testing without importing real class."""

    input_tokens: int = 0
    output_tokens: int = 0
    messages_sent: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def duration_seconds(self) -> float:
        """Session duration in seconds."""
        return (datetime.now(UTC) - self.created_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Time since last activity."""
        return (datetime.now(UTC) - self.last_activity_at).total_seconds()


@dataclass
class FakeSessionInfo:
    """Minimal fake of SessionInfo for testing without importing real class."""

    session_id: str
    task_id: str
    agent_type: str = "executor"
    model: str = "sonnet"
    state: str = "active"
    health: str = "healthy"
    metrics: FakeSessionMetrics = field(default_factory=FakeSessionMetrics)
    sdk_session: object | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create an AgentConfig with defaults.

    Returns:
        AgentConfig with default thresholds.
    """
    return AgentConfig()


@pytest.fixture
def low_threshold_config() -> AgentConfig:
    """Create an AgentConfig with low context_warning_threshold for easy triggering.

    Returns:
        AgentConfig with 0.5 context warning threshold.
    """
    return AgentConfig(context_warning_threshold=0.5)


@pytest.fixture
def detector(agent_config: AgentConfig) -> ContextExhaustionDetector:
    """Create a ContextExhaustionDetector with default config.

    Args:
        agent_config: Default agent config.

    Returns:
        ContextExhaustionDetector instance.
    """
    return ContextExhaustionDetector(agent_config)


@pytest.fixture
def prompt_generator() -> HandoverPromptGenerator:
    """Create a HandoverPromptGenerator.

    Returns:
        HandoverPromptGenerator instance.
    """
    return HandoverPromptGenerator()


@pytest.fixture
def handover_store() -> HandoverStore:
    """Create a HandoverStore (in-memory).

    Returns:
        HandoverStore instance.
    """
    return HandoverStore()


@pytest.fixture
def mock_session_manager() -> MagicMock:
    """Create a mock AgentSessionManager.

    Returns:
        MagicMock with async methods matching AgentSessionManager interface.
    """
    manager = MagicMock()
    manager.get_session_info = MagicMock(
        return_value=FakeSessionInfo(
            session_id="session-abc",
            task_id="task-123",
            metrics=FakeSessionMetrics(input_tokens=150_000, output_tokens=10_000),
        )
    )
    manager.send_message = AsyncMock(
        return_value=json.dumps(
            {
                "task_id": "task-123",
                "progress_summary": "Completed 3 of 5 steps",
                "files_modified": ["src/main.py", "src/utils.py"],
                "remaining_work": ["Step 4", "Step 5"],
                "current_step": "Working on step 3",
                "context_data": {"key": "value"},
            }
        )
    )
    manager.end_session = AsyncMock(return_value={"session_id": "session-abc"})
    manager.start_session = AsyncMock(return_value="session-new-456")
    return manager


@pytest.fixture
def handover_manager(
    agent_config: AgentConfig,
    mock_session_manager: MagicMock,
    handover_store: HandoverStore,
) -> SessionHandoverManager:
    """Create a SessionHandoverManager with mocked dependencies.

    Args:
        agent_config: Agent configuration.
        mock_session_manager: Mocked session manager.
        handover_store: In-memory handover store.

    Returns:
        SessionHandoverManager instance.
    """
    return SessionHandoverManager(
        config=agent_config,
        session_manager=mock_session_manager,
        handover_store=handover_store,
    )


# ---------------------------------------------------------------------------
# P6-001: HandoverReason Enum
# ---------------------------------------------------------------------------


class TestHandoverReasonEnum:
    """Tests for HandoverReason enum values."""

    def test_context_exhaustion_value(self) -> None:
        """Test CONTEXT_EXHAUSTION has correct string value."""
        assert HandoverReason.CONTEXT_EXHAUSTION == "context_exhaustion"

    def test_session_timeout_value(self) -> None:
        """Test SESSION_TIMEOUT has correct string value."""
        assert HandoverReason.SESSION_TIMEOUT == "session_timeout"

    def test_manual_trigger_value(self) -> None:
        """Test MANUAL_TRIGGER has correct string value."""
        assert HandoverReason.MANUAL_TRIGGER == "manual_trigger"

    def test_error_recovery_value(self) -> None:
        """Test ERROR_RECOVERY has correct string value."""
        assert HandoverReason.ERROR_RECOVERY == "error_recovery"

    def test_all_reasons_are_str(self) -> None:
        """Test all HandoverReason values are strings."""
        for reason in HandoverReason:
            assert isinstance(reason.value, str)


# ---------------------------------------------------------------------------
# P6-001: Context Exhaustion Detection
# ---------------------------------------------------------------------------


class TestContextExhaustionDetector:
    """Tests for ContextExhaustionDetector."""

    def test_no_trigger_below_threshold(self, detector: ContextExhaustionDetector) -> None:
        """Test no handover trigger when usage is below threshold."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=50_000, output_tokens=10_000),
        )

        result = detector.should_trigger_handover(session_info)

        assert result is None

    def test_trigger_at_threshold(self, detector: ContextExhaustionDetector) -> None:
        """Test handover triggers when usage is exactly at threshold (0.8)."""
        # 0.8 * 200_000 = 160_000 tokens
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=130_000, output_tokens=30_000),
        )

        result = detector.should_trigger_handover(session_info)

        assert result is not None
        assert result.trigger_reason == HandoverReason.CONTEXT_EXHAUSTION
        assert result.session_id == "s-1"
        assert result.task_id == "t-1"

    def test_trigger_above_threshold(self, detector: ContextExhaustionDetector) -> None:
        """Test handover triggers when usage is above threshold."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=170_000, output_tokens=20_000),
        )

        result = detector.should_trigger_handover(session_info)

        assert result is not None
        assert result.trigger_reason == HandoverReason.CONTEXT_EXHAUSTION
        assert result.token_usage_ratio >= 0.8

    def test_trigger_with_low_threshold(self) -> None:
        """Test handover triggers with low threshold config."""
        config = AgentConfig(context_warning_threshold=0.3)
        det = ContextExhaustionDetector(config)

        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=40_000, output_tokens=25_000),
        )

        result = det.should_trigger_handover(session_info)

        # 65_000 / 200_000 = 0.325 > 0.3
        assert result is not None
        assert result.trigger_reason == HandoverReason.CONTEXT_EXHAUSTION

    def test_trigger_on_session_timeout(self) -> None:
        """Test handover triggers when session duration approaches timeout."""
        config = AgentConfig(session_timeout_seconds=100)
        det = ContextExhaustionDetector(config)

        # Create metrics with a start time 95 seconds ago (>90% of 100s)
        created_at = datetime.now(UTC) - timedelta(seconds=95)
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(
                input_tokens=1_000,
                output_tokens=500,
                created_at=created_at,
            ),
        )

        result = det.should_trigger_handover(session_info)

        assert result is not None
        assert result.trigger_reason == HandoverReason.SESSION_TIMEOUT

    def test_no_trigger_when_time_under_threshold(self) -> None:
        """Test no timeout trigger when duration is under 90% of timeout."""
        config = AgentConfig(session_timeout_seconds=1000)
        det = ContextExhaustionDetector(config)

        # Only 50 seconds old (5% of 1000s, well under 90%)
        created_at = datetime.now(UTC) - timedelta(seconds=50)
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(
                input_tokens=1_000,
                output_tokens=500,
                created_at=created_at,
            ),
        )

        result = det.should_trigger_handover(session_info)

        assert result is None

    def test_trigger_token_usage_ratio_capped_at_one(
        self, detector: ContextExhaustionDetector
    ) -> None:
        """Test that token_usage_ratio is capped at 1.0 even if over."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=180_000, output_tokens=50_000),
        )

        result = detector.should_trigger_handover(session_info)

        assert result is not None
        assert result.token_usage_ratio <= 1.0

    def test_custom_max_context_tokens(self, detector: ContextExhaustionDetector) -> None:
        """Test detection with a custom max_context_tokens value."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=800, output_tokens=200),
        )

        # With max 1000, 1000/1000 = 1.0 => should trigger
        result = detector.should_trigger_handover(session_info, max_context_tokens=1000)

        assert result is not None
        assert result.trigger_reason == HandoverReason.CONTEXT_EXHAUSTION


# ---------------------------------------------------------------------------
# P6-001: Token Estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    """Tests for token remaining estimation."""

    def test_estimate_with_default_max(self, detector: ContextExhaustionDetector) -> None:
        """Test token estimation with default 200k max."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=100_000, output_tokens=50_000),
        )

        remaining = detector.estimate_remaining_tokens(session_info)

        assert remaining == 50_000

    def test_estimate_with_custom_max(self, detector: ContextExhaustionDetector) -> None:
        """Test token estimation with custom max."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=3_000, output_tokens=2_000),
        )

        remaining = detector.estimate_remaining_tokens(session_info, max_context_tokens=10_000)

        assert remaining == 5_000

    def test_estimate_clamped_to_zero(self, detector: ContextExhaustionDetector) -> None:
        """Test estimation never goes below zero."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=150_000, output_tokens=100_000),
        )

        remaining = detector.estimate_remaining_tokens(session_info)

        assert remaining == 0

    def test_estimate_zero_tokens_used(self, detector: ContextExhaustionDetector) -> None:
        """Test estimation when no tokens have been used."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=0, output_tokens=0),
        )

        remaining = detector.estimate_remaining_tokens(session_info)

        assert remaining == 200_000

    def test_estimate_with_small_max(self, detector: ContextExhaustionDetector) -> None:
        """Test estimation with very small max context."""
        session_info = FakeSessionInfo(
            session_id="s-1",
            task_id="t-1",
            metrics=FakeSessionMetrics(input_tokens=50, output_tokens=30),
        )

        remaining = detector.estimate_remaining_tokens(session_info, max_context_tokens=100)

        assert remaining == 20


# ---------------------------------------------------------------------------
# P6-002: Save-and-Exit Prompt Generation
# ---------------------------------------------------------------------------


class TestHandoverPromptGenerator:
    """Tests for save-exit and continuation prompt generation."""

    def test_save_prompt_contains_task_id(self, prompt_generator: HandoverPromptGenerator) -> None:
        """Test save prompt includes the task ID."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="task-abc",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )

        prompt = prompt_generator.generate_save_prompt(trigger, {"description": "Test task"})

        assert "task-abc" in prompt

    def test_save_prompt_contains_reason(self, prompt_generator: HandoverPromptGenerator) -> None:
        """Test save prompt includes the trigger reason."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="task-abc",
            trigger_reason=HandoverReason.SESSION_TIMEOUT,
            token_usage_ratio=0.5,
            estimated_remaining_tokens=100_000,
        )

        prompt = prompt_generator.generate_save_prompt(trigger, {})

        assert "session_timeout" in prompt

    def test_save_prompt_contains_json_schema(
        self, prompt_generator: HandoverPromptGenerator
    ) -> None:
        """Test save prompt includes the expected JSON schema fields."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="task-abc",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )

        prompt = prompt_generator.generate_save_prompt(trigger, {})

        assert "progress_summary" in prompt
        assert "files_modified" in prompt
        assert "remaining_work" in prompt
        assert "current_step" in prompt
        assert "context_data" in prompt

    def test_save_prompt_includes_task_context(
        self, prompt_generator: HandoverPromptGenerator
    ) -> None:
        """Test save prompt includes task context fields."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="task-abc",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )
        context = {
            "description": "Implement authentication module",
            "files": ["src/auth.py", "src/middleware.py"],
            "branch": "feature/auth",
        }

        prompt = prompt_generator.generate_save_prompt(trigger, context)

        assert "Implement authentication module" in prompt
        assert "src/auth.py" in prompt
        assert "feature/auth" in prompt

    def test_save_prompt_with_empty_files(self, prompt_generator: HandoverPromptGenerator) -> None:
        """Test save prompt handles empty files list."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="task-abc",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )

        prompt = prompt_generator.generate_save_prompt(trigger, {"files": []})

        assert "(none)" in prompt

    def test_save_prompt_token_percentage(self, prompt_generator: HandoverPromptGenerator) -> None:
        """Test save prompt includes token usage percentage."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="task-abc",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )

        prompt = prompt_generator.generate_save_prompt(trigger, {})

        assert "85" in prompt  # 85%

    def test_continuation_prompt_contains_task_info(
        self, prompt_generator: HandoverPromptGenerator
    ) -> None:
        """Test continuation prompt includes task metadata."""
        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
            branch_name="feature/handover",
            worktree_path="/workspace/wt-1",
        )

        prompt = prompt_generator.generate_continuation_prompt(ctx)

        assert "task-123" in prompt
        assert "s-old" in prompt
        assert "feature/handover" in prompt
        assert "/workspace/wt-1" in prompt

    def test_continuation_prompt_with_save_response(
        self, prompt_generator: HandoverPromptGenerator
    ) -> None:
        """Test continuation prompt includes save-exit response details."""
        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        save_resp = SaveExitResponse(
            task_id="task-123",
            progress_summary="Completed steps 1-3",
            files_modified=["src/a.py", "src/b.py"],
            remaining_work=["Step 4", "Step 5"],
            current_step="Working on step 3",
            context_data={"iteration": 2},
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
            save_exit_response=save_resp,
        )

        prompt = prompt_generator.generate_continuation_prompt(ctx)

        assert "Completed steps 1-3" in prompt
        assert "src/a.py" in prompt
        assert "Step 4" in prompt
        assert "Working on step 3" in prompt
        assert '"iteration": 2' in prompt

    def test_continuation_prompt_without_save_response(
        self, prompt_generator: HandoverPromptGenerator
    ) -> None:
        """Test continuation prompt handles missing save response."""
        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
            save_exit_response=None,
        )

        prompt = prompt_generator.generate_continuation_prompt(ctx)

        assert "No save-exit response" in prompt
        assert "re-assess the task" in prompt


# ---------------------------------------------------------------------------
# P6-002: Save Exit Response Parsing
# ---------------------------------------------------------------------------


class TestSaveExitResponseParsing:
    """Tests for parsing agent save-exit responses."""

    def test_parse_valid_json(self, handover_manager: SessionHandoverManager) -> None:
        """Test parsing a valid JSON response."""
        raw = json.dumps(
            {
                "task_id": "t-1",
                "progress_summary": "Done step 1",
                "files_modified": ["a.py"],
                "remaining_work": ["step 2"],
                "current_step": "step 1",
                "context_data": {"x": 1},
            }
        )

        result = handover_manager._parse_save_response(raw, "t-1")

        assert result.task_id == "t-1"
        assert result.progress_summary == "Done step 1"
        assert result.files_modified == ["a.py"]
        assert result.remaining_work == ["step 2"]
        assert result.context_data == {"x": 1}

    def test_parse_json_in_markdown_fence(self, handover_manager: SessionHandoverManager) -> None:
        """Test parsing JSON wrapped in markdown code fences."""
        raw = '```json\n{"task_id": "t-1", "progress_summary": "Step done"}\n```'

        result = handover_manager._parse_save_response(raw, "t-1")

        assert result.task_id == "t-1"
        assert result.progress_summary == "Step done"

    def test_parse_json_in_plain_fence(self, handover_manager: SessionHandoverManager) -> None:
        """Test parsing JSON wrapped in plain code fences."""
        raw = '```\n{"task_id": "t-1", "progress_summary": "Done"}\n```'

        result = handover_manager._parse_save_response(raw, "t-1")

        assert result.task_id == "t-1"

    def test_parse_invalid_json_fallback(self, handover_manager: SessionHandoverManager) -> None:
        """Test fallback when response is not valid JSON."""
        raw = "I was working on the authentication module and made good progress."

        result = handover_manager._parse_save_response(raw, "t-1")

        assert result.task_id == "t-1"
        assert result.progress_summary == raw


# ---------------------------------------------------------------------------
# P6-003: Handover Context Persistence
# ---------------------------------------------------------------------------


class TestHandoverStore:
    """Tests for HandoverStore (in-memory)."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, handover_store: HandoverStore) -> None:
        """Test saving and loading a handover context."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="t-1",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )
        ctx = HandoverContext(
            session_id="s-1",
            task_id="t-1",
            trigger=trigger,
        )

        key = await handover_store.save_handover(ctx)
        loaded = await handover_store.load_handover(key)

        assert loaded is not None
        assert loaded.session_id == "s-1"
        assert loaded.task_id == "t-1"
        assert loaded.trigger.trigger_reason == HandoverReason.CONTEXT_EXHAUSTION

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, handover_store: HandoverStore) -> None:
        """Test loading a non-existent session ID returns None."""
        result = await handover_store.load_handover("does-not-exist")

        assert result is None

    @pytest.mark.asyncio
    async def test_save_returns_session_id(self, handover_store: HandoverStore) -> None:
        """Test that save_handover returns the session ID."""
        trigger = HandoverTrigger(
            session_id="s-42",
            task_id="t-1",
            trigger_reason=HandoverReason.MANUAL_TRIGGER,
            token_usage_ratio=0.5,
            estimated_remaining_tokens=100_000,
        )
        ctx = HandoverContext(session_id="s-42", task_id="t-1", trigger=trigger)

        key = await handover_store.save_handover(ctx)

        assert key == "s-42"

    @pytest.mark.asyncio
    async def test_list_pending_all(self, handover_store: HandoverStore) -> None:
        """Test listing all pending handovers (no continuation yet)."""
        for i in range(3):
            trigger = HandoverTrigger(
                session_id=f"s-{i}",
                task_id=f"t-{i}",
                trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
                token_usage_ratio=0.9,
                estimated_remaining_tokens=20_000,
            )
            ctx = HandoverContext(session_id=f"s-{i}", task_id=f"t-{i}", trigger=trigger)
            await handover_store.save_handover(ctx)

        pending = await handover_store.list_pending_handovers()

        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_list_pending_excludes_continued(self, handover_store: HandoverStore) -> None:
        """Test that pending list excludes handovers with continuation sessions."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="t-1",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )

        # One pending
        ctx1 = HandoverContext(session_id="s-1", task_id="t-1", trigger=trigger)
        await handover_store.save_handover(ctx1)

        # One already continued
        trigger2 = HandoverTrigger(
            session_id="s-2",
            task_id="t-2",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx2 = HandoverContext(
            session_id="s-2",
            task_id="t-2",
            trigger=trigger2,
            continuation_session_id="s-3",
        )
        await handover_store.save_handover(ctx2)

        pending = await handover_store.list_pending_handovers()

        assert len(pending) == 1
        assert pending[0].session_id == "s-1"

    @pytest.mark.asyncio
    async def test_list_pending_filter_by_task(self, handover_store: HandoverStore) -> None:
        """Test filtering pending handovers by task ID."""
        for i, tid in enumerate(["task-A", "task-B", "task-A"]):
            trigger = HandoverTrigger(
                session_id=f"s-{i}",
                task_id=tid,
                trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
                token_usage_ratio=0.9,
                estimated_remaining_tokens=20_000,
            )
            ctx = HandoverContext(session_id=f"s-{i}", task_id=tid, trigger=trigger)
            await handover_store.save_handover(ctx)

        pending_a = await handover_store.list_pending_handovers(task_id="task-A")
        pending_b = await handover_store.list_pending_handovers(task_id="task-B")

        assert len(pending_a) == 2
        assert len(pending_b) == 1

    @pytest.mark.asyncio
    async def test_list_pending_empty_store(self, handover_store: HandoverStore) -> None:
        """Test listing pending on empty store returns empty list."""
        pending = await handover_store.list_pending_handovers()

        assert pending == []

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self, handover_store: HandoverStore) -> None:
        """Test that saving with same session_id overwrites."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="t-1",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )
        ctx1 = HandoverContext(
            session_id="s-1",
            task_id="t-1",
            trigger=trigger,
            task_status_snapshot="in_progress",
        )
        await handover_store.save_handover(ctx1)

        ctx2 = HandoverContext(
            session_id="s-1",
            task_id="t-1",
            trigger=trigger,
            task_status_snapshot="completed",
            continuation_session_id="s-new",
        )
        await handover_store.save_handover(ctx2)

        loaded = await handover_store.load_handover("s-1")

        assert loaded is not None
        assert loaded.task_status_snapshot == "completed"
        assert loaded.continuation_session_id == "s-new"


# ---------------------------------------------------------------------------
# P6-004: Full Handover Initiation
# ---------------------------------------------------------------------------


class TestSessionHandoverManagerInitiation:
    """Tests for the full handover initiation flow."""

    @pytest.mark.asyncio
    async def test_initiate_handover_basic(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test basic handover initiation creates a HandoverContext."""
        ctx = await handover_manager.initiate_handover(
            session_id="session-abc",
            reason=HandoverReason.CONTEXT_EXHAUSTION,
        )

        assert isinstance(ctx, HandoverContext)
        assert ctx.session_id == "session-abc"
        assert ctx.task_id == "task-123"
        assert ctx.trigger.trigger_reason == HandoverReason.CONTEXT_EXHAUSTION

    @pytest.mark.asyncio
    async def test_initiate_handover_sends_save_prompt(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that initiation sends a save prompt to the session."""
        await handover_manager.initiate_handover("session-abc")

        mock_session_manager.send_message.assert_called_once()
        call_args = mock_session_manager.send_message.call_args
        assert call_args[0][0] == "session-abc"
        assert "Session Handover Required" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_initiate_handover_ends_session(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that initiation ends the old session."""
        await handover_manager.initiate_handover("session-abc")

        mock_session_manager.end_session.assert_called_once_with("session-abc", status="completed")

    @pytest.mark.asyncio
    async def test_initiate_handover_parses_response(
        self,
        handover_manager: SessionHandoverManager,
    ) -> None:
        """Test that initiation parses the save-exit response."""
        ctx = await handover_manager.initiate_handover("session-abc")

        assert ctx.save_exit_response is not None
        assert ctx.save_exit_response.task_id == "task-123"
        assert ctx.save_exit_response.progress_summary == "Completed 3 of 5 steps"
        assert ctx.save_exit_response.files_modified == ["src/main.py", "src/utils.py"]

    @pytest.mark.asyncio
    async def test_initiate_handover_persists_context(
        self,
        handover_manager: SessionHandoverManager,
        handover_store: HandoverStore,
    ) -> None:
        """Test that initiation persists the handover context."""
        await handover_manager.initiate_handover("session-abc")

        loaded = await handover_store.load_handover("session-abc")

        assert loaded is not None
        assert loaded.task_id == "task-123"

    @pytest.mark.asyncio
    async def test_initiate_handover_with_task_context(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test initiation with explicit task context."""
        task_ctx = {
            "description": "Build auth module",
            "files": ["src/auth.py"],
            "branch": "feature/auth",
        }

        await handover_manager.initiate_handover(
            "session-abc",
            task_context=task_ctx,
        )

        call_args = mock_session_manager.send_message.call_args
        prompt = call_args[0][1]
        assert "Build auth module" in prompt
        assert "src/auth.py" in prompt

    @pytest.mark.asyncio
    async def test_initiate_handover_send_message_fails(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test initiation continues even if send_message fails."""
        mock_session_manager.send_message.side_effect = RuntimeError("agent dead")

        ctx = await handover_manager.initiate_handover("session-abc")

        # Should still create context, just without save response
        assert ctx.save_exit_response is None
        mock_session_manager.end_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_initiate_handover_end_session_fails(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test initiation continues even if end_session fails."""
        mock_session_manager.end_session.side_effect = RuntimeError("already ended")

        ctx = await handover_manager.initiate_handover("session-abc")

        # Should still save the handover context
        assert ctx.session_id == "session-abc"

    @pytest.mark.asyncio
    async def test_initiate_handover_session_not_found(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test initiation raises when session is not found."""
        mock_session_manager.get_session_info.side_effect = ValueError("Session not found")

        with pytest.raises(ValueError, match="Session not found"):
            await handover_manager.initiate_handover("nonexistent-session")

    @pytest.mark.asyncio
    async def test_initiate_handover_manual_reason(
        self,
        handover_manager: SessionHandoverManager,
    ) -> None:
        """Test initiation with manual trigger reason."""
        ctx = await handover_manager.initiate_handover(
            "session-abc",
            reason=HandoverReason.MANUAL_TRIGGER,
        )

        assert ctx.trigger.trigger_reason == HandoverReason.MANUAL_TRIGGER


# ---------------------------------------------------------------------------
# P6-004: Continuation Session Spawning
# ---------------------------------------------------------------------------


class TestSpawnContinuation:
    """Tests for continuation session spawning."""

    @pytest.mark.asyncio
    async def test_spawn_creates_new_session(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that spawn_continuation creates a new session."""
        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
        )

        new_id = await handover_manager.spawn_continuation(ctx)

        assert new_id == "session-new-456"
        mock_session_manager.start_session.assert_called_once()
        call_kwargs = mock_session_manager.start_session.call_args.kwargs
        assert call_kwargs["task_id"] == "task-123"

    @pytest.mark.asyncio
    async def test_spawn_sends_continuation_prompt(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that spawn sends a continuation prompt to the new session."""
        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
        )

        await handover_manager.spawn_continuation(ctx)

        # send_message called for the continuation prompt
        mock_session_manager.send_message.assert_called_once()
        call_args = mock_session_manager.send_message.call_args
        assert call_args[0][0] == "session-new-456"
        assert "Session Continuation" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_spawn_updates_context_with_new_id(
        self,
        handover_manager: SessionHandoverManager,
        handover_store: HandoverStore,
    ) -> None:
        """Test that spawn updates the handover context with continuation ID."""
        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
        )

        await handover_manager.spawn_continuation(ctx)

        assert ctx.continuation_session_id == "session-new-456"

        # Also persisted in the store
        loaded = await handover_store.load_handover("s-old")
        assert loaded is not None
        assert loaded.continuation_session_id == "session-new-456"

    @pytest.mark.asyncio
    async def test_spawn_with_custom_agent_type_and_model(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test spawn with custom agent_type and model."""
        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
        )

        await handover_manager.spawn_continuation(ctx, agent_type="executor", model="opus")

        call_kwargs = mock_session_manager.start_session.call_args.kwargs
        assert call_kwargs["agent_type"] == "executor"
        assert call_kwargs["model"] == "opus"

    @pytest.mark.asyncio
    async def test_spawn_default_agent_type_and_model(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test spawn uses default agent_type and model."""
        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
        )

        await handover_manager.spawn_continuation(ctx)

        call_kwargs = mock_session_manager.start_session.call_args.kwargs
        assert call_kwargs["agent_type"] == "continuation"
        assert call_kwargs["model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_spawn_raises_on_send_failure(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test spawn raises when continuation prompt send fails."""
        mock_session_manager.send_message.side_effect = RuntimeError("send failed")

        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
        )

        with pytest.raises(RuntimeError, match="send failed"):
            await handover_manager.spawn_continuation(ctx)

    @pytest.mark.asyncio
    async def test_spawn_raises_on_start_failure(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test spawn raises when session creation fails."""
        mock_session_manager.start_session.side_effect = RuntimeError("creation failed")

        trigger = HandoverTrigger(
            session_id="s-old",
            task_id="task-123",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.9,
            estimated_remaining_tokens=20_000,
        )
        ctx = HandoverContext(
            session_id="s-old",
            task_id="task-123",
            trigger=trigger,
        )

        with pytest.raises(RuntimeError, match="creation failed"):
            await handover_manager.spawn_continuation(ctx)


# ---------------------------------------------------------------------------
# P6-004: Check and Handle Exhaustion
# ---------------------------------------------------------------------------


class TestCheckAndHandleExhaustion:
    """Tests for the convenience check_and_handle_exhaustion method."""

    @pytest.mark.asyncio
    async def test_triggers_when_exhausted(
        self,
        handover_manager: SessionHandoverManager,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test check_and_handle returns context when exhaustion detected."""
        session_info = FakeSessionInfo(
            session_id="session-abc",
            task_id="task-123",
            metrics=FakeSessionMetrics(input_tokens=170_000, output_tokens=10_000),
        )

        result = await handover_manager.check_and_handle_exhaustion(session_info)

        assert result is not None
        assert isinstance(result, HandoverContext)
        assert result.task_id == "task-123"

    @pytest.mark.asyncio
    async def test_returns_none_when_healthy(
        self,
        handover_manager: SessionHandoverManager,
    ) -> None:
        """Test check_and_handle returns None when no exhaustion."""
        session_info = FakeSessionInfo(
            session_id="session-abc",
            task_id="task-123",
            metrics=FakeSessionMetrics(input_tokens=10_000, output_tokens=5_000),
        )

        result = await handover_manager.check_and_handle_exhaustion(session_info)

        assert result is None


# ---------------------------------------------------------------------------
# Pydantic Model Tests
# ---------------------------------------------------------------------------


class TestPydanticModels:
    """Tests for Pydantic model defaults and validation."""

    def test_handover_trigger_defaults(self) -> None:
        """Test HandoverTrigger default values."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="t-1",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )

        assert trigger.triggered_at is not None
        assert len(trigger.triggered_at) > 0

    def test_save_exit_response_defaults(self) -> None:
        """Test SaveExitResponse default values."""
        resp = SaveExitResponse(
            task_id="t-1",
            progress_summary="Some progress",
        )

        assert resp.files_modified == []
        assert resp.remaining_work == []
        assert resp.current_step == ""
        assert resp.context_data == {}

    def test_handover_context_defaults(self) -> None:
        """Test HandoverContext default values."""
        trigger = HandoverTrigger(
            session_id="s-1",
            task_id="t-1",
            trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
            token_usage_ratio=0.85,
            estimated_remaining_tokens=30_000,
        )
        ctx = HandoverContext(
            session_id="s-1",
            task_id="t-1",
            trigger=trigger,
        )

        assert ctx.save_exit_response is None
        assert ctx.task_status_snapshot == "in_progress"
        assert ctx.branch_name is None
        assert ctx.worktree_path is None
        assert ctx.continuation_session_id is None
        assert ctx.created_at is not None

    def test_handover_trigger_ratio_validation(self) -> None:
        """Test HandoverTrigger rejects ratio > 1.0."""
        with pytest.raises(ValueError):
            HandoverTrigger(
                session_id="s-1",
                task_id="t-1",
                trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
                token_usage_ratio=1.5,
                estimated_remaining_tokens=0,
            )

    def test_handover_trigger_negative_ratio_rejected(self) -> None:
        """Test HandoverTrigger rejects negative ratio."""
        with pytest.raises(ValueError):
            HandoverTrigger(
                session_id="s-1",
                task_id="t-1",
                trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
                token_usage_ratio=-0.1,
                estimated_remaining_tokens=0,
            )

    def test_handover_trigger_negative_remaining_rejected(self) -> None:
        """Test HandoverTrigger rejects negative remaining tokens."""
        with pytest.raises(ValueError):
            HandoverTrigger(
                session_id="s-1",
                task_id="t-1",
                trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
                token_usage_ratio=0.5,
                estimated_remaining_tokens=-100,
            )
