"""Unit tests for review cycle orchestration.

Tests the ReviewCycleOrchestrator class, state machine, trigger logic,
task generation, and result aggregation for the review cycle system.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.review.cycle import (
    VALID_REVIEW_TRANSITIONS,
    FindingSeverity,
    InvalidReviewTransitionError,
    ReviewCycle,
    ReviewCycleOrchestrator,
    ReviewCycleState,
    ReviewFinding,
    ReviewResult,
    ReviewTrigger,
    ReviewTriggerConfig,
    validate_review_transition,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Test enum definitions and values."""

    def test_review_cycle_state_enum(self) -> None:
        """Test that ReviewCycleState has all expected values."""
        assert ReviewCycleState.PENDING.value == "pending"
        assert ReviewCycleState.COLLECTING.value == "collecting"
        assert ReviewCycleState.DISPATCHING.value == "dispatching"
        assert ReviewCycleState.IN_PROGRESS.value == "in_progress"
        assert ReviewCycleState.AGGREGATING.value == "aggregating"
        assert ReviewCycleState.COMPLETED.value == "completed"
        assert ReviewCycleState.FAILED.value == "failed"

    def test_review_trigger_enum(self) -> None:
        """Test that ReviewTrigger has all expected values."""
        assert ReviewTrigger.PHASE_COMPLETE.value == "phase_complete"
        assert ReviewTrigger.TASK_COUNT.value == "task_count"
        assert ReviewTrigger.ON_DEMAND.value == "on_demand"
        assert ReviewTrigger.SCHEDULED.value == "scheduled"

    def test_finding_severity_enum(self) -> None:
        """Test that FindingSeverity has all expected values."""
        assert FindingSeverity.CRITICAL.value == "critical"
        assert FindingSeverity.HIGH.value == "high"
        assert FindingSeverity.MEDIUM.value == "medium"
        assert FindingSeverity.LOW.value == "low"
        assert FindingSeverity.INFO.value == "info"


# ---------------------------------------------------------------------------
# State Transition Tests
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Test state transition validation."""

    def test_valid_review_transitions_structure(self) -> None:
        """Test that VALID_REVIEW_TRANSITIONS contains all states."""
        for state in ReviewCycleState:
            assert state in VALID_REVIEW_TRANSITIONS

    def test_validate_pending_transitions(self) -> None:
        """Test valid transitions from PENDING state."""
        assert validate_review_transition(
            ReviewCycleState.PENDING, ReviewCycleState.COLLECTING
        )
        assert validate_review_transition(
            ReviewCycleState.PENDING, ReviewCycleState.FAILED
        )
        assert not validate_review_transition(
            ReviewCycleState.PENDING, ReviewCycleState.IN_PROGRESS
        )

    def test_validate_collecting_transitions(self) -> None:
        """Test valid transitions from COLLECTING state."""
        assert validate_review_transition(
            ReviewCycleState.COLLECTING, ReviewCycleState.DISPATCHING
        )
        assert validate_review_transition(
            ReviewCycleState.COLLECTING, ReviewCycleState.FAILED
        )
        assert not validate_review_transition(
            ReviewCycleState.COLLECTING, ReviewCycleState.COMPLETED
        )

    def test_validate_dispatching_transitions(self) -> None:
        """Test valid transitions from DISPATCHING state."""
        assert validate_review_transition(
            ReviewCycleState.DISPATCHING, ReviewCycleState.IN_PROGRESS
        )
        assert validate_review_transition(
            ReviewCycleState.DISPATCHING, ReviewCycleState.FAILED
        )
        assert not validate_review_transition(
            ReviewCycleState.DISPATCHING, ReviewCycleState.COLLECTING
        )

    def test_validate_in_progress_transitions(self) -> None:
        """Test valid transitions from IN_PROGRESS state."""
        assert validate_review_transition(
            ReviewCycleState.IN_PROGRESS, ReviewCycleState.AGGREGATING
        )
        assert validate_review_transition(
            ReviewCycleState.IN_PROGRESS, ReviewCycleState.FAILED
        )
        assert not validate_review_transition(
            ReviewCycleState.IN_PROGRESS, ReviewCycleState.PENDING
        )

    def test_validate_aggregating_transitions(self) -> None:
        """Test valid transitions from AGGREGATING state."""
        assert validate_review_transition(
            ReviewCycleState.AGGREGATING, ReviewCycleState.COMPLETED
        )
        assert validate_review_transition(
            ReviewCycleState.AGGREGATING, ReviewCycleState.FAILED
        )
        assert not validate_review_transition(
            ReviewCycleState.AGGREGATING, ReviewCycleState.IN_PROGRESS
        )

    def test_validate_terminal_states_no_transitions(self) -> None:
        """Test that terminal states (COMPLETED, FAILED) have no valid transitions."""
        for target in ReviewCycleState:
            assert not validate_review_transition(ReviewCycleState.COMPLETED, target)
            assert not validate_review_transition(ReviewCycleState.FAILED, target)

    def test_invalid_review_transition_error(self) -> None:
        """Test InvalidReviewTransitionError construction."""
        error = InvalidReviewTransitionError(
            ReviewCycleState.PENDING,
            ReviewCycleState.COMPLETED,
            "test-cycle-id",
        )
        assert error.current == ReviewCycleState.PENDING
        assert error.target == ReviewCycleState.COMPLETED
        assert error.cycle_id == "test-cycle-id"
        assert "pending" in str(error)
        assert "completed" in str(error)
        assert "test-cycle-id" in str(error)

    def test_invalid_review_transition_error_no_cycle_id(self) -> None:
        """Test InvalidReviewTransitionError without cycle_id."""
        error = InvalidReviewTransitionError(
            ReviewCycleState.COLLECTING,
            ReviewCycleState.PENDING,
        )
        assert error.cycle_id is None
        assert "collecting" in str(error)
        assert "pending" in str(error)


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestModels:
    """Test Pydantic model construction and defaults."""

    def test_review_finding_defaults(self) -> None:
        """Test ReviewFinding model with defaults."""
        finding = ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.HIGH,
            title="SQL Injection Risk",
            description="Unsanitized user input in query",
        )
        assert finding.reviewer_type == "security"
        assert finding.severity == FindingSeverity.HIGH
        assert finding.title == "SQL Injection Risk"
        assert finding.description == "Unsanitized user input in query"
        assert finding.file_path is None
        assert finding.line_number is None
        assert finding.suggested_fix is None
        assert finding.category is None
        # Check that ID was generated
        assert uuid.UUID(finding.id)

    def test_review_finding_full(self) -> None:
        """Test ReviewFinding with all fields."""
        finding = ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.MEDIUM,
            title="Missing error handling",
            description="Function does not handle exceptions",
            file_path="src/api/routes.py",
            line_number=42,
            suggested_fix="Add try/except block",
            category="error_handling",
        )
        assert finding.file_path == "src/api/routes.py"
        assert finding.line_number == 42
        assert finding.suggested_fix == "Add try/except block"
        assert finding.category == "error_handling"

    def test_review_result_defaults(self) -> None:
        """Test ReviewResult model with defaults."""
        result = ReviewResult(
            reviewer_type="spec",
            task_id="task-123",
            summary="All checks passed",
            passed=True,
        )
        assert result.reviewer_type == "spec"
        assert result.task_id == "task-123"
        assert result.summary == "All checks passed"
        assert result.passed is True
        assert result.findings == []
        assert result.confidence_score == 1.0

    def test_review_result_with_findings(self) -> None:
        """Test ReviewResult with multiple findings."""
        finding1 = ReviewFinding(
            reviewer_type="errors",
            severity=FindingSeverity.LOW,
            title="Bare except",
            description="Bare except clause found",
        )
        finding2 = ReviewFinding(
            reviewer_type="errors",
            severity=FindingSeverity.HIGH,
            title="Swallowed exception",
            description="Exception caught and ignored",
        )
        result = ReviewResult(
            reviewer_type="errors",
            task_id="task-456",
            findings=[finding1, finding2],
            summary="Found 2 issues",
            passed=False,
            confidence_score=0.85,
        )
        assert len(result.findings) == 2
        assert result.passed is False
        assert result.confidence_score == 0.85

    def test_review_cycle_defaults(self) -> None:
        """Test ReviewCycle model with defaults."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.TASK_COUNT,
        )
        assert cycle.project_id == "proj-123"
        assert cycle.trigger == ReviewTrigger.TASK_COUNT
        assert cycle.state == ReviewCycleState.PENDING
        assert cycle.tasks_under_review == []
        assert cycle.reviewer_types == []
        assert cycle.review_task_ids == []
        assert cycle.results == []
        assert cycle.all_findings == []
        assert cycle.started_at is None
        assert cycle.completed_at is None
        # Check that cycle_id and created_at were generated
        assert uuid.UUID(cycle.cycle_id)
        assert isinstance(cycle.created_at, datetime)

    def test_review_cycle_full(self) -> None:
        """Test ReviewCycle with all fields populated."""
        cycle = ReviewCycle(
            project_id="proj-456",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.IN_PROGRESS,
            tasks_under_review=["task-1", "task-2"],
            reviewer_types=["security", "backend"],
            review_task_ids=["review-1", "review-2"],
        )
        assert len(cycle.tasks_under_review) == 2
        assert len(cycle.reviewer_types) == 2
        assert len(cycle.review_task_ids) == 2

    def test_review_trigger_config_defaults(self) -> None:
        """Test ReviewTriggerConfig defaults."""
        config = ReviewTriggerConfig()
        assert config.task_count_threshold == 10
        assert config.enable_phase_trigger is True
        assert config.enable_scheduled_trigger is False
        assert config.schedule_interval_hours == 24
        assert config.default_reviewer_types == ["backend", "security", "spec", "errors"]

    def test_review_trigger_config_custom(self) -> None:
        """Test ReviewTriggerConfig with custom values."""
        config = ReviewTriggerConfig(
            task_count_threshold=5,
            enable_phase_trigger=False,
            enable_scheduled_trigger=True,
            schedule_interval_hours=12,
            default_reviewer_types=["security"],
        )
        assert config.task_count_threshold == 5
        assert config.enable_phase_trigger is False
        assert config.enable_scheduled_trigger is True
        assert config.schedule_interval_hours == 12
        assert config.default_reviewer_types == ["security"]


# ---------------------------------------------------------------------------
# Orchestrator Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session_factory() -> Any:
    """Create a mock session factory for testing."""

    @asynccontextmanager
    async def factory() -> Any:
        session = AsyncMock()
        yield session

    return factory


@pytest.fixture
def orchestrator(mock_session_factory: Any) -> ReviewCycleOrchestrator:
    """Create a ReviewCycleOrchestrator for testing."""
    return ReviewCycleOrchestrator(mock_session_factory)


@pytest.fixture
def custom_config_orchestrator(mock_session_factory: Any) -> ReviewCycleOrchestrator:
    """Create a ReviewCycleOrchestrator with custom trigger config."""
    config = ReviewTriggerConfig(
        task_count_threshold=5,
        enable_scheduled_trigger=True,
        schedule_interval_hours=12,
        default_reviewer_types=["security", "backend"],
    )
    return ReviewCycleOrchestrator(mock_session_factory, config)


class TestOrchestratorLifecycle:
    """Test review cycle lifecycle methods."""

    @pytest.mark.asyncio
    async def test_create_cycle_minimal(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test creating a review cycle with minimal parameters."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        assert cycle.project_id == "proj-123"
        assert cycle.trigger == ReviewTrigger.ON_DEMAND
        assert cycle.state == ReviewCycleState.PENDING
        assert cycle.tasks_under_review == []
        assert cycle.reviewer_types == ["backend", "security", "spec", "errors"]

    @pytest.mark.asyncio
    async def test_create_cycle_with_tasks(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test creating a review cycle with explicit tasks."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.TASK_COUNT,
            tasks_to_review=["task-1", "task-2", "task-3"],
        )
        assert cycle.tasks_under_review == ["task-1", "task-2", "task-3"]

    @pytest.mark.asyncio
    async def test_create_cycle_with_reviewer_types(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test creating a review cycle with custom reviewer types."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.PHASE_COMPLETE,
            reviewer_types=["security", "performance"],
        )
        assert cycle.reviewer_types == ["security", "performance"]

    @pytest.mark.asyncio
    async def test_create_cycle_uses_custom_config(
        self, custom_config_orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that custom trigger config is used for default reviewer types."""
        cycle = await custom_config_orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        assert cycle.reviewer_types == ["security", "backend"]

    @pytest.mark.asyncio
    async def test_start_cycle_from_pending(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test starting a cycle from PENDING state."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            tasks_to_review=["task-1", "task-2"],
            reviewer_types=["security"],
        )

        # Mock the file collection
        orchestrator._collect_files_for_tasks = AsyncMock(
            return_value=["file1.py", "file2.py"]
        )

        started_cycle = await orchestrator.start_cycle(cycle.cycle_id)

        assert started_cycle.state == ReviewCycleState.IN_PROGRESS
        assert started_cycle.started_at is not None
        assert len(started_cycle.review_task_ids) == 1

    @pytest.mark.asyncio
    async def test_start_cycle_auto_collects_tasks(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that start_cycle auto-collects tasks if none provided."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.SCHEDULED,
            reviewer_types=["backend"],
        )

        # Mock task and file collection
        orchestrator._collect_completed_tasks = AsyncMock(
            return_value=["task-1", "task-2", "task-3"]
        )
        orchestrator._collect_files_for_tasks = AsyncMock(
            return_value=["file1.py", "file2.py"]
        )

        started_cycle = await orchestrator.start_cycle(cycle.cycle_id)

        assert started_cycle.tasks_under_review == ["task-1", "task-2", "task-3"]
        orchestrator._collect_completed_tasks.assert_called_once_with("proj-123")

    @pytest.mark.asyncio
    async def test_start_cycle_invalid_id(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test starting a cycle with invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="Review cycle .* not found"):
            await orchestrator.start_cycle("nonexistent-cycle-id")

    @pytest.mark.asyncio
    async def test_start_cycle_invalid_state(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test starting a cycle not in PENDING state raises error."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            tasks_to_review=["task-1"],
            reviewer_types=["security"],
        )

        # Mock file collection
        orchestrator._collect_files_for_tasks = AsyncMock(return_value=["file1.py"])

        # Start once (moves to IN_PROGRESS)
        await orchestrator.start_cycle(cycle.cycle_id)

        # Try to start again
        with pytest.raises(InvalidReviewTransitionError):
            await orchestrator.start_cycle(cycle.cycle_id)


class TestTriggerLogic:
    """Test review trigger detection logic."""

    @pytest.mark.asyncio
    async def test_record_task_completion(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test recording task completions increments counter."""
        count1 = orchestrator.record_task_completion("proj-123")
        assert count1 == 1

        count2 = orchestrator.record_task_completion("proj-123")
        assert count2 == 2

        count3 = orchestrator.record_task_completion("proj-123")
        assert count3 == 3

    @pytest.mark.asyncio
    async def test_reset_task_counter(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test resetting task counter."""
        orchestrator.record_task_completion("proj-123")
        orchestrator.record_task_completion("proj-123")
        orchestrator.record_task_completion("proj-123")

        orchestrator.reset_task_counter("proj-123")

        assert orchestrator._tasks_since_last_review["proj-123"] == 0
        assert "proj-123" in orchestrator._last_review_time

    @pytest.mark.asyncio
    async def test_check_triggers_task_count(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that task count trigger fires at threshold."""
        project_id = "proj-123"

        # Below threshold
        for _ in range(9):
            orchestrator.record_task_completion(project_id)

        trigger = await orchestrator.check_triggers(project_id)
        assert trigger is None

        # At threshold
        orchestrator.record_task_completion(project_id)
        trigger = await orchestrator.check_triggers(project_id)
        assert trigger == ReviewTrigger.TASK_COUNT

    @pytest.mark.asyncio
    async def test_check_triggers_scheduled(
        self, custom_config_orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that scheduled trigger fires after interval."""
        project_id = "proj-123"

        # Set last review time to 13 hours ago (interval is 12 hours)
        past_time = datetime.now(timezone.utc) - timedelta(hours=13)
        custom_config_orchestrator._last_review_time[project_id] = past_time

        trigger = await custom_config_orchestrator.check_triggers(project_id)
        assert trigger == ReviewTrigger.SCHEDULED

    @pytest.mark.asyncio
    async def test_check_triggers_scheduled_not_elapsed(
        self, custom_config_orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that scheduled trigger does not fire before interval."""
        project_id = "proj-123"

        # Set last review time to 1 hour ago (interval is 12 hours)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        custom_config_orchestrator._last_review_time[project_id] = recent_time

        trigger = await custom_config_orchestrator.check_triggers(project_id)
        assert trigger is None

    @pytest.mark.asyncio
    async def test_check_triggers_no_trigger(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that no trigger returns None."""
        project_id = "proj-123"

        trigger = await orchestrator.check_triggers(project_id)
        assert trigger is None

    @pytest.mark.asyncio
    async def test_should_trigger_review_true(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test should_trigger_review returns True when trigger fires."""
        project_id = "proj-123"

        # Hit threshold
        for _ in range(10):
            orchestrator.record_task_completion(project_id)

        should_trigger = await orchestrator.should_trigger_review(project_id)
        assert should_trigger is True

    @pytest.mark.asyncio
    async def test_should_trigger_review_false(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test should_trigger_review returns False when no trigger."""
        project_id = "proj-123"

        should_trigger = await orchestrator.should_trigger_review(project_id)
        assert should_trigger is False


class TestTaskGeneration:
    """Test review task generation."""

    @pytest.mark.asyncio
    async def test_generate_review_tasks(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test generating review tasks for multiple reviewer types."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            tasks_under_review=["task-1", "task-2"],
            reviewer_types=["security", "backend", "spec"],
        )
        orchestrator._cycles[cycle.cycle_id] = cycle

        orchestrator._collect_files_for_tasks = AsyncMock(
            return_value=["file1.py", "file2.py", "file3.py"]
        )

        task_ids = await orchestrator.generate_review_tasks(cycle)

        assert len(task_ids) == 3
        for task_id in task_ids:
            assert uuid.UUID(task_id)

    @pytest.mark.asyncio
    async def test_build_review_task_structure(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that _build_review_task creates correct structure."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.TASK_COUNT,
            tasks_under_review=["task-1", "task-2"],
        )

        files = ["src/api/routes.py", "src/models/user.py"]
        task_def = orchestrator._build_review_task(cycle, "security", files)

        assert "task_id" in task_def
        assert uuid.UUID(task_def["task_id"])
        assert task_def["title"].startswith("Review: security review")
        assert "security" in task_def["description"].lower()
        assert "src/api/routes.py" in task_def["description"]
        assert "src/models/user.py" in task_def["description"]
        assert task_def["agent_type"] == "reviewer_security"
        assert task_def["model_tier"] == "sonnet"
        assert task_def["priority"] == 50
        assert task_def["files_touched"] == files
        assert task_def["metadata"]["review_cycle_id"] == cycle.cycle_id
        assert task_def["metadata"]["reviewer_type"] == "security"

    @pytest.mark.asyncio
    async def test_build_review_task_known_reviewers(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that known reviewer types get specific descriptions."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )

        known_types = ["backend", "security", "spec", "errors", "frontend", "performance"]

        for reviewer_type in known_types:
            task_def = orchestrator._build_review_task(cycle, reviewer_type, [])
            assert reviewer_type in task_def["description"].lower()
            assert task_def["agent_type"] == f"reviewer_{reviewer_type}"

    @pytest.mark.asyncio
    async def test_build_review_task_unknown_reviewer(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that unknown reviewer type gets generic description."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )

        task_def = orchestrator._build_review_task(cycle, "custom_reviewer", [])
        assert "custom_reviewer specialist" in task_def["description"]

    @pytest.mark.asyncio
    async def test_generate_review_tasks_no_files(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test generating review tasks when no files are found."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            tasks_under_review=["task-1"],
            reviewer_types=["security"],
        )
        orchestrator._cycles[cycle.cycle_id] = cycle

        orchestrator._collect_files_for_tasks = AsyncMock(return_value=[])

        task_ids = await orchestrator.generate_review_tasks(cycle)

        assert len(task_ids) == 1
        # Should still generate task even with no files


class TestResultAggregation:
    """Test review result submission and aggregation."""

    @pytest.mark.asyncio
    async def test_submit_result(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test submitting a review result."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.IN_PROGRESS,
        )
        orchestrator._cycles[cycle.cycle_id] = cycle

        finding = ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.HIGH,
            title="SQL Injection",
            description="User input not sanitized",
        )
        result = ReviewResult(
            reviewer_type="security",
            task_id="review-task-1",
            findings=[finding],
            summary="Found 1 issue",
            passed=False,
        )

        await orchestrator.submit_result(cycle.cycle_id, result)

        assert len(cycle.results) == 1
        assert cycle.results[0] == result
        assert len(cycle.all_findings) == 1
        assert cycle.all_findings[0] == finding

    @pytest.mark.asyncio
    async def test_submit_result_invalid_cycle(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test submitting result to invalid cycle raises ValueError."""
        result = ReviewResult(
            reviewer_type="security",
            task_id="review-task-1",
            summary="Summary",
            passed=True,
        )

        with pytest.raises(ValueError, match="Review cycle .* not found"):
            await orchestrator.submit_result("nonexistent-cycle", result)

    @pytest.mark.asyncio
    async def test_aggregate_results_partial(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test aggregating when not all reviewers have reported."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.IN_PROGRESS,
            reviewer_types=["security", "backend", "spec"],
        )
        orchestrator._cycles[cycle.cycle_id] = cycle

        # Submit result for only security
        result = ReviewResult(
            reviewer_type="security",
            task_id="review-task-1",
            summary="Security check passed",
            passed=True,
        )
        await orchestrator.submit_result(cycle.cycle_id, result)

        # Aggregate (should remain IN_PROGRESS)
        aggregated = await orchestrator.aggregate_results(cycle.cycle_id)

        assert aggregated.state == ReviewCycleState.IN_PROGRESS
        assert aggregated.completed_at is None

    @pytest.mark.asyncio
    async def test_aggregate_results_complete(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test aggregating when all reviewers have reported."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.IN_PROGRESS,
            reviewer_types=["security", "backend"],
        )
        orchestrator._cycles[cycle.cycle_id] = cycle

        # Submit both results
        result1 = ReviewResult(
            reviewer_type="security",
            task_id="review-task-1",
            summary="Security passed",
            passed=True,
        )
        result2 = ReviewResult(
            reviewer_type="backend",
            task_id="review-task-2",
            summary="Backend passed",
            passed=True,
        )

        await orchestrator.submit_result(cycle.cycle_id, result1)
        await orchestrator.submit_result(cycle.cycle_id, result2)

        # Aggregate
        aggregated = await orchestrator.aggregate_results(cycle.cycle_id)

        assert aggregated.state == ReviewCycleState.COMPLETED
        assert aggregated.completed_at is not None

    @pytest.mark.asyncio
    async def test_aggregate_results_deduplicates_findings(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that aggregation deduplicates findings by ID."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.IN_PROGRESS,
            reviewer_types=["security", "backend"],
        )
        orchestrator._cycles[cycle.cycle_id] = cycle

        # Create duplicate finding with same ID
        finding_id = str(uuid.uuid4())
        finding1 = ReviewFinding(
            id=finding_id,
            reviewer_type="security",
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Description",
        )
        finding2 = ReviewFinding(
            id=finding_id,
            reviewer_type="backend",
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Description",
        )

        result1 = ReviewResult(
            reviewer_type="security",
            task_id="task-1",
            findings=[finding1],
            summary="Summary",
            passed=False,
        )
        result2 = ReviewResult(
            reviewer_type="backend",
            task_id="task-2",
            findings=[finding2],
            summary="Summary",
            passed=False,
        )

        await orchestrator.submit_result(cycle.cycle_id, result1)
        await orchestrator.submit_result(cycle.cycle_id, result2)

        aggregated = await orchestrator.aggregate_results(cycle.cycle_id)

        # Should only have one finding after deduplication
        assert len(aggregated.all_findings) == 1

    @pytest.mark.asyncio
    async def test_aggregate_results_resets_counter(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that aggregation resets task counter."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.IN_PROGRESS,
            reviewer_types=["security"],
        )
        orchestrator._cycles[cycle.cycle_id] = cycle

        # Set task counter
        orchestrator._tasks_since_last_review["proj-123"] = 5

        # Submit result and aggregate
        result = ReviewResult(
            reviewer_type="security",
            task_id="task-1",
            summary="Summary",
            passed=True,
        )
        await orchestrator.submit_result(cycle.cycle_id, result)
        await orchestrator.aggregate_results(cycle.cycle_id)

        # Counter should be reset
        assert orchestrator._tasks_since_last_review["proj-123"] == 0


class TestSummaries:
    """Test findings summary generation."""

    def test_get_findings_summary_empty(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test summary for cycle with no findings."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.COMPLETED,
        )

        summary = orchestrator.get_findings_summary(cycle)

        assert summary["cycle_id"] == cycle.cycle_id
        assert summary["state"] == ReviewCycleState.COMPLETED.value
        assert summary["total_findings"] == 0
        assert summary["severity_counts"] == {}
        assert summary["category_counts"] == {}
        assert summary["reviewer_status"] == {}
        assert summary["overall_passed"] is False
        assert summary["critical_findings"] == []
        assert summary["high_findings"] == []

    def test_get_findings_summary_with_findings(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test summary with multiple findings."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.COMPLETED,
        )

        # Add findings
        cycle.all_findings = [
            ReviewFinding(
                reviewer_type="security",
                severity=FindingSeverity.CRITICAL,
                title="Critical issue",
                description="Description",
                category="security",
            ),
            ReviewFinding(
                reviewer_type="security",
                severity=FindingSeverity.HIGH,
                title="High issue",
                description="Description",
                category="security",
            ),
            ReviewFinding(
                reviewer_type="backend",
                severity=FindingSeverity.MEDIUM,
                title="Medium issue",
                description="Description",
                category="error_handling",
            ),
            ReviewFinding(
                reviewer_type="backend",
                severity=FindingSeverity.LOW,
                title="Low issue",
                description="Description",
                category="performance",
            ),
        ]

        # Add results
        cycle.results = [
            ReviewResult(
                reviewer_type="security",
                task_id="task-1",
                summary="Security issues found",
                passed=False,
            ),
            ReviewResult(
                reviewer_type="backend",
                task_id="task-2",
                summary="Backend issues found",
                passed=False,
            ),
        ]

        summary = orchestrator.get_findings_summary(cycle)

        assert summary["total_findings"] == 4
        assert summary["severity_counts"]["critical"] == 1
        assert summary["severity_counts"]["high"] == 1
        assert summary["severity_counts"]["medium"] == 1
        assert summary["severity_counts"]["low"] == 1
        assert summary["category_counts"]["security"] == 2
        assert summary["category_counts"]["error_handling"] == 1
        assert summary["category_counts"]["performance"] == 1
        assert summary["reviewer_status"]["security"] is False
        assert summary["reviewer_status"]["backend"] is False
        assert summary["overall_passed"] is False
        assert len(summary["critical_findings"]) == 1
        assert len(summary["high_findings"]) == 1

    def test_get_findings_summary_all_passed(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test summary when all reviewers passed."""
        cycle = ReviewCycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            state=ReviewCycleState.COMPLETED,
        )

        # Add results (all passed)
        cycle.results = [
            ReviewResult(
                reviewer_type="security",
                task_id="task-1",
                summary="No issues",
                passed=True,
            ),
            ReviewResult(
                reviewer_type="backend",
                task_id="task-2",
                summary="No issues",
                passed=True,
            ),
        ]

        summary = orchestrator.get_findings_summary(cycle)

        assert summary["overall_passed"] is True
        assert summary["reviewer_status"]["security"] is True
        assert summary["reviewer_status"]["backend"] is True


class TestStatusQueries:
    """Test status query methods."""

    @pytest.mark.asyncio
    async def test_get_cycle_found(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test getting a cycle by ID."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )

        retrieved = await orchestrator.get_cycle(cycle.cycle_id)

        assert retrieved is not None
        assert retrieved.cycle_id == cycle.cycle_id

    @pytest.mark.asyncio
    async def test_get_cycle_not_found(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test getting a non-existent cycle returns None."""
        retrieved = await orchestrator.get_cycle("nonexistent-id")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_active_cycles_empty(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test getting active cycles when none exist."""
        active = await orchestrator.get_active_cycles("proj-123")

        assert active == []

    @pytest.mark.asyncio
    async def test_get_active_cycles_filters_terminal(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that get_active_cycles filters out terminal states."""
        # Create cycles in various states
        cycle1 = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        # cycle1 is PENDING (active)

        cycle2 = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        cycle2.state = ReviewCycleState.COMPLETED
        orchestrator._cycles[cycle2.cycle_id] = cycle2

        cycle3 = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        cycle3.state = ReviewCycleState.FAILED
        orchestrator._cycles[cycle3.cycle_id] = cycle3

        cycle4 = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        cycle4.state = ReviewCycleState.IN_PROGRESS
        orchestrator._cycles[cycle4.cycle_id] = cycle4

        active = await orchestrator.get_active_cycles("proj-123")

        assert len(active) == 2
        active_ids = {c.cycle_id for c in active}
        assert cycle1.cycle_id in active_ids
        assert cycle4.cycle_id in active_ids

    @pytest.mark.asyncio
    async def test_get_active_cycles_filters_by_project(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that get_active_cycles filters by project ID."""
        cycle1 = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )

        cycle2 = await orchestrator.create_cycle(
            project_id="proj-456",
            trigger=ReviewTrigger.ON_DEMAND,
        )

        active = await orchestrator.get_active_cycles("proj-123")

        assert len(active) == 1
        assert active[0].cycle_id == cycle1.cycle_id

    @pytest.mark.asyncio
    async def test_get_active_cycles_sorted(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test that get_active_cycles returns cycles sorted by created_at."""
        # Create cycles with different timestamps
        cycle1 = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        cycle1.created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        cycle2 = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        cycle2.created_at = datetime.now(timezone.utc) - timedelta(hours=1)

        cycle3 = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        cycle3.created_at = datetime.now(timezone.utc)

        active = await orchestrator.get_active_cycles("proj-123")

        assert len(active) == 3
        assert active[0].cycle_id == cycle1.cycle_id
        assert active[1].cycle_id == cycle2.cycle_id
        assert active[2].cycle_id == cycle3.cycle_id


class TestFailCycle:
    """Test cycle failure handling."""

    @pytest.mark.asyncio
    async def test_fail_cycle_from_pending(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test failing a cycle from PENDING state."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )

        failed = await orchestrator.fail_cycle(cycle.cycle_id, "Test failure reason")

        assert failed.state == ReviewCycleState.FAILED
        assert failed.completed_at is not None

    @pytest.mark.asyncio
    async def test_fail_cycle_from_in_progress(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test failing a cycle from IN_PROGRESS state."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
            tasks_to_review=["task-1"],
            reviewer_types=["security"],
        )
        orchestrator._collect_files_for_tasks = AsyncMock(return_value=["file1.py"])
        await orchestrator.start_cycle(cycle.cycle_id)

        failed = await orchestrator.fail_cycle(cycle.cycle_id, "Agent timeout")

        assert failed.state == ReviewCycleState.FAILED

    @pytest.mark.asyncio
    async def test_fail_cycle_invalid_id(self, orchestrator: ReviewCycleOrchestrator) -> None:
        """Test failing a non-existent cycle raises ValueError."""
        with pytest.raises(ValueError, match="Review cycle .* not found"):
            await orchestrator.fail_cycle("nonexistent-id", "Reason")

    @pytest.mark.asyncio
    async def test_fail_cycle_already_completed(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test failing an already completed cycle raises error."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        cycle.state = ReviewCycleState.COMPLETED
        orchestrator._cycles[cycle.cycle_id] = cycle

        with pytest.raises(InvalidReviewTransitionError):
            await orchestrator.fail_cycle(cycle.cycle_id, "Reason")

    @pytest.mark.asyncio
    async def test_fail_cycle_already_failed(
        self, orchestrator: ReviewCycleOrchestrator
    ) -> None:
        """Test failing an already failed cycle raises error."""
        cycle = await orchestrator.create_cycle(
            project_id="proj-123",
            trigger=ReviewTrigger.ON_DEMAND,
        )
        await orchestrator.fail_cycle(cycle.cycle_id, "First failure")

        with pytest.raises(InvalidReviewTransitionError):
            await orchestrator.fail_cycle(cycle.cycle_id, "Second failure")
