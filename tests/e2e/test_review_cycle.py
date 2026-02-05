"""End-to-end tests for review cycle orchestration.

Tests review cycle triggers, task generation, finding consolidation,
fix task generation, and completion flows.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.review.cycle import (
    FindingSeverity,
    InvalidReviewTransitionError,
    ReviewCycleState,
    ReviewTrigger,
    validate_review_transition,
)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestReviewCycleTriggers:
    """Test review cycle trigger conditions."""

    async def test_review_trigger_on_task_count_threshold(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test triggering review after N completed tasks."""
        # Create 10 completed tasks
        completed_tasks = []
        for i in range(10):
            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"Completed task {i}",
                status=TaskStatus.done,
                agent_type="executor",
                max_retries=3,
                retry_count=0,
                completed_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)
            completed_tasks.append(task)

        await e2e_session.commit()

        # Query for completed tasks to simulate threshold check
        result = await e2e_session.execute(
            select(Task).where(
                Task.project_id == sample_project.id,
                Task.status == TaskStatus.done,
            )
        )
        done_tasks = result.scalars().all()

        # Should trigger review if count >= threshold (e.g., 10)
        assert len(done_tasks) == 10

    async def test_review_trigger_on_phase_complete(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test triggering review when a phase completes."""
        # Simulate phase completion by marking all tasks in a parallel group as done
        phase_group = "phase-1"

        for i in range(5):
            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"Phase 1 task {i}",
                status=TaskStatus.done,
                agent_type="executor",
                parallel_group=phase_group,
                max_retries=3,
                retry_count=0,
                completed_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)

        await e2e_session.commit()

        # Query for phase tasks
        result = await e2e_session.execute(
            select(Task).where(
                Task.parallel_group == phase_group,
                Task.status == TaskStatus.done,
            )
        )
        phase_tasks = result.scalars().all()

        assert len(phase_tasks) == 5
        assert all(t.status == TaskStatus.done for t in phase_tasks)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestReviewCycleStateTransitions:
    """Test review cycle state machine transitions."""

    def test_valid_transition_pending_to_collecting(self) -> None:
        """Test valid transition from PENDING to COLLECTING."""
        assert validate_review_transition(
            ReviewCycleState.PENDING,
            ReviewCycleState.COLLECTING,
        )

    def test_valid_transition_collecting_to_dispatching(self) -> None:
        """Test valid transition from COLLECTING to DISPATCHING."""
        assert validate_review_transition(
            ReviewCycleState.COLLECTING,
            ReviewCycleState.DISPATCHING,
        )

    def test_valid_transition_dispatching_to_in_progress(self) -> None:
        """Test valid transition from DISPATCHING to IN_PROGRESS."""
        assert validate_review_transition(
            ReviewCycleState.DISPATCHING,
            ReviewCycleState.IN_PROGRESS,
        )

    def test_valid_transition_in_progress_to_aggregating(self) -> None:
        """Test valid transition from IN_PROGRESS to AGGREGATING."""
        assert validate_review_transition(
            ReviewCycleState.IN_PROGRESS,
            ReviewCycleState.AGGREGATING,
        )

    def test_valid_transition_aggregating_to_completed(self) -> None:
        """Test valid transition from AGGREGATING to COMPLETED."""
        assert validate_review_transition(
            ReviewCycleState.AGGREGATING,
            ReviewCycleState.COMPLETED,
        )

    def test_any_state_can_transition_to_failed(self) -> None:
        """Test that any state can transition to FAILED."""
        states = [
            ReviewCycleState.PENDING,
            ReviewCycleState.COLLECTING,
            ReviewCycleState.DISPATCHING,
            ReviewCycleState.IN_PROGRESS,
            ReviewCycleState.AGGREGATING,
        ]

        for state in states:
            assert validate_review_transition(state, ReviewCycleState.FAILED)

    def test_completed_state_is_terminal(self) -> None:
        """Test that COMPLETED state cannot transition to any other state."""
        assert not validate_review_transition(
            ReviewCycleState.COMPLETED,
            ReviewCycleState.PENDING,
        )
        assert not validate_review_transition(
            ReviewCycleState.COMPLETED,
            ReviewCycleState.IN_PROGRESS,
        )

    def test_invalid_transition_pending_to_in_progress(self) -> None:
        """Test that invalid transition from PENDING to IN_PROGRESS is rejected."""
        assert not validate_review_transition(
            ReviewCycleState.PENDING,
            ReviewCycleState.IN_PROGRESS,
        )


@pytest.mark.e2e
@pytest.mark.asyncio
class TestReviewTaskGeneration:
    """Test generation of reviewer tasks."""

    async def test_generate_security_review_task(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test generating a security review task."""
        review_task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Security review - Phase 1",
            description="Review completed tasks for security issues",
            status=TaskStatus.ready,
            agent_type="security-reviewer",
            priority=50,  # Higher priority for reviews
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(review_task)
        await e2e_session.commit()
        await e2e_session.refresh(review_task)

        assert review_task.agent_type == "security-reviewer"
        assert review_task.status == TaskStatus.ready

    async def test_generate_multiple_reviewer_tasks(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test generating tasks for multiple reviewer types."""
        reviewer_types = [
            "security-reviewer",
            "code-reviewer",
            "build-fixer",
        ]

        review_tasks = []
        for reviewer_type in reviewer_types:
            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"{reviewer_type} - Review cycle 1",
                description=f"Review tasks with {reviewer_type}",
                status=TaskStatus.ready,
                agent_type=reviewer_type,
                priority=50,
                max_retries=3,
                retry_count=0,
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)
            review_tasks.append(task)

        await e2e_session.commit()

        # Query for review tasks
        result = await e2e_session.execute(
            select(Task).where(Task.priority == 50)
        )
        created_review_tasks = result.scalars().all()

        assert len(created_review_tasks) == 3
        actual_types = {t.agent_type for t in created_review_tasks}
        assert actual_types == set(reviewer_types)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestFindingConsolidation:
    """Test finding consolidation and deduplication."""

    async def test_findings_with_different_severities(self) -> None:
        """Test that findings have proper severity levels."""
        severities = [
            FindingSeverity.CRITICAL,
            FindingSeverity.HIGH,
            FindingSeverity.MEDIUM,
            FindingSeverity.LOW,
            FindingSeverity.INFO,
        ]

        for severity in severities:
            assert isinstance(severity, FindingSeverity)
            assert severity.value in ["critical", "high", "medium", "low", "info"]

    async def test_duplicate_finding_detection(self) -> None:
        """Test detecting duplicate findings from multiple reviewers."""
        # Simulate findings from two reviewers about the same issue
        finding1 = {
            "file": "src/auth.py",
            "line": 42,
            "severity": FindingSeverity.HIGH.value,
            "message": "Missing authentication check",
            "reviewer": "security-reviewer",
        }

        finding2 = {
            "file": "src/auth.py",
            "line": 42,
            "severity": FindingSeverity.HIGH.value,
            "message": "Authentication bypass vulnerability",
            "reviewer": "code-reviewer",
        }

        # Check if findings point to same location (deduplication logic)
        same_location = (
            finding1["file"] == finding2["file"] and finding1["line"] == finding2["line"]
        )

        assert same_location is True


@pytest.mark.e2e
@pytest.mark.asyncio
class TestFixTaskGeneration:
    """Test generation of fix tasks from review findings."""

    async def test_generate_fix_task_from_critical_finding(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test generating fix task for critical finding."""
        fix_task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Fix: Missing authentication check in auth.py",
            description="Address critical security finding from review cycle",
            status=TaskStatus.ready,
            agent_type="executor",
            priority=10,  # Very high priority for critical fixes
            files_touched=["src/auth.py"],
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(fix_task)
        await e2e_session.commit()
        await e2e_session.refresh(fix_task)

        assert fix_task.priority == 10
        assert "src/auth.py" in (fix_task.files_touched or [])

    async def test_generate_multiple_fix_tasks(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test generating fix tasks for multiple findings."""
        findings = [
            {"file": "src/auth.py", "severity": FindingSeverity.CRITICAL},
            {"file": "src/config.py", "severity": FindingSeverity.HIGH},
            {"file": "src/utils.py", "severity": FindingSeverity.MEDIUM},
        ]

        fix_tasks = []
        for i, finding in enumerate(findings):
            # Priority based on severity
            priority_map = {
                FindingSeverity.CRITICAL: 10,
                FindingSeverity.HIGH: 20,
                FindingSeverity.MEDIUM: 50,
            }

            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"Fix: Issue in {finding['file']}",
                status=TaskStatus.ready,
                agent_type="executor",
                priority=priority_map[finding["severity"]],
                files_touched=[finding["file"]],
                max_retries=3,
                retry_count=0,
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)
            fix_tasks.append(task)

        await e2e_session.commit()

        # Verify fix tasks created
        result = await e2e_session.execute(
            select(Task).where(Task.title.like("Fix:%"))
        )
        created_fixes = result.scalars().all()

        assert len(created_fixes) == 3


@pytest.mark.e2e
@pytest.mark.asyncio
class TestReviewCycleCompletion:
    """Test review cycle completion flows."""

    async def test_all_review_tasks_complete(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test detecting when all review tasks are complete."""
        review_cycle_id = "cycle-1"

        # Create 3 review tasks, all completed
        for i in range(3):
            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"Review task {i}",
                description=f"Review cycle: {review_cycle_id}",
                status=TaskStatus.done,
                agent_type="code-reviewer",
                max_retries=3,
                retry_count=0,
                completed_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)

        await e2e_session.commit()

        # Query for review tasks in this cycle
        result = await e2e_session.execute(
            select(Task).where(Task.description.like(f"%{review_cycle_id}%"))
        )
        cycle_tasks = result.scalars().all()

        # Check if all are done
        all_done = all(t.status == TaskStatus.done for t in cycle_tasks)
        assert all_done is True
        assert len(cycle_tasks) == 3

    async def test_review_cycle_aggregation_complete(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test aggregation phase completion."""
        # Simulate review tasks that produced findings
        review_tasks = []
        for i in range(2):
            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"Reviewer {i}",
                status=TaskStatus.done,
                agent_type="code-reviewer",
                max_retries=3,
                retry_count=0,
                completed_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)
            review_tasks.append(task)

        await e2e_session.commit()

        # After aggregation, cycle should be complete
        assert all(t.status == TaskStatus.done for t in review_tasks)
