"""Review cycle orchestration for Forgemaster.

Implements the review cycle state machine, trigger logic, review task
generation for specialist reviewers, and result aggregation. Review cycles
are periodic cross-cutting reviews that span multiple completed tasks,
checking for security issues, architectural consistency, spec compliance,
and general code quality.

The orchestrator manages the full lifecycle:
    PENDING -> COLLECTING -> DISPATCHING -> IN_PROGRESS -> AGGREGATING -> COMPLETED

Any state can transition to FAILED on unrecoverable errors.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.task import Task, TaskStatus

logger = structlog.get_logger(__name__)

# Type alias matching the project's convention (see dispatcher.py)
SessionFactory = Callable[[], AsyncSession]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReviewCycleState(str, Enum):
    """Lifecycle states for a review cycle.

    States:
        PENDING: Cycle created but not yet started.
        COLLECTING: Gathering completed tasks and files to review.
        DISPATCHING: Generating and spawning reviewer tasks.
        IN_PROGRESS: Reviewer agents are actively running.
        AGGREGATING: Collecting and merging reviewer results.
        COMPLETED: All results aggregated, cycle finished.
        FAILED: Cycle encountered an unrecoverable error.
    """

    PENDING = "pending"
    COLLECTING = "collecting"
    DISPATCHING = "dispatching"
    IN_PROGRESS = "in_progress"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewTrigger(str, Enum):
    """Conditions that trigger a new review cycle.

    Triggers:
        PHASE_COMPLETE: A project phase has finished.
        TASK_COUNT: A threshold number of tasks have completed since the last review.
        ON_DEMAND: Manually requested by the user or orchestrator.
        SCHEDULED: Periodic time-based trigger.
    """

    PHASE_COMPLETE = "phase_complete"
    TASK_COUNT = "task_count"
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"


class FindingSeverity(str, Enum):
    """Severity levels for review findings.

    Levels:
        CRITICAL: Must be fixed before proceeding.
        HIGH: Should be fixed soon.
        MEDIUM: Should be addressed in a future task.
        LOW: Minor improvement opportunity.
        INFO: Informational note, no action required.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ---------------------------------------------------------------------------
# Valid state transitions
# ---------------------------------------------------------------------------

VALID_REVIEW_TRANSITIONS: dict[ReviewCycleState, set[ReviewCycleState]] = {
    ReviewCycleState.PENDING: {ReviewCycleState.COLLECTING, ReviewCycleState.FAILED},
    ReviewCycleState.COLLECTING: {ReviewCycleState.DISPATCHING, ReviewCycleState.FAILED},
    ReviewCycleState.DISPATCHING: {ReviewCycleState.IN_PROGRESS, ReviewCycleState.FAILED},
    ReviewCycleState.IN_PROGRESS: {ReviewCycleState.AGGREGATING, ReviewCycleState.FAILED},
    ReviewCycleState.AGGREGATING: {ReviewCycleState.COMPLETED, ReviewCycleState.FAILED},
    ReviewCycleState.COMPLETED: set(),
    ReviewCycleState.FAILED: set(),
}


class InvalidReviewTransitionError(Exception):
    """Raised when an invalid review cycle state transition is attempted.

    Attributes:
        current: The current review cycle state.
        target: The attempted target state.
        cycle_id: The ID of the cycle that failed to transition.
    """

    def __init__(
        self,
        current: ReviewCycleState,
        target: ReviewCycleState,
        cycle_id: str | None = None,
    ) -> None:
        self.current = current
        self.target = target
        self.cycle_id = cycle_id
        msg = f"Invalid review cycle transition from {current.value} to {target.value}"
        if cycle_id:
            msg += f" for cycle {cycle_id}"
        super().__init__(msg)


def validate_review_transition(
    current: ReviewCycleState,
    target: ReviewCycleState,
) -> bool:
    """Validate if a review cycle state transition is allowed.

    Args:
        current: Current review cycle state.
        target: Target review cycle state.

    Returns:
        True if the transition is valid according to VALID_REVIEW_TRANSITIONS.
    """
    return target in VALID_REVIEW_TRANSITIONS.get(current, set())


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ReviewFinding(BaseModel):
    """A single finding from a reviewer.

    Attributes:
        id: UUID identifier for this finding.
        reviewer_type: The type of reviewer that produced this finding
            (e.g., "security", "backend", "spec").
        severity: Severity level of the finding.
        title: Short summary of the finding.
        description: Detailed explanation of the issue.
        file_path: File path where the issue was found, if applicable.
        line_number: Line number in the file, if applicable.
        suggested_fix: Suggested remediation, if any.
        category: Finding category (e.g., "security", "performance").
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_type: str
    severity: FindingSeverity
    title: str
    description: str
    file_path: str | None = None
    line_number: int | None = None
    suggested_fix: str | None = None
    category: str | None = None


class ReviewResult(BaseModel):
    """Aggregated result from one reviewer.

    Attributes:
        reviewer_type: The type of reviewer (e.g., "security", "backend").
        task_id: The review task ID that produced this result.
        findings: List of individual findings from the reviewer.
        summary: Human-readable summary of the review.
        passed: Whether the review passed overall.
        confidence_score: Reviewer's confidence in the result (0.0 to 1.0).
    """

    reviewer_type: str
    task_id: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str
    passed: bool
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ReviewCycle(BaseModel):
    """A complete review cycle spanning multiple tasks.

    Tracks the full lifecycle from creation through result aggregation,
    including which tasks are under review, which reviewer types are
    engaged, and the collected findings.

    Attributes:
        cycle_id: UUID identifier for this review cycle.
        project_id: UUID of the project being reviewed.
        state: Current lifecycle state of the cycle.
        trigger: What triggered this review cycle.
        tasks_under_review: List of task IDs being reviewed.
        reviewer_types: List of reviewer types to use.
        review_task_ids: List of spawned review task IDs.
        results: List of reviewer results received so far.
        all_findings: Flattened list of all findings across reviewers.
        created_at: When the cycle was created.
        started_at: When the cycle started executing.
        completed_at: When the cycle finished.
    """

    cycle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    state: ReviewCycleState = ReviewCycleState.PENDING
    trigger: ReviewTrigger
    tasks_under_review: list[str] = Field(default_factory=list)
    reviewer_types: list[str] = Field(default_factory=list)
    review_task_ids: list[str] = Field(default_factory=list)
    results: list[ReviewResult] = Field(default_factory=list)
    all_findings: list[ReviewFinding] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ReviewTriggerConfig(BaseModel):
    """Configuration for review triggers.

    Attributes:
        task_count_threshold: Number of completed tasks before triggering a review.
        enable_phase_trigger: Whether to trigger reviews on phase completion.
        enable_scheduled_trigger: Whether to enable periodic scheduled reviews.
        schedule_interval_hours: Hours between scheduled reviews.
        default_reviewer_types: Default set of reviewer types to use.
    """

    task_count_threshold: int = Field(
        default=10,
        description="Trigger after N tasks complete",
    )
    enable_phase_trigger: bool = Field(default=True)
    enable_scheduled_trigger: bool = Field(default=False)
    schedule_interval_hours: int = Field(default=24)
    default_reviewer_types: list[str] = Field(
        default=["backend", "security", "spec", "errors"],
    )


# ---------------------------------------------------------------------------
# Reviewer task templates
# ---------------------------------------------------------------------------

_REVIEWER_DESCRIPTIONS: dict[str, str] = {
    "backend": (
        "Review the following files for backend code quality issues including: "
        "error handling, input validation, resource management, API consistency, "
        "logging, and adherence to project coding standards."
    ),
    "security": (
        "Perform a security review of the following files. Look for: "
        "injection vulnerabilities, authentication/authorization issues, "
        "secrets in code, insecure defaults, SSRF risks, and "
        "dependency-related security concerns."
    ),
    "spec": (
        "Verify the following files conform to the project specification. "
        "Check: interface contracts, required fields, naming conventions, "
        "expected behavior, and completeness against the spec requirements."
    ),
    "errors": (
        "Review the following files for error-handling correctness. Check: "
        "bare except clauses, swallowed exceptions, missing error propagation, "
        "inconsistent error types, and missing validation."
    ),
    "frontend": (
        "Review the following files for frontend quality: component structure, "
        "accessibility, responsive design, state management patterns, and "
        "performance considerations."
    ),
    "performance": (
        "Review the following files for performance issues: N+1 queries, "
        "unnecessary allocations, blocking I/O in async paths, missing caching "
        "opportunities, and algorithmic inefficiencies."
    ),
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ReviewCycleOrchestrator:
    """Orchestrates review cycles across project tasks.

    Manages the lifecycle of review cycles: triggering, generating review
    tasks for specialist reviewers, dispatching them, and aggregating results.

    The orchestrator maintains an in-memory registry of active cycles and
    interacts with the database to query completed tasks and their files.

    Attributes:
        session_factory: Callable that produces database sessions.
        trigger_config: Configuration for review trigger conditions.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        trigger_config: ReviewTriggerConfig | None = None,
    ) -> None:
        """Initialize the review cycle orchestrator.

        Args:
            session_factory: Callable returning AsyncSession instances or
                an async context manager yielding one.
            trigger_config: Optional trigger configuration. Uses defaults
                if not provided.
        """
        self.session_factory = session_factory
        self.trigger_config = trigger_config or ReviewTriggerConfig()

        self._cycles: dict[str, ReviewCycle] = {}
        self._last_review_time: dict[str, datetime] = {}
        self._tasks_since_last_review: dict[str, int] = {}
        self._logger = logger.bind(component="ReviewCycleOrchestrator")

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------

    def _transition(
        self,
        cycle: ReviewCycle,
        target: ReviewCycleState,
    ) -> ReviewCycle:
        """Transition a review cycle to a new state with validation.

        Args:
            cycle: The review cycle to transition.
            target: The target state.

        Returns:
            The updated ReviewCycle.

        Raises:
            InvalidReviewTransitionError: If the transition is not valid.
        """
        if not validate_review_transition(cycle.state, target):
            raise InvalidReviewTransitionError(cycle.state, target, cycle.cycle_id)

        old_state = cycle.state
        cycle.state = target

        # Set timestamps on key transitions
        if target == ReviewCycleState.COLLECTING:
            cycle.started_at = datetime.now(timezone.utc)
        elif target == ReviewCycleState.COMPLETED:
            cycle.completed_at = datetime.now(timezone.utc)

        self._logger.info(
            "review_cycle_transition",
            cycle_id=cycle.cycle_id,
            from_state=old_state.value,
            to_state=target.value,
        )

        return cycle

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def create_cycle(
        self,
        project_id: str,
        trigger: ReviewTrigger,
        tasks_to_review: list[str] | None = None,
        reviewer_types: list[str] | None = None,
    ) -> ReviewCycle:
        """Create a new review cycle.

        If ``tasks_to_review`` is not provided, the orchestrator will
        collect completed tasks during the COLLECTING phase.

        Args:
            project_id: UUID string of the project.
            trigger: What triggered this review cycle.
            tasks_to_review: Explicit list of task IDs to review.
                If None, tasks are collected automatically.
            reviewer_types: List of reviewer types. If None, uses
                the configured defaults.

        Returns:
            The newly created ReviewCycle in PENDING state.
        """
        cycle = ReviewCycle(
            project_id=project_id,
            trigger=trigger,
            tasks_under_review=tasks_to_review or [],
            reviewer_types=reviewer_types or list(self.trigger_config.default_reviewer_types),
        )

        self._cycles[cycle.cycle_id] = cycle

        self._logger.info(
            "review_cycle_created",
            cycle_id=cycle.cycle_id,
            project_id=project_id,
            trigger=trigger.value,
            reviewer_types=cycle.reviewer_types,
            task_count=len(cycle.tasks_under_review),
        )

        return cycle

    async def start_cycle(self, cycle_id: str) -> ReviewCycle:
        """Start a review cycle, advancing it through COLLECTING and DISPATCHING.

        This method drives the cycle from PENDING through:
        1. COLLECTING - gathers tasks and files to review
        2. DISPATCHING - generates review tasks for each reviewer type

        After dispatching, the cycle moves to IN_PROGRESS awaiting results.

        Args:
            cycle_id: UUID string of the cycle to start.

        Returns:
            The updated ReviewCycle in IN_PROGRESS state.

        Raises:
            ValueError: If the cycle ID is not found.
            InvalidReviewTransitionError: If the cycle is not in PENDING state.
        """
        cycle = self._cycles.get(cycle_id)
        if cycle is None:
            raise ValueError(f"Review cycle {cycle_id} not found")

        # Phase 1: COLLECTING
        self._transition(cycle, ReviewCycleState.COLLECTING)

        if not cycle.tasks_under_review:
            # Auto-collect completed tasks since last review
            collected_ids = await self._collect_completed_tasks(cycle.project_id)
            cycle.tasks_under_review = collected_ids

        self._logger.info(
            "review_cycle_tasks_collected",
            cycle_id=cycle_id,
            task_count=len(cycle.tasks_under_review),
        )

        # Phase 2: DISPATCHING
        self._transition(cycle, ReviewCycleState.DISPATCHING)
        review_task_ids = await self.generate_review_tasks(cycle)
        cycle.review_task_ids = review_task_ids

        # Phase 3: IN_PROGRESS
        self._transition(cycle, ReviewCycleState.IN_PROGRESS)

        return cycle

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    async def check_triggers(self, project_id: str) -> ReviewTrigger | None:
        """Check if any trigger conditions are met for a new review cycle.

        Evaluates triggers in priority order:
        1. Phase completion (if enabled)
        2. Task count threshold
        3. Scheduled interval (if enabled)

        Args:
            project_id: UUID string of the project to check.

        Returns:
            The ReviewTrigger that fired, or None if no triggers are met.
        """
        # Check task count trigger
        count = self._tasks_since_last_review.get(project_id, 0)
        if count >= self.trigger_config.task_count_threshold:
            self._logger.info(
                "review_trigger_task_count",
                project_id=project_id,
                count=count,
                threshold=self.trigger_config.task_count_threshold,
            )
            return ReviewTrigger.TASK_COUNT

        # Check scheduled trigger
        if self.trigger_config.enable_scheduled_trigger:
            last_time = self._last_review_time.get(project_id)
            if last_time is not None:
                elapsed_hours = (
                    datetime.now(timezone.utc) - last_time
                ).total_seconds() / 3600
                if elapsed_hours >= self.trigger_config.schedule_interval_hours:
                    self._logger.info(
                        "review_trigger_scheduled",
                        project_id=project_id,
                        elapsed_hours=elapsed_hours,
                        interval_hours=self.trigger_config.schedule_interval_hours,
                    )
                    return ReviewTrigger.SCHEDULED

        return None

    async def should_trigger_review(self, project_id: str) -> bool:
        """Check if a review should be triggered for the given project.

        Convenience wrapper around ``check_triggers`` that returns a boolean.

        Args:
            project_id: UUID string of the project.

        Returns:
            True if any trigger condition is met.
        """
        trigger = await self.check_triggers(project_id)
        return trigger is not None

    def record_task_completion(self, project_id: str) -> int:
        """Record that a task has completed for trigger tracking.

        Should be called by the orchestrator whenever a task reaches DONE.

        Args:
            project_id: UUID string of the project.

        Returns:
            The new count of tasks since last review.
        """
        current = self._tasks_since_last_review.get(project_id, 0)
        new_count = current + 1
        self._tasks_since_last_review[project_id] = new_count
        return new_count

    def reset_task_counter(self, project_id: str) -> None:
        """Reset the task completion counter after a review cycle starts.

        Args:
            project_id: UUID string of the project.
        """
        self._tasks_since_last_review[project_id] = 0
        self._last_review_time[project_id] = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Task collection
    # ------------------------------------------------------------------

    async def _collect_completed_tasks(self, project_id: str) -> list[str]:
        """Collect IDs of completed tasks since the last review.

        Queries the database for tasks with status DONE that were completed
        after the last review time (or all DONE tasks if no prior review).

        Args:
            project_id: UUID string of the project.

        Returns:
            List of task ID strings.
        """
        from uuid import UUID

        last_time = self._last_review_time.get(project_id)

        async with self.session_factory() as session:
            stmt = (
                select(Task.id)
                .where(Task.project_id == UUID(project_id))
                .where(Task.status == TaskStatus.done)
            )
            if last_time is not None:
                stmt = stmt.where(Task.completed_at >= last_time)

            stmt = stmt.order_by(Task.completed_at.asc())

            result = await session.execute(stmt)
            task_ids = [str(row[0]) for row in result.all()]

        self._logger.info(
            "completed_tasks_collected",
            project_id=project_id,
            count=len(task_ids),
            since=last_time.isoformat() if last_time else "beginning",
        )

        return task_ids

    async def _collect_files_for_tasks(self, task_ids: list[str]) -> list[str]:
        """Aggregate all files_touched from a set of tasks.

        Deduplicates and sorts the resulting file list.

        Args:
            task_ids: List of task ID strings.

        Returns:
            Sorted, deduplicated list of file paths.
        """
        from uuid import UUID

        if not task_ids:
            return []

        uuids = [UUID(tid) for tid in task_ids]

        async with self.session_factory() as session:
            stmt = (
                select(Task.files_touched)
                .where(Task.id.in_(uuids))
                .where(Task.files_touched.isnot(None))
            )
            result = await session.execute(stmt)
            rows = result.all()

        all_files: set[str] = set()
        for row in rows:
            files = row[0]
            if files:
                all_files.update(files)

        return sorted(all_files)

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    async def generate_review_tasks(self, cycle: ReviewCycle) -> list[str]:
        """Generate review tasks for each reviewer type in the cycle.

        For each reviewer type, builds a task definition targeting the
        files modified by the tasks under review.

        Args:
            cycle: The review cycle to generate tasks for.

        Returns:
            List of generated review task ID strings.
        """
        files_to_review = await self._collect_files_for_tasks(cycle.tasks_under_review)

        if not files_to_review:
            self._logger.warning(
                "no_files_to_review",
                cycle_id=cycle.cycle_id,
                tasks_under_review=cycle.tasks_under_review,
            )

        generated_ids: list[str] = []

        for reviewer_type in cycle.reviewer_types:
            task_def = self._build_review_task(cycle, reviewer_type, files_to_review)
            generated_ids.append(task_def["task_id"])

            self._logger.info(
                "review_task_generated",
                cycle_id=cycle.cycle_id,
                reviewer_type=reviewer_type,
                task_id=task_def["task_id"],
                file_count=len(files_to_review),
            )

        return generated_ids

    def _build_review_task(
        self,
        cycle: ReviewCycle,
        reviewer_type: str,
        files_to_review: list[str],
    ) -> dict[str, Any]:
        """Build a task definition dict for a reviewer.

        Creates a structured task definition that can be used to spawn
        a review task in the task queue. The description includes the
        reviewer-specific instructions and the list of files to review.

        Args:
            cycle: The parent review cycle.
            reviewer_type: Type of reviewer (e.g., "security", "backend").
            files_to_review: List of file paths to include in the review.

        Returns:
            Dictionary with task definition fields including task_id,
            title, description, agent_type, model_tier, priority,
            files_touched, and metadata.
        """
        task_id = str(uuid.uuid4())

        description_template = _REVIEWER_DESCRIPTIONS.get(
            reviewer_type,
            f"Review the following files as a {reviewer_type} specialist. "
            f"Look for issues, improvements, and potential bugs.",
        )

        files_section = "\n".join(f"- {f}" for f in files_to_review) if files_to_review else "- (no files identified)"

        description = (
            f"{description_template}\n\n"
            f"## Files to Review\n{files_section}\n\n"
            f"## Tasks Under Review\n"
            f"This review covers {len(cycle.tasks_under_review)} completed task(s).\n\n"
            f"## Output Format\n"
            f"Return your result as a JSON object with fields:\n"
            f"- findings: list of {{severity, title, description, file_path, line_number, suggested_fix, category}}\n"
            f"- summary: overall summary string\n"
            f"- passed: boolean indicating overall pass/fail\n"
            f"- confidence_score: float 0.0-1.0"
        )

        return {
            "task_id": task_id,
            "title": f"Review: {reviewer_type} review for cycle {cycle.cycle_id[:8]}",
            "description": description,
            "agent_type": f"reviewer_{reviewer_type}",
            "model_tier": "sonnet",
            "priority": 50,  # Higher priority than normal tasks
            "files_touched": files_to_review,
            "metadata": {
                "review_cycle_id": cycle.cycle_id,
                "reviewer_type": reviewer_type,
                "project_id": cycle.project_id,
            },
        }

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    async def submit_result(
        self,
        cycle_id: str,
        result: ReviewResult,
    ) -> None:
        """Submit a review result from a reviewer.

        Adds the result to the cycle and flattens its findings into
        the cycle's ``all_findings`` list.

        Args:
            cycle_id: UUID string of the review cycle.
            result: The reviewer's result.

        Raises:
            ValueError: If the cycle ID is not found.
        """
        cycle = self._cycles.get(cycle_id)
        if cycle is None:
            raise ValueError(f"Review cycle {cycle_id} not found")

        cycle.results.append(result)
        cycle.all_findings.extend(result.findings)

        self._logger.info(
            "review_result_submitted",
            cycle_id=cycle_id,
            reviewer_type=result.reviewer_type,
            finding_count=len(result.findings),
            passed=result.passed,
        )

    async def aggregate_results(self, cycle_id: str) -> ReviewCycle:
        """Aggregate all reviewer results into the cycle.

        Checks if all expected reviewers have reported. If so, transitions
        the cycle through AGGREGATING to COMPLETED and resets the task
        counter for the project.

        Args:
            cycle_id: UUID string of the review cycle.

        Returns:
            The updated ReviewCycle (COMPLETED if all results in,
            otherwise still IN_PROGRESS).

        Raises:
            ValueError: If the cycle ID is not found.
        """
        cycle = self._cycles.get(cycle_id)
        if cycle is None:
            raise ValueError(f"Review cycle {cycle_id} not found")

        reported_types = {r.reviewer_type for r in cycle.results}
        expected_types = set(cycle.reviewer_types)

        if reported_types >= expected_types:
            # All reviewers have reported
            self._transition(cycle, ReviewCycleState.AGGREGATING)

            # Deduplicate findings by ID
            seen_ids: set[str] = set()
            unique_findings: list[ReviewFinding] = []
            for finding in cycle.all_findings:
                if finding.id not in seen_ids:
                    seen_ids.add(finding.id)
                    unique_findings.append(finding)
            cycle.all_findings = unique_findings

            self._transition(cycle, ReviewCycleState.COMPLETED)
            self.reset_task_counter(cycle.project_id)

            self._logger.info(
                "review_cycle_completed",
                cycle_id=cycle_id,
                total_findings=len(cycle.all_findings),
                all_passed=all(r.passed for r in cycle.results),
            )
        else:
            missing = expected_types - reported_types
            self._logger.info(
                "review_cycle_awaiting_results",
                cycle_id=cycle_id,
                reported=list(reported_types),
                missing=list(missing),
            )

        return cycle

    def get_findings_summary(self, cycle: ReviewCycle) -> dict[str, Any]:
        """Get a summary of findings from a review cycle.

        Provides counts by severity, counts by category, per-reviewer
        pass/fail status, and the top critical/high findings.

        Args:
            cycle: The review cycle to summarize.

        Returns:
            Dictionary containing:
                - cycle_id: The cycle identifier.
                - state: Current cycle state.
                - total_findings: Total number of findings.
                - severity_counts: Dict mapping severity to count.
                - category_counts: Dict mapping category to count.
                - reviewer_status: Dict mapping reviewer type to passed bool.
                - overall_passed: True if all reviewers passed.
                - critical_findings: List of CRITICAL severity findings.
                - high_findings: List of HIGH severity findings.
        """
        severity_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}

        for finding in cycle.all_findings:
            sev = finding.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if finding.category:
                category_counts[finding.category] = (
                    category_counts.get(finding.category, 0) + 1
                )

        reviewer_status = {r.reviewer_type: r.passed for r in cycle.results}
        overall_passed = all(r.passed for r in cycle.results) if cycle.results else False

        critical_findings = [
            f.model_dump()
            for f in cycle.all_findings
            if f.severity == FindingSeverity.CRITICAL
        ]
        high_findings = [
            f.model_dump()
            for f in cycle.all_findings
            if f.severity == FindingSeverity.HIGH
        ]

        return {
            "cycle_id": cycle.cycle_id,
            "state": cycle.state.value,
            "total_findings": len(cycle.all_findings),
            "severity_counts": severity_counts,
            "category_counts": category_counts,
            "reviewer_status": reviewer_status,
            "overall_passed": overall_passed,
            "critical_findings": critical_findings,
            "high_findings": high_findings,
        }

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    async def get_cycle(self, cycle_id: str) -> ReviewCycle | None:
        """Retrieve a review cycle by ID.

        Args:
            cycle_id: UUID string of the cycle.

        Returns:
            The ReviewCycle if found, None otherwise.
        """
        return self._cycles.get(cycle_id)

    async def get_active_cycles(self, project_id: str) -> list[ReviewCycle]:
        """Get all active (non-terminal) review cycles for a project.

        Active cycles are those not in COMPLETED or FAILED state.

        Args:
            project_id: UUID string of the project.

        Returns:
            List of active ReviewCycle instances, sorted by creation time.
        """
        terminal_states = {ReviewCycleState.COMPLETED, ReviewCycleState.FAILED}
        active = [
            c
            for c in self._cycles.values()
            if c.project_id == project_id and c.state not in terminal_states
        ]
        active.sort(key=lambda c: c.created_at)
        return active

    async def fail_cycle(self, cycle_id: str, reason: str) -> ReviewCycle:
        """Transition a review cycle to FAILED state.

        Args:
            cycle_id: UUID string of the cycle.
            reason: Human-readable reason for the failure.

        Returns:
            The updated ReviewCycle in FAILED state.

        Raises:
            ValueError: If the cycle ID is not found.
            InvalidReviewTransitionError: If the cycle is in a terminal state.
        """
        cycle = self._cycles.get(cycle_id)
        if cycle is None:
            raise ValueError(f"Review cycle {cycle_id} not found")

        previous_state = cycle.state
        self._transition(cycle, ReviewCycleState.FAILED)
        cycle.completed_at = datetime.now(timezone.utc)

        self._logger.error(
            "review_cycle_failed",
            cycle_id=cycle_id,
            reason=reason,
            state_at_failure=previous_state.value,
        )

        return cycle
