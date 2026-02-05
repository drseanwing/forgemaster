"""Finding consolidation for Forgemaster review cycles.

Implements deduplication, severity ranking, and fix task generation from
raw review findings. The consolidator takes findings from multiple
specialized reviewers (security, backend, spec, etc.) and produces a
deduplicated, prioritized list of issues with actionable fix tasks.

The consolidation pipeline:
    1. Deduplicate similar findings by file + title similarity or line range
    2. Rank by severity and compute priority scores
    3. Generate fix tasks for high-priority findings
    4. Group findings by file to create combined tasks

This module produces the final actionable output from a review cycle.
"""

from __future__ import annotations

import uuid
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from forgemaster.review.cycle import FindingSeverity, ReviewFinding

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ConsolidatedFinding(BaseModel):
    """A finding after deduplication and ranking.

    When multiple reviewers report the same issue, their findings are merged
    into a single ConsolidatedFinding with increased confidence and combined
    metadata from all reporters.

    Attributes:
        id: UUID for this consolidated finding.
        original_finding_ids: IDs of merged findings that contributed to this.
        reviewer_types: List of reviewer types that reported this issue.
        severity: Highest severity among merged findings.
        title: Representative title (longest from merged findings).
        description: Combined description from all merged findings.
        file_path: File path where issue was found, if applicable.
        line_number: Line number in the file, if applicable.
        suggested_fix: Best suggested fix from merged findings (longest).
        category: Issue category (e.g., "security", "performance").
        confidence: Confidence score (higher when multiple reviewers agree).
        priority_score: Computed priority for fix ordering.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_finding_ids: list[str] = Field(default_factory=list)
    reviewer_types: list[str] = Field(default_factory=list)
    severity: FindingSeverity
    title: str
    description: str
    file_path: str | None = None
    line_number: int | None = None
    suggested_fix: str | None = None
    category: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    priority_score: float = Field(default=0.0, ge=0.0)


class TaskComplexity(str, Enum):
    """Estimated complexity for a fix task.

    Complexity:
        SIMPLE: One-file change, straightforward fix.
        MEDIUM: Multi-file change or moderate complexity.
        COMPLEX: Significant refactoring or architectural change.
    """

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class FixTask(BaseModel):
    """A generated fix task from a consolidated finding.

    Attributes:
        task_id: UUID for the fix task.
        finding_id: ID of the consolidated finding this addresses.
        title: Task title describing the fix.
        description: Task description with detailed fix instructions.
        files_to_modify: List of file paths that need modification.
        priority: Task priority score (higher = more urgent).
        estimated_complexity: Simple/Medium/Complex complexity level.
        agent_type: Recommended agent type for performing the fix.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    finding_id: str
    title: str
    description: str
    files_to_modify: list[str] = Field(default_factory=list)
    priority: float = Field(default=0.0, ge=0.0)
    estimated_complexity: TaskComplexity = TaskComplexity.MEDIUM
    agent_type: str = "executor"


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

SEVERITY_SCORES: dict[FindingSeverity, float] = {
    FindingSeverity.CRITICAL: 100.0,
    FindingSeverity.HIGH: 75.0,
    FindingSeverity.MEDIUM: 50.0,
    FindingSeverity.LOW: 25.0,
    FindingSeverity.INFO: 10.0,
}


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------


class FindingConsolidator:
    """Consolidates, deduplicates, and ranks review findings.

    Takes raw findings from multiple reviewers and produces a
    deduplicated, severity-ranked list with generated fix tasks.

    The consolidator uses string similarity and location-based matching
    to identify duplicate findings across different reviewers, then
    merges them into high-confidence consolidated findings.

    Attributes:
        similarity_threshold: Minimum string similarity (0.0-1.0) for
            deduplication. Default 0.8.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
    ) -> None:
        """Initialize the finding consolidator.

        Args:
            similarity_threshold: Minimum Jaccard similarity for two
                findings to be considered duplicates. Should be between
                0.0 and 1.0.
        """
        self.similarity_threshold = similarity_threshold
        self._logger = logger.bind(component="FindingConsolidator")

    def deduplicate(
        self, findings: list[ReviewFinding]
    ) -> list[ConsolidatedFinding]:
        """Deduplicate findings by similarity.

        Two findings are considered duplicates when:
        - Same file_path AND similar title (string similarity > threshold)
        - OR same category AND same file_path AND overlapping line range

        When merging duplicates:
        - Use highest severity among merged findings
        - Combine reviewer_types
        - Keep longest description
        - Keep most detailed suggested_fix (longest non-None)
        - Increase confidence by 0.1 per additional reviewer

        Args:
            findings: List of raw findings from reviewers.

        Returns:
            List of consolidated findings with duplicates merged.
        """
        if not findings:
            return []

        consolidated: list[ConsolidatedFinding] = []
        used_ids: set[str] = set()

        for finding in findings:
            if finding.id in used_ids:
                continue

            # Find all duplicates of this finding
            duplicates = [finding]
            used_ids.add(finding.id)

            for other in findings:
                if other.id in used_ids:
                    continue

                if self._is_duplicate(finding, other):
                    duplicates.append(other)
                    used_ids.add(other.id)

            # Merge duplicates into one consolidated finding
            merged = self._merge_findings(duplicates)
            consolidated.append(merged)

        self._logger.info(
            "findings_deduplicated",
            original_count=len(findings),
            consolidated_count=len(consolidated),
            deduplication_rate=1.0 - (len(consolidated) / len(findings))
            if findings
            else 0.0,
        )

        return consolidated

    def _is_duplicate(self, a: ReviewFinding, b: ReviewFinding) -> bool:
        """Check if two findings are duplicates.

        Args:
            a: First finding.
            b: Second finding.

        Returns:
            True if findings are considered duplicates.
        """
        # Case 1: Same file and similar title
        if a.file_path and b.file_path and a.file_path == b.file_path:
            similarity = self._compute_similarity(a.title, b.title)
            if similarity >= self.similarity_threshold:
                return True

        # Case 2: Same category, same file, overlapping line range
        if (
            a.category
            and b.category
            and a.category == b.category
            and a.file_path
            and b.file_path
            and a.file_path == b.file_path
        ):
            if a.line_number is not None and b.line_number is not None:
                # Consider overlapping if within 5 lines
                if abs(a.line_number - b.line_number) <= 5:
                    return True

        return False

    def _merge_findings(self, findings: list[ReviewFinding]) -> ConsolidatedFinding:
        """Merge multiple findings into one consolidated finding.

        Args:
            findings: List of duplicate findings to merge.

        Returns:
            A single ConsolidatedFinding with merged data.
        """
        # Use highest severity
        highest_severity = max(findings, key=lambda f: SEVERITY_SCORES[f.severity])

        # Collect all reviewer types
        reviewer_types = [f.reviewer_type for f in findings]

        # Keep longest description
        longest_desc = max(findings, key=lambda f: len(f.description))

        # Keep longest suggested_fix (among non-None values)
        fixes_with_content = [f for f in findings if f.suggested_fix]
        suggested_fix = None
        if fixes_with_content:
            longest_fix = max(fixes_with_content, key=lambda f: len(f.suggested_fix or ""))
            suggested_fix = longest_fix.suggested_fix

        # Keep longest title
        longest_title = max(findings, key=lambda f: len(f.title))

        # Confidence increases with multiple reviewers
        base_confidence = 0.7
        bonus_per_reviewer = 0.1
        confidence = min(1.0, base_confidence + (len(findings) - 1) * bonus_per_reviewer)

        # Use first finding's location data as representative
        first = findings[0]

        return ConsolidatedFinding(
            original_finding_ids=[f.id for f in findings],
            reviewer_types=reviewer_types,
            severity=highest_severity.severity,
            title=longest_title.title,
            description=longest_desc.description,
            file_path=first.file_path,
            line_number=first.line_number,
            suggested_fix=suggested_fix,
            category=first.category,
            confidence=confidence,
        )

    def rank_by_severity(
        self, findings: list[ConsolidatedFinding]
    ) -> list[ConsolidatedFinding]:
        """Sort findings by severity and priority score.

        Priority score formula:
        - CRITICAL = 100, HIGH = 75, MEDIUM = 50, LOW = 25, INFO = 10
        - Bonus: +10 per additional reviewer that reported it
        - Bonus: +5 if suggested_fix is present

        Args:
            findings: List of consolidated findings.

        Returns:
            Sorted list (highest priority first) with priority_score populated.
        """
        for finding in findings:
            base_score = SEVERITY_SCORES[finding.severity]

            # Bonus for multiple reviewers
            reviewer_bonus = (len(finding.reviewer_types) - 1) * 10.0

            # Bonus for having a suggested fix
            fix_bonus = 5.0 if finding.suggested_fix else 0.0

            finding.priority_score = base_score + reviewer_bonus + fix_bonus

        # Sort by priority_score descending
        ranked = sorted(findings, key=lambda f: f.priority_score, reverse=True)

        self._logger.info(
            "findings_ranked",
            count=len(ranked),
            top_priority=ranked[0].priority_score if ranked else 0.0,
        )

        return ranked

    def generate_fix_tasks(
        self, findings: list[ConsolidatedFinding]
    ) -> list[FixTask]:
        """Generate fix tasks from consolidated findings.

        Rules:
        - CRITICAL/HIGH findings always get a fix task
        - MEDIUM findings get a fix task if confidence > 0.7
        - LOW/INFO findings do not get fix tasks
        - Group findings by file_path to create combined tasks
        - Set agent_type based on finding category:
          - security -> reviewer_security
          - performance -> executor
          - documentation -> writer
          - default -> executor

        Args:
            findings: List of consolidated findings (should be ranked).

        Returns:
            List of generated fix tasks.
        """
        # Filter findings that need fix tasks
        eligible_findings: list[ConsolidatedFinding] = []

        for finding in findings:
            if finding.severity in {FindingSeverity.CRITICAL, FindingSeverity.HIGH}:
                eligible_findings.append(finding)
            elif (
                finding.severity == FindingSeverity.MEDIUM and finding.confidence > 0.7
            ):
                eligible_findings.append(finding)
            # LOW and INFO do not get fix tasks

        # Group findings by file_path (None grouped together)
        file_groups: dict[str | None, list[ConsolidatedFinding]] = {}
        for finding in eligible_findings:
            key = finding.file_path
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(finding)

        # Generate fix tasks per file group
        fix_tasks: list[FixTask] = []

        for file_path, group_findings in file_groups.items():
            if len(group_findings) == 1:
                # Single finding: create straightforward fix task
                task = self._create_fix_task(group_findings[0])
            else:
                # Multiple findings in same file: combine into one task
                task = self._create_combined_fix_task(group_findings, file_path)

            fix_tasks.append(task)

        self._logger.info(
            "fix_tasks_generated",
            eligible_findings=len(eligible_findings),
            fix_tasks_created=len(fix_tasks),
        )

        return fix_tasks

    def _create_fix_task(self, finding: ConsolidatedFinding) -> FixTask:
        """Create a fix task from a single finding.

        Args:
            finding: The consolidated finding.

        Returns:
            A FixTask addressing the finding.
        """
        agent_type = self._select_agent_type(finding.category)

        files = [finding.file_path] if finding.file_path else []

        description_parts = [
            f"# Fix: {finding.title}",
            "",
            finding.description,
            "",
            f"**Severity:** {finding.severity.value}",
            f"**Confidence:** {finding.confidence:.2f}",
        ]

        if finding.file_path:
            loc = f"{finding.file_path}"
            if finding.line_number:
                loc += f":{finding.line_number}"
            description_parts.append(f"**Location:** {loc}")

        if finding.suggested_fix:
            description_parts.extend(
                [
                    "",
                    "## Suggested Fix",
                    finding.suggested_fix,
                ]
            )

        description = "\n".join(description_parts)

        # Use finding's priority_score, but ensure it's not 0 if unset
        priority = finding.priority_score
        if priority == 0.0:
            # Compute it if not set
            priority = SEVERITY_SCORES[finding.severity]

        return FixTask(
            finding_id=finding.id,
            title=f"Fix: {finding.title}",
            description=description,
            files_to_modify=files,
            priority=priority,
            estimated_complexity=self._estimate_complexity(finding),
            agent_type=agent_type,
        )

    def _create_combined_fix_task(
        self, findings: list[ConsolidatedFinding], file_path: str | None
    ) -> FixTask:
        """Create a fix task from multiple findings in the same file.

        Args:
            findings: List of findings to combine.
            file_path: The file path common to all findings (or None).

        Returns:
            A combined FixTask addressing all findings.
        """
        # Use highest priority finding as primary
        primary = max(findings, key=lambda f: f.priority_score)

        agent_type = self._select_agent_type(primary.category)

        files = [file_path] if file_path else []

        # Combine titles
        combined_title = f"Fix {len(findings)} issues in {file_path or 'multiple files'}"

        # Build description with all findings
        description_parts = [
            f"# {combined_title}",
            "",
            f"This task addresses {len(findings)} related findings:",
            "",
        ]

        for idx, finding in enumerate(findings, start=1):
            description_parts.extend(
                [
                    f"## {idx}. {finding.title}",
                    f"**Severity:** {finding.severity.value}",
                    f"**Confidence:** {finding.confidence:.2f}",
                    "",
                    finding.description,
                    "",
                ]
            )
            if finding.suggested_fix:
                description_parts.extend(
                    [
                        "**Suggested Fix:**",
                        finding.suggested_fix,
                        "",
                    ]
                )

        description = "\n".join(description_parts)

        # Use max priority among all findings
        max_priority = max(f.priority_score for f in findings)

        # Complexity: COMPLEX if 3+ findings, MEDIUM if 2, SIMPLE if 1
        if len(findings) >= 3:
            complexity = TaskComplexity.COMPLEX
        elif len(findings) == 2:
            complexity = TaskComplexity.MEDIUM
        else:
            complexity = TaskComplexity.SIMPLE

        return FixTask(
            finding_id=primary.id,  # Use primary finding ID
            title=combined_title,
            description=description,
            files_to_modify=files,
            priority=max_priority,
            estimated_complexity=complexity,
            agent_type=agent_type,
        )

    def _select_agent_type(self, category: str | None) -> str:
        """Select agent type based on finding category.

        Args:
            category: The finding category (e.g., "security", "performance").

        Returns:
            Appropriate agent type string.
        """
        if category is None:
            return "executor"

        category_lower = category.lower()

        if "security" in category_lower:
            return "reviewer_security"
        elif "performance" in category_lower:
            return "executor"
        elif "documentation" in category_lower or "docs" in category_lower:
            return "writer"
        else:
            return "executor"

    def _estimate_complexity(self, finding: ConsolidatedFinding) -> TaskComplexity:
        """Estimate complexity of fixing a finding.

        Args:
            finding: The consolidated finding.

        Returns:
            TaskComplexity estimate.
        """
        # CRITICAL severity tends to be more complex
        if finding.severity == FindingSeverity.CRITICAL:
            return TaskComplexity.COMPLEX

        # Multiple reviewers agree -> likely more fundamental issue
        if len(finding.reviewer_types) >= 3:
            return TaskComplexity.COMPLEX

        # Has suggested fix and single reviewer -> likely simple
        if finding.suggested_fix and len(finding.reviewer_types) == 1:
            return TaskComplexity.SIMPLE

        return TaskComplexity.MEDIUM

    def consolidate(
        self, findings: list[ReviewFinding]
    ) -> tuple[list[ConsolidatedFinding], list[FixTask]]:
        """Full consolidation pipeline: deduplicate -> rank -> generate tasks.

        This is the primary entry point that runs the complete consolidation
        workflow on raw review findings.

        Args:
            findings: List of raw findings from all reviewers.

        Returns:
            Tuple of (consolidated_findings, fix_tasks) where both lists
            are sorted by priority descending.
        """
        self._logger.info("consolidation_started", raw_finding_count=len(findings))

        # Step 1: Deduplicate
        consolidated = self.deduplicate(findings)

        # Step 2: Rank by severity and priority
        ranked = self.rank_by_severity(consolidated)

        # Step 3: Generate fix tasks
        fix_tasks = self.generate_fix_tasks(ranked)

        self._logger.info(
            "consolidation_completed",
            consolidated_findings=len(ranked),
            fix_tasks=len(fix_tasks),
        )

        return ranked, fix_tasks

    @staticmethod
    def _compute_similarity(a: str, b: str) -> float:
        """Compute string similarity between two strings.

        Uses word-level Jaccard similarity: intersection over union of words
        after lowercasing and splitting.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Float between 0.0 (no similarity) and 1.0 (identical).
        """
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())

        if not words_a and not words_b:
            return 1.0

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)
