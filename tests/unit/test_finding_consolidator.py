"""Unit tests for the finding consolidator.

Tests deduplication logic, severity ranking, fix task generation,
file grouping, similarity computation, and the full consolidation pipeline.
"""

from __future__ import annotations

import pytest

from forgemaster.review.consolidator import (
    ConsolidatedFinding,
    FindingConsolidator,
    FixTask,
    TaskComplexity,
)
from forgemaster.review.cycle import FindingSeverity, ReviewFinding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def consolidator() -> FindingConsolidator:
    """Create a consolidator with default settings."""
    return FindingConsolidator(similarity_threshold=0.8)


@pytest.fixture
def sample_findings() -> list[ReviewFinding]:
    """Create sample findings for testing."""
    return [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.CRITICAL,
            title="SQL injection vulnerability",
            description="User input not sanitized before query execution.",
            file_path="src/db/queries.py",
            line_number=42,
            category="security",
        ),
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.HIGH,
            title="Missing error handling",
            description="Function does not handle exceptions.",
            file_path="src/api/handlers.py",
            line_number=100,
            category="errors",
        ),
        ReviewFinding(
            reviewer_type="spec",
            severity=FindingSeverity.MEDIUM,
            title="Missing required field validation",
            description="API spec requires 'email' field validation.",
            file_path="src/api/validation.py",
            line_number=25,
            category="spec",
        ),
    ]


# ---------------------------------------------------------------------------
# Similarity computation tests
# ---------------------------------------------------------------------------


def test_compute_similarity_identical() -> None:
    """Test similarity computation for identical strings."""
    result = FindingConsolidator._compute_similarity(
        "Missing error handling", "Missing error handling"
    )
    assert result == 1.0


def test_compute_similarity_completely_different() -> None:
    """Test similarity computation for completely different strings."""
    result = FindingConsolidator._compute_similarity(
        "SQL injection vulnerability", "Performance bottleneck detected"
    )
    # No common words
    assert result == 0.0


def test_compute_similarity_partial_match() -> None:
    """Test similarity computation for partial matches."""
    result = FindingConsolidator._compute_similarity(
        "Missing error handling in module", "Missing error handling"
    )
    # Common words: missing, error, handling
    # Union: missing, error, handling, in, module (5 words)
    # Jaccard: 3/5 = 0.6
    assert 0.55 <= result <= 0.65


def test_compute_similarity_case_insensitive() -> None:
    """Test that similarity computation is case-insensitive."""
    result = FindingConsolidator._compute_similarity(
        "Missing ERROR Handling", "missing error handling"
    )
    assert result == 1.0


def test_compute_similarity_empty_strings() -> None:
    """Test similarity computation with empty strings."""
    assert FindingConsolidator._compute_similarity("", "") == 1.0
    assert FindingConsolidator._compute_similarity("", "test") == 0.0
    assert FindingConsolidator._compute_similarity("test", "") == 0.0


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


def test_deduplicate_no_duplicates(
    consolidator: FindingConsolidator, sample_findings: list[ReviewFinding]
) -> None:
    """Test deduplication when there are no duplicates."""
    consolidated = consolidator.deduplicate(sample_findings)

    assert len(consolidated) == 3
    assert all(isinstance(f, ConsolidatedFinding) for f in consolidated)
    assert all(len(f.original_finding_ids) == 1 for f in consolidated)


def test_deduplicate_same_file_similar_title(
    consolidator: FindingConsolidator,
) -> None:
    """Test deduplication when findings have same file and similar title."""
    findings = [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.HIGH,
            title="Missing input validation",
            description="User input not validated in handler.",
            file_path="src/api/handlers.py",
            line_number=50,
            category="security",
        ),
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.MEDIUM,
            title="Missing input validation",  # Exact match
            description="Input validation is not implemented.",
            file_path="src/api/handlers.py",
            line_number=52,
            category="errors",
        ),
    ]

    consolidated = consolidator.deduplicate(findings)

    assert len(consolidated) == 1
    merged = consolidated[0]
    assert len(merged.original_finding_ids) == 2
    assert set(merged.reviewer_types) == {"security", "backend"}
    assert merged.severity == FindingSeverity.HIGH  # Highest severity
    assert merged.confidence > 0.7  # Bonus for multiple reviewers


def test_deduplicate_same_category_overlapping_lines(
    consolidator: FindingConsolidator,
) -> None:
    """Test deduplication when findings have same category and overlapping line range."""
    findings = [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.CRITICAL,
            title="Injection vulnerability",
            description="SQL injection possible.",
            file_path="src/db/queries.py",
            line_number=100,
            category="security",
        ),
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.HIGH,
            title="Database query issue",
            description="Unsafe query construction.",
            file_path="src/db/queries.py",
            line_number=103,
            category="security",
        ),
    ]

    consolidated = consolidator.deduplicate(findings)

    assert len(consolidated) == 1
    merged = consolidated[0]
    assert len(merged.original_finding_ids) == 2
    assert merged.severity == FindingSeverity.CRITICAL  # Highest severity


def test_deduplicate_different_files(consolidator: FindingConsolidator) -> None:
    """Test that findings in different files are not deduplicated."""
    findings = [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.HIGH,
            title="Missing input validation",
            description="Validation missing.",
            file_path="src/api/handlers.py",
            line_number=50,
            category="security",
        ),
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.MEDIUM,
            title="Missing input validation",
            description="Validation missing.",
            file_path="src/web/views.py",
            line_number=50,
            category="security",
        ),
    ]

    consolidated = consolidator.deduplicate(findings)

    assert len(consolidated) == 2  # Different files, not merged


def test_deduplicate_keeps_longest_description(
    consolidator: FindingConsolidator,
) -> None:
    """Test that deduplication keeps the longest description."""
    findings = [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.HIGH,
            title="SQL injection vulnerability",  # Exact match
            description="Short description.",
            file_path="src/db/queries.py",
            line_number=50,
        ),
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.HIGH,
            title="SQL injection vulnerability",
            description="This is a much longer and more detailed description of the issue.",
            file_path="src/db/queries.py",
            line_number=50,
        ),
    ]

    consolidated = consolidator.deduplicate(findings)

    assert len(consolidated) == 1
    assert "much longer and more detailed" in consolidated[0].description


def test_deduplicate_keeps_best_suggested_fix(
    consolidator: FindingConsolidator,
) -> None:
    """Test that deduplication keeps the longest suggested fix."""
    findings = [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.HIGH,
            title="SQL injection vulnerability",  # Exact match
            description="Issue found.",
            file_path="src/db/queries.py",
            line_number=50,
            suggested_fix="Use parameterized queries.",
        ),
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.HIGH,
            title="SQL injection vulnerability",
            description="Issue found.",
            file_path="src/db/queries.py",
            line_number=50,
            suggested_fix="Use parameterized queries with SQLAlchemy's text() and bindparams.",
        ),
    ]

    consolidated = consolidator.deduplicate(findings)

    assert len(consolidated) == 1
    assert "SQLAlchemy" in (consolidated[0].suggested_fix or "")


def test_deduplicate_empty_list(consolidator: FindingConsolidator) -> None:
    """Test deduplication with empty input."""
    consolidated = consolidator.deduplicate([])
    assert consolidated == []


# ---------------------------------------------------------------------------
# Severity ranking tests
# ---------------------------------------------------------------------------


def test_rank_by_severity_correct_order(
    consolidator: FindingConsolidator,
) -> None:
    """Test that ranking produces correct severity order."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.MEDIUM,
            title="Medium issue",
            description="Medium priority.",
            reviewer_types=["backend"],
        ),
        ConsolidatedFinding(
            severity=FindingSeverity.CRITICAL,
            title="Critical issue",
            description="Critical priority.",
            reviewer_types=["security"],
        ),
        ConsolidatedFinding(
            severity=FindingSeverity.LOW,
            title="Low issue",
            description="Low priority.",
            reviewer_types=["spec"],
        ),
    ]

    ranked = consolidator.rank_by_severity(findings)

    assert ranked[0].severity == FindingSeverity.CRITICAL
    assert ranked[1].severity == FindingSeverity.MEDIUM
    assert ranked[2].severity == FindingSeverity.LOW


def test_rank_by_severity_priority_scores(
    consolidator: FindingConsolidator,
) -> None:
    """Test that priority scores are computed correctly."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Desc.",
            reviewer_types=["security"],
            suggested_fix=None,
        ),
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="Issue with fix",
            description="Desc.",
            reviewer_types=["security", "backend"],
            suggested_fix="Fix it like this.",
        ),
    ]

    ranked = consolidator.rank_by_severity(findings)

    # Both HIGH (75), first has 1 reviewer (0), no fix (0) = 75
    assert ranked[1].priority_score == 75.0

    # Second has 2 reviewers (+10), has fix (+5) = 90
    assert ranked[0].priority_score == 90.0


def test_rank_by_severity_multiple_reviewers_bonus(
    consolidator: FindingConsolidator,
) -> None:
    """Test that multiple reviewers increase priority."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.MEDIUM,
            title="Issue",
            description="Desc.",
            reviewer_types=["security", "backend", "spec"],
        ),
    ]

    ranked = consolidator.rank_by_severity(findings)

    # MEDIUM = 50, 3 reviewers = +20, no fix = 0
    assert ranked[0].priority_score == 70.0


def test_rank_by_severity_suggested_fix_bonus(
    consolidator: FindingConsolidator,
) -> None:
    """Test that having a suggested fix increases priority."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.LOW,
            title="Issue",
            description="Desc.",
            reviewer_types=["backend"],
            suggested_fix="Fix this way.",
        ),
    ]

    ranked = consolidator.rank_by_severity(findings)

    # LOW = 25, 1 reviewer = 0, fix = +5
    assert ranked[0].priority_score == 30.0


def test_rank_by_severity_empty_list(consolidator: FindingConsolidator) -> None:
    """Test ranking with empty input."""
    ranked = consolidator.rank_by_severity([])
    assert ranked == []


# ---------------------------------------------------------------------------
# Fix task generation tests
# ---------------------------------------------------------------------------


def test_generate_fix_tasks_critical_always_included(
    consolidator: FindingConsolidator,
) -> None:
    """Test that CRITICAL findings always get fix tasks."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.CRITICAL,
            title="Critical issue",
            description="Must fix.",
            reviewer_types=["security"],
            file_path="src/app.py",
            confidence=0.5,  # Low confidence, but still CRITICAL
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].priority > 0


def test_generate_fix_tasks_high_always_included(
    consolidator: FindingConsolidator,
) -> None:
    """Test that HIGH findings always get fix tasks."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="High issue",
            description="Should fix.",
            reviewer_types=["backend"],
            file_path="src/app.py",
            confidence=0.5,
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1


def test_generate_fix_tasks_medium_confidence_threshold(
    consolidator: FindingConsolidator,
) -> None:
    """Test that MEDIUM findings need confidence > 0.7."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.MEDIUM,
            title="Medium high confidence",
            description="Desc.",
            reviewer_types=["backend", "security"],
            file_path="src/app.py",
            confidence=0.8,
        ),
        ConsolidatedFinding(
            severity=FindingSeverity.MEDIUM,
            title="Medium low confidence",
            description="Desc.",
            reviewer_types=["backend"],
            file_path="src/other.py",
            confidence=0.6,
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert "high confidence" in tasks[0].title


def test_generate_fix_tasks_low_not_included(
    consolidator: FindingConsolidator,
) -> None:
    """Test that LOW findings do not get fix tasks."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.LOW,
            title="Low priority",
            description="Minor issue.",
            reviewer_types=["spec"],
            file_path="src/app.py",
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 0


def test_generate_fix_tasks_info_not_included(
    consolidator: FindingConsolidator,
) -> None:
    """Test that INFO findings do not get fix tasks."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.INFO,
            title="Informational note",
            description="FYI.",
            reviewer_types=["backend"],
            file_path="src/app.py",
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 0


def test_generate_fix_tasks_groups_by_file(
    consolidator: FindingConsolidator,
) -> None:
    """Test that findings for the same file are grouped into one task."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="Issue 1",
            description="First issue.",
            reviewer_types=["security"],
            file_path="src/app.py",
            priority_score=75.0,
        ),
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="Issue 2",
            description="Second issue.",
            reviewer_types=["backend"],
            file_path="src/app.py",
            priority_score=75.0,
        ),
        ConsolidatedFinding(
            severity=FindingSeverity.CRITICAL,
            title="Issue 3",
            description="Third issue.",
            reviewer_types=["security"],
            file_path="src/other.py",
            priority_score=100.0,
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    # Two files, so two tasks
    assert len(tasks) == 2

    # Find the combined task for src/app.py
    app_tasks = [t for t in tasks if "src/app.py" in t.files_to_modify]
    assert len(app_tasks) == 1
    assert "2 issues" in app_tasks[0].title


def test_generate_fix_tasks_agent_type_security(
    consolidator: FindingConsolidator,
) -> None:
    """Test that security findings get reviewer_security agent."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.CRITICAL,
            title="Security issue",
            description="Desc.",
            reviewer_types=["security"],
            file_path="src/app.py",
            category="security",
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].agent_type == "reviewer_security"


def test_generate_fix_tasks_agent_type_performance(
    consolidator: FindingConsolidator,
) -> None:
    """Test that performance findings get executor agent."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="Performance issue",
            description="Desc.",
            reviewer_types=["backend"],
            file_path="src/app.py",
            category="performance",
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].agent_type == "executor"


def test_generate_fix_tasks_agent_type_documentation(
    consolidator: FindingConsolidator,
) -> None:
    """Test that documentation findings get writer agent."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="Missing docs",
            description="Desc.",
            reviewer_types=["spec"],
            file_path="src/app.py",
            category="documentation",
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].agent_type == "writer"


def test_generate_fix_tasks_complexity_critical(
    consolidator: FindingConsolidator,
) -> None:
    """Test that CRITICAL findings get COMPLEX complexity."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.CRITICAL,
            title="Critical issue",
            description="Desc.",
            reviewer_types=["security"],
            file_path="src/app.py",
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].estimated_complexity == TaskComplexity.COMPLEX


def test_generate_fix_tasks_complexity_multiple_reviewers(
    consolidator: FindingConsolidator,
) -> None:
    """Test that 3+ reviewers results in COMPLEX complexity."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Desc.",
            reviewer_types=["security", "backend", "spec"],
            file_path="src/app.py",
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].estimated_complexity == TaskComplexity.COMPLEX


def test_generate_fix_tasks_complexity_simple_with_fix(
    consolidator: FindingConsolidator,
) -> None:
    """Test that single reviewer + suggested fix = SIMPLE complexity."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Desc.",
            reviewer_types=["backend"],
            file_path="src/app.py",
            suggested_fix="Apply this fix.",
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].estimated_complexity == TaskComplexity.SIMPLE


def test_generate_fix_tasks_combined_complexity(
    consolidator: FindingConsolidator,
) -> None:
    """Test complexity for combined tasks (3+ findings = COMPLEX)."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.HIGH,
            title=f"Issue {i}",
            description="Desc.",
            reviewer_types=["backend"],
            file_path="src/app.py",
            priority_score=75.0,
        )
        for i in range(3)
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].estimated_complexity == TaskComplexity.COMPLEX


def test_generate_fix_tasks_empty_list(consolidator: FindingConsolidator) -> None:
    """Test fix task generation with empty input."""
    tasks = consolidator.generate_fix_tasks([])
    assert tasks == []


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


def test_consolidate_full_pipeline(
    consolidator: FindingConsolidator,
) -> None:
    """Test the full consolidation pipeline end-to-end."""
    findings = [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.CRITICAL,
            title="SQL injection vulnerability",
            description="User input not sanitized.",
            file_path="src/db/queries.py",
            line_number=42,
            category="security",
        ),
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.HIGH,
            title="SQL injection issue",
            description="Unsafe query construction.",
            file_path="src/db/queries.py",
            line_number=44,
            category="security",
        ),
        ReviewFinding(
            reviewer_type="spec",
            severity=FindingSeverity.LOW,
            title="Minor style issue",
            description="Code style inconsistency.",
            file_path="src/utils.py",
            line_number=10,
        ),
    ]

    consolidated, fix_tasks = consolidator.consolidate(findings)

    # Two findings should be merged into one
    assert len(consolidated) == 2

    # First finding should be highest priority (CRITICAL + merged)
    assert consolidated[0].severity == FindingSeverity.CRITICAL
    assert len(consolidated[0].reviewer_types) == 2

    # Only one fix task (LOW finding excluded)
    assert len(fix_tasks) == 1
    assert fix_tasks[0].priority > 0


def test_consolidate_empty_list(consolidator: FindingConsolidator) -> None:
    """Test full pipeline with empty input."""
    consolidated, fix_tasks = consolidator.consolidate([])

    assert consolidated == []
    assert fix_tasks == []


def test_consolidate_single_finding(consolidator: FindingConsolidator) -> None:
    """Test full pipeline with a single finding."""
    findings = [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.CRITICAL,
            title="Security issue",
            description="Critical security problem.",
            file_path="src/app.py",
            category="security",
        ),
    ]

    consolidated, fix_tasks = consolidator.consolidate(findings)

    assert len(consolidated) == 1
    assert len(fix_tasks) == 1
    assert fix_tasks[0].finding_id == consolidated[0].id


def test_consolidate_all_same_severity(
    consolidator: FindingConsolidator,
) -> None:
    """Test consolidation when all findings have the same severity."""
    findings = [
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.MEDIUM,
            title=f"Issue {i}",
            description=f"Description {i}.",
            file_path=f"src/file{i}.py",
        )
        for i in range(3)
    ]

    consolidated, fix_tasks = consolidator.consolidate(findings)

    assert len(consolidated) == 3
    assert all(f.severity == FindingSeverity.MEDIUM for f in consolidated)
    # MEDIUM findings need confidence > 0.7, default is 0.7 so won't generate tasks
    assert len(fix_tasks) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_deduplicate_line_range_boundary(
    consolidator: FindingConsolidator,
) -> None:
    """Test that line range boundary is exactly 5 lines."""
    findings = [
        ReviewFinding(
            reviewer_type="security",
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Desc.",
            file_path="src/app.py",
            line_number=100,
            category="security",
        ),
        ReviewFinding(
            reviewer_type="backend",
            severity=FindingSeverity.HIGH,
            title="Different issue",
            description="Desc.",
            file_path="src/app.py",
            line_number=105,
            category="security",
        ),
        ReviewFinding(
            reviewer_type="spec",
            severity=FindingSeverity.HIGH,
            title="Another issue",
            description="Desc.",
            file_path="src/app.py",
            line_number=106,
            category="security",
        ),
    ]

    consolidated = consolidator.deduplicate(findings)

    # First two should merge (within 5 lines), third should not (6 lines away)
    assert len(consolidated) == 2


def test_generate_fix_tasks_none_file_path(
    consolidator: FindingConsolidator,
) -> None:
    """Test fix task generation with None file_path."""
    findings = [
        ConsolidatedFinding(
            severity=FindingSeverity.CRITICAL,
            title="General issue",
            description="No specific file.",
            reviewer_types=["backend"],
            file_path=None,
        ),
    ]

    tasks = consolidator.generate_fix_tasks(findings)

    assert len(tasks) == 1
    assert tasks[0].files_to_modify == []
    assert "General issue" in tasks[0].title
