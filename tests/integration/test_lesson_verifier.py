"""Integration tests for lesson verification protocol.

Tests the LessonVerifier class which verifies that lessons learned
are still valid by discovering and running associated tests. Covers
enum values, models, test discovery strategies, test execution,
verification protocol, status determination, and batch verification.

Note: These tests mock subprocess execution to avoid dependency on
a real test suite. They focus on testing the verification logic and
decision matrices rather than actual pytest execution.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from forgemaster.database.queries.lesson import create_lesson
from forgemaster.database.queries.project import create_project
from forgemaster.database.queries.task import create_task
from forgemaster.intelligence.lesson_verifier import (
    LessonVerifier,
    TestDiscoveryResult,
    TestExecutionResult,
    VerificationResult,
    VerificationStatus,
)


# ========================================================================
# Enum tests
# ========================================================================


@pytest.mark.asyncio
async def test_verification_status_enum_values() -> None:
    """Test that VerificationStatus enum has all expected values."""
    assert VerificationStatus.UNVERIFIED.value == "unverified"
    assert VerificationStatus.PENDING.value == "pending"
    assert VerificationStatus.VERIFIED.value == "verified"
    assert VerificationStatus.INVALIDATED.value == "invalidated"
    assert VerificationStatus.SKIPPED.value == "skipped"

    # Verify all five values exist
    all_values = {s.value for s in VerificationStatus}
    assert all_values == {
        "unverified",
        "pending",
        "verified",
        "invalidated",
        "skipped",
    }


# ========================================================================
# Model tests
# ========================================================================


@pytest.mark.asyncio
async def test_test_discovery_result_defaults() -> None:
    """Test TestDiscoveryResult model with default values."""
    result = TestDiscoveryResult(lesson_id="test-lesson-id")

    assert result.lesson_id == "test-lesson-id"
    assert result.test_files == []
    assert result.test_functions == []
    assert result.discovery_method == "file_mapping"
    assert result.confidence == 0.5


@pytest.mark.asyncio
async def test_test_discovery_result_with_values() -> None:
    """Test TestDiscoveryResult model with all fields populated."""
    result = TestDiscoveryResult(
        lesson_id="lesson-123",
        test_files=["tests/unit/test_module.py"],
        test_functions=["test_function_a", "test_function_b"],
        discovery_method="pattern_matching",
        confidence=0.8,
    )

    assert result.lesson_id == "lesson-123"
    assert result.test_files == ["tests/unit/test_module.py"]
    assert result.test_functions == ["test_function_a", "test_function_b"]
    assert result.discovery_method == "pattern_matching"
    assert result.confidence == 0.8


@pytest.mark.asyncio
async def test_test_execution_result_passed() -> None:
    """Test TestExecutionResult model for a passing test."""
    result = TestExecutionResult(
        test_file="tests/unit/test_sample.py",
        test_function="test_passes",
        passed=True,
        output="1 passed in 0.05s",
        duration_seconds=0.05,
    )

    assert result.test_file == "tests/unit/test_sample.py"
    assert result.test_function == "test_passes"
    assert result.passed is True
    assert result.output == "1 passed in 0.05s"
    assert result.duration_seconds == 0.05
    assert result.error_message is None


@pytest.mark.asyncio
async def test_test_execution_result_failed() -> None:
    """Test TestExecutionResult model for a failing test."""
    result = TestExecutionResult(
        test_file="tests/unit/test_sample.py",
        test_function="test_fails",
        passed=False,
        output="FAILED test_sample.py::test_fails",
        duration_seconds=0.12,
        error_message="AssertionError: expected 5, got 3",
    )

    assert result.passed is False
    assert result.error_message == "AssertionError: expected 5, got 3"


@pytest.mark.asyncio
async def test_verification_result_defaults() -> None:
    """Test VerificationResult model with default values."""
    result = VerificationResult(
        lesson_id="lesson-456",
        status=VerificationStatus.VERIFIED,
    )

    assert result.lesson_id == "lesson-456"
    assert result.status == VerificationStatus.VERIFIED
    assert result.pre_fix_results == []
    assert result.post_fix_results == []
    assert result.pre_fix_pass_rate == 0.0
    assert result.post_fix_pass_rate == 0.0
    assert result.notes is None
    # verified_at should be a recent datetime
    assert isinstance(result.verified_at, datetime)
    assert result.verified_at.tzinfo == timezone.utc


# ========================================================================
# Source-to-test mapping tests
# ========================================================================


@pytest.mark.asyncio
async def test_map_source_to_test_nested_module(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test mapping nested source module to test file."""
    # Create directory structure
    test_dir = tmp_path / "tests" / "unit"
    test_dir.mkdir(parents=True)
    (test_dir / "test_review_cycle.py").touch()

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # Map nested module: src/forgemaster/review/cycle.py
    result = verifier._map_source_to_test("src/forgemaster/review/cycle.py")

    assert result == "tests/unit/test_review_cycle.py"


@pytest.mark.asyncio
async def test_map_source_to_test_flat_module(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test mapping flat source module to test file."""
    test_dir = tmp_path / "tests" / "unit"
    test_dir.mkdir(parents=True)
    (test_dir / "test_config.py").touch()

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = verifier._map_source_to_test("src/forgemaster/config.py")

    assert result == "tests/unit/test_config.py"


@pytest.mark.asyncio
async def test_map_source_to_test_integration_fallback(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test mapping falls back to integration tests when unit test missing."""
    integration_dir = tmp_path / "tests" / "integration"
    integration_dir.mkdir(parents=True)
    (integration_dir / "test_database.py").touch()

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = verifier._map_source_to_test("src/forgemaster/database.py")

    assert result == "tests/integration/test_database.py"


@pytest.mark.asyncio
async def test_map_source_to_test_init_py_skipped(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test that __init__.py files are not mapped."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = verifier._map_source_to_test("src/forgemaster/__init__.py")

    assert result is None


@pytest.mark.asyncio
async def test_map_source_to_test_non_python_skipped(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test that non-Python files are not mapped."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = verifier._map_source_to_test("README.md")

    assert result is None


# ========================================================================
# Test discovery tests
# ========================================================================


@pytest.mark.asyncio
async def test_discover_tests_file_mapping_success(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test discovery via file mapping when test files exist."""
    # Setup database
    project = await create_project(db_session, name="Test Project", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Import error",
        root_cause="Missing dependency",
        fix_applied="Added to pyproject.toml",
        files_affected=["src/forgemaster/config.py"],
    )

    # Create test file
    test_dir = tmp_path / "tests" / "unit"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_config.py"
    test_file.write_text("def test_config_loads():\n    pass\n")

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = await verifier.discover_tests(str(lesson.id))

    assert result.lesson_id == str(lesson.id)
    assert result.test_files == ["tests/unit/test_config.py"]
    assert "test_config_loads" in result.test_functions
    assert result.discovery_method == "file_mapping"
    assert result.confidence == 0.8


@pytest.mark.asyncio
async def test_discover_tests_pattern_matching_fallback(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test discovery via pattern matching when file mapping fails."""
    # Setup database
    project = await create_project(db_session, name="Pattern Test", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Function error",
        root_cause="Logic bug",
        fix_applied="Fixed condition",
        files_affected=["src/forgemaster/helper.py"],
    )

    # Create test file that imports the module (not in conventional location)
    test_dir = tmp_path / "tests" / "custom"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_helpers.py"
    test_file.write_text(
        "from forgemaster.helper import some_function\n\n"
        "def test_helper():\n    pass\n"
    )

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = await verifier.discover_tests(str(lesson.id))

    assert result.discovery_method == "pattern_matching"
    assert len(result.test_files) == 1
    assert result.test_files[0].endswith("test_helpers.py")
    assert result.confidence == 0.6


@pytest.mark.asyncio
async def test_discover_tests_keyword_search_fallback(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test discovery via keyword search when other methods fail."""
    # Setup database
    project = await create_project(db_session, name="Keyword Test", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Database connection timeout",
        root_cause="Pool exhausted",
        fix_applied="Increased pool size",
        files_affected=["src/other/unknown.py"],
    )

    # Create test file with symptom keywords
    test_dir = tmp_path / "tests" / "integration"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_database_connection.py"
    test_file.write_text(
        "def test_connection_timeout():\n"
        "    # Test database connection handling\n"
        "    pass\n"
    )

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = await verifier.discover_tests(str(lesson.id))

    assert result.discovery_method == "keyword_search"
    assert len(result.test_files) >= 1
    assert any("test_database_connection" in f for f in result.test_files)
    assert result.confidence == 0.3


@pytest.mark.asyncio
async def test_discover_tests_no_tests_found(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test discovery when no tests are found."""
    # Setup database
    project = await create_project(db_session, name="No Tests", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Something",
        root_cause="Unknown",
        fix_applied="Fixed it",
        files_affected=["nonexistent.py"],
    )

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = await verifier.discover_tests(str(lesson.id))

    assert result.test_files == []
    assert result.test_functions == []


@pytest.mark.asyncio
async def test_discover_tests_lesson_not_found(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test discovery when lesson does not exist."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    fake_id = str(uuid.uuid4())
    result = await verifier.discover_tests(fake_id)

    assert result.lesson_id == fake_id
    assert result.discovery_method == "none"
    assert result.confidence == 0.0


# ========================================================================
# Test execution tests
# ========================================================================


@pytest.mark.asyncio
async def test_run_tests_single_file_passes(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test running a single test file that passes."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(b"1 passed in 0.02s", b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        results = await verifier.run_tests(["tests/unit/test_sample.py"])

    assert len(results) == 1
    assert results[0].test_file == "tests/unit/test_sample.py"
    assert results[0].passed is True
    assert "1 passed" in results[0].output


@pytest.mark.asyncio
async def test_run_tests_single_file_fails(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test running a single test file that fails."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(
        return_value=(
            b"FAILED tests/unit/test_sample.py::test_fails",
            b"AssertionError: expected 5, got 3",
        )
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        results = await verifier.run_tests(["tests/unit/test_sample.py"])

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].error_message is not None
    assert "AssertionError" in results[0].error_message


@pytest.mark.asyncio
async def test_run_tests_with_specific_functions(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test running specific test functions."""
    # Create test file
    test_dir = tmp_path / "tests" / "unit"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_module.py"
    test_file.write_text(
        "def test_function_a():\n    pass\n\n"
        "def test_function_b():\n    pass\n"
    )

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(b"1 passed", b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        results = await verifier.run_tests(
            ["tests/unit/test_module.py"],
            ["test_function_a", "test_function_b"],
        )

    # Should run each function individually
    assert len(results) == 2


@pytest.mark.asyncio
async def test_run_tests_timeout(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test handling of test execution timeout."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        results = await verifier.run_tests(["tests/unit/test_slow.py"])

    assert len(results) == 1
    assert results[0].passed is False
    assert "timed out" in results[0].error_message.lower()


@pytest.mark.asyncio
async def test_run_tests_command_not_found(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test handling when test command is not found."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("pytest not found"),
    ):
        results = await verifier.run_tests(["tests/unit/test_sample.py"])

    assert len(results) == 1
    assert results[0].passed is False
    assert "not found" in results[0].error_message.lower()


@pytest.mark.asyncio
async def test_run_tests_empty_list(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test running tests with empty file list."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    results = await verifier.run_tests([])

    assert results == []


# ========================================================================
# Verification protocol tests
# ========================================================================


@pytest.mark.asyncio
async def test_verify_lesson_full_protocol(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test full verification protocol from discovery to status update."""
    # Setup database
    project = await create_project(db_session, name="Verify Test", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error occurred",
        root_cause="Bug in code",
        fix_applied="Fixed bug",
        files_affected=["src/forgemaster/module.py"],
    )

    # Create test file
    test_dir = tmp_path / "tests" / "unit"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_module.py"
    test_file.write_text("def test_module():\n    pass\n")

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # Mock test execution to return passing tests
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(b"1 passed", b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await verifier.verify_lesson(str(lesson.id))

    assert result.lesson_id == str(lesson.id)
    assert result.status == VerificationStatus.VERIFIED
    assert len(result.pre_fix_results) > 0
    assert len(result.post_fix_results) > 0
    assert result.post_fix_pass_rate == 1.0


@pytest.mark.asyncio
async def test_verify_lesson_skipped_no_tests(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test verification skipped when no tests are found."""
    # Setup database
    project = await create_project(db_session, name="Skip Test", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error",
        root_cause="Bug",
        fix_applied="Fixed",
        files_affected=["nonexistent.py"],
    )

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    result = await verifier.verify_lesson(str(lesson.id))

    assert result.status == VerificationStatus.SKIPPED
    assert "No related test files" in result.notes


# ========================================================================
# Status determination tests (decision matrix)
# ========================================================================


@pytest.mark.asyncio
async def test_determine_status_both_pass_verified(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test status determination when both pre and post tests pass."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    pre_results = [
        TestExecutionResult(
            test_file="test.py",
            passed=True,
            output="passed",
            duration_seconds=0.1,
        ),
    ]
    post_results = [
        TestExecutionResult(
            test_file="test.py",
            passed=True,
            output="passed",
            duration_seconds=0.1,
        ),
    ]

    status = verifier._determine_verification_status(pre_results, post_results)

    assert status == VerificationStatus.VERIFIED


@pytest.mark.asyncio
async def test_determine_status_post_fails_invalidated(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test status determination when post-fix tests fail."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    pre_results = [
        TestExecutionResult(
            test_file="test.py",
            passed=True,
            output="passed",
            duration_seconds=0.1,
        ),
    ]
    post_results = [
        TestExecutionResult(
            test_file="test.py",
            passed=False,
            output="failed",
            duration_seconds=0.1,
            error_message="Test failed",
        ),
    ]

    status = verifier._determine_verification_status(pre_results, post_results)

    assert status == VerificationStatus.INVALIDATED


@pytest.mark.asyncio
async def test_determine_status_both_fail_invalidated(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test status determination when both pre and post tests fail."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    pre_results = [
        TestExecutionResult(
            test_file="test.py",
            passed=False,
            output="failed",
            duration_seconds=0.1,
            error_message="Failed",
        ),
    ]
    post_results = [
        TestExecutionResult(
            test_file="test.py",
            passed=False,
            output="failed",
            duration_seconds=0.1,
            error_message="Failed",
        ),
    ]

    status = verifier._determine_verification_status(pre_results, post_results)

    assert status == VerificationStatus.INVALIDATED


@pytest.mark.asyncio
async def test_determine_status_pre_fails_post_passes_verified(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test status when pre-fix fails but post-fix passes (fix improved)."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    pre_results = [
        TestExecutionResult(
            test_file="test.py",
            passed=False,
            output="failed",
            duration_seconds=0.1,
            error_message="Failed",
        ),
    ]
    post_results = [
        TestExecutionResult(
            test_file="test.py",
            passed=True,
            output="passed",
            duration_seconds=0.1,
        ),
    ]

    status = verifier._determine_verification_status(pre_results, post_results)

    assert status == VerificationStatus.VERIFIED


@pytest.mark.asyncio
async def test_determine_status_no_results_skipped(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test status determination when no results are available."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    status = verifier._determine_verification_status([], [])

    assert status == VerificationStatus.SKIPPED


@pytest.mark.asyncio
async def test_determine_status_high_pass_rate_verified(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test status determination with high pass rate (>= 80%)."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # 4 out of 5 tests pass (80%)
    post_results = [
        TestExecutionResult(
            test_file=f"test{i}.py",
            passed=i < 4,
            output="result",
            duration_seconds=0.1,
        )
        for i in range(5)
    ]

    status = verifier._determine_verification_status([], post_results)

    assert status == VerificationStatus.VERIFIED


@pytest.mark.asyncio
async def test_determine_status_post_better_than_pre_verified(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test status when post-fix improves over pre-fix (marginal case)."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # Pre: 30% pass, Post: 60% pass
    pre_results = [
        TestExecutionResult(
            test_file=f"test{i}.py",
            passed=i < 3,
            output="result",
            duration_seconds=0.1,
        )
        for i in range(10)
    ]
    post_results = [
        TestExecutionResult(
            test_file=f"test{i}.py",
            passed=i < 6,
            output="result",
            duration_seconds=0.1,
        )
        for i in range(10)
    ]

    status = verifier._determine_verification_status(pre_results, post_results)

    assert status == VerificationStatus.VERIFIED


# ========================================================================
# Confidence computation tests
# ========================================================================


@pytest.mark.asyncio
async def test_compute_confidence_high_discovery_high_pass(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test confidence computation with high discovery confidence and pass rate."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # 0.8 * 0.3 + 1.0 * 0.7 = 0.24 + 0.7 = 0.94
    confidence = verifier._compute_confidence(
        discovery_confidence=0.8,
        post_fix_pass_rate=1.0,
    )

    assert confidence == 0.94


@pytest.mark.asyncio
async def test_compute_confidence_low_discovery_low_pass(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test confidence computation with low values."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # 0.3 * 0.3 + 0.2 * 0.7 = 0.09 + 0.14 = 0.23
    confidence = verifier._compute_confidence(
        discovery_confidence=0.3,
        post_fix_pass_rate=0.2,
    )

    assert confidence == 0.23


@pytest.mark.asyncio
async def test_compute_confidence_weighted_average(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test that pass rate is weighted more heavily (70%) than discovery (30%)."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # 0.5 * 0.3 + 0.5 * 0.7 = 0.15 + 0.35 = 0.5
    confidence = verifier._compute_confidence(
        discovery_confidence=0.5,
        post_fix_pass_rate=0.5,
    )

    assert confidence == 0.5


@pytest.mark.asyncio
async def test_compute_confidence_clamped_to_range(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test that confidence is clamped to [0.0, 1.0] range."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # Should not exceed 1.0
    confidence_high = verifier._compute_confidence(
        discovery_confidence=1.0,
        post_fix_pass_rate=1.0,
    )
    assert confidence_high <= 1.0

    # Should not go below 0.0
    confidence_low = verifier._compute_confidence(
        discovery_confidence=0.0,
        post_fix_pass_rate=0.0,
    )
    assert confidence_low >= 0.0


# ========================================================================
# Status update tests
# ========================================================================


@pytest.mark.asyncio
async def test_update_verification_status(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test updating lesson verification status in database."""
    # Setup database
    project = await create_project(db_session, name="Update Test", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error",
        root_cause="Bug",
        fix_applied="Fixed",
    )

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # Update status
    await verifier.update_verification_status(
        str(lesson.id),
        VerificationStatus.VERIFIED,
        confidence_score=0.85,
    )

    # Verify update
    async with session_factory() as session:
        from forgemaster.database.queries.lesson import get_lesson
        updated = await get_lesson(session, lesson.id)
        assert updated is not None
        assert updated.verification_status == "verified"
        assert updated.confidence_score == 0.85


@pytest.mark.asyncio
async def test_update_verification_status_lesson_not_found(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test updating status for non-existent lesson raises error."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    fake_id = str(uuid.uuid4())

    with pytest.raises(ValueError, match="not found"):
        await verifier.update_verification_status(
            fake_id,
            VerificationStatus.VERIFIED,
        )


# ========================================================================
# Batch verification tests
# ========================================================================


@pytest.mark.asyncio
async def test_verify_all_pending(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test batch verification of all pending lessons."""
    # Setup database with multiple lessons
    project = await create_project(db_session, name="Batch Test", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )

    # Create test file
    test_dir = tmp_path / "tests" / "unit"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_batch.py"
    test_file.write_text("def test_batch():\n    pass\n")

    lesson1 = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error 1",
        root_cause="Bug 1",
        fix_applied="Fix 1",
        files_affected=["src/forgemaster/batch.py"],
    )
    lesson2 = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error 2",
        root_cause="Bug 2",
        fix_applied="Fix 2",
        files_affected=["src/forgemaster/batch.py"],
    )

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # Mock test execution
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(b"1 passed", b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        results = await verifier.verify_all_pending(str(project.id))

    assert len(results) == 2
    assert all(r.status == VerificationStatus.VERIFIED for r in results)


@pytest.mark.asyncio
async def test_verify_all_pending_mixed_results(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test batch verification with mixed verified/invalidated/skipped results."""
    # Setup database
    project = await create_project(db_session, name="Mixed Test", config={})
    task = await create_task(
        db_session,
        project_id=project.id,
        title="Test Task",
        agent_type="executor",
    )

    # Create test file for first lesson
    test_dir = tmp_path / "tests" / "unit"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_mixed.py"
    test_file.write_text("def test_mixed():\n    pass\n")

    # Lesson with tests (will be verified)
    lesson1 = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error 1",
        root_cause="Bug 1",
        fix_applied="Fix 1",
        files_affected=["src/forgemaster/mixed.py"],
    )

    # Lesson without tests (will be skipped)
    lesson2 = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error 2",
        root_cause="Bug 2",
        fix_applied="Fix 2",
        files_affected=["nonexistent.py"],
    )

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # Mock test execution to pass
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(b"1 passed", b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        results = await verifier.verify_all_pending(str(project.id))

    assert len(results) == 2
    statuses = {r.status for r in results}
    assert VerificationStatus.VERIFIED in statuses
    assert VerificationStatus.SKIPPED in statuses


@pytest.mark.asyncio
async def test_verify_all_pending_no_lessons(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test batch verification when no pending lessons exist."""
    # Setup database with project but no lessons
    project = await create_project(db_session, name="Empty Test", config={})

    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    results = await verifier.verify_all_pending(str(project.id))

    assert results == []


# ========================================================================
# Pass rate calculation tests
# ========================================================================


@pytest.mark.asyncio
async def test_calculate_pass_rate_all_pass(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test pass rate calculation when all tests pass."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    results = [
        TestExecutionResult(
            test_file=f"test{i}.py",
            passed=True,
            output="passed",
            duration_seconds=0.1,
        )
        for i in range(5)
    ]

    pass_rate = verifier._calculate_pass_rate(results)

    assert pass_rate == 1.0


@pytest.mark.asyncio
async def test_calculate_pass_rate_all_fail(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test pass rate calculation when all tests fail."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    results = [
        TestExecutionResult(
            test_file=f"test{i}.py",
            passed=False,
            output="failed",
            duration_seconds=0.1,
            error_message="Failed",
        )
        for i in range(5)
    ]

    pass_rate = verifier._calculate_pass_rate(results)

    assert pass_rate == 0.0


@pytest.mark.asyncio
async def test_calculate_pass_rate_partial(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test pass rate calculation with partial passes."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    # 3 out of 5 pass = 0.6
    results = [
        TestExecutionResult(
            test_file=f"test{i}.py",
            passed=i < 3,
            output="result",
            duration_seconds=0.1,
        )
        for i in range(5)
    ]

    pass_rate = verifier._calculate_pass_rate(results)

    assert pass_rate == 0.6


@pytest.mark.asyncio
async def test_calculate_pass_rate_empty(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Test pass rate calculation with empty results list."""
    verifier = LessonVerifier(
        session_factory=session_factory,
        project_root=tmp_path,
    )

    pass_rate = verifier._calculate_pass_rate([])

    assert pass_rate == 0.0
