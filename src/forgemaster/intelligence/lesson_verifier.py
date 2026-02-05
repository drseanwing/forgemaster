"""Lesson verification protocol for Forgemaster.

Verifies that lessons learned are still valid by discovering and running
associated tests. The protocol follows these steps:

1. Discover tests related to the lesson's affected files
2. Run tests as a baseline (pre-fix state)
3. Run tests again to verify the fix holds (post-fix state)
4. Compare results and update the lesson's verification status

This module implements P4-034 through P4-038 of the Review Cycles +
Intelligence phase.

Example usage:
    >>> from forgemaster.intelligence.lesson_verifier import LessonVerifier
    >>> verifier = LessonVerifier(session_factory=get_session, project_root=Path("."))
    >>> result = await verifier.verify_lesson(str(lesson.id))
    >>> print(result.status)
    VerificationStatus.VERIFIED
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.lesson import LessonLearned

logger = structlog.get_logger(__name__)

# Type alias matching codebase convention
SessionFactory = Callable[[], AsyncSession]


class VerificationStatus(str, Enum):
    """Status of a lesson's verification.

    Attributes:
        UNVERIFIED: Lesson has not been verified yet.
        PENDING: Verification is in progress.
        VERIFIED: Lesson has been confirmed valid by passing tests.
        INVALIDATED: Lesson is no longer valid (tests contradict it).
        SKIPPED: Verification was skipped (no tests found, etc.).
    """

    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    INVALIDATED = "invalidated"
    SKIPPED = "skipped"


class TestDiscoveryResult(BaseModel):
    """Result of test discovery for a lesson.

    Attributes:
        lesson_id: UUID of the lesson that was searched for.
        test_files: List of test file paths discovered.
        test_functions: List of specific test function names found.
        discovery_method: Strategy used to find tests.
        confidence: Confidence in the discovery result (0.0 to 1.0).
    """

    lesson_id: str
    test_files: list[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)
    discovery_method: str = "file_mapping"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TestExecutionResult(BaseModel):
    """Result of a single test execution run.

    Attributes:
        test_file: Path to the test file that was executed.
        test_function: Specific test function name, if targeted.
        passed: Whether the test passed.
        output: Raw stdout/stderr output from the test runner.
        duration_seconds: Wall-clock time for the test run.
        error_message: Error details if the test failed.
    """

    test_file: str
    test_function: str | None = None
    passed: bool
    output: str
    duration_seconds: float
    error_message: str | None = None


class VerificationResult(BaseModel):
    """Complete verification result for a lesson.

    Aggregates pre-fix and post-fix test results into a single
    verification outcome with pass rates and status determination.

    Attributes:
        lesson_id: UUID of the verified lesson.
        status: Final verification status.
        pre_fix_results: Test results from baseline run.
        post_fix_results: Test results from post-fix run.
        pre_fix_pass_rate: Fraction of pre-fix tests that passed.
        post_fix_pass_rate: Fraction of post-fix tests that passed.
        verified_at: Timestamp when verification completed.
        notes: Optional human-readable notes about the verification.
    """

    lesson_id: str
    status: VerificationStatus
    pre_fix_results: list[TestExecutionResult] = Field(default_factory=list)
    post_fix_results: list[TestExecutionResult] = Field(default_factory=list)
    pre_fix_pass_rate: float = 0.0
    post_fix_pass_rate: float = 0.0
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = None


class LessonVerifier:
    """Verifies lessons learned by running associated tests.

    The verification protocol:
    1. Discover tests related to the lesson's affected files
    2. Run tests as a baseline (pre-fix)
    3. Apply the lesson's fix (conceptually -- we check if fix is already applied)
    4. Run tests again (post-fix)
    5. Compare results to determine if lesson is still valid

    Attributes:
        session_factory: Callable that returns an AsyncSession.
        project_root: Root directory of the project under test.
        test_command: Base command used to invoke tests (default: ``pytest``).
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        project_root: Path | None = None,
        test_command: str = "pytest",
    ) -> None:
        """Initialize the lesson verifier.

        Args:
            session_factory: Callable returning an AsyncSession.
            project_root: Project root for resolving file paths.
                Defaults to the current working directory.
            test_command: Shell command to run tests (default ``pytest``).
        """
        self.session_factory = session_factory
        self.project_root = project_root or Path.cwd()
        self.test_command = test_command

        logger.info(
            "lesson_verifier_initialized",
            project_root=str(self.project_root),
            test_command=self.test_command,
        )

    # ------------------------------------------------------------------
    # Test discovery (P4-034)
    # ------------------------------------------------------------------

    async def discover_tests(self, lesson_id: str) -> TestDiscoveryResult:
        """Discover tests related to a lesson's ``files_affected``.

        Strategy (in priority order):
        1. **File mapping**: Convert source paths to conventional test paths
           (e.g. ``src/forgemaster/review/cycle.py`` ->
           ``tests/unit/test_review_cycle.py``).
        2. **Pattern matching**: Search test directories for imports or
           references to the affected modules.
        3. **Keyword search**: Grep test files for symptom keywords from
           the lesson.

        Args:
            lesson_id: UUID string of the lesson to discover tests for.

        Returns:
            A ``TestDiscoveryResult`` describing what was found.
        """
        session = self.session_factory()
        try:
            lesson = await self._get_lesson(session, lesson_id)
        finally:
            await session.close()

        if lesson is None:
            logger.warning("lesson_not_found_for_discovery", lesson_id=lesson_id)
            return TestDiscoveryResult(
                lesson_id=lesson_id,
                discovery_method="none",
                confidence=0.0,
            )

        files_affected = lesson.files_affected or []
        test_files: list[str] = []
        test_functions: list[str] = []
        discovery_method = "file_mapping"
        confidence = 0.5

        # Strategy 1: File mapping
        for source_path in files_affected:
            mapped = self._map_source_to_test(source_path)
            if mapped is not None:
                full_path = self.project_root / mapped
                if full_path.exists():
                    test_files.append(mapped)

        if test_files:
            confidence = 0.8
            logger.info(
                "test_discovery_file_mapping",
                lesson_id=lesson_id,
                test_files=test_files,
                count=len(test_files),
            )
        else:
            # Strategy 2: Pattern matching -- search for module references
            discovery_method = "pattern_matching"
            pattern_files = await self._discover_by_pattern(files_affected)
            if pattern_files:
                test_files.extend(pattern_files)
                confidence = 0.6
                logger.info(
                    "test_discovery_pattern_matching",
                    lesson_id=lesson_id,
                    test_files=pattern_files,
                    count=len(pattern_files),
                )

        if not test_files:
            # Strategy 3: Keyword search from symptom
            discovery_method = "keyword_search"
            keyword_files = await self._discover_by_keyword(lesson.symptom)
            if keyword_files:
                test_files.extend(keyword_files)
                confidence = 0.3
                logger.info(
                    "test_discovery_keyword_search",
                    lesson_id=lesson_id,
                    test_files=keyword_files,
                    count=len(keyword_files),
                )

        # Deduplicate
        test_files = list(dict.fromkeys(test_files))

        # Discover specific test functions inside found files
        for tf in test_files:
            funcs = self._extract_test_functions(tf)
            test_functions.extend(funcs)

        return TestDiscoveryResult(
            lesson_id=lesson_id,
            test_files=test_files,
            test_functions=test_functions,
            discovery_method=discovery_method,
            confidence=confidence,
        )

    def _map_source_to_test(self, source_path: str) -> str | None:
        """Map a source file path to its expected test file path.

        Handles two conventions:
        - ``src/<package>/<module>.py`` -> ``tests/unit/test_<module>.py``
        - Also checks integration: ``tests/integration/test_<module>.py``

        For nested subpackages the module name is constructed by joining
        the last directory and file name with an underscore:
        ``src/forgemaster/review/cycle.py`` -> ``tests/unit/test_review_cycle.py``

        Args:
            source_path: Relative path to the source file.

        Returns:
            Relative path to the expected test file, or ``None`` if the
            source path cannot be mapped.
        """
        path = Path(source_path)

        # Skip non-Python files
        if path.suffix != ".py":
            return None

        # Skip __init__.py
        if path.name == "__init__.py":
            return None

        parts = path.parts

        # Find the index after 'src' or after the package root
        src_idx = -1
        for i, part in enumerate(parts):
            if part == "src":
                src_idx = i
                break

        if src_idx >= 0 and len(parts) > src_idx + 2:
            # Skip package name (e.g. "forgemaster")
            module_parts = list(parts[src_idx + 2 :])
        elif src_idx >= 0 and len(parts) > src_idx + 1:
            module_parts = list(parts[src_idx + 1 :])
        else:
            # No src prefix; use last two parts at most
            module_parts = list(parts[-2:]) if len(parts) >= 2 else list(parts)

        # Remove .py extension from the last part
        module_parts[-1] = Path(module_parts[-1]).stem

        # Build test file name: test_<parent>_<module>.py or test_<module>.py
        if len(module_parts) > 1:
            # e.g. review/cycle -> test_review_cycle.py
            test_name = "test_" + "_".join(module_parts) + ".py"
        else:
            test_name = "test_" + module_parts[0] + ".py"

        # Check unit first, then integration
        unit_path = Path("tests") / "unit" / test_name
        integration_path = Path("tests") / "integration" / test_name

        unit_full = self.project_root / unit_path
        integration_full = self.project_root / integration_path

        if unit_full.exists():
            return str(unit_path)
        if integration_full.exists():
            return str(integration_path)

        # Return the unit path as the conventional default even if it
        # doesn't exist yet (caller checks existence)
        return str(unit_path)

    async def _discover_by_pattern(self, files_affected: list[str]) -> list[str]:
        """Discover test files by searching for module name references.

        Scans the ``tests/`` directory for Python files that import or
        reference the modules named by ``files_affected``.

        Args:
            files_affected: Source file paths to search for references to.

        Returns:
            List of test file paths (relative to project root).
        """
        found: list[str] = []
        test_dir = self.project_root / "tests"

        if not test_dir.exists():
            return found

        # Extract module names from source paths
        module_names: list[str] = []
        for src in files_affected:
            stem = Path(src).stem
            if stem != "__init__":
                module_names.append(stem)

        if not module_names:
            return found

        # Walk all test files and check for references
        for test_file in test_dir.rglob("test_*.py"):
            try:
                content = test_file.read_text(encoding="utf-8")
                for module_name in module_names:
                    if module_name in content:
                        rel_path = str(test_file.relative_to(self.project_root))
                        # Normalize to forward slashes for consistency
                        rel_path = rel_path.replace("\\", "/")
                        found.append(rel_path)
                        break
            except OSError:
                continue

        return found

    async def _discover_by_keyword(self, symptom: str) -> list[str]:
        """Discover test files by searching for symptom keywords.

        Extracts meaningful words from the lesson symptom and searches
        test files for occurrences.

        Args:
            symptom: The lesson's symptom text.

        Returns:
            List of test file paths (relative to project root).
        """
        found: list[str] = []
        test_dir = self.project_root / "tests"

        if not test_dir.exists():
            return found

        # Extract keywords: words of 4+ characters, lowercased, excluding
        # common stop-words
        stop_words = {
            "with", "from", "that", "this", "have", "been", "were", "will",
            "would", "could", "should", "when", "what", "which", "where",
            "does", "about", "into", "then", "than", "some", "more", "also",
            "after", "before",
        }
        words = re.findall(r"[a-zA-Z_]\w{3,}", symptom)
        keywords = [w.lower() for w in words if w.lower() not in stop_words]

        if not keywords:
            return found

        for test_file in test_dir.rglob("test_*.py"):
            try:
                content = test_file.read_text(encoding="utf-8").lower()
                if any(kw in content for kw in keywords):
                    rel_path = str(test_file.relative_to(self.project_root))
                    rel_path = rel_path.replace("\\", "/")
                    found.append(rel_path)
            except OSError:
                continue

        return found

    def _extract_test_functions(self, test_file: str) -> list[str]:
        """Extract ``test_`` function names from a test file.

        Args:
            test_file: Path to the test file (relative to project root).

        Returns:
            List of function names starting with ``test_``.
        """
        full_path = self.project_root / test_file
        functions: list[str] = []

        if not full_path.exists():
            return functions

        try:
            content = full_path.read_text(encoding="utf-8")
            # Match def test_... or async def test_...
            matches = re.findall(r"(?:async\s+)?def\s+(test_\w+)", content)
            functions.extend(matches)
        except OSError:
            logger.warning("test_file_read_error", path=test_file)

        return functions

    # ------------------------------------------------------------------
    # Test execution (P4-035 / P4-036)
    # ------------------------------------------------------------------

    async def run_tests(
        self,
        test_files: list[str],
        test_functions: list[str] | None = None,
    ) -> list[TestExecutionResult]:
        """Run specified tests and collect results.

        Uses ``asyncio.create_subprocess_exec`` to invoke ``pytest`` with
        ``--tb=short`` for compact tracebacks and ``-q`` for quiet output.

        When *test_functions* is provided, each function is executed
        individually so that per-function pass/fail granularity is captured.
        Otherwise, each file is run as a whole.

        Args:
            test_files: Test file paths (relative to project root).
            test_functions: Optional specific test function names to run.

        Returns:
            List of ``TestExecutionResult`` objects, one per invocation.
        """
        results: list[TestExecutionResult] = []

        if not test_files:
            return results

        if test_functions:
            # Run each function individually for granularity
            for func_name in test_functions:
                # Find which file contains this function
                target_file = self._find_file_for_function(test_files, func_name)
                if target_file is None:
                    continue

                result = await self._run_single_test(target_file, func_name)
                results.append(result)
        else:
            # Run each test file
            for test_file in test_files:
                result = await self._run_single_test(test_file)
                results.append(result)

        return results

    def _find_file_for_function(
        self,
        test_files: list[str],
        func_name: str,
    ) -> str | None:
        """Find which test file contains a given function.

        Args:
            test_files: Candidate test file paths.
            func_name: Function name to search for.

        Returns:
            The test file path containing the function, or ``None``.
        """
        for tf in test_files:
            full_path = self.project_root / tf
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8")
                if re.search(rf"def\s+{re.escape(func_name)}\s*\(", content):
                    return tf
            except OSError:
                continue
        return None

    async def _run_single_test(
        self,
        test_file: str,
        test_function: str | None = None,
    ) -> TestExecutionResult:
        """Execute a single test file or function via subprocess.

        Args:
            test_file: Path to the test file (relative to project root).
            test_function: Optional specific function to run.

        Returns:
            A ``TestExecutionResult`` capturing outcome and timing.
        """
        # Build command
        cmd_parts = [self.test_command, "--tb=short", "-q"]

        if test_function:
            # pytest node ID: file::function
            node_id = f"{test_file}::{test_function}"
            cmd_parts.append(node_id)
        else:
            cmd_parts.append(test_file)

        start_time = time.monotonic()

        logger.info(
            "test_execution_started",
            test_file=test_file,
            test_function=test_function,
            command=cmd_parts,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120.0,
            )

            duration = time.monotonic() - start_time
            output = stdout.decode("utf-8", errors="replace")
            err_output = stderr.decode("utf-8", errors="replace")
            combined_output = output + ("\n" + err_output if err_output else "")

            passed = process.returncode == 0

            error_message: str | None = None
            if not passed:
                error_message = err_output or output or f"Exit code {process.returncode}"

            logger.info(
                "test_execution_completed",
                test_file=test_file,
                test_function=test_function,
                passed=passed,
                return_code=process.returncode,
                duration_seconds=round(duration, 3),
            )

            return TestExecutionResult(
                test_file=test_file,
                test_function=test_function,
                passed=passed,
                output=combined_output,
                duration_seconds=round(duration, 3),
                error_message=error_message,
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            logger.error(
                "test_execution_timeout",
                test_file=test_file,
                test_function=test_function,
                timeout_seconds=120.0,
            )
            return TestExecutionResult(
                test_file=test_file,
                test_function=test_function,
                passed=False,
                output="",
                duration_seconds=round(duration, 3),
                error_message="Test execution timed out after 120 seconds",
            )

        except FileNotFoundError:
            duration = time.monotonic() - start_time
            logger.error(
                "test_command_not_found",
                test_command=self.test_command,
            )
            return TestExecutionResult(
                test_file=test_file,
                test_function=test_function,
                passed=False,
                output="",
                duration_seconds=round(duration, 3),
                error_message=f"Test command not found: {self.test_command}",
            )

        except OSError as exc:
            duration = time.monotonic() - start_time
            logger.error(
                "test_execution_error",
                test_file=test_file,
                error=str(exc),
            )
            return TestExecutionResult(
                test_file=test_file,
                test_function=test_function,
                passed=False,
                output="",
                duration_seconds=round(duration, 3),
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Verification protocol (P4-035 / P4-036 / P4-037)
    # ------------------------------------------------------------------

    async def verify_lesson(self, lesson_id: str) -> VerificationResult:
        """Execute the full verification protocol for a single lesson.

        Steps:
        1. Mark lesson status as ``pending``.
        2. Discover related tests.
        3. Run pre-fix tests (current state baseline).
        4. Run post-fix tests (same tests, conceptually after fix is applied
           -- since the fix is already in the codebase, we treat the current
           state as post-fix).
        5. Determine verification status from results.
        6. Update the lesson in the database.

        Args:
            lesson_id: UUID string of the lesson to verify.

        Returns:
            A ``VerificationResult`` summarising the outcome.
        """
        logger.info("lesson_verification_started", lesson_id=lesson_id)

        # Mark as pending
        await self.update_verification_status(
            lesson_id, VerificationStatus.PENDING,
        )

        # Step 1: Discover tests
        discovery = await self.discover_tests(lesson_id)

        if not discovery.test_files:
            logger.info(
                "lesson_verification_skipped_no_tests",
                lesson_id=lesson_id,
            )
            await self.update_verification_status(
                lesson_id, VerificationStatus.SKIPPED,
            )
            return VerificationResult(
                lesson_id=lesson_id,
                status=VerificationStatus.SKIPPED,
                notes="No related test files discovered",
            )

        # Step 2: Run pre-fix tests (baseline)
        pre_fix_results = await self.run_tests(
            discovery.test_files,
            discovery.test_functions if discovery.test_functions else None,
        )
        pre_fix_pass_rate = self._calculate_pass_rate(pre_fix_results)

        # Step 3: Run post-fix tests
        # Since the fix is already applied in the codebase, we run the
        # same tests again. In a more advanced implementation this would
        # involve checking out the pre-fix state for step 2 and then
        # re-applying the fix. For now, the two runs serve as a stability
        # check and the post-fix run is authoritative.
        post_fix_results = await self.run_tests(
            discovery.test_files,
            discovery.test_functions if discovery.test_functions else None,
        )
        post_fix_pass_rate = self._calculate_pass_rate(post_fix_results)

        # Step 4: Determine status
        status = self._determine_verification_status(
            pre_fix_results, post_fix_results,
        )

        # Step 5: Calculate confidence adjustment
        confidence = self._compute_confidence(
            discovery.confidence,
            post_fix_pass_rate,
        )

        # Step 6: Persist
        await self.update_verification_status(
            lesson_id, status, confidence_score=confidence,
        )

        notes = (
            f"Discovery method: {discovery.discovery_method}, "
            f"pre-fix pass rate: {pre_fix_pass_rate:.0%}, "
            f"post-fix pass rate: {post_fix_pass_rate:.0%}"
        )

        logger.info(
            "lesson_verification_completed",
            lesson_id=lesson_id,
            status=status.value,
            pre_fix_pass_rate=pre_fix_pass_rate,
            post_fix_pass_rate=post_fix_pass_rate,
            confidence=confidence,
        )

        return VerificationResult(
            lesson_id=lesson_id,
            status=status,
            pre_fix_results=pre_fix_results,
            post_fix_results=post_fix_results,
            pre_fix_pass_rate=pre_fix_pass_rate,
            post_fix_pass_rate=post_fix_pass_rate,
            notes=notes,
        )

    async def verify_all_pending(
        self,
        project_id: str,
    ) -> list[VerificationResult]:
        """Verify all lessons with ``unverified`` or ``pending`` status.

        Args:
            project_id: UUID string of the project whose lessons to verify.

        Returns:
            List of ``VerificationResult`` for each processed lesson.
        """
        session = self.session_factory()
        try:
            stmt = (
                select(LessonLearned)
                .where(LessonLearned.project_id == uuid.UUID(project_id))
                .where(
                    LessonLearned.verification_status.in_(
                        [VerificationStatus.UNVERIFIED.value, VerificationStatus.PENDING.value]
                    )
                )
                .order_by(LessonLearned.created_at.asc())
            )
            result = await session.execute(stmt)
            lessons = list(result.scalars().all())
        finally:
            await session.close()

        logger.info(
            "verify_all_pending_started",
            project_id=project_id,
            lesson_count=len(lessons),
        )

        results: list[VerificationResult] = []
        for lesson in lessons:
            vr = await self.verify_lesson(str(lesson.id))
            results.append(vr)

        logger.info(
            "verify_all_pending_completed",
            project_id=project_id,
            total=len(results),
            verified=sum(1 for r in results if r.status == VerificationStatus.VERIFIED),
            invalidated=sum(1 for r in results if r.status == VerificationStatus.INVALIDATED),
            skipped=sum(1 for r in results if r.status == VerificationStatus.SKIPPED),
        )

        return results

    # ------------------------------------------------------------------
    # Status update (P4-037)
    # ------------------------------------------------------------------

    async def update_verification_status(
        self,
        lesson_id: str,
        status: VerificationStatus,
        confidence_score: float | None = None,
    ) -> None:
        """Update a lesson's ``verification_status`` in the database.

        Args:
            lesson_id: UUID string of the lesson to update.
            status: New verification status.
            confidence_score: Optional updated confidence (0.0 to 1.0).

        Raises:
            ValueError: If the lesson does not exist.
        """
        session = self.session_factory()
        try:
            lesson = await self._get_lesson(session, lesson_id)
            if lesson is None:
                raise ValueError(f"Lesson {lesson_id} not found")

            values: dict[str, object] = {"verification_status": status.value}
            if confidence_score is not None:
                values["confidence_score"] = confidence_score

            stmt = (
                update(LessonLearned)
                .where(LessonLearned.id == uuid.UUID(lesson_id))
                .values(**values)
            )

            async with session.begin():
                await session.execute(stmt)

            logger.info(
                "verification_status_updated",
                lesson_id=lesson_id,
                status=status.value,
                confidence_score=confidence_score,
            )
        finally:
            await session.close()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _calculate_pass_rate(self, results: list[TestExecutionResult]) -> float:
        """Calculate the fraction of tests that passed.

        Args:
            results: List of test execution results.

        Returns:
            A float between 0.0 and 1.0 representing the pass rate.
            Returns 0.0 when the list is empty.
        """
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.passed)
        return passed / len(results)

    def _determine_verification_status(
        self,
        pre_results: list[TestExecutionResult],
        post_results: list[TestExecutionResult],
    ) -> VerificationStatus:
        """Determine the verification status from test results.

        Decision matrix:
        - Both runs pass (post-fix pass rate >= 0.8): **VERIFIED**
        - Post-fix passes but pre-fix fails: **VERIFIED** (fix improved
          the situation)
        - Both runs fail: **INVALIDATED** (lesson may be stale)
        - Post-fix fails: **INVALIDATED**
        - No results: **SKIPPED**

        Args:
            pre_results: Baseline test results.
            post_results: Post-fix test results.

        Returns:
            The determined ``VerificationStatus``.
        """
        if not post_results:
            return VerificationStatus.SKIPPED

        post_pass_rate = self._calculate_pass_rate(post_results)
        pre_pass_rate = self._calculate_pass_rate(pre_results)

        if post_pass_rate >= 0.8:
            # Tests are passing after fix -- lesson is valid
            return VerificationStatus.VERIFIED

        if pre_pass_rate < 0.5 and post_pass_rate < 0.5:
            # Both pre and post are failing -- lesson is likely stale
            return VerificationStatus.INVALIDATED

        if post_pass_rate < 0.5:
            # Post-fix tests are failing -- lesson not holding up
            return VerificationStatus.INVALIDATED

        # Marginal case: some tests pass, some fail
        # If post is better than pre, still treat as verified
        if post_pass_rate > pre_pass_rate:
            return VerificationStatus.VERIFIED

        return VerificationStatus.INVALIDATED

    def _compute_confidence(
        self,
        discovery_confidence: float,
        post_fix_pass_rate: float,
    ) -> float:
        """Compute the updated confidence score for a lesson.

        Combines the test discovery confidence with the post-fix pass rate
        using a weighted average.

        Args:
            discovery_confidence: Confidence from test discovery (0.0-1.0).
            post_fix_pass_rate: Post-fix test pass rate (0.0-1.0).

        Returns:
            A combined confidence score between 0.0 and 1.0.
        """
        # Weight pass-rate more heavily since it's direct evidence
        combined = (discovery_confidence * 0.3) + (post_fix_pass_rate * 0.7)
        return round(min(max(combined, 0.0), 1.0), 4)

    async def _get_lesson(
        self,
        session: AsyncSession,
        lesson_id: str,
    ) -> LessonLearned | None:
        """Retrieve a lesson by its UUID string.

        Args:
            session: Active async database session.
            lesson_id: UUID string of the lesson.

        Returns:
            The ``LessonLearned`` instance or ``None`` if not found.
        """
        stmt = select(LessonLearned).where(
            LessonLearned.id == uuid.UUID(lesson_id),
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
