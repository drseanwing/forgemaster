"""Integration tests for merge coordinator service.

This module tests the MergeCoordinator's queue management, automatic merging,
conflict detection, architect escalation, retry logic, and resource cleanup.
All git dependencies are mocked to isolate the coordinator logic from real
git operations (which are tested separately in test_git_ops.py).

Tests cover:
- Successful merge with no conflicts (P3-018)
- Merge with conflict detection (P3-018)
- FIFO queue processing order (P3-017)
- Conflict escalation to architect agent (P3-019)
- Retry logic under and over max_retries (P3-018)
- Queue status reporting (P3-016)
- Resolution application and re-merge (P3-019)
- File lock release on successful merge (P3-018)
- Worktree release on successful merge (P3-018)
- Concurrent queue processing with asyncio lock (P3-017)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forgemaster.orchestrator.merge_coordinator import (
    MergeCoordinator,
    MergeRequest,
    MergeStatus,
)
from forgemaster.pipeline.git_ops import MergeResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_git_manager() -> MagicMock:
    """Create a mock GitManager.

    Returns:
        MagicMock configured to behave like GitManager.
    """
    manager = MagicMock()
    manager.detect_conflicts.return_value = []
    manager.merge.return_value = MergeResult(
        success=True,
        commit_sha="abc123def456",
        conflicts=[],
    )
    return manager


@pytest.fixture
def mock_worktree_pool() -> MagicMock:
    """Create a mock WorktreePool.

    Returns:
        MagicMock configured to behave like WorktreePool with async release.
    """
    pool = MagicMock()
    pool.release = AsyncMock()
    return pool


@pytest.fixture
def mock_file_locker() -> MagicMock:
    """Create a mock FileLocker.

    Returns:
        MagicMock configured to behave like FileLocker with async release_locks.
    """
    locker = MagicMock()
    locker.release_locks = AsyncMock(return_value=3)
    return locker


@pytest.fixture
def mock_session_manager() -> MagicMock:
    """Create a mock AgentSessionManager.

    Returns:
        MagicMock configured to behave like AgentSessionManager with async methods.
    """
    manager = MagicMock()
    manager.start_session = AsyncMock(return_value="session-123")
    manager.send_message = AsyncMock(
        return_value="Resolution: Prioritise target branch changes for README.md."
    )
    manager.end_session = AsyncMock()
    return manager


@pytest.fixture
def mock_context_generator() -> MagicMock:
    """Create a mock ContextGenerator.

    Returns:
        MagicMock configured to behave like ContextGenerator.
    """
    generator = MagicMock()
    generator.generate_agent_context.return_value = "System prompt for architect."
    return generator


@pytest.fixture
def coordinator(
    mock_git_manager: MagicMock,
    mock_worktree_pool: MagicMock,
    mock_file_locker: MagicMock,
) -> MergeCoordinator:
    """Create a MergeCoordinator with mocked dependencies (no session manager).

    Args:
        mock_git_manager: Mocked GitManager.
        mock_worktree_pool: Mocked WorktreePool.
        mock_file_locker: Mocked FileLocker.

    Returns:
        MergeCoordinator instance without escalation support.
    """
    return MergeCoordinator(
        git_manager=mock_git_manager,
        worktree_pool=mock_worktree_pool,
        file_locker=mock_file_locker,
        target_branch="main",
        max_retries=3,
    )


@pytest.fixture
def coordinator_with_escalation(
    mock_git_manager: MagicMock,
    mock_worktree_pool: MagicMock,
    mock_file_locker: MagicMock,
    mock_session_manager: MagicMock,
    mock_context_generator: MagicMock,
) -> MergeCoordinator:
    """Create a MergeCoordinator with session manager for escalation tests.

    Args:
        mock_git_manager: Mocked GitManager.
        mock_worktree_pool: Mocked WorktreePool.
        mock_file_locker: Mocked FileLocker.
        mock_session_manager: Mocked AgentSessionManager.
        mock_context_generator: Mocked ContextGenerator.

    Returns:
        MergeCoordinator instance with escalation support.
    """
    return MergeCoordinator(
        git_manager=mock_git_manager,
        worktree_pool=mock_worktree_pool,
        file_locker=mock_file_locker,
        session_manager=mock_session_manager,
        context_generator=mock_context_generator,
        target_branch="main",
        max_retries=3,
    )


# ---------------------------------------------------------------------------
# P3-016: Merge Coordinator Service
# ---------------------------------------------------------------------------


class TestMergeCoordinatorInit:
    """Tests for MergeCoordinator initialization and configuration."""

    def test_init_defaults(
        self,
        mock_git_manager: MagicMock,
        mock_worktree_pool: MagicMock,
        mock_file_locker: MagicMock,
    ) -> None:
        """Test coordinator initializes with correct defaults."""
        coord = MergeCoordinator(
            git_manager=mock_git_manager,
            worktree_pool=mock_worktree_pool,
            file_locker=mock_file_locker,
        )

        assert coord.target_branch == "main"
        assert coord.max_retries == 3
        assert coord.session_manager is None
        assert coord.context_generator is None

    def test_init_custom_settings(
        self,
        mock_git_manager: MagicMock,
        mock_worktree_pool: MagicMock,
        mock_file_locker: MagicMock,
    ) -> None:
        """Test coordinator initializes with custom settings."""
        coord = MergeCoordinator(
            git_manager=mock_git_manager,
            worktree_pool=mock_worktree_pool,
            file_locker=mock_file_locker,
            target_branch="develop",
            max_retries=5,
        )

        assert coord.target_branch == "develop"
        assert coord.max_retries == 5

    def test_queue_status_empty(self, coordinator: MergeCoordinator) -> None:
        """Test queue status when queue is empty."""
        status = coordinator.get_queue_status()

        assert status["total"] == 0
        assert status["queued"] == 0
        assert status["merged"] == 0
        assert status["conflict"] == 0
        assert status["failed"] == 0

    def test_get_pending_merges_empty(self, coordinator: MergeCoordinator) -> None:
        """Test get_pending_merges when queue is empty."""
        pending = coordinator.get_pending_merges()
        assert pending == []

    def test_get_conflicts_empty(self, coordinator: MergeCoordinator) -> None:
        """Test get_conflicts when queue is empty."""
        conflicts = coordinator.get_conflicts()
        assert conflicts == []


# ---------------------------------------------------------------------------
# P3-017: Merge Queue Logic
# ---------------------------------------------------------------------------


class TestMergeQueue:
    """Tests for merge queue management and FIFO processing."""

    @pytest.mark.asyncio
    async def test_enqueue_merge_creates_request(
        self, coordinator: MergeCoordinator
    ) -> None:
        """Test that enqueue_merge creates and returns a MergeRequest."""
        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["src/main.py", "src/utils.py"],
        )

        assert isinstance(request, MergeRequest)
        assert request.task_id == "task-1"
        assert request.worker_id == "worker-1"
        assert request.worktree_name == "wt-1"
        assert request.source_branch == "worktree/wt-1"
        assert request.target_branch == "main"
        assert request.files_modified == ["src/main.py", "src/utils.py"]
        assert request.status == MergeStatus.QUEUED
        assert request.merged_at is None
        assert request.conflict_files == []
        assert request.retry_count == 0
        assert request.max_retries == 3

    @pytest.mark.asyncio
    async def test_enqueue_multiple_requests(
        self, coordinator: MergeCoordinator
    ) -> None:
        """Test that multiple requests are added to the queue."""
        for i in range(3):
            await coordinator.enqueue_merge(
                task_id=f"task-{i}",
                worker_id=f"worker-{i}",
                worktree_name=f"wt-{i}",
                source_branch=f"worktree/wt-{i}",
                files_modified=[f"src/file{i}.py"],
            )

        status = coordinator.get_queue_status()
        assert status["total"] == 3
        assert status["queued"] == 3

    @pytest.mark.asyncio
    async def test_get_pending_merges(self, coordinator: MergeCoordinator) -> None:
        """Test getting pending merge requests."""
        await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["src/a.py"],
        )
        await coordinator.enqueue_merge(
            task_id="task-2",
            worker_id="worker-2",
            worktree_name="wt-2",
            source_branch="worktree/wt-2",
            files_modified=["src/b.py"],
        )

        pending = coordinator.get_pending_merges()

        assert len(pending) == 2
        assert all(r.status == MergeStatus.QUEUED for r in pending)

    @pytest.mark.asyncio
    async def test_process_queue_fifo_order(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test that queue is processed in FIFO order."""
        merge_order: list[str] = []

        original_merge = mock_git_manager.merge

        def tracking_merge(source: str, target: str | None = None) -> MergeResult:
            merge_order.append(source)
            return MergeResult(success=True, commit_sha="sha123", conflicts=[])

        mock_git_manager.merge.side_effect = tracking_merge

        # Enqueue in specific order
        for i in range(3):
            await coordinator.enqueue_merge(
                task_id=f"task-{i}",
                worker_id=f"worker-{i}",
                worktree_name=f"wt-{i}",
                source_branch=f"branch-{i}",
                files_modified=[f"file{i}.py"],
            )

        # Process queue
        processed = await coordinator.process_queue()

        assert len(processed) == 3
        assert merge_order == ["branch-0", "branch-1", "branch-2"]

    @pytest.mark.asyncio
    async def test_process_queue_skips_non_queued(
        self,
        coordinator: MergeCoordinator,
    ) -> None:
        """Test that process_queue only processes QUEUED requests."""
        # Enqueue and process one
        req1 = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["a.py"],
        )

        # Process it
        await coordinator.process_queue()
        assert req1.status == MergeStatus.MERGED

        # Enqueue another
        await coordinator.enqueue_merge(
            task_id="task-2",
            worker_id="worker-2",
            worktree_name="wt-2",
            source_branch="branch-2",
            files_modified=["b.py"],
        )

        # Process again - should only process task-2
        processed = await coordinator.process_queue()
        assert len(processed) == 1
        assert processed[0].task_id == "task-2"

    @pytest.mark.asyncio
    async def test_concurrent_queue_processing_serialised(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test that concurrent process_queue calls are serialised by the lock."""
        call_count = 0

        async def slow_merge(source: str, target: str | None = None) -> MergeResult:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return MergeResult(success=True, commit_sha="sha123", conflicts=[])

        # Make merge async-compatible by using the sync version
        mock_git_manager.merge.side_effect = lambda s, t=None: MergeResult(
            success=True, commit_sha="sha123", conflicts=[]
        )

        # Enqueue requests
        for i in range(2):
            await coordinator.enqueue_merge(
                task_id=f"task-{i}",
                worker_id=f"worker-{i}",
                worktree_name=f"wt-{i}",
                source_branch=f"branch-{i}",
                files_modified=[f"file{i}.py"],
            )

        # Run two process_queue calls concurrently
        results = await asyncio.gather(
            coordinator.process_queue(),
            coordinator.process_queue(),
        )

        # One should process the requests, the other should find none queued
        total_processed = sum(len(r) for r in results)
        assert total_processed == 2


# ---------------------------------------------------------------------------
# P3-018: Automatic Merge Attempt
# ---------------------------------------------------------------------------


class TestAutomaticMerge:
    """Tests for automatic merge attempt logic."""

    @pytest.mark.asyncio
    async def test_successful_merge_no_conflicts(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
        mock_file_locker: MagicMock,
        mock_worktree_pool: MagicMock,
    ) -> None:
        """Test successful merge when no conflicts are detected."""
        mock_git_manager.detect_conflicts.return_value = []
        mock_git_manager.merge.return_value = MergeResult(
            success=True, commit_sha="abc123", conflicts=[]
        )

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["src/main.py"],
        )

        result = await coordinator.attempt_merge(request)

        assert result.status == MergeStatus.MERGED
        assert result.merged_at is not None
        assert result.conflict_files == []

        # Verify git operations called
        mock_git_manager.detect_conflicts.assert_called_once_with(
            "worktree/wt-1", "main"
        )
        mock_git_manager.merge.assert_called_once_with("worktree/wt-1", "main")

    @pytest.mark.asyncio
    async def test_merge_with_pre_check_conflicts(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test merge aborted when pre-check detects conflicts."""
        mock_git_manager.detect_conflicts.return_value = ["src/main.py", "src/utils.py"]

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["src/main.py"],
        )

        result = await coordinator.attempt_merge(request)

        assert result.status == MergeStatus.CONFLICT
        assert result.conflict_files == ["src/main.py", "src/utils.py"]
        assert result.merged_at is None

        # Merge should not have been called
        mock_git_manager.merge.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_with_result_conflicts(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test merge detects conflicts from merge result (not pre-check)."""
        mock_git_manager.detect_conflicts.return_value = []
        mock_git_manager.merge.return_value = MergeResult(
            success=False, commit_sha=None, conflicts=["README.md"]
        )

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["README.md"],
        )

        result = await coordinator.attempt_merge(request)

        assert result.status == MergeStatus.CONFLICT
        assert result.conflict_files == ["README.md"]

    @pytest.mark.asyncio
    async def test_merge_releases_file_locks(
        self,
        coordinator: MergeCoordinator,
        mock_file_locker: MagicMock,
    ) -> None:
        """Test that file locks are released after successful merge."""
        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["src/main.py"],
        )

        await coordinator.attempt_merge(request)

        mock_file_locker.release_locks.assert_called_once_with("task-1")

    @pytest.mark.asyncio
    async def test_merge_releases_worktree(
        self,
        coordinator: MergeCoordinator,
        mock_worktree_pool: MagicMock,
    ) -> None:
        """Test that worktree is released after successful merge."""
        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["src/main.py"],
        )

        await coordinator.attempt_merge(request)

        mock_worktree_pool.release.assert_called_once_with("wt-1")

    @pytest.mark.asyncio
    async def test_conflict_does_not_release_resources(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
        mock_file_locker: MagicMock,
        mock_worktree_pool: MagicMock,
    ) -> None:
        """Test that resources are NOT released when conflicts are detected."""
        mock_git_manager.detect_conflicts.return_value = ["conflict.py"]

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["conflict.py"],
        )

        await coordinator.attempt_merge(request)

        mock_file_locker.release_locks.assert_not_called()
        mock_worktree_pool.release.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_conflict_detection_exception(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test merge is marked FAILED when conflict detection raises."""
        mock_git_manager.detect_conflicts.side_effect = RuntimeError("git error")

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["src/main.py"],
        )

        result = await coordinator.attempt_merge(request)

        assert result.status == MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_merge_operation_exception(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test merge is marked FAILED when merge operation raises."""
        mock_git_manager.detect_conflicts.return_value = []
        mock_git_manager.merge.side_effect = RuntimeError("merge failed")

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="worktree/wt-1",
            files_modified=["src/main.py"],
        )

        result = await coordinator.attempt_merge(request)

        assert result.status == MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_process_queue_merges_all_conflict_free(
        self,
        coordinator: MergeCoordinator,
    ) -> None:
        """Test that process_queue merges all conflict-free requests."""
        for i in range(3):
            await coordinator.enqueue_merge(
                task_id=f"task-{i}",
                worker_id=f"worker-{i}",
                worktree_name=f"wt-{i}",
                source_branch=f"branch-{i}",
                files_modified=[f"file{i}.py"],
            )

        processed = await coordinator.process_queue()

        assert len(processed) == 3
        assert all(r.status == MergeStatus.MERGED for r in processed)


# ---------------------------------------------------------------------------
# P3-018: Retry Logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for merge retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_under_max_retries(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test retry succeeds when under max retries."""
        # First attempt: conflict
        mock_git_manager.detect_conflicts.return_value = ["conflict.py"]

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["conflict.py"],
        )
        await coordinator.attempt_merge(request)
        assert request.status == MergeStatus.CONFLICT

        # Fix the conflict (no more conflicts)
        mock_git_manager.detect_conflicts.return_value = []
        mock_git_manager.merge.return_value = MergeResult(
            success=True, commit_sha="sha123", conflicts=[]
        )

        # Retry
        result = await coordinator.retry_merge(request)

        assert result.status == MergeStatus.MERGED
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_increments_count(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test that retry_count is incremented on each retry."""
        mock_git_manager.detect_conflicts.return_value = ["conflict.py"]

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["conflict.py"],
        )

        # Multiple retries, all still conflicting
        for i in range(3):
            request.status = MergeStatus.CONFLICT
            request.conflict_files = []
            try:
                await coordinator.retry_merge(request)
            except ValueError:
                pass

        assert request.retry_count == 3

    @pytest.mark.asyncio
    async def test_retry_over_max_retries_raises(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test that exceeding max_retries raises ValueError."""
        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["file.py"],
        )

        # Exhaust retries
        request.retry_count = 3  # Already at max

        with pytest.raises(ValueError, match="exceeded max retries"):
            await coordinator.retry_merge(request)

        assert request.status == MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_resets_conflict_files(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test that retry clears previous conflict_files before re-attempting."""
        mock_git_manager.detect_conflicts.return_value = []
        mock_git_manager.merge.return_value = MergeResult(
            success=True, commit_sha="sha123", conflicts=[]
        )

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["file.py"],
        )

        request.status = MergeStatus.CONFLICT
        request.conflict_files = ["old_conflict.py"]

        result = await coordinator.retry_merge(request)

        assert result.conflict_files == []
        assert result.status == MergeStatus.MERGED


# ---------------------------------------------------------------------------
# P3-019: Conflict Escalation to Architect
# ---------------------------------------------------------------------------


class TestConflictEscalation:
    """Tests for architect escalation flow."""

    @pytest.mark.asyncio
    async def test_escalate_creates_architect_session(
        self,
        coordinator_with_escalation: MergeCoordinator,
        mock_session_manager: MagicMock,
        mock_context_generator: MagicMock,
    ) -> None:
        """Test that escalation creates an architect session."""
        request = await coordinator_with_escalation.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py"],
        )
        request.status = MergeStatus.CONFLICT
        request.conflict_files = ["src/main.py"]

        await coordinator_with_escalation.escalate_to_architect(request)

        mock_session_manager.start_session.assert_called_once()
        call_kwargs = mock_session_manager.start_session.call_args
        assert call_kwargs.kwargs["task_id"] == "task-1"
        assert call_kwargs.kwargs["agent_type"] == "architect"
        assert call_kwargs.kwargs["model"] == "opus"

    @pytest.mark.asyncio
    async def test_escalate_sends_conflict_message(
        self,
        coordinator_with_escalation: MergeCoordinator,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that escalation sends a descriptive conflict message."""
        request = await coordinator_with_escalation.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py", "src/utils.py"],
        )
        request.status = MergeStatus.CONFLICT
        request.conflict_files = ["src/main.py"]

        await coordinator_with_escalation.escalate_to_architect(request)

        mock_session_manager.send_message.assert_called_once()
        message = mock_session_manager.send_message.call_args[0][1]

        assert "src/main.py" in message
        assert "branch-1" in message
        assert "task-1" in message
        assert "Merge Conflict" in message

    @pytest.mark.asyncio
    async def test_escalate_returns_resolution(
        self,
        coordinator_with_escalation: MergeCoordinator,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that escalation returns the architect's resolution text."""
        request = await coordinator_with_escalation.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py"],
        )
        request.status = MergeStatus.CONFLICT
        request.conflict_files = ["src/main.py"]

        resolution = await coordinator_with_escalation.escalate_to_architect(request)

        assert "Resolution" in resolution
        assert request.status == MergeStatus.ESCALATED
        assert request.resolution_notes == resolution

    @pytest.mark.asyncio
    async def test_escalate_ends_session(
        self,
        coordinator_with_escalation: MergeCoordinator,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that architect session is always closed after escalation."""
        request = await coordinator_with_escalation.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py"],
        )
        request.status = MergeStatus.CONFLICT
        request.conflict_files = ["src/main.py"]

        await coordinator_with_escalation.escalate_to_architect(request)

        mock_session_manager.end_session.assert_called_once_with(
            "session-123", status="completed"
        )

    @pytest.mark.asyncio
    async def test_escalate_ends_session_on_error(
        self,
        coordinator_with_escalation: MergeCoordinator,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that architect session is closed even when send_message fails."""
        mock_session_manager.send_message.side_effect = RuntimeError("agent error")

        request = await coordinator_with_escalation.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py"],
        )
        request.status = MergeStatus.CONFLICT
        request.conflict_files = ["src/main.py"]

        with pytest.raises(RuntimeError, match="agent error"):
            await coordinator_with_escalation.escalate_to_architect(request)

        # Session should still be ended
        mock_session_manager.end_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_without_session_manager_raises(
        self,
        coordinator: MergeCoordinator,
    ) -> None:
        """Test that escalation raises when session_manager is not configured."""
        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py"],
        )
        request.status = MergeStatus.CONFLICT

        with pytest.raises(RuntimeError, match="session_manager is not configured"):
            await coordinator.escalate_to_architect(request)

    @pytest.mark.asyncio
    async def test_escalate_without_context_generator_raises(
        self,
        mock_git_manager: MagicMock,
        mock_worktree_pool: MagicMock,
        mock_file_locker: MagicMock,
        mock_session_manager: MagicMock,
    ) -> None:
        """Test that escalation raises when context_generator is not configured."""
        coord = MergeCoordinator(
            git_manager=mock_git_manager,
            worktree_pool=mock_worktree_pool,
            file_locker=mock_file_locker,
            session_manager=mock_session_manager,
            # No context_generator
        )

        request = await coord.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py"],
        )
        request.status = MergeStatus.CONFLICT

        with pytest.raises(RuntimeError, match="context_generator is not configured"):
            await coord.escalate_to_architect(request)


# ---------------------------------------------------------------------------
# P3-019: Resolution Application
# ---------------------------------------------------------------------------


class TestResolutionApplication:
    """Tests for applying architect resolutions and retrying merges."""

    @pytest.mark.asyncio
    async def test_apply_resolution_sets_notes(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test that apply_resolution records the resolution text."""
        mock_git_manager.detect_conflicts.return_value = []
        mock_git_manager.merge.return_value = MergeResult(
            success=True, commit_sha="sha123", conflicts=[]
        )

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py"],
        )
        request.status = MergeStatus.ESCALATED

        result = await coordinator.apply_resolution(
            request, "Use target branch version for README.md."
        )

        assert result.resolution_notes == "Use target branch version for README.md."

    @pytest.mark.asyncio
    async def test_apply_resolution_retries_merge(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test that apply_resolution retries the merge after recording resolution."""
        mock_git_manager.detect_conflicts.return_value = []
        mock_git_manager.merge.return_value = MergeResult(
            success=True, commit_sha="sha456", conflicts=[]
        )

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["src/main.py"],
        )
        request.status = MergeStatus.ESCALATED

        result = await coordinator.apply_resolution(request, "Resolution text.")

        assert result.status == MergeStatus.MERGED
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_apply_resolution_still_conflicts(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test apply_resolution when conflicts remain after retry."""
        mock_git_manager.detect_conflicts.return_value = ["still_conflict.py"]

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["still_conflict.py"],
        )
        request.status = MergeStatus.ESCALATED

        result = await coordinator.apply_resolution(request, "Try again.")

        assert result.status == MergeStatus.CONFLICT
        assert result.retry_count == 1


# ---------------------------------------------------------------------------
# P3-016: Status Reporting
# ---------------------------------------------------------------------------


class TestStatusReporting:
    """Tests for queue status and monitoring methods."""

    @pytest.mark.asyncio
    async def test_queue_status_counts(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test queue status reflects current request states."""
        # Enqueue 3 requests
        for i in range(3):
            await coordinator.enqueue_merge(
                task_id=f"task-{i}",
                worker_id=f"worker-{i}",
                worktree_name=f"wt-{i}",
                source_branch=f"branch-{i}",
                files_modified=[f"file{i}.py"],
            )

        # Process to merge all
        await coordinator.process_queue()

        status = coordinator.get_queue_status()

        assert status["total"] == 3
        assert status["merged"] == 3
        assert status["queued"] == 0

    @pytest.mark.asyncio
    async def test_queue_status_mixed_states(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test queue status with a mix of merged and conflicted requests."""
        # First request: conflict-free
        mock_git_manager.detect_conflicts.return_value = []
        await coordinator.enqueue_merge(
            task_id="task-ok",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-ok",
            files_modified=["ok.py"],
        )

        # Process first (should merge)
        await coordinator.process_queue()

        # Second request: will have conflicts
        mock_git_manager.detect_conflicts.return_value = ["bad.py"]
        await coordinator.enqueue_merge(
            task_id="task-conflict",
            worker_id="worker-2",
            worktree_name="wt-2",
            source_branch="branch-conflict",
            files_modified=["bad.py"],
        )

        # Process second (should conflict)
        await coordinator.process_queue()

        status = coordinator.get_queue_status()
        assert status["merged"] == 1
        assert status["conflict"] == 1

    @pytest.mark.asyncio
    async def test_get_conflicts_returns_conflicted_requests(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test get_conflicts returns only conflicted and escalated requests."""
        mock_git_manager.detect_conflicts.return_value = ["conflict.py"]

        req = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["conflict.py"],
        )

        await coordinator.attempt_merge(req)

        conflicts = coordinator.get_conflicts()

        assert len(conflicts) == 1
        assert conflicts[0].task_id == "task-1"
        assert conflicts[0].status == MergeStatus.CONFLICT

    @pytest.mark.asyncio
    async def test_get_conflicts_includes_escalated(
        self,
        coordinator_with_escalation: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test get_conflicts also returns ESCALATED requests."""
        mock_git_manager.detect_conflicts.return_value = ["conflict.py"]

        req = await coordinator_with_escalation.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["conflict.py"],
        )

        await coordinator_with_escalation.attempt_merge(req)
        await coordinator_with_escalation.escalate_to_architect(req)

        conflicts = coordinator_with_escalation.get_conflicts()

        assert len(conflicts) == 1
        assert conflicts[0].status == MergeStatus.ESCALATED


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error resilience."""

    @pytest.mark.asyncio
    async def test_lock_release_failure_does_not_fail_merge(
        self,
        coordinator: MergeCoordinator,
        mock_file_locker: MagicMock,
    ) -> None:
        """Test that lock release failure does not revert merge status."""
        mock_file_locker.release_locks.side_effect = RuntimeError("db error")

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["file.py"],
        )

        result = await coordinator.attempt_merge(request)

        # Merge should still be marked as successful
        assert result.status == MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_worktree_release_failure_does_not_fail_merge(
        self,
        coordinator: MergeCoordinator,
        mock_worktree_pool: MagicMock,
    ) -> None:
        """Test that worktree release failure does not revert merge status."""
        mock_worktree_pool.release.side_effect = RuntimeError("pool error")

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["file.py"],
        )

        result = await coordinator.attempt_merge(request)

        # Merge should still be marked as successful
        assert result.status == MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_merge_request_timestamps(
        self, coordinator: MergeCoordinator
    ) -> None:
        """Test that MergeRequest has proper timestamps."""
        before = datetime.now(timezone.utc)

        request = await coordinator.enqueue_merge(
            task_id="task-1",
            worker_id="worker-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            files_modified=["file.py"],
        )

        after = datetime.now(timezone.utc)

        assert before <= request.created_at <= after
        assert request.merged_at is None

        # After merge
        await coordinator.attempt_merge(request)

        assert request.merged_at is not None
        assert request.merged_at >= request.created_at

    @pytest.mark.asyncio
    async def test_process_queue_handles_exception_in_merge(
        self,
        coordinator: MergeCoordinator,
        mock_git_manager: MagicMock,
    ) -> None:
        """Test that process_queue handles exceptions from individual merges."""
        call_count = 0

        def failing_detect(source: str, target: str) -> list[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Unexpected git error")
            return []

        mock_git_manager.detect_conflicts.side_effect = failing_detect

        # Enqueue 3 requests
        for i in range(3):
            await coordinator.enqueue_merge(
                task_id=f"task-{i}",
                worker_id=f"worker-{i}",
                worktree_name=f"wt-{i}",
                source_branch=f"branch-{i}",
                files_modified=[f"file{i}.py"],
            )

        # Process queue - should not raise even though task-1 fails
        processed = await coordinator.process_queue()

        assert len(processed) == 3

        # task-0: merged, task-1: failed (exception), task-2: merged
        assert processed[0].status == MergeStatus.MERGED
        assert processed[1].status == MergeStatus.FAILED
        assert processed[2].status == MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_merge_request_model_defaults(self) -> None:
        """Test MergeRequest pydantic model defaults."""
        request = MergeRequest(
            task_id="t-1",
            worker_id="w-1",
            worktree_name="wt-1",
            source_branch="branch-1",
            target_branch="main",
            files_modified=["a.py"],
        )

        assert request.status == MergeStatus.QUEUED
        assert request.merged_at is None
        assert request.conflict_files == []
        assert request.resolution_notes is None
        assert request.retry_count == 0
        assert request.max_retries == 3
        assert request.created_at is not None

    def test_merge_status_values(self) -> None:
        """Test MergeStatus enum values."""
        assert MergeStatus.QUEUED == "queued"
        assert MergeStatus.MERGING == "merging"
        assert MergeStatus.MERGED == "merged"
        assert MergeStatus.CONFLICT == "conflict"
        assert MergeStatus.ESCALATED == "escalated"
        assert MergeStatus.RESOLVED == "resolved"
        assert MergeStatus.FAILED == "failed"
