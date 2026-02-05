"""Integration tests for git worktree management.

This module contains integration tests for the WorktreeManager and WorktreePool
classes, testing actual git operations in temporary directories.

Tests cover:
- Worktree creation and cleanup
- Pool lifecycle (acquire, release, reset)
- Task assignment and lookup
- Pool capacity limits (semaphore)
- Error handling
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import git
import pytest

from forgemaster.config import GitConfig
from forgemaster.pipeline.git_ops import GitManager
from forgemaster.pipeline.worktree import (
    WorktreeInfo,
    WorktreeManager,
    WorktreePool,
    WorktreeStatus,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> git.Repo:
    """Create a temporary git repository for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Initialized git.Repo with an initial commit
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    repo = git.Repo.init(repo_path)

    # Create initial commit so repo is valid
    readme = repo_path / "README.md"
    readme.write_text("# Test Repository\n")
    repo.index.add(["README.md"])
    repo.index.commit("initial commit")

    # Rename default branch from master to main
    if "master" in repo.heads:
        repo.heads.master.rename("main")

    return repo


@pytest.fixture
def git_config(tmp_path: Path) -> GitConfig:
    """Create GitConfig for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        GitConfig with worktree_base_path in tmp_path
    """
    worktree_base = tmp_path / "worktrees"
    return GitConfig(
        main_branch="main",
        auto_push=False,
        worktree_base_path=worktree_base,
    )


@pytest.fixture
def git_manager(git_repo: git.Repo, git_config: GitConfig) -> GitManager:
    """Create GitManager for testing.

    Args:
        git_repo: Temporary git repository
        git_config: Git configuration

    Returns:
        GitManager instance
    """
    return GitManager(repo_path=Path(git_repo.working_dir), config=git_config)


@pytest.fixture
def worktree_manager(
    git_manager: GitManager, git_config: GitConfig
) -> WorktreeManager:
    """Create WorktreeManager for testing.

    Args:
        git_manager: GitManager instance
        git_config: Git configuration

    Returns:
        WorktreeManager instance
    """
    return WorktreeManager(git_manager, base_path=git_config.worktree_base_path)


@pytest.fixture
def worktree_pool(worktree_manager: WorktreeManager) -> WorktreePool:
    """Create WorktreePool for testing.

    Args:
        worktree_manager: WorktreeManager instance

    Returns:
        WorktreePool with max_workers=3
    """
    return WorktreePool(worktree_manager, max_workers=3)


class TestWorktreeManager:
    """Test suite for WorktreeManager."""

    def test_create_worktree(
        self, worktree_manager: WorktreeManager, git_config: GitConfig
    ) -> None:
        """Test creating a new worktree."""
        info = worktree_manager.create_worktree("test-worker")

        assert info.name == "test-worker"
        assert info.path == git_config.worktree_base_path / "worker-test-worker"
        assert info.branch == "worktree/test-worker"
        assert info.status == WorktreeStatus.IDLE
        assert info.task_id is None
        assert info.path.exists()

        # Check it's registered
        assert "test-worker" in worktree_manager.worktrees

    def test_create_worktree_with_custom_branch(
        self, worktree_manager: WorktreeManager
    ) -> None:
        """Test creating a worktree with a custom branch name."""
        info = worktree_manager.create_worktree("test-worker", branch="feature/custom")

        assert info.branch == "feature/custom"
        assert info.name == "test-worker"

    def test_create_duplicate_worktree_fails(
        self, worktree_manager: WorktreeManager
    ) -> None:
        """Test that creating a duplicate worktree raises ValueError."""
        worktree_manager.create_worktree("test-worker")

        with pytest.raises(ValueError, match="already exists"):
            worktree_manager.create_worktree("test-worker")

    def test_cleanup_worktree(self, worktree_manager: WorktreeManager) -> None:
        """Test cleaning up a worktree."""
        info = worktree_manager.create_worktree("test-worker")
        path = info.path

        assert path.exists()
        assert "test-worker" in worktree_manager.worktrees

        # Cleanup
        result = worktree_manager.cleanup_worktree("test-worker")

        assert result is True
        assert not path.exists()
        assert "test-worker" not in worktree_manager.worktrees

    def test_cleanup_nonexistent_worktree(
        self, worktree_manager: WorktreeManager
    ) -> None:
        """Test cleaning up a nonexistent worktree returns False."""
        result = worktree_manager.cleanup_worktree("nonexistent")
        assert result is False

    def test_cleanup_all(self, worktree_manager: WorktreeManager) -> None:
        """Test cleaning up all worktrees."""
        # Create multiple worktrees
        worktree_manager.create_worktree("worker-1")
        worktree_manager.create_worktree("worker-2")
        worktree_manager.create_worktree("worker-3")

        assert len(worktree_manager.worktrees) == 3

        # Cleanup all
        count = worktree_manager.cleanup_all()

        assert count == 3
        assert len(worktree_manager.worktrees) == 0

    def test_assign_worktree(self, worktree_manager: WorktreeManager) -> None:
        """Test assigning a task to a worktree."""
        info = worktree_manager.create_worktree("test-worker")

        worktree_manager.assign_worktree("test-worker", "TASK-123")

        assert info.task_id == "TASK-123"
        assert info.status == WorktreeStatus.ACTIVE
        assert worktree_manager._task_mapping["TASK-123"] == "test-worker"
        assert worktree_manager._worktree_tasks["test-worker"] == "TASK-123"

    def test_assign_worktree_already_assigned_fails(
        self, worktree_manager: WorktreeManager
    ) -> None:
        """Test that assigning an already-assigned worktree raises ValueError."""
        worktree_manager.create_worktree("test-worker")
        worktree_manager.assign_worktree("test-worker", "TASK-123")

        with pytest.raises(ValueError, match="already assigned"):
            worktree_manager.assign_worktree("test-worker", "TASK-456")

    def test_assign_nonexistent_worktree_fails(
        self, worktree_manager: WorktreeManager
    ) -> None:
        """Test that assigning a nonexistent worktree raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            worktree_manager.assign_worktree("nonexistent", "TASK-123")

    def test_unassign_worktree(self, worktree_manager: WorktreeManager) -> None:
        """Test unassigning a task from a worktree."""
        info = worktree_manager.create_worktree("test-worker")
        worktree_manager.assign_worktree("test-worker", "TASK-123")

        worktree_manager.unassign_worktree("test-worker")

        assert info.task_id is None
        assert info.status == WorktreeStatus.IDLE
        assert "TASK-123" not in worktree_manager._task_mapping
        assert "test-worker" not in worktree_manager._worktree_tasks

    def test_get_worktree_for_task(self, worktree_manager: WorktreeManager) -> None:
        """Test looking up a worktree by task ID."""
        info = worktree_manager.create_worktree("test-worker")
        worktree_manager.assign_worktree("test-worker", "TASK-123")

        result = worktree_manager.get_worktree_for_task("TASK-123")

        assert result is not None
        assert result.name == "test-worker"
        assert result.task_id == "TASK-123"

    def test_get_worktree_for_unassigned_task(
        self, worktree_manager: WorktreeManager
    ) -> None:
        """Test looking up a nonexistent task returns None."""
        result = worktree_manager.get_worktree_for_task("TASK-999")
        assert result is None

    def test_cleanup_removes_task_mappings(
        self, worktree_manager: WorktreeManager
    ) -> None:
        """Test that cleanup removes task mappings."""
        worktree_manager.create_worktree("test-worker")
        worktree_manager.assign_worktree("test-worker", "TASK-123")

        worktree_manager.cleanup_worktree("test-worker")

        assert "TASK-123" not in worktree_manager._task_mapping
        assert "test-worker" not in worktree_manager._worktree_tasks


class TestWorktreePool:
    """Test suite for WorktreePool."""

    @pytest.mark.asyncio
    async def test_acquire_creates_new_worktree(
        self, worktree_pool: WorktreePool
    ) -> None:
        """Test that acquire creates a new worktree when pool is empty."""
        info = await worktree_pool.acquire()

        assert info.status == WorktreeStatus.ACTIVE
        assert info.name == "1"
        assert len(worktree_pool.manager.worktrees) == 1

    @pytest.mark.asyncio
    async def test_acquire_reuses_idle_worktree(
        self, worktree_pool: WorktreePool
    ) -> None:
        """Test that acquire reuses an idle worktree."""
        # Acquire and release a worktree
        info1 = await worktree_pool.acquire()
        name1 = info1.name
        await worktree_pool.release(name1)

        # Acquire again - should reuse
        info2 = await worktree_pool.acquire()

        assert info2.name == name1
        assert info2.status == WorktreeStatus.ACTIVE
        assert len(worktree_pool.manager.worktrees) == 1  # Still only one

    @pytest.mark.asyncio
    async def test_release_marks_idle(self, worktree_pool: WorktreePool) -> None:
        """Test that release marks worktree as idle."""
        info = await worktree_pool.acquire()
        name = info.name

        await worktree_pool.release(name)

        assert info.status == WorktreeStatus.IDLE
        assert info.task_id is None

    @pytest.mark.asyncio
    async def test_pool_max_workers_limit(self, worktree_pool: WorktreePool) -> None:
        """Test that pool enforces max_workers limit via semaphore."""
        # Pool has max_workers=3

        # Acquire 3 worktrees (should succeed immediately)
        infos = []
        for _ in range(3):
            info = await worktree_pool.acquire()
            infos.append(info)

        assert len(infos) == 3
        assert all(info.status == WorktreeStatus.ACTIVE for info in infos)

        # Try to acquire a 4th - should block
        acquire_task = asyncio.create_task(worktree_pool.acquire())

        # Give it a moment to try (should block on semaphore)
        await asyncio.sleep(0.1)
        assert not acquire_task.done()

        # Release one worktree
        await worktree_pool.release(infos[0].name)

        # Now the 4th should complete
        info4 = await asyncio.wait_for(acquire_task, timeout=1.0)
        assert info4 is not None
        assert info4.status == WorktreeStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_available(self, worktree_pool: WorktreePool) -> None:
        """Test getting list of available worktrees."""
        # Acquire 2, release 1
        info1 = await worktree_pool.acquire()
        info2 = await worktree_pool.acquire()
        await worktree_pool.release(info1.name)

        available = worktree_pool.get_available()

        assert len(available) == 1
        assert available[0].name == info1.name
        assert available[0].status == WorktreeStatus.IDLE

    @pytest.mark.asyncio
    async def test_get_active(self, worktree_pool: WorktreePool) -> None:
        """Test getting list of active worktrees."""
        # Acquire 2, release 1
        info1 = await worktree_pool.acquire()
        info2 = await worktree_pool.acquire()
        await worktree_pool.release(info1.name)

        active = worktree_pool.get_active()

        assert len(active) == 1
        assert active[0].name == info2.name
        assert active[0].status == WorktreeStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_assign_worktree_via_pool(
        self, worktree_pool: WorktreePool
    ) -> None:
        """Test assigning a task via pool."""
        info = await worktree_pool.acquire()

        worktree_pool.assign_worktree(info.name, "TASK-123")

        assert info.task_id == "TASK-123"
        assert info.status == WorktreeStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_release_nonexistent_worktree(
        self, worktree_pool: WorktreePool
    ) -> None:
        """Test that releasing a nonexistent worktree doesn't raise."""
        # Should not raise
        await worktree_pool.release("nonexistent")

    @pytest.mark.asyncio
    async def test_worktree_reset_on_release(
        self, worktree_pool: WorktreePool
    ) -> None:
        """Test that worktree is reset (cleaned) on release."""
        info = await worktree_pool.acquire()

        # Create a test file in the worktree
        test_file = info.path / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()

        # Release (should clean)
        await worktree_pool.release(info.name)

        # File should be gone after reset
        assert not test_file.exists()
