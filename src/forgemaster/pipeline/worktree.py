"""Git worktree management for parallel agent execution.

This module provides worktree management for FORGEMASTER's parallelization system,
allowing multiple agents to work on separate branches in isolated worktrees without
interfering with each other.

The WorktreeManager handles:
- Creating and removing worktrees with dedicated branches
- Managing a pool of reusable worktrees (WorktreePool)
- Tracking task-to-worktree assignments
- Lifecycle management (acquire, release, cleanup)

Example usage:
    >>> from pathlib import Path
    >>> from forgemaster.pipeline.git_ops import GitManager
    >>> from forgemaster.pipeline.worktree import WorktreeManager, WorktreePool
    >>> from forgemaster.config import GitConfig
    >>>
    >>> git_config = GitConfig(main_branch="main", worktree_base_path=Path("/workspace"))
    >>> git_manager = GitManager(repo_path=Path("/repo"), config=git_config)
    >>> worktree_manager = WorktreeManager(git_manager, base_path=Path("/workspace"))
    >>>
    >>> # Create a worktree pool for parallel workers
    >>> pool = WorktreePool(worktree_manager, max_workers=5)
    >>>
    >>> # Acquire a worktree for a task
    >>> worktree = await pool.acquire()
    >>> pool.assign_worktree(worktree.name, "TASK-123")
    >>>
    >>> # Work in the worktree...
    >>>
    >>> # Release back to pool
    >>> await pool.release(worktree.name)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from forgemaster.logging import get_logger
from forgemaster.pipeline.git_ops import GitManager


class WorktreeStatus(str, Enum):
    """Status of a git worktree in the lifecycle.

    Attributes:
        ACTIVE: Worktree is currently assigned to a task
        IDLE: Worktree is available in the pool
        CLEANING: Worktree is being cleaned/reset
        REMOVED: Worktree has been removed
    """

    ACTIVE = "active"
    IDLE = "idle"
    CLEANING = "cleaning"
    REMOVED = "removed"


class WorktreeInfo(BaseModel):
    """Information about a git worktree.

    Attributes:
        name: Unique identifier for this worktree (e.g., "worker-1")
        path: Filesystem path to the worktree directory
        branch: Git branch associated with this worktree
        status: Current lifecycle status
        created_at: UTC timestamp when worktree was created
        task_id: ID of task currently assigned to this worktree (None if unassigned)
    """

    name: str
    path: Path
    branch: str
    status: WorktreeStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    task_id: str | None = None


class WorktreeManager:
    """Manages git worktrees for parallel agent execution.

    This class provides low-level worktree operations including creation,
    removal, and task assignment tracking. For pool-based management with
    automatic lifecycle handling, use WorktreePool.

    Attributes:
        git_manager: GitManager instance for git operations
        base_path: Base directory where worktrees are created
        logger: Structured logger instance
        worktrees: Registry of all managed worktrees
        _task_mapping: Maps task_id to worktree name
        _worktree_tasks: Maps worktree name to task_id (reverse mapping)
    """

    def __init__(
        self, git_manager: GitManager, base_path: Path | None = None
    ) -> None:
        """Initialize WorktreeManager.

        Args:
            git_manager: GitManager instance for git operations
            base_path: Base directory for worktrees (default: from git_manager config)
        """
        self.git_manager = git_manager
        self.base_path = base_path or git_manager.config.worktree_base_path
        self.logger = get_logger(__name__)
        self.worktrees: dict[str, WorktreeInfo] = {}
        self._task_mapping: dict[str, str] = {}  # task_id -> worktree name
        self._worktree_tasks: dict[str, str] = {}  # worktree name -> task_id

        self.logger.info(
            "worktree_manager_initialized",
            base_path=str(self.base_path),
            repo_path=str(git_manager.repo_path),
        )

    def create_worktree(
        self, name: str, branch: str | None = None
    ) -> WorktreeInfo:
        """Create a new git worktree with a dedicated branch.

        Args:
            name: Unique name for the worktree (e.g., "worker-1")
            branch: Branch name to use (default: f"worktree/{name}")

        Returns:
            WorktreeInfo describing the created worktree

        Raises:
            ValueError: If worktree with this name already exists
            GitCommandError: If worktree creation fails
        """
        if name in self.worktrees:
            self.logger.error(
                "worktree_already_exists",
                name=name,
            )
            raise ValueError(f"Worktree '{name}' already exists")

        # Generate branch name if not provided
        if branch is None:
            branch = f"worktree/{name}"

        worktree_path = self.base_path / f"worker-{name}"

        try:
            # Ensure base path exists
            self.base_path.mkdir(parents=True, exist_ok=True)

            # Create branch from main
            self.git_manager.create_branch(branch)

            # Create worktree at the specified path with the branch
            self.git_manager.repo.git.worktree("add", str(worktree_path), branch)

            # Create WorktreeInfo
            info = WorktreeInfo(
                name=name,
                path=worktree_path,
                branch=branch,
                status=WorktreeStatus.IDLE,
            )

            self.worktrees[name] = info

            self.logger.info(
                "worktree_created",
                name=name,
                path=str(worktree_path),
                branch=branch,
            )

            return info

        except Exception as e:
            self.logger.error(
                "worktree_creation_failed",
                name=name,
                path=str(worktree_path),
                branch=branch,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def cleanup_worktree(self, name: str) -> bool:
        """Remove a specific worktree and its branch.

        Args:
            name: Name of the worktree to remove

        Returns:
            True if successfully cleaned, False if worktree not found

        Raises:
            RuntimeError: If cleanup fails
        """
        if name not in self.worktrees:
            self.logger.warning(
                "worktree_not_found",
                name=name,
            )
            return False

        info = self.worktrees[name]

        try:
            # Update status
            info.status = WorktreeStatus.CLEANING

            # Remove worktree (--force handles uncommitted changes)
            self.git_manager.repo.git.worktree("remove", "--force", str(info.path))

            # Delete the branch
            if info.branch in self.git_manager.repo.heads:
                self.git_manager.repo.delete_head(info.branch, force=True)

            # Remove from registry
            info.status = WorktreeStatus.REMOVED
            del self.worktrees[name]

            # Clean up task mappings
            if name in self._worktree_tasks:
                task_id = self._worktree_tasks[name]
                del self._worktree_tasks[name]
                if task_id in self._task_mapping:
                    del self._task_mapping[task_id]

            self.logger.info(
                "worktree_cleaned",
                name=name,
                path=str(info.path),
                branch=info.branch,
            )

            return True

        except Exception as e:
            self.logger.error(
                "worktree_cleanup_failed",
                name=name,
                path=str(info.path),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Failed to cleanup worktree '{name}': {e}") from e

    def cleanup_all(self) -> int:
        """Remove all managed worktrees.

        Returns:
            Number of worktrees successfully removed
        """
        count = 0
        worktree_names = list(self.worktrees.keys())

        for name in worktree_names:
            try:
                if self.cleanup_worktree(name):
                    count += 1
            except Exception as e:
                self.logger.warning(
                    "worktree_cleanup_skipped",
                    name=name,
                    error=str(e),
                )

        self.logger.info(
            "worktrees_cleaned",
            total_count=len(worktree_names),
            success_count=count,
        )

        return count

    def get_worktree_for_task(self, task_id: str) -> WorktreeInfo | None:
        """Lookup which worktree is assigned to a task.

        Args:
            task_id: Task identifier to look up

        Returns:
            WorktreeInfo if task is assigned to a worktree, None otherwise
        """
        worktree_name = self._task_mapping.get(task_id)
        if worktree_name is None:
            return None
        return self.worktrees.get(worktree_name)

    def assign_worktree(self, worktree_name: str, task_id: str) -> None:
        """Associate a task with a worktree.

        Args:
            worktree_name: Name of the worktree
            task_id: Task identifier to assign

        Raises:
            ValueError: If worktree doesn't exist or is already assigned
        """
        if worktree_name not in self.worktrees:
            raise ValueError(f"Worktree '{worktree_name}' not found")

        info = self.worktrees[worktree_name]

        if info.task_id is not None:
            raise ValueError(
                f"Worktree '{worktree_name}' is already assigned to task '{info.task_id}'"
            )

        # Update bidirectional mapping
        self._task_mapping[task_id] = worktree_name
        self._worktree_tasks[worktree_name] = task_id
        info.task_id = task_id
        info.status = WorktreeStatus.ACTIVE

        self.logger.info(
            "worktree_assigned",
            worktree_name=worktree_name,
            task_id=task_id,
        )

    def unassign_worktree(self, worktree_name: str) -> None:
        """Remove task association from a worktree.

        Args:
            worktree_name: Name of the worktree to unassign
        """
        if worktree_name not in self.worktrees:
            return

        info = self.worktrees[worktree_name]

        if info.task_id is not None:
            # Remove bidirectional mapping
            task_id = info.task_id
            if task_id in self._task_mapping:
                del self._task_mapping[task_id]
            if worktree_name in self._worktree_tasks:
                del self._worktree_tasks[worktree_name]

            self.logger.info(
                "worktree_unassigned",
                worktree_name=worktree_name,
                task_id=task_id,
            )

        info.task_id = None
        info.status = WorktreeStatus.IDLE


class WorktreePool:
    """Pool manager for reusable worktrees.

    This class provides a high-level interface for managing a fixed-size pool
    of worktrees. Worktrees are automatically created on-demand and reused
    through acquire/release cycles. The pool enforces a maximum size using
    an asyncio.Semaphore.

    Attributes:
        manager: WorktreeManager for low-level operations
        max_workers: Maximum number of concurrent worktrees
        logger: Structured logger instance
        _semaphore: Semaphore to limit concurrent worktrees
        _worker_counter: Counter for generating unique worker names
    """

    def __init__(self, manager: WorktreeManager, max_workers: int = 5) -> None:
        """Initialize WorktreePool.

        Args:
            manager: WorktreeManager instance for worktree operations
            max_workers: Maximum number of concurrent worktrees
        """
        self.manager = manager
        self.max_workers = max_workers
        self.logger = get_logger(__name__)
        self._semaphore = asyncio.Semaphore(max_workers)
        self._worker_counter = 0

        self.logger.info(
            "worktree_pool_initialized",
            max_workers=max_workers,
        )

    async def acquire(self) -> WorktreeInfo:
        """Get an available worktree from the pool.

        If an idle worktree exists, it's reused. Otherwise, a new worktree
        is created if the pool has capacity. Blocks if the pool is at max
        capacity until a worktree is released.

        Returns:
            WorktreeInfo for an available worktree

        Raises:
            RuntimeError: If worktree creation fails
        """
        # Block if pool is at capacity
        await self._semaphore.acquire()

        try:
            # Try to find an idle worktree
            idle_worktrees = self.get_available()
            if idle_worktrees:
                info = idle_worktrees[0]
                info.status = WorktreeStatus.ACTIVE
                self.logger.info(
                    "worktree_acquired_from_pool",
                    name=info.name,
                )
                return info

            # No idle worktree - create a new one
            self._worker_counter += 1
            name = f"{self._worker_counter}"
            info = self.manager.create_worktree(name)
            info.status = WorktreeStatus.ACTIVE

            self.logger.info(
                "worktree_acquired_new",
                name=info.name,
            )

            return info

        except Exception as e:
            # Release semaphore on failure
            self._semaphore.release()
            self.logger.error(
                "worktree_acquire_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Failed to acquire worktree: {e}") from e

    async def release(self, name: str) -> None:
        """Return a worktree to the pool.

        The worktree is reset to a clean state (unassigned, branch cleaned,
        status set to IDLE) before being returned to the pool.

        Args:
            name: Name of the worktree to release
        """
        try:
            if name not in self.manager.worktrees:
                self.logger.warning(
                    "worktree_not_found_for_release",
                    name=name,
                )
                return

            info = self.manager.worktrees[name]

            # Unassign task
            self.manager.unassign_worktree(name)

            # Reset the worktree
            await self._reset_worktree(info)

            # Mark as idle
            info.status = WorktreeStatus.IDLE

            self.logger.info(
                "worktree_released",
                name=name,
            )

        finally:
            # Always release the semaphore
            self._semaphore.release()

    def get_available(self) -> list[WorktreeInfo]:
        """List available (IDLE) worktrees in the pool.

        Returns:
            List of WorktreeInfo objects with IDLE status
        """
        return [
            info
            for info in self.manager.worktrees.values()
            if info.status == WorktreeStatus.IDLE
        ]

    def get_active(self) -> list[WorktreeInfo]:
        """List in-use (ACTIVE) worktrees in the pool.

        Returns:
            List of WorktreeInfo objects with ACTIVE status
        """
        return [
            info
            for info in self.manager.worktrees.values()
            if info.status == WorktreeStatus.ACTIVE
        ]

    def assign_worktree(self, worktree_name: str, task_id: str) -> None:
        """Associate a task with a worktree.

        Args:
            worktree_name: Name of the worktree
            task_id: Task identifier to assign
        """
        self.manager.assign_worktree(worktree_name, task_id)

    async def _reset_worktree(self, info: WorktreeInfo) -> None:
        """Reset worktree to clean state.

        This performs:
        - Reset to HEAD (uncommitted changes)
        - Clean all untracked files and directories

        Note: We don't checkout main branch because each worktree has its own
        dedicated branch, and main is typically locked by the main repo.

        Args:
            info: WorktreeInfo to reset
        """
        try:
            info.status = WorktreeStatus.CLEANING

            # Use git from the worktree's repo
            import git

            worktree_repo = git.Repo(info.path)

            # Reset any uncommitted changes to HEAD
            worktree_repo.git.reset("--hard", "HEAD")

            # Clean all untracked files and directories
            worktree_repo.git.clean("-fdx")

            self.logger.debug(
                "worktree_reset",
                name=info.name,
                path=str(info.path),
            )

        except Exception as e:
            self.logger.error(
                "worktree_reset_failed",
                name=info.name,
                path=str(info.path),
                error=str(e),
                error_type=type(e).__name__,
            )
            # Don't raise - best effort cleanup
