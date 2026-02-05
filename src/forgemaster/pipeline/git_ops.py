"""Git operations wrapper for FORGEMASTER.

This module provides a high-level interface to git operations using GitPython,
with comprehensive error handling and structured logging. Supports branch
creation, committing, merging, and conflict detection.

Example usage:
    >>> from pathlib import Path
    >>> from forgemaster.config import GitConfig
    >>> from forgemaster.pipeline.git_ops import GitManager, MergeResult
    >>>
    >>> config = GitConfig(main_branch="main", auto_push=False)
    >>> git_manager = GitManager(repo_path=Path("/workspace/repo"), config=config)
    >>>
    >>> # Create a new branch for agent work
    >>> branch_name = git_manager.create_branch("feature/agent-123")
    >>>
    >>> # Commit changes
    >>> commit_sha = git_manager.commit("feat: implement feature X", files=["src/foo.py"])
    >>>
    >>> # Check for conflicts before merging
    >>> conflicts = git_manager.detect_conflicts("feature/agent-123", "main")
    >>> if not conflicts:
    ...     result = git_manager.merge("feature/agent-123", "main")
    ...     if result.success:
    ...         print(f"Merged successfully: {result.commit_sha}")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import git
from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

from forgemaster.config import GitConfig
from forgemaster.logging import get_logger


@dataclass
class MergeResult:
    """Result of a merge operation.

    Attributes:
        success: True if merge completed without conflicts
        commit_sha: SHA of the merge commit (None if conflicts)
        conflicts: List of file paths with conflicts (empty if successful)
    """

    success: bool
    commit_sha: str | None
    conflicts: list[str]


class GitManager:
    """High-level git operations manager using GitPython.

    This class wraps GitPython operations with error handling, logging,
    and FORGEMASTER-specific conventions for branch management and merging.

    Attributes:
        repo_path: Path to the git repository
        config: Git configuration from ForgemasterConfig
        repo: GitPython Repo object
        logger: Structured logger instance
    """

    def __init__(self, repo_path: Path, config: GitConfig) -> None:
        """Initialize GitManager with repository and configuration.

        Args:
            repo_path: Path to the git repository root
            config: Git configuration settings

        Raises:
            InvalidGitRepositoryError: If repo_path is not a valid git repository
            NoSuchPathError: If repo_path does not exist
        """
        self.repo_path = repo_path
        self.config = config
        self.logger = get_logger(__name__)

        try:
            self.repo = git.Repo(repo_path)
            self.logger.info(
                "git_manager_initialized",
                repo_path=str(repo_path),
                main_branch=config.main_branch,
                auto_push=config.auto_push,
            )
        except (InvalidGitRepositoryError, NoSuchPathError) as e:
            self.logger.error(
                "git_manager_init_failed",
                repo_path=str(repo_path),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def create_branch(
        self, branch_name: str, base_branch: str | None = None
    ) -> str:
        """Create a new git branch from base branch.

        Args:
            branch_name: Name of the new branch to create
            base_branch: Base branch to create from (default: config.main_branch)

        Returns:
            Name of the created branch

        Raises:
            GitCommandError: If branch already exists or base branch not found
        """
        if base_branch is None:
            base_branch = self.config.main_branch

        try:
            # Ensure we're tracking the base branch
            if base_branch not in self.repo.heads:
                self.logger.error(
                    "base_branch_not_found",
                    base_branch=base_branch,
                    available_branches=[h.name for h in self.repo.heads],
                )
                raise GitCommandError(
                    f"create_branch",
                    f"Base branch '{base_branch}' not found",
                )

            # Check if branch already exists
            if branch_name in self.repo.heads:
                self.logger.warning(
                    "branch_already_exists",
                    branch_name=branch_name,
                    base_branch=base_branch,
                )
                raise GitCommandError(
                    f"create_branch",
                    f"Branch '{branch_name}' already exists",
                )

            # Create the new branch from base
            base_commit = self.repo.heads[base_branch].commit
            new_branch = self.repo.create_head(branch_name, base_commit)

            self.logger.info(
                "branch_created",
                branch_name=branch_name,
                base_branch=base_branch,
                commit_sha=str(base_commit),
            )

            return new_branch.name

        except GitCommandError as e:
            self.logger.error(
                "branch_creation_failed",
                branch_name=branch_name,
                base_branch=base_branch,
                error=str(e),
            )
            raise

    def commit(
        self, message: str, files: list[str] | None = None
    ) -> str:
        """Create a git commit with the specified message.

        Args:
            message: Commit message
            files: List of file paths to stage (None to stage all changes)

        Returns:
            SHA of the created commit

        Raises:
            GitCommandError: If commit fails or there's nothing to commit
        """
        try:
            # Stage files
            if files is not None:
                # Stage specific files
                for file_path in files:
                    self.repo.index.add([file_path])
                self.logger.debug(
                    "files_staged",
                    file_count=len(files),
                    files=files,
                )
            else:
                # Stage all changes (modified, deleted, new)
                self.repo.git.add(A=True)
                self.logger.debug("all_changes_staged")

            # Check if there are changes to commit
            if not self.repo.index.diff("HEAD") and not self.repo.untracked_files:
                self.logger.warning(
                    "nothing_to_commit",
                    message=message,
                )
                raise GitCommandError(
                    "commit",
                    "No changes to commit",
                )

            # Create the commit
            commit = self.repo.index.commit(message)

            self.logger.info(
                "commit_created",
                commit_sha=commit.hexsha[:8],
                full_sha=commit.hexsha,
                message=message,
                files_changed=len(commit.stats.files),
            )

            return commit.hexsha

        except GitCommandError as e:
            self.logger.error(
                "commit_failed",
                message=message,
                files=files,
                error=str(e),
            )
            raise

    def merge(
        self, source_branch: str, target_branch: str | None = None
    ) -> MergeResult:
        """Merge source branch into target branch.

        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into (default: current branch)

        Returns:
            MergeResult with success status, commit SHA, and conflict list

        Raises:
            GitCommandError: If branches don't exist or merge operation fails
        """
        try:
            # Validate source branch exists
            if source_branch not in self.repo.heads:
                self.logger.error(
                    "source_branch_not_found",
                    source_branch=source_branch,
                )
                raise GitCommandError(
                    "merge",
                    f"Source branch '{source_branch}' not found",
                )

            # Checkout target branch if specified
            if target_branch is not None:
                if target_branch not in self.repo.heads:
                    self.logger.error(
                        "target_branch_not_found",
                        target_branch=target_branch,
                    )
                    raise GitCommandError(
                        "merge",
                        f"Target branch '{target_branch}' not found",
                    )
                self.repo.heads[target_branch].checkout()
                self.logger.debug(
                    "checked_out_target_branch",
                    target_branch=target_branch,
                )
            else:
                target_branch = self.repo.active_branch.name

            # Perform the merge
            source_commit = self.repo.heads[source_branch].commit
            base = self.repo.merge_base(source_commit, "HEAD")[0]

            try:
                # Attempt the merge
                self.repo.index.merge_tree(
                    "HEAD",
                    base,
                    source_commit,
                )

                # Check for conflicts
                conflicts = [
                    entry[0]
                    for entry in self.repo.index.unmerged_blobs().items()
                ]

                if conflicts:
                    # Conflicts detected - abort the merge
                    self.repo.git.merge("--abort")
                    self.logger.warning(
                        "merge_conflicts_detected",
                        source_branch=source_branch,
                        target_branch=target_branch,
                        conflict_count=len(conflicts),
                        conflicts=conflicts,
                    )
                    return MergeResult(
                        success=False,
                        commit_sha=None,
                        conflicts=conflicts,
                    )

                # No conflicts - complete the merge
                merge_msg = f"Merge branch '{source_branch}' into {target_branch}"
                commit = self.repo.index.commit(
                    merge_msg,
                    parent_commits=(self.repo.head.commit, source_commit),
                )

                self.logger.info(
                    "merge_successful",
                    source_branch=source_branch,
                    target_branch=target_branch,
                    commit_sha=commit.hexsha[:8],
                    full_sha=commit.hexsha,
                )

                return MergeResult(
                    success=True,
                    commit_sha=commit.hexsha,
                    conflicts=[],
                )

            except GitCommandError as e:
                self.logger.error(
                    "merge_operation_failed",
                    source_branch=source_branch,
                    target_branch=target_branch,
                    error=str(e),
                )
                # Attempt to abort the merge
                try:
                    self.repo.git.merge("--abort")
                except Exception:
                    pass
                raise

        except GitCommandError as e:
            self.logger.error(
                "merge_failed",
                source_branch=source_branch,
                target_branch=target_branch,
                error=str(e),
            )
            raise

    def detect_conflicts(self, branch_a: str, branch_b: str) -> list[str]:
        """Detect potential merge conflicts between two branches.

        This performs a test merge without committing to detect conflicts.
        The repository state is not modified.

        Args:
            branch_a: First branch name
            branch_b: Second branch name

        Returns:
            List of file paths that would conflict (empty if no conflicts)

        Raises:
            GitCommandError: If branches don't exist
        """
        try:
            # Validate both branches exist
            if branch_a not in self.repo.heads:
                self.logger.error(
                    "branch_not_found",
                    branch=branch_a,
                )
                raise GitCommandError(
                    "detect_conflicts",
                    f"Branch '{branch_a}' not found",
                )
            if branch_b not in self.repo.heads:
                self.logger.error(
                    "branch_not_found",
                    branch=branch_b,
                )
                raise GitCommandError(
                    "detect_conflicts",
                    f"Branch '{branch_b}' not found",
                )

            # Save current branch
            current_branch = self.repo.active_branch.name

            try:
                # Checkout branch_a
                self.repo.heads[branch_a].checkout()

                # Attempt merge with no-commit and no-ff
                try:
                    self.repo.git.merge(
                        branch_b,
                        no_commit=True,
                        no_ff=True,
                    )
                    # No conflicts - abort the merge
                    self.repo.git.merge("--abort")
                    self.logger.debug(
                        "no_conflicts_detected",
                        branch_a=branch_a,
                        branch_b=branch_b,
                    )
                    return []
                except GitCommandError:
                    # Merge failed - check for conflicts
                    conflicts = [
                        entry[0]
                        for entry in self.repo.index.unmerged_blobs().items()
                    ]

                    # Abort the merge
                    self.repo.git.merge("--abort")

                    self.logger.info(
                        "conflicts_detected",
                        branch_a=branch_a,
                        branch_b=branch_b,
                        conflict_count=len(conflicts),
                        conflicts=conflicts,
                    )

                    return conflicts

            finally:
                # Restore original branch
                self.repo.heads[current_branch].checkout()

        except GitCommandError as e:
            self.logger.error(
                "conflict_detection_failed",
                branch_a=branch_a,
                branch_b=branch_b,
                error=str(e),
            )
            raise
