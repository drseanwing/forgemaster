"""Integration tests for git operations module.

These tests create real git repositories in temporary directories to verify
git operations work correctly with actual git commands and state.
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest
from git import GitCommandError

from forgemaster.config import GitConfig
from forgemaster.pipeline.git_ops import GitManager, MergeResult


@pytest.fixture
def temp_repo(tmp_path: Path) -> git.Repo:
    """Create a temporary git repository with initial commit.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        GitPython Repo object for the temporary repository
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize repository
    repo = git.Repo.init(repo_path)

    # Configure user (required for commits)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Create initial commit on main branch
    initial_file = repo_path / "README.md"
    initial_file.write_text("# Test Repository\n")
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")

    return repo


@pytest.fixture
def git_manager(temp_repo: git.Repo) -> GitManager:
    """Create GitManager instance for temporary repository.

    Args:
        temp_repo: Temporary git repository fixture

    Returns:
        GitManager configured for the temporary repository
    """
    config = GitConfig(
        main_branch="main",
        auto_push=False,
        worktree_base_path=Path("/tmp"),
    )
    return GitManager(repo_path=Path(temp_repo.working_dir), config=config)


def test_create_branch_from_main(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test creating a new branch from main branch."""
    branch_name = "feature/test-branch"

    # Create branch
    result = git_manager.create_branch(branch_name)

    # Verify branch was created
    assert result == branch_name
    assert branch_name in temp_repo.heads
    assert temp_repo.heads[branch_name].commit == temp_repo.heads["main"].commit


def test_create_branch_from_custom_base(
    git_manager: GitManager, temp_repo: git.Repo
) -> None:
    """Test creating a branch from a non-main base branch."""
    # Create base branch
    base_branch = "develop"
    temp_repo.create_head(base_branch, temp_repo.heads["main"].commit)

    # Create feature branch from develop
    feature_branch = "feature/from-develop"
    result = git_manager.create_branch(feature_branch, base_branch=base_branch)

    assert result == feature_branch
    assert feature_branch in temp_repo.heads
    assert (
        temp_repo.heads[feature_branch].commit == temp_repo.heads[base_branch].commit
    )


def test_create_branch_already_exists(
    git_manager: GitManager, temp_repo: git.Repo
) -> None:
    """Test error handling when creating a branch that already exists."""
    branch_name = "feature/duplicate"

    # Create branch first time
    git_manager.create_branch(branch_name)

    # Attempt to create same branch again
    with pytest.raises(GitCommandError, match="already exists"):
        git_manager.create_branch(branch_name)


def test_create_branch_invalid_base(git_manager: GitManager) -> None:
    """Test error handling when base branch doesn't exist."""
    with pytest.raises(GitCommandError, match="not found"):
        git_manager.create_branch("feature/test", base_branch="nonexistent")


def test_commit_with_specific_files(
    git_manager: GitManager, temp_repo: git.Repo
) -> None:
    """Test committing specific files."""
    repo_path = Path(temp_repo.working_dir)

    # Create test files
    file1 = repo_path / "file1.txt"
    file2 = repo_path / "file2.txt"
    file1.write_text("content 1")
    file2.write_text("content 2")

    # Commit only file1
    commit_sha = git_manager.commit(
        "Add file1",
        files=["file1.txt"],
    )

    # Verify commit was created
    assert commit_sha is not None
    assert len(commit_sha) == 40  # Full SHA-1 hash

    # Verify only file1 is in the commit
    commit = temp_repo.commit(commit_sha)
    assert "file1.txt" in commit.stats.files
    assert "file2.txt" not in commit.stats.files


def test_commit_all_changes(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test committing all changes when files=None."""
    repo_path = Path(temp_repo.working_dir)

    # Create multiple files
    for i in range(3):
        file = repo_path / f"file{i}.txt"
        file.write_text(f"content {i}")

    # Commit all changes
    commit_sha = git_manager.commit("Add multiple files")

    # Verify all files are in the commit
    commit = temp_repo.commit(commit_sha)
    assert len(commit.stats.files) == 3
    for i in range(3):
        assert f"file{i}.txt" in commit.stats.files


def test_commit_nothing_to_commit(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test error handling when there are no changes to commit."""
    with pytest.raises(GitCommandError, match="No changes to commit"):
        git_manager.commit("Empty commit")


def test_commit_modified_file(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test committing modifications to an existing file."""
    repo_path = Path(temp_repo.working_dir)

    # Modify existing file
    readme = repo_path / "README.md"
    readme.write_text("# Updated README\n\nNew content\n")

    # Commit the modification
    commit_sha = git_manager.commit("Update README")

    # Verify commit was created
    commit = temp_repo.commit(commit_sha)
    assert "README.md" in commit.stats.files


def test_merge_clean(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test merging branches with no conflicts."""
    repo_path = Path(temp_repo.working_dir)

    # Create feature branch
    feature_branch = "feature/add-file"
    git_manager.create_branch(feature_branch)
    temp_repo.heads[feature_branch].checkout()

    # Add file in feature branch
    new_file = repo_path / "feature.txt"
    new_file.write_text("feature content")
    git_manager.commit("Add feature file", files=["feature.txt"])

    # Switch back to main
    temp_repo.heads["main"].checkout()

    # Merge feature into main
    result = git_manager.merge(feature_branch, "main")

    # Verify merge succeeded
    assert isinstance(result, MergeResult)
    assert result.success is True
    assert result.commit_sha is not None
    assert len(result.commit_sha) == 40
    assert result.conflicts == []

    # Verify file exists in main branch
    assert new_file.exists()


def test_merge_with_conflicts(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test merging branches with conflicts."""
    repo_path = Path(temp_repo.working_dir)

    # Create feature branch
    feature_branch = "feature/conflict"
    git_manager.create_branch(feature_branch)

    # Modify README in main branch
    temp_repo.heads["main"].checkout()
    readme = repo_path / "README.md"
    readme.write_text("# Main branch version\n")
    git_manager.commit("Update README in main")

    # Modify README differently in feature branch
    temp_repo.heads[feature_branch].checkout()
    readme.write_text("# Feature branch version\n")
    git_manager.commit("Update README in feature")

    # Switch back to main and attempt merge
    temp_repo.heads["main"].checkout()
    result = git_manager.merge(feature_branch, "main")

    # Verify merge detected conflicts
    assert result.success is False
    assert result.commit_sha is None
    assert len(result.conflicts) > 0
    assert "README.md" in result.conflicts


def test_merge_current_branch(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test merging into current branch when target_branch=None."""
    repo_path = Path(temp_repo.working_dir)

    # Create and checkout feature branch
    feature_branch = "feature/merge-to-current"
    git_manager.create_branch(feature_branch)
    temp_repo.heads[feature_branch].checkout()
    current_file = repo_path / "current.txt"
    current_file.write_text("current content")
    git_manager.commit("Add current file")

    # Switch to main (this will be the target)
    temp_repo.heads["main"].checkout()

    # Merge feature into current branch (main)
    result = git_manager.merge(feature_branch)

    assert result.success is True
    assert current_file.exists()


def test_merge_nonexistent_source(git_manager: GitManager) -> None:
    """Test error handling when source branch doesn't exist."""
    with pytest.raises(GitCommandError, match="not found"):
        git_manager.merge("nonexistent-branch", "main")


def test_merge_nonexistent_target(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test error handling when target branch doesn't exist."""
    # Create source branch
    git_manager.create_branch("feature/source")

    with pytest.raises(GitCommandError, match="not found"):
        git_manager.merge("feature/source", "nonexistent-target")


def test_detect_conflicts_clean(git_manager: GitManager, temp_repo: git.Repo) -> None:
    """Test conflict detection with no conflicts."""
    repo_path = Path(temp_repo.working_dir)

    # Create feature branch with new file
    feature_branch = "feature/no-conflict"
    git_manager.create_branch(feature_branch)
    temp_repo.heads[feature_branch].checkout()
    new_file = repo_path / "new.txt"
    new_file.write_text("new content")
    git_manager.commit("Add new file")

    # Detect conflicts between feature and main
    conflicts = git_manager.detect_conflicts(feature_branch, "main")

    assert conflicts == []


def test_detect_conflicts_with_conflicts(
    git_manager: GitManager, temp_repo: git.Repo
) -> None:
    """Test conflict detection when conflicts exist."""
    repo_path = Path(temp_repo.working_dir)

    # Create feature branch
    feature_branch = "feature/with-conflict"
    git_manager.create_branch(feature_branch)

    # Modify file in main
    temp_repo.heads["main"].checkout()
    readme = repo_path / "README.md"
    readme.write_text("# Main version\n")
    git_manager.commit("Update in main")

    # Modify same file in feature
    temp_repo.heads[feature_branch].checkout()
    readme.write_text("# Feature version\n")
    git_manager.commit("Update in feature")

    # Detect conflicts
    conflicts = git_manager.detect_conflicts(feature_branch, "main")

    assert len(conflicts) > 0
    assert "README.md" in conflicts


def test_detect_conflicts_preserves_state(
    git_manager: GitManager, temp_repo: git.Repo
) -> None:
    """Test that conflict detection doesn't modify repository state."""
    original_branch = temp_repo.active_branch.name

    # Create feature branch
    feature_branch = "feature/state-test"
    git_manager.create_branch(feature_branch)

    # Detect conflicts (this should checkout branches temporarily)
    git_manager.detect_conflicts(feature_branch, "main")

    # Verify we're back on original branch
    assert temp_repo.active_branch.name == original_branch


def test_detect_conflicts_nonexistent_branch_a(git_manager: GitManager) -> None:
    """Test error handling when branch_a doesn't exist."""
    with pytest.raises(GitCommandError, match="not found"):
        git_manager.detect_conflicts("nonexistent", "main")


def test_detect_conflicts_nonexistent_branch_b(
    git_manager: GitManager, temp_repo: git.Repo
) -> None:
    """Test error handling when branch_b doesn't exist."""
    git_manager.create_branch("feature/test")

    with pytest.raises(GitCommandError, match="not found"):
        git_manager.detect_conflicts("feature/test", "nonexistent")


def test_git_manager_invalid_repo_path(tmp_path: Path) -> None:
    """Test error handling when initializing with invalid repository path."""
    invalid_path = tmp_path / "not-a-repo"
    invalid_path.mkdir()

    config = GitConfig(main_branch="main", auto_push=False)

    with pytest.raises(git.InvalidGitRepositoryError):
        GitManager(repo_path=invalid_path, config=config)


def test_git_manager_nonexistent_path() -> None:
    """Test error handling when repository path doesn't exist."""
    config = GitConfig(main_branch="main", auto_push=False)

    with pytest.raises(git.NoSuchPathError):
        GitManager(repo_path=Path("/nonexistent/path"), config=config)
