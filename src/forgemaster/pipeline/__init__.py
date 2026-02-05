"""Deterministic pipeline operations for Forgemaster.

This module implements git operations (worktree management, branching, merging),
Docker build/push, registry interactions, and deployment steps.
"""

from forgemaster.pipeline.docker_ops import (
    BuildLogEntry,
    BuildResult,
    BuildStatus,
    DockerBuildClient,
    DockerHealth,
    RootlessConfig,
    SemverTag,
)
from forgemaster.pipeline.git_ops import GitManager, MergeResult
from forgemaster.pipeline.worktree import (
    WorktreeInfo,
    WorktreeManager,
    WorktreePool,
    WorktreeStatus,
)

__all__ = [
    "BuildLogEntry",
    "BuildResult",
    "BuildStatus",
    "DockerBuildClient",
    "DockerHealth",
    "GitManager",
    "MergeResult",
    "RootlessConfig",
    "SemverTag",
    "WorktreeInfo",
    "WorktreeManager",
    "WorktreePool",
    "WorktreeStatus",
]
