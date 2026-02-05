"""Deterministic pipeline operations for Forgemaster.

This module implements git operations (worktree management, branching, merging),
Docker build/push, registry interactions, and deployment steps.
"""

from forgemaster.pipeline.git_ops import GitManager, MergeResult

__all__ = [
    "GitManager",
    "MergeResult",
]
