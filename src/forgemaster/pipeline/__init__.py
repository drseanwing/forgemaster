"""Deterministic pipeline operations for Forgemaster.

This module implements git operations (worktree management, branching, merging),
Docker build/push, registry interactions, container management, health monitoring,
and deployment steps.
"""

from __future__ import annotations

from forgemaster.pipeline.container import (
    ComposeAction,
    ComposeManager,
    ContainerAction,
    ContainerInfo,
    ContainerManager,
    ContainerStatus,
)
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
from forgemaster.pipeline.health import (
    HealthCheckConfig,
    HealthCheckResult,
    HealthCheckTimeout,
    HealthPoller,
    HealthStatus,
    RollbackConfig,
    RollbackDecision,
    RollbackExecutor,
    RollbackResult,
    RollbackStrategy,
    RollbackTrigger,
)
from forgemaster.pipeline.registry import (
    PushResult,
    PushStatus,
    RegistryAuth,
    RegistryClient,
    RetryConfig,
)
from forgemaster.pipeline.worktree import (
    WorktreeInfo,
    WorktreeManager,
    WorktreePool,
    WorktreeStatus,
)

__all__ = [
    # Docker build
    "BuildLogEntry",
    "BuildResult",
    "BuildStatus",
    "DockerBuildClient",
    "DockerHealth",
    "RootlessConfig",
    "SemverTag",
    # Git operations
    "GitManager",
    "MergeResult",
    # Worktree management
    "WorktreeInfo",
    "WorktreeManager",
    "WorktreePool",
    "WorktreeStatus",
    # Registry operations
    "PushResult",
    "PushStatus",
    "RegistryAuth",
    "RegistryClient",
    "RetryConfig",
    # Container management
    "ComposeAction",
    "ComposeManager",
    "ContainerAction",
    "ContainerInfo",
    "ContainerManager",
    "ContainerStatus",
    # Health check system
    "HealthCheckConfig",
    "HealthCheckResult",
    "HealthCheckTimeout",
    "HealthPoller",
    "HealthStatus",
    "RollbackConfig",
    "RollbackDecision",
    "RollbackExecutor",
    "RollbackResult",
    "RollbackStrategy",
    "RollbackTrigger",
]
