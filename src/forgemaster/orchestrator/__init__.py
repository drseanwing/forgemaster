"""Orchestrator subsystem for Forgemaster.

This module implements the task dispatcher, state machine, session health
monitor, file conflict detection, parallel group scheduling, parallel
worker coordination, merge coordination, session handover, crash recovery,
idle watchdog, and API rate limiting logic.
"""

from __future__ import annotations

from forgemaster.orchestrator.dispatcher import (
    Dispatcher,
    MultiWorkerDispatcher,
    WorkerSlot,
    WorkerState,
)
from forgemaster.orchestrator.file_locker import FileLocker, LockConflict
from forgemaster.orchestrator.handover import (
    ContextExhaustionDetector,
    HandoverContext,
    HandoverPromptGenerator,
    HandoverReason,
    HandoverStore,
    HandoverTrigger,
    SaveExitResponse,
    SessionHandoverManager,
)
from forgemaster.orchestrator.health_monitor import HealthMonitor
from forgemaster.orchestrator.merge_coordinator import (
    MergeCoordinator,
    MergeRequest,
    MergeStatus,
)
from forgemaster.orchestrator.rate_limiter import (
    AdaptiveThrottler,
    BackoffConfig,
    ExponentialBackoff,
    ParallelismReduction,
    RateLimitConfig,
    RateLimitHandler,
    RateLimitResponse,
    RateLimitState,
    TokenBucket,
)
from forgemaster.orchestrator.recovery import (
    CleanupAction,
    CleanupResult,
    OrphanDetector,
    OrphanReason,
    OrphanSession,
    RecoveryManager,
    RecoveryReport,
    RetryDecision,
    SessionCleaner,
    RetryScheduler,
)
from forgemaster.orchestrator.result_handler import ResultHandler
from forgemaster.orchestrator.scheduler import (
    GroupStatus,
    ParallelGroupScheduler,
    ScheduledGroup,
)
from forgemaster.orchestrator.state_machine import (
    VALID_TRANSITIONS,
    InvalidTransitionError,
    TaskStateMachine,
    validate_transition,
)
from forgemaster.orchestrator.watchdog import (
    ActivityRecord,
    ActivityTracker,
    ActivityType,
    IdleDetector,
    IdleSeverity,
    IdleSession,
    IdleWatchdog,
    WatchdogAction,
    WatchdogActionType,
)

__all__ = [
    # Dispatcher
    "Dispatcher",
    "MultiWorkerDispatcher",
    "WorkerSlot",
    "WorkerState",
    # File locking
    "FileLocker",
    "LockConflict",
    # Session handover
    "ContextExhaustionDetector",
    "HandoverContext",
    "HandoverPromptGenerator",
    "HandoverReason",
    "HandoverStore",
    "HandoverTrigger",
    "SaveExitResponse",
    "SessionHandoverManager",
    # Health monitor
    "HealthMonitor",
    # Merge coordinator
    "MergeCoordinator",
    "MergeRequest",
    "MergeStatus",
    # Rate limiter
    "AdaptiveThrottler",
    "BackoffConfig",
    "ExponentialBackoff",
    "ParallelismReduction",
    "RateLimitConfig",
    "RateLimitHandler",
    "RateLimitResponse",
    "RateLimitState",
    "TokenBucket",
    # Crash recovery
    "CleanupAction",
    "CleanupResult",
    "OrphanDetector",
    "OrphanReason",
    "OrphanSession",
    "RecoveryManager",
    "RecoveryReport",
    "RetryDecision",
    "RetryScheduler",
    "SessionCleaner",
    # Result handler
    "ResultHandler",
    # Scheduler
    "GroupStatus",
    "ParallelGroupScheduler",
    "ScheduledGroup",
    # State machine
    "InvalidTransitionError",
    "TaskStateMachine",
    "VALID_TRANSITIONS",
    "validate_transition",
    # Watchdog
    "ActivityRecord",
    "ActivityTracker",
    "ActivityType",
    "IdleDetector",
    "IdleSeverity",
    "IdleSession",
    "IdleWatchdog",
    "WatchdogAction",
    "WatchdogActionType",
]
