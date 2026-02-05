"""Orchestrator subsystem for Forgemaster.

This module implements the task dispatcher, state machine, session health
monitor, file conflict detection, parallel group scheduling, parallel
worker coordination, and merge coordination logic.
"""

from forgemaster.orchestrator.dispatcher import (
    Dispatcher,
    MultiWorkerDispatcher,
    WorkerSlot,
    WorkerState,
)
from forgemaster.orchestrator.file_locker import FileLocker, LockConflict
from forgemaster.orchestrator.health_monitor import HealthMonitor
from forgemaster.orchestrator.merge_coordinator import (
    MergeCoordinator,
    MergeRequest,
    MergeStatus,
)
from forgemaster.orchestrator.result_handler import ResultHandler
from forgemaster.orchestrator.scheduler import (
    GroupStatus,
    ParallelGroupScheduler,
    ScheduledGroup,
)
from forgemaster.orchestrator.state_machine import (
    InvalidTransitionError,
    TaskStateMachine,
    VALID_TRANSITIONS,
    validate_transition,
)

__all__ = [
    "Dispatcher",
    "FileLocker",
    "GroupStatus",
    "HealthMonitor",
    "InvalidTransitionError",
    "LockConflict",
    "MergeCoordinator",
    "MergeRequest",
    "MergeStatus",
    "MultiWorkerDispatcher",
    "ParallelGroupScheduler",
    "ResultHandler",
    "ScheduledGroup",
    "TaskStateMachine",
    "VALID_TRANSITIONS",
    "WorkerSlot",
    "WorkerState",
    "validate_transition",
]
