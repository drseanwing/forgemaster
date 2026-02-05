"""Orchestrator subsystem for Forgemaster.

This module implements the task dispatcher, state machine, session health
monitor, and parallel worker coordination logic.
"""

from forgemaster.orchestrator.dispatcher import Dispatcher
from forgemaster.orchestrator.health_monitor import HealthMonitor
from forgemaster.orchestrator.result_handler import ResultHandler
from forgemaster.orchestrator.state_machine import (
    InvalidTransitionError,
    TaskStateMachine,
    VALID_TRANSITIONS,
    validate_transition,
)

__all__ = [
    "Dispatcher",
    "HealthMonitor",
    "InvalidTransitionError",
    "ResultHandler",
    "TaskStateMachine",
    "VALID_TRANSITIONS",
    "validate_transition",
]
