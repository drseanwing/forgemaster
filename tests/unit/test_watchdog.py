"""Unit tests for watchdog idle session monitoring.

This module tests all components of the watchdog system including:
- ActivityTracker: Recording and retrieving activity state
- IdleDetector: Classifying sessions by idle severity
- IdleWatchdog: Monitoring loop and corrective actions
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forgemaster.agents.session import HealthStatus, SessionInfo, SessionMetrics, SessionState
from forgemaster.config import AgentConfig
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


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent config with short timeouts."""
    return AgentConfig(
        idle_timeout_seconds=300,  # 5 minutes
        session_timeout_seconds=1800,  # 30 minutes
        max_retries=3,
    )


@pytest.fixture
def activity_tracker() -> ActivityTracker:
    """Create fresh activity tracker."""
    return ActivityTracker()


@pytest.fixture
def mock_session_manager() -> AsyncMock:
    """Create mock session manager."""
    manager = AsyncMock()
    manager.list_sessions = MagicMock(return_value=[])
    manager.end_session = AsyncMock(return_value={"session_id": "test-session", "status": "failed"})
    manager.send_message = AsyncMock(return_value="Acknowledged")
    return manager


@pytest.fixture
def idle_detector(agent_config: AgentConfig, activity_tracker: ActivityTracker) -> IdleDetector:
    """Create idle detector."""
    return IdleDetector(agent_config, activity_tracker)


@pytest.fixture
def idle_watchdog(
    agent_config: AgentConfig,
    activity_tracker: ActivityTracker,
    mock_session_manager: AsyncMock,
) -> IdleWatchdog:
    """Create idle watchdog."""
    return IdleWatchdog(agent_config, activity_tracker, mock_session_manager, check_interval=1)


# =====================================================================
# ActivityTracker Tests
# =====================================================================


def test_activity_tracker_record_new_session(activity_tracker: ActivityTracker) -> None:
    """Test recording activity for a new session."""
    activity_tracker.record_activity("sess-1", ActivityType.SESSION_START, token_delta=0)

    record = activity_tracker.get_last_activity("sess-1")
    assert record is not None
    assert record.session_id == "sess-1"
    assert record.activity_type == ActivityType.SESSION_START
    assert record.message_count == 1
    assert record.token_delta == 0


def test_activity_tracker_record_update_existing(activity_tracker: ActivityTracker) -> None:
    """Test updating activity for existing session."""
    activity_tracker.record_activity("sess-1", ActivityType.SESSION_START)
    activity_tracker.record_activity("sess-1", ActivityType.MESSAGE_SENT, token_delta=100)

    record = activity_tracker.get_last_activity("sess-1")
    assert record is not None
    assert record.activity_type == ActivityType.MESSAGE_SENT
    assert record.message_count == 2
    assert record.token_delta == 100


def test_activity_tracker_get_unknown_session(activity_tracker: ActivityTracker) -> None:
    """Test getting activity for unknown session."""
    record = activity_tracker.get_last_activity("unknown")
    assert record is None


def test_activity_tracker_idle_duration_new_session(activity_tracker: ActivityTracker) -> None:
    """Test idle duration for newly recorded session."""
    activity_tracker.record_activity("sess-1", ActivityType.SESSION_START)
    idle_duration = activity_tracker.get_idle_duration_seconds("sess-1")

    # Should be close to 0 (within 1 second tolerance)
    assert idle_duration < 1.0


def test_activity_tracker_idle_duration_old_activity(activity_tracker: ActivityTracker) -> None:
    """Test idle duration for session with old activity."""
    # Manually inject old timestamp
    old_time = datetime.now(timezone.utc) - timedelta(seconds=150)
    record = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
        token_delta=0,
    )
    activity_tracker._activities["sess-1"] = record

    idle_duration = activity_tracker.get_idle_duration_seconds("sess-1")

    # Should be around 150 seconds (with small tolerance)
    assert 149 < idle_duration < 151


def test_activity_tracker_idle_duration_unknown_session(activity_tracker: ActivityTracker) -> None:
    """Test idle duration for unknown session returns 0."""
    idle_duration = activity_tracker.get_idle_duration_seconds("unknown")
    assert idle_duration == 0.0


def test_activity_tracker_clear_session(activity_tracker: ActivityTracker) -> None:
    """Test clearing session tracking."""
    activity_tracker.record_activity("sess-1", ActivityType.SESSION_START)
    assert activity_tracker.get_last_activity("sess-1") is not None

    activity_tracker.clear_session("sess-1")
    assert activity_tracker.get_last_activity("sess-1") is None


def test_activity_tracker_clear_unknown_session(activity_tracker: ActivityTracker) -> None:
    """Test clearing unknown session is safe."""
    activity_tracker.clear_session("unknown")  # Should not raise


# =====================================================================
# ActivityType Enum Tests
# =====================================================================


def test_activity_type_enum_values() -> None:
    """Test ActivityType enum has expected values."""
    assert ActivityType.MESSAGE_SENT.value == "message_sent"
    assert ActivityType.MESSAGE_RECEIVED.value == "message_received"
    assert ActivityType.TOOL_CALL.value == "tool_call"
    assert ActivityType.HEARTBEAT.value == "heartbeat"
    assert ActivityType.SESSION_START.value == "session_start"


# =====================================================================
# IdleDetector Tests
# =====================================================================


def test_idle_detector_no_idle_sessions(idle_detector: IdleDetector) -> None:
    """Test detector with no idle sessions."""
    sessions = [
        SessionInfo(
            session_id="sess-1",
            task_id="task-1",
            agent_type="executor",
            model="sonnet",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        )
    ]

    # Record recent activity
    idle_detector.activity_tracker.record_activity("sess-1", ActivityType.MESSAGE_SENT)

    idle_sessions = idle_detector.detect_idle_sessions(sessions)
    assert len(idle_sessions) == 0


def test_idle_detector_warning_severity(
    idle_detector: IdleDetector, activity_tracker: ActivityTracker
) -> None:
    """Test detector identifies WARNING severity (50-80% of threshold)."""
    sessions = [
        SessionInfo(
            session_id="sess-1",
            task_id="task-1",
            agent_type="executor",
            model="sonnet",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        )
    ]

    # Inject activity at 60% of threshold (300s * 0.6 = 180s)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=180)
    record = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )
    activity_tracker._activities["sess-1"] = record

    idle_sessions = idle_detector.detect_idle_sessions(sessions)
    assert len(idle_sessions) == 1
    assert idle_sessions[0].severity == IdleSeverity.WARNING
    assert idle_sessions[0].session_id == "sess-1"
    assert idle_sessions[0].idle_seconds >= 179  # Allow small tolerance


def test_idle_detector_critical_severity(
    idle_detector: IdleDetector, activity_tracker: ActivityTracker
) -> None:
    """Test detector identifies CRITICAL severity (80-100% of threshold)."""
    sessions = [
        SessionInfo(
            session_id="sess-1",
            task_id="task-1",
            agent_type="executor",
            model="sonnet",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        )
    ]

    # Inject activity at 90% of threshold (300s * 0.9 = 270s)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=270)
    record = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )
    activity_tracker._activities["sess-1"] = record

    idle_sessions = idle_detector.detect_idle_sessions(sessions)
    assert len(idle_sessions) == 1
    assert idle_sessions[0].severity == IdleSeverity.CRITICAL
    assert idle_sessions[0].idle_seconds >= 269


def test_idle_detector_terminal_severity(
    idle_detector: IdleDetector, activity_tracker: ActivityTracker
) -> None:
    """Test detector identifies TERMINAL severity (>100% of threshold)."""
    sessions = [
        SessionInfo(
            session_id="sess-1",
            task_id="task-1",
            agent_type="executor",
            model="sonnet",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        )
    ]

    # Inject activity at 120% of threshold (300s * 1.2 = 360s)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=360)
    record = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )
    activity_tracker._activities["sess-1"] = record

    idle_sessions = idle_detector.detect_idle_sessions(sessions)
    assert len(idle_sessions) == 1
    assert idle_sessions[0].severity == IdleSeverity.TERMINAL
    assert idle_sessions[0].idle_seconds >= 359


def test_idle_detector_multiple_sessions(
    idle_detector: IdleDetector, activity_tracker: ActivityTracker
) -> None:
    """Test detector with multiple sessions at different severities."""
    sessions = [
        SessionInfo(
            session_id="sess-healthy",
            task_id="task-1",
            agent_type="executor",
            model="sonnet",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        ),
        SessionInfo(
            session_id="sess-warning",
            task_id="task-2",
            agent_type="executor",
            model="sonnet",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        ),
        SessionInfo(
            session_id="sess-terminal",
            task_id="task-3",
            agent_type="executor",
            model="sonnet",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        ),
    ]

    # Healthy: recent activity
    activity_tracker.record_activity("sess-healthy", ActivityType.MESSAGE_SENT)

    # Warning: 60% threshold
    old_time_warning = datetime.now(timezone.utc) - timedelta(seconds=180)
    activity_tracker._activities["sess-warning"] = ActivityRecord(
        session_id="sess-warning",
        last_activity_at=old_time_warning.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )

    # Terminal: 120% threshold
    old_time_terminal = datetime.now(timezone.utc) - timedelta(seconds=360)
    activity_tracker._activities["sess-terminal"] = ActivityRecord(
        session_id="sess-terminal",
        last_activity_at=old_time_terminal.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )

    idle_sessions = idle_detector.detect_idle_sessions(sessions)
    assert len(idle_sessions) == 2

    # Find sessions by ID
    warning_session = next(s for s in idle_sessions if s.session_id == "sess-warning")
    terminal_session = next(s for s in idle_sessions if s.session_id == "sess-terminal")

    assert warning_session.severity == IdleSeverity.WARNING
    assert terminal_session.severity == IdleSeverity.TERMINAL


def test_idle_detector_empty_session_list(idle_detector: IdleDetector) -> None:
    """Test detector with empty session list."""
    idle_sessions = idle_detector.detect_idle_sessions([])
    assert len(idle_sessions) == 0


# =====================================================================
# IdleSeverity Tests
# =====================================================================


def test_idle_severity_enum_values() -> None:
    """Test IdleSeverity enum has expected values."""
    assert IdleSeverity.WARNING.value == "warning"
    assert IdleSeverity.CRITICAL.value == "critical"
    assert IdleSeverity.TERMINAL.value == "terminal"


# =====================================================================
# IdleWatchdog Tests
# =====================================================================


@pytest.mark.asyncio
async def test_watchdog_check_sessions_no_idle(idle_watchdog: IdleWatchdog) -> None:
    """Test watchdog check with no idle sessions."""
    # Create healthy session
    session = SessionInfo(
        session_id="sess-1",
        task_id="task-1",
        agent_type="executor",
        model="sonnet",
        state=SessionState.ACTIVE,
        health=HealthStatus.HEALTHY,
        metrics=SessionMetrics(),
    )

    idle_watchdog.session_manager.list_sessions = MagicMock(return_value=[session])
    idle_watchdog.activity_tracker.record_activity("sess-1", ActivityType.MESSAGE_SENT)

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_watchdog_check_sessions_warning_only_logs(
    idle_watchdog: IdleWatchdog, activity_tracker: ActivityTracker
) -> None:
    """Test watchdog logs WARNING severity but doesn't create action."""
    session = SessionInfo(
        session_id="sess-1",
        task_id="task-1",
        agent_type="executor",
        model="sonnet",
        state=SessionState.ACTIVE,
        health=HealthStatus.HEALTHY,
        metrics=SessionMetrics(),
    )

    idle_watchdog.session_manager.list_sessions = MagicMock(return_value=[session])

    # Set warning level idle (60% threshold)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=180)
    activity_tracker._activities["sess-1"] = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 0  # WARNING only logs, no action


@pytest.mark.asyncio
async def test_watchdog_check_sessions_critical_warns(
    idle_watchdog: IdleWatchdog, activity_tracker: ActivityTracker, mock_session_manager: AsyncMock
) -> None:
    """Test watchdog sends warning for CRITICAL severity."""
    session = SessionInfo(
        session_id="sess-1",
        task_id="task-1",
        agent_type="executor",
        model="sonnet",
        state=SessionState.ACTIVE,
        health=HealthStatus.HEALTHY,
        metrics=SessionMetrics(),
    )

    mock_session_manager.list_sessions = MagicMock(return_value=[session])

    # Set critical level idle (90% threshold)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=270)
    activity_tracker._activities["sess-1"] = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 1
    assert actions[0].action == WatchdogActionType.WARN
    assert actions[0].success is True
    assert actions[0].session_id == "sess-1"

    # Verify warning message was sent
    mock_session_manager.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_watchdog_check_sessions_terminal_kills(
    idle_watchdog: IdleWatchdog, activity_tracker: ActivityTracker, mock_session_manager: AsyncMock
) -> None:
    """Test watchdog kills session at TERMINAL severity."""
    session = SessionInfo(
        session_id="sess-1",
        task_id="task-1",
        agent_type="executor",
        model="sonnet",
        state=SessionState.ACTIVE,
        health=HealthStatus.HEALTHY,
        metrics=SessionMetrics(),
    )

    mock_session_manager.list_sessions = MagicMock(return_value=[session])

    # Set terminal level idle (120% threshold)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=360)
    activity_tracker._activities["sess-1"] = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 1
    assert actions[0].action == WatchdogActionType.KILL
    assert actions[0].success is True
    assert actions[0].session_id == "sess-1"

    # Verify session was ended
    mock_session_manager.end_session.assert_called_once_with(
        session_id="sess-1",
        status="failed",
    )

    # Verify activity tracking was cleared
    assert activity_tracker.get_last_activity("sess-1") is None


@pytest.mark.asyncio
async def test_watchdog_start_stop_lifecycle(idle_watchdog: IdleWatchdog) -> None:
    """Test watchdog start/stop lifecycle."""
    assert not idle_watchdog._running

    await idle_watchdog.start()
    assert idle_watchdog._running
    assert idle_watchdog._watchdog_task is not None

    await idle_watchdog.stop()
    assert not idle_watchdog._running


@pytest.mark.asyncio
async def test_watchdog_start_idempotent(idle_watchdog: IdleWatchdog) -> None:
    """Test starting watchdog multiple times is safe."""
    await idle_watchdog.start()
    task1 = idle_watchdog._watchdog_task

    await idle_watchdog.start()  # Should be no-op
    task2 = idle_watchdog._watchdog_task

    assert task1 is task2  # Same task

    await idle_watchdog.stop()


@pytest.mark.asyncio
async def test_watchdog_stop_idempotent(idle_watchdog: IdleWatchdog) -> None:
    """Test stopping watchdog when not running is safe."""
    assert not idle_watchdog._running

    await idle_watchdog.stop()  # Should be no-op
    assert not idle_watchdog._running


@pytest.mark.asyncio
async def test_watchdog_monitoring_loop_runs(
    idle_watchdog: IdleWatchdog, mock_session_manager: AsyncMock
) -> None:
    """Test watchdog monitoring loop executes checks."""
    mock_session_manager.list_sessions = MagicMock(return_value=[])

    await idle_watchdog.start()
    await asyncio.sleep(0.1)  # Let loop run briefly
    await idle_watchdog.stop()

    # Should have checked sessions at least once
    mock_session_manager.list_sessions.assert_called()


@pytest.mark.asyncio
async def test_watchdog_kill_session_error_handling(
    idle_watchdog: IdleWatchdog, activity_tracker: ActivityTracker, mock_session_manager: AsyncMock
) -> None:
    """Test watchdog handles errors when killing session."""
    session = SessionInfo(
        session_id="sess-1",
        task_id="task-1",
        agent_type="executor",
        model="sonnet",
        state=SessionState.ACTIVE,
        health=HealthStatus.HEALTHY,
        metrics=SessionMetrics(),
    )

    mock_session_manager.list_sessions = MagicMock(return_value=[session])
    mock_session_manager.end_session = AsyncMock(side_effect=RuntimeError("Kill failed"))

    # Set terminal level idle
    old_time = datetime.now(timezone.utc) - timedelta(seconds=360)
    activity_tracker._activities["sess-1"] = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 1
    assert actions[0].action == WatchdogActionType.KILL
    assert actions[0].success is False
    assert actions[0].error == "Kill failed"


@pytest.mark.asyncio
async def test_watchdog_warn_session_error_handling(
    idle_watchdog: IdleWatchdog, activity_tracker: ActivityTracker, mock_session_manager: AsyncMock
) -> None:
    """Test watchdog handles errors when warning session."""
    session = SessionInfo(
        session_id="sess-1",
        task_id="task-1",
        agent_type="executor",
        model="sonnet",
        state=SessionState.ACTIVE,
        health=HealthStatus.HEALTHY,
        metrics=SessionMetrics(),
    )

    mock_session_manager.list_sessions = MagicMock(return_value=[session])
    mock_session_manager.send_message = AsyncMock(side_effect=RuntimeError("Message failed"))

    # Set critical level idle
    old_time = datetime.now(timezone.utc) - timedelta(seconds=270)
    activity_tracker._activities["sess-1"] = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 1
    assert actions[0].action == WatchdogActionType.WARN
    assert actions[0].success is False
    assert actions[0].error == "Message failed"


# =====================================================================
# WatchdogAction Model Tests
# =====================================================================


def test_watchdog_action_model_validation() -> None:
    """Test WatchdogAction model validates correctly."""
    action = WatchdogAction(
        session_id="sess-1",
        task_id="task-1",
        action=WatchdogActionType.KILL,
        reason="Idle timeout",
        success=True,
        error=None,
        duration_seconds=0.5,
    )

    assert action.session_id == "sess-1"
    assert action.task_id == "task-1"
    assert action.action == WatchdogActionType.KILL
    assert action.reason == "Idle timeout"
    assert action.success is True
    assert action.error is None
    assert action.duration_seconds == 0.5


def test_watchdog_action_model_with_error() -> None:
    """Test WatchdogAction model with error."""
    action = WatchdogAction(
        session_id="sess-1",
        task_id="task-1",
        action=WatchdogActionType.WARN,
        reason="Critical idle",
        success=False,
        error="Connection timeout",
        duration_seconds=1.0,
    )

    assert action.success is False
    assert action.error == "Connection timeout"


# =====================================================================
# Edge Cases
# =====================================================================


@pytest.mark.asyncio
async def test_watchdog_no_sessions(
    idle_watchdog: IdleWatchdog, mock_session_manager: AsyncMock
) -> None:
    """Test watchdog with no active sessions."""
    mock_session_manager.list_sessions = MagicMock(return_value=[])

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_watchdog_all_healthy_sessions(
    idle_watchdog: IdleWatchdog, mock_session_manager: AsyncMock, activity_tracker: ActivityTracker
) -> None:
    """Test watchdog with all healthy sessions."""
    sessions = [
        SessionInfo(
            session_id="sess-1",
            task_id="task-1",
            agent_type="executor",
            model="sonnet",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        ),
        SessionInfo(
            session_id="sess-2",
            task_id="task-2",
            agent_type="architect",
            model="opus",
            state=SessionState.ACTIVE,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        ),
    ]

    mock_session_manager.list_sessions = MagicMock(return_value=sessions)

    # Record recent activity for all
    activity_tracker.record_activity("sess-1", ActivityType.MESSAGE_SENT)
    activity_tracker.record_activity("sess-2", ActivityType.MESSAGE_SENT)

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_watchdog_session_ended_between_detection_and_action(
    idle_watchdog: IdleWatchdog, activity_tracker: ActivityTracker, mock_session_manager: AsyncMock
) -> None:
    """Test watchdog handles session ending between detection and action."""
    session = SessionInfo(
        session_id="sess-1",
        task_id="task-1",
        agent_type="executor",
        model="sonnet",
        state=SessionState.ACTIVE,
        health=HealthStatus.HEALTHY,
        metrics=SessionMetrics(),
    )

    mock_session_manager.list_sessions = MagicMock(return_value=[session])
    # Session not found when trying to end it
    mock_session_manager.end_session = AsyncMock(side_effect=ValueError("Session not found"))

    # Set terminal level idle
    old_time = datetime.now(timezone.utc) - timedelta(seconds=360)
    activity_tracker._activities["sess-1"] = ActivityRecord(
        session_id="sess-1",
        last_activity_at=old_time.isoformat(),
        activity_type=ActivityType.MESSAGE_SENT,
        message_count=1,
    )

    actions = await idle_watchdog.check_sessions()
    assert len(actions) == 1
    assert actions[0].success is False
    assert "Session not found" in actions[0].error
