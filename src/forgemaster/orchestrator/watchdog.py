"""Idle session watchdog for Forgemaster.

This module provides activity tracking, idle session detection, and automated session
killing for agent sessions that exceed idle timeout thresholds. The watchdog monitors
sessions continuously and takes corrective action (warning, killing) based on severity.

The watchdog complements HealthMonitor by focusing specifically on activity-based
idle detection, providing finer-grained tracking of session activity types and
severity-based response escalation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from forgemaster.config import AgentConfig

if TYPE_CHECKING:
    from forgemaster.agents.session import AgentSessionManager, SessionInfo

logger = structlog.get_logger(__name__)


class ActivityType(str, Enum):
    """Types of session activity that reset idle timers.

    These activity types are tracked to determine when a session last had
    meaningful interaction, helping detect truly idle vs. actively working sessions.
    """

    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    TOOL_CALL = "tool_call"
    HEARTBEAT = "heartbeat"
    SESSION_START = "session_start"


class IdleSeverity(str, Enum):
    """Severity levels for idle session classification.

    Severity thresholds:
    - WARNING: idle > 50% of threshold
    - CRITICAL: idle > 80% of threshold
    - TERMINAL: idle > 100% of threshold
    """

    WARNING = "warning"
    CRITICAL = "critical"
    TERMINAL = "terminal"


class WatchdogActionType(str, Enum):
    """Actions the watchdog can take on sessions."""

    WARN = "warn"
    KILL = "kill"
    RETRY = "retry"


class ActivityRecord(BaseModel):
    """Record of a session's activity state.

    Attributes:
        session_id: Unique session identifier
        last_activity_at: ISO timestamp of most recent activity
        activity_type: Type of the last activity
        message_count: Total messages exchanged in session
        token_delta: Tokens used in last activity
    """

    session_id: str = Field(description="Session identifier")
    last_activity_at: str = Field(description="ISO timestamp of last activity")
    activity_type: ActivityType = Field(description="Type of last activity")
    message_count: int = Field(default=0, description="Total message count")
    token_delta: int = Field(default=0, description="Tokens in last activity")


class IdleSession(BaseModel):
    """Information about an idle session.

    Attributes:
        session_id: Session identifier
        task_id: Associated task identifier
        idle_seconds: Seconds since last activity
        threshold_seconds: Configured idle timeout threshold
        severity: Severity level of idle state
    """

    session_id: str = Field(description="Session identifier")
    task_id: str = Field(description="Associated task ID")
    idle_seconds: float = Field(description="Seconds idle")
    threshold_seconds: float = Field(description="Idle timeout threshold")
    severity: IdleSeverity = Field(description="Idle severity level")


class WatchdogAction(BaseModel):
    """Record of a watchdog action taken.

    Attributes:
        session_id: Session identifier
        task_id: Associated task identifier
        action: Type of action taken
        reason: Human-readable reason for action
        success: Whether action succeeded
        error: Error message if action failed
        duration_seconds: Time taken to execute action
    """

    session_id: str = Field(description="Session identifier")
    task_id: str = Field(description="Associated task ID")
    action: WatchdogActionType = Field(description="Action type")
    reason: str = Field(description="Reason for action")
    success: bool = Field(description="Whether action succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_seconds: float = Field(default=0.0, description="Action duration")


class ActivityTracker:
    """Tracks activity timestamps for sessions.

    This class maintains in-memory tracking of session activity to detect
    idle sessions more accurately than database-only tracking allows.
    """

    def __init__(self) -> None:
        """Initialize activity tracker with empty state."""
        self._activities: dict[str, ActivityRecord] = {}
        self._logger = structlog.get_logger(__name__)

    def record_activity(
        self, session_id: str, activity_type: ActivityType, token_delta: int = 0
    ) -> None:
        """Record activity for a session.

        Args:
            session_id: Session identifier
            activity_type: Type of activity
            token_delta: Tokens used in this activity
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        if session_id in self._activities:
            record = self._activities[session_id]
            record.last_activity_at = now_iso
            record.activity_type = activity_type
            record.message_count += 1
            record.token_delta = token_delta
        else:
            record = ActivityRecord(
                session_id=session_id,
                last_activity_at=now_iso,
                activity_type=activity_type,
                message_count=1,
                token_delta=token_delta,
            )
            self._activities[session_id] = record

        self._logger.debug(
            "activity_recorded",
            session_id=session_id,
            activity_type=activity_type.value,
            token_delta=token_delta,
        )

    def get_last_activity(self, session_id: str) -> ActivityRecord | None:
        """Get the last activity record for a session.

        Args:
            session_id: Session identifier

        Returns:
            ActivityRecord if found, None otherwise
        """
        return self._activities.get(session_id)

    def get_idle_duration_seconds(self, session_id: str) -> float:
        """Calculate seconds since last activity for a session.

        Args:
            session_id: Session identifier

        Returns:
            Seconds since last activity, or 0.0 if session not tracked
        """
        record = self._activities.get(session_id)
        if record is None:
            return 0.0

        last_activity = datetime.fromisoformat(record.last_activity_at)
        now = datetime.now(timezone.utc)
        return (now - last_activity).total_seconds()

    def clear_session(self, session_id: str) -> None:
        """Remove tracking for an ended session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._activities:
            del self._activities[session_id]
            self._logger.debug("activity_tracking_cleared", session_id=session_id)


class IdleDetector:
    """Detects idle sessions based on activity tracking.

    This class compares session idle times against configured thresholds
    and classifies sessions into severity levels.
    """

    def __init__(self, config: AgentConfig, activity_tracker: ActivityTracker):
        """Initialize idle detector.

        Args:
            config: Agent configuration with idle timeout threshold
            activity_tracker: Activity tracker for session state
        """
        self.config = config
        self.activity_tracker = activity_tracker
        self._logger = structlog.get_logger(__name__)

    def detect_idle_sessions(self, sessions: list[SessionInfo]) -> list[IdleSession]:
        """Detect idle sessions and classify by severity.

        Args:
            sessions: List of session info to check

        Returns:
            List of idle sessions at WARNING or above severity
        """
        idle_sessions: list[IdleSession] = []
        threshold = float(self.config.idle_timeout_seconds)

        for session in sessions:
            # Get idle duration from activity tracker
            idle_seconds = self.activity_tracker.get_idle_duration_seconds(session.session_id)

            # Calculate severity
            severity = self._calculate_severity(idle_seconds, threshold)

            # Only include sessions at WARNING or above
            if severity is not None:
                idle_session = IdleSession(
                    session_id=session.session_id,
                    task_id=session.task_id,
                    idle_seconds=idle_seconds,
                    threshold_seconds=threshold,
                    severity=severity,
                )
                idle_sessions.append(idle_session)

                self._logger.debug(
                    "idle_session_detected",
                    session_id=session.session_id,
                    task_id=session.task_id,
                    idle_seconds=idle_seconds,
                    severity=severity.value,
                )

        return idle_sessions

    def _calculate_severity(self, idle_seconds: float, threshold: float) -> IdleSeverity | None:
        """Calculate severity level based on idle time.

        Args:
            idle_seconds: Seconds since last activity
            threshold: Configured idle timeout threshold

        Returns:
            IdleSeverity if idle exceeds 50% of threshold, None otherwise
        """
        idle_ratio = idle_seconds / threshold

        if idle_ratio >= 1.0:
            return IdleSeverity.TERMINAL
        elif idle_ratio >= 0.8:
            return IdleSeverity.CRITICAL
        elif idle_ratio >= 0.5:
            return IdleSeverity.WARNING

        return None


class IdleWatchdog:
    """Idle session watchdog service.

    The watchdog continuously monitors active sessions for idle timeout issues
    and takes corrective action based on severity:
    - WARNING: Log only
    - CRITICAL: Send warning message to agent (nudge)
    - TERMINAL: Kill session and schedule retry

    This complements HealthMonitor by providing activity-based idle detection
    rather than relying solely on database timestamps.
    """

    def __init__(
        self,
        config: AgentConfig,
        activity_tracker: ActivityTracker,
        session_manager: AgentSessionManager,
        check_interval: int = 30,
    ):
        """Initialize idle watchdog.

        Args:
            config: Agent configuration with timeouts
            activity_tracker: Activity tracker for session state
            session_manager: Session manager for lifecycle operations
            check_interval: Seconds between watchdog checks (default: 30)
        """
        self.config = config
        self.activity_tracker = activity_tracker
        self.session_manager = session_manager
        self.check_interval = check_interval
        self.idle_detector = IdleDetector(config, activity_tracker)
        self._running = False
        self._watchdog_task: asyncio.Task[None] | None = None
        self._logger = structlog.get_logger(__name__)

    async def start(self) -> None:
        """Start the watchdog monitoring loop.

        Creates and starts the background monitoring task. If already running,
        this is a no-op.
        """
        if self._running:
            self._logger.warning("watchdog_already_running")
            return

        self._running = True
        self._watchdog_task = asyncio.create_task(self._monitoring_loop())
        self._logger.info(
            "watchdog_started",
            check_interval=self.check_interval,
            idle_timeout=self.config.idle_timeout_seconds,
        )

    async def stop(self) -> None:
        """Gracefully stop the watchdog monitoring loop.

        Cancels the background task and waits for clean shutdown.
        """
        if not self._running:
            self._logger.warning("watchdog_not_running")
            return

        self._running = False

        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass

        self._logger.info("watchdog_stopped")

    async def check_sessions(self) -> list[WatchdogAction]:
        """Check all active sessions and take action on idle ones.

        Returns:
            List of actions taken
        """
        actions: list[WatchdogAction] = []

        # Get all active sessions
        sessions = self.session_manager.list_sessions()
        active_sessions = [s for s in sessions if s.state.value in ("initializing", "active")]

        if not active_sessions:
            return actions

        # Detect idle sessions
        idle_sessions = self.idle_detector.detect_idle_sessions(active_sessions)

        # Take action based on severity
        for idle_session in idle_sessions:
            if idle_session.severity == IdleSeverity.TERMINAL:
                # Kill session and schedule retry
                action = await self._kill_session(idle_session)
                actions.append(action)
            elif idle_session.severity == IdleSeverity.CRITICAL:
                # Send warning message (nudge)
                action = await self._warn_session(idle_session)
                actions.append(action)
            elif idle_session.severity == IdleSeverity.WARNING:
                # Log only
                self._logger.warning(
                    "session_idle_warning",
                    session_id=idle_session.session_id,
                    task_id=idle_session.task_id,
                    idle_seconds=idle_session.idle_seconds,
                    threshold=idle_session.threshold_seconds,
                )

        return actions

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop that periodically checks sessions.

        This loop runs until stop() is called, checking all active sessions
        at each interval and taking corrective action as needed.
        """
        self._logger.info("watchdog_loop_started")

        while self._running:
            try:
                await self.check_sessions()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                self._logger.info("watchdog_loop_cancelled")
                break
            except Exception as e:
                self._logger.error(
                    "watchdog_loop_error",
                    error=str(e),
                    exc_info=True,
                )
                # Continue monitoring despite errors
                await asyncio.sleep(self.check_interval)

    async def _kill_session(self, idle_session: IdleSession) -> WatchdogAction:
        """Kill an idle session and schedule retry.

        Args:
            idle_session: Idle session information

        Returns:
            WatchdogAction recording the result
        """
        start_time = datetime.now(timezone.utc)
        reason = (
            f"Terminal idle timeout: {idle_session.idle_seconds:.1f}s "
            f"exceeds {idle_session.threshold_seconds:.1f}s"
        )

        self._logger.warning(
            "killing_idle_session",
            session_id=idle_session.session_id,
            task_id=idle_session.task_id,
            idle_seconds=idle_session.idle_seconds,
            reason=reason,
        )

        try:
            # End the session
            await self.session_manager.end_session(
                session_id=idle_session.session_id,
                status="failed",
            )

            # Clear activity tracking
            self.activity_tracker.clear_session(idle_session.session_id)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            self._logger.info(
                "idle_session_killed",
                session_id=idle_session.session_id,
                task_id=idle_session.task_id,
                duration_seconds=duration,
            )

            return WatchdogAction(
                session_id=idle_session.session_id,
                task_id=idle_session.task_id,
                action=WatchdogActionType.KILL,
                reason=reason,
                success=True,
                error=None,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = str(e)

            self._logger.error(
                "failed_to_kill_idle_session",
                session_id=idle_session.session_id,
                task_id=idle_session.task_id,
                error=error_msg,
                exc_info=True,
            )

            return WatchdogAction(
                session_id=idle_session.session_id,
                task_id=idle_session.task_id,
                action=WatchdogActionType.KILL,
                reason=reason,
                success=False,
                error=error_msg,
                duration_seconds=duration,
            )

    async def _warn_session(self, idle_session: IdleSession) -> WatchdogAction:
        """Send warning message to an idle session (nudge it).

        Args:
            idle_session: Idle session information

        Returns:
            WatchdogAction recording the result
        """
        start_time = datetime.now(timezone.utc)
        reason = (
            f"Critical idle timeout: {idle_session.idle_seconds:.1f}s "
            f"approaching threshold {idle_session.threshold_seconds:.1f}s"
        )

        self._logger.warning(
            "warning_idle_session",
            session_id=idle_session.session_id,
            task_id=idle_session.task_id,
            idle_seconds=idle_session.idle_seconds,
        )

        try:
            # Send nudge message to agent
            nudge_message = (
                f"WATCHDOG: You have been idle for {idle_session.idle_seconds:.1f} seconds. "
                f"Please respond to avoid session termination at {idle_session.threshold_seconds:.1f}s."
            )

            await self.session_manager.send_message(
                session_id=idle_session.session_id,
                message=nudge_message,
            )

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            self._logger.info(
                "idle_session_warned",
                session_id=idle_session.session_id,
                task_id=idle_session.task_id,
                duration_seconds=duration,
            )

            return WatchdogAction(
                session_id=idle_session.session_id,
                task_id=idle_session.task_id,
                action=WatchdogActionType.WARN,
                reason=reason,
                success=True,
                error=None,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = str(e)

            self._logger.error(
                "failed_to_warn_idle_session",
                session_id=idle_session.session_id,
                task_id=idle_session.task_id,
                error=error_msg,
                exc_info=True,
            )

            return WatchdogAction(
                session_id=idle_session.session_id,
                task_id=idle_session.task_id,
                action=WatchdogActionType.WARN,
                reason=reason,
                success=False,
                error=error_msg,
                duration_seconds=duration,
            )
