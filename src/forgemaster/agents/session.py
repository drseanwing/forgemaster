"""Agent session lifecycle management for Forgemaster.

This module provides session management, health monitoring, and token tracking
for Claude agent sessions. It tracks session state transitions, monitors for
idle or stuck sessions, and provides token usage statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from forgemaster.agents.sdk_wrapper import AgentClient, AgentSession
from forgemaster.config import AgentConfig


class SessionState(str, Enum):
    """Session lifecycle states.

    State transitions:
        INITIALIZING → ACTIVE → COMPLETING → COMPLETED
                              ↓
                            FAILED
    """

    INITIALIZING = "initializing"
    ACTIVE = "active"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthStatus(str, Enum):
    """Session health status indicators."""

    HEALTHY = "healthy"
    IDLE = "idle"
    STUCK = "stuck"
    CONTEXT_WARNING = "context_warning"
    FAILED = "failed"


@dataclass
class SessionMetrics:
    """Token usage and performance metrics for a session.

    Attributes:
        input_tokens: Cumulative input tokens sent to the agent
        output_tokens: Cumulative output tokens received from the agent
        total_tokens: Sum of input and output tokens
        messages_sent: Number of messages sent in this session
        created_at: Session creation timestamp
        last_activity_at: Timestamp of most recent activity
        duration_seconds: Total session duration in seconds
    """

    input_tokens: int = 0
    output_tokens: int = 0
    messages_sent: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output).

        Returns:
            Sum of input and output tokens
        """
        return self.input_tokens + self.output_tokens

    @property
    def duration_seconds(self) -> float:
        """Session duration in seconds.

        Returns:
            Time elapsed since session creation
        """
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Time since last activity in seconds.

        Returns:
            Time elapsed since last activity
        """
        now = datetime.now(timezone.utc)
        return (now - self.last_activity_at).total_seconds()


@dataclass
class SessionInfo:
    """Complete session information including state and metrics.

    Attributes:
        session_id: Unique session identifier
        task_id: Associated task identifier
        agent_type: Type of agent (e.g., "architect", "executor")
        model: Claude model being used
        state: Current session lifecycle state
        health: Current health status
        metrics: Token usage and performance metrics
        sdk_session: Underlying SDK session object
    """

    session_id: str
    task_id: str
    agent_type: str
    model: str
    state: SessionState
    health: HealthStatus
    metrics: SessionMetrics
    sdk_session: AgentSession | None = None


class AgentSessionManager:
    """Manages agent session lifecycle, health, and metrics.

    This class provides the primary interface for creating, monitoring, and
    terminating agent sessions. It tracks session state, monitors health,
    and collects token usage metrics.

    Attributes:
        config: Agent configuration settings
        agent_client: Underlying Claude SDK client
    """

    def __init__(self, config: AgentConfig, agent_client: AgentClient):
        """Initialize session manager.

        Args:
            config: Agent configuration for timeouts and thresholds
            agent_client: Initialized Claude SDK client
        """
        self.config = config
        self.agent_client = agent_client
        self._sessions: dict[str, SessionInfo] = {}
        self._logger = structlog.get_logger(__name__)

    async def start_session(
        self,
        task_id: str,
        agent_type: str,
        model: str,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create and start a new agent session.

        This method creates a new session with the specified configuration,
        initializes it with the SDK, and tracks it in the session registry.

        Args:
            task_id: Task identifier this session is associated with
            agent_type: Type of agent (e.g., "architect", "executor")
            model: Claude model identifier
            system_prompt: System prompt to configure agent behavior
            tools: Optional list of tool definitions

        Returns:
            Unique session identifier

        Raises:
            RuntimeError: If SDK client is not initialized or session creation fails
        """
        session_id = f"{task_id}-{agent_type}-{int(time.time())}"

        self._logger.info(
            "session_starting",
            session_id=session_id,
            task_id=task_id,
            agent_type=agent_type,
            model=model,
        )

        # Create session info with initializing state
        session_info = SessionInfo(
            session_id=session_id,
            task_id=task_id,
            agent_type=agent_type,
            model=model,
            state=SessionState.INITIALIZING,
            health=HealthStatus.HEALTHY,
            metrics=SessionMetrics(),
        )

        self._sessions[session_id] = session_info

        try:
            # Create SDK session
            sdk_session = await self.agent_client.create_session(
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

            session_info.sdk_session = sdk_session
            session_info.state = SessionState.ACTIVE

            self._logger.info(
                "session_started",
                session_id=session_id,
                state=session_info.state.value,
            )

            return session_id

        except Exception as e:
            session_info.state = SessionState.FAILED
            session_info.health = HealthStatus.FAILED

            self._logger.error(
                "session_start_failed",
                session_id=session_id,
                error=str(e),
            )
            raise

    async def send_message(self, session_id: str, message: str) -> str:
        """Send a message to an active session and receive response.

        This method sends a message to the agent, updates activity timestamps,
        and returns the agent's response.

        Args:
            session_id: Session identifier
            message: Message text to send to the agent

        Returns:
            Agent's response text

        Raises:
            ValueError: If session does not exist
            RuntimeError: If session is not active or SDK session is invalid
        """
        session_info = self._sessions.get(session_id)
        if session_info is None:
            raise ValueError(f"Session not found: {session_id}")

        if session_info.state != SessionState.ACTIVE:
            raise RuntimeError(
                f"Session {session_id} is not active (state: {session_info.state.value})"
            )

        if session_info.sdk_session is None:
            raise RuntimeError(f"Session {session_id} has no SDK session")

        self._logger.info(
            "session_message_sending",
            session_id=session_id,
            message_length=len(message),
        )

        try:
            # Send message via SDK
            response = await session_info.sdk_session.send_message(message)

            # Update metrics
            session_info.metrics.messages_sent += 1
            session_info.metrics.last_activity_at = datetime.now(timezone.utc)

            # Re-evaluate health after activity
            session_info.health = self._evaluate_health(session_info)

            self._logger.info(
                "session_message_received",
                session_id=session_id,
                response_length=len(response),
                health=session_info.health.value,
            )

            return response

        except Exception as e:
            session_info.state = SessionState.FAILED
            session_info.health = HealthStatus.FAILED

            self._logger.error(
                "session_message_failed",
                session_id=session_id,
                error=str(e),
            )
            raise

    async def end_session(self, session_id: str, status: str = "completed") -> dict[str, Any]:
        """End a session and return final metrics.

        This method transitions the session to completing state, closes the
        underlying SDK session, and returns final metrics and results.

        Args:
            session_id: Session identifier
            status: Final status ("completed" or "failed")

        Returns:
            Dictionary containing session metrics and final state

        Raises:
            ValueError: If session does not exist
        """
        session_info = self._sessions.get(session_id)
        if session_info is None:
            raise ValueError(f"Session not found: {session_id}")

        self._logger.info(
            "session_ending",
            session_id=session_id,
            current_state=session_info.state.value,
            requested_status=status,
        )

        # Transition to completing
        session_info.state = SessionState.COMPLETING

        # Close SDK session if exists
        if session_info.sdk_session is not None:
            try:
                await session_info.sdk_session.close()
            except Exception as e:
                self._logger.warning(
                    "session_close_error",
                    session_id=session_id,
                    error=str(e),
                )

        # Set final state
        if status == "completed":
            session_info.state = SessionState.COMPLETED
        else:
            session_info.state = SessionState.FAILED

        # Build result dictionary
        result = {
            "session_id": session_id,
            "task_id": session_info.task_id,
            "agent_type": session_info.agent_type,
            "model": session_info.model,
            "state": session_info.state.value,
            "health": session_info.health.value,
            "metrics": {
                "input_tokens": session_info.metrics.input_tokens,
                "output_tokens": session_info.metrics.output_tokens,
                "total_tokens": session_info.metrics.total_tokens,
                "messages_sent": session_info.metrics.messages_sent,
                "duration_seconds": session_info.metrics.duration_seconds,
            },
        }

        self._logger.info(
            "session_ended",
            session_id=session_id,
            final_state=session_info.state.value,
            total_tokens=session_info.metrics.total_tokens,
        )

        return result

    def check_health(self, session_id: str) -> HealthStatus:
        """Check the health status of a session.

        This method evaluates session health based on:
        - Idle time (no activity for idle_timeout_seconds)
        - Stuck time (running longer than session_timeout_seconds)
        - Context usage (approaching context_warning_threshold)
        - Session state

        Args:
            session_id: Session identifier

        Returns:
            Current health status

        Raises:
            ValueError: If session does not exist
        """
        session_info = self._sessions.get(session_id)
        if session_info is None:
            raise ValueError(f"Session not found: {session_id}")

        health = self._evaluate_health(session_info)
        session_info.health = health

        self._logger.debug(
            "session_health_checked",
            session_id=session_id,
            health=health.value,
            idle_seconds=session_info.metrics.idle_seconds,
            duration_seconds=session_info.metrics.duration_seconds,
        )

        return health

    def update_token_count(
        self, session_id: str, input_tokens: int, output_tokens: int
    ) -> dict[str, Any]:
        """Update token usage for a session.

        This method updates cumulative token counts and checks if the session
        is approaching the context warning threshold.

        Args:
            session_id: Session identifier
            input_tokens: Input tokens used in recent operation
            output_tokens: Output tokens generated in recent operation

        Returns:
            Dictionary with updated token counts and warning status

        Raises:
            ValueError: If session does not exist
        """
        session_info = self._sessions.get(session_id)
        if session_info is None:
            raise ValueError(f"Session not found: {session_id}")

        # Update cumulative counts
        session_info.metrics.input_tokens += input_tokens
        session_info.metrics.output_tokens += output_tokens

        # Check for context warning (simplified - would need model context limit)
        # Using a heuristic: warn if total tokens exceed 80% of estimated 200k context
        estimated_context_limit = 200_000
        usage_ratio = session_info.metrics.total_tokens / estimated_context_limit
        at_warning_threshold = usage_ratio >= self.config.context_warning_threshold

        if at_warning_threshold and session_info.health == HealthStatus.HEALTHY:
            session_info.health = HealthStatus.CONTEXT_WARNING
            self._logger.warning(
                "session_context_warning",
                session_id=session_id,
                total_tokens=session_info.metrics.total_tokens,
                usage_ratio=usage_ratio,
            )

        return {
            "session_id": session_id,
            "input_tokens": session_info.metrics.input_tokens,
            "output_tokens": session_info.metrics.output_tokens,
            "total_tokens": session_info.metrics.total_tokens,
            "context_warning": at_warning_threshold,
            "usage_ratio": usage_ratio,
        }

    def get_session_info(self, session_id: str) -> SessionInfo:
        """Get complete session information.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo object with current state and metrics

        Raises:
            ValueError: If session does not exist
        """
        session_info = self._sessions.get(session_id)
        if session_info is None:
            raise ValueError(f"Session not found: {session_id}")
        return session_info

    def list_sessions(self) -> list[SessionInfo]:
        """List all tracked sessions.

        Returns:
            List of all SessionInfo objects
        """
        return list(self._sessions.values())

    def _evaluate_health(self, session_info: SessionInfo) -> HealthStatus:
        """Evaluate health status based on session metrics and state.

        Args:
            session_info: Session to evaluate

        Returns:
            Calculated health status
        """
        # Failed state overrides all other health checks
        if session_info.state == SessionState.FAILED:
            return HealthStatus.FAILED

        # Check for stuck session (running too long)
        if session_info.metrics.duration_seconds > self.config.session_timeout_seconds:
            return HealthStatus.STUCK

        # Check for idle session (no activity)
        if (
            session_info.state == SessionState.ACTIVE
            and session_info.metrics.idle_seconds > self.config.idle_timeout_seconds
        ):
            return HealthStatus.IDLE

        # Check for context warning
        estimated_context_limit = 200_000
        usage_ratio = session_info.metrics.total_tokens / estimated_context_limit
        if usage_ratio >= self.config.context_warning_threshold:
            return HealthStatus.CONTEXT_WARNING

        return HealthStatus.HEALTHY
