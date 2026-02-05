"""Unit tests for the session health monitor.

Tests cover idle session detection, session kill logic, retry scheduling,
and monitoring loop lifecycle management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from forgemaster.config import AgentConfig
from forgemaster.database.models.session import AgentSession, SessionStatus
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.orchestrator.health_monitor import HealthMonitor


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        max_concurrent_workers=3,
        session_timeout_seconds=1800,
        idle_timeout_seconds=300,
        max_retries=3,
        context_warning_threshold=0.8,
    )


@pytest.fixture
def mock_session_manager():
    """Create mock session manager."""
    manager = MagicMock()
    manager.end_session = AsyncMock()
    return manager


@pytest.fixture
def mock_session_factory():
    """Create mock async session factory."""
    factory = MagicMock()
    # The factory itself is callable and returns an async context manager
    mock_session = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    factory.return_value.__aexit__ = AsyncMock()
    return factory


@pytest.fixture
def health_monitor(agent_config, mock_session_factory, mock_session_manager):
    """Create health monitor instance."""
    return HealthMonitor(
        config=agent_config,
        session_factory=mock_session_factory,
        session_manager=mock_session_manager,
        monitor_interval=1,  # Short interval for testing
    )


class TestHealthMonitorLifecycle:
    """Tests for health monitor start/stop lifecycle."""

    async def test_start_monitor(self, health_monitor):
        """Test starting the health monitor."""
        await health_monitor.start()
        assert health_monitor._running is True
        assert health_monitor._monitor_task is not None
        await health_monitor.stop()

    async def test_stop_monitor(self, health_monitor):
        """Test stopping the health monitor."""
        await health_monitor.start()
        await health_monitor.stop()
        assert health_monitor._running is False

    async def test_start_already_running(self, health_monitor):
        """Test starting when already running is a no-op."""
        await health_monitor.start()
        await health_monitor.start()  # Should log warning but not error
        assert health_monitor._running is True
        await health_monitor.stop()

    async def test_stop_not_running(self, health_monitor):
        """Test stopping when not running is a no-op."""
        await health_monitor.stop()  # Should log warning but not error
        assert health_monitor._running is False


class TestIdleSessionDetection:
    """Tests for idle session detection and killing."""

    async def test_detect_idle_sessions(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test detection of idle sessions."""
        # Create mock idle session
        session_id = uuid4()
        task_id = uuid4()
        idle_session = AgentSession(
            id=session_id,
            task_id=task_id,
            model="claude-sonnet-4",
            status=SessionStatus.active,
            started_at=datetime.utcnow() - timedelta(minutes=10),
            last_activity_at=datetime.utcnow() - timedelta(minutes=6),
        )

        # Mock database queries
        mock_db_session = mock_session_factory.return_value.__aenter__.return_value

        with patch(
            "forgemaster.orchestrator.health_monitor.get_idle_sessions",
            new=AsyncMock(return_value=[idle_session]),
        ), patch(
            "forgemaster.orchestrator.health_monitor.end_session",
            new=AsyncMock(),
        ) as mock_end_session, patch(
            "forgemaster.orchestrator.health_monitor.get_task",
            new=AsyncMock(
                return_value=Task(
                    id=task_id,
                    project_id=uuid4(),
                    title="Test task",
                    agent_type="executor",
                    status=TaskStatus.running,
                    retry_count=0,
                    max_retries=3,
                )
            ),
        ), patch(
            "forgemaster.orchestrator.health_monitor.increment_retry_count",
            new=AsyncMock(
                return_value=Task(
                    id=task_id,
                    project_id=uuid4(),
                    title="Test task",
                    agent_type="executor",
                    status=TaskStatus.running,
                    retry_count=1,
                    max_retries=3,
                )
            ),
        ), patch(
            "forgemaster.orchestrator.health_monitor.update_task_status",
            new=AsyncMock(),
        ):
            killed_ids = await health_monitor._check_idle_sessions()

        assert len(killed_ids) == 1
        assert killed_ids[0] == session_id
        mock_end_session.assert_called_once()

    async def test_no_idle_sessions(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test when no idle sessions exist."""
        with patch(
            "forgemaster.orchestrator.health_monitor.get_idle_sessions",
            new=AsyncMock(return_value=[]),
        ):
            killed_ids = await health_monitor._check_idle_sessions()

        assert len(killed_ids) == 0


class TestSessionKillLogic:
    """Tests for session killing logic."""

    async def test_kill_session(
        self,
        health_monitor,
        mock_session_factory,
        mock_session_manager,
    ):
        """Test killing a session."""
        session_id = uuid4()

        with patch(
            "forgemaster.orchestrator.health_monitor.end_session",
            new=AsyncMock(),
        ) as mock_end_session:
            await health_monitor._kill_session(
                session_id,
                reason="Test timeout",
            )

        # Should attempt to end via session manager
        mock_session_manager.end_session.assert_called_once_with(
            session_id=str(session_id),
            status="failed",
        )

        # Should update database session
        mock_end_session.assert_called_once()
        call_args = mock_end_session.call_args
        assert call_args[1]["session_id"] == session_id
        assert call_args[1]["status"] == SessionStatus.killed
        assert call_args[1]["error_message"] == "Test timeout"

    async def test_kill_session_not_in_manager(
        self,
        health_monitor,
        mock_session_factory,
        mock_session_manager,
    ):
        """Test killing a session not found in session manager."""
        session_id = uuid4()

        # Session manager raises ValueError for missing session
        mock_session_manager.end_session.side_effect = ValueError("Session not found")

        with patch(
            "forgemaster.orchestrator.health_monitor.end_session",
            new=AsyncMock(),
        ) as mock_end_session:
            # Should not raise - handles ValueError gracefully
            await health_monitor._kill_session(
                session_id,
                reason="Test timeout",
            )

        # Database session should still be updated
        mock_end_session.assert_called_once()


class TestRetryScheduling:
    """Tests for task retry scheduling logic."""

    async def test_schedule_retry_within_limits(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test scheduling retry when within retry limits."""
        task_id = uuid4()

        task_after_increment = Task(
            id=task_id,
            project_id=uuid4(),
            title="Test task",
            agent_type="executor",
            status=TaskStatus.running,
            retry_count=1,
            max_retries=3,
        )

        with patch(
            "forgemaster.orchestrator.health_monitor.get_task",
            new=AsyncMock(
                return_value=Task(
                    id=task_id,
                    project_id=uuid4(),
                    title="Test task",
                    agent_type="executor",
                    status=TaskStatus.running,
                    retry_count=0,
                    max_retries=3,
                )
            ),
        ), patch(
            "forgemaster.orchestrator.health_monitor.increment_retry_count",
            new=AsyncMock(return_value=task_after_increment),
        ), patch(
            "forgemaster.orchestrator.health_monitor.update_task_status",
            new=AsyncMock(),
        ) as mock_update_status:
            await health_monitor._schedule_retry(task_id)

        # Should transition to READY for retry
        mock_update_status.assert_called_once()
        call_args = mock_update_status.call_args
        assert call_args[0][1] == task_id
        assert call_args[0][2] == TaskStatus.ready

    async def test_schedule_retry_max_exceeded(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test scheduling retry when max retries exceeded."""
        task_id = uuid4()

        task_after_increment = Task(
            id=task_id,
            project_id=uuid4(),
            title="Test task",
            agent_type="executor",
            status=TaskStatus.running,
            retry_count=3,
            max_retries=3,
        )

        with patch(
            "forgemaster.orchestrator.health_monitor.get_task",
            new=AsyncMock(
                return_value=Task(
                    id=task_id,
                    project_id=uuid4(),
                    title="Test task",
                    agent_type="executor",
                    status=TaskStatus.running,
                    retry_count=2,
                    max_retries=3,
                )
            ),
        ), patch(
            "forgemaster.orchestrator.health_monitor.increment_retry_count",
            new=AsyncMock(return_value=task_after_increment),
        ), patch(
            "forgemaster.orchestrator.health_monitor.update_task_status",
            new=AsyncMock(),
        ) as mock_update_status:
            await health_monitor._schedule_retry(task_id)

        # Should transition to FAILED
        mock_update_status.assert_called_once()
        call_args = mock_update_status.call_args
        assert call_args[0][1] == task_id
        assert call_args[0][2] == TaskStatus.failed

    async def test_schedule_retry_task_not_found(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test scheduling retry when task not found."""
        task_id = uuid4()

        with patch(
            "forgemaster.orchestrator.health_monitor.get_task",
            new=AsyncMock(return_value=None),
        ):
            # Should log error but not raise
            await health_monitor._schedule_retry(task_id)


class TestMonitoringLoop:
    """Tests for the monitoring loop behavior."""

    async def test_monitoring_loop_runs(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test that monitoring loop executes periodically."""
        check_count = 0

        async def mock_check_idle():
            nonlocal check_count
            check_count += 1
            return []

        with patch.object(
            health_monitor,
            "_check_idle_sessions",
            new=mock_check_idle,
        ):
            await health_monitor.start()
            await asyncio.sleep(2.5)  # Should trigger ~2 checks
            await health_monitor.stop()

        assert check_count >= 2

    async def test_monitoring_loop_handles_errors(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test that monitoring loop continues after errors."""
        call_count = 0

        async def mock_check_idle_with_error():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Test error")
            return []

        with patch.object(
            health_monitor,
            "_check_idle_sessions",
            new=mock_check_idle_with_error,
        ):
            await health_monitor.start()
            await asyncio.sleep(2.5)  # Should continue despite error
            await health_monitor.stop()

        # Should have recovered and continued
        assert call_count >= 2


class TestTaskStatusUpdates:
    """Tests for task status updates after session kill."""

    async def test_running_to_ready_on_retry(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test task transitions from RUNNING to READY on retry."""
        task_id = uuid4()
        session_id = uuid4()

        idle_session = AgentSession(
            id=session_id,
            task_id=task_id,
            model="claude-sonnet-4",
            status=SessionStatus.active,
            started_at=datetime.utcnow() - timedelta(minutes=10),
            last_activity_at=datetime.utcnow() - timedelta(minutes=6),
        )

        task_after_increment = Task(
            id=task_id,
            project_id=uuid4(),
            title="Test task",
            agent_type="executor",
            status=TaskStatus.running,
            retry_count=1,
            max_retries=3,
        )

        with patch(
            "forgemaster.orchestrator.health_monitor.get_idle_sessions",
            new=AsyncMock(return_value=[idle_session]),
        ), patch(
            "forgemaster.orchestrator.health_monitor.end_session",
            new=AsyncMock(),
        ), patch(
            "forgemaster.orchestrator.health_monitor.get_task",
            new=AsyncMock(
                return_value=Task(
                    id=task_id,
                    project_id=uuid4(),
                    title="Test task",
                    agent_type="executor",
                    status=TaskStatus.running,
                    retry_count=0,
                    max_retries=3,
                )
            ),
        ), patch(
            "forgemaster.orchestrator.health_monitor.increment_retry_count",
            new=AsyncMock(return_value=task_after_increment),
        ), patch(
            "forgemaster.orchestrator.health_monitor.update_task_status",
            new=AsyncMock(),
        ) as mock_update_status:
            await health_monitor._check_idle_sessions()

        # Verify task was transitioned to READY
        mock_update_status.assert_called_once()
        call_args = mock_update_status.call_args
        assert call_args[0][2] == TaskStatus.ready

    async def test_running_to_failed_on_max_retries(
        self,
        health_monitor,
        mock_session_factory,
    ):
        """Test task transitions from RUNNING to FAILED when max retries exceeded."""
        task_id = uuid4()
        session_id = uuid4()

        idle_session = AgentSession(
            id=session_id,
            task_id=task_id,
            model="claude-sonnet-4",
            status=SessionStatus.active,
            started_at=datetime.utcnow() - timedelta(minutes=10),
            last_activity_at=datetime.utcnow() - timedelta(minutes=6),
        )

        task_after_increment = Task(
            id=task_id,
            project_id=uuid4(),
            title="Test task",
            agent_type="executor",
            status=TaskStatus.running,
            retry_count=3,
            max_retries=3,
        )

        with patch(
            "forgemaster.orchestrator.health_monitor.get_idle_sessions",
            new=AsyncMock(return_value=[idle_session]),
        ), patch(
            "forgemaster.orchestrator.health_monitor.end_session",
            new=AsyncMock(),
        ), patch(
            "forgemaster.orchestrator.health_monitor.get_task",
            new=AsyncMock(
                return_value=Task(
                    id=task_id,
                    project_id=uuid4(),
                    title="Test task",
                    agent_type="executor",
                    status=TaskStatus.running,
                    retry_count=2,
                    max_retries=3,
                )
            ),
        ), patch(
            "forgemaster.orchestrator.health_monitor.increment_retry_count",
            new=AsyncMock(return_value=task_after_increment),
        ), patch(
            "forgemaster.orchestrator.health_monitor.update_task_status",
            new=AsyncMock(),
        ) as mock_update_status:
            await health_monitor._check_idle_sessions()

        # Verify task was transitioned to FAILED
        mock_update_status.assert_called_once()
        call_args = mock_update_status.call_args
        assert call_args[0][2] == TaskStatus.failed
