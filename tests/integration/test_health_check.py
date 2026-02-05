"""Integration tests for health check and rollback system.

Tests health endpoint polling, threshold-based status transitions, rollback
trigger logic, and rollback execution strategies.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from forgemaster.config import DockerConfig
from forgemaster.pipeline.container import (
    ContainerAction,
    ContainerInfo,
    ContainerManager,
    ContainerStatus,
)
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

# --- HealthCheckConfig Tests ---


def test_health_check_config_defaults() -> None:
    """Test HealthCheckConfig with default values."""
    config = HealthCheckConfig(url="http://localhost:8080/health")

    assert config.url == "http://localhost:8080/health"
    assert config.interval_seconds == 10.0
    assert config.timeout_seconds == 5.0
    assert config.healthy_threshold == 3
    assert config.unhealthy_threshold == 3
    assert config.expected_status_codes == [200]
    assert config.expected_body is None


def test_health_check_config_custom() -> None:
    """Test HealthCheckConfig with custom values."""
    config = HealthCheckConfig(
        url="http://api.example.com/status",
        interval_seconds=5.0,
        timeout_seconds=2.0,
        healthy_threshold=2,
        unhealthy_threshold=5,
        expected_status_codes=[200, 204],
        expected_body="OK",
    )

    assert config.url == "http://api.example.com/status"
    assert config.interval_seconds == 5.0
    assert config.timeout_seconds == 2.0
    assert config.healthy_threshold == 2
    assert config.unhealthy_threshold == 5
    assert config.expected_status_codes == [200, 204]
    assert config.expected_body == "OK"


def test_health_check_config_validation() -> None:
    """Test HealthCheckConfig validation constraints."""
    # interval_seconds must be positive
    with pytest.raises(ValueError):
        HealthCheckConfig(url="http://localhost", interval_seconds=0.0)

    # timeout_seconds must be positive
    with pytest.raises(ValueError):
        HealthCheckConfig(url="http://localhost", timeout_seconds=-1.0)

    # healthy_threshold must be >= 1
    with pytest.raises(ValueError):
        HealthCheckConfig(url="http://localhost", healthy_threshold=0)

    # unhealthy_threshold must be >= 1
    with pytest.raises(ValueError):
        HealthCheckConfig(url="http://localhost", unhealthy_threshold=0)


# --- HealthPoller Tests ---


@pytest.mark.asyncio
async def test_health_poller_initialization() -> None:
    """Test HealthPoller initialization."""
    config = HealthCheckConfig(url="http://localhost:8080/health")
    poller = HealthPoller(config)

    assert poller.config == config
    assert poller._consecutive_successes == 0
    assert poller._consecutive_failures == 0
    assert poller._current_status == HealthStatus.UNKNOWN

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_success() -> None:
    """Test successful health check."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        expected_status_codes=[200],
        healthy_threshold=1,
    )
    poller = HealthPoller(config)

    # Mock httpx response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "OK"

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    with patch.object(poller, "_get_session", return_value=mock_session):
        result = await poller.check_health()

    assert result.status == HealthStatus.HEALTHY
    assert result.response_code == 200
    assert result.error is None
    assert result.consecutive_successes == 1
    assert result.consecutive_failures == 0
    assert result.response_time_seconds is not None
    assert result.response_time_seconds > 0

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_bad_status_code() -> None:
    """Test health check with unexpected status code."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        expected_status_codes=[200],
        unhealthy_threshold=1,
    )
    poller = HealthPoller(config)

    mock_response = Mock()
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    with patch.object(poller, "_get_session", return_value=mock_session):
        result = await poller.check_health()

    assert result.status == HealthStatus.UNHEALTHY
    assert result.response_code == 503
    assert result.error == "Unexpected status code: 503"
    assert result.consecutive_successes == 0
    assert result.consecutive_failures == 1

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_expected_body_match() -> None:
    """Test health check with expected body content matching."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        expected_status_codes=[200],
        expected_body="healthy",
        healthy_threshold=1,
    )
    poller = HealthPoller(config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"status": "healthy"}'

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    with patch.object(poller, "_get_session", return_value=mock_session):
        result = await poller.check_health()

    assert result.status == HealthStatus.HEALTHY
    assert result.error is None
    assert result.consecutive_successes == 1

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_expected_body_mismatch() -> None:
    """Test health check with expected body content not found."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        expected_status_codes=[200],
        expected_body="healthy",
        unhealthy_threshold=1,
    )
    poller = HealthPoller(config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"status": "degraded"}'

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    with patch.object(poller, "_get_session", return_value=mock_session):
        result = await poller.check_health()

    assert result.status == HealthStatus.UNHEALTHY
    assert result.error == "Expected body content not found: healthy"
    assert result.consecutive_failures == 1

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_timeout() -> None:
    """Test health check timeout handling."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        timeout_seconds=0.1,
        unhealthy_threshold=1,
    )
    poller = HealthPoller(config)

    # Mock timeout exception with httpx
    mock_session = AsyncMock()
    mock_session.get = AsyncMock(side_effect=httpx.ReadTimeout("Request timed out"))

    with patch.object(poller, "_get_session", return_value=mock_session):
        result = await poller.check_health()

    assert result.status == HealthStatus.UNHEALTHY
    assert "connection error" in result.error.lower()
    assert result.response_code is None
    assert result.consecutive_failures == 1

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_connection_error() -> None:
    """Test health check connection error handling."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        unhealthy_threshold=1,
    )
    poller = HealthPoller(config)

    # Mock connection error
    mock_session = AsyncMock()
    mock_session.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    with patch.object(poller, "_get_session", return_value=mock_session):
        result = await poller.check_health()

    assert result.status == HealthStatus.UNHEALTHY
    assert "connection error" in result.error.lower()
    assert result.response_code is None
    assert result.consecutive_failures == 1

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_consecutive_success_threshold() -> None:
    """Test health status transition based on consecutive success threshold."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        healthy_threshold=3,
    )
    poller = HealthPoller(config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "OK"

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    with patch.object(poller, "_get_session", return_value=mock_session):
        # First check - still UNKNOWN
        result1 = await poller.check_health()
        assert result1.status == HealthStatus.UNKNOWN
        assert result1.consecutive_successes == 1

        # Second check - still UNKNOWN
        result2 = await poller.check_health()
        assert result2.status == HealthStatus.UNKNOWN
        assert result2.consecutive_successes == 2

        # Third check - now HEALTHY
        result3 = await poller.check_health()
        assert result3.status == HealthStatus.HEALTHY
        assert result3.consecutive_successes == 3

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_consecutive_failure_threshold() -> None:
    """Test health status transition based on consecutive failure threshold."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        unhealthy_threshold=2,
    )
    poller = HealthPoller(config)

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Error"

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    with patch.object(poller, "_get_session", return_value=mock_session):
        # First failure - still UNKNOWN (or DEGRADED if was HEALTHY)
        result1 = await poller.check_health()
        assert result1.consecutive_failures == 1
        assert result1.status != HealthStatus.UNHEALTHY

        # Second failure - now UNHEALTHY
        result2 = await poller.check_health()
        assert result2.consecutive_failures == 2
        assert result2.status == HealthStatus.UNHEALTHY

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_with_timeout_override() -> None:
    """Test check_health_with_timeout with custom timeout."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        timeout_seconds=10.0,  # Default
    )
    poller = HealthPoller(config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "OK"

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    with patch.object(poller, "_get_session", return_value=mock_session):
        # Use override timeout
        result = await poller.check_health_with_timeout(timeout_seconds=2.0)

    assert result.response_code == 200
    # Timeout should be restored to original
    assert poller.config.timeout_seconds == 10.0

    await poller.close()


@pytest.mark.asyncio
async def test_poll_until_healthy_success() -> None:
    """Test poll_until_healthy that eventually succeeds."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        interval_seconds=0.1,
        healthy_threshold=1,
    )
    poller = HealthPoller(config)

    call_count = 0

    async def mock_check_health():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            # First two checks fail
            poller._consecutive_failures += 1
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                url=config.url,
                response_code=500,
                checked_at=datetime.now(timezone.utc).isoformat(),
                consecutive_failures=poller._consecutive_failures,
            )
        else:
            # Third check succeeds
            poller._consecutive_successes = 1
            poller._consecutive_failures = 0
            poller._current_status = HealthStatus.HEALTHY
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                url=config.url,
                response_code=200,
                checked_at=datetime.now(timezone.utc).isoformat(),
                consecutive_successes=1,
            )

    with patch.object(poller, "check_health", side_effect=mock_check_health):
        result = await poller.poll_until_healthy(timeout_seconds=5.0)

    assert result.status == HealthStatus.HEALTHY
    assert call_count == 3

    await poller.close()


@pytest.mark.asyncio
async def test_poll_until_healthy_timeout() -> None:
    """Test poll_until_healthy that times out."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        interval_seconds=0.1,
    )
    poller = HealthPoller(config)

    async def mock_check_health():
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            url=config.url,
            response_code=500,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    with patch.object(poller, "check_health", side_effect=mock_check_health):
        with pytest.raises(HealthCheckTimeout) as exc_info:
            await poller.poll_until_healthy(timeout_seconds=0.3)

    assert "timed out" in str(exc_info.value).lower()

    await poller.close()


@pytest.mark.asyncio
async def test_start_stop_polling() -> None:
    """Test continuous background polling start and stop."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        interval_seconds=0.1,
    )
    poller = HealthPoller(config)

    check_count = 0

    async def mock_check_health():
        nonlocal check_count
        check_count += 1
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            url=config.url,
            response_code=200,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    with patch.object(poller, "check_health", side_effect=mock_check_health):
        await poller.start_polling()

        # Let it poll a few times
        await asyncio.sleep(0.35)

        await poller.stop_polling()

    # Should have polled at least 2-3 times
    assert check_count >= 2

    await poller.close()


@pytest.mark.asyncio
async def test_start_polling_already_active() -> None:
    """Test starting polling when already active does nothing."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        interval_seconds=0.1,
    )
    poller = HealthPoller(config)

    async def mock_check_health():
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            url=config.url,
            response_code=200,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    with patch.object(poller, "check_health", side_effect=mock_check_health):
        await poller.start_polling()
        task1 = poller._polling_task

        # Try to start again
        await poller.start_polling()
        task2 = poller._polling_task

        # Should be the same task
        assert task1 is task2

        await poller.stop_polling()

    await poller.close()


@pytest.mark.asyncio
async def test_stop_polling_not_active() -> None:
    """Test stopping polling when not active is safe."""
    config = HealthCheckConfig(url="http://localhost:8080/health")
    poller = HealthPoller(config)

    # Should not raise
    await poller.stop_polling()

    await poller.close()


# --- RollbackConfig Tests ---


def test_rollback_config_defaults() -> None:
    """Test RollbackConfig with default values."""
    config = RollbackConfig()

    assert config.enabled is True
    assert config.max_rollback_attempts == 3
    assert config.cooldown_seconds == 60.0
    assert config.trigger_on_consecutive_failures == 3


def test_rollback_config_custom() -> None:
    """Test RollbackConfig with custom values."""
    config = RollbackConfig(
        enabled=False,
        max_rollback_attempts=5,
        cooldown_seconds=120.0,
        trigger_on_consecutive_failures=5,
    )

    assert config.enabled is False
    assert config.max_rollback_attempts == 5
    assert config.cooldown_seconds == 120.0
    assert config.trigger_on_consecutive_failures == 5


# --- RollbackTrigger Tests ---


@pytest.mark.asyncio
async def test_rollback_trigger_initialization() -> None:
    """Test RollbackTrigger initialization."""
    health_config = HealthCheckConfig(url="http://localhost:8080/health")
    poller = HealthPoller(health_config)
    rollback_config = RollbackConfig()
    trigger = RollbackTrigger(poller, rollback_config)

    assert trigger.poller is poller
    assert trigger.config == rollback_config
    assert trigger._rollback_count == 0
    assert trigger._last_rollback_time is None

    await poller.close()


@pytest.mark.asyncio
async def test_rollback_trigger_disabled() -> None:
    """Test rollback trigger when disabled."""
    health_config = HealthCheckConfig(url="http://localhost:8080/health")
    poller = HealthPoller(health_config)
    rollback_config = RollbackConfig(enabled=False)
    trigger = RollbackTrigger(poller, rollback_config)

    health_result = HealthCheckResult(
        status=HealthStatus.UNHEALTHY,
        url=health_config.url,
        consecutive_failures=5,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    decision = await trigger.evaluate(health_result)

    assert decision.should_rollback is False
    assert "disabled" in decision.reason.lower()

    await poller.close()


@pytest.mark.asyncio
async def test_rollback_trigger_below_threshold() -> None:
    """Test rollback trigger when failures are below threshold."""
    health_config = HealthCheckConfig(url="http://localhost:8080/health")
    poller = HealthPoller(health_config)
    rollback_config = RollbackConfig(trigger_on_consecutive_failures=5)
    trigger = RollbackTrigger(poller, rollback_config)

    health_result = HealthCheckResult(
        status=HealthStatus.UNHEALTHY,
        url=health_config.url,
        consecutive_failures=3,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    decision = await trigger.evaluate(health_result)

    assert decision.should_rollback is False
    assert "below threshold" in decision.reason.lower()
    assert decision.consecutive_failures == 3

    await poller.close()


@pytest.mark.asyncio
async def test_rollback_trigger_meets_threshold() -> None:
    """Test rollback trigger when failures meet threshold."""
    health_config = HealthCheckConfig(url="http://localhost:8080/health")
    poller = HealthPoller(health_config)
    rollback_config = RollbackConfig(trigger_on_consecutive_failures=3)
    trigger = RollbackTrigger(poller, rollback_config)

    health_result = HealthCheckResult(
        status=HealthStatus.UNHEALTHY,
        url=health_config.url,
        consecutive_failures=3,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    decision = await trigger.evaluate(health_result)

    assert decision.should_rollback is True
    assert "exceeded threshold" in decision.reason.lower()
    assert decision.rollback_attempt == 1
    assert trigger._rollback_count == 1

    await poller.close()


@pytest.mark.asyncio
async def test_rollback_trigger_cooldown_active() -> None:
    """Test rollback trigger during cooldown period."""
    health_config = HealthCheckConfig(url="http://localhost:8080/health")
    poller = HealthPoller(health_config)
    rollback_config = RollbackConfig(
        trigger_on_consecutive_failures=3,
        cooldown_seconds=60.0,
    )
    trigger = RollbackTrigger(poller, rollback_config)

    # Simulate a previous rollback
    trigger._rollback_count = 1
    trigger._last_rollback_time = asyncio.get_event_loop().time()

    health_result = HealthCheckResult(
        status=HealthStatus.UNHEALTHY,
        url=health_config.url,
        consecutive_failures=5,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    decision = await trigger.evaluate(health_result)

    assert decision.should_rollback is False
    assert "cooldown" in decision.reason.lower()
    assert decision.cooldown_remaining_seconds > 0

    await poller.close()


@pytest.mark.asyncio
async def test_rollback_trigger_max_attempts_reached() -> None:
    """Test rollback trigger when max attempts are reached."""
    health_config = HealthCheckConfig(url="http://localhost:8080/health")
    poller = HealthPoller(health_config)
    rollback_config = RollbackConfig(
        trigger_on_consecutive_failures=3,
        max_rollback_attempts=3,
    )
    trigger = RollbackTrigger(poller, rollback_config)

    # Simulate reaching max attempts
    trigger._rollback_count = 3

    health_result = HealthCheckResult(
        status=HealthStatus.UNHEALTHY,
        url=health_config.url,
        consecutive_failures=5,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    decision = await trigger.evaluate(health_result)

    assert decision.should_rollback is False
    assert "maximum" in decision.reason.lower()
    assert decision.rollback_attempt == 3

    await poller.close()


# --- RollbackExecutor Tests ---


@pytest.mark.asyncio
async def test_rollback_executor_initialization() -> None:
    """Test RollbackExecutor initialization."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    assert executor.container_manager is container_manager
    assert executor.config == rollback_config

    await container_manager.close()


@pytest.mark.asyncio
async def test_rollback_restart_strategy_success() -> None:
    """Test rollback with RESTART_CONTAINER strategy (success)."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    # Mock successful restart
    mock_action = ContainerAction(
        success=True,
        container_id="test-container",
        action="restart",
        previous_status="running",
        current_status="running",
        duration_seconds=1.5,
    )

    with patch.object(container_manager, "restart_container", return_value=mock_action):
        result = await executor.execute_rollback(
            "test-container",
            strategy=RollbackStrategy.RESTART_CONTAINER,
        )

    assert result.success is True
    assert result.action == "restart_container"
    assert result.containers_affected == ["test-container"]
    assert result.error is None

    await container_manager.close()


@pytest.mark.asyncio
async def test_rollback_restart_strategy_failure() -> None:
    """Test rollback with RESTART_CONTAINER strategy (failure)."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    # Mock failed restart
    mock_action = ContainerAction(
        success=False,
        container_id="test-container",
        action="restart",
        error="Container not found",
        duration_seconds=0.5,
    )

    with patch.object(container_manager, "restart_container", return_value=mock_action):
        result = await executor.execute_rollback(
            "test-container",
            strategy=RollbackStrategy.RESTART_CONTAINER,
        )

    assert result.success is False
    assert result.action == "restart_container"
    assert result.error == "Container not found"

    await container_manager.close()


@pytest.mark.asyncio
async def test_rollback_revert_image_strategy_success() -> None:
    """Test rollback with REVERT_IMAGE strategy (success)."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    # Mock container status
    mock_info = ContainerInfo(
        container_id="test-container",
        name="test",
        image="myapp:v2",
        status=ContainerStatus.RUNNING,
    )

    # Mock successful stop and start
    mock_stop = ContainerAction(
        success=True,
        container_id="test-container",
        action="stop",
        previous_status="running",
        current_status="stopped",
    )

    mock_start = ContainerAction(
        success=True,
        container_id="test-container",
        action="start",
        previous_status="stopped",
        current_status="running",
    )

    with patch.object(container_manager, "get_container_status", return_value=mock_info):
        with patch.object(container_manager, "stop_container", return_value=mock_stop):
            with patch.object(container_manager, "start_container", return_value=mock_start):
                result = await executor.execute_rollback(
                    "test-container",
                    strategy=RollbackStrategy.REVERT_IMAGE,
                    previous_image="myapp:v1",
                )

    assert result.success is True
    assert result.action == "revert_image"
    assert result.previous_image == "myapp:v2"
    assert result.rollback_image == "myapp:v1"
    assert result.containers_affected == ["test-container"]

    await container_manager.close()


@pytest.mark.asyncio
async def test_rollback_revert_image_strategy_missing_image() -> None:
    """Test rollback with REVERT_IMAGE strategy without previous_image."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    result = await executor.execute_rollback(
        "test-container",
        strategy=RollbackStrategy.REVERT_IMAGE,
        previous_image=None,
    )

    assert result.success is False
    assert result.action == "revert_image"
    assert "previous_image required" in result.error.lower()

    await container_manager.close()


@pytest.mark.asyncio
async def test_rollback_revert_image_strategy_stop_fails() -> None:
    """Test rollback with REVERT_IMAGE strategy when stop fails."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    mock_info = ContainerInfo(
        container_id="test-container",
        name="test",
        image="myapp:v2",
        status=ContainerStatus.RUNNING,
    )

    mock_stop = ContainerAction(
        success=False,
        container_id="test-container",
        action="stop",
        error="Failed to stop",
    )

    with patch.object(container_manager, "get_container_status", return_value=mock_info):
        with patch.object(container_manager, "stop_container", return_value=mock_stop):
            result = await executor.execute_rollback(
                "test-container",
                strategy=RollbackStrategy.REVERT_IMAGE,
                previous_image="myapp:v1",
            )

    assert result.success is False
    assert result.action == "revert_image"
    assert "failed to stop" in result.error.lower()

    await container_manager.close()


@pytest.mark.asyncio
async def test_rollback_stop_service_strategy_success() -> None:
    """Test rollback with STOP_SERVICE strategy (success)."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    mock_stop = ContainerAction(
        success=True,
        container_id="test-container",
        action="stop",
        previous_status="running",
        current_status="stopped",
    )

    with patch.object(container_manager, "stop_container", return_value=mock_stop):
        result = await executor.execute_rollback(
            "test-container",
            strategy=RollbackStrategy.STOP_SERVICE,
        )

    assert result.success is True
    assert result.action == "stop_service"
    assert result.containers_affected == ["test-container"]

    await container_manager.close()


@pytest.mark.asyncio
async def test_rollback_stop_service_strategy_failure() -> None:
    """Test rollback with STOP_SERVICE strategy (failure)."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    mock_stop = ContainerAction(
        success=False,
        container_id="test-container",
        action="stop",
        error="Container not found",
    )

    with patch.object(container_manager, "stop_container", return_value=mock_stop):
        result = await executor.execute_rollback(
            "test-container",
            strategy=RollbackStrategy.STOP_SERVICE,
        )

    assert result.success is False
    assert result.action == "stop_service"
    assert result.error == "Container not found"

    await container_manager.close()


@pytest.mark.asyncio
async def test_rollback_unknown_strategy() -> None:
    """Test rollback with unknown strategy."""
    docker_config = DockerConfig()
    container_manager = ContainerManager(docker_config)
    rollback_config = RollbackConfig()
    executor = RollbackExecutor(container_manager, rollback_config)

    # Cast to bypass type checking
    invalid_strategy = "invalid_strategy"  # type: ignore

    result = await executor.execute_rollback(
        "test-container",
        strategy=invalid_strategy,  # type: ignore
    )

    assert result.success is False
    assert "unknown" in result.error.lower()

    await container_manager.close()


# --- Model Validation Tests ---


def test_health_check_result_model() -> None:
    """Test HealthCheckResult model validation."""
    result = HealthCheckResult(
        status=HealthStatus.HEALTHY,
        url="http://localhost:8080/health",
        response_code=200,
        response_time_seconds=0.123,
        checked_at=datetime.now(timezone.utc).isoformat(),
        consecutive_successes=5,
        consecutive_failures=0,
    )

    assert result.status == HealthStatus.HEALTHY
    assert result.response_code == 200
    assert result.consecutive_successes == 5
    assert result.consecutive_failures == 0


def test_rollback_decision_model() -> None:
    """Test RollbackDecision model validation."""
    decision = RollbackDecision(
        should_rollback=True,
        reason="Threshold exceeded",
        consecutive_failures=5,
        cooldown_remaining_seconds=0.0,
        rollback_attempt=2,
    )

    assert decision.should_rollback is True
    assert decision.consecutive_failures == 5
    assert decision.rollback_attempt == 2


def test_rollback_result_model() -> None:
    """Test RollbackResult model validation."""
    result = RollbackResult(
        success=True,
        action="restart_container",
        containers_affected=["container1", "container2"],
        duration_seconds=2.5,
    )

    assert result.success is True
    assert result.action == "restart_container"
    assert len(result.containers_affected) == 2


# --- Edge Cases ---


@pytest.mark.asyncio
async def test_health_check_invalid_url() -> None:
    """Test health check with invalid URL."""
    config = HealthCheckConfig(url="not-a-valid-url", unhealthy_threshold=1)
    poller = HealthPoller(config)

    result = await poller.check_health()

    assert result.status == HealthStatus.UNHEALTHY
    assert result.error is not None
    assert result.response_code is None

    await poller.close()


@pytest.mark.asyncio
async def test_health_check_empty_response_body() -> None:
    """Test health check with empty response body."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        expected_status_codes=[200],
        expected_body="OK",
        unhealthy_threshold=1,
    )
    poller = HealthPoller(config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = ""

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    with patch.object(poller, "_get_session", return_value=mock_session):
        result = await poller.check_health()

    assert result.status == HealthStatus.UNHEALTHY
    assert "expected body content not found" in result.error.lower()

    await poller.close()


@pytest.mark.asyncio
async def test_multiple_status_codes_accepted() -> None:
    """Test health check with multiple acceptable status codes."""
    config = HealthCheckConfig(
        url="http://localhost:8080/health",
        expected_status_codes=[200, 204, 304],
        healthy_threshold=1,
    )
    poller = HealthPoller(config)

    for status_code in [200, 204, 304]:
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.text = ""

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        with patch.object(poller, "_get_session", return_value=mock_session):
            # Reset consecutive counters
            poller._consecutive_successes = 0
            poller._consecutive_failures = 0
            poller._current_status = HealthStatus.UNKNOWN

            result = await poller.check_health()

        assert result.status == HealthStatus.HEALTHY, f"Failed for status code {status_code}"
        assert result.response_code == status_code

    await poller.close()
