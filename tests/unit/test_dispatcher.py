"""Unit tests for the single-worker dispatcher and result handler.

Tests cover:
- Dispatcher lifecycle (start/stop)
- Poll loop mechanics
- Priority-based task selection
- Task assignment flow (READY -> ASSIGNED -> RUNNING)
- Result handling (success, partial, failure)
- Retry logic with max retry enforcement
- Lesson extraction from agent results
- Error resilience in the dispatch loop
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forgemaster.agents.result_schema import (
    AgentResult,
    IssueDiscovered,
    LessonLearned,
)
from forgemaster.agents.session import AgentSessionManager
from forgemaster.config import AgentConfig
from forgemaster.context.generator import ContextGenerator
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.orchestrator.dispatcher import Dispatcher
from forgemaster.orchestrator.result_handler import ResultHandler
from forgemaster.orchestrator.state_machine import (
    InvalidTransitionError,
    TaskStateMachine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create a default AgentConfig for testing."""
    return AgentConfig(
        max_concurrent_workers=1,
        session_timeout_seconds=1800,
        idle_timeout_seconds=300,
        max_retries=3,
        context_warning_threshold=0.8,
    )


@pytest.fixture
def mock_state_machine() -> TaskStateMachine:
    """Create a mock TaskStateMachine."""
    sm = AsyncMock(spec=TaskStateMachine)
    sm.update_pending_tasks = AsyncMock(return_value=[])
    sm.transition = AsyncMock()
    return sm


@pytest.fixture
def mock_session_manager() -> AgentSessionManager:
    """Create a mock AgentSessionManager."""
    mgr = AsyncMock(spec=AgentSessionManager)
    mgr.start_session = AsyncMock(return_value="session-001")
    mgr.send_message = AsyncMock(
        return_value='{"status": "success", "summary": "Done", "details": "All good"}'
    )
    mgr.end_session = AsyncMock(return_value={"session_id": "session-001"})
    return mgr


@pytest.fixture
def mock_context_generator() -> ContextGenerator:
    """Create a mock ContextGenerator."""
    gen = MagicMock(spec=ContextGenerator)
    gen.generate_agent_context = MagicMock(return_value="System prompt for agent")
    return gen


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create a mock async database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def session_factory(mock_db_session):
    """Create a session factory that yields the mock db session."""

    @asynccontextmanager
    async def _factory():
        yield mock_db_session

    return _factory


@pytest.fixture
def sample_project_id() -> str:
    """Return a stable project UUID string."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_task(sample_project_id) -> Task:
    """Create a sample Task in READY status."""
    task = Task(
        id=uuid.uuid4(),
        project_id=uuid.UUID(sample_project_id),
        title="Implement feature X",
        description="Add the X feature to the codebase",
        status=TaskStatus.ready,
        agent_type="executor",
        model_tier="sonnet",
        priority=10,
        retry_count=0,
        max_retries=3,
        files_touched=["src/feature_x.py"],
    )
    return task


@pytest.fixture
def success_result() -> AgentResult:
    """Create a successful AgentResult."""
    return AgentResult(
        status="success",
        summary="Feature X implemented successfully",
        details="Added feature X with full test coverage",
        files_modified=["src/feature_x.py"],
        confidence_score=0.95,
    )


@pytest.fixture
def partial_result() -> AgentResult:
    """Create a partial AgentResult."""
    return AgentResult(
        status="partial",
        summary="Feature X partially implemented",
        details="Completed core logic but tests still needed",
        files_modified=["src/feature_x.py"],
        confidence_score=0.5,
    )


@pytest.fixture
def failure_result() -> AgentResult:
    """Create a failed AgentResult."""
    return AgentResult(
        status="failed",
        summary="Feature X implementation failed",
        details="Could not resolve dependency conflicts",
        issues_discovered=[
            IssueDiscovered(
                description="Missing dependency libfoo",
                severity="high",
                location="src/feature_x.py:42",
            )
        ],
        confidence_score=0.0,
    )


@pytest.fixture
def result_with_lessons() -> AgentResult:
    """Create an AgentResult that includes lessons learned."""
    return AgentResult(
        status="success",
        summary="Feature implemented with insights",
        details="Completed with important findings",
        files_modified=["src/feature_x.py"],
        lessons_learned=[
            LessonLearned(
                context="Async database calls in tight loops",
                observation="Connection pool exhaustion under load",
                recommendation="Use batch queries instead of per-item queries",
            ),
            LessonLearned(
                context="Pydantic model validation",
                observation="field_validator runs before model_validator",
                recommendation="Place cross-field checks in model_validator",
            ),
        ],
        confidence_score=0.9,
    )


@pytest.fixture
def dispatcher(
    agent_config,
    session_factory,
    mock_state_machine,
    mock_session_manager,
    mock_context_generator,
) -> Dispatcher:
    """Create a Dispatcher instance with all mocked dependencies."""
    return Dispatcher(
        config=agent_config,
        session_factory=session_factory,
        state_machine=mock_state_machine,
        session_manager=mock_session_manager,
        context_generator=mock_context_generator,
        poll_interval=0.01,  # Very short for fast tests
    )


@pytest.fixture
def result_handler(agent_config, mock_state_machine) -> ResultHandler:
    """Create a ResultHandler instance."""
    return ResultHandler(
        config=agent_config,
        state_machine=mock_state_machine,
    )


# ---------------------------------------------------------------------------
# Dispatcher Lifecycle Tests
# ---------------------------------------------------------------------------


class TestDispatcherLifecycle:
    """Test dispatcher start and stop mechanics."""

    def test_initial_state(self, dispatcher):
        """Dispatcher should start in non-running state."""
        assert dispatcher.is_running is False
        assert dispatcher._current_task_id is None
        assert dispatcher._poll_task is None

    @pytest.mark.asyncio
    async def test_start_sets_running(self, dispatcher, sample_project_id):
        """Start should set running flag and create background task."""
        await dispatcher.start(sample_project_id)

        assert dispatcher.is_running is True
        assert dispatcher._poll_task is not None

        # Clean up
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_start_raises_if_already_running(self, dispatcher, sample_project_id):
        """Start should raise RuntimeError if already running."""
        await dispatcher.start(sample_project_id)

        with pytest.raises(RuntimeError, match="already running"):
            await dispatcher.start(sample_project_id)

        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_state(self, dispatcher, sample_project_id):
        """Stop should clear running flag and poll task."""
        await dispatcher.start(sample_project_id)
        await dispatcher.stop()

        assert dispatcher.is_running is False
        assert dispatcher._poll_task is None

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_running(self, dispatcher):
        """Stop should be safe to call when not running."""
        await dispatcher.stop()
        assert dispatcher.is_running is False


# ---------------------------------------------------------------------------
# Task Selection Tests
# ---------------------------------------------------------------------------


class TestTaskSelection:
    """Test priority-based task selection logic."""

    @pytest.mark.asyncio
    async def test_select_next_task_found(
        self, dispatcher, mock_db_session, sample_task, sample_project_id
    ):
        """Should return the highest priority ready task."""
        with patch(
            "forgemaster.orchestrator.dispatcher.get_next_task",
            new_callable=AsyncMock,
            return_value=sample_task,
        ):
            result = await dispatcher._select_next_task(mock_db_session, sample_project_id)

        assert result is sample_task

    @pytest.mark.asyncio
    async def test_select_next_task_none(
        self, dispatcher, mock_db_session, sample_project_id
    ):
        """Should return None when no tasks are ready."""
        with patch(
            "forgemaster.orchestrator.dispatcher.get_next_task",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await dispatcher._select_next_task(mock_db_session, sample_project_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_select_logs_task_info(
        self, dispatcher, mock_db_session, sample_task, sample_project_id
    ):
        """Should log task selection details."""
        with patch(
            "forgemaster.orchestrator.dispatcher.get_next_task",
            new_callable=AsyncMock,
            return_value=sample_task,
        ):
            with patch.object(dispatcher._logger, "info") as mock_log:
                await dispatcher._select_next_task(mock_db_session, sample_project_id)

                # Verify logging occurred with task details
                calls = [c for c in mock_log.call_args_list if c[0][0] == "task_selected"]
                assert len(calls) == 1
                assert calls[0][1]["task_id"] == str(sample_task.id)
                assert calls[0][1]["priority"] == sample_task.priority


# ---------------------------------------------------------------------------
# Task Assignment Tests
# ---------------------------------------------------------------------------


class TestTaskAssignment:
    """Test the task assignment flow."""

    @pytest.mark.asyncio
    async def test_assign_task_transitions(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task
    ):
        """Assignment should transition READY -> ASSIGNED -> RUNNING."""
        with patch(
            "forgemaster.orchestrator.dispatcher.create_db_session",
            new_callable=AsyncMock,
        ):
            session_id = await dispatcher._assign_task(sample_task, mock_db_session)

        assert session_id is not None

        # Verify both transitions occurred in order
        transition_calls = mock_state_machine.transition.call_args_list
        assert len(transition_calls) == 2
        assert transition_calls[0][0][1] == TaskStatus.assigned
        assert transition_calls[1][0][1] == TaskStatus.running

    @pytest.mark.asyncio
    async def test_assign_task_starts_session(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Assignment should start an agent session."""
        with patch(
            "forgemaster.orchestrator.dispatcher.create_db_session",
            new_callable=AsyncMock,
        ):
            session_id = await dispatcher._assign_task(sample_task, mock_db_session)

        assert session_id == "session-001"
        mock_session_manager.start_session.assert_called_once()
        call_kwargs = mock_session_manager.start_session.call_args[1]
        assert call_kwargs["task_id"] == str(sample_task.id)
        assert call_kwargs["agent_type"] == "executor"

    @pytest.mark.asyncio
    async def test_assign_task_generates_context(
        self, dispatcher, mock_context_generator, mock_db_session, sample_task
    ):
        """Assignment should generate agent context from the task."""
        with patch(
            "forgemaster.orchestrator.dispatcher.create_db_session",
            new_callable=AsyncMock,
        ):
            await dispatcher._assign_task(sample_task, mock_db_session)

        mock_context_generator.generate_agent_context.assert_called_once()
        call_kwargs = mock_context_generator.generate_agent_context.call_args[1]
        assert call_kwargs["agent_type"] == "executor"

    @pytest.mark.asyncio
    async def test_assign_task_returns_none_on_transition_error(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task
    ):
        """Should return None if state transition fails."""
        mock_state_machine.transition.side_effect = InvalidTransitionError(
            TaskStatus.ready, TaskStatus.assigned, str(sample_task.id)
        )

        session_id = await dispatcher._assign_task(sample_task, mock_db_session)

        assert session_id is None

    @pytest.mark.asyncio
    async def test_assign_task_returns_none_on_session_error(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Should return None if agent session creation fails."""
        mock_session_manager.start_session.side_effect = RuntimeError("SDK not ready")

        session_id = await dispatcher._assign_task(sample_task, mock_db_session)

        assert session_id is None


# ---------------------------------------------------------------------------
# Task Execution Tests
# ---------------------------------------------------------------------------


class TestTaskExecution:
    """Test task execution and response parsing."""

    @pytest.mark.asyncio
    async def test_execute_task_sends_message(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Should send task message to the agent session."""
        result = await dispatcher._execute_task(sample_task, "session-001", mock_db_session)

        mock_session_manager.send_message.assert_called_once()
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_execute_task_ends_session_on_success(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Should end the session as completed on success."""
        await dispatcher._execute_task(sample_task, "session-001", mock_db_session)

        mock_session_manager.end_session.assert_called_once_with(
            "session-001", status="completed"
        )

    @pytest.mark.asyncio
    async def test_execute_task_returns_failure_on_error(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Should return failure result if execution throws."""
        mock_session_manager.send_message.side_effect = RuntimeError("Connection lost")

        result = await dispatcher._execute_task(sample_task, "session-001", mock_db_session)

        assert result.status == "failed"
        assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_execute_task_ends_session_as_failed_on_error(
        self, dispatcher, mock_session_manager, mock_db_session, sample_task
    ):
        """Should end session as failed if execution throws."""
        mock_session_manager.send_message.side_effect = RuntimeError("Connection lost")

        await dispatcher._execute_task(sample_task, "session-001", mock_db_session)

        mock_session_manager.end_session.assert_called_once_with(
            "session-001", status="failed"
        )


# ---------------------------------------------------------------------------
# Default Result Handling Tests
# ---------------------------------------------------------------------------


class TestDefaultResultHandling:
    """Test the dispatcher's default result handling (no ResultHandler)."""

    @pytest.mark.asyncio
    async def test_success_transitions_to_review(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task, success_result
    ):
        """Success should transition task to REVIEW."""
        await dispatcher._default_handle_result(sample_task, success_result, mock_db_session)

        mock_state_machine.transition.assert_called_once_with(
            str(sample_task.id), TaskStatus.review, mock_db_session
        )

    @pytest.mark.asyncio
    async def test_failure_transitions_to_failed(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task, failure_result
    ):
        """Failure should transition task to FAILED."""
        await dispatcher._default_handle_result(sample_task, failure_result, mock_db_session)

        mock_state_machine.transition.assert_called_once_with(
            str(sample_task.id), TaskStatus.failed, mock_db_session
        )

    @pytest.mark.asyncio
    async def test_partial_keeps_running(
        self, dispatcher, mock_state_machine, mock_db_session, sample_task, partial_result
    ):
        """Partial result should not trigger any transition."""
        await dispatcher._default_handle_result(sample_task, partial_result, mock_db_session)

        mock_state_machine.transition.assert_not_called()


# ---------------------------------------------------------------------------
# ResultHandler Tests
# ---------------------------------------------------------------------------


class TestResultHandler:
    """Test the ResultHandler class."""

    @pytest.mark.asyncio
    async def test_handle_success_transitions_to_review(
        self, result_handler, mock_state_machine, mock_db_session, sample_task, success_result
    ):
        """Success should transition task to REVIEW."""
        await result_handler.handle_result(sample_task, mock_db_session, success_result)

        mock_state_machine.transition.assert_called_once_with(
            str(sample_task.id), TaskStatus.review, mock_db_session
        )

    @pytest.mark.asyncio
    async def test_handle_partial_stores_handover(
        self, result_handler, mock_db_session, sample_task, partial_result
    ):
        """Partial result should store handover context."""
        mock_session_record = MagicMock()
        mock_session_record.handover_context = None

        with patch(
            "forgemaster.orchestrator.result_handler.list_sessions",
            new_callable=AsyncMock,
            return_value=[mock_session_record],
        ):
            await result_handler.handle_result(
                sample_task, mock_db_session, partial_result
            )

        assert mock_session_record.handover_context is not None
        assert mock_session_record.handover_context["summary"] == partial_result.summary

    @pytest.mark.asyncio
    async def test_handle_failure_increments_retry(
        self, result_handler, mock_state_machine, mock_db_session, sample_task, failure_result
    ):
        """Failure should increment retry count."""
        updated_task = MagicMock()
        updated_task.retry_count = 1

        with patch(
            "forgemaster.orchestrator.result_handler.increment_retry_count",
            new_callable=AsyncMock,
            return_value=updated_task,
        ):
            await result_handler.handle_result(
                sample_task, mock_db_session, failure_result
            )

        # Should transition RUNNING -> FAILED -> READY for retry
        transition_calls = mock_state_machine.transition.call_args_list
        assert len(transition_calls) == 2
        assert transition_calls[0][0][1] == TaskStatus.failed
        assert transition_calls[1][0][1] == TaskStatus.ready

    @pytest.mark.asyncio
    async def test_handle_failure_max_retries_exceeded(
        self, result_handler, mock_state_machine, mock_db_session, sample_task, failure_result
    ):
        """Should transition to FAILED when max retries exhausted."""
        updated_task = MagicMock()
        updated_task.retry_count = 3  # Matches max_retries

        with patch(
            "forgemaster.orchestrator.result_handler.increment_retry_count",
            new_callable=AsyncMock,
            return_value=updated_task,
        ):
            await result_handler.handle_result(
                sample_task, mock_db_session, failure_result
            )

        # Should only transition to FAILED (no retry)
        transition_calls = mock_state_machine.transition.call_args_list
        assert len(transition_calls) == 1
        assert transition_calls[0][0][1] == TaskStatus.failed

    @pytest.mark.asyncio
    async def test_handle_failure_transition_error_resilience(
        self, result_handler, mock_state_machine, mock_db_session, sample_task, failure_result
    ):
        """Should handle transition errors gracefully during failure handling."""
        updated_task = MagicMock()
        updated_task.retry_count = 5  # Exceeds max

        mock_state_machine.transition.side_effect = InvalidTransitionError(
            TaskStatus.running, TaskStatus.failed, str(sample_task.id)
        )

        with patch(
            "forgemaster.orchestrator.result_handler.increment_retry_count",
            new_callable=AsyncMock,
            return_value=updated_task,
        ):
            # Should not raise
            await result_handler.handle_result(
                sample_task, mock_db_session, failure_result
            )


# ---------------------------------------------------------------------------
# Lesson Extraction Tests
# ---------------------------------------------------------------------------


class TestLessonExtraction:
    """Test lesson extraction from agent results."""

    @pytest.mark.asyncio
    async def test_extract_lessons_creates_records(
        self, result_handler, mock_db_session, sample_task, result_with_lessons
    ):
        """Should create lesson records for each lesson in the result."""
        mock_lesson = MagicMock()
        mock_lesson.id = uuid.uuid4()

        with (
            patch(
                "forgemaster.orchestrator.result_handler.create_lesson",
                new_callable=AsyncMock,
                return_value=mock_lesson,
            ) as mock_create,
            patch(
                "forgemaster.orchestrator.result_handler.enqueue_embedding",
                new_callable=AsyncMock,
            ) as mock_enqueue,
        ):
            await result_handler._extract_lessons(
                sample_task, mock_db_session, result_with_lessons
            )

        # Should create 2 lessons
        assert mock_create.call_count == 2

        # Verify first lesson mapping
        first_call = mock_create.call_args_list[0]
        assert first_call[1]["symptom"] == "Async database calls in tight loops"
        assert first_call[1]["root_cause"] == "Connection pool exhaustion under load"
        assert first_call[1]["fix_applied"] == "Use batch queries instead of per-item queries"

    @pytest.mark.asyncio
    async def test_extract_lessons_enqueues_embeddings(
        self, result_handler, mock_db_session, sample_task, result_with_lessons
    ):
        """Should enqueue embeddings for each lesson."""
        mock_lesson = MagicMock()
        mock_lesson.id = uuid.uuid4()

        with (
            patch(
                "forgemaster.orchestrator.result_handler.create_lesson",
                new_callable=AsyncMock,
                return_value=mock_lesson,
            ),
            patch(
                "forgemaster.orchestrator.result_handler.enqueue_embedding",
                new_callable=AsyncMock,
            ) as mock_enqueue,
        ):
            await result_handler._extract_lessons(
                sample_task, mock_db_session, result_with_lessons
            )

        # 2 lessons * 2 embeddings each (content + symptom) = 4
        assert mock_enqueue.call_count == 4

        # Verify embedding targets
        enqueue_calls = mock_enqueue.call_args_list
        target_columns = [c[1]["target_column"] for c in enqueue_calls]
        assert target_columns.count("content_embedding") == 2
        assert target_columns.count("symptom_embedding") == 2

    @pytest.mark.asyncio
    async def test_extract_lessons_skips_without_project_id(
        self, result_handler, mock_db_session, result_with_lessons
    ):
        """Should skip extraction if task has no project_id."""
        task = Task(
            id=uuid.uuid4(),
            project_id=None,
            title="Orphan task",
            status=TaskStatus.running,
            agent_type="executor",
        )

        with patch(
            "forgemaster.orchestrator.result_handler.create_lesson",
            new_callable=AsyncMock,
        ) as mock_create:
            await result_handler._extract_lessons(
                task, mock_db_session, result_with_lessons
            )

        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_lessons_handles_individual_errors(
        self, result_handler, mock_db_session, sample_task, result_with_lessons
    ):
        """Should continue extracting if one lesson fails."""
        mock_lesson = MagicMock()
        mock_lesson.id = uuid.uuid4()

        call_count = 0

        async def create_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("DB error on first lesson")
            return mock_lesson

        with (
            patch(
                "forgemaster.orchestrator.result_handler.create_lesson",
                new_callable=AsyncMock,
                side_effect=create_side_effect,
            ),
            patch(
                "forgemaster.orchestrator.result_handler.enqueue_embedding",
                new_callable=AsyncMock,
            ) as mock_enqueue,
        ):
            await result_handler._extract_lessons(
                sample_task, mock_db_session, result_with_lessons
            )

        # Second lesson should still be processed (2 embeddings)
        assert mock_enqueue.call_count == 2

    @pytest.mark.asyncio
    async def test_success_with_lessons_triggers_extraction(
        self, result_handler, mock_state_machine, mock_db_session,
        sample_task, result_with_lessons
    ):
        """Success result with lessons should trigger lesson extraction."""
        mock_lesson = MagicMock()
        mock_lesson.id = uuid.uuid4()

        with (
            patch(
                "forgemaster.orchestrator.result_handler.create_lesson",
                new_callable=AsyncMock,
                return_value=mock_lesson,
            ) as mock_create,
            patch(
                "forgemaster.orchestrator.result_handler.enqueue_embedding",
                new_callable=AsyncMock,
            ),
        ):
            await result_handler.handle_result(
                sample_task, mock_db_session, result_with_lessons
            )

        # Lessons should be created
        assert mock_create.call_count == 2


# ---------------------------------------------------------------------------
# Model Tier Resolution Tests
# ---------------------------------------------------------------------------


class TestModelTierResolution:
    """Test model tier resolution logic in the dispatcher."""

    def test_resolve_sonnet(self, dispatcher):
        """Should resolve 'sonnet' to the sonnet model ID."""
        result = dispatcher._resolve_model_tier("sonnet")
        assert "sonnet" in result

    def test_resolve_haiku(self, dispatcher):
        """Should resolve 'haiku' to the haiku model ID."""
        result = dispatcher._resolve_model_tier("haiku")
        assert "haiku" in result

    def test_resolve_opus(self, dispatcher):
        """Should resolve 'opus' to the opus model ID."""
        result = dispatcher._resolve_model_tier("opus")
        assert "opus" in result

    def test_resolve_auto_defaults_to_sonnet(self, dispatcher):
        """Should default to sonnet for 'auto' tier."""
        result = dispatcher._resolve_model_tier("auto")
        assert "sonnet" in result

    def test_resolve_none_defaults_to_sonnet(self, dispatcher):
        """Should default to sonnet for None tier."""
        result = dispatcher._resolve_model_tier(None)
        assert "sonnet" in result


# ---------------------------------------------------------------------------
# Task Message Building Tests
# ---------------------------------------------------------------------------


class TestTaskMessageBuilding:
    """Test task message construction."""

    def test_build_task_message_includes_title(self, dispatcher, sample_task):
        """Message should include the task title."""
        message = dispatcher._build_task_message(sample_task)
        assert sample_task.title in message

    def test_build_task_message_includes_description(self, dispatcher, sample_task):
        """Message should include the task description."""
        message = dispatcher._build_task_message(sample_task)
        assert sample_task.description in message

    def test_build_task_message_includes_files(self, dispatcher, sample_task):
        """Message should include files to modify."""
        message = dispatcher._build_task_message(sample_task)
        assert "src/feature_x.py" in message

    def test_build_task_message_without_description(self, dispatcher):
        """Message should handle tasks without description."""
        task = Task(
            id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            title="Simple task",
            description=None,
            status=TaskStatus.ready,
            agent_type="executor",
        )
        message = dispatcher._build_task_message(task)
        assert "Simple task" in message
        assert "Description" not in message

    def test_build_task_message_without_files(self, dispatcher):
        """Message should handle tasks without files_touched."""
        task = Task(
            id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            title="No files task",
            description="Just a task",
            status=TaskStatus.ready,
            agent_type="executor",
            files_touched=None,
        )
        message = dispatcher._build_task_message(task)
        assert "Files to Modify" not in message


# ---------------------------------------------------------------------------
# Poll Loop Integration Tests
# ---------------------------------------------------------------------------


class TestPollLoopIntegration:
    """Test the poll loop behaviour with mocked internals."""

    @pytest.mark.asyncio
    async def test_poll_loop_stops_when_flag_cleared(self, dispatcher, sample_project_id):
        """Poll loop should exit when _running is set to False."""
        dispatcher._running = True

        # Make the loop exit after one iteration
        async def stop_after_one(*args, **kwargs):
            dispatcher._running = False
            return None

        with patch.object(
            dispatcher, "_select_next_task", side_effect=stop_after_one
        ):
            await dispatcher._poll_loop(sample_project_id)

        assert dispatcher.is_running is False

    @pytest.mark.asyncio
    async def test_poll_loop_handles_exceptions(self, dispatcher, sample_project_id):
        """Poll loop should survive exceptions and continue."""
        call_count = 0

        dispatcher._running = True

        async def error_then_stop(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient DB error")
            dispatcher._running = False
            return None

        with patch.object(
            dispatcher, "_select_next_task", side_effect=error_then_stop
        ):
            # Should not raise
            await dispatcher._poll_loop(sample_project_id)

        assert call_count == 2  # Survived error and ran again


# ---------------------------------------------------------------------------
# Dispatcher with ResultHandler Integration Tests
# ---------------------------------------------------------------------------


class TestDispatcherWithResultHandler:
    """Test dispatcher delegates to ResultHandler when configured."""

    @pytest.mark.asyncio
    async def test_handle_result_delegates_to_handler(
        self,
        agent_config,
        session_factory,
        mock_state_machine,
        mock_session_manager,
        mock_context_generator,
        mock_db_session,
        sample_task,
        success_result,
    ):
        """Should delegate to result_handler.handle_result when configured."""
        mock_handler = AsyncMock(spec=ResultHandler)

        disp = Dispatcher(
            config=agent_config,
            session_factory=session_factory,
            state_machine=mock_state_machine,
            session_manager=mock_session_manager,
            context_generator=mock_context_generator,
            result_handler=mock_handler,
            poll_interval=0.01,
        )

        await disp._handle_result(
            sample_task, "session-001", success_result, mock_db_session
        )

        mock_handler.handle_result.assert_called_once_with(
            sample_task, mock_db_session, success_result
        )
