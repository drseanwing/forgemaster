"""Single-worker task dispatcher for Forgemaster orchestrator.

This module implements the core dispatch loop that polls for ready tasks,
assigns them to agents, monitors execution, and routes results through
the result handler. This is the single-worker version; Phase 3 extends
to multi-worker parallel dispatch.

The dispatcher coordinates between the task state machine, agent session
manager, context generator, and result handler to drive tasks through
their lifecycle from READY to completion.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.agents.result_schema import AgentResult, parse_agent_result_safe
from forgemaster.agents.session import AgentSessionManager, SessionInfo
from forgemaster.config import AgentConfig
from forgemaster.context.generator import ContextGenerator
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.database.queries.task import get_next_task
from forgemaster.orchestrator.state_machine import InvalidTransitionError, TaskStateMachine

logger = structlog.get_logger(__name__)

# Type alias for the async session factory callable
SessionFactory = Callable[[], AsyncSession]


class Dispatcher:
    """Single-worker task dispatcher for the orchestrator loop.

    The dispatcher is the core engine of Forgemaster's orchestration. It
    continuously polls for ready tasks, assigns them to agent sessions,
    monitors execution, and handles results. This single-worker version
    processes one task at a time sequentially.

    Attributes:
        config: Agent configuration settings.
        session_factory: Callable that produces new database sessions.
        state_machine: Task state machine for managing transitions.
        session_manager: Agent session lifecycle manager.
        context_generator: Jinja2-based context generator for agent prompts.
        poll_interval: Seconds between polling cycles.
    """

    def __init__(
        self,
        config: AgentConfig,
        session_factory: SessionFactory,
        state_machine: TaskStateMachine,
        session_manager: AgentSessionManager,
        context_generator: ContextGenerator,
        result_handler: Any | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            config: Agent configuration with retry limits and timeouts.
            session_factory: Callable returning new AsyncSession instances.
            state_machine: Task lifecycle state machine.
            session_manager: Agent session lifecycle manager.
            context_generator: Context generator for building agent prompts.
            result_handler: Optional ResultHandler instance for processing results.
                           If None, results are logged but not fully processed.
            poll_interval: Seconds to sleep between poll cycles. Defaults to 5.0.
        """
        self.config = config
        self.session_factory = session_factory
        self.state_machine = state_machine
        self.session_manager = session_manager
        self.context_generator = context_generator
        self.result_handler = result_handler
        self.poll_interval = poll_interval

        self._running: bool = False
        self._current_task_id: str | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._logger = logger.bind(component="Dispatcher")

    @property
    def is_running(self) -> bool:
        """Whether the dispatch loop is currently active.

        Returns:
            True if the loop is running, False otherwise.
        """
        return self._running

    async def start(self, project_id: str) -> None:
        """Start the dispatch loop for a project.

        Launches the asynchronous poll loop as a background task. The loop
        runs until ``stop()`` is called or an unrecoverable error occurs.

        Args:
            project_id: UUID string of the project to dispatch tasks for.

        Raises:
            RuntimeError: If the dispatcher is already running.
        """
        if self._running:
            raise RuntimeError("Dispatcher is already running")

        self._running = True
        self._logger.info("dispatcher_starting", project_id=project_id)
        self._poll_task = asyncio.create_task(
            self._poll_loop(project_id),
            name=f"dispatcher-{project_id}",
        )

    async def stop(self) -> None:
        """Gracefully stop the dispatch loop.

        Signals the loop to stop after the current cycle completes, then
        waits for the background task to finish. Safe to call if the
        dispatcher is not running.
        """
        if not self._running:
            self._logger.debug("dispatcher_stop_noop", reason="not running")
            return

        self._logger.info("dispatcher_stopping")
        self._running = False

        if self._poll_task is not None:
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            finally:
                self._poll_task = None

        self._logger.info("dispatcher_stopped")

    async def _poll_loop(self, project_id: str) -> None:
        """Main async poll loop that drives the dispatch cycle.

        Each cycle:
        1. Resolve PENDING tasks whose dependencies are now met.
        2. Select the highest-priority READY task.
        3. Assign the task to an agent and execute.
        4. Handle the result (success, partial, failure).
        5. Sleep for the configured poll interval.

        Args:
            project_id: UUID string of the project to process.
        """
        self._logger.info("poll_loop_started", project_id=project_id)

        while self._running:
            try:
                async with self.session_factory() as db_session:
                    # Step 1: Resolve dependencies for pending tasks
                    newly_ready = await self.state_machine.update_pending_tasks(
                        project_id, db_session
                    )
                    if newly_ready:
                        await db_session.commit()
                        self._logger.info(
                            "pending_tasks_promoted",
                            count=len(newly_ready),
                            task_ids=[str(t.id) for t in newly_ready],
                        )

                    # Step 2: Select next task
                    task = await self._select_next_task(db_session, project_id)
                    if task is None:
                        await asyncio.sleep(self.poll_interval)
                        continue

                    # Step 3: Assign and execute
                    session_id = await self._assign_task(task, db_session)
                    if session_id is None:
                        await asyncio.sleep(self.poll_interval)
                        continue

                    # Step 4: Execute the agent session and collect result
                    agent_result = await self._execute_task(task, session_id, db_session)

                    # Step 5: Handle the result
                    await self._handle_result(task, session_id, agent_result, db_session)
                    await db_session.commit()

            except Exception:
                self._logger.exception("poll_loop_error", project_id=project_id)

            if self._running:
                await asyncio.sleep(self.poll_interval)

        self._logger.info("poll_loop_exited", project_id=project_id)

    async def _select_next_task(
        self,
        db_session: AsyncSession,
        project_id: str,
    ) -> Task | None:
        """Select the next highest-priority task ready for dispatch.

        Uses the query layer's ``get_next_task`` which returns tasks sorted
        by priority (lower number = higher priority), then by creation time.

        Args:
            db_session: Active database session.
            project_id: UUID string of the project.

        Returns:
            The next Task to dispatch, or None if no tasks are ready.
        """
        from uuid import UUID

        task = await get_next_task(db_session, UUID(project_id))

        if task is None:
            self._logger.debug("no_ready_tasks", project_id=project_id)
            return None

        self._logger.info(
            "task_selected",
            task_id=str(task.id),
            title=task.title,
            priority=task.priority,
            agent_type=task.agent_type,
        )
        return task

    async def _assign_task(
        self,
        task: Task,
        db_session: AsyncSession,
    ) -> str | None:
        """Assign a task to an agent by transitioning it and starting a session.

        Performs the state transitions READY -> ASSIGNED -> RUNNING and
        initialises an agent session with the appropriate context.

        Args:
            task: The task to assign.
            db_session: Active database session.

        Returns:
            The agent session ID string, or None if assignment failed.
        """
        task_id = str(task.id)
        self._current_task_id = task_id

        try:
            # Transition: READY -> ASSIGNED
            await self.state_machine.transition(task_id, TaskStatus.assigned, db_session)

            # Transition: ASSIGNED -> RUNNING
            await self.state_machine.transition(task_id, TaskStatus.running, db_session)

            # Generate agent context
            model = self._resolve_model_tier(task.model_tier)
            project_info = self._build_project_info(task)
            system_prompt = self.context_generator.generate_agent_context(
                agent_type=task.agent_type,
                task=task.description or task.title,
                project=project_info,
                model_tier=model,
            )

            # Start agent session
            session_id = await self.session_manager.start_session(
                task_id=task_id,
                agent_type=task.agent_type,
                model=model,
                system_prompt=system_prompt,
            )

            # Create database record for the agent session
            from forgemaster.database.queries.session import create_session as create_db_session

            await create_db_session(
                session=db_session,
                task_id=task.id,
                model=model,
            )
            await db_session.commit()

            self._logger.info(
                "task_assigned",
                task_id=task_id,
                session_id=session_id,
                agent_type=task.agent_type,
                model=model,
            )
            return session_id

        except InvalidTransitionError as e:
            self._logger.error(
                "task_assignment_transition_error",
                task_id=task_id,
                error=str(e),
            )
            self._current_task_id = None
            return None
        except Exception:
            self._logger.exception("task_assignment_error", task_id=task_id)
            self._current_task_id = None
            return None

    async def _execute_task(
        self,
        task: Task,
        session_id: str,
        db_session: AsyncSession,
    ) -> AgentResult:
        """Execute a task by sending it to the agent and parsing the response.

        Sends the task description to the agent session and parses the
        raw response into a structured AgentResult. Uses the safe parser
        to handle malformed outputs gracefully.

        Args:
            task: The task being executed.
            session_id: The agent session identifier.
            db_session: Active database session.

        Returns:
            Parsed AgentResult from the agent's response.
        """
        task_id = str(task.id)
        self._logger.info("task_execution_started", task_id=task_id, session_id=session_id)

        try:
            # Build the task message for the agent
            message = self._build_task_message(task)

            # Send to agent and get response
            raw_response = await self.session_manager.send_message(session_id, message)

            # Parse the response
            agent_result = parse_agent_result_safe(raw_response)

            self._logger.info(
                "task_execution_completed",
                task_id=task_id,
                session_id=session_id,
                result_status=agent_result.status,
                confidence=agent_result.confidence_score,
            )

            # End the agent session
            await self.session_manager.end_session(session_id, status="completed")

            return agent_result

        except Exception as e:
            self._logger.exception(
                "task_execution_error",
                task_id=task_id,
                session_id=session_id,
            )

            # End session as failed
            try:
                await self.session_manager.end_session(session_id, status="failed")
            except Exception:
                self._logger.exception("session_cleanup_error", session_id=session_id)

            # Return a failure result
            return AgentResult(
                status="failed",
                summary=f"Task execution failed: {e}",
                details=str(e),
                confidence_score=0.0,
            )

    async def _handle_result(
        self,
        task: Task,
        session_id: str,
        agent_result: AgentResult,
        db_session: AsyncSession,
    ) -> None:
        """Route the agent result to the result handler or apply defaults.

        If a result handler is configured, delegates to it. Otherwise
        applies basic transitions: success -> REVIEW, failure -> FAILED.

        Args:
            task: The task that was executed.
            session_id: The agent session identifier.
            agent_result: Parsed result from the agent.
            db_session: Active database session.
        """
        task_id = str(task.id)

        if self.result_handler is not None:
            await self.result_handler.handle_result(task, db_session, agent_result)
        else:
            # Default handling when no result handler is configured
            await self._default_handle_result(task, agent_result, db_session)

        self._current_task_id = None

        self._logger.info(
            "result_handled",
            task_id=task_id,
            result_status=agent_result.status,
        )

    async def _default_handle_result(
        self,
        task: Task,
        agent_result: AgentResult,
        db_session: AsyncSession,
    ) -> None:
        """Apply default result handling when no ResultHandler is configured.

        Args:
            task: The task that was executed.
            agent_result: Parsed result from the agent.
            db_session: Active database session.
        """
        task_id = str(task.id)

        if agent_result.status == "success":
            await self.state_machine.transition(task_id, TaskStatus.review, db_session)
        elif agent_result.status == "failed":
            await self.state_machine.transition(task_id, TaskStatus.failed, db_session)
        else:
            # partial - keep running, agent may continue
            self._logger.info(
                "task_partial_result",
                task_id=task_id,
                summary=agent_result.summary,
            )

    def _resolve_model_tier(self, model_tier: str | None) -> str:
        """Resolve a model tier preference to a concrete model identifier.

        Args:
            model_tier: Tier preference from the task ('auto', 'haiku', 'sonnet', 'opus').

        Returns:
            Claude model identifier string.
        """
        tier_map = {
            "haiku": "claude-3-5-haiku-20241022",
            "sonnet": "claude-sonnet-4-20250514",
            "opus": "claude-opus-4-20250514",
        }
        if model_tier and model_tier in tier_map:
            return tier_map[model_tier]
        # Default to sonnet for 'auto' or unknown
        return tier_map["sonnet"]

    def _build_project_info(self, task: Task) -> dict[str, Any]:
        """Build a project info dictionary for context generation.

        Args:
            task: Task containing project relationship.

        Returns:
            Dictionary with project metadata for template rendering.
        """
        project = getattr(task, "project", None)
        if project is not None:
            return {
                "name": getattr(project, "name", "Unknown Project"),
                "context": getattr(project, "description", "No context provided."),
                "standards": "Follow project coding standards.",
            }
        return {
            "name": "Unknown Project",
            "context": "No context provided.",
            "standards": "Follow standard best practices.",
        }

    def _build_task_message(self, task: Task) -> str:
        """Build the message to send to the agent for task execution.

        Args:
            task: The task to build a message for.

        Returns:
            Formatted task message string.
        """
        parts = [
            f"## Task: {task.title}",
        ]
        if task.description:
            parts.append(f"\n### Description\n{task.description}")
        if task.files_touched:
            parts.append("\n### Files to Modify")
            for fp in task.files_touched:
                parts.append(f"- `{fp}`")
        parts.append(
            "\n### Instructions\n"
            "Complete the task described above. Return your result as a JSON object "
            "with fields: status, summary, details, tests_run, issues_discovered, "
            "lessons_learned, files_modified, confidence_score."
        )
        return "\n".join(parts)
