"""Task dispatchers for Forgemaster orchestrator.

This module implements the core dispatch loops that poll for ready tasks,
assign them to agents, monitor execution, and route results through
the result handler.

Two dispatchers are provided:

- ``Dispatcher``: Single-worker sequential dispatcher (Phase 2).
- ``MultiWorkerDispatcher``: Multi-worker parallel dispatcher using
  git worktrees for isolation (Phase 3).

The dispatchers coordinate between the task state machine, agent session
manager, context generator, worktree pool, and result handler to drive
tasks through their lifecycle from READY to completion.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import structlog
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.agents.result_schema import AgentResult, parse_agent_result_safe
from forgemaster.agents.session import AgentSessionManager, SessionInfo
from forgemaster.config import AgentConfig
from forgemaster.context.generator import ContextGenerator
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.database.queries.task import get_next_task, get_ready_tasks
from forgemaster.orchestrator.state_machine import InvalidTransitionError, TaskStateMachine
from forgemaster.pipeline.worktree import WorktreePool

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


# ---------------------------------------------------------------------------
# Multi-Worker Parallel Dispatcher (Phase 3)
# ---------------------------------------------------------------------------


class WorkerState(str, Enum):
    """Lifecycle states for a worker slot.

    Attributes:
        IDLE: Worker has a worktree but no task assigned.
        ASSIGNED: Task selected and being set up in this worker.
        RUNNING: Agent is actively executing the task.
        COMPLETING: Task finished, result being processed.
        FAILED: Worker encountered an unrecoverable error.
    """

    IDLE = "idle"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETING = "completing"
    FAILED = "failed"


class WorkerSlot(BaseModel):
    """Represents an active worker slot with its worktree and metrics.

    Each worker slot is backed by a git worktree from the WorktreePool,
    allowing it to run an agent session in isolation from other workers.

    Attributes:
        worker_id: Unique identifier for this worker (e.g., "worker-1").
        worktree_name: Name of the associated worktree in the pool.
        worktree_path: Filesystem path to the worktree directory.
        task_id: ID of the currently assigned task, or None if idle.
        session_id: ID of the current agent session, or None.
        state: Current lifecycle state of this worker.
        started_at: UTC timestamp when the current task started.
        tasks_completed: Total number of tasks successfully completed.
        tasks_failed: Total number of tasks that failed.
        last_health_check: UTC timestamp of the most recent health check.
        created_at: UTC timestamp when this worker slot was created.
    """

    worker_id: str
    worktree_name: str
    worktree_path: str
    task_id: str | None = None
    session_id: str | None = None
    state: WorkerState = WorkerState.IDLE
    started_at: datetime | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_health_check: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class MultiWorkerDispatcher:
    """Multi-worker parallel task dispatcher.

    Extends the dispatch pattern to run multiple tasks concurrently,
    each in its own git worktree with its own agent session. Tasks are
    selected by priority and dispatched to available worker slots up to
    the configured ``max_concurrent_workers`` limit.

    The dispatcher uses a ``WorktreePool`` for worktree lifecycle management
    and an ``asyncio.Semaphore`` to enforce its own concurrency limit
    independently of the pool's internal semaphore.

    Attributes:
        config: Agent configuration settings.
        session_factory: Callable that produces new database sessions.
        state_machine: Task state machine for managing transitions.
        session_manager: Agent session lifecycle manager.
        context_generator: Jinja2-based context generator for agent prompts.
        worktree_pool: Pool of git worktrees for parallel execution.
        result_handler: Optional result handler for post-execution processing.
        poll_interval: Seconds between polling cycles.
    """

    def __init__(
        self,
        config: AgentConfig,
        session_factory: SessionFactory,
        state_machine: TaskStateMachine,
        session_manager: AgentSessionManager,
        context_generator: ContextGenerator,
        worktree_pool: WorktreePool,
        result_handler: Any | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        """Initialize the multi-worker dispatcher.

        Args:
            config: Agent configuration with concurrency limits and timeouts.
            session_factory: Callable returning new AsyncSession instances.
            state_machine: Task lifecycle state machine.
            session_manager: Agent session lifecycle manager.
            context_generator: Context generator for building agent prompts.
            worktree_pool: WorktreePool instance for worktree management.
            result_handler: Optional ResultHandler instance for processing
                results. If None, results are logged but not fully processed.
            poll_interval: Seconds to sleep between poll cycles. Defaults to 5.0.
        """
        self.config = config
        self.session_factory = session_factory
        self.state_machine = state_machine
        self.session_manager = session_manager
        self.context_generator = context_generator
        self.worktree_pool = worktree_pool
        self.result_handler = result_handler
        self.poll_interval = poll_interval

        self._max_workers: int = config.max_concurrent_workers
        self._semaphore = asyncio.Semaphore(self._max_workers)
        self._active_count: int = 0
        self._running: bool = False
        self._poll_task: asyncio.Task[None] | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._workers: dict[str, WorkerSlot] = {}
        self._worker_tasks: dict[str, asyncio.Task[None]] = {}
        self._worker_counter: int = 0
        self._logger = logger.bind(component="MultiWorkerDispatcher")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the dispatch loop is currently active.

        Returns:
            True if the loop is running, False otherwise.
        """
        return self._running

    @property
    def active_workers(self) -> int:
        """Number of workers currently executing tasks.

        Returns:
            Count of active workers.
        """
        return self._active_count

    @property
    def available_slots(self) -> int:
        """Number of worker slots available for new tasks.

        Returns:
            Difference between max workers and active workers.
        """
        return max(0, self._max_workers - self._active_count)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, project_id: str) -> None:
        """Start the multi-worker dispatch loop for a project.

        Launches the asynchronous poll loop and health monitor as background
        tasks. The loops run until ``stop()`` is called.

        Args:
            project_id: UUID string of the project to dispatch tasks for.

        Raises:
            RuntimeError: If the dispatcher is already running.
        """
        if self._running:
            raise RuntimeError("MultiWorkerDispatcher is already running")

        self._running = True
        self._logger.info(
            "multi_worker_dispatcher_starting",
            project_id=project_id,
            max_workers=self._max_workers,
        )

        self._poll_task = asyncio.create_task(
            self._poll_loop(project_id),
            name=f"mw-dispatcher-{project_id}",
        )
        self._health_task = asyncio.create_task(
            self._health_loop(),
            name=f"mw-health-{project_id}",
        )

    async def stop(self) -> None:
        """Gracefully stop the multi-worker dispatch loop.

        Signals the loop to stop, waits for all running workers to complete
        their current tasks, releases all worktrees back to the pool, and
        cleans up background tasks.

        Safe to call if the dispatcher is not running.
        """
        if not self._running:
            self._logger.debug("multi_worker_dispatcher_stop_noop", reason="not running")
            return

        self._logger.info(
            "multi_worker_dispatcher_stopping",
            active_workers=self.active_workers,
        )
        self._running = False

        # Wait for all active worker tasks to complete
        active_tasks = list(self._worker_tasks.values())
        if active_tasks:
            self._logger.info(
                "waiting_for_workers",
                count=len(active_tasks),
            )
            results = await asyncio.gather(*active_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._logger.warning(
                        "worker_shutdown_error",
                        error=str(result),
                        error_type=type(result).__name__,
                    )

        # Release all worktrees
        for worker_id in list(self._workers.keys()):
            try:
                await self._release_worker(worker_id)
            except Exception as e:
                self._logger.warning(
                    "worker_release_error_on_shutdown",
                    worker_id=worker_id,
                    error=str(e),
                )

        # Cancel background tasks
        if self._poll_task is not None:
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            finally:
                self._poll_task = None

        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            finally:
                self._health_task = None

        self._worker_tasks.clear()
        self._logger.info("multi_worker_dispatcher_stopped")

    # ------------------------------------------------------------------
    # Poll Loop
    # ------------------------------------------------------------------

    async def _poll_loop(self, project_id: str) -> None:
        """Main async poll loop for multi-worker dispatch.

        Each cycle:
        1. Resolve PENDING tasks whose dependencies are now met.
        2. Select up to ``available_slots`` highest-priority READY tasks.
        3. Dispatch each task to a worker concurrently.
        4. Sleep for the configured poll interval.

        Workers handle their own execution and result processing
        asynchronously, so the poll loop does not block on individual
        task completion.

        Args:
            project_id: UUID string of the project to process.
        """
        self._logger.info("multi_worker_poll_loop_started", project_id=project_id)

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

                    # Step 2: Select tasks up to available capacity
                    slots = self.available_slots
                    if slots > 0:
                        tasks = await self._select_ready_tasks(
                            db_session, project_id, max_count=slots
                        )

                        # Step 3: Dispatch each task to a worker
                        for task in tasks:
                            worker_id = await self._dispatch_to_worker(task)
                            if worker_id is not None:
                                self._logger.info(
                                    "task_dispatched_to_worker",
                                    task_id=str(task.id),
                                    worker_id=worker_id,
                                    active_workers=self.active_workers,
                                    available_slots=self.available_slots,
                                )

            except Exception:
                self._logger.exception(
                    "multi_worker_poll_loop_error",
                    project_id=project_id,
                )

            if self._running:
                await asyncio.sleep(self.poll_interval)

        self._logger.info("multi_worker_poll_loop_exited", project_id=project_id)

    # ------------------------------------------------------------------
    # Task Selection
    # ------------------------------------------------------------------

    async def _select_ready_tasks(
        self,
        db_session: AsyncSession,
        project_id: str,
        max_count: int,
    ) -> list[Task]:
        """Select up to ``max_count`` highest-priority ready tasks.

        Filters out tasks already assigned to active workers to avoid
        double-dispatch.

        Args:
            db_session: Active database session.
            project_id: UUID string of the project.
            max_count: Maximum number of tasks to return.

        Returns:
            List of Task instances ready for dispatch, sorted by priority.
        """
        from uuid import UUID

        ready = await get_ready_tasks(db_session, UUID(project_id))

        # Filter out tasks already being processed by a worker
        active_task_ids = {
            w.task_id for w in self._workers.values() if w.task_id is not None
        }
        available = [t for t in ready if str(t.id) not in active_task_ids]

        selected = available[:max_count]

        if selected:
            self._logger.info(
                "tasks_selected",
                count=len(selected),
                total_ready=len(ready),
                task_ids=[str(t.id) for t in selected],
            )
        else:
            self._logger.debug("no_ready_tasks_for_dispatch", project_id=project_id)

        return selected

    # ------------------------------------------------------------------
    # Worker Dispatch & Release
    # ------------------------------------------------------------------

    async def _dispatch_to_worker(self, task: Task) -> str | None:
        """Acquire a worker slot, assign the task, and start execution.

        Acquires a worktree from the pool, creates a WorkerSlot, and
        launches an asyncio task for the worker's execution coroutine.
        The semaphore ensures that no more than ``max_concurrent_workers``
        are running simultaneously.

        Args:
            task: The task to dispatch to a worker.

        Returns:
            The worker_id string if dispatch succeeded, or None on failure.
        """
        task_id = str(task.id)

        # Try to acquire the concurrency semaphore (non-blocking)
        if self._active_count >= self._max_workers:
            self._logger.debug(
                "dispatch_blocked_by_semaphore",
                task_id=task_id,
                active_workers=self.active_workers,
            )
            return None

        await self._semaphore.acquire()
        self._active_count += 1

        try:
            # Acquire worktree from pool
            worktree = await self.worktree_pool.acquire()

            # Create worker slot
            self._worker_counter += 1
            worker_id = f"worker-{self._worker_counter}"

            slot = WorkerSlot(
                worker_id=worker_id,
                worktree_name=worktree.name,
                worktree_path=str(worktree.path),
                task_id=task_id,
                state=WorkerState.ASSIGNED,
                started_at=datetime.now(timezone.utc),
            )
            self._workers[worker_id] = slot

            # Assign the task to the worktree in the pool
            self.worktree_pool.assign_worktree(worktree.name, task_id)

            # Launch worker coroutine as background task
            worker_coro = self._run_worker(worker_id, task)
            async_task = asyncio.create_task(
                worker_coro,
                name=f"worker-{worker_id}-task-{task_id}",
            )
            self._worker_tasks[worker_id] = async_task

            self._logger.info(
                "worker_dispatched",
                worker_id=worker_id,
                task_id=task_id,
                worktree_name=worktree.name,
            )

            return worker_id

        except Exception as e:
            self._semaphore.release()
            self._active_count -= 1
            self._logger.exception(
                "worker_dispatch_failed",
                task_id=task_id,
                error=str(e),
            )
            return None

    async def _release_worker(self, worker_id: str) -> None:
        """Release a worker slot and return its worktree to the pool.

        Updates the worker state, releases the worktree back to the pool,
        releases the concurrency semaphore, and removes the worker from
        the active registry.

        Args:
            worker_id: Identifier of the worker to release.
        """
        slot = self._workers.get(worker_id)
        if slot is None:
            self._logger.debug("release_worker_not_found", worker_id=worker_id)
            return

        worktree_name = slot.worktree_name

        try:
            # Release worktree back to pool
            await self.worktree_pool.release(worktree_name)
        except Exception as e:
            self._logger.warning(
                "worktree_release_error",
                worker_id=worker_id,
                worktree_name=worktree_name,
                error=str(e),
            )

        # Release semaphore
        self._semaphore.release()
        self._active_count -= 1

        # Remove from registries
        self._workers.pop(worker_id, None)
        self._worker_tasks.pop(worker_id, None)

        self._logger.info(
            "worker_released",
            worker_id=worker_id,
            worktree_name=worktree_name,
        )

    # ------------------------------------------------------------------
    # Worker Execution
    # ------------------------------------------------------------------

    async def _run_worker(self, worker_id: str, task: Task) -> None:
        """Execute a task within a worker slot.

        This coroutine is the main execution path for a single worker.
        It performs task assignment (state transitions), agent execution,
        result handling, and cleanup. On completion or failure, the worker
        slot is released.

        Args:
            worker_id: Identifier of the worker slot.
            task: The task to execute.
        """
        slot = self._workers.get(worker_id)
        if slot is None:
            return

        task_id = str(task.id)

        try:
            async with self.session_factory() as db_session:
                # Transition: READY -> ASSIGNED -> RUNNING
                slot.state = WorkerState.RUNNING

                session_id = await self._assign_task(task, db_session)
                if session_id is None:
                    slot.state = WorkerState.FAILED
                    slot.tasks_failed += 1
                    return

                slot.session_id = session_id

                # Execute the agent session
                agent_result = await self._execute_task(task, session_id, db_session)

                # Handle the result
                slot.state = WorkerState.COMPLETING
                await self._handle_result(task, session_id, agent_result, db_session)
                await db_session.commit()

                # Update metrics
                if agent_result.status == "failed":
                    slot.tasks_failed += 1
                else:
                    slot.tasks_completed += 1

                self._logger.info(
                    "worker_task_completed",
                    worker_id=worker_id,
                    task_id=task_id,
                    result_status=agent_result.status,
                )

        except Exception:
            self._logger.exception(
                "worker_execution_error",
                worker_id=worker_id,
                task_id=task_id,
            )
            if slot is not None:
                slot.state = WorkerState.FAILED
                slot.tasks_failed += 1

        finally:
            await self._release_worker(worker_id)

    # ------------------------------------------------------------------
    # Task Assignment (reuses Dispatcher patterns)
    # ------------------------------------------------------------------

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
            return None
        except Exception:
            self._logger.exception("task_assignment_error", task_id=task_id)
            return None

    # ------------------------------------------------------------------
    # Task Execution (reuses Dispatcher patterns)
    # ------------------------------------------------------------------

    async def _execute_task(
        self,
        task: Task,
        session_id: str,
        db_session: AsyncSession,
    ) -> AgentResult:
        """Execute a task by sending it to the agent and parsing the response.

        Args:
            task: The task being executed.
            session_id: The agent session identifier.
            db_session: Active database session.

        Returns:
            Parsed AgentResult from the agent's response.
        """
        task_id = str(task.id)
        self._logger.info(
            "task_execution_started",
            task_id=task_id,
            session_id=session_id,
        )

        try:
            message = self._build_task_message(task)
            raw_response = await self.session_manager.send_message(session_id, message)
            agent_result = parse_agent_result_safe(raw_response)

            self._logger.info(
                "task_execution_completed",
                task_id=task_id,
                session_id=session_id,
                result_status=agent_result.status,
                confidence=agent_result.confidence_score,
            )

            await self.session_manager.end_session(session_id, status="completed")
            return agent_result

        except Exception as e:
            self._logger.exception(
                "task_execution_error",
                task_id=task_id,
                session_id=session_id,
            )

            try:
                await self.session_manager.end_session(session_id, status="failed")
            except Exception:
                self._logger.exception("session_cleanup_error", session_id=session_id)

            return AgentResult(
                status="failed",
                summary=f"Task execution failed: {e}",
                details=str(e),
                confidence_score=0.0,
            )

    # ------------------------------------------------------------------
    # Result Handling (reuses Dispatcher patterns)
    # ------------------------------------------------------------------

    async def _handle_result(
        self,
        task: Task,
        session_id: str,
        agent_result: AgentResult,
        db_session: AsyncSession,
    ) -> None:
        """Route the agent result to the result handler or apply defaults.

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
            await self._default_handle_result(task, agent_result, db_session)

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
            self._logger.info(
                "task_partial_result",
                task_id=task_id,
                summary=agent_result.summary,
            )

    # ------------------------------------------------------------------
    # Health Tracking
    # ------------------------------------------------------------------

    async def _health_loop(self) -> None:
        """Background loop that periodically checks worker health.

        Runs every ``poll_interval * 2`` seconds (health checks are less
        urgent than task dispatch). Identifies stuck workers that have
        exceeded the idle timeout and flags them.
        """
        health_interval = self.poll_interval * 2

        while self._running:
            try:
                await self._check_worker_health()
            except Exception:
                self._logger.exception("health_check_error")

            await asyncio.sleep(health_interval)

    async def _check_worker_health(self) -> list[WorkerSlot]:
        """Check health of all active workers.

        Identifies workers that are stuck (no state change beyond
        the configured idle timeout) and flags them as FAILED.

        Returns:
            List of WorkerSlot instances that are unhealthy.
        """
        now = datetime.now(timezone.utc)
        idle_timeout = self.config.idle_timeout_seconds
        unhealthy: list[WorkerSlot] = []

        for worker_id, slot in list(self._workers.items()):
            slot.last_health_check = now

            # Check for stuck workers
            if slot.state in (WorkerState.RUNNING, WorkerState.ASSIGNED) and slot.started_at:
                elapsed = (now - slot.started_at).total_seconds()
                if elapsed > idle_timeout:
                    self._logger.warning(
                        "worker_stuck",
                        worker_id=worker_id,
                        task_id=slot.task_id,
                        state=slot.state.value,
                        elapsed_seconds=elapsed,
                        idle_timeout=idle_timeout,
                    )
                    slot.state = WorkerState.FAILED
                    unhealthy.append(slot)

            # Check for workers in FAILED state
            elif slot.state == WorkerState.FAILED:
                unhealthy.append(slot)

        if unhealthy:
            self._logger.info(
                "unhealthy_workers_detected",
                count=len(unhealthy),
                worker_ids=[w.worker_id for w in unhealthy],
            )

        return unhealthy

    def get_worker_stats(self) -> dict[str, Any]:
        """Get aggregate statistics for all workers.

        Returns:
            Dictionary containing worker count, state distribution,
            total tasks completed and failed, and per-worker details.
        """
        now = datetime.now(timezone.utc)

        state_counts: dict[str, int] = {}
        total_completed = 0
        total_failed = 0
        worker_details: list[dict[str, Any]] = []

        for slot in self._workers.values():
            state_name = slot.state.value
            state_counts[state_name] = state_counts.get(state_name, 0) + 1
            total_completed += slot.tasks_completed
            total_failed += slot.tasks_failed

            uptime = (now - slot.created_at).total_seconds() if slot.created_at else 0.0

            worker_details.append({
                "worker_id": slot.worker_id,
                "state": state_name,
                "task_id": slot.task_id,
                "session_id": slot.session_id,
                "tasks_completed": slot.tasks_completed,
                "tasks_failed": slot.tasks_failed,
                "uptime_seconds": uptime,
                "worktree_name": slot.worktree_name,
            })

        return {
            "max_workers": self._max_workers,
            "active_workers": self.active_workers,
            "available_slots": self.available_slots,
            "total_workers_created": self._worker_counter,
            "state_distribution": state_counts,
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "workers": worker_details,
        }

    # ------------------------------------------------------------------
    # Shared helpers (same as Dispatcher)
    # ------------------------------------------------------------------

    def _resolve_model_tier(self, model_tier: str | None) -> str:
        """Resolve a model tier preference to a concrete model identifier.

        Args:
            model_tier: Tier preference from the task.

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
        parts = [f"## Task: {task.title}"]
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
