"""Result handler for agent task execution in Forgemaster orchestrator.

This module processes agent results after task execution, handling success,
partial completion, and failure outcomes. It also extracts lessons learned
from agent results and persists them to the knowledge base with embedding
queue entries for future semantic search.
"""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.agents.result_schema import AgentResult, LessonLearned as LessonSchema
from forgemaster.config import AgentConfig
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.database.queries.embedding import enqueue_embedding
from forgemaster.database.queries.lesson import create_lesson
from forgemaster.database.queries.task import increment_retry_count
from forgemaster.orchestrator.state_machine import InvalidTransitionError, TaskStateMachine

logger = structlog.get_logger(__name__)


class ResultHandler:
    """Handles agent execution results and drives post-execution logic.

    Responsible for:
    - Transitioning tasks based on result status (success, partial, failure).
    - Incrementing retry counts and enforcing max-retry limits.
    - Extracting lessons learned from agent results.
    - Persisting lessons to the database and enqueuing embeddings.

    Attributes:
        config: Agent configuration with retry limits.
        state_machine: Task state machine for transitions.
    """

    def __init__(
        self,
        config: AgentConfig,
        state_machine: TaskStateMachine,
    ) -> None:
        """Initialize the result handler.

        Args:
            config: Agent configuration containing max_retries and other limits.
            state_machine: Task lifecycle state machine for transitions.
        """
        self.config = config
        self.state_machine = state_machine
        self._logger = logger.bind(component="ResultHandler")

    async def handle_result(
        self,
        task: Task,
        db_session: AsyncSession,
        agent_result: AgentResult,
    ) -> None:
        """Process an agent's execution result for a task.

        Routes to the appropriate handler based on the result status:
        - ``success``: Transition task to REVIEW, extract lessons.
        - ``partial``: Keep task RUNNING for continuation.
        - ``failed``: Increment retry counter; transition to FAILED if exhausted.

        Args:
            task: The task that was executed.
            db_session: Active database session for persistence.
            agent_result: Structured result from the agent.
        """
        task_id = str(task.id)

        self._logger.info(
            "handling_result",
            task_id=task_id,
            result_status=agent_result.status,
            confidence=agent_result.confidence_score,
            files_modified=len(agent_result.files_modified),
            issues_found=len(agent_result.issues_discovered),
            lessons_count=len(agent_result.lessons_learned),
        )

        if agent_result.status == "success":
            await self._handle_success(task, db_session, agent_result)
        elif agent_result.status == "partial":
            await self._handle_partial(task, db_session, agent_result)
        elif agent_result.status == "failed":
            await self._handle_failure(task, db_session, agent_result)
        else:
            self._logger.warning(
                "unknown_result_status",
                task_id=task_id,
                status=agent_result.status,
            )

    async def _handle_success(
        self,
        task: Task,
        db_session: AsyncSession,
        agent_result: AgentResult,
    ) -> None:
        """Handle a successful agent result.

        Transitions the task to REVIEW and extracts any lessons learned.

        Args:
            task: The completed task.
            db_session: Active database session.
            agent_result: Successful agent result.
        """
        task_id = str(task.id)

        try:
            await self.state_machine.transition(task_id, TaskStatus.review, db_session)
        except InvalidTransitionError as e:
            self._logger.error(
                "success_transition_failed",
                task_id=task_id,
                error=str(e),
            )
            return

        self._logger.info(
            "task_success",
            task_id=task_id,
            summary=agent_result.summary,
            confidence=agent_result.confidence_score,
        )

        # Extract lessons from the result
        if agent_result.lessons_learned:
            await self._extract_lessons(task, db_session, agent_result)

    async def _handle_partial(
        self,
        task: Task,
        db_session: AsyncSession,
        agent_result: AgentResult,
    ) -> None:
        """Handle a partial agent result.

        The task remains in RUNNING status. The partial result details
        can serve as handover context for a continuation session.

        Args:
            task: The partially completed task.
            db_session: Active database session.
            agent_result: Partial agent result.
        """
        task_id = str(task.id)

        self._logger.info(
            "task_partial",
            task_id=task_id,
            summary=agent_result.summary,
            confidence=agent_result.confidence_score,
        )

        # Store handover context in the session record for continuation
        from forgemaster.database.queries.session import list_sessions

        sessions = await list_sessions(db_session, task_id=task.id)
        if sessions:
            latest_session = sessions[0]
            latest_session.handover_context = {
                "summary": agent_result.summary,
                "details": agent_result.details,
                "files_modified": agent_result.files_modified,
                "confidence": agent_result.confidence_score,
            }
            await db_session.flush()

        # Extract any lessons from partial work
        if agent_result.lessons_learned:
            await self._extract_lessons(task, db_session, agent_result)

    async def _handle_failure(
        self,
        task: Task,
        db_session: AsyncSession,
        agent_result: AgentResult,
    ) -> None:
        """Handle a failed agent result.

        Increments the retry count. If max retries are exceeded, transitions
        the task to FAILED. Otherwise the task remains RUNNING, eligible
        for retry on the next dispatch cycle (after being transitioned
        RUNNING -> FAILED -> READY by the orchestrator).

        Args:
            task: The failed task.
            db_session: Active database session.
            agent_result: Failed agent result.
        """
        task_id = str(task.id)
        max_retries = task.max_retries or self.config.max_retries

        # Increment retry count
        updated_task = await increment_retry_count(db_session, task.id)
        current_retries = updated_task.retry_count

        self._logger.warning(
            "task_failure",
            task_id=task_id,
            summary=agent_result.summary,
            retry_count=current_retries,
            max_retries=max_retries,
            issues_found=len(agent_result.issues_discovered),
        )

        if current_retries >= max_retries:
            # Max retries exhausted - transition to FAILED
            try:
                await self.state_machine.transition(
                    task_id, TaskStatus.failed, db_session
                )
                self._logger.error(
                    "task_max_retries_exceeded",
                    task_id=task_id,
                    retry_count=current_retries,
                    max_retries=max_retries,
                )
            except InvalidTransitionError as e:
                self._logger.error(
                    "failure_transition_failed",
                    task_id=task_id,
                    error=str(e),
                )
        else:
            # Transition RUNNING -> FAILED -> READY for retry
            try:
                await self.state_machine.transition(
                    task_id, TaskStatus.failed, db_session
                )
                await self.state_machine.transition(
                    task_id, TaskStatus.ready, db_session
                )
                self._logger.info(
                    "task_queued_for_retry",
                    task_id=task_id,
                    retry_count=current_retries,
                    max_retries=max_retries,
                )
            except InvalidTransitionError as e:
                self._logger.error(
                    "retry_transition_failed",
                    task_id=task_id,
                    error=str(e),
                )

        # Extract lessons even from failures - they are often the most valuable
        if agent_result.lessons_learned:
            await self._extract_lessons(task, db_session, agent_result)

    async def _extract_lessons(
        self,
        task: Task,
        db_session: AsyncSession,
        agent_result: AgentResult,
    ) -> None:
        """Extract and persist lessons learned from an agent result.

        Parses each LessonLearned entry from the agent result, creates a
        database record, and enqueues embedding generation for semantic search.

        Args:
            task: The task that produced the lessons.
            db_session: Active database session.
            agent_result: Agent result containing lessons_learned entries.
        """
        task_id = str(task.id)
        project_id = task.project_id

        if not project_id:
            self._logger.warning(
                "lesson_extraction_skipped",
                task_id=task_id,
                reason="no project_id",
            )
            return

        extracted_count = 0

        for lesson_data in agent_result.lessons_learned:
            try:
                lesson = await self._persist_lesson(
                    task=task,
                    db_session=db_session,
                    lesson_data=lesson_data,
                    agent_result=agent_result,
                )

                # Enqueue embedding for the lesson content
                content_text = (
                    f"{lesson_data.context} {lesson_data.observation} "
                    f"{lesson_data.recommendation}"
                )
                await enqueue_embedding(
                    session=db_session,
                    target_table="lessons_learned",
                    target_id=lesson.id,
                    target_column="content_embedding",
                    source_text=content_text,
                )

                # Also enqueue symptom embedding for symptom-based search
                await enqueue_embedding(
                    session=db_session,
                    target_table="lessons_learned",
                    target_id=lesson.id,
                    target_column="symptom_embedding",
                    source_text=lesson_data.context,
                )

                extracted_count += 1

            except Exception:
                self._logger.exception(
                    "lesson_extraction_error",
                    task_id=task_id,
                    lesson_context=lesson_data.context[:100],
                )

        self._logger.info(
            "lessons_extracted",
            task_id=task_id,
            extracted_count=extracted_count,
            total_lessons=len(agent_result.lessons_learned),
        )

    async def _persist_lesson(
        self,
        task: Task,
        db_session: AsyncSession,
        lesson_data: LessonSchema,
        agent_result: AgentResult,
    ) -> Any:
        """Persist a single lesson learned to the database.

        Maps the agent's LessonLearned schema to the database model fields:
        - context -> symptom (the situation/problem observed)
        - observation -> root_cause (what was discovered)
        - recommendation -> fix_applied (the recommended solution)

        Args:
            task: The originating task.
            db_session: Active database session.
            lesson_data: Lesson data from the agent result.
            agent_result: Full agent result for additional context.

        Returns:
            The created LessonLearned database record.
        """
        lesson = await create_lesson(
            session=db_session,
            project_id=task.project_id,
            task_id=task.id,
            symptom=lesson_data.context,
            root_cause=lesson_data.observation,
            fix_applied=lesson_data.recommendation,
            files_affected=agent_result.files_modified or None,
            confidence_score=agent_result.confidence_score,
        )

        self._logger.debug(
            "lesson_persisted",
            lesson_id=str(lesson.id),
            task_id=str(task.id),
            context=lesson_data.context[:80],
        )

        return lesson
