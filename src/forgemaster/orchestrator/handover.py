"""Session handover protocol for context exhaustion scenarios.

Implements detection of context window exhaustion, save-and-exit prompt
injection, handover context persistence, and continuation session spawning.
When an agent session approaches its context limit or times out, this module
orchestrates a graceful handover to a fresh session, preserving task progress
and context across the boundary.

Flow:
    1. ContextExhaustionDetector monitors token usage ratio and session duration.
    2. When threshold is exceeded, HandoverPromptGenerator creates a structured
       save prompt instructing the agent to summarise its progress.
    3. HandoverStore persists the handover context (in-memory for MVP).
    4. SessionHandoverManager spawns a continuation session with injected context.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from forgemaster.agents.session import AgentSessionManager, SessionInfo

from forgemaster.config import AgentConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HandoverReason(str, Enum):
    """Reasons that trigger a session handover.

    Values:
        CONTEXT_EXHAUSTION: Token usage ratio exceeded threshold.
        SESSION_TIMEOUT: Session approaching maximum duration.
        MANUAL_TRIGGER: Handover requested manually.
        ERROR_RECOVERY: Handover initiated due to recoverable errors.
    """

    CONTEXT_EXHAUSTION = "context_exhaustion"
    SESSION_TIMEOUT = "session_timeout"
    MANUAL_TRIGGER = "manual_trigger"
    ERROR_RECOVERY = "error_recovery"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class HandoverTrigger(BaseModel):
    """Trigger information for a session handover.

    Captures the reason, timing, and token metrics at the moment a handover
    is triggered.

    Attributes:
        session_id: The session that triggered the handover.
        task_id: The task the session was working on.
        trigger_reason: Why the handover was triggered.
        token_usage_ratio: Current token usage as a fraction of max context.
        estimated_remaining_tokens: Estimated tokens remaining before exhaustion.
        triggered_at: ISO-8601 timestamp of the trigger event.
    """

    session_id: str
    task_id: str
    trigger_reason: HandoverReason
    token_usage_ratio: float = Field(ge=0.0, le=1.0)
    estimated_remaining_tokens: int = Field(ge=0)
    triggered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SaveExitResponse(BaseModel):
    """Structured response from an agent performing a save-and-exit.

    The departing agent is instructed to output this JSON summary so the
    continuation session knows exactly where to pick up.

    Attributes:
        task_id: The task being worked on.
        progress_summary: Human-readable summary of progress so far.
        files_modified: List of file paths modified in this session.
        remaining_work: List of work items still to be done.
        current_step: Description of the step that was in progress.
        context_data: Arbitrary key-value data the agent wants to pass forward.
    """

    task_id: str
    progress_summary: str
    files_modified: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)
    current_step: str = Field(default="")
    context_data: dict[str, Any] = Field(default_factory=dict)


class HandoverContext(BaseModel):
    """Complete context for a session handover.

    Links the trigger, save-exit response, and task metadata so that a
    continuation session can be spawned with full context.

    Attributes:
        session_id: The original session ID.
        task_id: The task being handed over.
        trigger: The trigger that caused this handover.
        save_exit_response: Parsed save-exit output from the departing agent.
        task_status_snapshot: Snapshot of the task status at handover time.
        branch_name: Git branch the agent was working on.
        worktree_path: Path to the agent's worktree.
        created_at: ISO-8601 timestamp of context creation.
        continuation_session_id: Session ID of the continuation session, once spawned.
    """

    session_id: str
    task_id: str
    trigger: HandoverTrigger
    save_exit_response: SaveExitResponse | None = None
    task_status_snapshot: str = Field(default="in_progress")
    branch_name: str | None = None
    worktree_path: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    continuation_session_id: str | None = None


# ---------------------------------------------------------------------------
# Context Exhaustion Detector
# ---------------------------------------------------------------------------


class ContextExhaustionDetector:
    """Detects when a session is approaching context window exhaustion.

    Evaluates token usage ratio and session duration against configured
    thresholds to determine whether a handover should be triggered.

    Attributes:
        config: Agent configuration with threshold values.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the detector.

        Args:
            config: Agent configuration with context_warning_threshold
                and session_timeout_seconds.
        """
        self.config = config
        self._logger = logger.bind(component="ContextExhaustionDetector")

    def should_trigger_handover(
        self,
        session_info: SessionInfo,
        max_context_tokens: int = 200_000,
    ) -> HandoverTrigger | None:
        """Check whether a handover should be triggered for the given session.

        Evaluates two conditions:
        1. Token usage ratio exceeds ``config.context_warning_threshold``.
        2. Session duration is within 10% of ``config.session_timeout_seconds``.

        Args:
            session_info: Current session information with metrics.
            max_context_tokens: Maximum context window size in tokens.

        Returns:
            A HandoverTrigger if a handover should be initiated, None otherwise.
        """
        total_tokens = session_info.metrics.total_tokens
        usage_ratio = total_tokens / max_context_tokens if max_context_tokens > 0 else 0.0

        # Check context exhaustion
        if usage_ratio >= self.config.context_warning_threshold:
            remaining = self.estimate_remaining_tokens(session_info, max_context_tokens)
            self._logger.warning(
                "context_exhaustion_detected",
                session_id=session_info.session_id,
                usage_ratio=usage_ratio,
                threshold=self.config.context_warning_threshold,
                remaining_tokens=remaining,
            )
            return HandoverTrigger(
                session_id=session_info.session_id,
                task_id=session_info.task_id,
                trigger_reason=HandoverReason.CONTEXT_EXHAUSTION,
                token_usage_ratio=min(usage_ratio, 1.0),
                estimated_remaining_tokens=remaining,
            )

        # Check session timeout (trigger at 90% of timeout)
        duration = session_info.metrics.duration_seconds
        timeout_threshold = self.config.session_timeout_seconds * 0.9
        if duration >= timeout_threshold:
            remaining = self.estimate_remaining_tokens(session_info, max_context_tokens)
            self._logger.warning(
                "session_timeout_approaching",
                session_id=session_info.session_id,
                duration_seconds=duration,
                timeout_seconds=self.config.session_timeout_seconds,
                timeout_threshold=timeout_threshold,
            )
            return HandoverTrigger(
                session_id=session_info.session_id,
                task_id=session_info.task_id,
                trigger_reason=HandoverReason.SESSION_TIMEOUT,
                token_usage_ratio=min(usage_ratio, 1.0),
                estimated_remaining_tokens=remaining,
            )

        return None

    def estimate_remaining_tokens(
        self,
        session_info: SessionInfo,
        max_context_tokens: int = 200_000,
    ) -> int:
        """Estimate the number of tokens remaining before context exhaustion.

        Args:
            session_info: Current session information with metrics.
            max_context_tokens: Maximum context window size in tokens.

        Returns:
            Estimated remaining tokens, clamped to a minimum of 0.
        """
        used = session_info.metrics.total_tokens
        remaining = max_context_tokens - used
        return max(0, remaining)


# ---------------------------------------------------------------------------
# Save-and-Exit Prompt Generator
# ---------------------------------------------------------------------------


class HandoverPromptGenerator:
    """Generates structured prompts for save-and-exit and continuation.

    Produces prompts that instruct agents to output structured JSON summaries
    of their progress, and continuation prompts that inject prior context
    into a new session.
    """

    def __init__(self) -> None:
        """Initialise the prompt generator."""
        self._logger = logger.bind(component="HandoverPromptGenerator")

    def generate_save_prompt(
        self,
        trigger: HandoverTrigger,
        task_context: dict[str, Any],
    ) -> str:
        """Generate a save-and-exit prompt for the departing agent.

        The prompt instructs the agent to output a JSON object matching the
        SaveExitResponse schema so that the continuation session can pick up
        exactly where this session left off.

        Args:
            trigger: The handover trigger that caused this prompt.
            task_context: Additional context about the task (e.g. description,
                files, branch).

        Returns:
            Formatted prompt string.
        """
        task_description = task_context.get("description", "No description available")
        files_list = task_context.get("files", [])
        branch = task_context.get("branch", "unknown")

        files_section = "\n".join(f"  - {f}" for f in files_list) if files_list else "  (none)"

        prompt = (
            "## IMPORTANT: Session Handover Required\n\n"
            f"Your session is being handed over due to: **{trigger.trigger_reason.value}**.\n"
            f"Token usage: {trigger.token_usage_ratio:.1%} "
            f"(~{trigger.estimated_remaining_tokens:,} tokens remaining).\n\n"
            "You MUST immediately output a JSON summary of your current progress "
            "so the next session can continue where you left off.\n\n"
            "### Task Information\n"
            f"- **Task ID**: {trigger.task_id}\n"
            f"- **Branch**: {branch}\n"
            f"- **Description**: {task_description}\n"
            f"- **Known files**:\n{files_section}\n\n"
            "### Required Output Format\n"
            "Output ONLY valid JSON matching this schema:\n"
            "```json\n"
            "{\n"
            f'  "task_id": "{trigger.task_id}",\n'
            '  "progress_summary": "<what you have accomplished so far>",\n'
            '  "files_modified": ["<list of files you modified>"],\n'
            '  "remaining_work": ["<list of work items still to do>"],\n'
            '  "current_step": "<what you were working on when interrupted>",\n'
            '  "context_data": {<any additional data the next session needs>}\n'
            "}\n"
            "```\n\n"
            "Do NOT include any text outside the JSON object."
        )

        self._logger.info(
            "save_prompt_generated",
            task_id=trigger.task_id,
            session_id=trigger.session_id,
            trigger_reason=trigger.trigger_reason.value,
        )

        return prompt

    def generate_continuation_prompt(
        self,
        handover_context: HandoverContext,
    ) -> str:
        """Generate a continuation prompt for the new session.

        Injects the previous session's progress, modified files, remaining
        work, and any context data so the new agent can resume seamlessly.

        Args:
            handover_context: The complete handover context from the
                departing session.

        Returns:
            Formatted continuation prompt string.
        """
        parts: list[str] = [
            "## Session Continuation\n",
            "You are continuing a task from a previous session that was "
            f"handed over due to: **{handover_context.trigger.trigger_reason.value}**.\n",
            f"- **Task ID**: {handover_context.task_id}",
            f"- **Original Session**: {handover_context.session_id}",
            f"- **Task Status**: {handover_context.task_status_snapshot}",
        ]

        if handover_context.branch_name:
            parts.append(f"- **Branch**: {handover_context.branch_name}")
        if handover_context.worktree_path:
            parts.append(f"- **Worktree**: {handover_context.worktree_path}")

        if handover_context.save_exit_response:
            resp = handover_context.save_exit_response
            parts.append(f"\n### Previous Progress\n{resp.progress_summary}")

            if resp.files_modified:
                parts.append("\n### Files Modified")
                for f in resp.files_modified:
                    parts.append(f"- {f}")

            if resp.remaining_work:
                parts.append("\n### Remaining Work")
                for item in resp.remaining_work:
                    parts.append(f"- {item}")

            if resp.current_step:
                parts.append(f"\n### Last Step in Progress\n{resp.current_step}")

            if resp.context_data:
                parts.append(
                    "\n### Additional Context\n"
                    f"```json\n{json.dumps(resp.context_data, indent=2)}\n```"
                )
        else:
            parts.append(
                "\n### Note\nNo save-exit response was captured from the "
                "previous session. Please re-assess the task from the beginning."
            )

        parts.append(
            "\n### Instructions\n"
            "Continue the task from where the previous session left off. "
            "Do NOT repeat work that has already been completed."
        )

        prompt = "\n".join(parts)

        self._logger.info(
            "continuation_prompt_generated",
            task_id=handover_context.task_id,
            original_session=handover_context.session_id,
            has_save_response=handover_context.save_exit_response is not None,
        )

        return prompt


# ---------------------------------------------------------------------------
# Handover Store (in-memory MVP)
# ---------------------------------------------------------------------------


class HandoverStore:
    """Persists handover contexts for retrieval by continuation sessions.

    For MVP, this uses an in-memory dictionary keyed by session ID. A future
    iteration should persist to the database for crash recovery.

    Attributes:
        session_factory: Callable that produces database sessions (reserved
            for future database persistence).
    """

    def __init__(self, session_factory: Callable[[], AsyncSession] | None = None) -> None:
        """Initialise the handover store.

        Args:
            session_factory: Optional async session factory for future
                database persistence. Currently unused (in-memory store).
        """
        self.session_factory = session_factory
        self._store: dict[str, HandoverContext] = {}
        self._logger = logger.bind(component="HandoverStore")

    async def save_handover(self, context: HandoverContext) -> str:
        """Persist a handover context.

        Args:
            context: The handover context to save.

        Returns:
            The session ID used as the storage key.
        """
        self._store[context.session_id] = context

        self._logger.info(
            "handover_context_saved",
            session_id=context.session_id,
            task_id=context.task_id,
            trigger_reason=context.trigger.trigger_reason.value,
        )

        return context.session_id

    async def load_handover(self, session_id: str) -> HandoverContext | None:
        """Load a handover context by session ID.

        Args:
            session_id: The session ID to look up.

        Returns:
            The HandoverContext if found, None otherwise.
        """
        context = self._store.get(session_id)

        if context is None:
            self._logger.debug(
                "handover_context_not_found",
                session_id=session_id,
            )
        else:
            self._logger.debug(
                "handover_context_loaded",
                session_id=session_id,
                task_id=context.task_id,
            )

        return context

    async def list_pending_handovers(
        self,
        task_id: str | None = None,
    ) -> list[HandoverContext]:
        """List handover contexts that have not yet been continued.

        A pending handover is one where ``continuation_session_id`` is None.

        Args:
            task_id: Optional filter by task ID. If None, returns all pending.

        Returns:
            List of pending HandoverContext objects.
        """
        pending: list[HandoverContext] = []
        for ctx in self._store.values():
            if ctx.continuation_session_id is not None:
                continue
            if task_id is not None and ctx.task_id != task_id:
                continue
            pending.append(ctx)

        self._logger.debug(
            "pending_handovers_listed",
            count=len(pending),
            filter_task_id=task_id,
        )

        return pending


# ---------------------------------------------------------------------------
# Session Handover Manager
# ---------------------------------------------------------------------------


class SessionHandoverManager:
    """Orchestrates the full session handover lifecycle.

    Coordinates detection, save-exit prompting, context persistence, and
    continuation session spawning. This is the main entry point for the
    handover protocol.

    Attributes:
        config: Agent configuration.
        session_manager: Agent session manager for session operations.
        handover_store: Store for persisting handover contexts.
        detector: Context exhaustion detector.
        prompt_generator: Save/continuation prompt generator.
    """

    def __init__(
        self,
        config: AgentConfig,
        session_manager: AgentSessionManager,
        handover_store: HandoverStore,
    ) -> None:
        """Initialise the session handover manager.

        Args:
            config: Agent configuration with thresholds and timeouts.
            session_manager: Agent session manager instance.
            handover_store: Store for handover context persistence.
        """
        self.config = config
        self.session_manager = session_manager
        self.handover_store = handover_store
        self.detector = ContextExhaustionDetector(config)
        self.prompt_generator = HandoverPromptGenerator()
        self._logger = logger.bind(component="SessionHandoverManager")

    async def initiate_handover(
        self,
        session_id: str,
        reason: HandoverReason = HandoverReason.CONTEXT_EXHAUSTION,
        task_context: dict[str, Any] | None = None,
    ) -> HandoverContext:
        """Initiate a full handover for a session.

        Performs the following steps:
        1. Get session info from the session manager.
        2. Create or use the provided trigger.
        3. Generate and send the save-exit prompt.
        4. Parse the agent's response into a SaveExitResponse.
        5. End the current session.
        6. Save and return the handover context.

        Args:
            session_id: The session to hand over.
            reason: The reason for the handover.
            task_context: Additional context about the task. If None, a
                minimal context is generated from session info.

        Returns:
            The persisted HandoverContext.

        Raises:
            ValueError: If the session is not found.
        """
        self._logger.info(
            "handover_initiated",
            session_id=session_id,
            reason=reason.value,
        )

        # 1. Get session info
        session_info = self.session_manager.get_session_info(session_id)

        # 2. Create trigger
        total_tokens = session_info.metrics.total_tokens
        max_tokens = 200_000
        usage_ratio = min(total_tokens / max_tokens, 1.0) if max_tokens > 0 else 0.0
        remaining = max(0, max_tokens - total_tokens)

        trigger = HandoverTrigger(
            session_id=session_id,
            task_id=session_info.task_id,
            trigger_reason=reason,
            token_usage_ratio=usage_ratio,
            estimated_remaining_tokens=remaining,
        )

        # 3. Generate and send save prompt
        ctx = task_context or {
            "description": f"Task {session_info.task_id}",
            "files": [],
            "branch": "unknown",
        }
        save_prompt = self.prompt_generator.generate_save_prompt(trigger, ctx)

        save_exit_response: SaveExitResponse | None = None
        try:
            response_text = await self.session_manager.send_message(session_id, save_prompt)

            # 4. Parse response
            save_exit_response = self._parse_save_response(response_text, session_info.task_id)
        except Exception as e:
            self._logger.warning(
                "save_prompt_response_failed",
                session_id=session_id,
                error=str(e),
            )

        # 5. End current session
        try:
            await self.session_manager.end_session(session_id, status="completed")
        except Exception as e:
            self._logger.warning(
                "session_end_failed_during_handover",
                session_id=session_id,
                error=str(e),
            )

        # 6. Build and save handover context
        handover_ctx = HandoverContext(
            session_id=session_id,
            task_id=session_info.task_id,
            trigger=trigger,
            save_exit_response=save_exit_response,
            task_status_snapshot="in_progress",
        )

        await self.handover_store.save_handover(handover_ctx)

        self._logger.info(
            "handover_context_created",
            session_id=session_id,
            task_id=session_info.task_id,
            has_save_response=save_exit_response is not None,
        )

        return handover_ctx

    async def spawn_continuation(
        self,
        handover_context: HandoverContext,
        system_prompt: str = "You are a Forgemaster agent continuing a previous session.",
        agent_type: str | None = None,
        model: str | None = None,
    ) -> str:
        """Spawn a continuation session from a handover context.

        Creates a new session for the same task and injects the previous
        session's context via a continuation prompt.

        Args:
            handover_context: The handover context to continue from.
            system_prompt: System prompt for the new session.
            agent_type: Agent type for the new session. If None, defaults
                to "continuation".
            model: Model for the new session. If None, defaults to "sonnet".

        Returns:
            The new session ID.

        Raises:
            RuntimeError: If session creation or message sending fails.
        """
        self._logger.info(
            "spawning_continuation",
            original_session=handover_context.session_id,
            task_id=handover_context.task_id,
        )

        # 1. Generate continuation prompt
        continuation_prompt = self.prompt_generator.generate_continuation_prompt(handover_context)

        # 2. Start new session
        new_session_id = await self.session_manager.start_session(
            task_id=handover_context.task_id,
            agent_type=agent_type or "continuation",
            model=model or "sonnet",
            system_prompt=system_prompt,
        )

        # 3. Send continuation prompt
        try:
            await self.session_manager.send_message(new_session_id, continuation_prompt)
        except Exception as e:
            self._logger.error(
                "continuation_prompt_send_failed",
                new_session_id=new_session_id,
                error=str(e),
            )
            raise

        # 4. Update handover context with new session ID
        handover_context.continuation_session_id = new_session_id
        await self.handover_store.save_handover(handover_context)

        self._logger.info(
            "continuation_spawned",
            original_session=handover_context.session_id,
            new_session_id=new_session_id,
            task_id=handover_context.task_id,
        )

        return new_session_id

    async def check_and_handle_exhaustion(
        self,
        session_info: SessionInfo,
        task_context: dict[str, Any] | None = None,
    ) -> HandoverContext | None:
        """Convenience method combining detection and handover initiation.

        Checks whether the given session needs a handover. If so, initiates
        the full handover flow and returns the context. Otherwise returns None.

        Args:
            session_info: Current session information.
            task_context: Optional task context for the save prompt.

        Returns:
            HandoverContext if a handover was initiated, None otherwise.
        """
        trigger = self.detector.should_trigger_handover(session_info)
        if trigger is None:
            return None

        self._logger.info(
            "exhaustion_detected_initiating_handover",
            session_id=session_info.session_id,
            trigger_reason=trigger.trigger_reason.value,
        )

        return await self.initiate_handover(
            session_id=session_info.session_id,
            reason=trigger.trigger_reason,
            task_context=task_context,
        )

    def _parse_save_response(
        self,
        response_text: str,
        task_id: str,
    ) -> SaveExitResponse:
        """Parse an agent's save-exit response text into a SaveExitResponse.

        Attempts to extract JSON from the response text. Falls back to
        creating a minimal response if parsing fails.

        Args:
            response_text: Raw response text from the agent.
            task_id: The task ID as a fallback.

        Returns:
            Parsed SaveExitResponse.
        """
        # Try to extract JSON from the response
        text = response_text.strip()

        # Handle markdown code fences
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + len("```")
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
            return SaveExitResponse(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self._logger.warning(
                "save_response_parse_failed",
                task_id=task_id,
                error=str(e),
                response_preview=response_text[:200],
            )
            # Fallback: treat the entire response as a progress summary
            return SaveExitResponse(
                task_id=task_id,
                progress_summary=response_text,
            )
