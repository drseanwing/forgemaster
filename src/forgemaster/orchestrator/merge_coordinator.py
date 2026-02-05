"""Merge coordinator for worktree branch integration.

This module implements the MergeCoordinator service which manages merging
worktree branches back to the main branch after tasks complete. It provides
a FIFO merge queue, automatic conflict-free merging, and escalation of
conflicting merges to architect agents for resolution guidance.

Merge flow:
1. Completed task calls ``enqueue_merge()`` with branch and file metadata.
2. ``process_queue()`` iterates queued requests in FIFO order.
3. For each request, ``attempt_merge()`` pre-checks for conflicts.
4. If conflict-free, the merge is performed, locks released, and worktree freed.
5. If conflicts exist, the request is marked CONFLICT and optionally escalated
   to an architect agent via ``escalate_to_architect()``.
6. After resolution, ``apply_resolution()`` retries the merge.

Example usage:
    >>> coordinator = MergeCoordinator(
    ...     git_manager=git_manager,
    ...     worktree_pool=worktree_pool,
    ...     file_locker=file_locker,
    ...     session_manager=session_manager,
    ...     context_generator=context_generator,
    ... )
    >>> request = await coordinator.enqueue_merge(
    ...     task_id="task-123",
    ...     worker_id="worker-1",
    ...     worktree_name="1",
    ...     source_branch="worktree/1",
    ...     files_modified=["src/main.py"],
    ... )
    >>> processed = await coordinator.process_queue()
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from forgemaster.agents.session import AgentSessionManager
from forgemaster.context.generator import ContextGenerator
from forgemaster.orchestrator.file_locker import FileLocker
from forgemaster.pipeline.git_ops import GitManager, MergeResult
from forgemaster.pipeline.worktree import WorktreePool

logger = structlog.get_logger(__name__)


class MergeStatus(str, Enum):
    """Status of a merge request in the queue.

    Attributes:
        QUEUED: Request is waiting to be processed.
        MERGING: Merge operation is in progress.
        MERGED: Merge completed successfully.
        CONFLICT: Merge conflicts were detected.
        ESCALATED: Conflicts have been escalated to an architect agent.
        RESOLVED: Architect provided a resolution; ready for retry.
        FAILED: Merge failed after exhausting retries.
    """

    QUEUED = "queued"
    MERGING = "merging"
    MERGED = "merged"
    CONFLICT = "conflict"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    FAILED = "failed"


class MergeRequest(BaseModel):
    """A request to merge a worktree branch back to the target branch.

    Attributes:
        task_id: Identifier of the completed task.
        worker_id: Identifier of the worker slot that ran the task.
        worktree_name: Name of the worktree to release after merge.
        source_branch: Branch to merge from (worktree branch).
        target_branch: Branch to merge into (typically main).
        files_modified: List of files modified by the task.
        status: Current merge status.
        created_at: UTC timestamp when the request was created.
        merged_at: UTC timestamp when the merge completed (None if pending).
        conflict_files: List of file paths with conflicts.
        resolution_notes: Architect's resolution guidance text.
        retry_count: Number of merge retry attempts so far.
        max_retries: Maximum allowed retry attempts.
    """

    task_id: str
    worker_id: str
    worktree_name: str
    source_branch: str
    target_branch: str
    files_modified: list[str]
    status: MergeStatus = MergeStatus.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    merged_at: datetime | None = None
    conflict_files: list[str] = Field(default_factory=list)
    resolution_notes: str | None = None
    retry_count: int = 0
    max_retries: int = 3


class MergeCoordinator:
    """Coordinates merging of worktree branches back to main.

    Manages a queue of merge requests, performs automatic merges for
    conflict-free branches, and escalates conflicts to architect agents
    for resolution guidance.

    The coordinator is thread-safe for asyncio: an internal ``asyncio.Lock``
    serialises queue processing so that only one merge runs at a time,
    preventing race conditions where two merges might conflict on the
    target branch.

    Attributes:
        git_manager: GitManager instance for git operations.
        worktree_pool: WorktreePool for releasing worktrees after merge.
        file_locker: FileLocker for releasing file locks after merge.
        session_manager: Optional AgentSessionManager for architect escalation.
        context_generator: Optional ContextGenerator for building architect prompts.
        target_branch: Target branch for merges (default: ``"main"``).
        max_retries: Default maximum retries for merge requests.
    """

    def __init__(
        self,
        git_manager: GitManager,
        worktree_pool: WorktreePool,
        file_locker: FileLocker,
        session_manager: AgentSessionManager | None = None,
        context_generator: ContextGenerator | None = None,
        target_branch: str = "main",
        max_retries: int = 3,
    ) -> None:
        """Initialize the merge coordinator.

        Args:
            git_manager: GitManager instance for branch and merge operations.
            worktree_pool: WorktreePool for releasing worktrees post-merge.
            file_locker: FileLocker for releasing file locks post-merge.
            session_manager: Optional session manager for spawning architect
                agents when conflict escalation is needed.
            context_generator: Optional context generator for building
                architect system prompts during escalation.
            target_branch: Target branch name for all merges.
            max_retries: Default maximum number of merge retry attempts.
        """
        self.git_manager = git_manager
        self.worktree_pool = worktree_pool
        self.file_locker = file_locker
        self.session_manager = session_manager
        self.context_generator = context_generator
        self.target_branch = target_branch
        self.max_retries = max_retries

        self._queue: list[MergeRequest] = []
        self._lock = asyncio.Lock()
        self._logger = logger.bind(component="MergeCoordinator")

        self._logger.info(
            "merge_coordinator_initialized",
            target_branch=target_branch,
            max_retries=max_retries,
            has_session_manager=session_manager is not None,
            has_context_generator=context_generator is not None,
        )

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    async def enqueue_merge(
        self,
        task_id: str,
        worker_id: str,
        worktree_name: str,
        source_branch: str,
        files_modified: list[str],
    ) -> MergeRequest:
        """Add a merge request to the queue.

        Creates a new ``MergeRequest`` with status ``QUEUED`` and appends
        it to the internal FIFO queue.

        Args:
            task_id: Identifier of the completed task.
            worker_id: Identifier of the worker slot.
            worktree_name: Name of the worktree to release after merge.
            source_branch: Branch to merge from.
            files_modified: List of files modified by the task.

        Returns:
            The created MergeRequest.
        """
        request = MergeRequest(
            task_id=task_id,
            worker_id=worker_id,
            worktree_name=worktree_name,
            source_branch=source_branch,
            target_branch=self.target_branch,
            files_modified=files_modified,
            max_retries=self.max_retries,
        )

        async with self._lock:
            self._queue.append(request)

        self._logger.info(
            "merge_enqueued",
            task_id=task_id,
            worker_id=worker_id,
            source_branch=source_branch,
            target_branch=self.target_branch,
            files_modified_count=len(files_modified),
            queue_depth=len(self._queue),
        )

        return request

    async def process_queue(self) -> list[MergeRequest]:
        """Process all queued merge requests in FIFO order.

        Iterates through the queue, attempting each ``QUEUED`` request.
        Processing is serialised via an internal lock to prevent concurrent
        merges from conflicting on the target branch.

        Returns:
            List of processed requests with updated statuses.
        """
        processed: list[MergeRequest] = []

        async with self._lock:
            # Snapshot queued items to process
            to_process = [r for r in self._queue if r.status == MergeStatus.QUEUED]

            self._logger.info(
                "queue_processing_started",
                queued_count=len(to_process),
                total_in_queue=len(self._queue),
            )

            for request in to_process:
                try:
                    updated = await self._do_attempt_merge(request)
                    processed.append(updated)
                except Exception as e:
                    self._logger.error(
                        "queue_processing_error",
                        task_id=request.task_id,
                        source_branch=request.source_branch,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    request.status = MergeStatus.FAILED
                    processed.append(request)

            self._logger.info(
                "queue_processing_completed",
                processed_count=len(processed),
                merged=[r.task_id for r in processed if r.status == MergeStatus.MERGED],
                conflicted=[r.task_id for r in processed if r.status == MergeStatus.CONFLICT],
                failed=[r.task_id for r in processed if r.status == MergeStatus.FAILED],
            )

        return processed

    # ------------------------------------------------------------------
    # Merge operations
    # ------------------------------------------------------------------

    async def attempt_merge(self, request: MergeRequest) -> MergeRequest:
        """Attempt to merge a single request (public, lock-acquiring).

        Acquires the internal lock before performing the merge to prevent
        concurrent operations on the target branch.

        Args:
            request: The merge request to process.

        Returns:
            The updated MergeRequest with new status.
        """
        async with self._lock:
            return await self._do_attempt_merge(request)

    async def _do_attempt_merge(self, request: MergeRequest) -> MergeRequest:
        """Attempt to merge a single request (internal, assumes lock held).

        Performs the following steps:
        1. Set status to MERGING.
        2. Detect conflicts via ``git_manager.detect_conflicts()``.
        3. If conflicts exist, set status to CONFLICT with conflict details.
        4. If no conflicts, perform ``git_manager.merge()``.
        5. On successful merge, release file locks and worktree.

        Args:
            request: The merge request to process.

        Returns:
            The updated MergeRequest with new status.
        """
        request.status = MergeStatus.MERGING

        self._logger.info(
            "merge_attempt_started",
            task_id=request.task_id,
            source_branch=request.source_branch,
            target_branch=request.target_branch,
            retry_count=request.retry_count,
        )

        # Step 1: Detect conflicts before merging
        try:
            conflicts = self.git_manager.detect_conflicts(
                request.source_branch, request.target_branch
            )
        except Exception as e:
            self._logger.error(
                "conflict_detection_failed",
                task_id=request.task_id,
                source_branch=request.source_branch,
                error=str(e),
            )
            request.status = MergeStatus.FAILED
            return request

        if conflicts:
            request.status = MergeStatus.CONFLICT
            request.conflict_files = conflicts

            self._logger.warning(
                "merge_conflicts_detected",
                task_id=request.task_id,
                source_branch=request.source_branch,
                conflict_count=len(conflicts),
                conflict_files=conflicts,
            )
            return request

        # Step 2: Perform the merge
        try:
            result: MergeResult = self.git_manager.merge(
                request.source_branch, request.target_branch
            )
        except Exception as e:
            self._logger.error(
                "merge_operation_failed",
                task_id=request.task_id,
                source_branch=request.source_branch,
                error=str(e),
            )
            request.status = MergeStatus.FAILED
            return request

        if not result.success:
            request.status = MergeStatus.CONFLICT
            request.conflict_files = result.conflicts

            self._logger.warning(
                "merge_result_conflicts",
                task_id=request.task_id,
                conflict_files=result.conflicts,
            )
            return request

        # Step 3: Merge succeeded - clean up resources
        request.status = MergeStatus.MERGED
        request.merged_at = datetime.now(timezone.utc)

        self._logger.info(
            "merge_successful",
            task_id=request.task_id,
            source_branch=request.source_branch,
            target_branch=request.target_branch,
            commit_sha=result.commit_sha,
        )

        # Release file locks for this task
        await self._release_locks(request)

        # Release worktree back to pool
        await self._release_worktree(request)

        return request

    async def retry_merge(self, request: MergeRequest) -> MergeRequest:
        """Retry a failed or conflicted merge.

        Increments the retry counter and re-attempts the merge. If the
        retry count exceeds ``max_retries``, the request is marked as
        FAILED.

        Args:
            request: The merge request to retry.

        Returns:
            The updated MergeRequest with new status.

        Raises:
            ValueError: If the request has exceeded max retries.
        """
        request.retry_count += 1

        if request.retry_count > request.max_retries:
            request.status = MergeStatus.FAILED

            self._logger.error(
                "merge_max_retries_exceeded",
                task_id=request.task_id,
                retry_count=request.retry_count,
                max_retries=request.max_retries,
            )
            raise ValueError(
                f"Merge request for task '{request.task_id}' has exceeded "
                f"max retries ({request.max_retries})"
            )

        self._logger.info(
            "merge_retry_attempt",
            task_id=request.task_id,
            retry_count=request.retry_count,
            max_retries=request.max_retries,
        )

        # Reset status to QUEUED for re-processing
        request.status = MergeStatus.QUEUED
        request.conflict_files = []

        return await self.attempt_merge(request)

    # ------------------------------------------------------------------
    # Conflict escalation
    # ------------------------------------------------------------------

    async def escalate_to_architect(self, request: MergeRequest) -> str:
        """Escalate a conflicting merge to an architect agent.

        Spawns an architect agent session with context about the merge
        conflict and asks it to analyse the conflicts and suggest a
        resolution strategy.

        Requires ``session_manager`` and ``context_generator`` to be set
        at construction time. If either is missing, raises ``RuntimeError``.

        Args:
            request: The merge request with conflicts to escalate.

        Returns:
            Resolution suggestion text from the architect agent.

        Raises:
            RuntimeError: If session_manager or context_generator is not configured.
        """
        if self.session_manager is None:
            raise RuntimeError(
                "Cannot escalate: session_manager is not configured"
            )
        if self.context_generator is None:
            raise RuntimeError(
                "Cannot escalate: context_generator is not configured"
            )

        self._logger.info(
            "escalating_to_architect",
            task_id=request.task_id,
            conflict_files=request.conflict_files,
            source_branch=request.source_branch,
            target_branch=request.target_branch,
        )

        request.status = MergeStatus.ESCALATED

        # Build architect system prompt with merge conflict context
        system_prompt = self.context_generator.generate_agent_context(
            agent_type="architect",
            task="Analyze merge conflicts and suggest resolution strategy.",
            project={
                "name": "Forgemaster",
                "context": "Parallel task execution system with worktree isolation.",
                "standards": "Python 3.12+, structlog, Pydantic v2.",
            },
        )

        # Create architect session
        session_id = await self.session_manager.start_session(
            task_id=request.task_id,
            agent_type="architect",
            model="opus",
            system_prompt=system_prompt,
        )

        try:
            # Build the conflict analysis message
            conflict_message = self._build_conflict_message(request)

            # Send message and get resolution
            resolution = await self.session_manager.send_message(
                session_id, conflict_message
            )

            self._logger.info(
                "architect_resolution_received",
                task_id=request.task_id,
                session_id=session_id,
                resolution_length=len(resolution),
            )

            request.resolution_notes = resolution

            return resolution

        finally:
            # Always end the session
            await self.session_manager.end_session(session_id, status="completed")

    async def apply_resolution(
        self, request: MergeRequest, resolution: str
    ) -> MergeRequest:
        """Apply an architect's resolution and retry the merge.

        Records the resolution notes on the request, transitions it to
        RESOLVED status, and retries the merge operation.

        Args:
            request: The merge request to resolve.
            resolution: Resolution text from the architect or manual input.

        Returns:
            The updated MergeRequest after retry attempt.
        """
        request.resolution_notes = resolution
        request.status = MergeStatus.RESOLVED

        self._logger.info(
            "resolution_applied",
            task_id=request.task_id,
            resolution_length=len(resolution),
            retry_count=request.retry_count,
        )

        # Retry the merge with the resolution context
        return await self.retry_merge(request)

    # ------------------------------------------------------------------
    # Status and monitoring
    # ------------------------------------------------------------------

    def get_queue_status(self) -> dict[str, Any]:
        """Get merge queue statistics.

        Returns:
            Dictionary containing:
            - total: Total number of requests in the queue.
            - queued: Count of requests waiting to be processed.
            - merging: Count of in-progress merges.
            - merged: Count of successfully merged requests.
            - conflict: Count of requests with unresolved conflicts.
            - escalated: Count of requests escalated to architect.
            - resolved: Count of requests with architect resolution.
            - failed: Count of failed requests.
        """
        status_counts: dict[str, int] = {
            "total": len(self._queue),
            "queued": 0,
            "merging": 0,
            "merged": 0,
            "conflict": 0,
            "escalated": 0,
            "resolved": 0,
            "failed": 0,
        }

        for request in self._queue:
            key = request.status.value
            if key in status_counts:
                status_counts[key] += 1

        return status_counts

    def get_pending_merges(self) -> list[MergeRequest]:
        """Get all unprocessed merge requests.

        Returns:
            List of MergeRequest objects with QUEUED status.
        """
        return [r for r in self._queue if r.status == MergeStatus.QUEUED]

    def get_conflicts(self) -> list[MergeRequest]:
        """Get all merge requests with unresolved conflicts.

        Returns:
            List of MergeRequest objects with CONFLICT or ESCALATED status.
        """
        return [
            r for r in self._queue
            if r.status in (MergeStatus.CONFLICT, MergeStatus.ESCALATED)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _release_locks(self, request: MergeRequest) -> None:
        """Release file locks held by the merged task.

        Args:
            request: The successfully merged request.
        """
        try:
            released = await self.file_locker.release_locks(request.task_id)
            self._logger.info(
                "locks_released_after_merge",
                task_id=request.task_id,
                released_count=released,
            )
        except Exception as e:
            self._logger.error(
                "lock_release_failed",
                task_id=request.task_id,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _release_worktree(self, request: MergeRequest) -> None:
        """Release the worktree back to the pool after merge.

        Args:
            request: The successfully merged request.
        """
        try:
            await self.worktree_pool.release(request.worktree_name)
            self._logger.info(
                "worktree_released_after_merge",
                task_id=request.task_id,
                worktree_name=request.worktree_name,
            )
        except Exception as e:
            self._logger.error(
                "worktree_release_failed",
                task_id=request.task_id,
                worktree_name=request.worktree_name,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _build_conflict_message(self, request: MergeRequest) -> str:
        """Build a message describing the merge conflict for the architect.

        Args:
            request: The merge request with conflicts.

        Returns:
            Formatted message string for the architect agent.
        """
        lines = [
            "## Merge Conflict Analysis Required",
            "",
            f"**Task ID**: {request.task_id}",
            f"**Source Branch**: {request.source_branch}",
            f"**Target Branch**: {request.target_branch}",
            f"**Worker**: {request.worker_id}",
            "",
            "### Conflicting Files",
        ]

        for conflict_file in request.conflict_files:
            lines.append(f"- `{conflict_file}`")

        lines.extend([
            "",
            "### Files Modified by Task",
        ])

        for modified_file in request.files_modified:
            lines.append(f"- `{modified_file}`")

        lines.extend([
            "",
            "### Instructions",
            "",
            "Please analyse the conflicts and provide a resolution strategy.",
            "Consider:",
            "1. Which version of conflicting changes should take priority?",
            "2. Can the changes be combined safely?",
            "3. Are there semantic conflicts beyond textual ones?",
            "4. What manual edits are needed to resolve the conflicts?",
            "",
            "Provide a clear, actionable resolution plan.",
        ])

        return "\n".join(lines)
