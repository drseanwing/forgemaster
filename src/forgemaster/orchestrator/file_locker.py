"""Advisory file locking service for parallel task dispatch.

This module implements the FileLocker service which prevents multiple
workers from modifying the same files simultaneously during parallel
task execution. Locks are tracked in the database for persistence and
crash recovery.

Lock semantics:
- **Exclusive locks** prevent any other task from holding a lock
  (exclusive or shared) on the same file.
- **Shared locks** allow multiple tasks to hold shared locks on the
  same file concurrently, but are denied if an exclusive lock is held.
- Lock acquisition is **atomic**: all locks for a task are acquired in
  a single transaction, or none are (all-or-nothing semantics).

Example:
    >>> locker = FileLocker(session_factory)
    >>> conflicts = await locker.detect_conflicts("task-1", ["src/main.py"])
    >>> if not conflicts:
    ...     acquired = await locker.acquire_locks("task-1", "worker-1", ["src/main.py"])
    >>> released = await locker.release_locks("task-1")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.file_lock import FileLock, LockType

logger = structlog.get_logger(__name__)

# Type alias matching dispatcher convention
SessionFactory = Callable[[], AsyncSession]


class LockConflict(BaseModel):
    """Describes a file conflict between tasks.

    Returned by ``detect_conflicts()`` when a requested file is already
    locked by another task.

    Attributes:
        file_path: The file path that conflicts.
        held_by_task_id: UUID string of the task holding the lock.
        held_by_worker_id: Worker slot identifier holding the lock.
        requesting_task_id: UUID string of the task requesting the lock.
        lock_type: The type of the existing lock.
    """

    file_path: str
    held_by_task_id: str
    held_by_worker_id: str
    requesting_task_id: str
    lock_type: LockType


class FileLocker:
    """Advisory file locking for parallel task dispatch.

    Prevents multiple workers from modifying the same files simultaneously.
    Tracks locks in the database for persistence and crash recovery.

    The locker enforces all-or-nothing lock acquisition: if any requested
    file is already locked (and the lock types conflict), no locks are
    acquired. This avoids partial locking scenarios that could lead to
    deadlocks.

    Attributes:
        session_factory: Callable that produces async database sessions.
    """

    def __init__(self, session_factory: SessionFactory) -> None:
        """Initialize the file locker.

        Args:
            session_factory: Callable returning new AsyncSession instances.
                Must be an async context manager (``async with session_factory() as session``).
        """
        self.session_factory = session_factory
        self._logger = logger.bind(component="FileLocker")

    async def acquire_locks(
        self,
        task_id: str,
        worker_id: str,
        file_paths: list[str],
        lock_type: LockType = LockType.EXCLUSIVE,
    ) -> bool:
        """Acquire locks for all specified files atomically.

        Checks for conflicts and, if none exist, creates lock records for
        every file in a single transaction. If any file is already locked
        with a conflicting lock type, no locks are created.

        Args:
            task_id: UUID string of the task requesting locks.
            worker_id: Identifier of the worker slot (e.g. ``"worker-1"``).
            file_paths: List of relative file paths to lock.
            lock_type: Type of lock to acquire. Defaults to ``EXCLUSIVE``.

        Returns:
            True if all locks were acquired successfully, False if any
            conflict prevented acquisition.
        """
        if not file_paths:
            return True

        # Normalise paths for consistent matching
        normalised = [self._normalise_path(p) for p in file_paths]

        async with self.session_factory() as session:
            # Check for conflicts within the transaction
            conflicts = await self._detect_conflicts_in_session(
                session, task_id, normalised, lock_type
            )

            if conflicts:
                self._logger.warning(
                    "lock_acquisition_denied",
                    task_id=task_id,
                    worker_id=worker_id,
                    conflict_count=len(conflicts),
                    conflicting_files=[c.file_path for c in conflicts],
                )
                return False

            # Acquire all locks atomically
            now = datetime.now(timezone.utc)
            for path in normalised:
                lock = FileLock(
                    file_path=path,
                    task_id=task_id,
                    worker_id=worker_id,
                    acquired_at=now,
                    lock_type=lock_type,
                )
                session.add(lock)

            await session.commit()

            self._logger.info(
                "locks_acquired",
                task_id=task_id,
                worker_id=worker_id,
                lock_count=len(normalised),
                lock_type=lock_type.value,
                file_paths=normalised,
            )
            return True

    async def release_locks(self, task_id: str) -> int:
        """Release all locks held by a task.

        Marks all active locks for the given task as released by setting
        the ``released_at`` timestamp. Does not delete the records; they
        are preserved for audit and debugging purposes.

        Args:
            task_id: UUID string of the task whose locks to release.

        Returns:
            Number of locks released.
        """
        async with self.session_factory() as session:
            now = datetime.now(timezone.utc)

            stmt = (
                update(FileLock)
                .where(
                    FileLock.task_id == task_id,
                    FileLock.released_at.is_(None),
                )
                .values(released_at=now)
            )
            result = await session.execute(stmt)
            await session.commit()

            count = result.rowcount  # type: ignore[union-attr]

            self._logger.info(
                "locks_released",
                task_id=task_id,
                count=count,
            )
            return count

    async def release_worker_locks(self, worker_id: str) -> int:
        """Release all locks held by a worker.

        Used for crash recovery when a worker fails or is terminated
        without cleanly releasing its locks.

        Args:
            worker_id: Identifier of the worker whose locks to release.

        Returns:
            Number of locks released.
        """
        async with self.session_factory() as session:
            now = datetime.now(timezone.utc)

            stmt = (
                update(FileLock)
                .where(
                    FileLock.worker_id == worker_id,
                    FileLock.released_at.is_(None),
                )
                .values(released_at=now)
            )
            result = await session.execute(stmt)
            await session.commit()

            count = result.rowcount  # type: ignore[union-attr]

            self._logger.info(
                "worker_locks_released",
                worker_id=worker_id,
                count=count,
            )
            return count

    async def detect_conflicts(
        self,
        task_id: str,
        file_paths: list[str],
        lock_type: LockType = LockType.EXCLUSIVE,
    ) -> list[LockConflict]:
        """Check if files would conflict with existing locks.

        Queries active locks (released_at IS NULL) and determines which
        of the requested files are already locked by another task with
        a conflicting lock type.

        Conflict rules:
        - Exclusive requested + any existing lock -> conflict
        - Shared requested + existing exclusive lock -> conflict
        - Shared requested + existing shared lock -> no conflict

        Args:
            task_id: UUID string of the task requesting locks.
            file_paths: List of relative file paths to check.
            lock_type: Type of lock being requested. Defaults to ``EXCLUSIVE``.

        Returns:
            List of LockConflict instances describing each conflict.
            Empty list means no conflicts.
        """
        if not file_paths:
            return []

        normalised = [self._normalise_path(p) for p in file_paths]

        async with self.session_factory() as session:
            return await self._detect_conflicts_in_session(
                session, task_id, normalised, lock_type
            )

    async def get_active_locks(self) -> list[FileLock]:
        """Get all currently held (active) locks.

        Active locks are those where ``released_at`` is NULL.

        Returns:
            List of FileLock instances representing active locks.
        """
        async with self.session_factory() as session:
            stmt = select(FileLock).where(FileLock.released_at.is_(None))
            result = await session.execute(stmt)
            locks = list(result.scalars().all())

            self._logger.debug(
                "active_locks_queried",
                count=len(locks),
            )
            return locks

    async def get_locks_for_task(self, task_id: str) -> list[FileLock]:
        """Get all active locks held by a specific task.

        Args:
            task_id: UUID string of the task.

        Returns:
            List of FileLock instances for the task.
        """
        async with self.session_factory() as session:
            stmt = select(FileLock).where(
                FileLock.task_id == task_id,
                FileLock.released_at.is_(None),
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def cleanup_stale_locks(self, max_age_seconds: int = 3600) -> int:
        """Remove locks older than the specified maximum age.

        Stale locks typically result from workers that crashed or were
        killed without cleanly releasing their locks. This method marks
        them as released so they no longer block new lock acquisitions.

        Args:
            max_age_seconds: Maximum age in seconds before a lock is
                considered stale. Defaults to 3600 (1 hour).

        Returns:
            Number of stale locks cleaned up.
        """
        async with self.session_factory() as session:
            now = datetime.now(timezone.utc)
            cutoff = datetime.fromtimestamp(
                now.timestamp() - max_age_seconds,
                tz=timezone.utc,
            )

            stmt = (
                update(FileLock)
                .where(
                    FileLock.released_at.is_(None),
                    FileLock.acquired_at < cutoff,
                )
                .values(released_at=now)
            )
            result = await session.execute(stmt)
            await session.commit()

            count = result.rowcount  # type: ignore[union-attr]

            if count > 0:
                self._logger.warning(
                    "stale_locks_cleaned",
                    count=count,
                    max_age_seconds=max_age_seconds,
                )
            else:
                self._logger.debug(
                    "no_stale_locks",
                    max_age_seconds=max_age_seconds,
                )

            return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _detect_conflicts_in_session(
        self,
        session: AsyncSession,
        task_id: str,
        normalised_paths: list[str],
        lock_type: LockType,
    ) -> list[LockConflict]:
        """Detect conflicts within an existing database session.

        This internal method is used by both ``detect_conflicts()`` and
        ``acquire_locks()`` to check for conflicting locks within the
        same transaction boundary.

        Args:
            session: Active async database session.
            task_id: UUID string of the requesting task.
            normalised_paths: Normalised file paths to check.
            lock_type: The type of lock being requested.

        Returns:
            List of LockConflict instances for each detected conflict.
        """
        # Query active locks on any of the requested files held by
        # OTHER tasks (same-task locks are not considered conflicts)
        stmt = select(FileLock).where(
            FileLock.file_path.in_(normalised_paths),
            FileLock.released_at.is_(None),
            FileLock.task_id != task_id,
        )
        result = await session.execute(stmt)
        active_locks = result.scalars().all()

        conflicts: list[LockConflict] = []
        for existing_lock in active_locks:
            if self._locks_conflict(lock_type, existing_lock.lock_type):
                conflicts.append(
                    LockConflict(
                        file_path=existing_lock.file_path,
                        held_by_task_id=str(existing_lock.task_id),
                        held_by_worker_id=existing_lock.worker_id,
                        requesting_task_id=task_id,
                        lock_type=existing_lock.lock_type,
                    )
                )

        return conflicts

    @staticmethod
    def _locks_conflict(requested: LockType, existing: LockType) -> bool:
        """Determine whether two lock types conflict.

        Conflict matrix:
            - Exclusive requested + any existing -> conflict
            - Shared requested + Exclusive existing -> conflict
            - Shared requested + Shared existing -> no conflict

        Args:
            requested: The lock type being requested.
            existing: The lock type already held.

        Returns:
            True if the lock types conflict, False otherwise.
        """
        if requested == LockType.EXCLUSIVE:
            return True
        if existing == LockType.EXCLUSIVE:
            return True
        # Both are SHARED
        return False

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Normalise a file path for consistent lock matching.

        Strips leading/trailing whitespace and normalises separators
        to forward slashes. Removes leading ``./`` prefix if present.

        Args:
            path: Raw file path string.

        Returns:
            Normalised path string.
        """
        normalised = path.strip().replace("\\", "/")
        if normalised.startswith("./"):
            normalised = normalised[2:]
        return normalised
