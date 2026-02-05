"""File lock model for Forgemaster.

Defines the FileLock table and LockType enum for advisory file locking
during parallel task execution. File locks prevent multiple workers from
modifying the same files simultaneously, ensuring consistency when tasks
run concurrently across git worktrees.

Locks are advisory (not enforced at the filesystem level) and are tracked
in the database for persistence and crash recovery.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgemaster.database.models.base import Base, TimestampMixin


class LockType(str, enum.Enum):
    """Type of file lock.

    Attributes:
        EXCLUSIVE: Only one task may hold an exclusive lock on a file.
            Other exclusive and shared lock requests will be denied.
        SHARED: Multiple tasks may hold shared locks on the same file
            concurrently. Shared locks are denied if an exclusive lock
            is already held.
    """

    EXCLUSIVE = "exclusive"
    SHARED = "shared"


class FileLock(TimestampMixin, Base):
    """An advisory file lock held by a task during parallel execution.

    File locks are acquired atomically before a task is dispatched to a
    worker. All locks for a task are acquired in a single transaction
    (all-or-nothing). Locks are released when the task completes,
    fails, or is cleaned up due to staleness.

    Attributes:
        id: UUID primary key (from TimestampMixin).
        file_path: The relative file path being locked.
        task_id: Foreign key to the task holding this lock.
        worker_id: Identifier of the worker slot holding the lock.
        acquired_at: UTC timestamp when the lock was acquired.
        released_at: UTC timestamp when the lock was released.
            None indicates the lock is still active.
        lock_type: Whether the lock is exclusive or shared.
        created_at: Row creation timestamp (from TimestampMixin).
        updated_at: Last modification timestamp (from TimestampMixin).
        task: Relationship to the parent Task.
    """

    __tablename__ = "file_locks"

    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    task_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("tasks.id"),
        nullable=False,
    )
    worker_id: Mapped[str] = mapped_column(Text, nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    lock_type: Mapped[LockType] = mapped_column(
        default=LockType.EXCLUSIVE,
        nullable=False,
    )

    # Relationships
    task: Mapped["Task"] = relationship(  # noqa: F821
        "Task",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_file_locks_file_path", "file_path"),
        Index("ix_file_locks_task_id", "task_id"),
        Index("ix_file_locks_worker_id", "worker_id"),
        Index(
            "ix_file_locks_active",
            "file_path",
            "released_at",
            postgresql_where=released_at.is_(None),  # type: ignore[arg-type]
        ),
    )
