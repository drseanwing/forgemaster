"""Agent session model for Forgemaster.

Defines the AgentSession table and SessionStatus enum for tracking
individual agent execution sessions. Each session represents a single
invocation of a Claude agent working on a task.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgemaster.database.models.base import Base, TimestampMixin


class SessionStatus(enum.Enum):
    """State machine for agent session lifecycle.

    States:
        initialising: Session created, agent being set up.
        active: Agent is actively processing.
        idle: Agent is waiting (e.g., for human input or dependency).
        completing: Agent is wrapping up work.
        completed: Session finished successfully.
        failed: Session encountered an unrecoverable error.
        killed: Session was forcibly terminated.
    """

    initialising = "initialising"
    active = "active"
    idle = "idle"
    completing = "completing"
    completed = "completed"
    failed = "failed"
    killed = "killed"


class AgentSession(TimestampMixin, Base):
    """A single agent execution session within Forgemaster.

    Attributes:
        id: UUID primary key (from TimestampMixin).
        task_id: Foreign key to the associated task.
        status: Current session lifecycle state.
        model: Claude model identifier used for this session.
        worktree_path: Git worktree path assigned to this session.
        started_at: Session start timestamp.
        last_activity_at: Timestamp of most recent agent activity.
        ended_at: Session termination timestamp.
        token_count: Total tokens consumed during the session.
        result: Session output/result as JSONB.
        handover_context: Context passed to successor sessions as JSONB.
        error_message: Error description if session failed.
        created_at: Row creation timestamp (from TimestampMixin).
        updated_at: Last modification timestamp (from TimestampMixin).
        task: Relationship to the parent Task.
    """

    __tablename__ = "agent_sessions"

    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id"),
        nullable=True,
    )
    status: Mapped[SessionStatus] = mapped_column(
        default=SessionStatus.initialising,
        nullable=False,
    )
    model: Mapped[str] = mapped_column(Text, nullable=False)
    worktree_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)
    token_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    handover_context: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    task: Mapped["Task"] = relationship(  # noqa: F821
        "Task",
        back_populates="sessions",
        lazy="selectin",
    )
