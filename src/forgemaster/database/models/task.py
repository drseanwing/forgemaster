"""Task model for Forgemaster.

Defines the Task table and TaskStatus enum for tracking individual
work items within a project. Tasks represent discrete units of work
assigned to AI agents, with dependency tracking and state machine
lifecycle management.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgemaster.database.models.base import Base, TimestampMixin


class TaskStatus(enum.Enum):
    """State machine for task lifecycle.

    States:
        pending: Task created but dependencies not yet met.
        ready: All dependencies satisfied, eligible for assignment.
        assigned: Task assigned to an agent but not yet started.
        running: Agent is actively working on the task.
        review: Work complete, pending review.
        done: Task successfully completed.
        failed: Task failed after exhausting retries.
        blocked: Task blocked by external factor.
    """

    pending = "pending"
    ready = "ready"
    assigned = "assigned"
    running = "running"
    review = "review"
    done = "done"
    failed = "failed"
    blocked = "blocked"


class Task(TimestampMixin, Base):
    """A discrete work item within a Forgemaster project.

    Attributes:
        id: UUID primary key (from TimestampMixin).
        project_id: Foreign key to the parent project.
        title: Short description of the task.
        description: Detailed task description and instructions.
        status: Current state in the task lifecycle.
        agent_type: Type of agent required (e.g., 'executor', 'architect').
        model_tier: Model tier preference ('auto', 'haiku', 'sonnet', 'opus').
        priority: Numeric priority (lower = higher priority).
        estimated_minutes: Estimated completion time in minutes.
        files_touched: List of file paths modified by this task.
        dependencies: List of task UUIDs this task depends on.
        parallel_group: Optional group identifier for parallel execution.
        retry_count: Number of retry attempts made.
        max_retries: Maximum allowed retry attempts.
        description_embedding: Vector embedding of the task description.
        created_at: Row creation timestamp (from TimestampMixin).
        started_at: Timestamp when task execution began.
        completed_at: Timestamp when task reached a terminal state.
        project: Relationship to the parent Project.
        sessions: Relationship to associated AgentSession records.
    """

    __tablename__ = "tasks"

    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("projects.id"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[TaskStatus] = mapped_column(
        default=TaskStatus.pending,
        nullable=False,
    )
    agent_type: Mapped[str] = mapped_column(Text, nullable=False)
    model_tier: Mapped[str | None] = mapped_column(
        Text,
        default="auto",
        nullable=True,
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
    )
    estimated_minutes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    files_touched: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
    )
    dependencies: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(Uuid),
        nullable=True,
    )
    parallel_group: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    max_retries: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    description_embedding = mapped_column(
        Vector(768),
        nullable=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        nullable=True,
    )

    # Relationships
    project: Mapped["Project"] = relationship(  # noqa: F821
        "Project",
        lazy="selectin",
    )
    sessions: Mapped[list["AgentSession"]] = relationship(  # noqa: F821
        "AgentSession",
        back_populates="task",
        lazy="selectin",
    )
