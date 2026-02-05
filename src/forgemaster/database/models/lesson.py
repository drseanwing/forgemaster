"""Lessons learned model for Forgemaster.

Defines the LessonLearned table for capturing knowledge from
completed tasks. Lessons include symptom/root-cause/fix patterns
with vector embeddings for semantic search and tsvector for
full-text search.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgemaster.database.models.base import Base, TimestampMixin


class LessonLearned(TimestampMixin, Base):
    """A captured lesson from task execution.

    Stores symptom, root cause, and fix information along with
    vector embeddings for semantic similarity search and a generated
    tsvector column for PostgreSQL full-text search.

    Attributes:
        id: UUID primary key (from TimestampMixin).
        project_id: Foreign key to the associated project.
        task_id: Foreign key to the originating task.
        symptom: Description of the observed problem.
        root_cause: Identified root cause of the problem.
        fix_applied: Description of the fix that resolved it.
        files_affected: List of file paths involved.
        pattern_tags: Classification tags for the pattern.
        verification_status: Whether the lesson has been verified.
        confidence_score: Confidence in the lesson (0.0 to 1.0).
        symptom_embedding: Vector embedding of the symptom text.
        content_embedding: Vector embedding of full content.
        content_tsv: Generated tsvector for full-text search.
        created_at: Row creation timestamp (from TimestampMixin).
        updated_at: Last modification timestamp (from TimestampMixin).
        archived_at: Timestamp when the lesson was archived.
        project: Relationship to the associated Project.
        task: Relationship to the originating Task.
    """

    __tablename__ = "lessons_learned"

    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("projects.id"),
        nullable=True,
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id"),
        nullable=True,
    )
    symptom: Mapped[str] = mapped_column(Text, nullable=False)
    root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    fix_applied: Mapped[str] = mapped_column(Text, nullable=False)
    files_affected: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
    )
    pattern_tags: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
    )
    verification_status: Mapped[str] = mapped_column(
        Text,
        default="unverified",
        nullable=False,
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        default=0.5,
        nullable=False,
    )
    symptom_embedding = mapped_column(
        Vector(768),
        nullable=True,
    )
    content_embedding = mapped_column(
        Vector(768),
        nullable=True,
    )
    content_tsv = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', symptom || ' ' || root_cause || ' ' || fix_applied)",
            persisted=True,
        ),
    )
    archived_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship(  # noqa: F821
        "Project",
        lazy="selectin",
    )
    task: Mapped["Task"] = relationship(  # noqa: F821
        "Task",
        lazy="selectin",
    )
