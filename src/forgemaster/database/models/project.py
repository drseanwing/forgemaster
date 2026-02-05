"""Project model for Forgemaster.

Defines the Project table and ProjectStatus enum for tracking
development projects managed by the orchestration system.

Each project contains a specification document, architecture document,
and configuration stored as JSONB, along with lifecycle status tracking.
"""

from __future__ import annotations

import enum
from typing import Any

from sqlalchemy import Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from forgemaster.database.models.base import Base, TimestampMixin


class ProjectStatus(enum.Enum):
    """Lifecycle status for a project.

    States:
        draft: Initial state, project is being defined.
        active: Project is actively being worked on.
        paused: Work temporarily suspended.
        completed: All tasks finished successfully.
        archived: Project archived for reference.
    """

    draft = "draft"
    active = "active"
    paused = "paused"
    completed = "completed"
    archived = "archived"


class Project(TimestampMixin, Base):
    """A development project managed by Forgemaster.

    Attributes:
        id: UUID primary key (from TimestampMixin).
        name: Human-readable project name.
        status: Current lifecycle status.
        spec_document: Project specification as JSONB.
        architecture_document: Architecture description as JSONB.
        config: Project-specific configuration as JSONB.
        created_at: Row creation timestamp (from TimestampMixin).
        updated_at: Last modification timestamp (from TimestampMixin).
    """

    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[ProjectStatus] = mapped_column(
        default=ProjectStatus.draft,
        nullable=False,
    )
    spec_document: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    architecture_document: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
    )
