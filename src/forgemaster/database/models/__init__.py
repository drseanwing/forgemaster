"""SQLAlchemy ORM models for Forgemaster.

This module defines the database schema including projects, tasks, agent
sessions, lessons learned, and embedding queue tables.

All models use SQLAlchemy 2.0 declarative style with Mapped[] type annotations.
"""

from forgemaster.database.models.base import Base, TimestampMixin
from forgemaster.database.models.embedding import EmbeddingQueueItem
from forgemaster.database.models.lesson import LessonLearned
from forgemaster.database.models.project import Project, ProjectStatus
from forgemaster.database.models.session import AgentSession, SessionStatus
from forgemaster.database.models.task import Task, TaskStatus

__all__ = [
    "Base",
    "TimestampMixin",
    "Project",
    "ProjectStatus",
    "Task",
    "TaskStatus",
    "AgentSession",
    "SessionStatus",
    "LessonLearned",
    "EmbeddingQueueItem",
]
