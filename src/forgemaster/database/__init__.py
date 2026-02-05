"""Database layer for Forgemaster.

This module handles database connections, session management, and provides
the SQLAlchemy async engine configuration for PostgreSQL with pgvector.

Public API:
    get_engine: Create an AsyncEngine from DatabaseConfig.
    get_session_factory: Create an async_sessionmaker from an engine.
    Base: SQLAlchemy declarative base for all models.
"""

from forgemaster.database.connection import get_engine, get_session_factory
from forgemaster.database.models import (
    AgentSession,
    Base,
    EmbeddingQueueItem,
    LessonLearned,
    Project,
    ProjectStatus,
    SessionStatus,
    Task,
    TaskStatus,
    TimestampMixin,
)

__all__ = [
    "get_engine",
    "get_session_factory",
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
