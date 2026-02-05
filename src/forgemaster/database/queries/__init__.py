"""Database query functions for Forgemaster.

This module provides async query functions for all database entities:
- Project CRUD operations
- Task CRUD and dependency-aware selection
- Agent session lifecycle management
- Lesson learned search and retrieval
- Embedding queue management
"""

from forgemaster.database.queries.embedding import (
    enqueue_embedding,
    get_pending_items,
    get_queue_stats,
    mark_failed,
    mark_processed,
)
from forgemaster.database.queries.lesson import (
    create_lesson,
    get_lesson,
    list_lessons,
    search_lessons_by_embedding,
    search_lessons_by_files,
    search_lessons_by_text,
    update_lesson_verification,
)
from forgemaster.database.queries.project import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    update_project,
)
from forgemaster.database.queries.session import (
    create_session,
    end_session,
    get_active_sessions,
    get_idle_sessions,
    get_session,
    list_sessions,
    update_session_activity,
)
from forgemaster.database.queries.task import (
    create_task,
    get_next_task,
    get_ready_tasks,
    get_task,
    increment_retry_count,
    list_tasks,
    update_task_status,
)

__all__ = [
    # Project queries
    "create_project",
    "get_project",
    "list_projects",
    "update_project",
    "delete_project",
    # Task queries
    "create_task",
    "get_task",
    "list_tasks",
    "update_task_status",
    "get_ready_tasks",
    "get_next_task",
    "increment_retry_count",
    # Session queries
    "create_session",
    "get_session",
    "list_sessions",
    "update_session_activity",
    "end_session",
    "get_active_sessions",
    "get_idle_sessions",
    # Lesson queries
    "create_lesson",
    "get_lesson",
    "list_lessons",
    "update_lesson_verification",
    "search_lessons_by_text",
    "search_lessons_by_embedding",
    "search_lessons_by_files",
    # Embedding queue queries
    "enqueue_embedding",
    "get_pending_items",
    "mark_processed",
    "mark_failed",
    "get_queue_stats",
]
