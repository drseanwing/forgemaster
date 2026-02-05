"""Database indexes for Forgemaster.

Creates performance indexes on frequently queried columns including
composite indexes on tasks, HNSW indexes for vector similarity search,
and GIN indexes for full-text search.

Revision ID: 002
Revises: 001
Create Date: 2025-02-05
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Tasks: composite index on project_id + status for task queue queries
    op.create_index(
        "ix_tasks_project_status",
        "tasks",
        ["project_id", "status"],
    )

    # Tasks: index on priority for ordering task queue
    op.create_index(
        "ix_tasks_priority",
        "tasks",
        ["priority"],
    )

    # Lessons learned: index on project_id for project-scoped queries
    op.create_index(
        "ix_lessons_learned_project_id",
        "lessons_learned",
        ["project_id"],
    )

    # HNSW index on task description embeddings for approximate nearest neighbor search
    op.execute(
        "CREATE INDEX ix_tasks_description_embedding "
        "ON tasks USING hnsw (description_embedding vector_cosine_ops)"
    )

    # HNSW index on lesson symptom embeddings
    op.execute(
        "CREATE INDEX ix_lessons_symptom_embedding "
        "ON lessons_learned USING hnsw (symptom_embedding vector_cosine_ops)"
    )

    # HNSW index on lesson content embeddings
    op.execute(
        "CREATE INDEX ix_lessons_content_embedding "
        "ON lessons_learned USING hnsw (content_embedding vector_cosine_ops)"
    )

    # GIN index on lessons_learned content_tsv for full-text search
    op.execute(
        "CREATE INDEX ix_lessons_content_tsv "
        "ON lessons_learned USING gin (content_tsv)"
    )

    # Embedding queue: index on status for processing queries
    op.create_index(
        "ix_embedding_queue_status",
        "embedding_queue",
        ["status"],
    )

    # Agent sessions: index on task_id for session lookups
    op.create_index(
        "ix_agent_sessions_task_id",
        "agent_sessions",
        ["task_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_sessions_task_id", table_name="agent_sessions")
    op.drop_index("ix_embedding_queue_status", table_name="embedding_queue")
    op.execute("DROP INDEX IF EXISTS ix_lessons_content_tsv")
    op.execute("DROP INDEX IF EXISTS ix_lessons_content_embedding")
    op.execute("DROP INDEX IF EXISTS ix_lessons_symptom_embedding")
    op.execute("DROP INDEX IF EXISTS ix_tasks_description_embedding")
    op.drop_index("ix_lessons_learned_project_id", table_name="lessons_learned")
    op.drop_index("ix_tasks_priority", table_name="tasks")
    op.drop_index("ix_tasks_project_status", table_name="tasks")
