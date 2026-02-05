"""Initial schema for Forgemaster.

Creates all core tables: projects, tasks, agent_sessions, lessons_learned,
and embedding_queue. Enables the pgvector extension for vector similarity
search.

Revision ID: 001
Revises: None
Create Date: 2025-02-05
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension (P1-024)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create project_status enum
    project_status = sa.Enum(
        "draft", "active", "paused", "completed", "archived",
        name="project_status",
    )
    project_status.create(op.get_bind(), checkfirst=True)

    # Create task_status enum
    task_status = sa.Enum(
        "pending", "ready", "assigned", "running", "review", "done", "failed", "blocked",
        name="task_status",
    )
    task_status.create(op.get_bind(), checkfirst=True)

    # Create session_status enum
    session_status = sa.Enum(
        "initialising", "active", "idle", "completing", "completed", "failed", "killed",
        name="session_status",
    )
    session_status.create(op.get_bind(), checkfirst=True)

    # Projects table
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "draft", "active", "paused", "completed", "archived",
                name="project_status",
                create_type=False,
            ),
            nullable=False,
            server_default="draft",
        ),
        sa.Column("spec_document", JSONB, nullable=True),
        sa.Column("architecture_document", JSONB, nullable=True),
        sa.Column("config", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Tasks table
    op.create_table(
        "tasks",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "pending", "ready", "assigned", "running", "review", "done", "failed", "blocked",
                name="task_status",
                create_type=False,
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("agent_type", sa.Text(), nullable=False),
        sa.Column("model_tier", sa.Text(), nullable=True, server_default="auto"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("100")),
        sa.Column("estimated_minutes", sa.Integer(), nullable=True),
        sa.Column("files_touched", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("dependencies", sa.ARRAY(sa.Uuid()), nullable=True),
        sa.Column("parallel_group", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default=sa.text("3")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add pgvector column via raw SQL (vector type not natively supported by sa.Column)
    op.execute("ALTER TABLE tasks ADD COLUMN description_embedding vector(768)")

    # Agent sessions table
    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "initialising", "active", "idle", "completing", "completed", "failed", "killed",
                name="session_status",
                create_type=False,
            ),
            nullable=False,
            server_default="initialising",
        ),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("worktree_path", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("handover_context", JSONB, nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Lessons learned table
    op.create_table(
        "lessons_learned",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("symptom", sa.Text(), nullable=False),
        sa.Column("root_cause", sa.Text(), nullable=False),
        sa.Column("fix_applied", sa.Text(), nullable=False),
        sa.Column("files_affected", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("pattern_tags", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column(
            "verification_status",
            sa.Text(),
            nullable=False,
            server_default="unverified",
        ),
        sa.Column(
            "confidence_score",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.5"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add pgvector columns and generated tsvector via raw SQL
    op.execute("ALTER TABLE lessons_learned ADD COLUMN symptom_embedding vector(768)")
    op.execute("ALTER TABLE lessons_learned ADD COLUMN content_embedding vector(768)")
    op.execute(
        "ALTER TABLE lessons_learned ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS ("
        "to_tsvector('english', symptom || ' ' || root_cause || ' ' || fix_applied)"
        ") STORED"
    )

    # Embedding queue table
    op.create_table(
        "embedding_queue",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("target_table", sa.Text(), nullable=False),
        sa.Column("target_id", sa.Uuid(), nullable=False),
        sa.Column("target_column", sa.Text(), nullable=False),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("embedding_queue")
    op.drop_table("lessons_learned")
    op.drop_table("agent_sessions")
    op.drop_table("tasks")
    op.drop_table("projects")

    # Drop enum types
    sa.Enum(name="session_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="task_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="project_status").drop(op.get_bind(), checkfirst=True)

    # Drop pgvector extension
    op.execute("DROP EXTENSION IF EXISTS vector")
