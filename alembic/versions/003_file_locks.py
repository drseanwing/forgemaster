"""File locks table for parallel task conflict detection.

Creates the file_locks table used for advisory file locking during
parallel task execution. Includes partial index on active locks
(released_at IS NULL) for efficient conflict detection queries.

Revision ID: 003
Revises: 002
Create Date: 2025-02-05
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create lock_type enum
    lock_type = sa.Enum(
        "exclusive", "shared",
        name="lock_type",
    )
    lock_type.create(op.get_bind(), checkfirst=True)

    # Create file_locks table
    op.create_table(
        "file_locks",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column(
            "task_id",
            sa.Uuid(),
            sa.ForeignKey("tasks.id"),
            nullable=False,
        ),
        sa.Column("worker_id", sa.Text(), nullable=False),
        sa.Column(
            "acquired_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "released_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "lock_type",
            sa.Enum(
                "exclusive", "shared",
                name="lock_type",
                create_type=False,
            ),
            nullable=False,
            server_default="exclusive",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Index on file_path for lock lookups
    op.create_index(
        "ix_file_locks_file_path",
        "file_locks",
        ["file_path"],
    )

    # Index on task_id for task-scoped lock queries
    op.create_index(
        "ix_file_locks_task_id",
        "file_locks",
        ["task_id"],
    )

    # Index on worker_id for worker-scoped lock release
    op.create_index(
        "ix_file_locks_worker_id",
        "file_locks",
        ["worker_id"],
    )

    # Partial index on active locks (released_at IS NULL) for fast
    # conflict detection. Only active locks need to be checked.
    op.execute(
        "CREATE INDEX ix_file_locks_active "
        "ON file_locks (file_path, released_at) "
        "WHERE released_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_file_locks_active")
    op.drop_index("ix_file_locks_worker_id", table_name="file_locks")
    op.drop_index("ix_file_locks_task_id", table_name="file_locks")
    op.drop_index("ix_file_locks_file_path", table_name="file_locks")
    op.drop_table("file_locks")

    # Drop enum type
    sa.Enum(name="lock_type").drop(op.get_bind(), checkfirst=True)
