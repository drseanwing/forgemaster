"""Embedding queue model for Forgemaster.

Defines the EmbeddingQueueItem table for asynchronous processing of
text-to-vector embedding requests. Items are queued when text content
is created or updated, then processed by the embedding worker to
populate vector columns in the target tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from forgemaster.database.models.base import Base, TimestampMixin


class EmbeddingQueueItem(TimestampMixin, Base):
    """A queued embedding request awaiting processing.

    Attributes:
        id: UUID primary key (from TimestampMixin).
        target_table: Name of the table containing the target column.
        target_id: UUID of the row to update with the embedding.
        target_column: Name of the vector column to populate.
        source_text: Text content to generate an embedding for.
        status: Processing status ('pending', 'processing', 'completed', 'failed').
        created_at: Row creation timestamp (from TimestampMixin).
        updated_at: Last modification timestamp (from TimestampMixin).
        processed_at: Timestamp when embedding was generated.
        error_message: Error description if processing failed.
    """

    __tablename__ = "embedding_queue"

    target_table: Mapped[str] = mapped_column(Text, nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    target_column: Mapped[str] = mapped_column(Text, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text,
        default="pending",
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
