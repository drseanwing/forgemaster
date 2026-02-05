"""Embedding queue query functions for Forgemaster.

Provides async functions for managing the embedding queue, including
enqueueing items, fetching pending items, marking processing status,
and retrieving queue statistics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.embedding import EmbeddingQueueItem

logger = structlog.get_logger(__name__)


async def enqueue_embedding(
    session: AsyncSession,
    target_table: str,
    target_id: UUID,
    target_column: str,
    source_text: str,
) -> EmbeddingQueueItem:
    """Enqueue an embedding request for asynchronous processing.

    Args:
        session: Active async database session.
        target_table: Name of the table containing the target column.
        target_id: UUID of the row to update with the embedding.
        target_column: Name of the vector column to populate.
        source_text: Text content to generate an embedding for.

    Returns:
        The newly created EmbeddingQueueItem instance.
    """
    item = EmbeddingQueueItem(
        target_table=target_table,
        target_id=target_id,
        target_column=target_column,
        source_text=source_text,
        status="pending",
    )

    async with session.begin():
        session.add(item)
        await session.flush()
        await session.refresh(item)

    logger.info(
        "embedding_enqueued",
        item_id=str(item.id),
        target_table=target_table,
        target_id=str(target_id),
        target_column=target_column,
    )

    return item


async def get_pending_items(
    session: AsyncSession,
    limit: int = 100,
) -> list[EmbeddingQueueItem]:
    """Get pending embedding queue items for processing.

    Args:
        session: Active async database session.
        limit: Maximum number of items to retrieve.

    Returns:
        List of pending EmbeddingQueueItem instances, ordered by creation time.
    """
    stmt = (
        select(EmbeddingQueueItem)
        .where(EmbeddingQueueItem.status == "pending")
        .order_by(EmbeddingQueueItem.created_at.asc())
        .limit(limit)
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def mark_processed(
    session: AsyncSession,
    item_id: UUID,
    embedding: list[float],
) -> EmbeddingQueueItem:
    """Mark an embedding queue item as successfully processed.

    Args:
        session: Active async database session.
        item_id: UUID of the queue item to update.
        embedding: Generated embedding vector.

    Returns:
        The updated EmbeddingQueueItem instance.

    Raises:
        ValueError: If queue item not found.
    """
    stmt_fetch = select(EmbeddingQueueItem).where(EmbeddingQueueItem.id == item_id)
    result = await session.execute(stmt_fetch)
    item = result.scalar_one_or_none()

    if item is None:
        raise ValueError(f"Embedding queue item {item_id} not found")

    # Update the target table with the embedding
    # This requires dynamic SQL based on target_table and target_column
    # For now, we'll just mark the queue item as completed
    # The actual embedding update should be done by the caller

    async with session.begin():
        stmt = (
            update(EmbeddingQueueItem)
            .where(EmbeddingQueueItem.id == item_id)
            .values(
                status="completed",
                processed_at=datetime.now(timezone.utc),
            )
        )
        await session.execute(stmt)
        await session.refresh(item)

    logger.info(
        "embedding_processed",
        item_id=str(item_id),
        target_table=item.target_table,
        target_id=str(item.target_id),
    )

    return item


async def mark_failed(
    session: AsyncSession,
    item_id: UUID,
    error_message: str,
) -> EmbeddingQueueItem:
    """Mark an embedding queue item as failed.

    Args:
        session: Active async database session.
        item_id: UUID of the queue item to update.
        error_message: Error description.

    Returns:
        The updated EmbeddingQueueItem instance.

    Raises:
        ValueError: If queue item not found.
    """
    stmt_fetch = select(EmbeddingQueueItem).where(EmbeddingQueueItem.id == item_id)
    result = await session.execute(stmt_fetch)
    item = result.scalar_one_or_none()

    if item is None:
        raise ValueError(f"Embedding queue item {item_id} not found")

    async with session.begin():
        stmt = (
            update(EmbeddingQueueItem)
            .where(EmbeddingQueueItem.id == item_id)
            .values(
                status="failed",
                processed_at=datetime.now(timezone.utc),
                error_message=error_message,
            )
        )
        await session.execute(stmt)
        await session.refresh(item)

    logger.warning(
        "embedding_failed",
        item_id=str(item_id),
        error_message=error_message,
    )

    return item


async def get_queue_stats(
    session: AsyncSession,
) -> dict[str, Any]:
    """Get statistics about the embedding queue.

    Returns:
        Dictionary with counts by status:
        - pending: Number of pending items
        - processing: Number of items being processed
        - completed: Number of completed items
        - failed: Number of failed items
        - total: Total number of items
    """
    stmt = (
        select(
            EmbeddingQueueItem.status,
            func.count(EmbeddingQueueItem.id).label("count"),
        )
        .group_by(EmbeddingQueueItem.status)
    )

    result = await session.execute(stmt)
    rows = result.all()

    stats: dict[str, Any] = {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "total": 0,
    }

    for row in rows:
        status = row.status
        count = row.count
        if status in stats:
            stats[status] = count
        stats["total"] += count

    return stats
