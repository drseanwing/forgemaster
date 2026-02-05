"""Embedding queue worker for background processing.

This module provides a worker that processes the embedding_queue table,
generating embeddings for pending items and updating target tables.

The worker polls the queue at configurable intervals, processes items in batches,
and handles failures with error tracking.

Example usage:
    >>> from forgemaster.intelligence.embedding_worker import EmbeddingWorker
    >>> from forgemaster.intelligence.embeddings import EmbeddingService
    >>>
    >>> worker = EmbeddingWorker(
    ...     embedding_service=service,
    ...     queue_repository=queue_repo,
    ...     batch_size=10,
    ...     poll_interval_seconds=5.0
    ... )
    >>> await worker.run()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import structlog

from forgemaster.intelligence.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


@dataclass
class QueueItem:
    """Represents an item in the embedding queue.

    Attributes:
        id: Unique queue item ID
        target_table: Name of table to update with embedding
        target_id: ID of record in target table
        text: Text to embed
        status: Processing status (pending, processing, completed, failed)
        error_message: Optional error message if failed
        created_at: Timestamp when queued
        updated_at: Timestamp of last update
    """

    id: int
    target_table: str
    target_id: int
    text: str
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class QueueRepository(Protocol):
    """Protocol for queue database operations.

    This protocol defines the interface that the worker expects from
    the database layer. The actual implementation will be provided
    by the database subsystem.
    """

    async def get_pending_items(self, limit: int) -> list[QueueItem]:
        """Fetch pending items from queue.

        Args:
            limit: Maximum number of items to fetch

        Returns:
            List of pending queue items
        """
        ...

    async def mark_processing(self, item_id: int) -> None:
        """Mark item as currently being processed.

        Args:
            item_id: Queue item ID
        """
        ...

    async def update_embedding(
        self, target_table: str, target_id: int, embedding: list[float]
    ) -> None:
        """Update target table with generated embedding.

        Args:
            target_table: Name of table to update
            target_id: ID of record in target table
            embedding: Generated embedding vector
        """
        ...

    async def mark_completed(self, item_id: int) -> None:
        """Mark item as successfully completed.

        Args:
            item_id: Queue item ID
        """
        ...

    async def mark_failed(self, item_id: int, error_message: str) -> None:
        """Mark item as failed with error message.

        Args:
            item_id: Queue item ID
            error_message: Error description
        """
        ...


class EmbeddingWorker:
    """Background worker for processing embedding queue.

    This worker continuously polls the embedding_queue table, processes
    pending items in batches, and updates target tables with generated
    embeddings.

    Attributes:
        embedding_service: Service for generating embeddings
        queue_repository: Repository for queue database operations
        batch_size: Number of items to process per cycle
        poll_interval_seconds: Delay between polling cycles
        is_running: Flag indicating if worker is active
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        queue_repository: QueueRepository,
        batch_size: int = 10,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        """Initialize embedding worker.

        Args:
            embedding_service: Service for generating embeddings
            queue_repository: Repository for queue operations
            batch_size: Items to process per cycle (default: 10)
            poll_interval_seconds: Polling interval (default: 5.0)
        """
        self.embedding_service = embedding_service
        self.queue_repository = queue_repository
        self.batch_size = batch_size
        self.poll_interval_seconds = poll_interval_seconds
        self.is_running = False

        logger.info(
            "embedding_worker_initialized",
            batch_size=batch_size,
            poll_interval=poll_interval_seconds,
        )

    async def run(self) -> None:
        """Start the worker main loop.

        Continuously polls for pending items and processes them until
        stop() is called.
        """
        self.is_running = True
        logger.info("embedding_worker_started")

        try:
            while self.is_running:
                try:
                    await self.process_queue()
                except Exception as e:
                    logger.error(
                        "embedding_worker_cycle_error",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

                # Wait before next cycle
                await asyncio.sleep(self.poll_interval_seconds)

        finally:
            logger.info("embedding_worker_stopped")

    def stop(self) -> None:
        """Signal the worker to stop processing.

        The worker will complete the current cycle before stopping.
        """
        logger.info("embedding_worker_stop_requested")
        self.is_running = False

    async def process_queue(self) -> None:
        """Process one batch of pending queue items.

        Fetches pending items, generates embeddings, and updates target tables.
        Handles errors gracefully by marking failed items.
        """
        # Fetch pending items
        items = await self.queue_repository.get_pending_items(self.batch_size)

        if not items:
            logger.debug("embedding_queue_empty")
            return

        logger.info("embedding_queue_batch_fetched", count=len(items))

        # Process each item
        for item in items:
            await self.process_item(item)

        logger.info("embedding_queue_batch_completed", count=len(items))

    async def process_item(self, item: QueueItem) -> None:
        """Process a single queue item.

        Generates embedding and updates target table on success,
        or marks item as failed on error.

        Args:
            item: Queue item to process
        """
        try:
            logger.debug(
                "embedding_queue_item_processing",
                item_id=item.id,
                target_table=item.target_table,
                target_id=item.target_id,
                text_length=len(item.text),
            )

            # Mark as processing
            await self.queue_repository.mark_processing(item.id)

            # Generate embedding
            embedding = await self.embedding_service.generate(item.text)

            # Update target table
            await self.queue_repository.update_embedding(
                target_table=item.target_table,
                target_id=item.target_id,
                embedding=embedding,
            )

            # Mark as completed
            await self.queue_repository.mark_completed(item.id)

            logger.info(
                "embedding_queue_item_completed",
                item_id=item.id,
                target_table=item.target_table,
                target_id=item.target_id,
                embedding_dim=len(embedding),
            )

        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"

            logger.error(
                "embedding_queue_item_failed",
                item_id=item.id,
                target_table=item.target_table,
                target_id=item.target_id,
                error=error_message,
            )

            # Mark as failed
            await self.queue_repository.mark_failed(item.id, error_message)
