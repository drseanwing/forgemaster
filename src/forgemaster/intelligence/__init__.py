"""Intelligence and knowledge subsystem for Forgemaster.

This module manages embedding generation via Ollama, semantic search over
lessons learned, and knowledge retrieval for agent context injection.
"""

from forgemaster.intelligence.embeddings import (
    EmbeddingClient,
    EmbeddingService,
    OpenAIEmbeddingClient,
)
from forgemaster.intelligence.embedding_worker import EmbeddingWorker, QueueItem, QueueRepository
from forgemaster.intelligence.ollama_client import (
    OllamaAPIError,
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaTimeoutError,
)

__all__ = [
    # Ollama client
    "OllamaClient",
    "OllamaClientError",
    "OllamaTimeoutError",
    "OllamaConnectionError",
    "OllamaAPIError",
    # Embedding service
    "EmbeddingClient",
    "EmbeddingService",
    "OpenAIEmbeddingClient",
    # Queue worker
    "EmbeddingWorker",
    "QueueItem",
    "QueueRepository",
]
