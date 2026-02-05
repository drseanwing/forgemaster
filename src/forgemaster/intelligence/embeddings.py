"""Embedding generation service with provider fallback.

This module orchestrates embedding generation using multiple providers:
- Primary: Ollama (local, nomic-embed-text)
- Fallback: OpenAI (cloud, text-embedding-3-small)

All embeddings are normalized to unit vectors for consistent cosine similarity.

Example usage:
    >>> from forgemaster.config import OllamaConfig
    >>> from forgemaster.intelligence.ollama_client import OllamaClient
    >>> from forgemaster.intelligence.embeddings import EmbeddingService
    >>>
    >>> ollama_config = OllamaConfig()
    >>> async with OllamaClient(ollama_config) as ollama:
    ...     service = EmbeddingService(ollama_client=ollama)
    ...     embedding = await service.generate("Hello world")
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import httpx
import numpy as np
import structlog

from forgemaster.intelligence.ollama_client import OllamaClient, OllamaClientError

logger = structlog.get_logger(__name__)


class EmbeddingClient(Protocol):
    """Protocol for embedding client implementations."""

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...

    async def health_check(self) -> bool:
        """Check if client is healthy."""
        ...


class OpenAIEmbeddingClient:
    """OpenAI embedding API client for fallback.

    Attributes:
        api_key: OpenAI API key from environment
        model: Embedding model name (default: text-embedding-3-small)
        timeout_seconds: Request timeout in seconds
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        timeout_seconds: int = 30,
    ) -> None:
        """Initialize OpenAI embedding client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model name
            timeout_seconds: Request timeout in seconds

        Raises:
            ValueError: If api_key is not provided and OPENAI_API_KEY env var not set
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required: pass api_key or set OPENAI_API_KEY env var"
            )

        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "openai_embedding_client_initialized",
            model=model,
            timeout=timeout_seconds,
        )

    async def __aenter__(self) -> OpenAIEmbeddingClient:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            timeout=httpx.Timeout(self.timeout_seconds),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            raise RuntimeError(
                "OpenAIEmbeddingClient must be used as async context manager"
            )
        return self._client

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If API request fails
        """
        client = self._get_client()
        endpoint = "/embeddings"
        payload = {"input": text, "model": self.model}

        try:
            logger.debug(
                "openai_embedding_request", text_length=len(text), model=self.model
            )

            response = await client.post(endpoint, json=payload)
            response.raise_for_status()

            data = response.json()
            embedding = data["data"][0]["embedding"]

            logger.info(
                "openai_embedding_generated",
                text_length=len(text),
                embedding_dim=len(embedding),
                model=self.model,
            )

            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(
                "openai_api_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("openai_embedding_failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        client = self._get_client()

        try:
            response = await client.get("/models")
            if response.status_code == 200:
                logger.info("openai_health_check_passed")
                return True
            else:
                logger.warning(
                    "openai_health_check_failed", status_code=response.status_code
                )
                return False

        except Exception as e:
            logger.warning("openai_health_check_error", error=str(e))
            return False


class EmbeddingService:
    """High-level embedding generation service with provider fallback.

    This service orchestrates embedding generation across multiple providers,
    implementing automatic fallback and normalization.

    Attributes:
        ollama_client: Primary Ollama embedding client
        openai_client: Optional OpenAI fallback client
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        openai_client: OpenAIEmbeddingClient | None = None,
    ) -> None:
        """Initialize embedding service.

        Args:
            ollama_client: Primary Ollama client
            openai_client: Optional OpenAI fallback client
        """
        self.ollama_client = ollama_client
        self.openai_client = openai_client

        logger.info(
            "embedding_service_initialized",
            has_fallback=openai_client is not None,
        )

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        """Normalize embedding to unit vector.

        Args:
            embedding: Raw embedding vector

        Returns:
            Normalized embedding vector (L2 norm = 1.0)
        """
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)

        if norm == 0:
            logger.warning("embedding_zero_norm", embedding_dim=len(embedding))
            return embedding

        normalized = arr / norm
        return normalized.tolist()

    async def generate(self, text: str) -> list[float]:
        """Generate normalized embedding for text.

        Attempts generation using Ollama first, falling back to OpenAI if configured
        and Ollama fails.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector

        Raises:
            ValueError: If text is empty or only whitespace
            Exception: If all providers fail
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace text")

        start_time = time.time()

        # Try Ollama first
        try:
            logger.debug("embedding_generation_started", provider="ollama", text_length=len(text))

            raw_embedding = await self.ollama_client.generate_embedding(text)
            normalized = self._normalize_embedding(raw_embedding)

            duration = time.time() - start_time
            logger.info(
                "embedding_generated",
                provider="ollama",
                text_length=len(text),
                embedding_dim=len(normalized),
                duration_seconds=round(duration, 3),
            )

            return normalized

        except OllamaClientError as e:
            logger.warning(
                "ollama_embedding_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

            # Try OpenAI fallback if available
            if self.openai_client is not None:
                try:
                    logger.info("embedding_fallback_to_openai", text_length=len(text))

                    raw_embedding = await self.openai_client.generate_embedding(text)
                    normalized = self._normalize_embedding(raw_embedding)

                    duration = time.time() - start_time
                    logger.info(
                        "embedding_generated",
                        provider="openai",
                        text_length=len(text),
                        embedding_dim=len(normalized),
                        duration_seconds=round(duration, 3),
                    )

                    return normalized

                except Exception as openai_error:
                    logger.error(
                        "all_embedding_providers_failed",
                        ollama_error=str(e),
                        openai_error=str(openai_error),
                    )
                    raise Exception(
                        f"All embedding providers failed. Ollama: {e}, OpenAI: {openai_error}"
                    ) from openai_error
            else:
                # No fallback available
                logger.error(
                    "embedding_failed_no_fallback",
                    error=str(e),
                )
                raise
