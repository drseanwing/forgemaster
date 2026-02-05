"""Ollama API client for embedding generation.

This module provides an async HTTP client for interacting with the Ollama
embedding API. It handles timeouts, retries with exponential backoff, and
comprehensive error logging.

Example usage:
    >>> from forgemaster.config import OllamaConfig
    >>> config = OllamaConfig(url="http://localhost:11434", model="nomic-embed-text")
    >>> client = OllamaClient(config)
    >>> embedding = await client.generate_embedding("Hello world")
    >>> is_healthy = await client.health_check()
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from forgemaster.config import OllamaConfig

logger = structlog.get_logger(__name__)


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""

    pass


class OllamaTimeoutError(OllamaClientError):
    """Raised when Ollama request times out."""

    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when unable to connect to Ollama service."""

    pass


class OllamaAPIError(OllamaClientError):
    """Raised when Ollama API returns an error response."""

    pass


class OllamaClient:
    """Async client for Ollama embedding API.

    This client provides embedding generation with automatic retries,
    timeout handling, and health checking capabilities.

    Attributes:
        config: Ollama configuration containing URL, model, and timeout settings
    """

    def __init__(self, config: OllamaConfig) -> None:
        """Initialize Ollama client.

        Args:
            config: OllamaConfig instance with connection settings
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None
        logger.info(
            "ollama_client_initialized",
            url=config.url,
            model=config.model,
            timeout=config.timeout_seconds,
        )

    async def __aenter__(self) -> OllamaClient:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.config.url,
            timeout=httpx.Timeout(self.config.timeout_seconds),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Active httpx.AsyncClient instance

        Raises:
            RuntimeError: If called outside async context manager
        """
        if self._client is None:
            raise RuntimeError("OllamaClient must be used as async context manager")
        return self._client

    async def generate_embedding(
        self, text: str, max_retries: int = 3, initial_backoff: float = 1.0
    ) -> list[float]:
        """Generate embedding vector for given text.

        Implements exponential backoff retry logic on transient failures.
        Retries on connection errors and 5xx status codes.

        Args:
            text: Input text to embed
            max_retries: Maximum number of retry attempts (default: 3)
            initial_backoff: Initial backoff delay in seconds (default: 1.0)

        Returns:
            Embedding vector as list of floats

        Raises:
            OllamaTimeoutError: If request times out after all retries
            OllamaConnectionError: If unable to connect after all retries
            OllamaAPIError: If API returns error response
        """
        client = self._get_client()
        endpoint = "/api/embeddings"
        payload = {"model": self.config.model, "prompt": text}

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "ollama_embedding_request",
                    attempt=attempt + 1,
                    max_retries=max_retries + 1,
                    text_length=len(text),
                )

                response = await client.post(endpoint, json=payload)

                # Check for successful response
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding")

                    if not embedding or not isinstance(embedding, list):
                        raise OllamaAPIError(
                            f"Invalid response format: missing or invalid 'embedding' field"
                        )

                    logger.info(
                        "ollama_embedding_generated",
                        text_length=len(text),
                        embedding_dim=len(embedding),
                        attempt=attempt + 1,
                    )
                    return embedding

                # Handle error responses
                error_msg = f"API error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = f"{error_msg}: {error_data}"
                except Exception:
                    error_msg = f"{error_msg}: {response.text}"

                # Retry on 5xx errors
                if 500 <= response.status_code < 600 and attempt < max_retries:
                    backoff = initial_backoff * (2**attempt)
                    logger.warning(
                        "ollama_server_error_retry",
                        status_code=response.status_code,
                        attempt=attempt + 1,
                        backoff_seconds=backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                raise OllamaAPIError(error_msg)

            except httpx.TimeoutException as e:
                if attempt < max_retries:
                    backoff = initial_backoff * (2**attempt)
                    logger.warning(
                        "ollama_timeout_retry",
                        attempt=attempt + 1,
                        backoff_seconds=backoff,
                        error=str(e),
                    )
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(
                        "ollama_timeout_exhausted",
                        max_retries=max_retries,
                        timeout_seconds=self.config.timeout_seconds,
                    )
                    raise OllamaTimeoutError(
                        f"Request timed out after {max_retries} retries"
                    ) from e

            except (httpx.ConnectError, httpx.NetworkError) as e:
                if attempt < max_retries:
                    backoff = initial_backoff * (2**attempt)
                    logger.warning(
                        "ollama_connection_error_retry",
                        attempt=attempt + 1,
                        backoff_seconds=backoff,
                        error=str(e),
                    )
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(
                        "ollama_connection_exhausted",
                        url=self.config.url,
                        max_retries=max_retries,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama at {self.config.url}"
                    ) from e

        # Should never reach here, but satisfy type checker
        raise OllamaClientError("Unexpected retry loop exit")

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy and responsive.

        Returns:
            True if service is healthy, False otherwise
        """
        client = self._get_client()

        try:
            # Ollama doesn't have a dedicated health endpoint,
            # so we check if the server responds to the tags endpoint
            response = await client.get("/api/tags")

            if response.status_code == 200:
                logger.info("ollama_health_check_passed", url=self.config.url)
                return True
            else:
                logger.warning(
                    "ollama_health_check_failed",
                    url=self.config.url,
                    status_code=response.status_code,
                )
                return False

        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            logger.warning(
                "ollama_health_check_error", url=self.config.url, error=str(e)
            )
            return False
        except Exception as e:
            logger.error(
                "ollama_health_check_unexpected_error",
                url=self.config.url,
                error=str(e),
            )
            return False
