"""Unit tests for embedding generation system.

Tests cover:
- Ollama client with mocked httpx
- Embedding normalization
- Empty text handling
- Retry logic on failure
- OpenAI fallback when Ollama fails
- Embedding worker queue processing
- Error handling and logging
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import httpx
import numpy as np
import pytest

from forgemaster.config import OllamaConfig
from forgemaster.intelligence.embedding_worker import EmbeddingWorker, QueueItem
from forgemaster.intelligence.embeddings import (
    EmbeddingService,
    OpenAIEmbeddingClient,
)
from forgemaster.intelligence.ollama_client import (
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaTimeoutError,
)


class TestOllamaClient:
    """Test OllamaClient with mocked httpx."""

    @pytest.fixture
    def config(self) -> OllamaConfig:
        """Create test Ollama configuration."""
        return OllamaConfig(
            url="http://localhost:11434",
            model="nomic-embed-text",
            timeout_seconds=30,
        )

    @pytest.fixture
    async def mock_client(self, config: OllamaConfig):
        """Create OllamaClient with mocked httpx."""
        async with OllamaClient(config) as client:
            yield client

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, config: OllamaConfig) -> None:
        """Test successful embedding generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4],
        }

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            async with OllamaClient(config) as client:
                embedding = await client.generate_embedding("test text")

                assert embedding == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_generate_embedding_retry_on_500(self, config: OllamaConfig) -> None:
        """Test retry logic on 5xx server errors."""
        # First call fails with 500, second succeeds
        fail_response = Mock()
        fail_response.status_code = 500
        fail_response.json.return_value = {"error": "Internal server error"}

        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"embedding": [0.1, 0.2]}

        with patch(
            "httpx.AsyncClient.post", side_effect=[fail_response, success_response]
        ):
            async with OllamaClient(config) as client:
                embedding = await client.generate_embedding("test", max_retries=2)
                assert embedding == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_generate_embedding_timeout(self, config: OllamaConfig) -> None:
        """Test timeout handling with retries."""
        with patch(
            "httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")
        ):
            async with OllamaClient(config) as client:
                with pytest.raises(OllamaTimeoutError):
                    await client.generate_embedding("test", max_retries=1)

    @pytest.mark.asyncio
    async def test_generate_embedding_connection_error(
        self, config: OllamaConfig
    ) -> None:
        """Test connection error handling."""
        with patch(
            "httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection failed")
        ):
            async with OllamaClient(config) as client:
                with pytest.raises(OllamaConnectionError):
                    await client.generate_embedding("test", max_retries=1)

    @pytest.mark.asyncio
    async def test_generate_embedding_api_error(self, config: OllamaConfig) -> None:
        """Test API error response handling."""
        error_response = Mock()
        error_response.status_code = 400
        error_response.json.return_value = {"error": "Invalid request"}

        with patch("httpx.AsyncClient.post", return_value=error_response):
            async with OllamaClient(config) as client:
                with pytest.raises(OllamaAPIError):
                    await client.generate_embedding("test")

    @pytest.mark.asyncio
    async def test_health_check_success(self, config: OllamaConfig) -> None:
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            async with OllamaClient(config) as client:
                is_healthy = await client.health_check()
                assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, config: OllamaConfig) -> None:
        """Test failed health check."""
        with patch(
            "httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection failed")
        ):
            async with OllamaClient(config) as client:
                is_healthy = await client.health_check()
                assert is_healthy is False

    @pytest.mark.asyncio
    async def test_context_manager_required(self, config: OllamaConfig) -> None:
        """Test that client must be used as context manager."""
        client = OllamaClient(config)
        with pytest.raises(RuntimeError, match="async context manager"):
            await client.generate_embedding("test")


class TestOpenAIEmbeddingClient:
    """Test OpenAIEmbeddingClient."""

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self) -> None:
        """Test successful OpenAI embedding generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            async with OpenAIEmbeddingClient(api_key="test-key") as client:
                embedding = await client.generate_embedding("test text")
                assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_api_key_required(self) -> None:
        """Test that API key is required."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                OpenAIEmbeddingClient()

    @pytest.mark.asyncio
    async def test_api_key_from_env(self) -> None:
        """Test API key loaded from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            client = OpenAIEmbeddingClient()
            assert client.api_key == "env-key"

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            async with OpenAIEmbeddingClient(api_key="test-key") as client:
                is_healthy = await client.health_check()
                assert is_healthy is True


class TestEmbeddingService:
    """Test EmbeddingService."""

    @pytest.fixture
    def mock_ollama_client(self) -> AsyncMock:
        """Create mock Ollama client."""
        client = AsyncMock()
        client.generate_embedding = AsyncMock(return_value=[0.6, 0.8])
        return client

    @pytest.fixture
    def mock_openai_client(self) -> AsyncMock:
        """Create mock OpenAI client."""
        client = AsyncMock()
        client.generate_embedding = AsyncMock(return_value=[0.6, 0.8])
        return client

    def test_normalize_embedding(self) -> None:
        """Test embedding normalization to unit vector."""
        # Vector [3, 4] has norm 5, normalized to [0.6, 0.8]
        embedding = [3.0, 4.0]
        normalized = EmbeddingService._normalize_embedding(embedding)

        assert len(normalized) == 2
        assert abs(normalized[0] - 0.6) < 1e-6
        assert abs(normalized[1] - 0.8) < 1e-6

        # Check L2 norm is 1.0
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6

    def test_normalize_zero_vector(self) -> None:
        """Test normalization of zero vector."""
        embedding = [0.0, 0.0, 0.0]
        normalized = EmbeddingService._normalize_embedding(embedding)

        # Zero vector returns unchanged (with warning)
        assert normalized == embedding

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_ollama_client: AsyncMock) -> None:
        """Test successful embedding generation via Ollama."""
        service = EmbeddingService(ollama_client=mock_ollama_client)

        embedding = await service.generate("test text")

        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

        mock_ollama_client.generate_embedding.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_generate_empty_text(self, mock_ollama_client: AsyncMock) -> None:
        """Test that empty text raises ValueError."""
        service = EmbeddingService(ollama_client=mock_ollama_client)

        with pytest.raises(ValueError, match="empty or whitespace"):
            await service.generate("")

        with pytest.raises(ValueError, match="empty or whitespace"):
            await service.generate("   ")

    @pytest.mark.asyncio
    async def test_fallback_to_openai(
        self, mock_ollama_client: AsyncMock, mock_openai_client: AsyncMock
    ) -> None:
        """Test fallback to OpenAI when Ollama fails."""
        # Ollama fails
        mock_ollama_client.generate_embedding.side_effect = OllamaConnectionError(
            "Connection failed"
        )

        # OpenAI succeeds
        mock_openai_client.generate_embedding.return_value = [0.6, 0.8]

        service = EmbeddingService(
            ollama_client=mock_ollama_client, openai_client=mock_openai_client
        )

        embedding = await service.generate("test text")

        # Should use OpenAI fallback
        mock_openai_client.generate_embedding.assert_called_once_with("test text")

        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_no_fallback_available(self, mock_ollama_client: AsyncMock) -> None:
        """Test that error is raised when no fallback available."""
        mock_ollama_client.generate_embedding.side_effect = OllamaConnectionError(
            "Connection failed"
        )

        service = EmbeddingService(ollama_client=mock_ollama_client)

        with pytest.raises(OllamaConnectionError):
            await service.generate("test text")

    @pytest.mark.asyncio
    async def test_all_providers_fail(
        self, mock_ollama_client: AsyncMock, mock_openai_client: AsyncMock
    ) -> None:
        """Test that error is raised when all providers fail."""
        mock_ollama_client.generate_embedding.side_effect = OllamaConnectionError(
            "Ollama failed"
        )
        mock_openai_client.generate_embedding.side_effect = Exception("OpenAI failed")

        service = EmbeddingService(
            ollama_client=mock_ollama_client, openai_client=mock_openai_client
        )

        with pytest.raises(Exception, match="All embedding providers failed"):
            await service.generate("test text")


class TestEmbeddingWorker:
    """Test EmbeddingWorker queue processing."""

    @pytest.fixture
    def mock_embedding_service(self) -> AsyncMock:
        """Create mock embedding service."""
        service = AsyncMock()
        service.generate = AsyncMock(return_value=[0.1, 0.2, 0.3])
        return service

    @pytest.fixture
    def mock_queue_repo(self) -> AsyncMock:
        """Create mock queue repository."""
        repo = AsyncMock()
        repo.get_pending_items = AsyncMock(return_value=[])
        repo.mark_processing = AsyncMock()
        repo.update_embedding = AsyncMock()
        repo.mark_completed = AsyncMock()
        repo.mark_failed = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_process_item_success(
        self, mock_embedding_service: AsyncMock, mock_queue_repo: AsyncMock
    ) -> None:
        """Test successful processing of a queue item."""
        worker = EmbeddingWorker(
            embedding_service=mock_embedding_service,
            queue_repository=mock_queue_repo,
        )

        item = QueueItem(
            id=1,
            target_table="lessons_learned",
            target_id=100,
            text="test lesson",
            status="pending",
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await worker.process_item(item)

        # Verify workflow
        mock_queue_repo.mark_processing.assert_called_once_with(1)
        mock_embedding_service.generate.assert_called_once_with("test lesson")
        mock_queue_repo.update_embedding.assert_called_once_with(
            target_table="lessons_learned",
            target_id=100,
            embedding=[0.1, 0.2, 0.3],
        )
        mock_queue_repo.mark_completed.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_process_item_failure(
        self, mock_embedding_service: AsyncMock, mock_queue_repo: AsyncMock
    ) -> None:
        """Test handling of item processing failure."""
        worker = EmbeddingWorker(
            embedding_service=mock_embedding_service,
            queue_repository=mock_queue_repo,
        )

        # Simulate embedding generation failure
        mock_embedding_service.generate.side_effect = Exception("Embedding failed")

        item = QueueItem(
            id=1,
            target_table="lessons_learned",
            target_id=100,
            text="test lesson",
            status="pending",
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await worker.process_item(item)

        # Should mark as failed
        mock_queue_repo.mark_failed.assert_called_once()
        call_args = mock_queue_repo.mark_failed.call_args
        assert call_args[0][0] == 1  # item_id
        assert "Embedding failed" in call_args[0][1]  # error_message

    @pytest.mark.asyncio
    async def test_process_queue_empty(
        self, mock_embedding_service: AsyncMock, mock_queue_repo: AsyncMock
    ) -> None:
        """Test processing when queue is empty."""
        worker = EmbeddingWorker(
            embedding_service=mock_embedding_service,
            queue_repository=mock_queue_repo,
            batch_size=10,
        )

        mock_queue_repo.get_pending_items.return_value = []

        await worker.process_queue()

        # Should fetch items but not process any
        mock_queue_repo.get_pending_items.assert_called_once_with(10)
        mock_embedding_service.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_queue_batch(
        self, mock_embedding_service: AsyncMock, mock_queue_repo: AsyncMock
    ) -> None:
        """Test batch processing of queue items."""
        worker = EmbeddingWorker(
            embedding_service=mock_embedding_service,
            queue_repository=mock_queue_repo,
            batch_size=2,
        )

        items = [
            QueueItem(
                id=1,
                target_table="lessons_learned",
                target_id=100,
                text="lesson 1",
                status="pending",
                error_message=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            QueueItem(
                id=2,
                target_table="lessons_learned",
                target_id=101,
                text="lesson 2",
                status="pending",
                error_message=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

        mock_queue_repo.get_pending_items.return_value = items

        await worker.process_queue()

        # Should process both items
        assert mock_embedding_service.generate.call_count == 2
        assert mock_queue_repo.mark_completed.call_count == 2

    def test_worker_initialization(
        self, mock_embedding_service: AsyncMock, mock_queue_repo: AsyncMock
    ) -> None:
        """Test worker initialization with custom settings."""
        worker = EmbeddingWorker(
            embedding_service=mock_embedding_service,
            queue_repository=mock_queue_repo,
            batch_size=20,
            poll_interval_seconds=10.0,
        )

        assert worker.batch_size == 20
        assert worker.poll_interval_seconds == 10.0
        assert worker.is_running is False
