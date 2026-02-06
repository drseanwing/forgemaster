"""Unit tests for FastAPI application setup.

Tests cover:
- Application factory creates FastAPI instance
- CORS middleware is configured correctly
- Request logging middleware is active
- Health endpoints return expected responses
- Readiness endpoint verifies database connectivity
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from forgemaster.config import ForgemasterConfig, WebConfig
from forgemaster.web.app import create_app
from forgemaster.web.middleware import RequestLoggingMiddleware

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class TestCreateApp:
    """Test application factory function."""

    def test_returns_fastapi_instance(self) -> None:
        """Test that create_app returns a FastAPI instance."""
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_correct_title(self) -> None:
        """Test that app has correct title."""
        app = create_app()
        assert app.title == "Forgemaster"

    def test_app_has_version(self) -> None:
        """Test that app has version set."""
        app = create_app()
        assert app.version == "0.1.0"

    def test_app_stores_config_in_state(self) -> None:
        """Test that config is stored in app.state."""
        config = ForgemasterConfig()
        app = create_app(config)
        assert app.state.config is config

    def test_uses_default_config_when_none_provided(self) -> None:
        """Test that default config is used when none provided."""
        app = create_app()
        assert isinstance(app.state.config, ForgemasterConfig)


class TestCorsMiddleware:
    """Test CORS middleware configuration."""

    def test_cors_middleware_is_registered(self) -> None:
        """Test that CORS middleware is added to the app."""
        config = ForgemasterConfig(
            web=WebConfig(cors_origins=["https://example.com"])
        )
        app = create_app(config)

        # Check middleware stack for CORSMiddleware
        cors_found = False
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                cors_found = True
                break

        assert cors_found, "CORSMiddleware not found in middleware stack"

    def test_cors_uses_config_origins(self) -> None:
        """Test that CORS middleware uses origins from config."""
        origins = ["https://app.example.com", "https://admin.example.com"]
        config = ForgemasterConfig(
            web=WebConfig(cors_origins=origins)
        )
        app = create_app(config)

        # Find CORS middleware and check its configuration
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                assert middleware.kwargs["allow_origins"] == origins
                assert middleware.kwargs["allow_credentials"] is True
                assert middleware.kwargs["allow_methods"] == ["*"]
                assert middleware.kwargs["allow_headers"] == ["*"]
                break


class TestRequestLoggingMiddleware:
    """Test request logging middleware configuration."""

    def test_logging_middleware_is_registered(self) -> None:
        """Test that RequestLoggingMiddleware is added to the app."""
        app = create_app()

        # Check middleware stack for RequestLoggingMiddleware
        logging_found = False
        for middleware in app.user_middleware:
            if middleware.cls == RequestLoggingMiddleware:
                logging_found = True
                break

        assert logging_found, "RequestLoggingMiddleware not found in middleware stack"


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.fixture
    def app_with_mock_db(self) -> FastAPI:
        """Create app with mocked database for testing."""
        app = create_app()
        # Mock session factory for tests (doesn't need real DB)
        mock_session_factory = MagicMock()
        app.state.session_factory = mock_session_factory
        return app

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self, app_with_mock_db: FastAPI) -> None:
        """Test that /health/ endpoint returns 200 OK."""
        async with AsyncClient(
            transport=ASGITransport(app=app_with_mock_db),
            base_url="http://test",
        ) as client:
            response = await client.get("/health/")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_ok_status(
        self, app_with_mock_db: FastAPI
    ) -> None:
        """Test that /health/ endpoint returns status ok."""
        async with AsyncClient(
            transport=ASGITransport(app=app_with_mock_db),
            base_url="http://test",
        ) as client:
            response = await client.get("/health/")
            data = response.json()
            assert data["status"] == "ok"


class TestReadinessEndpoint:
    """Test readiness check endpoint with database verification."""

    @pytest.fixture
    def app_with_healthy_db(self) -> FastAPI:
        """Create app with healthy (mocked) database."""
        app = create_app()

        # Create async context manager mock for session
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())

        async def session_context() -> AsyncIterator[AsyncMock]:
            yield mock_session

        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        app.state.session_factory = mock_session_factory
        return app

    @pytest.fixture
    def app_with_unhealthy_db(self) -> FastAPI:
        """Create app with unhealthy (failing) database."""
        app = create_app()

        # Create mock that raises exception
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        app.state.session_factory = mock_session_factory
        return app

    @pytest.mark.asyncio
    async def test_readiness_returns_200_when_db_healthy(
        self, app_with_healthy_db: FastAPI
    ) -> None:
        """Test that /health/ready returns 200 when database is healthy."""
        async with AsyncClient(
            transport=ASGITransport(app=app_with_healthy_db),
            base_url="http://test",
        ) as client:
            response = await client.get("/health/ready")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_readiness_returns_ok_status_when_healthy(
        self, app_with_healthy_db: FastAPI
    ) -> None:
        """Test that readiness returns ok status when database is healthy."""
        async with AsyncClient(
            transport=ASGITransport(app=app_with_healthy_db),
            base_url="http://test",
        ) as client:
            response = await client.get("/health/ready")
            data = response.json()
            assert data["status"] == "ok"
            assert data["database"] == "connected"

    @pytest.mark.asyncio
    async def test_readiness_returns_unhealthy_when_db_fails(
        self, app_with_unhealthy_db: FastAPI
    ) -> None:
        """Test that readiness returns unhealthy when database fails."""
        async with AsyncClient(
            transport=ASGITransport(app=app_with_unhealthy_db),
            base_url="http://test",
        ) as client:
            response = await client.get("/health/ready")
            # Still returns 200 but with unhealthy status
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["database"] == "disconnected"


class TestCorrelationId:
    """Test correlation ID handling in middleware."""

    @pytest.fixture
    def app_with_mock_db(self) -> FastAPI:
        """Create app with mocked database."""
        app = create_app()
        mock_session_factory = MagicMock()
        app.state.session_factory = mock_session_factory
        return app

    @pytest.mark.asyncio
    async def test_response_includes_correlation_id(
        self, app_with_mock_db: FastAPI
    ) -> None:
        """Test that response includes X-Correlation-ID header."""
        async with AsyncClient(
            transport=ASGITransport(app=app_with_mock_db),
            base_url="http://test",
        ) as client:
            response = await client.get("/health/")
            assert "X-Correlation-ID" in response.headers

    @pytest.mark.asyncio
    async def test_response_echoes_provided_correlation_id(
        self, app_with_mock_db: FastAPI
    ) -> None:
        """Test that provided correlation ID is echoed in response."""
        custom_id = "test-correlation-id-12345"
        async with AsyncClient(
            transport=ASGITransport(app=app_with_mock_db),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/health/",
                headers={"X-Correlation-ID": custom_id},
            )
            assert response.headers["X-Correlation-ID"] == custom_id


class TestRouterRegistration:
    """Test that routers are properly registered."""

    def test_health_router_is_registered(self) -> None:
        """Test that health router is included in the app."""
        app = create_app()

        # Check that /health/ route exists
        routes = [route.path for route in app.routes]
        assert "/health/" in routes or "/health" in routes

    def test_readiness_route_exists(self) -> None:
        """Test that /health/ready route exists."""
        app = create_app()

        routes = [route.path for route in app.routes]
        assert "/health/ready" in routes
