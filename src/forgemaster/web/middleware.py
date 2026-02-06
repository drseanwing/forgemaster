"""Request logging middleware for Forgemaster.

This module provides middleware for logging HTTP requests with:
- Request method, path, and status code
- Request duration in milliseconds
- Correlation IDs for distributed tracing
- Structured logging via structlog

Example:
    >>> from fastapi import FastAPI
    >>> from forgemaster.web.middleware import RequestLoggingMiddleware
    >>>
    >>> app = FastAPI()
    >>> app.add_middleware(RequestLoggingMiddleware)
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from forgemaster.logging import get_logger, set_correlation_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs HTTP requests with timing and correlation IDs.

    For each incoming request, this middleware:
    1. Generates or extracts a correlation ID
    2. Sets the correlation ID in the logging context
    3. Logs the request start
    4. Measures request duration
    5. Logs the response with status and duration

    The correlation ID is extracted from the X-Correlation-ID header if present,
    otherwise a new UUID is generated.

    Attributes:
        None

    Example:
        >>> from fastapi import FastAPI
        >>> from forgemaster.web.middleware import RequestLoggingMiddleware
        >>>
        >>> app = FastAPI()
        >>> app.add_middleware(RequestLoggingMiddleware)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request with logging.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response from downstream handlers
        """
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        # Set correlation ID in logging context
        set_correlation_id(correlation_id)

        # Record start time
        start_time = time.perf_counter()

        # Log request start
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query) if request.url.query else None,
        )

        try:
            # Call downstream handler
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as exc:
            # Calculate duration even on error
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log error
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error=str(exc),
                exc_info=True,
            )
            raise

        finally:
            # Clear correlation ID from context
            set_correlation_id(None)
