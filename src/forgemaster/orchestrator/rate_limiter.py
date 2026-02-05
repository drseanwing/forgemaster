"""API rate limit handling with token bucket and adaptive throttling.

This module implements rate limiting and throttling for API calls to the Claude API,
including token bucket rate limiting, HTTP 429 response handling, exponential backoff,
and adaptive parallelism reduction.

Key Components:
- TokenBucket: Token bucket algorithm for request rate limiting
- RateLimitHandler: HTTP 429 response parsing and handling
- ExponentialBackoff: Exponential backoff with jitter
- AdaptiveThrottler: Adaptive parallelism reduction based on rate limit responses
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class RateLimitConfig(BaseModel):
    """Rate limit configuration.

    Attributes:
        requests_per_minute: Maximum requests per minute
        tokens_per_minute: Maximum tokens per minute
        burst_multiplier: Burst capacity multiplier (e.g., 1.5 = 50% burst)
        min_requests_per_minute: Minimum requests per minute when throttled
    """

    requests_per_minute: int = Field(default=60, ge=1, le=10000)
    tokens_per_minute: int = Field(default=100000, ge=100, le=10000000)
    burst_multiplier: float = Field(default=1.5, ge=1.0, le=5.0)
    min_requests_per_minute: int = Field(default=5, ge=1, le=100)


class BackoffConfig(BaseModel):
    """Exponential backoff configuration.

    Attributes:
        initial_delay_seconds: Initial backoff delay
        max_delay_seconds: Maximum backoff delay
        multiplier: Backoff multiplier per attempt
        jitter: Add random jitter to delays
    """

    initial_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    multiplier: float = Field(default=2.0, ge=1.1, le=10.0)
    jitter: bool = Field(default=True)


class RateLimitState(str, Enum):
    """Rate limit state.

    Values:
        NORMAL: Normal operation, no rate limiting
        THROTTLED: Throttled due to approaching limits
        BACKING_OFF: Backing off after rate limit hit
        BLOCKED: Blocked by rate limit, waiting for retry
    """

    NORMAL = "normal"
    THROTTLED = "throttled"
    BACKING_OFF = "backing_off"
    BLOCKED = "blocked"


class RateLimitResponse(BaseModel):
    """Parsed rate limit response.

    Attributes:
        status_code: HTTP status code
        retry_after_seconds: Seconds to wait before retry (from Retry-After header)
        rate_limit_remaining: Remaining requests in current window
        rate_limit_reset_at: Timestamp when rate limit resets
        error_message: Error message from response body
    """

    status_code: int
    retry_after_seconds: float | None = None
    rate_limit_remaining: int | None = None
    rate_limit_reset_at: str | None = None
    error_message: str | None = None


class ParallelismReduction(BaseModel):
    """Parallelism reduction status.

    Attributes:
        current_max: Current maximum concurrency
        recommended_max: Recommended maximum concurrency
        reduction_reason: Reason for reduction
        rate_limit_state: Current rate limit state
        consecutive_429s: Number of consecutive 429 responses
        last_429_at: Timestamp of last 429 response
    """

    current_max: int
    recommended_max: int
    reduction_reason: str | None = None
    rate_limit_state: RateLimitState
    consecutive_429s: int
    last_429_at: str | None = None


class TokenBucket:
    """Token bucket rate limiter.

    Implements the token bucket algorithm for rate limiting with burst capacity.
    Thread-safe using asyncio.Lock.

    Args:
        rate: Token refill rate (tokens per second)
        capacity: Maximum token capacity (burst capacity)
    """

    def __init__(self, rate: float, capacity: float) -> None:
        """Initialize token bucket.

        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum token capacity
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

        logger.info(
            "token_bucket_initialized",
            rate=rate,
            capacity=capacity,
        )

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_refill).total_seconds()
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                logger.debug(
                    "tokens_acquired",
                    tokens=tokens,
                    remaining=self._tokens,
                    wait_time=0.0,
                )
                return 0.0

            # Calculate wait time for tokens to refill
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self._rate

            logger.info(
                "waiting_for_tokens",
                tokens_needed=tokens_needed,
                wait_time=wait_time,
            )

            # Wait outside the lock to allow other operations
            async with self._lock:
                self._refill()

            await asyncio.sleep(wait_time)

            async with self._lock:
                self._refill()
                self._tokens -= tokens

            logger.debug(
                "tokens_acquired_after_wait",
                tokens=tokens,
                remaining=self._tokens,
                wait_time=wait_time,
            )

            return wait_time

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            logger.debug(
                "tokens_try_acquire_success",
                tokens=tokens,
                remaining=self._tokens,
            )
            return True

        logger.debug(
            "tokens_try_acquire_failed",
            tokens=tokens,
            available=self._tokens,
        )
        return False

    def available_tokens(self) -> float:
        """Get current available token count.

        Returns:
            Available tokens
        """
        self._refill()
        return self._tokens


class RateLimitHandler:
    """HTTP 429 response handler.

    Parses rate limit information from HTTP responses and calculates
    appropriate retry delays.
    """

    def __init__(self) -> None:
        """Initialize rate limit handler."""
        logger.info("rate_limit_handler_initialized")

    def parse_rate_limit_response(
        self,
        status_code: int,
        headers: dict[str, str],
        body: str | None = None,
    ) -> RateLimitResponse:
        """Parse rate limit response from HTTP headers and body.

        Parses standard headers:
        - Retry-After: Seconds to wait (integer) or HTTP-date
        - X-RateLimit-Remaining: Remaining requests in window
        - X-RateLimit-Reset: Timestamp when limit resets
        - anthropic-ratelimit-*: Anthropic-specific headers

        Args:
            status_code: HTTP status code
            headers: Response headers (case-insensitive keys)
            body: Response body (optional)

        Returns:
            Parsed rate limit response
        """
        # Normalize header keys to lowercase for case-insensitive lookup
        headers_lower = {k.lower(): v for k, v in headers.items()}

        retry_after: float | None = None
        if "retry-after" in headers_lower:
            retry_after_value = headers_lower["retry-after"]
            try:
                # Try parsing as integer (seconds)
                retry_after = float(retry_after_value)
            except ValueError:
                # Try parsing as HTTP-date
                try:
                    retry_date = parsedate_to_datetime(retry_after_value)
                    retry_after = max(
                        0.0,
                        (retry_date - datetime.now(timezone.utc)).total_seconds(),
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "failed_to_parse_retry_after",
                        value=retry_after_value,
                    )

        # Parse X-RateLimit headers
        rate_limit_remaining: int | None = None
        if "x-ratelimit-remaining" in headers_lower:
            try:
                rate_limit_remaining = int(headers_lower["x-ratelimit-remaining"])
            except ValueError:
                logger.warning(
                    "failed_to_parse_x_ratelimit_remaining",
                    value=headers_lower["x-ratelimit-remaining"],
                )

        # Parse X-RateLimit-Reset (can be timestamp or seconds)
        rate_limit_reset_at: str | None = None
        if "x-ratelimit-reset" in headers_lower:
            rate_limit_reset_at = headers_lower["x-ratelimit-reset"]

        # Parse Anthropic-specific headers
        if "anthropic-ratelimit-requests-remaining" in headers_lower:
            try:
                rate_limit_remaining = int(
                    headers_lower["anthropic-ratelimit-requests-remaining"]
                )
            except ValueError:
                pass

        if "anthropic-ratelimit-requests-reset" in headers_lower:
            rate_limit_reset_at = headers_lower["anthropic-ratelimit-requests-reset"]

        # Parse error message from body if available
        error_message: str | None = None
        if body:
            # Simple extraction - in real implementation might parse JSON
            error_message = body[:200]  # First 200 chars

        response = RateLimitResponse(
            status_code=status_code,
            retry_after_seconds=retry_after,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset_at=rate_limit_reset_at,
            error_message=error_message,
        )

        logger.info(
            "rate_limit_response_parsed",
            status_code=status_code,
            retry_after=retry_after,
            remaining=rate_limit_remaining,
            reset_at=rate_limit_reset_at,
        )

        return response

    async def handle_rate_limit(self, response: RateLimitResponse) -> float:
        """Handle rate limit response and wait appropriately.

        Determines wait time from response and applies delay. Prefers
        retry_after_seconds if available, otherwise uses default backoff.

        Args:
            response: Parsed rate limit response

        Returns:
            Wait time applied in seconds
        """
        if response.retry_after_seconds is not None:
            wait_time = response.retry_after_seconds
        else:
            # Default backoff if no retry-after header
            wait_time = 60.0

        logger.info(
            "handling_rate_limit",
            status_code=response.status_code,
            wait_time=wait_time,
            remaining=response.rate_limit_remaining,
        )

        await asyncio.sleep(wait_time)

        logger.info(
            "rate_limit_wait_complete",
            wait_time=wait_time,
        )

        return wait_time


class ExponentialBackoff:
    """Exponential backoff with jitter.

    Calculates exponential backoff delays with optional jitter for retry logic.

    Args:
        config: Backoff configuration
    """

    def __init__(self, config: BackoffConfig) -> None:
        """Initialize exponential backoff.

        Args:
            config: Backoff configuration
        """
        self._config = config
        self._attempt = 0

        logger.info(
            "exponential_backoff_initialized",
            initial_delay=config.initial_delay_seconds,
            max_delay=config.max_delay_seconds,
            multiplier=config.multiplier,
            jitter=config.jitter,
        )

    def next_delay(self, attempt: int) -> float:
        """Calculate next backoff delay.

        Formula: min(initial_delay * multiplier^attempt, max_delay)
        With jitter: delay * (0.5 + random() * 0.5)

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self._config.initial_delay_seconds * (self._config.multiplier**attempt)
        delay = min(delay, self._config.max_delay_seconds)

        if self._config.jitter:
            # Add jitter: multiply by random factor between 0.5 and 1.0
            jitter_factor = 0.5 + random.random() * 0.5
            delay *= jitter_factor

        logger.debug(
            "backoff_delay_calculated",
            attempt=attempt,
            delay=delay,
            jitter=self._config.jitter,
        )

        return delay

    async def wait(self, attempt: int) -> float:
        """Wait for calculated backoff delay.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Actual wait time in seconds
        """
        delay = self.next_delay(attempt)

        logger.info(
            "backoff_waiting",
            attempt=attempt,
            delay=delay,
        )

        await asyncio.sleep(delay)

        logger.info(
            "backoff_wait_complete",
            attempt=attempt,
            delay=delay,
        )

        return delay

    def reset(self) -> None:
        """Reset attempt counter."""
        self._attempt = 0
        logger.debug("backoff_reset")


class AdaptiveThrottler:
    """Adaptive rate limiter with parallelism reduction.

    Main rate limit manager that coordinates token bucket rate limiting,
    HTTP 429 response handling, and adaptive parallelism reduction.

    Args:
        config: Rate limit configuration
        backoff_config: Backoff configuration
    """

    def __init__(
        self,
        config: RateLimitConfig,
        backoff_config: BackoffConfig,
    ) -> None:
        """Initialize adaptive throttler.

        Args:
            config: Rate limit configuration
            backoff_config: Backoff configuration
        """
        self._config = config
        self._backoff_config = backoff_config

        # Token buckets for requests and tokens
        self._request_bucket = TokenBucket(
            rate=config.requests_per_minute / 60.0,
            capacity=config.requests_per_minute * config.burst_multiplier / 60.0,
        )

        self._handler = RateLimitHandler()
        self._backoff = ExponentialBackoff(backoff_config)

        # Adaptive throttling state
        self._state = RateLimitState.NORMAL
        self._consecutive_429s = 0
        self._last_429_at: datetime | None = None
        self._last_success_at = datetime.now(timezone.utc)
        self._current_reduction = 0
        self._lock = asyncio.Lock()

        logger.info(
            "adaptive_throttler_initialized",
            requests_per_minute=config.requests_per_minute,
            tokens_per_minute=config.tokens_per_minute,
        )

    async def on_request(self) -> float:
        """Called before making a request. Acquires from token bucket.

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        wait_time = await self._request_bucket.acquire(1.0)

        logger.debug(
            "request_acquired",
            wait_time=wait_time,
            state=self._state.value,
        )

        return wait_time

    async def on_response(self, status_code: int, headers: dict[str, str]) -> None:
        """Called after receiving a response. Updates throttling state.

        Args:
            status_code: HTTP status code
            headers: Response headers
        """
        async with self._lock:
            if status_code == 429:
                # Rate limit hit
                self._consecutive_429s += 1
                self._last_429_at = datetime.now(timezone.utc)
                self._state = RateLimitState.BACKING_OFF

                # Parse response and handle
                response = self._handler.parse_rate_limit_response(
                    status_code=status_code,
                    headers=headers,
                )
                wait_time = await self._handler.handle_rate_limit(response)

                # Increase reduction
                self._current_reduction = min(
                    self._current_reduction + 1,
                    10,  # Max reduction
                )

                logger.warning(
                    "rate_limit_hit",
                    consecutive_429s=self._consecutive_429s,
                    wait_time=wait_time,
                    reduction=self._current_reduction,
                )

            elif 200 <= status_code < 300:
                # Success - gradually restore parallelism
                self._last_success_at = datetime.now(timezone.utc)

                # If no 429s for recovery period (60s), reduce reduction
                if self._last_429_at:
                    time_since_429 = (
                        datetime.now(timezone.utc) - self._last_429_at
                    ).total_seconds()
                    if time_since_429 > 60.0:
                        # Gradually restore (one level per recovery period)
                        if self._current_reduction > 0:
                            self._current_reduction -= 1
                            logger.info(
                                "parallelism_restored",
                                reduction=self._current_reduction,
                                time_since_429=time_since_429,
                            )

                # Update state
                if self._consecutive_429s > 0 and self._current_reduction == 0:
                    self._consecutive_429s = 0
                    self._state = RateLimitState.NORMAL
                    logger.info("rate_limit_state_normal")

    def get_recommended_concurrency(self) -> int:
        """Get recommended maximum concurrency level.

        Reduces from max_concurrent_workers based on throttling state.
        Formula: max(min_workers, current - reduction_per_429)

        Returns:
            Recommended maximum concurrency
        """
        from forgemaster.config import AgentConfig

        agent_config = AgentConfig()
        base_max = agent_config.max_concurrent_workers

        # Reduce by 1 per consecutive 429
        recommended = max(
            self._config.min_requests_per_minute,
            base_max - self._current_reduction,
        )

        return recommended

    def get_parallelism_status(self) -> ParallelismReduction:
        """Get current parallelism reduction status.

        Returns:
            Parallelism reduction status
        """
        from forgemaster.config import AgentConfig

        agent_config = AgentConfig()
        current_max = agent_config.max_concurrent_workers
        recommended_max = self.get_recommended_concurrency()

        reduction_reason = None
        if self._current_reduction > 0:
            reduction_reason = f"Reduced by {self._current_reduction} due to rate limiting"

        return ParallelismReduction(
            current_max=current_max,
            recommended_max=recommended_max,
            reduction_reason=reduction_reason,
            rate_limit_state=self._state,
            consecutive_429s=self._consecutive_429s,
            last_429_at=self._last_429_at.isoformat() if self._last_429_at else None,
        )
