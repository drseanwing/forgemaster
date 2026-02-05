"""Unit tests for rate limiter module.

Tests for token bucket, HTTP 429 handling, exponential backoff,
and adaptive throttling components.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from forgemaster.orchestrator.rate_limiter import (
    AdaptiveThrottler,
    BackoffConfig,
    ExponentialBackoff,
    ParallelismReduction,
    RateLimitConfig,
    RateLimitHandler,
    RateLimitResponse,
    RateLimitState,
    TokenBucket,
)


class TestTokenBucket:
    """Tests for TokenBucket class."""

    def test_initialize(self) -> None:
        """Test token bucket initialization."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        assert bucket.available_tokens() == 20.0

    def test_try_acquire_success(self) -> None:
        """Test successful non-blocking acquire."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        assert bucket.try_acquire(5.0) is True
        # Allow small tolerance for timing precision
        assert 14.9 <= bucket.available_tokens() <= 15.1

    def test_try_acquire_failure(self) -> None:
        """Test failed non-blocking acquire."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        assert bucket.try_acquire(25.0) is False
        assert bucket.available_tokens() == 20.0  # No tokens consumed

    @pytest.mark.asyncio
    async def test_acquire_immediate(self) -> None:
        """Test immediate acquire when tokens available."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        wait_time = await bucket.acquire(10.0)
        assert wait_time == 0.0
        # Allow small tolerance for timing precision
        assert 9.9 <= bucket.available_tokens() <= 10.1

    @pytest.mark.asyncio
    async def test_acquire_with_wait(self) -> None:
        """Test acquire that requires waiting."""
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        # Exhaust tokens
        await bucket.acquire(10.0)
        # Allow small tolerance for timing precision
        assert bucket.available_tokens() <= 0.01

        # Acquire more - should wait
        start = datetime.now(timezone.utc)
        wait_time = await bucket.acquire(5.0)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        # Should wait approximately 0.5 seconds (5 tokens / 10 per second)
        assert 0.4 <= wait_time <= 0.7
        assert 0.4 <= elapsed <= 0.8

    @pytest.mark.asyncio
    async def test_refill_over_time(self) -> None:
        """Test token refill over time."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        await bucket.acquire(20.0)  # Exhaust all tokens
        # Allow small tolerance for timing precision
        assert bucket.available_tokens() <= 0.01

        # Wait for refill
        await asyncio.sleep(0.5)
        available = bucket.available_tokens()

        # Should have ~5 tokens (10 per second * 0.5 seconds)
        assert 4.0 <= available <= 6.0

    @pytest.mark.asyncio
    async def test_refill_capped_at_capacity(self) -> None:
        """Test that refill doesn't exceed capacity."""
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        await bucket.acquire(5.0)

        # Wait for more than capacity worth of refill time
        await asyncio.sleep(2.0)
        available = bucket.available_tokens()

        # Should be capped at capacity (10.0)
        assert available == 10.0

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self) -> None:
        """Test concurrent acquire operations."""
        bucket = TokenBucket(rate=20.0, capacity=20.0)

        async def acquire_tokens() -> float:
            return await bucket.acquire(5.0)

        # Launch concurrent acquires
        results = await asyncio.gather(
            acquire_tokens(),
            acquire_tokens(),
            acquire_tokens(),
            acquire_tokens(),
        )

        # All should complete
        assert len(results) == 4
        # Bucket should be empty (4 * 5 = 20), allow small tolerance
        assert bucket.available_tokens() <= 0.01

    def test_try_acquire_zero_tokens(self) -> None:
        """Test try_acquire with zero tokens."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        assert bucket.try_acquire(0.0) is True
        assert bucket.available_tokens() == 20.0

    @pytest.mark.asyncio
    async def test_acquire_fractional_tokens(self) -> None:
        """Test acquire with fractional token amounts."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        wait_time = await bucket.acquire(2.5)
        assert wait_time == 0.0
        # Allow small tolerance for timing precision
        assert 17.4 <= bucket.available_tokens() <= 17.6


class TestRateLimitConfig:
    """Tests for RateLimitConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 100000
        assert config.burst_multiplier == 1.5
        assert config.min_requests_per_minute == 5

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_minute=120,
            tokens_per_minute=200000,
            burst_multiplier=2.0,
            min_requests_per_minute=10,
        )
        assert config.requests_per_minute == 120
        assert config.tokens_per_minute == 200000
        assert config.burst_multiplier == 2.0
        assert config.min_requests_per_minute == 10

    def test_validation_requests_per_minute(self) -> None:
        """Test validation of requests_per_minute."""
        with pytest.raises(Exception):  # Pydantic validation error
            RateLimitConfig(requests_per_minute=0)

        with pytest.raises(Exception):
            RateLimitConfig(requests_per_minute=20000)

    def test_validation_burst_multiplier(self) -> None:
        """Test validation of burst_multiplier."""
        with pytest.raises(Exception):
            RateLimitConfig(burst_multiplier=0.5)

        with pytest.raises(Exception):
            RateLimitConfig(burst_multiplier=10.0)


class TestRateLimitHandler:
    """Tests for RateLimitHandler class."""

    def test_parse_retry_after_seconds(self) -> None:
        """Test parsing Retry-After header as seconds."""
        handler = RateLimitHandler()
        headers = {"Retry-After": "60"}

        response = handler.parse_rate_limit_response(
            status_code=429,
            headers=headers,
        )

        assert response.status_code == 429
        assert response.retry_after_seconds == 60.0
        assert response.rate_limit_remaining is None

    def test_parse_retry_after_http_date(self) -> None:
        """Test parsing Retry-After header as HTTP-date."""
        handler = RateLimitHandler()
        # Future date
        headers = {"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"}

        response = handler.parse_rate_limit_response(
            status_code=429,
            headers=headers,
        )

        assert response.status_code == 429
        # Should calculate seconds until that date
        assert response.retry_after_seconds is not None

    def test_parse_x_ratelimit_headers(self) -> None:
        """Test parsing X-RateLimit-* headers."""
        handler = RateLimitHandler()
        headers = {
            "X-RateLimit-Remaining": "10",
            "X-RateLimit-Reset": "1234567890",
        }

        response = handler.parse_rate_limit_response(
            status_code=429,
            headers=headers,
        )

        assert response.rate_limit_remaining == 10
        assert response.rate_limit_reset_at == "1234567890"

    def test_parse_anthropic_headers(self) -> None:
        """Test parsing Anthropic-specific headers."""
        handler = RateLimitHandler()
        headers = {
            "anthropic-ratelimit-requests-remaining": "5",
            "anthropic-ratelimit-requests-reset": "2025-01-15T10:00:00Z",
        }

        response = handler.parse_rate_limit_response(
            status_code=429,
            headers=headers,
        )

        assert response.rate_limit_remaining == 5
        assert response.rate_limit_reset_at == "2025-01-15T10:00:00Z"

    def test_parse_case_insensitive_headers(self) -> None:
        """Test case-insensitive header parsing."""
        handler = RateLimitHandler()
        headers = {
            "retry-after": "30",
            "X-RATELIMIT-REMAINING": "15",
        }

        response = handler.parse_rate_limit_response(
            status_code=429,
            headers=headers,
        )

        assert response.retry_after_seconds == 30.0
        assert response.rate_limit_remaining == 15

    def test_parse_with_body(self) -> None:
        """Test parsing with response body."""
        handler = RateLimitHandler()
        headers = {"Retry-After": "60"}
        body = "Rate limit exceeded. Please try again later."

        response = handler.parse_rate_limit_response(
            status_code=429,
            headers=headers,
            body=body,
        )

        assert response.error_message == body

    def test_parse_invalid_retry_after(self) -> None:
        """Test parsing invalid Retry-After header."""
        handler = RateLimitHandler()
        headers = {"Retry-After": "invalid"}

        response = handler.parse_rate_limit_response(
            status_code=429,
            headers=headers,
        )

        # Should handle gracefully
        assert response.retry_after_seconds is None

    @pytest.mark.asyncio
    async def test_handle_rate_limit_with_retry_after(self) -> None:
        """Test handling rate limit with Retry-After."""
        handler = RateLimitHandler()
        response = RateLimitResponse(
            status_code=429,
            retry_after_seconds=0.1,  # Short delay for test
        )

        start = datetime.now(timezone.utc)
        wait_time = await handler.handle_rate_limit(response)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        assert wait_time == 0.1
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_handle_rate_limit_without_retry_after(self) -> None:
        """Test handling rate limit without Retry-After."""
        handler = RateLimitHandler()
        response = RateLimitResponse(
            status_code=429,
            retry_after_seconds=None,
        )

        # Should use default backoff (mocked to 0.1 for test)
        wait_time = await handler.handle_rate_limit(response)

        # Default is 60 seconds
        assert wait_time == 60.0


class TestExponentialBackoff:
    """Tests for ExponentialBackoff class."""

    def test_initialize(self) -> None:
        """Test exponential backoff initialization."""
        config = BackoffConfig(
            initial_delay_seconds=2.0,
            max_delay_seconds=30.0,
            multiplier=2.0,
        )
        backoff = ExponentialBackoff(config)
        assert backoff is not None

    def test_next_delay_first_attempt(self) -> None:
        """Test delay calculation for first attempt."""
        config = BackoffConfig(
            initial_delay_seconds=1.0,
            multiplier=2.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        delay = backoff.next_delay(0)
        assert delay == 1.0

    def test_next_delay_exponential_growth(self) -> None:
        """Test exponential delay growth."""
        config = BackoffConfig(
            initial_delay_seconds=1.0,
            multiplier=2.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        assert backoff.next_delay(0) == 1.0
        assert backoff.next_delay(1) == 2.0
        assert backoff.next_delay(2) == 4.0
        assert backoff.next_delay(3) == 8.0

    def test_next_delay_capped_at_max(self) -> None:
        """Test delay capped at maximum."""
        config = BackoffConfig(
            initial_delay_seconds=1.0,
            max_delay_seconds=5.0,
            multiplier=2.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        assert backoff.next_delay(10) == 5.0

    def test_next_delay_with_jitter(self) -> None:
        """Test delay with jitter."""
        config = BackoffConfig(
            initial_delay_seconds=10.0,
            multiplier=2.0,
            jitter=True,
        )
        backoff = ExponentialBackoff(config)

        delays = [backoff.next_delay(0) for _ in range(10)]

        # All delays should be between 5.0 (0.5 * 10) and 10.0 (1.0 * 10)
        assert all(5.0 <= d <= 10.0 for d in delays)
        # Delays should vary (not all the same)
        assert len(set(delays)) > 1

    @pytest.mark.asyncio
    async def test_wait(self) -> None:
        """Test wait method."""
        config = BackoffConfig(
            initial_delay_seconds=0.1,
            multiplier=2.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        start = datetime.now(timezone.utc)
        wait_time = await backoff.wait(1)  # 0.1 * 2^1 = 0.2
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        assert wait_time == 0.2
        assert elapsed >= 0.2

    def test_reset(self) -> None:
        """Test reset method."""
        config = BackoffConfig()
        backoff = ExponentialBackoff(config)

        backoff.reset()
        # Should not raise any errors


class TestAdaptiveThrottler:
    """Tests for AdaptiveThrottler class."""

    def test_initialize(self) -> None:
        """Test adaptive throttler initialization."""
        config = RateLimitConfig()
        backoff_config = BackoffConfig()
        throttler = AdaptiveThrottler(config, backoff_config)

        status = throttler.get_parallelism_status()
        assert status.rate_limit_state == RateLimitState.NORMAL
        assert status.consecutive_429s == 0

    @pytest.mark.asyncio
    async def test_on_request(self) -> None:
        """Test on_request acquires tokens."""
        config = RateLimitConfig(requests_per_minute=60)
        backoff_config = BackoffConfig()
        throttler = AdaptiveThrottler(config, backoff_config)

        wait_time = await throttler.on_request()
        # Should not wait on first request
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_on_response_success(self) -> None:
        """Test on_response with successful status code."""
        config = RateLimitConfig()
        backoff_config = BackoffConfig()
        throttler = AdaptiveThrottler(config, backoff_config)

        await throttler.on_response(200, {})

        status = throttler.get_parallelism_status()
        assert status.rate_limit_state == RateLimitState.NORMAL

    @pytest.mark.asyncio
    async def test_on_response_rate_limit(self) -> None:
        """Test on_response with 429 status code."""
        config = RateLimitConfig()
        backoff_config = BackoffConfig(initial_delay_seconds=0.1)
        throttler = AdaptiveThrottler(config, backoff_config)

        # Trigger rate limit
        await throttler.on_response(429, {"Retry-After": "0.1"})

        status = throttler.get_parallelism_status()
        assert status.consecutive_429s == 1
        assert status.last_429_at is not None

    @pytest.mark.asyncio
    async def test_parallelism_reduction_after_429(self) -> None:
        """Test parallelism reduction after 429 response."""
        config = RateLimitConfig()
        backoff_config = BackoffConfig(initial_delay_seconds=0.1)
        throttler = AdaptiveThrottler(config, backoff_config)

        # Get initial recommendation
        initial = throttler.get_recommended_concurrency()

        # Trigger rate limit
        await throttler.on_response(429, {"Retry-After": "0.1"})

        # Should reduce recommendation
        reduced = throttler.get_recommended_concurrency()
        assert reduced < initial

    @pytest.mark.asyncio
    async def test_parallelism_gradual_recovery(self) -> None:
        """Test gradual parallelism recovery."""
        config = RateLimitConfig()
        backoff_config = BackoffConfig(initial_delay_seconds=0.1)
        throttler = AdaptiveThrottler(config, backoff_config)

        # Trigger rate limit
        await throttler.on_response(429, {"Retry-After": "0.1"})
        reduced = throttler.get_recommended_concurrency()

        # Wait for recovery period
        await asyncio.sleep(0.1)

        # Successful response after recovery period
        await throttler.on_response(200, {})

        # Should start restoring (might take multiple successes)
        current = throttler.get_recommended_concurrency()
        assert current >= reduced

    @pytest.mark.asyncio
    async def test_multiple_consecutive_429s(self) -> None:
        """Test multiple consecutive 429 responses."""
        config = RateLimitConfig()
        backoff_config = BackoffConfig(initial_delay_seconds=0.1)
        throttler = AdaptiveThrottler(config, backoff_config)

        # Trigger multiple rate limits
        for _ in range(3):
            await throttler.on_response(429, {"Retry-After": "0.1"})

        status = throttler.get_parallelism_status()
        assert status.consecutive_429s == 3

    @pytest.mark.asyncio
    async def test_recommended_concurrency_minimum(self) -> None:
        """Test recommended concurrency has minimum."""
        config = RateLimitConfig(min_requests_per_minute=2)
        backoff_config = BackoffConfig(initial_delay_seconds=0.1)
        throttler = AdaptiveThrottler(config, backoff_config)

        # Trigger many rate limits
        for _ in range(20):
            await throttler.on_response(429, {"Retry-After": "0.1"})

        # Should not go below minimum
        recommended = throttler.get_recommended_concurrency()
        assert recommended >= config.min_requests_per_minute

    def test_get_parallelism_status(self) -> None:
        """Test get_parallelism_status returns correct structure."""
        config = RateLimitConfig()
        backoff_config = BackoffConfig()
        throttler = AdaptiveThrottler(config, backoff_config)

        status = throttler.get_parallelism_status()

        assert isinstance(status, ParallelismReduction)
        assert status.current_max > 0
        assert status.recommended_max > 0
        assert status.rate_limit_state == RateLimitState.NORMAL
        assert status.consecutive_429s == 0
        assert status.last_429_at is None

    @pytest.mark.asyncio
    async def test_rapid_429_burst(self) -> None:
        """Test handling rapid burst of 429 responses."""
        config = RateLimitConfig()
        backoff_config = BackoffConfig(initial_delay_seconds=0.1)
        throttler = AdaptiveThrottler(config, backoff_config)

        # Rapid burst
        tasks = [
            throttler.on_response(429, {"Retry-After": "0.05"}) for _ in range(5)
        ]
        await asyncio.gather(*tasks)

        status = throttler.get_parallelism_status()
        assert status.consecutive_429s == 5


class TestRateLimitState:
    """Tests for RateLimitState enum."""

    def test_enum_values(self) -> None:
        """Test enum has expected values."""
        assert RateLimitState.NORMAL.value == "normal"
        assert RateLimitState.THROTTLED.value == "throttled"
        assert RateLimitState.BACKING_OFF.value == "backing_off"
        assert RateLimitState.BLOCKED.value == "blocked"


class TestBackoffConfig:
    """Tests for BackoffConfig validation."""

    def test_default_config(self) -> None:
        """Test default backoff configuration."""
        config = BackoffConfig()
        assert config.initial_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.multiplier == 2.0
        assert config.jitter is True

    def test_custom_config(self) -> None:
        """Test custom backoff configuration."""
        config = BackoffConfig(
            initial_delay_seconds=0.5,
            max_delay_seconds=30.0,
            multiplier=3.0,
            jitter=False,
        )
        assert config.initial_delay_seconds == 0.5
        assert config.max_delay_seconds == 30.0
        assert config.multiplier == 3.0
        assert config.jitter is False


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_token_bucket_empty(self) -> None:
        """Test token bucket when completely empty."""
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        bucket.try_acquire(10.0)
        # Allow small tolerance for timing precision
        assert bucket.available_tokens() <= 0.01
        assert bucket.try_acquire(0.1) is False

    def test_rate_limit_response_no_headers(self) -> None:
        """Test parsing rate limit response with no headers."""
        handler = RateLimitHandler()
        response = handler.parse_rate_limit_response(
            status_code=429,
            headers={},
        )

        assert response.status_code == 429
        assert response.retry_after_seconds is None
        assert response.rate_limit_remaining is None

    def test_exponential_backoff_zero_attempt(self) -> None:
        """Test exponential backoff with zero attempt."""
        config = BackoffConfig(initial_delay_seconds=5.0, jitter=False)
        backoff = ExponentialBackoff(config)

        delay = backoff.next_delay(0)
        assert delay == 5.0

    @pytest.mark.asyncio
    async def test_concurrent_token_bucket_stress(self) -> None:
        """Test token bucket under concurrent stress."""
        bucket = TokenBucket(rate=100.0, capacity=100.0)

        async def stress_acquire() -> bool:
            try:
                await bucket.acquire(1.0)
                return True
            except Exception:
                return False

        # Launch many concurrent acquires
        results = await asyncio.gather(*[stress_acquire() for _ in range(50)])

        # All should succeed eventually
        assert all(results)

    def test_negative_retry_after_handled(self) -> None:
        """Test handling of past HTTP-date in Retry-After."""
        handler = RateLimitHandler()
        # Past date
        headers = {"Retry-After": "Wed, 21 Oct 2020 07:28:00 GMT"}

        response = handler.parse_rate_limit_response(
            status_code=429,
            headers=headers,
        )

        # Should be 0 or None (past date)
        if response.retry_after_seconds is not None:
            assert response.retry_after_seconds >= 0.0
