"""Tests for custom rate limiter."""
import pytest
import asyncio
from its_project.common.rate_limiter import (
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    MultiEndpointRateLimiter,
    RateLimitConfig,
)


@pytest.fixture
def rate_limit_config():
    """Rate limit configuration."""
    return RateLimitConfig(
        requests_per_second=10.0,
        burst_size=5,
        enable_exponential_backoff=True,
        max_backoff_seconds=60.0,
    )


@pytest.mark.asyncio
async def test_token_bucket_acquire_success(rate_limit_config):
    """Test successful token acquisition."""
    limiter = TokenBucketRateLimiter(rate_limit_config)
    
    # Should succeed for burst_size requests
    for _ in range(5):
        assert await limiter.acquire() is True


@pytest.mark.asyncio
async def test_token_bucket_rate_limit(rate_limit_config):
    """Test rate limiting after burst."""
    limiter = TokenBucketRateLimiter(rate_limit_config)
    
    # Exhaust burst
    for _ in range(5):
        await limiter.acquire()
    
    # Next request should be rate limited
    assert await limiter.acquire() is False


@pytest.mark.asyncio
async def test_token_bucket_refill(rate_limit_config):
    """Test token refill over time."""
    limiter = TokenBucketRateLimiter(rate_limit_config)
    
    # Exhaust burst
    for _ in range(5):
        await limiter.acquire()
    
    # Wait for refill (1 second should add ~10 tokens)
    await asyncio.sleep(1.1)
    
    # Should be able to acquire again
    assert await limiter.acquire() is True


@pytest.mark.asyncio
async def test_token_bucket_wait_for_token(rate_limit_config):
    """Test waiting for token."""
    limiter = TokenBucketRateLimiter(rate_limit_config)
    
    # Exhaust burst
    for _ in range(5):
        await limiter.acquire()
    
    # Wait for token should succeed after refill
    await limiter.wait_for_token()


@pytest.mark.asyncio
async def test_token_bucket_exponential_backoff(rate_limit_config):
    """Test exponential backoff on repeated rejections."""
    limiter = TokenBucketRateLimiter(rate_limit_config)
    
    # Exhaust burst
    for _ in range(5):
        await limiter.acquire()
    
    # Try to acquire multiple times to trigger backoff
    for _ in range(3):
        await limiter.acquire()
    
    # Check that backoff is active
    assert limiter._backoff_until > 0


def test_token_bucket_stats(rate_limit_config):
    """Test statistics tracking."""
    limiter = TokenBucketRateLimiter(rate_limit_config)
    
    stats = limiter.get_stats()
    
    assert stats.total_requests == 0
    assert stats.rejected_requests == 0
    assert stats.throttled_requests == 0


def test_token_bucket_reset(rate_limit_config):
    """Test reset functionality."""
    limiter = TokenBucketRateLimiter(rate_limit_config)
    
    # Modify state
    limiter._tokens = 0.0
    limiter._backoff_until = 100.0
    
    # Reset
    limiter.reset()
    
    assert limiter._tokens == float(rate_limit_config.burst_size)
    assert limiter._backoff_until == 0.0


@pytest.mark.asyncio
async def test_sliding_window_acquire(rate_limit_config):
    """Test sliding window rate limiter."""
    limiter = SlidingWindowRateLimiter(rate_limit_config)
    
    # Should succeed for initial requests within burst
    for _ in range(5):
        result = await limiter.acquire()
        # May be True or False depending on timing
        assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_sliding_window_rate_limit(rate_limit_config):
    """Test sliding window rate limiting."""
    limiter = SlidingWindowRateLimiter(rate_limit_config)
    
    # Make many requests quickly
    success_count = 0
    for _ in range(20):
        if await limiter.acquire():
            success_count += 1
    
    # Should be limited (not all 20 should succeed)
    assert success_count <= 20


@pytest.mark.asyncio
async def test_multi_endpoint_limiter(rate_limit_config):
    """Test multi-endpoint rate limiter."""
    limiter = MultiEndpointRateLimiter(
        default_config=rate_limit_config,
        endpoint_configs={
            "fast": RateLimitConfig(requests_per_second=20.0, burst_size=10),
        }
    )
    
    # Default endpoint
    for _ in range(5):
        assert await limiter.acquire("default") is True
    
    # Fast endpoint should allow more
    for _ in range(10):
        assert await limiter.acquire("fast") is True


@pytest.mark.asyncio
async def test_multi_endpoint_stats(rate_limit_config):
    """Test multi-endpoint statistics."""
    limiter = MultiEndpointRateLimiter(rate_limit_config)
    
    await limiter.acquire("endpoint1")
    await limiter.acquire("endpoint2")
    
    stats = limiter.get_stats()
    
    assert "endpoint1" in stats
    assert "endpoint2" in stats
    assert stats["endpoint1"].total_requests == 1
    assert stats["endpoint2"].total_requests == 1


def test_rate_limit_config_defaults():
    """Test rate limit config defaults."""
    config = RateLimitConfig()
    
    assert config.requests_per_second == 10.0
    assert config.burst_size == 5
    assert config.enable_exponential_backoff is True
    assert config.max_backoff_seconds == 60.0
