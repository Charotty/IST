from __future__ import annotations

import asyncio
import time
import logging
from typing import Optional, Dict
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    requests_per_minute: Optional[float] = None
    burst_size: int = 5
    enable_exponential_backoff: bool = True
    max_backoff_seconds: float = 60.0
    backoff_base: float = 2.0


@dataclass
class RateLimitStats:
    """Statistics for rate limiter."""
    total_requests: int = 0
    rejected_requests: int = 0
    throttled_requests: int = 0
    current_wait_time: float = 0.0
    last_request_time: Optional[datetime] = None


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with exponential backoff.
    
    Uses token bucket algorithm for smooth rate limiting and
    exponential backoff when limits are exceeded.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._tokens = float(config.burst_size)
        self._last_update = time.time()
        self._lock = asyncio.Lock()
        self._stats = RateLimitStats()
        self._backoff_until: float = 0.0
        self._consecutive_rejections: int = 0

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now
        
        # Add tokens based on rate
        tokens_to_add = elapsed * self.config.requests_per_second
        self._tokens = min(
            self._tokens + tokens_to_add,
            float(self.config.burst_size)
        )

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            # Check if we're in backoff period
            now = time.time()
            if now < self._backoff_until:
                wait_time = self._backoff_until - now
                self._stats.current_wait_time = wait_time
                self._stats.throttled_requests += 1
                logger.debug(f"In backoff, waiting {wait_time:.2f}s")
                return False

            self._add_tokens()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.total_requests += 1
                self._stats.last_request_time = datetime.now()
                self._consecutive_rejections = 0
                return True
            else:
                # Rate limit exceeded
                self._stats.rejected_requests += 1
                self._consecutive_rejections += 1
                
                if self.config.enable_exponential_backoff:
                    backoff_time = min(
                        self.config.backoff_base ** self._consecutive_rejections,
                        self.config.max_backoff_seconds
                    )
                    self._backoff_until = now + backoff_time
                    logger.warning(
                        f"Rate limit exceeded, backing off for {backoff_time:.2f}s "
                        f"(rejections: {self._consecutive_rejections})"
                    )
                
                return False

    async def wait_for_token(self, tokens: float = 1.0) -> None:
        """
        Wait until a token is available.
        
        Args:
            tokens: Number of tokens to acquire
        """
        while not await self.acquire(tokens):
            wait_time = self._stats.current_wait_time
            if wait_time > 0:
                await asyncio.sleep(min(wait_time, 1.0))
            else:
                await asyncio.sleep(0.1)

    def get_stats(self) -> RateLimitStats:
        """Get current statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self._tokens = float(self.config.burst_size)
        self._last_update = time.time()
        self._backoff_until = 0.0
        self._consecutive_rejections = 0
        self._stats = RateLimitStats()


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with exponential backoff.
    
    Tracks requests in a sliding time window for precise rate limiting.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._window: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._stats = RateLimitStats()
        self._backoff_until: float = 0.0
        self._consecutive_rejections: int = 0

    def _cleanup_old_requests(self, now: float) -> None:
        """Remove requests outside the time window."""
        window_size = 1.0 / self.config.requests_per_second
        while self._window and now - self._window[0] > window_size:
            self._window.popleft()

    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.
        
        Returns:
            True if allowed, False if rate limited
        """
        async with self._lock:
            now = time.time()
            
            # Check if we're in backoff period
            if now < self._backoff_until:
                wait_time = self._backoff_until - now
                self._stats.current_wait_time = wait_time
                self._stats.throttled_requests += 1
                return False

            self._cleanup_old_requests(now)
            
            # Check if we can make a request
            window_size = 1.0 / self.config.requests_per_second
            max_requests = int(window_size * self.config.requests_per_second)
            
            if len(self._window) < max_requests:
                self._window.append(now)
                self._stats.total_requests += 1
                self._stats.last_request_time = datetime.now()
                self._consecutive_rejections = 0
                return True
            else:
                # Rate limit exceeded
                self._stats.rejected_requests += 1
                self._consecutive_rejections += 1
                
                if self.config.enable_exponential_backoff:
                    backoff_time = min(
                        self.config.backoff_base ** self._consecutive_rejections,
                        self.config.max_backoff_seconds
                    )
                    self._backoff_until = now + backoff_time
                    logger.warning(
                        f"Rate limit exceeded, backing off for {backoff_time:.2f}s "
                        f"(rejections: {self._consecutive_rejections})"
                    )
                
                return False

    async def wait_for_token(self) -> None:
        """Wait until a request is allowed."""
        while not await self.acquire():
            wait_time = self._stats.current_wait_time
            if wait_time > 0:
                await asyncio.sleep(min(wait_time, 1.0))
            else:
                await asyncio.sleep(0.1)

    def get_stats(self) -> RateLimitStats:
        """Get current statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self._window.clear()
        self._backoff_until = 0.0
        self._consecutive_rejections = 0
        self._stats = RateLimitStats()


class MultiEndpointRateLimiter:
    """
    Rate limiter for multiple endpoints with separate limits.
    """

    def __init__(
        self,
        default_config: RateLimitConfig,
        endpoint_configs: Optional[Dict[str, RateLimitConfig]] = None,
    ) -> None:
        self.default_config = default_config
        self.endpoint_configs = endpoint_configs or {}
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._lock = asyncio.Lock()

    async def get_limiter(self, endpoint: str) -> TokenBucketRateLimiter:
        """Get or create rate limiter for endpoint."""
        async with self._lock:
            if endpoint not in self._limiters:
                config = self.endpoint_configs.get(endpoint, self.default_config)
                self._limiters[endpoint] = TokenBucketRateLimiter(config)
            return self._limiters[endpoint]

    async def acquire(self, endpoint: str, tokens: float = 1.0) -> bool:
        """Acquire tokens for specific endpoint."""
        limiter = await self.get_limiter(endpoint)
        return await limiter.acquire(tokens)

    async def wait_for_token(self, endpoint: str, tokens: float = 1.0) -> None:
        """Wait for token for specific endpoint."""
        limiter = await self.get_limiter(endpoint)
        await limiter.wait_for_token(tokens)

    def get_stats(self, endpoint: Optional[str] = None) -> Dict[str, RateLimitStats]:
        """Get statistics for all or specific endpoint."""
        if endpoint:
            limiter = self._limiters.get(endpoint)
            return {endpoint: limiter.get_stats()} if limiter else {}
        
        return {ep: lim.get_stats() for ep, lim in self._limiters.items()}

    def reset(self, endpoint: Optional[str] = None) -> None:
        """Reset limiter(s)."""
        if endpoint:
            if endpoint in self._limiters:
                self._limiters[endpoint].reset()
        else:
            for limiter in self._limiters.values():
                limiter.reset()


def rate_limiter_decorator(
    limiter: TokenBucketRateLimiter,
    tokens: float = 1.0,
):
    """
    Decorator for rate limiting function calls.
    
    Args:
        limiter: Rate limiter instance
        tokens: Number of tokens required per call
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            await limiter.wait_for_token(tokens)
            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper(*args, **kwargs)
            else:
                # For sync functions, run in event loop
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    limiter.wait_for_token(tokens)
                )
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
