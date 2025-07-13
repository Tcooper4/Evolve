"""
Rate Limiter for QuantGPT

Provides rate limiting functionality to prevent API abuse
and ensure fair usage of external services.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API calls and operations.

    Supports sliding window rate limiting with configurable
    limits per time period.
    """

    def __init__(self, max_calls: int = 100, time_window: int = 3600):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self._call_timestamps = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """
        Check if a call is allowed.

        Args:
            key: Rate limit key (e.g., 'openai_api', 'model_build')

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        with self._lock:
            now = time.time()
            timestamps = self._call_timestamps[key]

            # Remove old timestamps outside the window
            cutoff = now - self.time_window
            timestamps = [ts for ts in timestamps if ts > cutoff]
            self._call_timestamps[key] = timestamps

            # Check if we're under the limit
            if len(timestamps) < self.max_calls:
                return True, None

            # Calculate retry after time
            oldest_timestamp = min(timestamps)
            retry_after = int(oldest_timestamp + self.time_window - now)

            return False, retry_after

    def record_call(self, key: str) -> bool:
        """
        Record a call for rate limiting.

        Args:
            key: Rate limit key

        Returns:
            True if call was recorded, False if rate limit exceeded
        """
        is_allowed, retry_after = self.is_allowed(key)

        if is_allowed:
            with self._lock:
                self._call_timestamps[key].append(time.time())
            return True
        else:
            logger.warning(f"Rate limit exceeded for {key}. Retry after {retry_after} seconds.")
            return False

    def get_status(self, key: str) -> Dict[str, any]:
        """
        Get rate limit status for a key.

        Args:
            key: Rate limit key

        Returns:
            Dictionary with rate limit status
        """
        with self._lock:
            now = time.time()
            timestamps = self._call_timestamps[key]

            # Remove old timestamps
            cutoff = now - self.time_window
            timestamps = [ts for ts in timestamps if ts > cutoff]
            self._call_timestamps[key] = timestamps

            calls_used = len(timestamps)
            calls_remaining = max(0, self.max_calls - calls_used)

            return {
                "key": key,
                "calls_used": calls_used,
                "calls_remaining": calls_remaining,
                "max_calls": self.max_calls,
                "time_window": self.time_window,
                "reset_time": now + self.time_window if timestamps else None,
            }

    def reset(self, key: str) -> None:
        """
        Reset rate limit for a key.

        Args:
            key: Rate limit key
        """
        with self._lock:
            self._call_timestamps[key] = []

    def reset_all(self) -> None:
        """Reset all rate limits."""
        with self._lock:
            self._call_timestamps.clear()


class RateLimitDecorator:
    """
    Decorator for rate limiting function calls.
    """

    def __init__(self, rate_limiter: RateLimiter, key: str):
        """
        Initialize rate limit decorator.

        Args:
            rate_limiter: Rate limiter instance
            key: Rate limit key
        """
        self.rate_limiter = rate_limiter
        self.key = key

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if not self.rate_limiter.record_call(self.key):
                raise Exception(f"Rate limit exceeded for {self.key}")
            return func(*args, **kwargs)

        return wrapper


def rate_limit(max_calls: int = 100, time_window: int = 3600, key: str = "default"):
    """
    Decorator for rate limiting.

    Args:
        max_calls: Maximum calls allowed
        time_window: Time window in seconds
        key: Rate limit key

    Returns:
        Decorated function
    """
    limiter = RateLimiter(max_calls, time_window)
    return RateLimitDecorator(limiter, key)
