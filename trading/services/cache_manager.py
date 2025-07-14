"""
Cache Manager for QuantGPT

Provides caching functionality for query results, model evaluations,
and other expensive operations to improve performance.
"""

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Optional

import redis

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching for QuantGPT operations.

    Provides in-memory and Redis-based caching with TTL support,
    cache invalidation, and performance metrics.
    """

    def __init__(
        self,
        redis_client: redis.Redis = None,
        cache_enabled: bool = True,
        ttl: int = 3600,
    ):
        """
        Initialize cache manager.

        Args:
            redis_client: Redis client for distributed caching
            cache_enabled: Whether caching is enabled
            ttl: Time-to-live for cache entries in seconds
        """
        self.redis_client = redis_client
        self.cache_enabled = cache_enabled
        self.ttl = ttl
        self._memory_cache = {}
        self._cache_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.cache_enabled:
            return None

        try:
            # Try Redis first
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    self._cache_stats["hits"] += 1
                    return json.loads(value)

            # Fallback to memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if time.time() < entry["expires"]:
                    self._cache_stats["hits"] += 1
                    return entry["value"]
                else:
                    del self._memory_cache[key]

            self._cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if successful, False otherwise
        """
        if not self.cache_enabled:
            return False

        try:
            ttl = ttl or self.ttl
            expires = time.time() + ttl

            # Try Redis first
            if self.redis_client:
                serialized = json.dumps(value)
                self.redis_client.setex(key, ttl, serialized)

            # Also store in memory cache
            self._memory_cache[key] = {"value": value, "expires": expires}

            self._cache_stats["sets"] += 1
            return True

        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from Redis
            if self.redis_client:
                self.redis_client.delete(key)

            # Delete from memory cache
            if key in self._memory_cache:
                del self._memory_cache[key]

            self._cache_stats["deletes"] += 1
            return True

        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear memory cache
            self._memory_cache.clear()

            # Clear Redis cache (if using pattern)
            if self.redis_client:
                # Note: This is a simplified clear - in production you might want
                # to use SCAN to avoid blocking
                pass

            return True

        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (
            self._cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            **self._cache_stats,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache),
            "cache_enabled": self.cache_enabled,
        }

    def cache_result(self, prefix: str, ttl: Optional[int] = None):
        """
        Decorator to cache function results.

        Args:
            prefix: Cache key prefix
            ttl: Time-to-live in seconds

        Returns:
            Decorated function
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(prefix, *args, **kwargs)

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator


def cache_result(prefix: str, ttl: Optional[int] = None):
    """
    Global cache decorator.

    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would use a global cache manager instance
            # For now, just execute the function
            return func(*args, **kwargs)

        return wrapper

    return decorator
