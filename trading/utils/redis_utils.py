"""Redis utilities with connection pooling and error handling.

This module provides utilities for managing Redis connections with connection
pooling, automatic reconnection, and comprehensive error handling.
"""

import json
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import redis
from redis.exceptions import ConnectionError, RedisError, ResponseError, TimeoutError

logger = logging.getLogger(__name__)


class RedisManager:
    """Manager for Redis connections with pooling and error handling."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
    ):
        """Initialize the Redis manager.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum number of connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            retry_on_timeout: Whether to retry on timeout
            health_check_interval: Health check interval in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval

        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
        )

        self.client = redis.Redis(connection_pool=self.pool)
        self.last_health_check = 0

    def _check_health(self) -> bool:
        """Check Redis connection health.

        Returns:
            Whether the connection is healthy
        """
        current_time = time.time()

        if current_time - self.last_health_check < self.health_check_interval:
            return True

        try:
            self.client.ping()
            self.last_health_check = current_time
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def _handle_error(self, error: RedisError) -> None:
        """Handle Redis errors.

        Args:
            error: Redis error to handle
        """
        if isinstance(error, ConnectionError):
            logger.error(f"Redis connection error: {error}")
        elif isinstance(error, TimeoutError):
            logger.error(f"Redis timeout error: {error}")
        elif isinstance(error, ResponseError):
            logger.error(f"Redis response error: {error}")
        else:
            logger.error(f"Redis error: {error}")

    def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis.

        Args:
            key: Key to get

        Returns:
            Value if found, None otherwise
        """
        try:
            if not self._check_health():
                return None

            value = self.client.get(key)
            return json.loads(value) if value else None
        except RedisError as e:
            self._handle_error(e)

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set a value in Redis.

        Args:
            key: Key to set
            value: Value to set
            expire: Expiration time in seconds

        Returns:
            Whether the operation was successful
        """
        try:
            if not self._check_health():
                return False

            value = json.dumps(value)
            if expire:
                return self.client.setex(key, expire, value)
            return self.client.set(key, value)
        except RedisError as e:
            self._handle_error(e)
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from Redis.

        Args:
            key: Key to delete

        Returns:
            Whether the operation was successful
        """
        try:
            if not self._check_health():
                return False

            return bool(self.client.delete(key))
        except RedisError as e:
            self._handle_error(e)
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis.

        Args:
            key: Key to check

        Returns:
            Whether the key exists
        """
        try:
            if not self._check_health():
                return False

            return bool(self.client.exists(key))
        except RedisError as e:
            self._handle_error(e)
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern.

        Args:
            pattern: Pattern to match

        Returns:
            List of matching keys
        """
        try:
            if not self._check_health():
                return []

            return [k.decode("utf-8") for k in self.client.keys(pattern)]
        except RedisError as e:
            self._handle_error(e)
            return []

    def hget(self, name: str, key: str) -> Optional[Any]:
        """Get a field from a hash.

        Args:
            name: Hash name
            key: Field key

        Returns:
            Field value if found, None otherwise
        """
        try:
            if not self._check_health():
                return None

            value = self.client.hget(name, key)
            return json.loads(value) if value else None
        except RedisError as e:
            self._handle_error(e)

    def hset(self, name: str, key: str, value: Any) -> bool:
        """Set a field in a hash.

        Args:
            name: Hash name
            key: Field key
            value: Field value

        Returns:
            Whether the operation was successful
        """
        try:
            if not self._check_health():
                return False

            value = json.dumps(value)
            return bool(self.client.hset(name, key, value))
        except RedisError as e:
            self._handle_error(e)
            return False

    def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all fields from a hash.

        Args:
            name: Hash name

        Returns:
            Dictionary of field-value pairs
        """
        try:
            if not self._check_health():
                return {}

            data = self.client.hgetall(name)
            return {k.decode("utf-8"): json.loads(v) for k, v in data.items()}
        except RedisError as e:
            self._handle_error(e)
            return {}

    def publish(self, channel: str, message: Any) -> bool:
        """Publish a message to a channel.

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Whether the operation was successful
        """
        try:
            if not self._check_health():
                return False

            message = json.dumps(message)
            return bool(self.client.publish(channel, message))
        except RedisError as e:
            self._handle_error(e)
            return False

    def subscribe(self, channel: str) -> Optional[redis.client.PubSub]:
        """Subscribe to a channel.

        Args:
            channel: Channel name

        Returns:
            PubSub object if successful, None otherwise
        """
        try:
            if not self._check_health():
                return None

            pubsub = self.client.pubsub()
            pubsub.subscribe(channel)
            return pubsub
        except RedisError as e:
            self._handle_error(e)

    def close(self) -> None:
        """Close the Redis connection."""
        try:
            self.client.close()
        except RedisError as e:
            self._handle_error(e)


def with_redis_retry(max_retries: int = 3, delay: float = 1.0) -> Callable:
    """Decorator for retrying Redis operations.

    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return {
                        "success": True,
                        "result": {
                            "success": True,
                            "result": {
                                "success": True,
                                "result": func(*args, **kwargs),
                                "message": "Operation completed successfully",
                                "timestamp": datetime.now().isoformat(),
                            },
                            "message": "Operation completed successfully",
                            "timestamp": datetime.now().isoformat(),
                        },
                        "message": "Operation completed successfully",
                        "timestamp": datetime.now().isoformat(),
                    }
                except (ConnectionError, TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Redis operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


# Create singleton instance
redis_manager = RedisManager()
