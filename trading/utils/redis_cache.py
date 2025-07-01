"""
Redis Cache Module

Centralized Redis cache with proper JSON serialization, TTLs, and memory sharing.
Provides caching for strategy signals, predictions, and agent memory states.
"""

import json
import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import os
import pickle
import hashlib

# Import Redis with fallback handling
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install with: pip install redis")

from trading.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class RedisCache:
    """Advanced Redis cache with JSON serialization and TTL management."""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 decode_responses: bool = True):
        """Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            decode_responses: Whether to decode responses
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.decode_responses = decode_responses
        
        # Initialize Redis client
        self.client = None
        self._initialize_client()
        
        # Default TTL values (in seconds)
        self.default_ttls = {
            'strategy_signals': 3600,  # 1 hour
            'predictions': 1800,       # 30 minutes
            'agent_memory': 7200,      # 2 hours
            'market_data': 300,        # 5 minutes
            'sentiment_data': 1800,    # 30 minutes
            'optimization_results': 86400,  # 24 hours
            'model_cache': 3600,       # 1 hour
            'session_data': 1800       # 30 minutes
        }
        
        logger.info(f"Redis cache initialized: {host}:{port}/db{db}")
    
    def _initialize_client(self):
        """Initialize Redis client with error handling."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - cache will be disabled")
            return
        
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=self.decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.client = None
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value to JSON string with proper handling of complex objects.
        
        Args:
            value: Value to serialize
            
        Returns:
            JSON string
        """
        try:
            # Handle datetime objects
            if isinstance(value, datetime):
                return json.dumps({'__datetime__': value.isoformat()})
            
            # Handle pandas DataFrames
            if hasattr(value, 'to_dict'):
                try:
                    return json.dumps({'__dataframe__': value.to_dict()})
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug(f"DataFrame serialization failed: {e}")
                    pass
            
            # Handle numpy arrays
            if hasattr(value, 'tolist'):
                try:
                    return json.dumps({'__numpy__': value.tolist()})
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug(f"NumPy array serialization failed: {e}")
                    pass
            
            # Default JSON serialization
            return json.dumps(value, default=str)
            
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            # Fallback to pickle for complex objects
            try:
                pickled = pickle.dumps(value)
                return json.dumps({'__pickle__': pickled.hex()})
            except Exception as e2:
                logger.error(f"Pickle serialization also failed: {e2}")
                return json.dumps({'__error__': str(value)})
    
    def _deserialize_value(self, value: str) -> Any:
        """Deserialize JSON string to original value.
        
        Args:
            value: JSON string to deserialize
            
        Returns:
            Deserialized value
        """
        try:
            data = json.loads(value)
            
            # Handle datetime objects
            if isinstance(data, dict) and '__datetime__' in data:
                return datetime.fromisoformat(data['__datetime__'])
            
            # Handle pandas DataFrames
            if isinstance(data, dict) and '__dataframe__' in data:
                import pandas as pd
                return pd.DataFrame(data['__dataframe__'])
            
            # Handle numpy arrays
            if isinstance(data, dict) and '__numpy__' in data:
                import numpy as np
                return np.array(data['__numpy__'])
            
            # Handle pickled objects
            if isinstance(data, dict) and '__pickle__' in data:
                pickled_bytes = bytes.fromhex(data['__pickle__'])
                return pickle.loads(pickled_bytes)
            
            # Handle error objects
            if isinstance(data, dict) and '__error__' in data:
                logger.warning(f"Deserialized error object: {data['__error__']}")
                return data['__error__']
            
            return data
            
        except Exception as e:
            logger.error(f"Error deserializing value: {e}")
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, category: str = 'default') -> bool:
        """Set a value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            category: Category for default TTL lookup
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Redis not available - cannot set cache")
            return False
        
        try:
            # Serialize value
            serialized_value = self._serialize_value(value)
            
            # Determine TTL
            if ttl is None:
                ttl = self.default_ttls.get(category, 3600)  # Default 1 hour
            
            # Set with TTL
            result = self.client.setex(key, ttl, serialized_value)
            
            if result:
                logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            else:
                logger.warning(f"Failed to set cache: {key}")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting cache {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.client:
            logger.warning("Redis not available - cannot get cache")
            return None
        
        try:
            value = self.client.get(key)
            
            if value is None:
                logger.debug(f"Cache miss: {key}")
                return None
            
            # Deserialize value
            deserialized_value = self._deserialize_value(value)
            logger.debug(f"Cache hit: {key}")
            
            return deserialized_value
            
        except Exception as e:
            logger.error(f"Error getting cache {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Redis not available - cannot delete cache")
            return False
        
        try:
            result = self.client.delete(key)
            logger.debug(f"Cache delete: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting cache {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.client:
            return False
        
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache existence {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            return bool(self.client.expire(key, ttl))
        except Exception as e:
            logger.error(f"Error setting expiration {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        if not self.client:
            return -2
        
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL {key}: {e}")
            return -2
    
    def set_strategy_signals(self, ticker: str, signals: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache strategy signals for a ticker.
        
        Args:
            ticker: Stock ticker
            signals: Strategy signals dictionary
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key = f"strategy_signals:{ticker}"
        return self.set(key, signals, ttl, 'strategy_signals')
    
    def get_strategy_signals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached strategy signals for a ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Strategy signals or None if not found
        """
        key = f"strategy_signals:{ticker}"
        return self.get(key)
    
    def set_predictions(self, model_name: str, ticker: str, predictions: Any, ttl: Optional[int] = None) -> bool:
        """Cache model predictions.
        
        Args:
            model_name: Name of the model
            ticker: Stock ticker
            predictions: Model predictions
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key = f"predictions:{model_name}:{ticker}"
        return self.set(key, predictions, ttl, 'predictions')
    
    def get_predictions(self, model_name: str, ticker: str) -> Optional[Any]:
        """Get cached model predictions.
        
        Args:
            model_name: Name of the model
            ticker: Stock ticker
            
        Returns:
            Model predictions or None if not found
        """
        key = f"predictions:{model_name}:{ticker}"
        return self.get(key)
    
    def set_agent_memory(self, agent_id: str, memory_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache agent memory state.
        
        Args:
            agent_id: Agent identifier
            memory_data: Agent memory data
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key = f"agent_memory:{agent_id}"
        return self.set(key, memory_data, ttl, 'agent_memory')
    
    def get_agent_memory(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get cached agent memory state.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent memory data or None if not found
        """
        key = f"agent_memory:{agent_id}"
        return self.get(key)
    
    def set_market_data(self, ticker: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Cache market data.
        
        Args:
            ticker: Stock ticker
            data: Market data
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key = f"market_data:{ticker}"
        return self.set(key, data, ttl, 'market_data')
    
    def get_market_data(self, ticker: str) -> Optional[Any]:
        """Get cached market data.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Market data or None if not found
        """
        key = f"market_data:{ticker}"
        return self.get(key)
    
    def set_sentiment_data(self, ticker: str, sentiment_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache sentiment data.
        
        Args:
            ticker: Stock ticker
            sentiment_data: Sentiment analysis data
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key = f"sentiment_data:{ticker}"
        return self.set(key, sentiment_data, ttl, 'sentiment_data')
    
    def get_sentiment_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached sentiment data.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Sentiment data or None if not found
        """
        key = f"sentiment_data:{ticker}"
        return self.get(key)
    
    def clear_category(self, category: str) -> int:
        """Clear all keys in a category.
        
        Args:
            category: Category to clear
            
        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0
        
        try:
            pattern = f"{category}:*"
            keys = self.client.keys(pattern)
            
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} keys for category: {category}")
                return deleted
            else:
                logger.debug(f"No keys found for category: {category}")
                return 0
                
        except Exception as e:
            logger.error(f"Error clearing category {category}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.client:
            return {'error': 'Redis not available'}
        
        try:
            info = self.client.info()
            
            stats = {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0),
                'db_size': self.client.dbsize()
            }
            
            # Calculate hit rate
            total_requests = stats['keyspace_hits'] + stats['keyspace_misses']
            if total_requests > 0:
                stats['hit_rate'] = stats['keyspace_hits'] / total_requests
            else:
                stats['hit_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection.
        
        Returns:
            Dictionary with health status
        """
        if not self.client:
            return {
                'status': 'unavailable',
                'error': 'Redis client not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Test ping
            ping_result = self.client.ping()
            
            # Test basic operations
            test_key = f"health_check:{datetime.now().timestamp()}"
            set_result = self.set(test_key, {'test': 'data'}, 60)
            get_result = self.get(test_key)
            delete_result = self.delete(test_key)
            
            health_status = {
                'status': 'healthy' if ping_result else 'unhealthy',
                'ping': ping_result,
                'set_operation': set_result,
                'get_operation': get_result is not None,
                'delete_operation': delete_result,
                'timestamp': datetime.now().isoformat()
            }
            
            if not all([ping_result, set_result, get_result is not None, delete_result]):
                health_status['status'] = 'degraded'
                health_status['warnings'] = []
                
                if not ping_result:
                    health_status['warnings'].append('Ping failed')
                if not set_result:
                    health_status['warnings'].append('Set operation failed')
                if get_result is None:
                    health_status['warnings'].append('Get operation failed')
                if not delete_result:
                    health_status['warnings'].append('Delete operation failed')
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global Redis cache instance
_redis_cache = None

def get_redis_cache() -> RedisCache:
    """Get the global Redis cache instance."""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache 