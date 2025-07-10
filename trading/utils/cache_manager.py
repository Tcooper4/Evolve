"""
Cache management utilities for the trading system.
"""

import functools
import hashlib
import json
import pickle
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Cache management utility class."""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 100):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_metadata = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata."""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.cache_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache metadata: {e}")
                self.cache_metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        metadata_file = self.cache_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save cache metadata: {e}")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        try:
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                return default
            
            # Check if cache is expired
            if key in self.cache_metadata:
                expiry = datetime.fromisoformat(self.cache_metadata[key]['expiry'])
                if datetime.now() > expiry:
                    self.delete(key)
                    return default
            
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            cache_path = self._get_cache_path(key)
            
            # Save value
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            expiry = datetime.now() + timedelta(seconds=ttl_seconds)
            self.cache_metadata[key] = {
                'expiry': expiry.isoformat(),
                'size': cache_path.stat().st_size,
                'created_at': datetime.now().isoformat()
            }
            
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            
            if key in self.cache_metadata:
                del self.cache_metadata[key]
                self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache."""
        try:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            # Clear metadata
            self.cache_metadata.clear()
            self._save_metadata()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            total_files = len(list(self.cache_dir.glob("*.pkl")))
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            
            # Count expired entries
            expired_count = 0
            for key, metadata in self.cache_metadata.items():
                expiry = datetime.fromisoformat(metadata['expiry'])
                if datetime.now() > expiry:
                    expired_count += 1
            
            return {
                'total_files': total_files,
                'total_size_mb': total_size / (1024 * 1024),
                'expired_entries': expired_count,
                'max_size_mb': self.max_size_mb
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

# Global cache manager instance
_cache_manager = CacheManager()

def cache_result(ttl_seconds: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results.
    
    Args:
        ttl_seconds: Time to live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            
            # Add args and kwargs to key
            if args:
                key_parts.append(str(args))
            if kwargs:
                key_parts.append(str(sorted(kwargs.items())))
            
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = _cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache_manager.set(cache_key, result, ttl_seconds)
            
            logger.debug(f"Cache miss for {func.__name__}, stored result")
            return result
        
        return wrapper
    return decorator

def clear_cache() -> bool:
    """Clear all cache."""
    return _cache_manager.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return _cache_manager.get_stats() 