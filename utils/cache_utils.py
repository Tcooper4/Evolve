"""
Cache Utilities for Model Operations

This module provides centralized caching functionality for model operations:
- Joblib-based caching for long-running model operations
- Cache management and cleanup
- Performance monitoring and statistics
- Consistent caching interface across all models
"""

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CacheConfig:
    """Configuration for caching operations."""
    
    def __init__(
        self,
        cache_dir: str = "cache/model_cache",
        max_size_mb: int = 1024,  # 1GB default
        ttl_hours: int = 24,  # 24 hours default
        compression_level: int = 3,
        enable_cache: bool = True,
        enable_stats: bool = True,
    ):
        """
        Initialize cache configuration.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live in hours
            compression_level: Joblib compression level (0-9)
            enable_cache: Whether caching is enabled
            enable_stats: Whether to collect cache statistics
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.ttl_hours = ttl_hours
        self.compression_level = compression_level
        self.enable_cache = enable_cache
        self.enable_stats = enable_stats
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_mb": 0,
            "last_cleanup": datetime.now(),
        }


class ModelCache:
    """Centralized model cache manager."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize model cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache directory
        self._init_cache_directory()
        
        # Load existing statistics
        self._load_stats()
        
        self.logger.info(f"ModelCache initialized: {self.config.cache_dir}")

    def _init_cache_directory(self):
        """Initialize cache directory structure."""
        # Create main cache directory
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different model types
        subdirs = ["lstm", "xgboost", "prophet", "ensemble", "hybrid", "other"]
        for subdir in subdirs:
            (self.config.cache_dir / subdir).mkdir(exist_ok=True)

    def _load_stats(self):
        """Load cache statistics from file."""
        stats_file = self.config.cache_dir / "cache_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    saved_stats = json.load(f)
                    self.config.stats.update(saved_stats)
            except Exception as e:
                self.logger.warning(f"Failed to load cache stats: {e}")

    def _save_stats(self):
        """Save cache statistics to file."""
        if not self.config.enable_stats:
            return
            
        stats_file = self.config.cache_dir / "cache_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.config.stats, f, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save cache stats: {e}")

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Generate a cache key for function call.
        
        Args:
            func_name: Name of the function
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Cache key string
        """
        # Create a hashable representation of arguments
        def make_hashable(obj):
            if isinstance(obj, (list, tuple)):
                return tuple(make_hashable(item) for item in obj)
            elif isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, pd.DataFrame):
                # Hash DataFrame content
                return f"df_{hashlib.md5(obj.to_string().encode()).hexdigest()[:16]}"
            elif isinstance(obj, np.ndarray):
                # Hash numpy array
                return f"np_{hashlib.md5(obj.tobytes()).hexdigest()[:16]}"
            elif hasattr(obj, '__hash__'):
                return obj
            else:
                return str(obj)
        
        # Create key components
        key_parts = [
            func_name,
            make_hashable(args),
            make_hashable(kwargs),
        ]
        
        # Generate hash
        key_string = json.dumps(key_parts, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, model_type: str = "other") -> Path:
        """
        Get cache file path for a key.
        
        Args:
            cache_key: Cache key
            model_type: Type of model (lstm, xgboost, etc.)
            
        Returns:
            Path to cache file
        """
        return self.config.cache_dir / model_type / f"{cache_key}.joblib"

    def get(self, cache_key: str, model_type: str = "other") -> Optional[Any]:
        """
        Get cached result.
        
        Args:
            cache_key: Cache key
            model_type: Type of model
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self.config.enable_cache:
            return None
            
        cache_path = self._get_cache_path(cache_key, model_type)
        
        if not cache_path.exists():
            self._record_miss()
            return None
        
        # Check if cache is expired
        if self._is_expired(cache_path):
            self._record_miss()
            cache_path.unlink(missing_ok=True)
            return None
        
        try:
            # Load cached result
            result = joblib.load(cache_path)
            self._record_hit()
            self.logger.debug(f"Cache hit: {cache_key}")
            return result
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_key}: {e}")
            self._record_miss()
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, cache_key: str, result: Any, model_type: str = "other") -> bool:
        """
        Cache a result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
            model_type: Type of model
            
        Returns:
            True if successfully cached
        """
        if not self.config.enable_cache:
            return False
            
        try:
            # Check cache size before storing
            if self._should_evict():
                self._evict_oldest()
            
            cache_path = self._get_cache_path(cache_key, model_type)
            
            # Save result with compression
            joblib.dump(
                result,
                cache_path,
                compress=self.config.compression_level,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            
            # Update statistics
            self._update_size_stats()
            
            self.logger.debug(f"Cached result: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache result {cache_key}: {e}")
            return False

    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache file is expired."""
        if not cache_path.exists():
            return True
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age > timedelta(hours=self.config.ttl_hours)

    def _should_evict(self) -> bool:
        """Check if cache eviction is needed."""
        return self.config.stats["total_size_mb"] > self.config.max_size_mb

    def _evict_oldest(self):
        """Evict oldest cache files."""
        cache_files = []
        
        # Collect all cache files with their modification times
        for subdir in self.config.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.joblib"):
                    cache_files.append((cache_file, cache_file.stat().st_mtime))
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove oldest files until under size limit
        for cache_file, _ in cache_files:
            if self.config.stats["total_size_mb"] <= self.config.max_size_mb * 0.8:
                break
                
            try:
                cache_file.unlink()
                self.config.stats["evictions"] += 1
                self.logger.debug(f"Evicted cache file: {cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to evict {cache_file}: {e}")
        
        # Update size statistics
        self._update_size_stats()

    def _update_size_stats(self):
        """Update cache size statistics."""
        total_size = 0
        
        for subdir in self.config.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.joblib"):
                    if cache_file.exists():
                        total_size += cache_file.stat().st_size
        
        self.config.stats["total_size_mb"] = total_size / (1024 * 1024)

    def _record_hit(self):
        """Record a cache hit."""
        self.config.stats["hits"] += 1
        self._save_stats()

    def _record_miss(self):
        """Record a cache miss."""
        self.config.stats["misses"] += 1
        self._save_stats()

    def clear(self, model_type: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            model_type: Specific model type to clear, or None for all
        """
        if model_type:
            cache_dir = self.config.cache_dir / model_type
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.joblib"):
                    cache_file.unlink(missing_ok=True)
                self.logger.info(f"Cleared cache for {model_type}")
        else:
            # Clear all cache
            for subdir in self.config.cache_dir.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.joblib"):
                        cache_file.unlink(missing_ok=True)
            self.logger.info("Cleared all cache")
        
        # Reset statistics
        self.config.stats["total_size_mb"] = 0
        self._save_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.config.stats["hits"] + self.config.stats["misses"]
        hit_rate = self.config.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.config.stats["hits"],
            "misses": self.config.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.config.stats["evictions"],
            "total_size_mb": self.config.stats["total_size_mb"],
            "max_size_mb": self.config.max_size_mb,
            "last_cleanup": self.config.stats["last_cleanup"],
        }

    def cleanup(self):
        """Clean up expired cache files."""
        expired_count = 0
        
        for subdir in self.config.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.joblib"):
                    if self._is_expired(cache_file):
                        cache_file.unlink()
                        expired_count += 1
        
        self.config.stats["last_cleanup"] = datetime.now()
        self._update_size_stats()
        self._save_stats()
        
        self.logger.info(f"Cleanup completed: removed {expired_count} expired files")


# Global cache instance
_model_cache = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


def cache_model_operation(
    model_type: str = "other",
    ttl_hours: Optional[int] = None,
    key_prefix: Optional[str] = None,
):
    """
    Decorator for caching model operations.
    
    Args:
        model_type: Type of model (lstm, xgboost, prophet, etc.)
        ttl_hours: Override TTL for this operation
        key_prefix: Prefix for cache key
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache instance
            cache = get_model_cache()
            
            # Generate cache key
            func_name = key_prefix or func.__name__
            cache_key = cache._generate_cache_key(func_name, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key, model_type)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, model_type)
            
            return result
        
        return wrapper
    return decorator


def cache_forecast_operation(model_type: str = "other"):
    """
    Specialized decorator for forecast operations.
    
    Args:
        model_type: Type of forecasting model
        
    Returns:
        Decorated function
    """
    return cache_model_operation(model_type=model_type, key_prefix="forecast")


def cache_training_operation(model_type: str = "other"):
    """
    Specialized decorator for training operations.
    
    Args:
        model_type: Type of model
        
    Returns:
        Decorated function
    """
    return cache_model_operation(model_type=model_type, key_prefix="train")


def cache_prediction_operation(model_type: str = "other"):
    """
    Specialized decorator for prediction operations.
    
    Args:
        model_type: Type of model
        
    Returns:
        Decorated function
    """
    return cache_model_operation(model_type=model_type, key_prefix="predict")


# Convenience functions for specific model types
def cache_lstm_operation():
    """Cache decorator for LSTM operations."""
    return cache_model_operation(model_type="lstm")


def cache_xgboost_operation():
    """Cache decorator for XGBoost operations."""
    return cache_model_operation(model_type="xgboost")


def cache_prophet_operation():
    """Cache decorator for Prophet operations."""
    return cache_model_operation(model_type="prophet")


def cache_ensemble_operation():
    """Cache decorator for ensemble operations."""
    return cache_model_operation(model_type="ensemble")


def cache_hybrid_operation():
    """Cache decorator for hybrid operations."""
    return cache_model_operation(model_type="hybrid")


# Cache management functions
def clear_model_cache(model_type: Optional[str] = None):
    """Clear model cache."""
    cache = get_model_cache()
    cache.clear(model_type)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache = get_model_cache()
    return cache.get_stats()


def cleanup_cache():
    """Clean up expired cache files."""
    cache = get_model_cache()
    cache.cleanup()


# Example usage:
# @cache_forecast_operation("lstm")
# def forecast_lstm(data, params):
#     # LSTM forecasting logic
#     return forecast_result
#
# @cache_training_operation("xgboost")
# def train_xgboost(data, params):
#     # XGBoost training logic
#     return model 