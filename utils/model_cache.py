"""
Model Caching Utility

Provides joblib-based caching for long-running model operations like
LSTM and XGBoost forecasts to improve performance and reduce redundant computations.
"""

import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Caching utility for model operations using joblib.

    Provides intelligent caching for expensive model operations like
    training, prediction, and forecasting.
    """

    def __init__(self, cache_dir: str = ".cache", verbose: int = 0):
        """
        Initialize the model cache.

        Args:
            cache_dir: Directory to store cache files
            verbose: Verbosity level for joblib Memory
        """
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.memory = joblib.Memory(location=cache_dir, verbose=verbose)

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Model cache initialized at: {cache_dir}")

    def _create_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """
        Create a unique cache key for function arguments.

        Args:
            func_name: Name of the function being cached
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            str: Unique cache key
        """
        # Convert args and kwargs to a hashable format
        key_data = {
            "func_name": func_name,
            "args": self._hash_data(args),
            "kwargs": self._hash_data(kwargs),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        }

        # Create hash
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _hash_data(self, data: Any) -> str:
        """
        Create a hash of data for caching.

        Args:
            data: Data to hash

        Returns:
            str: Hash of the data
        """
        if isinstance(data, pd.DataFrame):
            # Hash DataFrame by converting to string representation
            return hashlib.md5(data.to_string().encode()).hexdigest()
        elif isinstance(data, np.ndarray):
            # Hash numpy array
            return hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, (list, tuple)):
            # Hash list/tuple by converting elements
            return hashlib.md5(
                str([self._hash_data(item) for item in data]).encode()
            ).hexdigest()
        elif isinstance(data, dict):
            # Hash dictionary by converting items
            return hashlib.md5(
                str({k: self._hash_data(v) for k, v in data.items()}).encode()
            ).hexdigest()
        else:
            # Hash other types as string
            return hashlib.md5(str(data).encode()).hexdigest()

    def cache_function(
        self, func: Callable, cache_key: Optional[str] = None
    ) -> Callable:
        """
        Cache a function using joblib.

        Args:
            func: Function to cache
            cache_key: Optional custom cache key

        Returns:
            Callable: Cached function
        """
        # For instance methods, ignore 'self' parameter
        import inspect
        sig = inspect.signature(func)
        is_method = 'self' in sig.parameters
        
        if cache_key:
            # Use custom cache key
            if is_method:
                cached_func = self.memory.cache(func, func_id=cache_key, ignore=['self'])
            else:
                cached_func = self.memory.cache(func, func_id=cache_key)
        else:
            # Use default caching
            if is_method:
                cached_func = self.memory.cache(func, ignore=['self'])
            else:
                cached_func = self.memory.cache(func)

        return cached_func

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.memory.clear()
        logger.info("Model cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.

        Returns:
            Dict containing cache information
        """
        try:
            cache_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in os.listdir(self.cache_dir)
                if os.path.isfile(os.path.join(self.cache_dir, f))
            )

            cache_files = len(
                [
                    f
                    for f in os.listdir(self.cache_dir)
                    if os.path.isfile(os.path.join(self.cache_dir, f))
                ]
            )

            return {
                "cache_dir": self.cache_dir,
                "cache_size_bytes": cache_size,
                "cache_size_mb": cache_size / (1024 * 1024),
                "cache_files": cache_files,
                "cache_available": True,
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {
                "cache_dir": self.cache_dir,
                "cache_available": False,
                "error": str(e),
            }


# Global cache instance
_model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """
    Get the global model cache instance.

    Returns:
        ModelCache: Global cache instance
    """
    return _model_cache


def cache_model_operation(func: Callable) -> Callable:
    """
    Decorator to cache model operations.

    Args:
        func: Function to cache

    Returns:
        Callable: Cached function
    """
    return _model_cache.cache_function(func)


def clear_model_cache() -> None:
    """Clear the global model cache."""
    _model_cache.clear_cache()


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the global model cache.

    Returns:
        Dict containing cache information
    """
    return _model_cache.get_cache_info()


# Convenience functions for common model operations
@cache_model_operation
def cached_lstm_forecast(
    data: pd.DataFrame, model_config: Dict[str, Any], horizon: int = 30, **kwargs
) -> Dict[str, Any]:
    """
    Cached LSTM forecast operation.

    Args:
        data: Input data
        model_config: Model configuration
        horizon: Forecast horizon
        **kwargs: Additional arguments

    Returns:
        Dict containing forecast results
    """
    from trading.models.lstm_model import LSTMForecaster

    model = LSTMForecaster(model_config)
    model.fit(data)
    return model.forecast(data, horizon)


@cache_model_operation
def cached_xgboost_forecast(
    data: pd.DataFrame, model_config: Dict[str, Any], horizon: int = 30, **kwargs
) -> Dict[str, Any]:
    """
    Cached XGBoost forecast operation.

    Args:
        data: Input data
        model_config: Model configuration
        horizon: Forecast horizon
        **kwargs: Additional arguments

    Returns:
        Dict containing forecast results
    """
    from trading.models.xgboost_model import XGBoostModel

    model = XGBoostModel(model_config)
    model.fit(data)
    return model.forecast(data, horizon)


@cache_model_operation
def cached_ensemble_forecast(
    data: pd.DataFrame, model_config: Dict[str, Any], horizon: int = 30, **kwargs
) -> Dict[str, Any]:
    """
    Cached ensemble forecast operation.

    Args:
        data: Input data
        model_config: Model configuration
        horizon: Forecast horizon
        **kwargs: Additional arguments

    Returns:
        Dict containing forecast results
    """
    from trading.models.ensemble_model import EnsembleModel

    model = EnsembleModel(model_config)
    model.fit(data)
    return model.forecast(data, horizon)


@cache_model_operation
def cached_tcn_forecast(
    data: pd.DataFrame, model_config: Dict[str, Any], horizon: int = 30, **kwargs
) -> Dict[str, Any]:
    """
    Cached TCN forecast operation.

    Args:
        data: Input data
        model_config: Model configuration
        horizon: Forecast horizon
        **kwargs: Additional arguments

    Returns:
        Dict containing forecast results
    """
    from trading.models.tcn_model import TCNModel

    model = TCNModel(model_config)
    model.fit(data)
    return model.forecast(data, horizon)


def create_cached_forecast_function(model_type: str) -> Callable:
    """
    Create a cached forecast function for a specific model type.

    Args:
        model_type: Type of model ('lstm', 'xgboost', 'ensemble', 'tcn')

    Returns:
        Callable: Cached forecast function
    """
    model_functions = {
        "lstm": cached_lstm_forecast,
        "xgboost": cached_xgboost_forecast,
        "ensemble": cached_ensemble_forecast,
        "tcn": cached_tcn_forecast,
    }

    if model_type.lower() not in model_functions:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model_functions[model_type.lower()]
