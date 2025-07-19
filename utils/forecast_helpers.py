"""
Forecast Helper Utilities

This module provides shared utilities for forecasting operations, including
exception handling decorators and common validation functions.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_forecast(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    log_errors: bool = True,
    return_fallback: bool = True,
    fallback_value: Optional[Any] = None
):
    """
    Decorator to safely handle forecasting operations with retry logic and error handling.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        log_errors: Whether to log errors
        return_fallback: Whether to return fallback value on failure
        fallback_value: Value to return if all retries fail

    Returns:
        Decorated function that handles exceptions gracefully
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[Tuple[np.ndarray, float], Any]:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Validate result format
                    if isinstance(result, tuple) and len(result) >= 2:
                        predictions, confidence = result[0], result[1]

                        # Basic validation
                        if not isinstance(predictions, (np.ndarray, list, pd.Series)):
                            raise ValueError(f"Invalid prediction format: {type(predictions)}")

                        if not isinstance(confidence, (int, float, np.number)):
                            raise ValueError(f"Invalid confidence format: {type(confidence)}")

                        if log_errors:
                            logger.info(f"Forecast successful in {execution_time:.3f}s (attempt {attempt + 1})")

                        return result
                    else:
                        raise ValueError(f"Invalid result format: expected tuple, got {type(result)}")

                except Exception as e:
                    last_exception = e
                    if log_errors:
                        logger.warning(
                            f"Forecast attempt {attempt + 1} failed for {func.__name__}: {str(e)}"
                        )

                    if attempt < max_retries:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        if log_errors:
                            logger.error(
                                f"All forecast attempts failed for {func.__name__}: {str(e)}"
                            )
                        break

            # Return fallback if specified
            if return_fallback:
                if log_errors:
                    logger.warning(f"Returning fallback value for {func.__name__}")
                return fallback_value
            else:
                raise last_exception

        return wrapper
    return decorator


def validate_forecast_input(
    data: Union[np.ndarray, pd.DataFrame, pd.Series],
    min_length: int = 10,
    require_numeric: bool = True
) -> bool:
    """
    Validate input data for forecasting operations.

    Args:
        data: Input data to validate
        min_length: Minimum required length
        require_numeric: Whether data must be numeric

    Returns:
        True if data is valid, False otherwise

    Raises:
        ValueError: If data is invalid
    """
    if data is None:
        raise ValueError("Input data cannot be None")

    if len(data) < min_length:
        raise ValueError(f"Data length {len(data)} is less than minimum {min_length}")

    if require_numeric:
        if isinstance(data, pd.DataFrame):
            if not data.select_dtypes(include=[np.number]).shape[1]:
                raise ValueError("DataFrame must contain numeric columns")
        elif isinstance(data, (pd.Series, np.ndarray)):
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError("Data must be numeric")

    return True


def calculate_forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:
    """
    Calculate common forecast accuracy metrics.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Dictionary containing accuracy metrics
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have same length")

    # Calculate metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }


def normalize_forecast_output(
    predictions: np.ndarray,
    confidence: float,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Normalize forecast output to ensure valid ranges.

    Args:
        predictions: Raw predictions
        confidence: Raw confidence score
        min_confidence: Minimum allowed confidence
        max_confidence: Maximum allowed confidence

    Returns:
        Tuple of (normalized_predictions, normalized_confidence)
    """
    # Ensure predictions are finite
    if not np.all(np.isfinite(predictions)):
        logger.warning("Non-finite values detected in predictions, replacing with NaN")
        predictions = np.where(np.isfinite(predictions), predictions, np.nan)

    # Normalize confidence to valid range
    confidence = np.clip(confidence, min_confidence, max_confidence)

    return predictions, confidence


def log_forecast_performance(
    model_name: str,
    execution_time: float,
    data_length: int,
    confidence: float,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Log forecast performance metrics.

    Args:
        model_name: Name of the forecasting model
        execution_time: Time taken for forecast
        data_length: Length of input data
        confidence: Confidence score
        metrics: Optional accuracy metrics
    """
    logger.info(
        f"Forecast performance - Model: {model_name}, "
        f"Time: {execution_time:.3f}s, "
        f"Data length: {data_length}, "
        f"Confidence: {confidence:.3f}"
    )

    if metrics:
        logger.info(f"Accuracy metrics: {metrics}") 