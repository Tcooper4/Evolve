"""
Feature engineering utilities for trading system.
Provides common functions for data preprocessing and feature creation.
"""

import logging
from typing import List, Union

import numpy as np
import pandas as pd
from trading.utils.safe_math import safe_normalize

logger = logging.getLogger(__name__)


def normalize_features(
    data: Union[pd.DataFrame, np.ndarray], method: str = "zscore"
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Normalize features using specified method.

    Args:
        data: Input data to normalize
        method: Normalization method ('zscore', 'minmax', 'robust')

    Returns:
        Normalized data
    """
    try:
        # Use safe_normalize to handle division by zero
        if isinstance(data, pd.DataFrame):
            # Normalize each column
            result = data.copy()
            for col in result.columns:
                if result[col].dtype in [np.number]:
                    result[col] = safe_normalize(result[col], method=method)
            return result
        else:
            # For Series or array
            return safe_normalize(data, method=method)
    except Exception as e:
        logger.error(f"Error normalizing features: {e}")
        return data


def create_lag_features(
    data: pd.DataFrame, columns: List[str], lags: List[int]
) -> pd.DataFrame:
    """
    Create lag features for specified columns.

    Args:
        data: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods

    Returns:
        DataFrame with lag features added
    """
    try:
        result = data.copy()
        for col in columns:
            for lag in lags:
                result[f"{col}_lag_{lag}"] = data[col].shift(lag)
        logger.debug(
            f"Created lag features for {len(columns)} columns with {len(lags)} lags"
        )
        return result
    except Exception as e:
        logger.error(f"Error creating lag features: {e}")
        return data


def remove_outliers(
    data: pd.DataFrame, columns: List[str], method: str = "iqr", threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from specified columns.

    Args:
        data: Input DataFrame
        columns: Columns to process
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outliers removed
    """
    try:
        result = data.copy()
        for col in columns:
            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (data[col] >= lower_bound) & (data[col] <= upper_bound)
            elif method == "zscore":
                from trading.utils.safe_math import safe_divide
                mean = data[col].mean()
                std = data[col].std()
                z_scores = np.abs(safe_divide(data[col] - mean, std, default=0.0))
                mask = z_scores < threshold
            else:
                logger.warning(f"Unknown outlier method: {method}. Using IQR.")
                continue

            result = result[mask]

        logger.debug(f"Removed outliers from {len(columns)} columns using {method}")
        return result
    except Exception as e:
        logger.error(f"Error removing outliers: {e}")
        return data


def calculate_rolling_features(
    data: pd.DataFrame, columns: List[str], windows: List[int], functions: List[str]
) -> pd.DataFrame:
    """
    Calculate rolling window features.

    Args:
        data: Input DataFrame
        columns: Columns to calculate features for
        windows: List of window sizes
        functions: List of functions to apply ('mean', 'std', 'min', 'max')

    Returns:
        DataFrame with rolling features added
    """
    try:
        result = data.copy()
        for col in columns:
            for window in windows:
                for func in functions:
                    if func == "mean":
                        result[f"{col}_rolling_mean_{window}"] = (
                            data[col].rolling(window).mean()
                        )
                    elif func == "std":
                        result[f"{col}_rolling_std_{window}"] = (
                            data[col].rolling(window).std()
                        )
                    elif func == "min":
                        result[f"{col}_rolling_min_{window}"] = (
                            data[col].rolling(window).min()
                        )
                    elif func == "max":
                        result[f"{col}_rolling_max_{window}"] = (
                            data[col].rolling(window).max()
                        )

        logger.debug(f"Created rolling features for {len(columns)} columns")
        return result
    except Exception as e:
        logger.error(f"Error calculating rolling features: {e}")
        return data
