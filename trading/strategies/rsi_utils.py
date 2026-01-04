"""
Shared RSI utilities for consolidating RSI logic across strategy files.

This module provides common RSI calculation and signal generation functions
to prevent code duplication between rsi_signals.py and rsi_strategy.py.
Enhanced with RSI normalization, edge case handling, and computation caching.
"""

import hashlib
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import pandas_ta, with fallback
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas_ta not available, using fallback RSI calculation")

logger = logging.getLogger(__name__)

# Cache for RSI calculations
_rsi_cache: Dict[str, pd.Series] = {}


def normalize_rsi_value(rsi_value: float) -> float:
    """Normalize RSI value to handle edge cases.

    Args:
        rsi_value: Raw RSI value

    Returns:
        Normalized RSI value between 0 and 100
    """
    if pd.isna(rsi_value) or np.isnan(rsi_value):
        return 50.0  # Neutral value for NaN

    if np.isinf(rsi_value):
        return 100.0 if rsi_value > 0 else 0.0  # Handle infinity

    # Clamp to valid range
    return np.clip(rsi_value, 0.0, 100.0)


def handle_constant_price(prices: pd.Series) -> pd.Series:
    """Handle case where price series is constant.

    Args:
        prices: Price series

    Returns:
        RSI series for constant prices (all 50.0)
    """
    if len(prices) == 0:
        return pd.Series(dtype=float)

    # Check if all prices are the same (within small tolerance)
    price_std = prices.std()
    if price_std < 1e-10:  # Very small standard deviation
        logger.warning("Constant price detected, returning neutral RSI values")
        return pd.Series(50.0, index=prices.index)

    return None  # Not constant, proceed with normal calculation


def create_cache_key(prices: pd.Series, period: int) -> str:
    """Create a cache key for RSI calculation.

    Args:
        prices: Price series
        period: RSI period

    Returns:
        Cache key string
    """
    # Create hash of price data and period
    price_hash = hashlib.md5(prices.values.tobytes()).hexdigest()
    return f"rsi_{period}_{price_hash}_{len(prices)}"


def clear_rsi_cache():
    """Clear the RSI calculation cache."""
    global _rsi_cache
    _rsi_cache.clear()
    logger.info("RSI cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get RSI cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    return {"cache_size": len(_rsi_cache), "cache_keys": list(_rsi_cache.keys())}


def calculate_rsi(
    prices: pd.Series, period: int = 14, use_cache: bool = True
) -> pd.Series:
    """Calculate RSI using the most efficient available method with caching.

    Args:
        prices: Price series (typically close prices)
        period: RSI period (default: 14)
        use_cache: Whether to use caching (default: True)

    Returns:
        RSI values as pandas Series
    """
    try:
        # Validate input
        if prices is None or prices.empty:
            logger.warning("Empty price series provided")
            return pd.Series(dtype=float)

        if period <= 0:
            raise ValueError("Period must be positive")

        if len(prices) < period:
            logger.warning(
                f"Insufficient data: {len(prices)} points, need at least {period}"
            )
            return pd.Series([np.nan] * len(prices), index=prices.index)

        # Check cache if enabled
        if use_cache:
            cache_key = create_cache_key(prices, period)
            if cache_key in _rsi_cache:
                logger.debug(f"RSI cache hit for period {period}")
                return _rsi_cache[cache_key].copy()

        # Handle constant price case
        constant_rsi = handle_constant_price(prices)
        if constant_rsi is not None:
            return constant_rsi

        # Calculate RSI
        # ALWAYS use our corrected implementation for consistency
        # pandas_ta has compatibility issues, so use safe_rsi directly
        from trading.utils.safe_math import safe_rsi
        logger.info("Using corrected Wilder's smoothing RSI implementation")
        rsi = safe_rsi(prices, period)

        # Normalize RSI values
        rsi = rsi.apply(normalize_rsi_value)

        # Cache result if enabled
        if use_cache:
            cache_key = create_cache_key(prices, period)
            _rsi_cache[cache_key] = rsi.copy()
            logger.debug(f"RSI cached for period {period}")

        return rsi

    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        # Return neutral RSI values on error
        return pd.Series(50.0, index=prices.index)


def calculate_rsi_fallback(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing when pandas_ta is not available.
    
    This is now just a wrapper around safe_rsi() which implements the correct formula.

    Args:
        prices: Price series
        period: RSI period

    Returns:
        RSI values as pandas Series
    """
    try:
        # Import the corrected safe_rsi function
        from trading.utils.safe_math import safe_rsi
        
        logger.info("Using corrected Wilder's smoothing for RSI calculation")
        return safe_rsi(prices, period)
        
    except Exception as e:
        logger.error(f"Error in RSI fallback calculation: {e}")
        return pd.Series(50.0, index=prices.index)


def generate_rsi_signals_core(
    df: pd.DataFrame,
    period: int = 14,
    buy_threshold: float = 30,
    sell_threshold: float = 70,
    tolerance: float = 0.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Core RSI signal generation logic with enhanced edge case handling.

    Args:
        df: Price data DataFrame with 'Close' column
        period: RSI period
        buy_threshold: RSI level for buy signals
        sell_threshold: RSI level for sell signals
        tolerance: Tolerance buffer to prevent repeated signals
        use_cache: Whether to use RSI caching

    Returns:
        DataFrame with RSI signals and indicators
    """
    # Validate input
    if "Close" not in df.columns:
        raise ValueError("Missing 'Close' column in DataFrame")

    if buy_threshold >= sell_threshold:
        raise ValueError("buy_threshold must be less than sell_threshold")

    # Create copy to avoid modifying original
    df = df.copy()

    # Calculate RSI with caching
    df["rsi"] = calculate_rsi(df["Close"], period, use_cache=use_cache)

    # Check for valid RSI values
    valid_rsi_mask = df["rsi"].notna() & (df["rsi"] >= 0) & (df["rsi"] <= 100)
    if not valid_rsi_mask.any():
        logger.warning("No valid RSI values generated")
        df["signal"] = 0
        df["returns"] = 0
        df["strategy_returns"] = 0
        df["cumulative_returns"] = 1
        df["strategy_cumulative_returns"] = 1
        return df

    # Generate signals with tolerance
    df["signal"] = 0

    # Buy signal when RSI crosses below buy threshold with tolerance
    buy_condition = (df["rsi"] < buy_threshold) & (
        df["rsi"].shift(1) >= buy_threshold + tolerance
    )
    df.loc[buy_condition, "signal"] = 1

    # Sell signal when RSI crosses above sell threshold with tolerance
    sell_condition = (df["rsi"] > sell_threshold) & (
        df["rsi"].shift(1) <= sell_threshold - tolerance
    )
    df.loc[sell_condition, "signal"] = -1

    # Calculate returns with error handling
    try:
        df["returns"] = df["Close"].pct_change()
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]

        # Calculate cumulative returns
        df["cumulative_returns"] = (1 + df["returns"]).cumprod()
        df["strategy_cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        df["returns"] = 0
        df["strategy_returns"] = 0
        df["cumulative_returns"] = 1
        df["strategy_cumulative_returns"] = 1

    return df


def validate_rsi_parameters(
    period: int, buy_threshold: float, sell_threshold: float
) -> Tuple[bool, Optional[str]]:
    """Validate RSI parameters.

    Args:
        period: RSI period
        buy_threshold: Buy threshold
        sell_threshold: Sell threshold

    Returns:
        Tuple of (is_valid, error_message)
    """
    if period <= 0:
        return False, "Period must be positive"

    if buy_threshold <= 0 or buy_threshold >= 100:
        return False, "Buy threshold must be between 0 and 100"

    if sell_threshold <= 0 or sell_threshold >= 100:
        return False, "Sell threshold must be between 0 and 100"

    if buy_threshold >= sell_threshold:
        return False, "Buy threshold must be less than sell threshold"

    return True, None


def get_rsi_parameter_space() -> Dict[str, Tuple[float, float]]:
    """Get standard RSI parameter space for optimization.

    Returns:
        Dictionary of parameter ranges
    """
    return {"period": (5, 30), "buy_threshold": (20, 40), "sell_threshold": (60, 80)}


def get_default_rsi_parameters() -> Dict[str, Any]:
    """Get default RSI parameters.

    Returns:
        Dictionary of default parameters
    """
    return {
        "period": 14,
        "buy_threshold": 30,
        "sell_threshold": 70,
        "tolerance": 0.0,
        "use_cache": True,
    }


def get_rsi_statistics(rsi_series: pd.Series) -> Dict[str, float]:
    """Get statistics about RSI values.

    Args:
        rsi_series: RSI values

    Returns:
        Dictionary with RSI statistics
    """
    if rsi_series is None or rsi_series.empty:
        return {}

    valid_rsi = rsi_series.dropna()
    if len(valid_rsi) == 0:
        return {}

    return {
        "mean": float(valid_rsi.mean()),
        "std": float(valid_rsi.std()),
        "min": float(valid_rsi.min()),
        "max": float(valid_rsi.max()),
        "median": float(valid_rsi.median()),
        "oversold_count": int((valid_rsi < 30).sum()),
        "overbought_count": int((valid_rsi > 70).sum()),
        "neutral_count": int(((valid_rsi >= 30) & (valid_rsi <= 70)).sum()),
    }
