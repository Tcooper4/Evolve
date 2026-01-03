"""Common custom indicator functions for :mod:`feature_engineer`.

This module provides a collection of helper functions that can be
registered with :class:`FeatureEngineer` for additional feature
calculations. Custom indicators are kept here to keep
``feature_engineer.py`` focused on orchestration.
"""

import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global registry for indicators
INDICATOR_REGISTRY: Dict[str, Callable] = {}


def register_indicator(name: Optional[str] = None):
    """Decorator to register indicator functions in the global registry.

    Args:
        name: Optional name for the indicator. If not provided, uses function name.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[pd.Series, pd.DataFrame]:
            try:
                result = func(*args, **kwargs)
                return {
                    "success": True,
                    "result": {
                        "success": True,
                        "result": {
                            "success": True,
                            "result": result,
                            "message": "Operation completed successfully",
                            "timestamp": datetime.now().isoformat(),
                        },
                        "message": "Operation completed successfully",
                        "timestamp": datetime.now().isoformat(),
                    },
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"Error in indicator {func.__name__}: {str(e)}")
                raise

        # Register the function
        indicator_name = name or func.__name__.upper()
        INDICATOR_REGISTRY[indicator_name] = wrapper
        return wrapper

    return decorator


def _check_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Check if required columns are present in DataFrame.

    Args:
        df: Input DataFrame
        required: List of required column names

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


@register_indicator()
def rolling_zscore(
    series: pd.Series, window: int = 20, fillna: bool = True
) -> pd.Series:
    """Calculate a rolling z-score for a series.

    Args:
        series: Input series
        window: Rolling window size
        fillna: Whether to fill NaN values

    Returns:
        Series with rolling z-scores
    """
    from trading.utils.safe_math import safe_divide
    
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    result = safe_divide(series - mean, std, default=0.0)
    return result.ffill() if fillna else result


@register_indicator()
def price_ratios(df: pd.DataFrame, fillna: bool = True) -> pd.DataFrame:
    """Compute high/low and close/open ratios.

    Args:
        df: Input DataFrame with OHLC data
        fillna: Whether to fill NaN values

    Returns:
        DataFrame with price ratios
    """
    _check_required_columns(df, ["high", "low", "close", "open"])
    out = pd.DataFrame(index=df.index)
    out["HL_RATIO"] = df["high"] / df["low"]
    out["CO_RATIO"] = df["close"] / df["open"]
    return out.ffill() if fillna else out


@register_indicator()
def rsi(df: pd.DataFrame, window: int = 14, fillna: bool = True) -> pd.Series:
    """Calculate Relative Strength Index using safe division.

    Args:
        df: Input DataFrame with OHLC data
        window: RSI window size
        fillna: Whether to fill NaN values

    Returns:
        Series with RSI values
    """
    from trading.utils.safe_math import safe_rsi
    
    _check_required_columns(df, ["close"])
    result = safe_rsi(df["close"], period=window)
    return result.ffill() if fillna else result


@register_indicator()
def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    fillna: bool = True,
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        df: Input DataFrame with OHLC data
        fast: Fast period
        slow: Slow period
        signal: Signal period
        fillna: Whether to fill NaN values

    Returns:
        DataFrame with MACD, Signal, and Histogram
    """
    _check_required_columns(df, ["close"])
    exp1 = df["close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["close"].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line

    result = pd.DataFrame({"MACD": macd, "SIGNAL": signal_line, "HIST": hist})
    return result.ffill() if fillna else result


@register_indicator()
def atr(df: pd.DataFrame, window: int = 14, fillna: bool = True) -> pd.Series:
    """Calculate Average True Range.

    Args:
        df: Input DataFrame with OHLC data
        window: ATR window size
        fillna: Whether to fill NaN values

    Returns:
        Series with ATR values
    """
    _check_required_columns(df, ["high", "low", "close"])
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result = tr.rolling(window=window).mean()
    return result.ffill() if fillna else result


@register_indicator()
def bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0, fillna: bool = True
) -> pd.DataFrame:
    """Calculate Bollinger Bands.

    Args:
        df: Input DataFrame with OHLC data
        window: Moving average window
        num_std: Number of standard deviations
        fillna: Whether to fill NaN values

    Returns:
        DataFrame with upper, middle, and lower bands
    """
    _check_required_columns(df, ["close"])
    middle = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    result = pd.DataFrame({"BB_UPPER": upper, "BB_MIDDLE": middle, "BB_LOWER": lower})
    return result.ffill() if fillna else result


@register_indicator()
def sma(df: pd.DataFrame, window: int = 20, fillna: bool = True) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        df: Input DataFrame with OHLC data
        window: SMA window size
        fillna: Whether to fill NaN values

    Returns:
        Series with SMA values
    """
    _check_required_columns(df, ["close"])
    result = df["close"].rolling(window=window).mean()
    return result.ffill() if fillna else result


@register_indicator()
def ema(df: pd.DataFrame, window: int = 20, fillna: bool = True) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        df: Input DataFrame with OHLC data
        window: EMA window size
        fillna: Whether to fill NaN values

    Returns:
        Series with EMA values
    """
    _check_required_columns(df, ["close"])
    result = df["close"].ewm(span=window, adjust=False).mean()
    return result.ffill() if fillna else result


def get_indicator_descriptions() -> Dict[str, str]:
    """Get descriptions for all registered indicators.

    Returns:
        Dictionary mapping indicator names to descriptions
    """
    return {
        "ROLLING_ZSCORE": "Rolling z-score normalization of price series",
        "PRICE_RATIOS": "High/Low and Close/Open price ratios",
        "RSI": "Relative Strength Index - momentum oscillator",
        "MACD": "Moving Average Convergence Divergence - trend following momentum indicator",
        "ATR": "Average True Range - volatility indicator",
        "BOLLINGER_BANDS": "Bollinger Bands - volatility indicator with upper, middle, and lower bands",
        "SMA": "Simple Moving Average - basic trend indicator",
        "EMA": "Exponential Moving Average - weighted trend indicator",
    }
