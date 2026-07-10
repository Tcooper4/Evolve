"""
Safe technical indicator calculations with division-by-zero protection.

This module provides utilities to calculate common technical indicators
safely, preventing the widespread RSI division bug that appears in 11+ files.
"""

import numpy as np
import pandas as pd
from typing import Union
from trading.utils.safe_math import safe_divide

def safe_rsi(
    prices: Union[pd.Series, pd.DataFrame],
    period: int = 14,
    price_column: str = None,
    epsilon: float = 1e-10
) -> pd.Series:
    """
    Calculate RSI with division-by-zero protection.

    BUG FIX: this previously implemented its own independent RSI formula
    using a simple moving average for gain/loss, rather than Wilder's
    smoothing (the standard method, and the one used by every other RSI
    calculation in this codebase - rsi_strategy.py, rsi_utils.py,
    adaptive_selector.py, market_scanner.py). Verified concretely: on
    identical price data, this produced a mean absolute difference of
    ~12 RSI points (max 50 points) versus the authoritative
    trading.utils.safe_math.safe_rsi - a severe, user-visible
    inconsistency, since this function is the one actually used by the
    live watchlist widget (components/watchlist_widget.py via
    pages/7_Settings.py). Separately, this also crashed outright on
    every call due to a bug in safe_divide's handling of a scalar
    numerator with an array denominator (fixed in safe_math.py directly,
    since other callers could hit the same crash).

    Now delegates the actual RSI calculation to the authoritative
    Wilder's-smoothing implementation, preserving only the DataFrame/
    price_column convenience this function offered that the other one
    doesn't.

    Args:
        prices: Price series or DataFrame
        period: RSI period (default 14)
        price_column: Column name if DataFrame provided
        epsilon: Unused (kept for backward-compatible signature); the
            underlying Wilder's implementation handles the zero-loss
            edge case directly.

    Returns:
        RSI series (0-100)

    Examples:
        >>> rsi = safe_rsi(data["close"])
        >>> rsi = safe_rsi(data, price_column="Close")
    """
    from trading.utils.safe_math import safe_rsi as _wilder_rsi

    # Handle DataFrame input
    if isinstance(prices, pd.DataFrame):
        if price_column is None:
            raise ValueError("price_column required when passing DataFrame")
        price_series = prices[price_column]
    else:
        price_series = prices

    result = _wilder_rsi(price_series, period=period)
    return pd.Series(result, index=price_series.index, name="RSI")


def safe_bollinger_bandwidth(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    epsilon: float = 1e-10
) -> pd.Series:
    """
    Calculate Bollinger Band bandwidth safely.
    
    Args:
        prices: Price series
        window: Rolling window
        num_std: Number of standard deviations
        epsilon: Minimum denominator
        
    Returns:
        Bandwidth series
    """
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # Safe division
    bandwidth = safe_divide(
        upper_band - lower_band,
        middle_band,
        default=0.0,
        epsilon=epsilon
    )
    
    return pd.Series(bandwidth, index=prices.index, name="BB_Bandwidth")


def safe_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    epsilon: float = 1e-10
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator safely.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D smoothing period
        epsilon: Minimum denominator
        
    Returns:
        DataFrame with Stoch_K and Stoch_D
    """
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    
    # Safe division
    price_range = high_max - low_min
    stoch_k_raw = safe_divide(
        close - low_min,
        price_range,
        default=0.5,  # Neutral value when range is zero
        epsilon=epsilon
    )
    stoch_k = 100 * stoch_k_raw
    
    stoch_k = pd.Series(stoch_k, index=close.index, name="Stoch_K")
    stoch_d = stoch_k.rolling(window=d_period).mean()
    stoch_d.name = "Stoch_D"
    
    return pd.DataFrame({"Stoch_K": stoch_k, "Stoch_D": stoch_d})

