"""
Safe technical indicator calculations with division-by-zero protection.

This module provides utilities to calculate common technical indicators
safely, preventing the widespread RSI division bug that appears in 11+ files.
"""

import numpy as np
import pandas as pd
from typing import Union

def safe_rsi(
    prices: Union[pd.Series, pd.DataFrame],
    period: int = 14,
    price_column: str = None,
    epsilon: float = 1e-10
) -> pd.Series:
    """
    Calculate RSI with division-by-zero protection.
    
    This function replaces the buggy pattern:
        rs = gain / loss  # BUG: loss can be zero!
        rsi = 100 - (100 / (1 + rs))
    
    Args:
        prices: Price series or DataFrame
        period: RSI period (default 14)
        price_column: Column name if DataFrame provided
        epsilon: Minimum denominator value
        
    Returns:
        RSI series (0-100)
        
    Examples:
        >>> rsi = safe_rsi(data["close"])
        >>> rsi = safe_rsi(data, price_column="Close")
    """
    # Handle DataFrame input
    if isinstance(prices, pd.DataFrame):
        if price_column is None:
            raise ValueError("price_column required when passing DataFrame")
        price_series = prices[price_column]
    else:
        price_series = prices
    
    # Calculate gains and losses
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Safe division: rs = gain / loss
    rs = np.where(loss > epsilon, gain / loss, 0.0)
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=price_series.index, name="RSI")


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
    bandwidth = np.where(
        middle_band > epsilon,
        (upper_band - lower_band) / middle_band,
        0.0
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
    stoch_k = np.where(
        price_range > epsilon,
        100 * ((close - low_min) / price_range),
        50.0  # Neutral value
    )
    
    stoch_k = pd.Series(stoch_k, index=close.index, name="Stoch_K")
    stoch_d = stoch_k.rolling(window=d_period).mean()
    stoch_d.name = "Stoch_D"
    
    return pd.DataFrame({"Stoch_K": stoch_k, "Stoch_D": stoch_d})

