"""
Shared RSI utilities for consolidating RSI logic across strategy files.

This module provides common RSI calculation and signal generation functions
to prevent code duplication between rsi_signals.py and rsi_strategy.py.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import pandas_ta, with fallback
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas_ta not available, using fallback RSI calculation")

logger = logging.getLogger(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using the most efficient available method.
    
    Args:
        prices: Price series (typically close prices)
        period: RSI period (default: 14)
        
    Returns:
        RSI values as pandas Series
    """
    if PANDAS_TA_AVAILABLE:
        return ta.rsi(prices, length=period)
    else:
        return calculate_rsi_fallback(prices, period)

def calculate_rsi_fallback(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using manual implementation when pandas_ta is not available.
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values as pandas Series
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_rsi_signals_core(
    df: pd.DataFrame,
    period: int = 14,
    buy_threshold: float = 30,
    sell_threshold: float = 70,
    tolerance: float = 0.0
) -> pd.DataFrame:
    """Core RSI signal generation logic.
    
    Args:
        df: Price data DataFrame with 'Close' column
        period: RSI period
        buy_threshold: RSI level for buy signals
        sell_threshold: RSI level for sell signals
        tolerance: Tolerance buffer to prevent repeated signals
        
    Returns:
        DataFrame with RSI signals and indicators
    """
    # Validate input
    if 'Close' not in df.columns:
        raise ValueError("Missing 'Close' column in DataFrame")
    
    if buy_threshold >= sell_threshold:
        raise ValueError("buy_threshold must be less than sell_threshold")
    
    # Calculate RSI
    df = df.copy()
    df['rsi'] = calculate_rsi(df['Close'], period)
    
    # Generate signals with tolerance
    df['signal'] = 0
    
    # Buy signal when RSI crosses below buy threshold with tolerance
    buy_condition = (df['rsi'] < buy_threshold) & (df['rsi'].shift(1) >= buy_threshold + tolerance)
    df.loc[buy_condition, 'signal'] = 1
    
    # Sell signal when RSI crosses above sell threshold with tolerance
    sell_condition = (df['rsi'] > sell_threshold) & (df['rsi'].shift(1) <= sell_threshold - tolerance)
    df.loc[sell_condition, 'signal'] = -1
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    # Calculate cumulative returns
    df['cumulative_returns'] = (1 + df['returns']).cumprod()
    df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    
    return df

def validate_rsi_parameters(
    period: int,
    buy_threshold: float,
    sell_threshold: float
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
    return {
        'period': (5, 30),
        'buy_threshold': (20, 40),
        'sell_threshold': (60, 80)
    }

def get_default_rsi_parameters() -> Dict[str, Any]:
    """Get default RSI parameters.
    
    Returns:
        Dictionary of default parameters
    """
    return {
        'period': 14,
        'buy_threshold': 30,
        'sell_threshold': 70,
        'tolerance': 0.0
    } 