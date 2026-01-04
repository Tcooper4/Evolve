"""
Technical indicators for trading analysis.
Replacement for the removed core.utils.technical_indicators module.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average over a specified period.

    Args:
        data (pd.Series): Price series.
        window (int): Period for SMA calculation.

    Returns:
        pd.Series: Simple Moving Average values.
    """
    try:
        return data.rolling(window=window).mean()
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return pd.Series(dtype=float)


def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average over a specified period.

    Args:
        data (pd.Series): Price series.
        window (int): Period for EMA calculation.

    Returns:
        pd.Series: Exponential Moving Average values.
    """
    try:
        return data.ewm(span=window).mean()
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return pd.Series(dtype=float)


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing (CORRECT METHOD).
    
    This is a wrapper around safe_rsi() to ensure consistent calculation
    across the entire codebase.
    
    Args:
        data (pd.Series): Price series.
        window (int): Period for RSI calculation.

    Returns:
        pd.Series: RSI values using Wilder's smoothing.
    """
    try:
        # Import the corrected safe_rsi function
        from trading.utils.safe_math import safe_rsi
        
        logger.info(f"Calculating RSI with period={window} using Wilder's smoothing")
        return safe_rsi(data, window)
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        # Return neutral RSI values on error
        return pd.Series(50.0, index=data.index)


def calculate_macd(
    data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    try:
        ema_fast = calculate_ema(data, fast)
        ema_slow = calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series, empty_series


def calculate_bollinger_bands(
    data: pd.Series, window: int = 20, num_std: float = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    try:
        sma = calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series, empty_series


def calculate_stochastic(
    data: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator with safe division.
    
    Handles edge case where high == low (no price range) by returning
    neutral 50% value instead of crashing.
    
    Args:
        data (pd.DataFrame): DataFrame with High, Low, Close columns
        k_window (int): Period for %K calculation
        d_window (int): Period for %D smoothing
        
    Returns:
        Tuple[pd.Series, pd.Series]: %K and %D values
    """
    try:
        from trading.utils.safe_math import safe_divide
        
        low_min = data["Low"].rolling(window=k_window).min()
        high_max = data["High"].rolling(window=k_window).max()
        
        # Safe division to handle case where high == low (no range)
        # When range is zero, return 50 (neutral value)
        k_percent = 100 * safe_divide(
            data["Close"] - low_min,
            high_max - low_min,
            default=0.5  # 50% when divided by 100
        )
        
        d_percent = calculate_sma(k_percent, d_window)
        
        return k_percent, d_percent
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range using Wilder's smoothing (CORRECT METHOD).
    
    ATR was invented by J. Welles Wilder (same as RSI) and uses the same
    Wilder's smoothing formula, NOT simple moving average.
    
    Formula:
    - First ATR = SMA of True Range over period
    - Subsequent ATR = (Previous ATR × (period-1) + Current TR) / period
    
    This is mathematically equivalent to EWM with alpha=1/period
    
    Args:
        data (pd.DataFrame): DataFrame with High, Low, Close columns
        window (int): ATR period (default: 14)
        
    Returns:
        pd.Series: ATR values using Wilder's smoothing
    """
    try:
        # Calculate True Range components
        high_low = data["High"] - data["Low"]
        high_close = np.abs(data["High"] - data["Close"].shift())
        low_close = np.abs(data["Low"] - data["Close"].shift())
        
        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Use Wilder's smoothing (equivalent to EWM with alpha=1/window)
        # This is the CORRECT formula for ATR
        atr = true_range.ewm(alpha=1/window, adjust=False).mean()
        
        logger.debug(f"Calculated ATR with period={window} using Wilder's smoothing")
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series(dtype=float)


def calculate_volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Volume Simple Moving Average."""
    try:
        return calculate_sma(volume, window)
    except Exception as e:
        logger.error(f"Error calculating Volume SMA: {e}")
        return pd.Series(dtype=float)


def calculate_price_momentum(data: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Price Momentum with safe division.
    
    Handles edge case where historical price is zero to prevent crashes.
    
    Formula: (Current Price / Price N periods ago) - 1
    
    Args:
        data (pd.Series): Price series
        period (int): Lookback period
        
    Returns:
        pd.Series: Momentum values (fractional change)
    """
    try:
        from trading.utils.safe_math import safe_divide
        
        shifted = data.shift(period)
        
        # Safe division to handle zero prices
        momentum = safe_divide(data, shifted, default=0.0) - 1
        
        return momentum
        
    except Exception as e:
        logger.error(f"Error calculating Price Momentum: {e}")
        return pd.Series(dtype=float)


def calculate_volatility(data: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Rolling Volatility."""
    try:
        returns = data.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)
    except Exception as e:
        logger.error(f"Error calculating Volatility: {e}")
        return pd.Series(dtype=float)


def calculate_beta(
    asset_returns: pd.Series, market_returns: pd.Series, window: int = 252
) -> pd.Series:
    """Calculate Rolling Beta."""
    try:
        covariance = asset_returns.rolling(window=window).cov(market_returns)
        market_variance = market_returns.rolling(window=window).var()
        beta = covariance / market_variance
        return beta
    except Exception as e:
        logger.error(f"Error calculating Beta: {e}")
        return pd.Series(dtype=float)


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, window: int = 252
) -> pd.Series:
    """Calculate Rolling Sharpe Ratio."""
    try:
        excess_returns = returns - risk_free_rate / 252
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()
        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        return sharpe
    except Exception as e:
        logger.error(f"Error calculating Sharpe Ratio: {e}")
        return pd.Series(dtype=float)


def calculate_max_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Calculate Rolling Maximum Drawdown."""
    try:
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown
    except Exception as e:
        logger.error(f"Error calculating Max Drawdown: {e}")
        return pd.Series(dtype=float)


def calculate_support_resistance(
    data: pd.Series, window: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Support and Resistance levels."""
    try:
        resistance = data.rolling(window=window).max()
        support = data.rolling(window=window).min()
        return support, resistance
    except Exception as e:
        logger.error(f"Error calculating Support/Resistance: {e}")
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels."""
    try:
        diff = high - low
        levels = {
            "0.0": low,
            "0.236": low + 0.236 * diff,
            "0.382": low + 0.382 * diff,
            "0.5": low + 0.5 * diff,
            "0.618": low + 0.618 * diff,
            "0.786": low + 0.786 * diff,
            "1.0": high,
        }
        return levels
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return {}


def calculate_pivot_points(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate Pivot Points."""
    try:
        high = data["High"].iloc[-1]
        low = data["Low"].iloc[-1]
        close = data["Close"].iloc[-1]

        pivot = (high + low + close) / 3

        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        return {
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3,
        }
    except Exception as e:
        logger.error(f"Error calculating Pivot Points: {e}")
        return {}


def calculate_ichimoku(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate Ichimoku Cloud indicators."""
    try:
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2

        # Kijun-sen (Base Line)
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        senkou_b = (
            (high.rolling(window=52).max() + low.rolling(window=52).min()) / 2
        ).shift(26)

        # Chikou Span (Lagging Span)
        chikou = close.shift(-26)

        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou,
        }
    except Exception as e:
        logger.error(f"Error calculating Ichimoku: {e}")
        empty_series = pd.Series(dtype=float)
        return {
            "tenkan": empty_series,
            "kijun": empty_series,
            "senkou_a": empty_series,
            "senkou_b": empty_series,
            "chikou": empty_series,
        }
