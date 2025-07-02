"""
Technical Indicators Module

Centralized technical indicator calculations for the Evolve trading system.
This module consolidates all indicator functions from across the codebase
to provide a single source of truth for technical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
import warnings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# MOVING AVERAGES
# ============================================================================

def calculate_sma(data: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: Price series
        window: Rolling window size
        min_periods: Minimum periods for calculation
        
    Returns:
        SMA series
    """
    try:
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window:
            logger.warning(f"Insufficient data for SMA calculation: {len(data)} < {window}")
            return pd.Series(index=data.index, dtype=float)
        
        return data.rolling(window=window, min_periods=min_periods).mean()
        
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return pd.Series(index=data.index, dtype=float)

def calculate_ema(data: pd.Series, span: int, adjust: bool = False) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        data: Price series
        span: EMA span
        adjust: Whether to adjust for bias
        
    Returns:
        EMA series
    """
    try:
        if span <= 0:
            raise ValueError("Span must be positive")
        
        if len(data) < span:
            logger.warning(f"Insufficient data for EMA calculation: {len(data)} < {span}")
            return pd.Series(index=data.index, dtype=float)
        
        return data.ewm(span=span, adjust=adjust).mean()
        
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return pd.Series(index=data.index, dtype=float)

def calculate_wma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Weighted Moving Average.
    
    Args:
        data: Price series
        window: Rolling window size
        
    Returns:
        WMA series
    """
    try:
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window:
            logger.warning(f"Insufficient data for WMA calculation: {len(data)} < {window}")
            return pd.Series(index=data.index, dtype=float)
        
        weights = np.arange(1, window + 1)
        return data.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        
    except Exception as e:
        logger.error(f"Error calculating WMA: {e}")
        return pd.Series(index=data.index, dtype=float)

# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: Price series
        window: RSI window (default: 14)
        
    Returns:
        RSI series (0-100)
    """
    try:
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window + 1:
            logger.warning(f"Insufficient data for RSI calculation: {len(data)} < {window + 1}")
            return pd.Series(index=data.index, dtype=float)
        
        # Calculate price changes
        delta = data.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.Series(index=data.index, dtype=float)

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    try:
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError("All periods must be positive")
        
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
        
        if len(data) < slow + signal:
            logger.warning(f"Insufficient data for MACD calculation: {len(data)} < {slow + signal}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series, empty_series
        
        # Calculate EMAs
        fast_ema = calculate_ema(data, fast)
        slow_ema = calculate_ema(data, slow)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = calculate_ema(macd_line, signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series, empty_series

def calculate_stochastic(data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        k_window: %K window
        d_window: %D window
        
    Returns:
        Tuple of (%K, %D)
    """
    try:
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        if k_window <= 0 or d_window <= 0:
            raise ValueError("Windows must be positive")
        
        if len(data) < k_window:
            logger.warning(f"Insufficient data for Stochastic calculation: {len(data)} < {k_window}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series
        
        # Calculate %K
        lowest_low = data['low'].rolling(window=k_window).min()
        highest_high = data['high'].rolling(window=k_window).max()
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (SMA of %K)
        d_percent = calculate_sma(k_percent, d_window)
        
        return k_percent, d_percent
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series

# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price series
        window: Rolling window size
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    try:
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window:
            logger.warning(f"Insufficient data for Bollinger Bands: {len(data)} < {window}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series, empty_series
        
        # Calculate middle band (SMA)
        middle = calculate_sma(data, window)
        
        # Calculate standard deviation
        std = data.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        return upper, middle, lower
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series, empty_series

def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        window: ATR window
        
    Returns:
        ATR series
    """
    try:
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window + 1:
            logger.warning(f"Insufficient data for ATR calculation: {len(data)} < {window + 1}")
            return pd.Series(index=data.index, dtype=float)
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR (EMA of True Range)
        atr = calculate_ema(true_range, window)
        
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series(index=data.index, dtype=float)

# ============================================================================
# VOLUME INDICATORS
# ============================================================================

def calculate_volume_sma(data: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Volume Simple Moving Average.
    
    Args:
        data: Volume series
        window: Rolling window size
        
    Returns:
        Volume SMA series
    """
    try:
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window:
            logger.warning(f"Insufficient data for Volume SMA: {len(data)} < {window}")
            return pd.Series(index=data.index, dtype=float)
        
        return calculate_sma(data, window)
        
    except Exception as e:
        logger.error(f"Error calculating Volume SMA: {e}")
        return pd.Series(index=data.index, dtype=float)

def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        data: DataFrame with 'close', 'volume' columns
        
    Returns:
        OBV series
    """
    try:
        required_cols = ['close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        if len(data) < 2:
            logger.warning("Insufficient data for OBV calculation")
            return pd.Series(index=data.index, dtype=float)
        
        # Calculate price changes
        price_change = data['close'].diff()
        
        # Initialize OBV
        obv = pd.Series(0.0, index=data.index)
        obv.iloc[0] = data['volume'].iloc[0]
        
        # Calculate OBV
        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
        
    except Exception as e:
        logger.error(f"Error calculating OBV: {e}")
        return pd.Series(index=data.index, dtype=float)

# ============================================================================
# TREND INDICATORS
# ============================================================================

def calculate_adx(data: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        window: ADX window
        
    Returns:
        Tuple of (+DI, -DI, ADX)
    """
    try:
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window + 1:
            logger.warning(f"Insufficient data for ADX calculation: {len(data)} < {window + 1}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series, empty_series
        
        # Calculate True Range
        atr = calculate_atr(data, window)
        
        # Calculate directional movement
        high_diff = data['high'].diff()
        low_diff = data['low'].diff()
        
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        # +DM
        plus_dm.loc[(high_diff > 0) & (high_diff > -low_diff)] = high_diff
        
        # -DM
        minus_dm.loc[(-low_diff > 0) & (-low_diff > high_diff)] = -low_diff
        
        # Calculate +DI and -DI
        plus_di = 100 * calculate_ema(plus_dm, window) / atr
        minus_di = 100 * calculate_ema(minus_dm, window) / atr
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = calculate_ema(dx, window)
        
        return plus_di, minus_di, adx
        
    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series, empty_series

def calculate_cci(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        window: CCI window
        
    Returns:
        CCI series
    """
    try:
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window:
            logger.warning(f"Insufficient data for CCI calculation: {len(data)} < {window}")
            return pd.Series(index=data.index, dtype=float)
        
        # Calculate Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate SMA of Typical Price
        sma_tp = calculate_sma(typical_price, window)
        
        # Calculate Mean Deviation
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        # Calculate CCI
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
        
    except Exception as e:
        logger.error(f"Error calculating CCI: {e}")
        return pd.Series(index=data.index, dtype=float)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_bullish(data: pd.DataFrame, strategy: str = 'sma', **kwargs) -> pd.Series:
    """
    Determine if market is bullish based on technical indicators.
    
    Args:
        data: DataFrame with price data
        strategy: Strategy to use ('sma', 'macd', 'rsi', 'bollinger')
        **kwargs: Strategy-specific parameters
        
    Returns:
        Boolean series indicating bullish periods
    """
    try:
        if strategy == 'sma':
            short_window = kwargs.get('short_window', 20)
            long_window = kwargs.get('long_window', 50)
            
            short_sma = calculate_sma(data['close'], short_window)
            long_sma = calculate_sma(data['close'], long_window)
            
            return short_sma > long_sma
            
        elif strategy == 'macd':
            macd_line, signal_line, _ = calculate_macd(data['close'])
            return macd_line > signal_line
            
        elif strategy == 'rsi':
            rsi = calculate_rsi(data['close'])
            return rsi > 50
            
        elif strategy == 'bollinger':
            upper, middle, lower = calculate_bollinger_bands(data['close'])
            return data['close'] > middle
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    except Exception as e:
        logger.error(f"Error determining bullish signal: {e}")
        return pd.Series(False, index=data.index)

def get_support_resistance(data: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate support and resistance levels.
    
    Args:
        data: DataFrame with 'high', 'low' columns
        window: Rolling window size
        
    Returns:
        Tuple of (Support, Resistance)
    """
    try:
        required_cols = ['high', 'low']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if len(data) < window:
            logger.warning(f"Insufficient data for support/resistance: {len(data)} < {window}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series
        
        # Calculate rolling min/max
        support = data['low'].rolling(window=window).min()
        resistance = data['high'].rolling(window=window).max()
        
        return support, resistance
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series

def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate all technical indicators for a dataset.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary of all indicators
    """
    try:
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = calculate_sma(data['close'], 20)
        indicators['sma_50'] = calculate_sma(data['close'], 50)
        indicators['ema_12'] = calculate_ema(data['close'], 12)
        indicators['ema_26'] = calculate_ema(data['close'], 26)
        
        # Momentum indicators
        indicators['rsi'] = calculate_rsi(data['close'])
        macd_line, signal_line, histogram = calculate_macd(data['close'])
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram
        
        # Volatility indicators
        upper, middle, lower = calculate_bollinger_bands(data['close'])
        indicators['bb_upper'] = upper
        indicators['bb_middle'] = middle
        indicators['bb_lower'] = lower
        
        indicators['atr'] = calculate_atr(data)
        
        # Volume indicators
        if 'volume' in data.columns:
            indicators['volume_sma'] = calculate_volume_sma(data['volume'])
            indicators['obv'] = calculate_obv(data)
        
        # Trend indicators
        plus_di, minus_di, adx = calculate_adx(data)
        indicators['plus_di'] = plus_di
        indicators['minus_di'] = minus_di
        indicators['adx'] = adx
        
        indicators['cci'] = calculate_cci(data)
        
        # Support/Resistance
        support, resistance = get_support_resistance(data)
        indicators['support'] = support
        indicators['resistance'] = resistance
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating all indicators: {e}")
        return {}

def get_indicator_signals(data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """
    Generate trading signals from technical indicators.
    
    Args:
        data: DataFrame with price data
        indicators: Dictionary of calculated indicators
        
    Returns:
        Dictionary of trading signals
    """
    try:
        signals = {}
        
        # RSI signals
        if 'rsi' in indicators:
            signals['rsi_oversold'] = indicators['rsi'] < 30
            signals['rsi_overbought'] = indicators['rsi'] > 70
        
        # MACD signals
        if 'macd' in indicators and 'macd_signal' in indicators:
            signals['macd_bullish'] = indicators['macd'] > indicators['macd_signal']
            signals['macd_bearish'] = indicators['macd'] < indicators['macd_signal']
        
        # Bollinger Bands signals
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            signals['bb_upper_breakout'] = data['close'] > indicators['bb_upper']
            signals['bb_lower_breakout'] = data['close'] < indicators['bb_lower']
        
        # Moving average signals
        if 'sma_20' in indicators and 'sma_50' in indicators:
            signals['sma_bullish'] = indicators['sma_20'] > indicators['sma_50']
            signals['sma_bearish'] = indicators['sma_20'] < indicators['sma_50']
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating indicator signals: {e}")
        return {} 