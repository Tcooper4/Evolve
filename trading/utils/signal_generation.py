"""Signal generation utilities for trading strategies."""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    lookback_period: int = 14
    threshold: float = 0.7
    smoothing: int = 3
    min_volume: float = 1000
    max_spread: float = 0.02

def validate_market_data(data: pd.DataFrame) -> None:
    """Validate market data for signal generation.
    
    Args:
        data: DataFrame containing market data
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    required_columns = ["open", "high", "low", "close", "volume"]
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for null values
    if data[required_columns].isnull().any().any():
        raise ValueError("Market data contains null values")
    
    # Check for negative values
    if (data[["open", "high", "low", "close", "volume"]] < 0).any().any():
        raise ValueError("Market data contains negative values")
    
    # Check for high-low consistency
    if not (data["high"] >= data["low"]).all():
        raise ValueError("High prices must be greater than or equal to low prices")

def calculate_technical_indicators(data: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    """Calculate technical indicators for signal generation.
    
    Args:
        data: DataFrame containing market data
        config: Signal generation configuration
        
    Returns:
        DataFrame with added technical indicators
    """
    # Calculate returns
    data["returns"] = data["close"].pct_change()
    
    # Calculate volatility
    data["volatility"] = data["returns"].rolling(window=config.lookback_period).std()
    
    # Calculate moving averages
    data["sma_short"] = data["close"].rolling(window=config.lookback_period).mean()
    data["sma_long"] = data["close"].rolling(window=config.lookback_period * 2).mean()
    
    # Calculate RSI
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config.lookback_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config.lookback_period).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))
    
    # Calculate MACD with span validation
    span1, span2, signal_span = 12, 26, 9
    
    # Validate MACD spans
    if span1 >= span2:
        raise ValueError("MACD span1 must be less than span2.")
    
    exp1 = data["close"].ewm(span=span1, adjust=False).mean()
    exp2 = data["close"].ewm(span=span2, adjust=False).mean()
    data["macd"] = exp1 - exp2
    data["macd_signal"] = data["macd"].ewm(span=signal_span, adjust=False).mean()
    
    # Calculate Bollinger Bands
    data["bb_middle"] = data["close"].rolling(window=config.lookback_period).mean()
    data["bb_std"] = data["close"].rolling(window=config.lookback_period).std()
    data["bb_upper"] = data["bb_middle"] + (data["bb_std"] * 2)
    data["bb_lower"] = data["bb_middle"] - (data["bb_std"] * 2)
    
    return data

def apply_signal_filters(signals: pd.Series, data: pd.DataFrame, config: SignalConfig) -> pd.Series:
    """Apply filters to raw trading signals.
    
    Args:
        signals: Series containing raw trading signals
        data: DataFrame containing market data
        config: Signal generation configuration
        
    Returns:
        Filtered trading signals
    """
    # Volume filter
    volume_filter = data["volume"] >= config.min_volume
    
    # Spread filter
    spread = (data["high"] - data["low"]) / data["low"]
    spread_filter = spread <= config.max_spread
    
    # Apply filters
    filtered_signals = signals.copy()
    filtered_signals[~volume_filter] = 0
    filtered_signals[~spread_filter] = 0
    
    # Apply smoothing if configured
    if config.smoothing > 1:
        filtered_signals = filtered_signals.rolling(window=config.smoothing).mean()
    
    return filtered_signals

def generate_signals(data: pd.DataFrame, config: SignalConfig) -> Dict[str, pd.Series]:
    """Generate trading signals using multiple strategies.
    
    Args:
        data: DataFrame containing market data
        config: Signal generation configuration
        
    Returns:
        Dictionary containing signals for each strategy
    """
    # Validate data
    validate_market_data(data)
    
    # Calculate indicators
    data = calculate_technical_indicators(data, config)
    
    # Initialize signals dictionary
    signals = {}
    
    # RSI strategy
    rsi_signals = pd.Series(0, index=data.index)
    rsi_signals[data["rsi"] < 30] = 1  # Oversold
    rsi_signals[data["rsi"] > 70] = -1  # Overbought
    signals["rsi"] = apply_signal_filters(rsi_signals, data, config)
    
    # MACD strategy
    macd_signals = pd.Series(0, index=data.index)
    macd_signals[data["macd"] > data["macd_signal"]] = 1  # Bullish crossover
    macd_signals[data["macd"] < data["macd_signal"]] = -1  # Bearish crossover
    signals["macd"] = apply_signal_filters(macd_signals, data, config)
    
    # Bollinger Bands strategy
    bb_signals = pd.Series(0, index=data.index)
    bb_signals[data["close"] < data["bb_lower"]] = 1  # Price below lower band
    bb_signals[data["close"] > data["bb_upper"]] = -1  # Price above upper band
    signals["bollinger"] = apply_signal_filters(bb_signals, data, config)
    
    # Moving Average Crossover strategy
    ma_signals = pd.Series(0, index=data.index)
    ma_signals[data["sma_short"] > data["sma_long"]] = 1  # Bullish crossover
    ma_signals[data["sma_short"] < data["sma_long"]] = -1  # Bearish crossover
    signals["ma_crossover"] = apply_signal_filters(ma_signals, data, config)
    
    return signals

def generate_custom_signals(data: pd.DataFrame, rules: List[Dict[str, Any]], config: SignalConfig) -> pd.Series:
    """Generate custom trading signals based on user-defined rules.
    
    Args:
        data: DataFrame containing market data
        rules: List of dictionaries containing signal rules
        config: Signal generation configuration
        
    Returns:
        Series containing custom trading signals
    """
    # Validate data
    validate_market_data(data)
    
    # Calculate indicators
    data = calculate_technical_indicators(data, config)
    
    # Initialize signals
    signals = pd.Series(0, index=data.index)
    
    # Apply each rule
    for rule in rules:
        condition = rule["condition"]
        value = rule["value"]
        signal = rule["signal"]
        
        # Evaluate condition
        if condition == "rsi":
            mask = data["rsi"] < value if signal == 1 else data["rsi"] > value
        elif condition == "macd":
            mask = data["macd"] > value if signal == 1 else data["macd"] < value
        elif condition == "bb":
            mask = data["close"] < data["bb_lower"] if signal == 1 else data["close"] > data["bb_upper"]
        elif condition == "ma":
            mask = data["sma_short"] > data["sma_long"] if signal == 1 else data["sma_short"] < data["sma_long"]
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Apply signal
        signals[mask] = signal
    
    # Apply filters
    return {'success': True, 'result': apply_signal_filters(signals, data, config), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}