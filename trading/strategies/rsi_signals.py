"""RSI strategy signal generator."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_optimized_settings(ticker: str) -> Dict[str, Any]:
    """Load optimized RSI settings for a ticker if available.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary of optimized settings or None if not found
    """
    try:
        settings_file = Path(f"memory/strategy_settings/rsi/{ticker}.json")
        if settings_file.exists():
            with open(settings_file, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading optimized settings for {ticker}: {str(e)}")
    return None

def generate_rsi_signals(
    df: pd.DataFrame,
    ticker: str = None,
    period: int = 14,
    buy_threshold: float = 30,
    sell_threshold: float = 70
) -> pd.DataFrame:
    """Generate RSI trading signals.
    
    Args:
        df: Price data DataFrame with OHLCV columns
        ticker: Optional ticker symbol for loading optimized settings
        period: RSI period
        buy_threshold: RSI level for buy signals
        sell_threshold: RSI level for sell signals
        
    Returns:
        DataFrame with RSI signals and returns
    """
    try:
        # Validate required columns
        if 'Close' not in df.columns:
            raise ValueError("Missing 'Close' column in DataFrame")
        
        # Load optimized settings if available
        if ticker:
            optimized = load_optimized_settings(ticker)
            if optimized:
                period = optimized["optimal_period"]
                buy_threshold = optimized["buy_threshold"]
                sell_threshold = optimized["sell_threshold"]
                logger.info(f"Using optimized RSI settings for {ticker}")
        
        # Calculate RSI
        df['rsi'] = ta.rsi(df['Close'], length=period)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['rsi'] < buy_threshold, 'signal'] = 1  # Buy signal
        df.loc[df['rsi'] > sell_threshold, 'signal'] = -1  # Sell signal
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        return df
        
    except Exception as e:
        error_msg = f"Error generating RSI signals: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def generate_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Generate trading signals using RSI strategy.
    
    This function implements the shared strategy interface that returns a DataFrame
    with signal columns for consistent usage across the system.
    
    Args:
        df: Price data DataFrame with OHLCV columns
        **kwargs: Additional parameters including:
            - ticker: Optional ticker symbol for loading optimized settings
            - period: RSI period (default: 14)
            - buy_threshold: RSI level for buy signals (default: 30)
            - sell_threshold: RSI level for sell signals (default: 70)
        
    Returns:
        DataFrame with original data plus signal columns:
            - signal: 1 for buy, -1 for sell, 0 for hold
            - rsi: RSI indicator values
            - returns: Price returns
            - strategy_returns: Strategy returns
            - cumulative_returns: Cumulative price returns
            - strategy_cumulative_returns: Cumulative strategy returns
            
    Raises:
        RuntimeError: If signal generation fails
    """
    try:
        # Extract parameters
        ticker = kwargs.get('ticker')
        period = kwargs.get('period', 14)
        buy_threshold = kwargs.get('buy_threshold', 30)
        sell_threshold = kwargs.get('sell_threshold', 70)
        
        # Generate RSI signals
        result_df = generate_rsi_signals(
            df, 
            ticker=ticker,
            period=period,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        
        # Ensure we have all required columns
        required_columns = ['signal', 'rsi', 'returns', 'strategy_returns']
        for col in required_columns:
            if col not in result_df.columns:
                logger.warning(f"Missing required column {col} in RSI strategy output")
                result_df[col] = 0
        
        # Add metadata columns
        result_df['strategy_name'] = 'RSI'
        result_df['strategy_params'] = json.dumps({
            'period': period,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'ticker': ticker
        })
        
        logger.info(f"Successfully generated RSI signals for {len(result_df)} data points")
        return result_df
        
    except Exception as e:
        error_msg = f"Error generating RSI signals: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) 