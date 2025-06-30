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

def generate_signals(df: pd.DataFrame, ticker: str = None, **kwargs) -> Dict[str, Any]:
    """Generate trading signals using RSI strategy.
    
    Args:
        df: Price data DataFrame with OHLCV columns
        ticker: Optional ticker symbol for loading optimized settings
        **kwargs: Additional parameters for signal generation
        
    Returns:
        Dictionary containing signals and metadata
    """
    try:
        # Generate RSI signals
        result_df = generate_rsi_signals(df, ticker, **kwargs)
        
        # Extract signals
        signals = result_df['signal'].dropna()
        
        # Calculate signal statistics
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        total_signals = buy_signals + sell_signals
        
        # Calculate performance metrics
        if 'strategy_returns' in result_df.columns:
            strategy_returns = result_df['strategy_returns'].dropna()
            total_return = (1 + strategy_returns).prod() - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
            max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
        else:
            total_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'signals': signals.to_dict(),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': total_signals,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'strategy': 'RSI',
            'ticker': ticker,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Error generating signals: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) 