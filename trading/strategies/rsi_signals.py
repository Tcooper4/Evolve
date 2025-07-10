"""RSI strategy signal generator."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.common_helpers import normalize_indicator_name
from .rsi_utils import (
    generate_rsi_signals_core,
    validate_rsi_parameters,
    get_default_rsi_parameters
)

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
    sell_threshold: float = 70,
    user_config: dict = None,
    streamlit_session: dict = None
) -> pd.DataFrame:
    """Generate RSI trading signals with user-configurable thresholds.
    
    Args:
        df: Price data DataFrame with OHLCV columns
        ticker: Optional ticker symbol for loading optimized settings
        period: RSI period
        buy_threshold: RSI level for buy signals
        sell_threshold: RSI level for sell signals
        user_config: Optional user configuration dict with thresholds
        
    Returns:
        DataFrame with RSI signals and returns
    """
    try:
        # Validate required columns
        if 'Close' not in df.columns:
            raise ValueError("Missing 'Close' column in DataFrame")
        
        # Apply user configuration if provided
        if user_config:
            buy_threshold = user_config.get('buy_threshold', buy_threshold)
            sell_threshold = user_config.get('sell_threshold', sell_threshold)
            period = user_config.get('period', period)
        
        # Apply Streamlit session configuration if provided
        if streamlit_session:
            buy_threshold = streamlit_session.get('rsi_buy_threshold', buy_threshold)
            sell_threshold = streamlit_session.get('rsi_sell_threshold', sell_threshold)
            period = streamlit_session.get('rsi_period', period)
        
        # Load optimized settings if available and no user config
        if ticker and not user_config and not streamlit_session:
            optimized = load_optimized_settings(ticker)
            if optimized:
                period = optimized["optimal_period"]
                buy_threshold = optimized["buy_threshold"]
                sell_threshold = optimized["sell_threshold"]
                logger.info(f"Using optimized RSI settings for {ticker}")
        
        # Validate parameters
        is_valid, error_msg = validate_rsi_parameters(period, buy_threshold, sell_threshold)
        if not is_valid:
            raise ValueError(f"Invalid RSI parameters: {error_msg}")
        
        # Use shared core function
        result_df = generate_rsi_signals_core(
            df, 
            period=period,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        
        return result_df
        
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

def calculate_rsi_fallback(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using a fallback implementation when pandas_ta is not available.
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values
    """
    try:
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.error(f"Error calculating RSI fallback: {e}")
        return pd.Series([np.nan] * len(prices), index=prices.index) 