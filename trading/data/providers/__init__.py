"""Unified data loading interface for multiple providers."""

import os
from typing import Optional, Dict, Any
import pandas as pd
from .alpha_vantage_provider import AlphaVantageProvider
from .yfinance_provider import YFinanceProvider

# Initialize providers
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
alpha_vantage = AlphaVantageProvider(api_key=ALPHA_VANTAGE_KEY)
yfinance = YFinanceProvider()

def load_data(
    symbol: str,
    source: str = 'auto',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d'
) -> pd.DataFrame:
    """Load data from specified or auto-selected source.
    
    Args:
        symbol: Stock symbol
        source: Data source ('auto', 'alpha_vantage', or 'yfinance')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, etc.)
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        ValueError: If source is invalid
        RuntimeError: If data loading fails
    """
    if source == 'auto':
        try:
            return alpha_vantage.get_data(symbol, start_date, end_date, interval)
        except Exception as e:
            print(f"Alpha Vantage failed, falling back to Yahoo Finance: {e}")
            return yfinance.get_data(symbol, start_date, end_date, interval)
    elif source == 'alpha_vantage':
        return alpha_vantage.get_data(symbol, start_date, end_date, interval)
    elif source == 'yfinance':
        return yfinance.get_data(symbol, start_date, end_date, interval)
    else:
        raise ValueError(f"Invalid source: {source}")

def load_multiple_data(
    symbols: list[str],
    source: str = 'auto',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d'
) -> Dict[str, pd.DataFrame]:
    """Load data for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        source: Data source ('auto', 'alpha_vantage', or 'yfinance')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, etc.)
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if source == 'auto':
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = load_data(symbol, 'auto', start_date, end_date, interval)
            except Exception as e:
                print(f"Failed to load {symbol}: {e}")
                continue
        return results
    elif source == 'alpha_vantage':
        return alpha_vantage.get_multiple_data(symbols, start_date, end_date, interval)
    elif source == 'yfinance':
        return yfinance.get_multiple_data(symbols, start_date, end_date, interval)
    else:
        raise ValueError(f"Invalid source: {source}")

__all__ = ['load_data', 'load_multiple_data', 'AlphaVantageProvider', 'YFinanceProvider'] 