"""Data loader for market data."""

import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def load_market_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y"
) -> pd.DataFrame:
    """Load market data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period: Time period if start_date/end_date not provided
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date)
        else:
            data = yf.download(ticker, period=period)
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        
        # Reset index to make date a column
        data = data.reset_index()
        
        # Rename columns to standard format
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Loaded {len(data)} rows of data for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()

def load_multiple_tickers(
    tickers: list,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y"
) -> Dict[str, pd.DataFrame]:
    """Load market data for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period: Time period if start_date/end_date not provided
        
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    data_dict = {}
    
    for ticker in tickers:
        data = load_market_data(ticker, start_date, end_date, period)
        if not data.empty:
            data_dict[ticker] = data
    
    return data_dict

def get_latest_price(ticker: str) -> Optional[float]:
    """Get the latest price for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Latest closing price or None if not available
    """
    try:
        data = yf.download(ticker, period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception as e:
        logger.error(f"Error getting latest price for {ticker}: {str(e)}")
        return None 