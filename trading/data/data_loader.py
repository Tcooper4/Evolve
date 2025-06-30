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
) -> Dict[str, Any]:
    """Load market data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period: Time period if start_date/end_date not provided
        
    Returns:
        Dictionary with success status, data, and metadata
    """
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date)
        else:
            data = yf.download(ticker, period=period)
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return {
                'success': False,
                'message': f'No data found for ticker {ticker}',
                'data': pd.DataFrame(),
                'ticker': ticker,
                'rows': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Reset index to make date a column
        data = data.reset_index()
        
        # Rename columns to standard format
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Loaded {len(data)} rows of data for {ticker}")
        return {
            'success': True,
            'message': f'Successfully loaded {len(data)} rows for {ticker}',
            'data': data,
            'ticker': ticker,
            'rows': len(data),
            'start_date': data['date'].min() if 'date' in data.columns else None,
            'end_date': data['date'].max() if 'date' in data.columns else None,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return {
            'success': False,
            'message': f'Error loading data for {ticker}: {str(e)}',
            'data': pd.DataFrame(),
            'ticker': ticker,
            'rows': 0,
            'timestamp': datetime.now().isoformat()
        }

def load_multiple_tickers(
    tickers: list,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y"
) -> Dict[str, Any]:
    """Load market data for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period: Time period if start_date/end_date not provided
        
    Returns:
        Dictionary with success status, data dictionary, and metadata
    """
    data_dict = {}
    successful_loads = 0
    failed_loads = 0
    
    for ticker in tickers:
        result = load_market_data(ticker, start_date, end_date, period)
        if result['success']:
            data_dict[ticker] = result['data']
            successful_loads += 1
        else:
            failed_loads += 1
    
    return {
        'success': successful_loads > 0,
        'message': f'Loaded {successful_loads} tickers successfully, {failed_loads} failed',
        'data': data_dict,
        'tickers': tickers,
        'successful_count': successful_loads,
        'failed_count': failed_loads,
        'total_count': len(tickers),
        'timestamp': datetime.now().isoformat()
    }

def get_latest_price(ticker: str) -> Dict[str, Any]:
    """Get the latest price for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with success status, price, and metadata
    """
    try:
        data = yf.download(ticker, period="1d")
        if not data.empty:
            price = data['Close'].iloc[-1]
            return {
                'success': True,
                'message': f'Successfully retrieved latest price for {ticker}',
                'price': price,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'message': f'No data available for {ticker}',
                'price': None,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting latest price for {ticker}: {str(e)}")
        return {
            'success': False,
            'message': f'Error getting latest price for {ticker}: {str(e)}',
            'price': None,
            'ticker': ticker,
            'timestamp': datetime.now().isoformat()
        } 