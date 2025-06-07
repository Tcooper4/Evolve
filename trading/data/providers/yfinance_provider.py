import time
import yfinance as yf
from typing import Dict, Any, Optional
import pandas as pd
import requests

class YFinanceProvider:
    """Provider for fetching data from Yahoo Finance."""
    
    def __init__(self, delay: float = 1.0):
        """Initialize the provider.
        
        Args:
            delay: Delay in seconds between requests
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the fetched data.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("No data received")
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if data.isnull().any().any():
            raise ValueError("Data contains missing values")

    def get_data(self, symbol: str, start_date: Optional[str] = None,
                end_date: Optional[str] = None, interval: str = '1d') -> pd.DataFrame:
        """Get data from Yahoo Finance with rate limiting.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            RuntimeError: If data fetching fails
        """
        try:
            # Create Ticker object
            ticker = yf.Ticker(symbol)
            
            # Download data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            # Validate data
            self._validate_data(data)
            
            # Add delay between requests
            time.sleep(self.delay)
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Error fetching data for {symbol}: {str(e)}")

    def get_multiple_data(self, symbols: list, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols with rate limiting.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            Dictionary mapping symbols to DataFrames
            
        Raises:
            RuntimeError: If data fetching fails for any symbol
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_data(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
            except Exception as e:
                raise RuntimeError(f"Error fetching data for {symbol}: {str(e)}")
        return results 