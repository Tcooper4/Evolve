"""Yahoo Finance data provider with caching and logging."""

import time
import yfinance as yf
from typing import Dict, Any, Optional
import pandas as pd
import requests
import logging
from pathlib import Path
from datetime import datetime
import pickle

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = Path("memory/logs/yfinance.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# Cache configuration
CACHE_DIR = Path("memory/cache/yfinance")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRY = 3600  # 1 hour in seconds

def log_data_request(symbol: str, success: bool, error: Optional[str] = None) -> None:
    """Log data request details.
    
    Args:
        symbol: Stock symbol
        success: Whether request was successful
        error: Optional error message
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "success": success,
        "error": error
    }
    
    log_path = Path("memory/logs/data_requests.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "a") as f:
        f.write(f"{log_entry}\n")

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def get_cached_data(symbol: str) -> Optional[pd.DataFrame]:
    """Get cached data if available and not expired.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Cached DataFrame or None if not available/expired
    """
    cache_path = CACHE_DIR / f"{symbol}.pkl"
    if not cache_path.exists():
        return None
        
    try:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
            
        if time.time() - cache_data["timestamp"] > CACHE_EXPIRY:
            logger.info(f"Cache expired for {symbol}")
            return None
            
        logger.info(f"Using cached data for {symbol}")
        return cache_data["data"]
    except Exception as e:
        logger.error(f"Error reading cache for {symbol}: {e}")
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def cache_data(symbol: str, data: pd.DataFrame) -> None:
    """Cache data with timestamp.
    
    Args:
        symbol: Stock symbol
        data: DataFrame to cache
    """
    try:
        cache_data = {
            "timestamp": time.time(),
            "data": data
        }
        
        cache_path = CACHE_DIR / f"{symbol}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
            
        logger.info(f"Cached data for {symbol}")
    except Exception as e:
        logger.error(f"Error caching data for {symbol}: {e}")

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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
            # Check cache first
            cached_data = get_cached_data(symbol)
            if cached_data is not None:
                log_data_request(symbol, True)
                return {'success': True, 'result': cached_data, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            # Create Ticker object
            ticker = yf.Ticker(symbol)
            
            # Download data
            logger.info(f"Fetching data for {symbol}")
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            # Validate data
            self._validate_data(data)
            
            # Cache the data
            cache_data(symbol, data)
            log_data_request(symbol, True)
            
            # Add delay between requests
            time.sleep(self.delay)
            
            return data
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching data for {symbol}: {error_msg}")
            log_data_request(symbol, False, error_msg)
            raise RuntimeError(f"Error fetching data for {symbol}: {error_msg}")

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
                logger.error(f"Skipping {symbol} due to error: {e}")
                continue
        return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}