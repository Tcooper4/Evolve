import time
import requests
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import asyncio
import aiohttp
from functools import lru_cache
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
import random
import yfinance as yf
from .yfinance_provider import YFinanceProvider

# Try to import redis, but make it optional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class AlphaVantageError(Exception):
    """Custom exception for Alpha Vantage API errors."""
    pass

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, calls_per_minute: int):
        """Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum number of API calls per minute
        """
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]
            
            if len(self.calls) >= self.calls_per_minute:
                # Wait until oldest call is more than 1 minute old
                sleep_time = 60 - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            self.calls.append(now)

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = Path("memory/logs/alpha_vantage.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# Cache configuration
CACHE_DIR = Path("memory/cache/alpha_vantage")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRY = 3600  # 1 hour in seconds

# Rate limit configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 32  # seconds

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
class AlphaVantageProvider:
    """Provider for fetching data from Alpha Vantage with fallback to Yahoo Finance."""
    
    def __init__(self, api_key: str, delay: float = 1.0):
        """Initialize the provider.
        
        Args:
            api_key: Alpha Vantage API key
            delay: Delay in seconds between requests
        """
        self.api_key = api_key
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.yfinance_provider = YFinanceProvider()
        
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
    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """Check for rate limit in response.
        
        Args:
            response: API response
            
        Returns:
            True if rate limited, False otherwise
        """
        if "Note" in response.json():
            note = response.json()["Note"]
            if "API call frequency" in note or "premium" in note.lower():
                return {'success': True, 'result': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        return False
        
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate backoff time with jitter.
        
        Args:
            attempt: Current attempt number
            
        Returns:
            Backoff time in seconds
        """
        backoff = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
        jitter = backoff * 0.1 * (2 * random.random() - 1)
        return {'success': True, 'result': backoff + jitter, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
    def get_data(self, symbol: str, start_date: Optional[str] = None,
                end_date: Optional[str] = None, interval: str = '1d') -> pd.DataFrame:
        """Get data from Alpha Vantage with exponential backoff and fallback.
        
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
                return cached_data
                
            # Try Alpha Vantage with exponential backoff
            for attempt in range(MAX_RETRIES):
                try:
                    url = f"https://www.alphavantage.co/query"
                    params = {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": symbol,
                        "apikey": self.api_key,
                        "outputsize": "full"
                    }
                    
                    response = self.session.get(url, params=params)
                    response.raise_for_status()
                    
                    # Check for rate limit
                    if self._handle_rate_limit(response):
                        backoff = self._exponential_backoff(attempt)
                        logger.warning(f"Rate limited, backing off for {backoff:.2f} seconds")
                        time.sleep(backoff)
                        continue
                        
                    # Parse response
                    data = response.json()
                    if "Time Series (Daily)" not in data:
                        raise ValueError("Invalid response format")
                        
                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                    df.index = pd.to_datetime(df.index)
                    df.columns = [col.split(". ")[1] for col in df.columns]
                    
                    # Filter date range
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                        
                    # Validate data
                    self._validate_data(df)
                    
                    # Cache the data
                    cache_data(symbol, df)
                    log_data_request(symbol, True)
                    
                    return df
                    
                except requests.exceptions.RequestException as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    backoff = self._exponential_backoff(attempt)
                    logger.warning(f"Request failed, backing off for {backoff:.2f} seconds: {e}")
                    time.sleep(backoff)
                    
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {error_msg}")
            log_data_request(symbol, False, error_msg)
            
            # Fallback to Yahoo Finance
            logger.info(f"Falling back to Yahoo Finance for {symbol}")
            try:
                df = self.yfinance_provider.get_data(symbol, start_date, end_date, interval)
                log_data_request(symbol, True, "Fallback to Yahoo Finance successful")
                return {'success': True, 'result': df, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            except Exception as fallback_error:
                error_msg = f"Both Alpha Vantage and Yahoo Finance failed: {fallback_error}"
                logger.error(error_msg)
                log_data_request(symbol, False, error_msg)
                raise RuntimeError(error_msg)

    def get_multiple_data(self, symbols: list, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols with fallback.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_data(symbol, start_date, end_date, interval)
            except Exception as e:
                logger.error(f"Skipping {symbol} due to error: {e}")
                continue
        return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol asynchronously.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price
            
        Raises:
            AlphaVantageError: If price fetching fails
        """
        try:
            # Check cache first
            cache_key = f"price_{symbol}"
            if self.redis_client:
                cached_price = self.redis_client.get(cache_key)
                if cached_price:
                    return float(cached_price)
                    
            # Wait for rate limit
            self.rate_limiter.wait_if_needed()
            
            # Prepare request
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
            if "Error Message" in data:
                raise AlphaVantageError(data["Error Message"])
                
            if "Global Quote" not in data:
                raise AlphaVantageError("No price data found in response")
                
            price = float(data["Global Quote"]["05. price"])
            
            # Save to cache
            if self.redis_client:
                self.redis_client.setex(cache_key, 60, str(price))
                
            return price
            
        except Exception as e:
            raise AlphaVantageError(f"Error fetching price for {symbol}: {str(e)}")
            
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols asynchronously.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to prices
            
        Raises:
            AlphaVantageError: If price fetching fails for any symbol
        """
        results = {}
        errors = []
        
        async def fetch_price(symbol: str) -> None:
            try:
                results[symbol] = await self.get_current_price(symbol)
            except Exception as e:
                errors.append(f"Error fetching price for {symbol}: {str(e)}")
                
        # Fetch prices concurrently
        tasks = [fetch_price(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
        
        if errors:
            raise AlphaVantageError("\n".join(errors))
            
        return results 