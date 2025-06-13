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

class AlphaVantageProvider:
    """Provider for fetching data from Alpha Vantage."""
    
    def __init__(self, api_key: str, delay: float = 12.0, cache_ttl: int = 3600,
                 redis_config: Optional[Dict] = None):
        """Initialize the provider.
        
        Args:
            api_key: Alpha Vantage API key
            delay: Delay in seconds between requests (default: 12.0 for free tier)
            cache_ttl: Cache TTL in seconds (default: 3600)
            redis_config: Redis configuration dictionary
        """
        self.api_key = api_key
        self.delay = delay
        self.cache_ttl = cache_ttl
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize rate limiter (5 calls per minute for free tier)
        self.rate_limiter = RateLimiter(calls_per_minute=5)
        
        # Initialize Redis cache if configured and available
        self.redis_client = None
        if redis_config and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(**redis_config)
                # Test connection
                self.redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {str(e)}")
                self.redis_client = None
            
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        # Create cache directory
        self.cache_dir = Path("cache/alpha_vantage")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _get_cache_key(self, symbol: str, interval: str) -> str:
        """Generate cache key for data.
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            
        Returns:
            Cache key string
        """
        return f"{symbol}_{interval}"
        
    def _get_cached_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Get cached data if available and valid.
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            
        Returns:
            Cached DataFrame if valid, None otherwise
        """
        cache_key = self._get_cache_key(symbol, interval)
        
        # Try Redis cache first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)
            except Exception as e:
                self.logger.warning(f"Redis cache error: {str(e)}")
                
        # Try file cache
        cache_file = self.cache_dir / f"{cache_key}.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if (datetime.now() - df.index[-1]).total_seconds() < self.cache_ttl:
                    return df
            except Exception as e:
                self.logger.warning(f"File cache error: {str(e)}")
                
        return None
        
    def _save_to_cache(self, symbol: str, interval: str, data: pd.DataFrame) -> None:
        """Save data to cache.
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            data: DataFrame to cache
        """
        cache_key = self._get_cache_key(symbol, interval)
        
        # Save to Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    data.to_json()
                )
            except Exception as e:
                self.logger.warning(f"Redis cache error: {str(e)}")
                
        # Save to file cache
        try:
            cache_file = self.cache_dir / f"{cache_key}.csv"
            data.to_csv(cache_file)
        except Exception as e:
            self.logger.warning(f"File cache error: {str(e)}")
            
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the fetched data.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            AlphaVantageError: If data is invalid
        """
        if data.empty:
            raise AlphaVantageError("No data received")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise AlphaVantageError(f"Missing required columns: {missing_columns}")
            
        if data.isnull().any().any():
            raise AlphaVantageError("Data contains missing values")
            
    def _parse_time_series(self, response: Dict[str, Any]) -> pd.DataFrame:
        """Parse time series data from API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            AlphaVantageError: If response is invalid
        """
        try:
            # Check for API errors
            if "Error Message" in response:
                raise AlphaVantageError(response["Error Message"])
                
            if "Note" in response:
                self.logger.warning(response["Note"])
                
            # Get time series data
            time_series_key = next((k for k in response.keys() if "Time Series" in k), None)
            if not time_series_key:
                raise AlphaVantageError("No time series data found in response")
                
            # Convert to DataFrame
            data = pd.DataFrame.from_dict(response[time_series_key], orient='index')
            data.index = pd.to_datetime(data.index)
            data = data.astype(float)
            
            # Rename columns
            column_map = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            data = data.rename(columns=column_map)
            
            return data
            
        except Exception as e:
            raise AlphaVantageError(f"Error parsing time series data: {str(e)}")
            
    def fetch_data(self, symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """Fetch data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            interval: Data interval (daily, weekly, monthly)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            AlphaVantageError: If data fetching fails
        """
        try:
            # Check cache first
            cached_data = self._get_cached_data(symbol, interval)
            if cached_data is not None:
                return cached_data
                
            # Validate interval
            valid_intervals = ['daily', 'weekly', 'monthly']
            if interval not in valid_intervals:
                raise AlphaVantageError(f"Invalid interval. Must be one of {valid_intervals}")
                
            # Wait for rate limit
            self.rate_limiter.wait_if_needed()
                
            # Prepare request
            params = {
                'function': f'TIME_SERIES_{interval.upper()}',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            # Make request
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            df = self._parse_time_series(data)
            
            # Validate data
            self._validate_data(df)
            
            # Save to cache
            self._save_to_cache(symbol, interval, df)
            
            # Add delay between requests
            time.sleep(self.delay)
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise AlphaVantageError(f"API request failed: {str(e)}")
        except Exception as e:
            raise AlphaVantageError(f"Error fetching data for {symbol}: {str(e)}")
            
    def get_historical_data(self, symbol: str, interval: str = 'daily',
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Get historical data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            interval: Data interval (daily, weekly, monthly)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            AlphaVantageError: If data fetching fails
        """
        try:
            # Fetch data
            df = self.fetch_data(symbol, interval)
            
            # Filter by date range
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            return df
            
        except Exception as e:
            raise AlphaVantageError(f"Error getting historical data for {symbol}: {str(e)}")
            
    def get_multiple_data(self, symbols: List[str], interval: str = 'daily',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            interval: Data interval (daily, weekly, monthly)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping symbols to DataFrames
            
        Raises:
            AlphaVantageError: If data fetching fails for any symbol
        """
        results = {}
        errors = []
        
        def fetch_symbol(symbol: str) -> None:
            try:
                results[symbol] = self.get_historical_data(
                    symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                errors.append(f"Error fetching {symbol}: {str(e)}")
                
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=min(len(symbols), 5)) as executor:
            executor.map(fetch_symbol, symbols)
            
        if errors:
            raise AlphaVantageError("\n".join(errors))
            
        return results
        
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