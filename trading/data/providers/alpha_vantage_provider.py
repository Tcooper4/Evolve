import asyncio
import logging
import pickle
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import pandas as pd
import requests

from .base_provider import BaseDataProvider, ProviderConfig
from .yfinance_provider import YFinanceProvider

# Try to import redis, but make it optional
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Constants
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0
CACHE_DIR = Path("memory/cache/alpha_vantage")
CACHE_EXPIRY = 3600  # 1 hour in seconds

# Configure logging
logger = logging.getLogger(__name__)


def get_cached_data(symbol: str) -> Optional[pd.DataFrame]:
    """Get cached data if available and not expired."""
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
        return None


def cache_data(symbol: str, data: pd.DataFrame) -> None:
    """Cache data with timestamp."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_data = {"timestamp": time.time(), "data": data}

        cache_path = CACHE_DIR / f"{symbol}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        logger.info(f"Cached data for {symbol}")
    except Exception as e:
        logger.error(f"Error caching data for {symbol}: {e}")


def log_data_request(symbol: str, success: bool, error: Optional[str] = None) -> None:
    """Log data request details."""
    log_entry = {"timestamp": datetime.now().isoformat(), "symbol": symbol, "success": success, "error": error}

    log_path = Path("memory/logs/data_requests.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"{log_entry}\n")


class AlphaVantageError(Exception):
    """Custom exception for Alpha Vantage API errors."""



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


class AlphaVantageProvider(BaseDataProvider):
    """Provider for fetching data from Alpha Vantage with fallback to Yahoo Finance."""

    def __init__(self, config: Optional[ProviderConfig] = None, api_key: Optional[str] = None):
        """Initialize the provider.

        Args:
            config: Provider configuration (optional)
            api_key: Alpha Vantage API key (optional, can be in config)
        """
        if config is None:
            config = ProviderConfig(
                name="alpha_vantage",
                enabled=True,
                priority=1,
                rate_limit_per_minute=60,
                timeout_seconds=30,
                retry_attempts=3,
                custom_config={"delay": 1.0, "api_key": api_key},
            )

        super().__init__(config)
        self.api_key = api_key or config.custom_config.get("api_key") if config.custom_config else None
        if not self.api_key:
            raise ValueError("API key is required for Alpha Vantage provider")

        self.delay = config.custom_config.get("delay", 1.0) if config.custom_config else 1.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        self.yfinance_provider = YFinanceProvider()
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)

        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                import os

                redis_password = os.getenv("REDIS_PASSWORD")
                self.redis_client = redis.Redis(host="localhost", port=6379, db=0, password=redis_password)
                self.redis_client.ping()
            except (redis.ConnectionError, redis.TimeoutError) as e:
                self.logger.debug(f"Redis connection failed: {e}")
                self.redis_client = None

        # Call setup method after all attributes are set
        self._setup()

    def _setup(self) -> None:
        """Setup method called during initialization."""
        # Only log if delay attribute exists
        if hasattr(self, "delay"):
            self.logger.info(f"Initialized Alpha Vantage provider with delay: {self.delay}s")
        else:
            self.logger.info("Initialized Alpha Vantage provider")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the fetched data.

        Args:
            data: DataFrame to validate

        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("No data received")

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if data.isnull().any().any():
            raise ValueError("Data contains missing values")

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """Check for rate limit in response.

        Args:
            response: API response

        Returns:
            True if rate limited, False otherwise
        """
        try:
            response_data = response.json()

            # Check for error messages
            if "Error Message" in response_data:
                error_msg = response_data["Error Message"]
                raise AlphaVantageError(f"AlphaVantage error: {error_msg}")

            # Check for rate limit notes
            if "Note" in response_data:
                note = response_data["Note"]
                if "API call frequency" in note or "premium" in note.lower():
                    return True
                else:
                    raise AlphaVantageError(f"AlphaVantage error: {note}")

            # Check for invalid symbol or other errors
            if "Information" in response_data:
                info = response_data["Information"]
                if "Invalid API call" in info or "symbol" in info.lower():
                    raise AlphaVantageError(f"AlphaVantage error: {info}")

            return False

        except ValueError:
            # Response is not JSON
            if "API call frequency" in response.text or "premium" in response.text.lower():
                return True
            return False
        except Exception as e:
            logger.warning(f"Error parsing AlphaVantage response: {e}")
            return False

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate backoff time with jitter.

        Args:
            attempt: Current attempt number

        Returns:
            Backoff time in seconds
        """
        backoff = min(INITIAL_BACKOFF * (2**attempt), MAX_BACKOFF)
        jitter = backoff * 0.1 * (2 * random.random() - 1)
        return backoff + jitter

    def fetch(self, symbol: str, interval: str = "1d", **kwargs) -> pd.DataFrame:
        """Fetch data for a given symbol and interval.

        Args:
            symbol: Stock symbol
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters (start_date, end_date)

        Returns:
            DataFrame with OHLCV data

        Raises:
            Exception: If data fetching fails
        """
        if not self.is_enabled():
            raise RuntimeError("Provider is disabled")

        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")

        if not self.validate_interval(interval):
            raise ValueError(f"Invalid interval: {interval}")

        try:
            self._update_status_on_request()

            # Check cache first
            cached_data = get_cached_data(symbol)
            if cached_data is not None:
                self._update_status_on_success()
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
                        "outputsize": "full",
                    }

                    response = self.session.get(url, params=params)
                    response.raise_for_status()

                    # Check for rate limit
                    if self._handle_rate_limit(response):
                        backoff = self._exponential_backoff(attempt)
                        self.logger.warning(f"Rate limited, backing off for {backoff:.2f} seconds")
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
                    start_date = kwargs.get("start_date")
                    end_date = kwargs.get("end_date")
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]

                    # Validate data
                    self._validate_data(df)

                    # Cache the data
                    cache_data(symbol, df)
                    self._update_status_on_success()
                    log_data_request(symbol, True)

                    return df

                except requests.exceptions.RequestException as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    backoff = self._exponential_backoff(attempt)
                    self.logger.warning(f"Request failed, backing off for {backoff:.2f} seconds: {e}")
                    time.sleep(backoff)

        except Exception as e:
            error_msg = str(e)
            self._update_status_on_failure(error_msg)
            self.logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {error_msg}")
            log_data_request(symbol, False, error_msg)

            # Fallback to Yahoo Finance
            self.logger.info(f"Falling back to Yahoo Finance for {symbol}")
            try:
                df = self.yfinance_provider.fetch(symbol, interval, **kwargs)
                log_data_request(symbol, True, "Fallback to Yahoo Finance successful")
                return df
            except Exception as fallback_error:
                error_msg = f"Both Alpha Vantage and Yahoo Finance failed: {fallback_error}"
                self.logger.error(error_msg)
                log_data_request(symbol, False, error_msg)
                raise RuntimeError(error_msg)

    def fetch_multiple(self, symbols: List[str], interval: str = "1d", **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols with fallback.

        Args:
            symbols: List of stock symbols
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters (start_date, end_date)

        Returns:
            Dictionary mapping symbols to DataFrames

        Raises:
            Exception: If data fetching fails for any symbol
        """
        if not self.is_enabled():
            raise RuntimeError("Provider is disabled")

        results = {}
        failed_symbols = []

        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol, interval, **kwargs)
            except Exception as e:
                failed_symbols.append(symbol)
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue

        if failed_symbols:
            self.logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")

        return results

    def get_data(
        self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, interval: str = "1d"
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility.

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        kwargs = {}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        return self.fetch(symbol, interval, **kwargs)

    def get_multiple_data(
        self, symbols: list, start_date: Optional[str] = None, end_date: Optional[str] = None, interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Legacy method for backward compatibility.

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        kwargs = {}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        return self.fetch_multiple(symbols, interval, **kwargs)

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
            params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": self.api_key}

            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get("https://www.alphavantage.co/query", params=params) as response:
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
