"""Yahoo Finance data provider with caching and logging."""

import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ yfinance not available. Disabling Yahoo Finance data provider.")
    print(f"   Missing: {e}")
    yf = None
    YFINANCE_AVAILABLE = False

# Try to import tenacity
try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    TENACITY_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ tenacity not available. Disabling retry logic.")
    print(f"   Missing: {e}")
    retry = None
    retry_if_exception_type = None
    stop_after_attempt = None
    wait_exponential = None
    TENACITY_AVAILABLE = False

from .base_provider import BaseDataProvider, ProviderConfig

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = Path("memory/logs/yfinance.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler(log_file)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
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
        "error": error,
    }

    log_path = Path("memory/logs/data_requests.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"{log_entry}\n")


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
        return None


def cache_data(symbol: str, data: pd.DataFrame) -> None:
    """Cache data with timestamp.

    Args:
        symbol: Stock symbol
        data: DataFrame to cache
    """
    try:
        cache_data = {"timestamp": time.time(), "data": data}

        cache_path = CACHE_DIR / f"{symbol}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        logger.info(f"Cached data for {symbol}")
    except Exception as e:
        logger.error(f"Error caching data for {symbol}: {e}")


class YFinanceProvider(BaseDataProvider):
    """Provider for fetching data from Yahoo Finance."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize the provider.

        Args:
            config: Provider configuration (optional)
        """
        if config is None:
            config = ProviderConfig(
                name="yfinance",
                enabled=True,
                priority=1,
                rate_limit_per_minute=60,
                timeout_seconds=30,
                retry_attempts=3,
                custom_config={"delay": 1.0},
            )

        # Set delay attribute before calling super().__init__()
        self.delay = (
            config.custom_config.get("delay", 1.0) if config.custom_config else 1.0
        )

        super().__init__(config)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def _setup(self) -> None:
        """Setup method called during initialization."""
        self.logger.info(f"Initialized YFinance provider with delay: {self.delay}s")

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

    def fetch(self, symbol: str, interval: str = "1d", **kwargs) -> pd.DataFrame:
        """Fetch data for a given symbol and interval with retry logic.

        Args:
            symbol: Stock symbol
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters (start_date, end_date)

        Returns:
            DataFrame with OHLCV data

        Raises:
            Exception: If data fetching fails after all retries
        """
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance is not available. Cannot fetch data.")
        
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

            # Create Ticker object
            ticker = yf.Ticker(symbol)

            # Download data
            self.logger.info(f"Fetching data for {symbol}")
            data = ticker.history(
                start=kwargs.get("start_date"),
                end=kwargs.get("end_date"),
                interval=interval,
            )

            # Validate data
            self._validate_data(data)

            # Cache the data
            cache_data(symbol, data)
            self._update_status_on_success()
            log_data_request(symbol, True)

            # Add delay between requests
            time.sleep(self.delay)

            return data

        except Exception as e:
            error_msg = str(e)
            self._update_status_on_failure(error_msg)
            self.logger.error(f"Error fetching data for {symbol}: {error_msg}")
            log_data_request(symbol, False, error_msg)
            raise RuntimeError(f"Error fetching data for {symbol}: {error_msg}")

    def fetch_multiple(
        self, symbols: List[str], interval: str = "1d", **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.

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
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
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
        self,
        symbols: list,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
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
