"""
Data Loader Module for Market Data

This module provides comprehensive data loading capabilities for market data,
including single and multiple ticker loading, price retrieval, and data validation.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ yfinance not available. Disabling Yahoo Finance data loading.")
    print(f"   Missing: {e}")
    yf = None
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Data Models ---


@dataclass
class DataLoadRequest:
    """Represents a data load request."""

    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: str = "1y"
    interval: str = "1d"
    auto_adjust: bool = True
    prepost: bool = False
    threads: bool = True
    proxy: Optional[str] = None
    progress: bool = False


@dataclass
class DataLoadResponse:
    """Represents a data load response."""

    success: bool
    message: str
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    ticker: str = ""
    rows: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriceResponse:
    """Represents a price response."""

    success: bool
    message: str
    price: Optional[float] = None
    ticker: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelLoadConfig:
    """Configuration for parallel data loading."""

    max_workers: int = 4
    timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    chunk_size: int = 10
    progress_callback: Optional[callable] = None


# --- Configuration Management ---


class DataLoaderConfig:
    """Configuration management for data loader."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "trading/data/config/data_loader_config.json"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        default_config = {
            "yfinance": {
                "auto_adjust": True,
                "prepost": False,
                "threads": True,
                "progress": False,
                "timeout": 30,
                "retry_attempts": 3,
            },
            "validation": {
                "check_data_quality": True,
                "min_data_points": 5,
                "max_price_change": 0.5,
            },
            "caching": {"enabled": True, "ttl_minutes": 15},
            "parallel": {
                "max_workers": 4,
                "timeout_seconds": 60,
                "retry_attempts": 3,
                "retry_delay_seconds": 1.0,
                "chunk_size": 10,
                "enable_progress": True,
            },
        }

        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    self._merge_configs(default_config, file_config)
                    logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not load configuration from {self.config_path}: {e}")

        return default_config

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def get_yfinance_config(self) -> Dict[str, Any]:
        """Get YFinance configuration.

        Returns:
            YFinance configuration dictionary
        """
        return self.config.get("yfinance", {})

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration.

        Returns:
            Validation configuration dictionary
        """
        return self.config.get("validation", {})

    def get_parallel_config(self) -> ParallelLoadConfig:
        """Get parallel loading configuration.

        Returns:
            ParallelLoadConfig object
        """
        parallel_config = self.config.get("parallel", {})
        return ParallelLoadConfig(
            max_workers=parallel_config.get("max_workers", 4),
            timeout_seconds=parallel_config.get("timeout_seconds", 60),
            retry_attempts=parallel_config.get("retry_attempts", 3),
            retry_delay_seconds=parallel_config.get("retry_delay_seconds", 1.0),
            chunk_size=parallel_config.get("chunk_size", 10),
            progress_callback=None,  # Set externally if needed
        )


# --- Data Validation ---


class DataValidator:
    """Validates market data quality and consistency."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the validator.

        Args:
            config: Validation configuration
        """
        self.check_quality = config.get("check_data_quality", True)
        self.min_data_points = config.get("min_data_points", 5)
        self.max_price_change = config.get("max_price_change", 0.5)

    def validate_market_data(
        self, data: pd.DataFrame, ticker: str
    ) -> tuple[bool, Optional[str]]:
        """Validate market data.

        Args:
            data: Market data DataFrame
            ticker: Stock ticker for logging

        Returns:
            Tuple of (is_valid, error_message)
        """
        if data is None or data.empty:
            return False, "Data is None or empty"

        # Check for required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        # Check data length
        if len(data) < self.min_data_points:
            return (
                False,
                f"Insufficient data points: {len(data)} < {self.min_data_points}",
            )

        # Check for reasonable data ranges
        if (data["high"] < data["low"]).any():
            return False, "High price is less than low price"

        if (data["close"] < 0).any() or (data["volume"] < 0).any():
            return False, "Negative prices or volumes detected"

        # Check for extreme price changes
        if self.check_quality and len(data) > 1:
            price_changes = abs(data["close"].pct_change()).dropna()
            if (price_changes > self.max_price_change).any():
                return (
                    False,
                    f"Extreme price changes detected (>{self.max_price_change*100}%)",
                )

        return True, None


# --- Parallel Processing Utilities ---


class ParallelProcessor:
    """Handles parallel processing of data loading tasks."""

    def __init__(self, config: ParallelLoadConfig):
        """Initialize the parallel processor.

        Args:
            config: Parallel processing configuration
        """
        self.config = config
        self._lock = threading.Lock()
        self._completed_count = 0
        self._total_count = 0

    def _update_progress(self, ticker: str, success: bool):
        """Update progress counter and call progress callback."""
        with self._lock:
            self._completed_count += 1
            if self.config.progress_callback:
                self.config.progress_callback(
                    ticker=ticker,
                    success=success,
                    completed=self._completed_count,
                    total=self._total_count,
                    progress=self._completed_count / self._total_count,
                )

    def _load_single_ticker_with_retry(
        self, request: DataLoadRequest, loader_instance
    ) -> tuple[str, DataLoadResponse]:
        """Load a single ticker with retry logic.

        Args:
            request: Data load request
            loader_instance: DataLoader instance

        Returns:
            Tuple of (ticker, response)
        """
        ticker = request.ticker
        last_error = None

        for attempt in range(self.config.retry_attempts):
            try:
                response = loader_instance.load_market_data(request)
                self._update_progress(ticker, response.success)
                return ticker, response

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")

                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)

        # All retries failed
        error_response = DataLoadResponse(
            success=False,
            message=f"Failed after {self.config.retry_attempts} attempts: {last_error}",
            ticker=ticker,
        )
        self._update_progress(ticker, False)
        return ticker, error_response

    def load_tickers_parallel(
        self, requests: List[DataLoadRequest], loader_instance
    ) -> Dict[str, DataLoadResponse]:
        """Load multiple tickers in parallel.

        Args:
            requests: List of data load requests
            loader_instance: DataLoader instance

        Returns:
            Dictionary mapping ticker to response
        """
        self._completed_count = 0
        self._total_count = len(requests)

        results: Dict[str, DataLoadResponse] = {}

        # Create partial function with loader instance
        load_func = partial(
            self._load_single_ticker_with_retry, loader_instance=loader_instance
        )

        logger.info(
            f"Starting parallel load of {len(requests)} tickers with {self.config.max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(load_func, request): request.ticker
                for request in requests
            }

            # Process completed tasks
            for future in as_completed(
                future_to_ticker, timeout=self.config.timeout_seconds
            ):
                ticker = future_to_ticker[future]
                try:
                    ticker, response = future.result()
                    results[ticker] = response

                    if response.success:
                        logger.debug(
                            f"Successfully loaded {ticker} ({response.rows} rows)"
                        )
                    else:
                        logger.warning(f"Failed to load {ticker}: {response.message}")

                except Exception as e:
                    logger.error(f"Exception occurred while loading {ticker}: {e}")
                    error_response = DataLoadResponse(
                        success=False, message=f"Exception: {str(e)}", ticker=ticker
                    )
                    results[ticker] = error_response
                    self._update_progress(ticker, False)

        logger.info(
            f"Completed parallel load: {len([r for r in results.values() if r.success])}/{len(results)} successful"
        )
        return results

    def load_tickers_in_chunks(
        self, requests: List[DataLoadRequest], loader_instance
    ) -> Dict[str, DataLoadResponse]:
        """Load tickers in chunks to avoid overwhelming the API.

        Args:
            requests: List of data load requests
            loader_instance: DataLoader instance

        Returns:
            Dictionary mapping ticker to response
        """
        all_results: Dict[str, DataLoadResponse] = {}

        # Split requests into chunks
        chunks = [
            requests[i : i + self.config.chunk_size]
            for i in range(0, len(requests), self.config.chunk_size)
        ]

        logger.info(
            f"Processing {len(requests)} tickers in {len(chunks)} chunks of {self.config.chunk_size}"
        )

        for i, chunk in enumerate(chunks):
            logger.info(
                f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} tickers)"
            )

            chunk_results = self.load_tickers_parallel(chunk, loader_instance)
            all_results.update(chunk_results)

            # Add delay between chunks to be respectful to the API
            if i < len(chunks) - 1:
                time.sleep(0.5)

        return all_results


# --- Main Data Loader Class ---


class DataLoader:
    """Main data loader class with comprehensive functionality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data loader.

        Args:
            config: Optional configuration dictionary
        """
        self.config_manager = DataLoaderConfig()
        self.validator = DataValidator(self.config_manager.get_validation_config())
        self.yfinance_config = self.config_manager.get_yfinance_config()
        self.parallel_config = self.config_manager.get_parallel_config()
        self.parallel_processor = ParallelProcessor(self.parallel_config)
        self.cache: Dict[str, DataLoadResponse] = {}
        self.cache_ttl = timedelta(
            minutes=self.config_manager.config.get("caching", {}).get("ttl_minutes", 15)
        )

        # Add data caching mechanism using joblib or diskcache
        self.cache_dir = Path("cache/data_loader")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety for cache operations
        self._cache_lock = threading.Lock()

    def load_market_data(self, request: DataLoadRequest) -> DataLoadResponse:
        """Load market data for a single ticker.

        Args:
            request: Data load request

        Returns:
            Data load response
        """
        if not YFINANCE_AVAILABLE:
            return DataLoadResponse(
                success=False,
                message="yfinance is not available. Cannot load market data.",
                ticker=request.ticker,
            )
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(request)
            cached_data = self._load_from_disk_cache(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for {request.ticker}")
                return cached_data

            # Load data from YFinance
            ticker_obj = yf.Ticker(request.ticker)

            if request.start_date and request.end_date:
                data = ticker_obj.history(
                    start=request.start_date,
                    end=request.end_date,
                    interval=request.interval,
                    auto_adjust=request.auto_adjust,
                    prepost=request.prepost,
                    threads=request.threads,
                    proxy=request.proxy,
                    progress=request.progress,
                )
            else:
                data = ticker_obj.history(
                    period=request.period,
                    interval=request.interval,
                    auto_adjust=request.auto_adjust,
                    prepost=request.prepost,
                    threads=request.threads,
                    proxy=request.proxy,
                    progress=request.progress,
                )

            # Validate data
            is_valid, error_msg = self.validator.validate_market_data(
                data, request.ticker
            )
            if not is_valid:
                return DataLoadResponse(
                    success=False,
                    message=f"Data validation failed: {error_msg}",
                    ticker=request.ticker,
                )

            # Create response
            response = DataLoadResponse(
                success=True,
                message=f"Successfully loaded {len(data)} rows for {request.ticker}",
                data=data,
                ticker=request.ticker,
                rows=len(data),
                start_date=request.start_date,
                end_date=request.end_date,
                metadata={
                    "source": "yfinance",
                    "interval": request.interval,
                    "auto_adjust": request.auto_adjust,
                    "prepost": request.prepost,
                },
            )

            # Cache the result
            self._save_to_disk_cache(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Error loading data for {request.ticker}: {e}")
            return DataLoadResponse(
                success=False,
                message=f"Error loading data: {str(e)}",
                ticker=request.ticker,
            )

    def load_multiple_tickers(
        self,
        requests: List[DataLoadRequest],
        use_parallel: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, DataLoadResponse]:
        """Load market data for multiple tickers.

        Args:
            requests: List of data load requests
            use_parallel: Whether to use parallel processing
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping ticker to response
        """
        if not requests:
            return {}

        # Set progress callback if provided
        if progress_callback:
            self.parallel_processor.config.progress_callback = progress_callback

        try:
            if use_parallel and len(requests) > 1:
                # Use parallel processing for multiple tickers
                if len(requests) > self.parallel_config.chunk_size:
                    # Use chunked processing for large numbers of tickers
                    return self.parallel_processor.load_tickers_in_chunks(
                        requests, self
                    )
                else:
                    # Use direct parallel processing for smaller batches
                    return self.parallel_processor.load_tickers_parallel(requests, self)
            else:
                # Sequential processing
                results = {}
                for i, request in enumerate(requests):
                    if progress_callback:
                        progress_callback(
                            ticker=request.ticker,
                            success=True,
                            completed=i,
                            total=len(requests),
                            progress=i / len(requests),
                        )

                    response = self.load_market_data(request)
                    results[request.ticker] = response

                    if progress_callback:
                        progress_callback(
                            ticker=request.ticker,
                            success=response.success,
                            completed=i + 1,
                            total=len(requests),
                            progress=(i + 1) / len(requests),
                        )

                return results

        except Exception as e:
            logger.error(f"Error in load_multiple_tickers: {e}")
            # Return error responses for all requests
            return {
                request.ticker: DataLoadResponse(
                    success=False,
                    message=f"Batch processing error: {str(e)}",
                    ticker=request.ticker,
                )
                for request in requests
            }

    def get_latest_price(self, ticker: str) -> PriceResponse:
        """Get the latest price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Price response
        """
        if not YFINANCE_AVAILABLE:
            return PriceResponse(
                success=False,
                message="yfinance is not available. Cannot get latest price.",
                ticker=ticker,
            )
        
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            # Try to get current price from info
            current_price = info.get("regularMarketPrice")
            if current_price is None:
                # Fallback to history
                hist = ticker_obj.history(period="1d")
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
                else:
                    return PriceResponse(
                        success=False,
                        message="Could not retrieve current price",
                        ticker=ticker,
                    )

            return PriceResponse(
                success=True,
                message=f"Successfully retrieved price for {ticker}",
                price=current_price,
                ticker=ticker,
                metadata={
                    "source": "yfinance",
                    "currency": info.get("currency", "USD"),
                    "market_cap": info.get("marketCap"),
                    "volume": info.get("volume"),
                },
            )

        except Exception as e:
            logger.error(f"Error getting latest price for {ticker}: {e}")
            return PriceResponse(
                success=False,
                message=f"Error retrieving price: {str(e)}",
                ticker=ticker,
            )

    def _get_cache_key(self, request: DataLoadRequest) -> str:
        """Generate cache key for a request."""
        return f"{request.ticker}_{request.start_date}_{request.end_date}_{request.period}_{request.interval}"

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.joblib"

    def _load_from_disk_cache(self, cache_key: str) -> Optional[DataLoadResponse]:
        """Load data from disk cache."""
        try:
            with self._cache_lock:
                cache_file = self._get_cache_file_path(cache_key)
                if cache_file.exists():
                    # Check if cache is still valid
                    cache_age = datetime.now() - datetime.fromtimestamp(
                        cache_file.stat().st_mtime
                    )
                    if cache_age < self.cache_ttl:
                        cached_data = joblib.load(cache_file)
                        return cached_data
                    else:
                        # Remove expired cache
                        cache_file.unlink()
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
        return None

    def _save_to_disk_cache(self, cache_key: str, data: DataLoadResponse) -> None:
        """Save data to disk cache."""
        try:
            with self._cache_lock:
                cache_file = self._get_cache_file_path(cache_key)
                joblib.dump(data, cache_file)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            with self._cache_lock:
                for cache_file in self.cache_dir.glob("*.joblib"):
                    cache_file.unlink()
                logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.joblib"))
            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                "cache_files": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# --- Global Data Loader Instance ---
_data_loader = None


def get_data_loader() -> DataLoader:
    """Get the global data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader


# --- Convenience Functions ---


def load_market_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y",
) -> Dict[str, Any]:
    """Load market data for a single ticker.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        period: Time period if start/end dates not provided

    Returns:
        Dictionary with load result
    """
    try:
        request = DataLoadRequest(
            ticker=ticker, start_date=start_date, end_date=end_date, period=period
        )

        loader = get_data_loader()
        response = loader.load_market_data(request)

        return {
            "success": response.success,
            "message": response.message,
            "data": response.data.to_dict("records") if response.success else [],
            "ticker": response.ticker,
            "rows": response.rows,
            "timestamp": response.timestamp,
        }

    except Exception as e:
        logger.error(f"Error in load_market_data: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": [],
            "ticker": ticker,
            "rows": 0,
            "timestamp": datetime.now().isoformat(),
        }


def load_multiple_tickers(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y",
    use_parallel: bool = True,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Load market data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        period: Time period if start/end dates not provided
        use_parallel: Whether to use parallel processing
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with load results
    """
    try:
        requests = [
            DataLoadRequest(
                ticker=ticker, start_date=start_date, end_date=end_date, period=period
            )
            for ticker in tickers
        ]

        loader = get_data_loader()
        responses = loader.load_multiple_tickers(
            requests, use_parallel, progress_callback
        )

        # Convert responses to dictionary format
        results = {}
        for ticker, response in responses.items():
            results[ticker] = {
                "success": response.success,
                "message": response.message,
                "data": response.data.to_dict("records") if response.success else [],
                "rows": response.rows,
                "timestamp": response.timestamp,
            }

        return {
            "success": True,
            "results": results,
            "total_tickers": len(tickers),
            "successful_loads": len([r for r in responses.values() if r.success]),
            "failed_loads": len([r for r in responses.values() if not r.success]),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in load_multiple_tickers: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": {},
            "total_tickers": len(tickers),
            "successful_loads": 0,
            "failed_loads": len(tickers),
            "timestamp": datetime.now().isoformat(),
        }


def get_latest_price(ticker: str) -> Dict[str, Any]:
    """Get the latest price for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with price result
    """
    try:
        loader = get_data_loader()
        response = loader.get_latest_price(ticker)

        return {
            "success": response.success,
            "message": response.message,
            "price": response.price,
            "ticker": response.ticker,
            "timestamp": response.timestamp,
        }

    except Exception as e:
        logger.error(f"Error in get_latest_price: {e}")
        return {
            "success": False,
            "error": str(e),
            "price": None,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
        }


# --- Exports ---
__all__ = [
    "DataLoader",
    "DataLoadRequest",
    "DataLoadResponse",
    "PriceResponse",
    "DataLoaderConfig",
    "DataValidator",
    "get_data_loader",
    "load_market_data",
    "load_multiple_tickers",
    "get_latest_price",
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Data Loader Module for Market Data"
