"""
Data Loader Module for Market Data

This module provides comprehensive data loading capabilities for market data,
including single and multiple ticker loading, price retrieval, and data validation.
"""

import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json

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
                "retry_attempts": 3
            },
            "validation": {
                "check_data_quality": True,
                "min_data_points": 5,
                "max_price_change": 0.5
            },
            "caching": {
                "enabled": True,
                "ttl_minutes": 15
            }
        }
        
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
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
    
    def validate_market_data(self, data: pd.DataFrame, ticker: str) -> tuple[bool, Optional[str]]:
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
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check data length
        if len(data) < self.min_data_points:
            return False, f"Insufficient data points: {len(data)} < {self.min_data_points}"
        
        # Check for reasonable data ranges
        if (data['high'] < data['low']).any():
            return False, "High price is less than low price"
        
        if (data['close'] < 0).any() or (data['volume'] < 0).any():
            return False, "Negative prices or volumes detected"
        
        # Check for extreme price changes
        if self.check_quality and len(data) > 1:
            price_changes = abs(data['close'].pct_change()).dropna()
            if (price_changes > self.max_price_change).any():
                return False, f"Extreme price changes detected (>{self.max_price_change*100}%)"
        
        return True, None

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
        self.cache: Dict[str, DataLoadResponse] = {}
        self.cache_ttl = timedelta(minutes=self.config_manager.config.get("caching", {}).get("ttl_minutes", 15))
        
        logger.info("DataLoader initialized")
    
    def load_market_data(self, request: DataLoadRequest) -> DataLoadResponse:
        """Load market data for a given ticker.
        
        Args:
            request: Data load request object
            
        Returns:
            DataLoadResponse with market data and metadata
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(request)
            if cache_key in self.cache:
                cached_response = self.cache[cache_key]
                if datetime.now() - datetime.fromisoformat(cached_response.timestamp) < self.cache_ttl:
                    logger.info(f"Cache hit for {request.ticker}")
                    return cached_response
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
            
            # Download data using YFinance
            logger.info(f"Loading market data for {request.ticker}")
            
            if request.start_date and request.end_date:
                data = yf.download(
                    request.ticker,
                    start=request.start_date,
                    end=request.end_date,
                    interval=request.interval,
                    auto_adjust=request.auto_adjust,
                    prepost=request.prepost,
                    threads=request.threads,
                    proxy=request.proxy,
                    progress=request.progress
                )
            else:
                data = yf.download(
                    request.ticker,
                    period=request.period,
                    interval=request.interval,
                    auto_adjust=request.auto_adjust,
                    prepost=request.prepost,
                    threads=request.threads,
                    proxy=request.proxy,
                    progress=request.progress
                )
            
            if data.empty:
                logger.warning(f"No data found for ticker {request.ticker}")
                return DataLoadResponse(
                    success=False,
                    message=f'No data found for ticker {request.ticker}',
                    ticker=request.ticker
                )
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]
            
            # Validate data
            is_valid, error_msg = self.validator.validate_market_data(data, request.ticker)
            if not is_valid:
                logger.warning(f"Data validation failed for {request.ticker}: {error_msg}")
                return DataLoadResponse(
                    success=False,
                    message=f'Data validation failed for {request.ticker}: {error_msg}',
                    ticker=request.ticker
                )
            
            # Create response
            response = DataLoadResponse(
                success=True,
                message=f'Successfully loaded {len(data)} rows for {request.ticker}',
                data=data,
                ticker=request.ticker,
                rows=len(data),
                start_date=data['date'].min().isoformat() if 'date' in data.columns else None,
                end_date=data['date'].max().isoformat() if 'date' in data.columns else None,
                metadata={
                    'period': request.period,
                    'interval': request.interval,
                    'auto_adjust': request.auto_adjust,
                    'prepost': request.prepost
                }
            )
            
            # Cache the response
            self.cache[cache_key] = response
            
            logger.info(f"Successfully loaded {len(data)} rows of data for {request.ticker}")
            return response
            
        except Exception as e:
            error_msg = f"Error loading data for {request.ticker}: {str(e)}"
            logger.error(error_msg)
            return DataLoadResponse(
                success=False,
                message=error_msg,
                ticker=request.ticker
            )
    
    def load_multiple_tickers(self, requests: List[DataLoadRequest]) -> Dict[str, DataLoadResponse]:
        """Load market data for multiple tickers.
        
        Args:
            requests: List of data load requests
            
        Returns:
            Dictionary mapping tickers to DataLoadResponse objects
        """
        results = {}
        successful_loads = 0
        failed_loads = 0
        
        for request in requests:
            try:
                response = self.load_market_data(request)
                results[request.ticker] = response
                
                if response.success:
                    successful_loads += 1
                else:
                    failed_loads += 1
                    
            except Exception as e:
                logger.error(f"Failed to load data for {request.ticker}: {e}")
                results[request.ticker] = DataLoadResponse(
                    success=False,
                    message=f"Failed to load data for {request.ticker}: {e}",
                    ticker=request.ticker
                )
                failed_loads += 1
        
        logger.info(f"Loaded {successful_loads} tickers successfully, {failed_loads} failed")
        return results
    
    def get_latest_price(self, ticker: str) -> PriceResponse:
        """Get the latest price for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            PriceResponse with price and metadata
        """
        try:
            logger.info(f"Getting latest price for {ticker}")
            
            data = yf.download(
                ticker,
                period="1d",
                auto_adjust=self.yfinance_config.get("auto_adjust", True),
                prepost=self.yfinance_config.get("prepost", False),
                threads=self.yfinance_config.get("threads", True)
            )
            
            if not data.empty:
                price = data['Close'].iloc[-1]
                return PriceResponse(
                    success=True,
                    message=f'Successfully retrieved latest price for {ticker}',
                    price=price,
                    ticker=ticker,
                    metadata={
                        'price_type': 'close',
                        'data_points': len(data)
                    }
                )
            else:
                return PriceResponse(
                    success=False,
                    message=f'No data available for {ticker}',
                    ticker=ticker
                )
                
        except Exception as e:
            error_msg = f"Error getting latest price for {ticker}: {str(e)}"
            logger.error(error_msg)
            return PriceResponse(
                success=False,
                message=error_msg,
                ticker=ticker
            )
    
    def _get_cache_key(self, request: DataLoadRequest) -> str:
        """Generate cache key for request."""
        return f"{request.ticker}_{request.start_date}_{request.end_date}_{request.period}_{request.interval}"
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Data loader cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return {
            'cache_size': len(self.cache),
            'cache_ttl_minutes': self.cache_ttl.total_seconds() / 60
        }

# --- Global Instance ---
_data_loader: Optional[DataLoader] = None

# --- Convenience Functions ---
def get_data_loader() -> DataLoader:
    """Get the global data loader instance.
    
    Returns:
        Global DataLoader instance
    """
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader

def load_market_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y"
) -> Dict[str, Any]:
    """Load market data for a given ticker (legacy function).
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period: Time period if start_date/end_date not provided
        
    Returns:
        Dictionary with success status, data, and metadata
    """
    loader = get_data_loader()
    request = DataLoadRequest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        period=period
    )
    response = loader.load_market_data(request)
    
    # Convert to legacy format
    return {
        'success': response.success,
        'message': response.message,
        'data': response.data,
        'ticker': response.ticker,
        'rows': response.rows,
        'start_date': response.start_date,
        'end_date': response.end_date,
        'timestamp': response.timestamp
    }

def load_multiple_tickers(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y"
) -> Dict[str, Any]:
    """Load market data for multiple tickers (legacy function).
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period: Time period if start_date/end_date not provided
        
    Returns:
        Dictionary with success status, data dictionary, and metadata
    """
    loader = get_data_loader()
    requests = [
        DataLoadRequest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
        for ticker in tickers
    ]
    
    responses = loader.load_multiple_tickers(requests)
    
    # Convert to legacy format
    data_dict = {}
    successful_loads = 0
    failed_loads = 0
    
    for ticker, response in responses.items():
        if response.success:
            data_dict[ticker] = response.data
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
    """Get the latest price for a ticker (legacy function).
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with success status, price, and metadata
    """
    loader = get_data_loader()
    response = loader.get_latest_price(ticker)
    
    # Convert to legacy format
    return {
        'success': response.success,
        'message': response.message,
        'price': response.price,
        'ticker': response.ticker,
        'timestamp': response.timestamp
    }

# --- Exports ---
__all__ = [
    'DataLoader',
    'DataLoadRequest',
    'DataLoadResponse',
    'PriceResponse',
    'DataLoaderConfig',
    'DataValidator',
    'get_data_loader',
    'load_market_data',
    'load_multiple_tickers',
    'get_latest_price'
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Data Loader Module for Market Data" 