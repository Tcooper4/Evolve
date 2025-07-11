"""
Consolidated Data Provider Module

This module provides a unified interface for accessing market data from multiple sources
with automatic fallback, caching, and error handling.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import json

# Import existing providers
from .providers.base_provider import BaseDataProvider, ProviderConfig
from .providers.alpha_vantage_provider import AlphaVantageProvider
from .providers.yfinance_provider import YFinanceProvider
from .providers.fallback_provider import FallbackProvider

logger = logging.getLogger(__name__)

# --- Data Models ---
@dataclass
class DataRequest:
    """Represents a data request with all necessary parameters."""
    symbol: str
    interval: str = '1d'
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    provider: Optional[str] = None
    use_cache: bool = True
    retry_count: int = 3
    timeout_seconds: int = 30
    priority: int = 1

@dataclass
class DataResponse:
    """Represents a data response with metadata."""
    data: pd.DataFrame
    provider: str
    cache_hit: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class CacheEntry:
    """Represents a cache entry with expiration."""
    response: DataResponse
    expires_at: datetime

@dataclass
class ProviderStatus:
    """Represents the status of a data provider."""
    name: str
    enabled: bool
    last_used: Optional[datetime] = None
    success_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    average_response_time: float = 0.0

# --- Configuration Management ---
class DataProviderConfig:
    """Configuration management for data providers."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "trading/data/config/data_provider_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        default_config = {
            "cache": {
                "enabled": True,
                "ttl_minutes": 15,
                "max_size": 1000
            },
            "providers": {
                "alpha_vantage": {
                    "enabled": True,
                    "priority": 1,
                    "timeout_seconds": 30,
                    "retry_attempts": 3
                },
                "yfinance": {
                    "enabled": True,
                    "priority": 2,
                    "timeout_seconds": 30,
                    "retry_attempts": 3
                },
                "fallback": {
                    "enabled": True,
                    "priority": 3,
                    "timeout_seconds": 30,
                    "retry_attempts": 3
                }
            },
            "validation": {
                "check_data_quality": True,
                "min_data_points": 10,
                "max_price_change": 0.5  # 50% max price change
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
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider configuration dictionary
        """
        return self.config.get("providers", {}).get(provider_name, {})
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration.
        
        Returns:
            Cache configuration dictionary
        """
        return self.config.get("cache", {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration.
        
        Returns:
            Validation configuration dictionary
        """
        return self.config.get("validation", {})

# --- Cache Management ---
class DataCache:
    """Manages data caching with TTL and size limits."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the cache.
        
        Args:
            config: Cache configuration
        """
        self.enabled = config.get("enabled", True)
        self.ttl = timedelta(minutes=config.get("ttl_minutes", 15))
        self.max_size = config.get("max_size", 1000)
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def get(self, key: str) -> Optional[DataResponse]:
        """Get data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data response or None if not found/expired
        """
        if not self.enabled:
            return None
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry.expires_at:
                self.stats["hits"] += 1
                return entry.response
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def put(self, key: str, response: DataResponse) -> None:
        """Store data in cache.
        
        Args:
            key: Cache key
            response: Data response to cache
        """
        if not self.enabled:
            return
        
        # Check cache size limit
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        expires_at = datetime.now() + self.ttl
        self.cache[key] = CacheEntry(response=response, expires_at=expires_at)
    
    def _evict_oldest(self) -> None:
        """Remove the oldest cache entry."""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].expires_at)
        del self.cache[oldest_key]
        self.stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Data cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_minutes": self.ttl.total_seconds() / 60
        }

# --- Data Validation ---
class DataValidator:
    """Validates market data quality and consistency."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the validator.
        
        Args:
            config: Validation configuration
        """
        self.check_quality = config.get("check_data_quality", True)
        self.min_data_points = config.get("min_data_points", 10)
        self.max_price_change = config.get("max_price_change", 0.5)
    
    def validate(self, data: pd.DataFrame, symbol: str) -> tuple[bool, Optional[str]]:
        """Validate market data.
        
        Args:
            data: Market data DataFrame
            symbol: Stock symbol for logging
            
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
        if self.check_quality:
            price_changes = abs(data['close'].pct_change()).dropna()
            if (price_changes > self.max_price_change).any():
                return False, f"Extreme price changes detected (>{self.max_price_change*100}%)"
        
        return True, None

# --- Main Data Provider Manager ---
class DataProviderManager:
    """
    Unified data provider manager with automatic fallback and caching.
    
    This class consolidates all data provider functionality into a single interface,
    providing automatic fallback between providers, caching, and comprehensive error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data provider manager.
        
        Args:
            config: Configuration dictionary for providers
        """
        self.config_manager = DataProviderConfig()
        self.providers: Dict[str, BaseDataProvider] = {}
        self.provider_status: Dict[str, ProviderStatus] = {}
        self.cache: Dict[str, DataResponse] = {}
        self.cache_ttl = timedelta(minutes=15)  # 15 minute cache TTL
        self.validator = DataValidator(self.config_manager.get_validation_config())
        self.request_history: List[DataRequest] = []
        
        self._setup_providers()
        logger.info("DataProviderManager initialized with fallback support")
    
    def _setup_providers(self) -> None:
        """Setup all available data providers."""
        try:
            # Setup AlphaVantage provider
            alpha_config = self.config_manager.get_provider_config('alpha_vantage')
            if alpha_config.get('enabled', True):
                try:
                    self.providers['alpha_vantage'] = AlphaVantageProvider(alpha_config)
                    self.provider_status['alpha_vantage'] = ProviderStatus(
                        name='alpha_vantage',
                        enabled=True
                    )
                    logger.info("AlphaVantage provider initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize AlphaVantage provider: {e}")
            
            # Setup YFinance provider
            yfinance_config = self.config_manager.get_provider_config('yfinance')
            if yfinance_config.get('enabled', True):
                try:
                    self.providers['yfinance'] = YFinanceProvider(yfinance_config)
                    self.provider_status['yfinance'] = ProviderStatus(
                        name='yfinance',
                        enabled=True
                    )
                    logger.info("YFinance provider initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize YFinance provider: {e}")
            
            # Setup fallback provider
            fallback_config = self.config_manager.get_provider_config('fallback')
            if fallback_config.get('enabled', True):
                try:
                    self.providers['fallback'] = FallbackProvider(fallback_config)
                    self.provider_status['fallback'] = ProviderStatus(
                        name='fallback',
                        enabled=True
                    )
                    logger.info("Fallback provider initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize fallback provider: {e}")
            
            if not self.providers:
                raise RuntimeError("No data providers could be initialized")
                
        except Exception as e:
            logger.error(f"Error setting up providers: {e}")
            raise RuntimeError(f"Failed to initialize data providers: {e}")
    
    def get_data(self, request: DataRequest) -> DataResponse:
        """
        Get market data with automatic fallback.
        
        Args:
            request: Data request object
            
        Returns:
            DataResponse with market data and metadata
            
        Raises:
            RuntimeError: If all providers fail
        """
        start_time = datetime.now()
        
        # Check cache first
        if request.use_cache:
            cache_key = self._get_cache_key(request)
            cached_response = self._get_cached_data(request)
            if cached_response is not None:
                logger.info(f"Cache hit for {request.symbol}")
                return cached_response
        
        # Try providers in order
        provider_order = self._get_provider_order(request.provider)
        
        for provider_name in provider_order:
            if provider_name not in self.providers:
                logger.warning(f"Provider {provider_name} not available, skipping")
                continue
            
            try:
                logger.info(f"Attempting to fetch data from {provider_name} for {request.symbol}")
                
                # Log fetch latency per provider call
                provider_start_time = datetime.now()
                
                provider = self.providers[provider_name]
                data = provider.fetch(
                    request.symbol,
                    request.interval,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                
                # Calculate and log latency
                provider_latency = (datetime.now() - provider_start_time).total_seconds()
                logger.info(f"Provider {provider_name} fetch latency for {request.symbol}: {provider_latency:.3f}s")
                
                # Validate data
                is_valid, error_msg = self.validator.validate(data, request.symbol)
                if is_valid:
                    response = DataResponse(
                        data=data,
                        provider=provider_name,
                        cache_hit=False,
                        metadata={
                            'symbol': request.symbol,
                            'interval': request.interval,
                            'data_points': len(data),
                            'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
                        }
                    )
                    
                    # Update provider status
                    self._update_provider_status(provider_name, True, None)
                    
                    # Cache the response
                    if request.use_cache:
                        self._cache_data(request, response)
                    
                    # Log request history
                    self.request_history.append(request)
                    
                    logger.info(f"Successfully fetched data from {provider_name} for {request.symbol}")
                    return response
                else:
                    logger.warning(f"Invalid data received from {provider_name} for {request.symbol}: {error_msg}")
                    self._update_provider_status(provider_name, False, error_msg)
                    
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {request.symbol}: {e}")
                self._update_provider_status(provider_name, False, str(e))
                continue
        
        # All providers failed
        error_msg = f"All data providers failed for {request.symbol}"
        logger.error(error_msg)
        return DataResponse(
            data=pd.DataFrame(),
            provider="none",
            cache_hit=False,
            error=error_msg
        )
    
    def get_multiple_data(self, requests: List[DataRequest]) -> Dict[str, DataResponse]:
        """
        Get data for multiple symbols efficiently.
        
        Args:
            requests: List of data requests
            
        Returns:
            Dictionary mapping symbols to DataResponse objects
        """
        results = {}
        failed_symbols = []
        
        for request in requests:
            try:
                response = self.get_data(request)
                if response.error is None:
                    results[request.symbol] = response
                else:
                    failed_symbols.append(request.symbol)
            except Exception as e:
                failed_symbols.append(request.symbol)
                logger.error(f"Failed to fetch data for {request.symbol}: {e}")
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
        
        return results
    
    async def get_data_async(self, request: DataRequest) -> DataResponse:
        """
        Get market data asynchronously.
        
        Args:
            request: Data request object
            
        Returns:
            DataResponse with market data and metadata
        """
        # For now, run synchronously in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_data, request)
    
    def _get_provider_order(self, preferred_provider: Optional[str] = None) -> List[str]:
        """Get the order of providers to try."""
        if preferred_provider and preferred_provider in self.providers:
            # Start with preferred provider, then fallback
            order = [preferred_provider]
            for provider in ['alpha_vantage', 'yfinance', 'fallback']:
                if provider != preferred_provider and provider in self.providers:
                    order.append(provider)
            return order
        
        # Default order: AlphaVantage -> YFinance -> Fallback
        return ['alpha_vantage', 'yfinance', 'fallback']
    
    def _get_cached_data(self, request: DataRequest) -> Optional[DataResponse]:
        """Get cached data if available and not expired."""
        cache_key = self._get_cache_key(request)
        
        if cache_key in self.cache:
            cached_response = self.cache[cache_key]
            if datetime.now() - cached_response.timestamp < self.cache_ttl:
                return cached_response
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    def _cache_data(self, request: DataRequest, response: DataResponse) -> None:
        """Cache the data response."""
        cache_key = self._get_cache_key(request)
        self.cache[cache_key] = response
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
    
    def _get_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request."""
        return f"{request.symbol}_{request.interval}_{request.start_date}_{request.end_date}"
    
    def _update_provider_status(self, provider_name: str, success: bool, error: Optional[str]) -> None:
        """Update provider status statistics."""
        if provider_name in self.provider_status:
            status = self.provider_status[provider_name]
            status.last_used = datetime.now()
            if success:
                status.success_count += 1
            else:
                status.error_count += 1
                status.last_error = error
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        for name, provider in self.providers.items():
            try:
                provider_status = self.provider_status.get(name, ProviderStatus(name=name, enabled=False))
                status[name] = {
                    'enabled': provider.is_enabled() and provider_status.enabled,
                    'metadata': provider.get_metadata(),
                    'success_count': provider_status.success_count,
                    'error_count': provider_status.error_count,
                    'last_error': provider_status.last_error,
                    'last_used': provider_status.last_used.isoformat() if provider_status.last_used else None
                }
            except Exception as e:
                status[name] = {
                    'enabled': False,
                    'error': str(e)
                }
        
        return status
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'cache_ttl_minutes': self.cache_ttl.total_seconds() / 60,
            'request_history_count': len(self.request_history)
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Data cache cleared")
    
    def get_request_history(self, limit: int = 100) -> List[DataRequest]:
        """Get recent request history."""
        return self.request_history[-limit:]

# --- Global Instance ---
_data_provider_manager: Optional[DataProviderManager] = None

# --- Convenience Functions ---
def get_data_provider(config: Optional[Dict[str, Any]] = None) -> DataProviderManager:
    """Get a configured data provider manager.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        DataProviderManager instance
    """
    global _data_provider_manager
    if _data_provider_manager is None:
        _data_provider_manager = DataProviderManager(config)
    return _data_provider_manager

def fetch_data(symbol: str, interval: str = '1d', **kwargs) -> pd.DataFrame:
    """Fetch data for a single symbol (convenience function).
    
    Args:
        symbol: Stock symbol
        interval: Data interval
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with market data
    """
    provider = get_data_provider()
    request = DataRequest(symbol=symbol, interval=interval, **kwargs)
    response = provider.get_data(request)
    if response.error:
        raise RuntimeError(f"Failed to fetch data for {symbol}: {response.error}")
    return response.data

def fetch_multiple_data(symbols: List[str], interval: str = '1d', **kwargs) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple symbols (convenience function).
    
    Args:
        symbols: List of stock symbols
        interval: Data interval
        **kwargs: Additional parameters
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    provider = get_data_provider()
    requests = [DataRequest(symbol=symbol, interval=interval, **kwargs) for symbol in symbols]
    responses = provider.get_multiple_data(requests)
    return {symbol: response.data for symbol, response in responses.items() if response.error is None}

# --- Exports ---
__all__ = [
    'DataRequest',
    'DataResponse',
    'CacheEntry',
    'ProviderStatus',
    'DataProviderConfig',
    'DataCache',
    'DataValidator',
    'DataProviderManager',
    'get_data_provider',
    'fetch_data',
    'fetch_multiple_data'
]

# Backward compatibility aliases
DataProvider = DataProviderManager


# Backward compatibility aliases
DataProvider = DataProviderManager

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Consolidated Data Provider Module"
