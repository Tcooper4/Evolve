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
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import existing providers
from .providers.base_provider import BaseDataProvider, ProviderConfig
from .providers.alpha_vantage_provider import AlphaVantageProvider
from .providers.yfinance_provider import YFinanceProvider
from .providers.fallback_provider import FallbackProvider

logger = logging.getLogger(__name__)

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

@dataclass
class DataResponse:
    """Represents a data response with metadata."""
    data: pd.DataFrame
    provider: str
    cache_hit: bool
    timestamp: datetime
    metadata: Dict[str, Any]

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
        self.config = config or {}
        self.providers: Dict[str, BaseDataProvider] = {}
        self.cache: Dict[str, DataResponse] = {}
        self.cache_ttl = timedelta(minutes=15)  # 15 minute cache TTL
        self.request_history: List[DataRequest] = []
        
        self._setup_providers()
        logger.info("DataProviderManager initialized with fallback support")
    
    def _setup_providers(self) -> None:
        """Setup all available data providers."""
        try:
            # Setup AlphaVantage provider
            alpha_config = self.config.get('alpha_vantage', {})
            if alpha_config.get('api_key'):
                self.providers['alpha_vantage'] = AlphaVantageProvider(alpha_config)
                logger.info("AlphaVantage provider initialized")
            
            # Setup YFinance provider
            yfinance_config = self.config.get('yfinance', {})
            self.providers['yfinance'] = YFinanceProvider(yfinance_config)
            logger.info("YFinance provider initialized")
            
            # Setup fallback provider
            fallback_config = self.config.get('fallback', {})
            self.providers['fallback'] = FallbackProvider(fallback_config)
            logger.info("Fallback provider initialized")
            
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
        # Check cache first
        if request.use_cache:
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
                
                provider = self.providers[provider_name]
                data = provider.fetch(
                    request.symbol,
                    request.interval,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                
                # Validate data
                if self._validate_data(data):
                    response = DataResponse(
                        data=data,
                        provider=provider_name,
                        cache_hit=False,
                        timestamp=datetime.now(),
                        metadata={'symbol': request.symbol, 'interval': request.interval}
                    )
                    
                    # Cache the response
                    if request.use_cache:
                        self._cache_data(request, response)
                    
                    # Log request history
                    self.request_history.append(request)
                    
                    logger.info(f"Successfully fetched data from {provider_name} for {request.symbol}")
                    return response
                else:
                    logger.warning(f"Invalid data received from {provider_name} for {request.symbol}")
                    
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {request.symbol}: {e}")
                continue
        
        # All providers failed
        error_msg = f"All data providers failed for {request.symbol}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
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
                results[request.symbol] = self.get_data(request)
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
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that the data is usable."""
        if data is None or data.empty:
            return False
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for reasonable data ranges
        if (data['high'] < data['low']).any():
            return False
        
        if (data['close'] < 0).any() or (data['volume'] < 0).any():
            return False
        
        return True
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        for name, provider in self.providers.items():
            try:
                status[name] = {
                    'enabled': provider.is_enabled(),
                    'metadata': provider.get_metadata(),
                    'last_error': getattr(provider, 'last_error', None)
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

# Convenience functions for backward compatibility
def get_data_provider(config: Optional[Dict[str, Any]] = None) -> DataProviderManager:
    """Get a configured data provider manager."""
    return DataProviderManager(config)

def fetch_data(symbol: str, interval: str = '1d', **kwargs) -> pd.DataFrame:
    """Fetch data for a single symbol (convenience function)."""
    provider = get_data_provider()
    request = DataRequest(symbol=symbol, interval=interval, **kwargs)
    response = provider.get_data(request)
    return response.data

def fetch_multiple_data(symbols: List[str], interval: str = '1d', **kwargs) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple symbols (convenience function)."""
    provider = get_data_provider()
    requests = [DataRequest(symbol=symbol, interval=interval, **kwargs) for symbol in symbols]
    responses = provider.get_multiple_data(requests)
    return {symbol: response.data for symbol, response in responses.items()} 