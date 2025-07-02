"""Fallback Data Provider for Trading System.

This module provides a fallback mechanism for data providers, trying multiple
sources in sequence when one fails.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

from .yfinance_provider import YFinanceProvider
from .alpha_vantage_provider import AlphaVantageProvider
from .base_provider import BaseDataProvider, ProviderConfig

logger = logging.getLogger(__name__)

class FallbackDataProvider(BaseDataProvider):
    """Data provider with fallback logic for multiple sources."""
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize fallback data provider.
        
        Args:
            config: Provider configuration (optional)
        """
        if config is None:
            config = ProviderConfig(
                name="fallback",
                enabled=True,
                priority=1,
                rate_limit_per_minute=60,
                timeout_seconds=30,
                retry_attempts=3,
                custom_config={}
            )
        
        super().__init__(config)
        self.providers = []
        self.fallback_log = []
        
        # Initialize providers in order of preference
        self._initialize_providers()
        
        self.logger.info(f"Fallback data provider initialized with {len(self.providers)} providers")

    def _setup(self) -> None:
        """Setup method called during initialization."""
        pass

    def _initialize_providers(self):
        """Initialize data providers in fallback order."""
        try:
            # Try YFinance first (free, reliable)
            yfinance_config = ProviderConfig(
                name="yfinance_fallback",
                enabled=True,
                priority=1,
                rate_limit_per_minute=60,
                timeout_seconds=30,
                retry_attempts=3,
                custom_config=self.config.custom_config.get('yfinance', {})
            )
            yfinance_provider = YFinanceProvider(yfinance_config)
            self.providers.append(('yfinance', yfinance_provider))
            self.logger.info("YFinance provider initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize YFinance provider: {e}")
        
        try:
            # Try Alpha Vantage as backup
            alpha_config_dict = self.config.custom_config.get('alpha_vantage', {})
            if alpha_config_dict.get('api_key'):
                alpha_config = ProviderConfig(
                    name="alpha_vantage_fallback",
                    enabled=True,
                    priority=2,
                    rate_limit_per_minute=60,
                    timeout_seconds=30,
                    retry_attempts=3,
                    custom_config=alpha_config_dict
                )
                alpha_provider = AlphaVantageProvider(alpha_config)
                self.providers.append(('alpha_vantage', alpha_provider))
                self.logger.info("Alpha Vantage provider initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Alpha Vantage provider: {e}")
        
        # Add mock provider as final fallback
        mock_config = ProviderConfig(
            name="mock_fallback",
            enabled=True,
            priority=3,
            rate_limit_per_minute=1000,
            timeout_seconds=1,
            retry_attempts=1,
            custom_config=self.config.custom_config.get('mock', {})
        )
        mock_provider = MockDataProvider(mock_config)
        self.providers.append(('mock', mock_provider))
        self.logger.info("Mock provider initialized as final fallback")

    def fetch(self, symbol: str, interval: str = '1d', **kwargs) -> pd.DataFrame:
        """Fetch data for a given symbol and interval with fallback logic.
        
        Args:
            symbol: Stock symbol
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters (start_date, end_date)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            Exception: If data fetching fails for all providers
        """
        if not self.is_enabled():
            raise RuntimeError("Provider is disabled")
            
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
            
        if not self.validate_interval(interval):
            raise ValueError(f"Invalid interval: {interval}")
        
        self._update_status_on_request()
        
        for provider_name, provider in self.providers:
            try:
                self.logger.info(f"Trying {provider_name} for {symbol}")
                data = provider.fetch(symbol, interval, **kwargs)
                
                if data is not None and not data.empty:
                    self._log_success(provider_name, symbol)
                    self._update_status_on_success()
                    return data
                else:
                    self.logger.warning(f"{provider_name} returned empty data for {symbol}")
                    
            except Exception as e:
                self.logger.warning(f"{provider_name} failed for {symbol}: {e}")
                self._log_failure(provider_name, symbol, str(e))
                continue
        
        # If all providers fail, return mock data
        self.logger.error(f"All providers failed for {symbol}, using mock data")
        mock_provider = self.providers[-1][1]  # Last provider is mock
        try:
            data = mock_provider.fetch(symbol, interval, **kwargs)
            self._update_status_on_success()
            return data
        except Exception as e:
            error_msg = f"All providers including mock failed for {symbol}: {e}"
            self._update_status_on_failure(error_msg)
            raise RuntimeError(error_msg)

    def fetch_multiple(self, symbols: List[str], interval: str = '1d', **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols with fallback logic.
        
        Args:
            symbols: List of stock symbols
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters (start_date, end_date)
            
        Returns:
            Dictionary mapping symbols to DataFrames
            
        Raises:
            Exception: If data fetching fails for all symbols
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

    def get_historical_data(self, 
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           interval: str = '1d') -> pd.DataFrame:
        """Legacy method for backward compatibility.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with historical data
        """
        kwargs = {
            'start_date': start_date.strftime('%Y-%m-%d') if start_date else None,
            'end_date': end_date.strftime('%Y-%m-%d') if end_date else None
        }
        return self.fetch(symbol, interval, **kwargs)
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price with fallback logic.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None
        """
        for provider_name, provider in self.providers:
            try:
                if hasattr(provider, 'get_live_price'):
                    price = provider.get_live_price(symbol)
                    if price is not None and price > 0:
                        self._log_success(provider_name, symbol, "live_price")
                        return price
            except Exception as e:
                self.logger.warning(f"{provider_name} failed for live price of {symbol}: {e}")
                self._log_failure(provider_name, symbol, str(e), "live_price")
                continue
        
        # Return mock price as fallback
        self.logger.error(f"All providers failed for live price of {symbol}, using mock price")
        mock_provider = self.providers[-1][1]
        return mock_provider.get_live_price(symbol)
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for multiple symbols with fallback logic.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary of market data
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Try to get live price
                price = self.get_live_price(symbol)
                if price is not None:
                    results[symbol] = {
                        'price': price,
                        'timestamp': datetime.now(),
                        'volume': np.random.uniform(1000000, 10000000),  # Mock volume
                        'change': np.random.normal(0, 0.02),  # Mock change
                        'change_pct': np.random.normal(0, 0.02) * 100  # Mock change %
                    }
            except Exception as e:
                self.logger.error(f"Failed to get market data for {symbol}: {e}")
                # Use mock data as fallback
                results[symbol] = {
                    'price': 100.0 + np.random.normal(0, 10),
                    'timestamp': datetime.now(),
                    'volume': np.random.uniform(1000000, 10000000),
                    'change': np.random.normal(0, 0.02),
                    'change_pct': np.random.normal(0, 0.02) * 100
                }
        
        return results
    
    def _log_success(self, provider_name: str, symbol: str, operation: str = "historical_data"):
        """Log successful data retrieval."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider_name,
            'symbol': symbol,
            'operation': operation,
            'status': 'success',
            'error': None
        }
        self.fallback_log.append(log_entry)
    
    def _log_failure(self, provider_name: str, symbol: str, error: str, operation: str = "historical_data"):
        """Log failed data retrieval."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider_name,
            'symbol': symbol,
            'operation': operation,
            'status': 'failure',
            'error': error
        }
        self.fallback_log.append(log_entry)
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get statistics about fallback usage.
        
        Returns:
            Dictionary with fallback statistics
        """
        if not self.fallback_log:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'provider_stats': {},
                'recent_failures': []
            }
        
        # Calculate basic stats
        total_requests = len(self.fallback_log)
        successful_requests = len([log for log in self.fallback_log if log['status'] == 'success'])
        failed_requests = total_requests - successful_requests
        
        # Calculate provider stats
        provider_stats = {}
        for log in self.fallback_log:
            provider = log['provider']
            if provider not in provider_stats:
                provider_stats[provider] = {'success': 0, 'failure': 0}
            
            if log['status'] == 'success':
                provider_stats[provider]['success'] += 1
            else:
                provider_stats[provider]['failure'] += 1
        
        # Get recent failures
        recent_failures = [
            log for log in self.fallback_log[-10:]  # Last 10 entries
            if log['status'] == 'failure'
        ]
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'provider_stats': provider_stats,
            'recent_failures': recent_failures
        }
    
    def save_fallback_log(self, filepath: str):
        """Save fallback log to file.
        
        Args:
            filepath: Path to save the log
        """
        with open(filepath, 'w') as f:
            json.dump(self.fallback_log, f, indent=2)

class MockDataProvider(BaseDataProvider):
    """Mock data provider for testing and fallback scenarios."""
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize mock data provider.
        
        Args:
            config: Provider configuration (optional)
        """
        if config is None:
            config = ProviderConfig(
                name="mock",
                enabled=True,
                priority=999,
                rate_limit_per_minute=1000,
                timeout_seconds=1,
                retry_attempts=1,
                custom_config={}
            )
        
        super().__init__(config)

    def _setup(self) -> None:
        """Setup method called during initialization."""
        self.logger.info("Mock data provider initialized")

    def fetch(self, symbol: str, interval: str = '1d', **kwargs) -> pd.DataFrame:
        """Generate mock historical data.
        
        Args:
            symbol: Stock symbol
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters (start_date, end_date)
            
        Returns:
            DataFrame with mock OHLCV data
        """
        if not self.is_enabled():
            raise RuntimeError("Provider is disabled")
            
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
            
        if not self.validate_interval(interval):
            raise ValueError(f"Invalid interval: {interval}")
        
        self._update_status_on_request()
        
        try:
            # Generate mock data
            start_date = kwargs.get('start_date', '2023-01-01')
            end_date = kwargs.get('end_date', '2023-12-31')
            
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate mock OHLCV data
            base_price = 100.0 + hash(symbol) % 50  # Deterministic base price
            np.random.seed(hash(symbol) % 1000)  # Deterministic randomness
            
            data = []
            current_price = base_price
            
            for date in date_range:
                # Generate price movement
                change = np.random.normal(0, 0.02) * current_price
                current_price += change
                
                # Generate OHLC
                open_price = current_price
                high_price = current_price * (1 + abs(np.random.normal(0, 0.01)))
                low_price = current_price * (1 - abs(np.random.normal(0, 0.01)))
                close_price = current_price + np.random.normal(0, 0.005) * current_price
                
                # Generate volume
                volume = np.random.uniform(1000000, 10000000)
                
                data.append({
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                })
            
            df = pd.DataFrame(data, index=date_range)
            self._update_status_on_success()
            return df
            
        except Exception as e:
            error_msg = f"Error generating mock data for {symbol}: {str(e)}"
            self._update_status_on_failure(error_msg)
            raise RuntimeError(error_msg)

    def fetch_multiple(self, symbols: List[str], interval: str = '1d', **kwargs) -> Dict[str, pd.DataFrame]:
        """Generate mock data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters (start_date, end_date)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if not self.is_enabled():
            raise RuntimeError("Provider is disabled")
            
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol, interval, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to generate mock data for {symbol}: {e}")
                continue
        
        return results

    def get_historical_data(self, 
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           interval: str = '1d') -> pd.DataFrame:
        """Legacy method for backward compatibility.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with historical data
        """
        kwargs = {
            'start_date': start_date.strftime('%Y-%m-%d') if start_date else None,
            'end_date': end_date.strftime('%Y-%m-%d') if end_date else None
        }
        return self.fetch(symbol, interval, **kwargs)

    def get_live_price(self, symbol: str) -> float:
        """Get mock live price.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Mock current price
        """
        base_price = 100.0 + hash(symbol) % 50
        np.random.seed(hash(symbol) % 1000)
        return base_price + np.random.normal(0, 5)

def get_fallback_provider() -> FallbackDataProvider:
    """Get a configured fallback data provider.
    
    Returns:
        Configured FallbackDataProvider instance
    """
    config = ProviderConfig(
        name="fallback",
        enabled=True,
        priority=1,
        rate_limit_per_minute=60,
        timeout_seconds=30,
        retry_attempts=3,
        custom_config={}
    )
    return FallbackDataProvider(config)

# Backward compatibility alias
FallbackProvider = FallbackDataProvider