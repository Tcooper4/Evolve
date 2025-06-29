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

logger = logging.getLogger(__name__)

class FallbackDataProvider:
    """Data provider with fallback logic for multiple sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fallback data provider.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.providers = []
        self.fallback_log = []
        
        # Initialize providers in order of preference
        self._initialize_providers()
        
        logger.info(f"Fallback data provider initialized with {len(self.providers)} providers")
    
    def _initialize_providers(self):
        """Initialize data providers in fallback order."""
        try:
            # Try YFinance first (free, reliable)
            yfinance_provider = YFinanceProvider(self.config.get('yfinance', {}))
            self.providers.append(('yfinance', yfinance_provider))
            logger.info("YFinance provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize YFinance provider: {e}")
        
        try:
            # Try Alpha Vantage as backup
            alpha_config = self.config.get('alpha_vantage', {})
            if alpha_config.get('api_key'):
                alpha_provider = AlphaVantageProvider(alpha_config)
                self.providers.append(('alpha_vantage', alpha_provider))
                logger.info("Alpha Vantage provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Alpha Vantage provider: {e}")
        
        # Add mock provider as final fallback
        mock_provider = MockDataProvider(self.config.get('mock', {}))
        self.providers.append(('mock', mock_provider))
        logger.info("Mock provider initialized as final fallback")
    
    def get_historical_data(self, 
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           interval: str = '1d') -> pd.DataFrame:
        """Get historical data with fallback logic.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with historical data
        """
        for provider_name, provider in self.providers:
            try:
                logger.info(f"Trying {provider_name} for {symbol}")
                data = provider.get_historical_data(symbol, start_date, end_date, interval)
                
                if data is not None and not data.empty:
                    self._log_success(provider_name, symbol)
                    return data
                else:
                    logger.warning(f"{provider_name} returned empty data for {symbol}")
                    
            except Exception as e:
                logger.warning(f"{provider_name} failed for {symbol}: {e}")
                self._log_failure(provider_name, symbol, str(e))
                continue
        
        # If all providers fail, return mock data
        logger.error(f"All providers failed for {symbol}, using mock data")
        mock_provider = self.providers[-1][1]  # Last provider is mock
        return mock_provider.get_historical_data(symbol, start_date, end_date, interval)
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price with fallback logic.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None
        """
        for provider_name, provider in self.providers:
            try:
                price = provider.get_live_price(symbol)
                if price is not None and price > 0:
                    self._log_success(provider_name, symbol, "live_price")
                    return price
            except Exception as e:
                logger.warning(f"{provider_name} failed for live price of {symbol}: {e}")
                self._log_failure(provider_name, symbol, str(e), "live_price")
                continue
        
        # Return mock price as fallback
        logger.error(f"All providers failed for live price of {symbol}, using mock price")
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
                logger.error(f"Failed to get market data for {symbol}: {e}")
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
        
        # Provider stats
        provider_stats = {}
        for log in self.fallback_log:
            provider = log['provider']
            if provider not in provider_stats:
                provider_stats[provider] = {'success': 0, 'failure': 0}
            
            if log['status'] == 'success':
                provider_stats[provider]['success'] += 1
            else:
                provider_stats[provider]['failure'] += 1
        
        # Recent failures (last 10)
        recent_failures = [
            log for log in self.fallback_log[-10:] 
            if log['status'] == 'failure'
        ]
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'provider_stats': provider_stats,
            'recent_failures': recent_failures
        }
    
    def save_fallback_log(self, filepath: str):
        """Save fallback log to file.
        
        Args:
            filepath: Path to save log file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.fallback_log, f, indent=2, default=str)
            logger.info(f"Fallback log saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save fallback log: {e}")

class MockDataProvider:
    """Mock data provider for fallback scenarios."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mock data provider."""
        self.config = config or {}
        self.base_price = self.config.get('base_price', 100.0)
        self.volatility = self.config.get('volatility', 0.02)
        
        logger.info("Mock data provider initialized")
    
    def get_historical_data(self, 
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           interval: str = '1d') -> pd.DataFrame:
        """Generate mock historical data.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with mock historical data
        """
        logger.warning(f"Generating mock historical data for {symbol}")
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price data with random walk
        np.random.seed(hash(symbol) % 2**32)  # Deterministic for same symbol
        
        prices = [self.base_price]
        for _ in range(len(date_range) - 1):
            change = np.random.normal(0, self.volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Ensure positive price
        
        # Generate other data
        volumes = np.random.uniform(1000000, 10000000, len(date_range))
        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        opens = [p * (1 + np.random.normal(0, 0.005)) for p in prices]
        
        # Create DataFrame
        data = pd.DataFrame({
            'Date': date_range,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        })
        
        data.set_index('Date', inplace=True)
        
        return data
    
    def get_live_price(self, symbol: str) -> float:
        """Get mock live price.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Mock current price
        """
        logger.warning(f"Generating mock live price for {symbol}")
        
        # Generate price based on symbol hash for consistency
        np.random.seed(hash(symbol) % 2**32)
        price = self.base_price * (1 + np.random.normal(0, self.volatility))
        return max(price, 0.01)

# Global fallback provider instance
fallback_provider = FallbackDataProvider()

def get_fallback_provider() -> FallbackDataProvider:
    """Get the global fallback data provider instance."""
    return fallback_provider 