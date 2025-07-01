"""Data providers for fetching market data from various sources."""

import os
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
from .base_provider import BaseDataProvider, ProviderConfig
from .alpha_vantage_provider import AlphaVantageProvider
from .yfinance_provider import YFinanceProvider
from .fallback_provider import FallbackDataProvider, get_fallback_provider

# Initialize providers with proper configuration
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Create provider configurations
yfinance_config = ProviderConfig(
    name="yfinance",
    enabled=True,
    priority=1,
    rate_limit_per_minute=60,
    timeout_seconds=30,
    retry_attempts=3,
    custom_config={"delay": 1.0}
)

alpha_vantage_config = ProviderConfig(
    name="alpha_vantage",
    enabled=True,
    priority=2,
    rate_limit_per_minute=60,
    timeout_seconds=30,
    retry_attempts=3,
    custom_config={"delay": 1.0, "api_key": ALPHA_VANTAGE_KEY}
)

# Initialize providers
try:
    yfinance = YFinanceProvider(yfinance_config)
except Exception as e:
    print(f"Failed to initialize YFinance provider: {e}")
    yfinance = None

try:
    if ALPHA_VANTAGE_KEY:
        alpha_vantage = AlphaVantageProvider(alpha_vantage_config)
    else:
        print("Alpha Vantage API key not found, skipping initialization")
        alpha_vantage = None
except Exception as e:
    print(f"Failed to initialize Alpha Vantage provider: {e}")
    alpha_vantage = None

# Initialize fallback provider
try:
    fallback = get_fallback_provider()
except Exception as e:
    print(f"Failed to initialize fallback provider: {e}")
    fallback = None

def load_data(
    symbol: str,
    source: str = 'auto',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d'
) -> pd.DataFrame:
    """Load data from specified or auto-selected source.
    
    Args:
        symbol: Stock symbol
        source: Data source ('auto', 'alpha_vantage', 'yfinance', or 'fallback')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, etc.)
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        ValueError: If source is invalid
        RuntimeError: If data loading fails
    """
    kwargs = {}
    if start_date:
        kwargs['start_date'] = start_date
    if end_date:
        kwargs['end_date'] = end_date
    
    if source == 'auto':
        # Try providers in order of preference
        providers = [
            ('alpha_vantage', alpha_vantage),
            ('yfinance', yfinance),
            ('fallback', fallback)
        ]
        
        for provider_name, provider in providers:
            if provider is None:
                continue
            try:
                return provider.fetch(symbol, interval, **kwargs)
            except Exception as e:
                print(f"{provider_name} failed for {symbol}: {e}")
                continue
        
        raise RuntimeError(f"All providers failed for {symbol}")
        
    elif source == 'alpha_vantage':
        if alpha_vantage is None:
            raise RuntimeError("Alpha Vantage provider not available")
        return alpha_vantage.fetch(symbol, interval, **kwargs)
        
    elif source == 'yfinance':
        if yfinance is None:
            raise RuntimeError("YFinance provider not available")
        return yfinance.fetch(symbol, interval, **kwargs)
        
    elif source == 'fallback':
        if fallback is None:
            raise RuntimeError("Fallback provider not available")
        return fallback.fetch(symbol, interval, **kwargs)
        
    else:
        raise ValueError(f"Invalid source: {source}")

def load_multiple_data(
    symbols: list[str],
    source: str = 'auto',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d'
) -> Dict[str, pd.DataFrame]:
    """Load data for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        source: Data source ('auto', 'alpha_vantage', 'yfinance', or 'fallback')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, etc.)
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    kwargs = {}
    if start_date:
        kwargs['start_date'] = start_date
    if end_date:
        kwargs['end_date'] = end_date
    
    if source == 'auto':
        if fallback is not None:
            return fallback.fetch_multiple(symbols, interval, **kwargs)
        else:
            # Manual fallback logic
            results = {}
            for symbol in symbols:
                try:
                    results[symbol] = load_data(symbol, 'auto', start_date, end_date, interval)
                except Exception as e:
                    print(f"Failed to load {symbol}: {e}")
                    continue
            return results
            
    elif source == 'alpha_vantage':
        if alpha_vantage is None:
            raise RuntimeError("Alpha Vantage provider not available")
        return alpha_vantage.fetch_multiple(symbols, interval, **kwargs)
        
    elif source == 'yfinance':
        if yfinance is None:
            raise RuntimeError("YFinance provider not available")
        return yfinance.fetch_multiple(symbols, interval, **kwargs)
        
    elif source == 'fallback':
        if fallback is None:
            raise RuntimeError("Fallback provider not available")
        return fallback.fetch_multiple(symbols, interval, **kwargs)
        
    else:
        raise ValueError(f"Invalid source: {source}")

def get_available_providers() -> Dict[str, bool]:
    """Get status of available providers.
    
    Returns:
        Dictionary mapping provider names to availability status
    """
    return {
        'alpha_vantage': alpha_vantage is not None and alpha_vantage.is_enabled(),
        'yfinance': yfinance is not None and yfinance.is_enabled(),
        'fallback': fallback is not None and fallback.is_enabled()
    }

def get_provider_status() -> Dict[str, Dict[str, Any]]:
    """Get detailed status of all providers.
    
    Returns:
        Dictionary with provider status information
    """
    status = {}
    
    if alpha_vantage is not None:
        status['alpha_vantage'] = alpha_vantage.get_metadata()
    if yfinance is not None:
        status['yfinance'] = yfinance.get_metadata()
    if fallback is not None:
        status['fallback'] = fallback.get_metadata()
        
    return status

__all__ = [
    'load_data', 
    'load_multiple_data', 
    'get_available_providers',
    'get_provider_status',
    'BaseDataProvider', 
    'ProviderConfig',
    'AlphaVantageProvider', 
    'YFinanceProvider',
    'FallbackDataProvider',
    'get_fallback_provider'
] 