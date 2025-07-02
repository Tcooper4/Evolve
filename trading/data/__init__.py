"""
Trading Data Management and Processing Module

This module provides comprehensive data management capabilities for the trading system,
including data providers, preprocessing, validation, and real-time data listening.
"""

from .preprocessing import (
    DataPreprocessor,
    FeatureEngineering,
    DataValidator,
    DataScaler
)
from .data_provider import DataProvider, DataProviderConfig
from .data_loader import (
    load_market_data,
    load_multiple_tickers,
    get_latest_price,
    DataLoader
)
from .data_listener import (
    DataListener,
    RealTimeDataFeed,
    MarketDataStream
)
from .macro_data_integration import (
    MacroDataIntegrator,
    EconomicIndicatorLoader
)
from .providers.alpha_vantage_provider import AlphaVantageProvider
from .providers.yfinance_provider import YFinanceProvider
from .providers.base_provider import BaseDataProvider

__all__ = [
    # Core data components
    'DataProvider',
    'DataProviderConfig',
    'DataLoader',
    'DataListener',
    'RealTimeDataFeed',
    'MarketDataStream',
    
    # Preprocessing components
    'DataPreprocessor',
    'FeatureEngineering',
    'DataValidator',
    'DataScaler',
    
    # Data providers
    'BaseDataProvider',
    'AlphaVantageProvider',
    'YFinanceProvider',
    
    # Macro data integration
    'MacroDataIntegrator',
    'EconomicIndicatorLoader',
    
    # Utility functions
    'load_market_data',
    'load_multiple_tickers',
    'get_latest_price'
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Trading Data Management and Processing" 