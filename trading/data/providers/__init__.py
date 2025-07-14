"""
Data Providers Module for Market Data Fetching

This module provides a unified interface for fetching market data from various sources
including Alpha Vantage, YFinance, and fallback providers with automatic failover.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .alpha_vantage_provider import AlphaVantageProvider
from .base_provider import BaseDataProvider, ProviderConfig
from .fallback_provider import FallbackDataProvider, get_fallback_provider
from .yfinance_provider import YFinanceProvider

# Configure logging
logger = logging.getLogger(__name__)

# --- Provider Configuration ---


class ProviderManager:
    """Manages data provider initialization and configuration."""

    def __init__(self):
        """Initialize the provider manager."""
        self.providers: Dict[str, BaseDataProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all available data providers."""
        # Get API keys from environment
        alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")

        # YFinance configuration
        yfinance_config = ProviderConfig(
            name="yfinance",
            enabled=True,
            priority=1,
            rate_limit_per_minute=60,
            timeout_seconds=30,
            retry_attempts=3,
            custom_config={"delay": 1.0},
        )

        # Alpha Vantage configuration
        alpha_vantage_config = ProviderConfig(
            name="alpha_vantage",
            enabled=True,
            priority=2,
            rate_limit_per_minute=60,
            timeout_seconds=30,
            retry_attempts=3,
            custom_config={"delay": 1.0, "api_key": alpha_vantage_key},
        )

        # Initialize YFinance provider
        try:
            self.providers["yfinance"] = YFinanceProvider(yfinance_config)
            logger.info("YFinance provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YFinance provider: {e}")

        # Initialize Alpha Vantage provider
        if alpha_vantage_key:
            try:
                self.providers["alpha_vantage"] = AlphaVantageProvider(
                    alpha_vantage_config
                )
                logger.info("Alpha Vantage provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Alpha Vantage provider: {e}")
        else:
            logger.warning("Alpha Vantage API key not found, skipping initialization")

        # Initialize fallback provider
        try:
            fallback_provider = get_fallback_provider()
            if fallback_provider:
                self.providers["fallback"] = fallback_provider
                logger.info("Fallback provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize fallback provider: {e}")

    def get_provider(self, name: str) -> Optional[BaseDataProvider]:
        """Get a specific provider by name.

        Args:
            name: Provider name ('yfinance', 'alpha_vantage', 'fallback')

        Returns:
            Provider instance or None if not available
        """
        return self.providers.get(name)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names.

        Returns:
            List of available provider names
        """
        return list(self.providers.keys())

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all providers.

        Returns:
            Dictionary with provider status information
        """
        status = {}
        for name, provider in self.providers.items():
            try:
                status[name] = provider.get_metadata()
            except Exception as e:
                logger.error(f"Error getting status for {name}: {e}")
                status[name] = {"error": str(e)}
        return status

    def is_provider_available(self, name: str) -> bool:
        """Check if a provider is available and enabled.

        Args:
            name: Provider name

        Returns:
            True if provider is available and enabled
        """
        provider = self.providers.get(name)
        return provider is not None and provider.is_enabled()


# --- Data Loading Functions ---


class DataLoader:
    """Unified data loading interface with automatic provider selection."""

    def __init__(self, provider_manager: ProviderManager):
        """Initialize the data loader.

        Args:
            provider_manager: Provider manager instance
        """
        self.provider_manager = provider_manager

    def load_data(
        self,
        symbol: str,
        source: str = "auto",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
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
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        if source == "auto":
            return self._load_data_auto(symbol, interval, **kwargs)
        else:
            return self._load_data_from_source(symbol, source, interval, **kwargs)

    def _load_data_auto(self, symbol: str, interval: str, **kwargs) -> pd.DataFrame:
        """Load data using automatic provider selection.

        Args:
            symbol: Stock symbol
            interval: Data interval
            **kwargs: Additional parameters

        Returns:
            DataFrame with market data

        Raises:
            RuntimeError: If all providers fail
        """
        # Try providers in order of preference
        provider_order = ["alpha_vantage", "yfinance", "fallback"]

        for provider_name in provider_order:
            provider = self.provider_manager.get_provider(provider_name)
            if provider is None or not provider.is_enabled():
                continue

            try:
                logger.info(f"Attempting to load {symbol} from {provider_name}")
                return provider.fetch(symbol, interval, **kwargs)
            except Exception as e:
                logger.warning(f"{provider_name} failed for {symbol}: {e}")
                continue

        raise RuntimeError(f"All providers failed for {symbol}")

    def _load_data_from_source(
        self, symbol: str, source: str, interval: str, **kwargs
    ) -> pd.DataFrame:
        """Load data from a specific source.

        Args:
            symbol: Stock symbol
            source: Data source name
            interval: Data interval
            **kwargs: Additional parameters

        Returns:
            DataFrame with market data

        Raises:
            RuntimeError: If provider is not available
            ValueError: If source is invalid
        """
        provider = self.provider_manager.get_provider(source)
        if provider is None:
            raise RuntimeError(f"{source} provider not available")

        if not provider.is_enabled():
            raise RuntimeError(f"{source} provider is disabled")

        return provider.fetch(symbol, interval, **kwargs)

    def load_multiple_data(
        self,
        symbols: List[str],
        source: str = "auto",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
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
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        if source == "auto":
            return self._load_multiple_data_auto(symbols, interval, **kwargs)
        else:
            return self._load_multiple_data_from_source(
                symbols, source, interval, **kwargs
            )

    def _load_multiple_data_auto(
        self, symbols: List[str], interval: str, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple symbols using automatic provider selection.

        Args:
            symbols: List of stock symbols
            interval: Data interval
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        # Try fallback provider first for multiple symbols
        fallback_provider = self.provider_manager.get_provider("fallback")
        if fallback_provider and fallback_provider.is_enabled():
            try:
                return fallback_provider.fetch_multiple(symbols, interval, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback provider failed for multiple symbols: {e}")

        # Manual fallback logic
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self._load_data_auto(symbol, interval, **kwargs)
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                continue

        return results

    def _load_multiple_data_from_source(
        self, symbols: List[str], source: str, interval: str, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple symbols from a specific source.

        Args:
            symbols: List of stock symbols
            source: Data source name
            interval: Data interval
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping symbols to DataFrames

        Raises:
            RuntimeError: If provider is not available
            ValueError: If source is invalid
        """
        provider = self.provider_manager.get_provider(source)
        if provider is None:
            raise RuntimeError(f"{source} provider not available")

        if not provider.is_enabled():
            raise RuntimeError(f"{source} provider is disabled")

        return provider.fetch_multiple(symbols, interval, **kwargs)


# --- Global Instances ---
_provider_manager = ProviderManager()
_data_loader = DataLoader(_provider_manager)

# --- Convenience Functions ---


def load_data(
    symbol: str,
    source: str = "auto",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
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
    return _data_loader.load_data(symbol, source, start_date, end_date, interval)


def load_multiple_data(
    symbols: List[str],
    source: str = "auto",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
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
    return _data_loader.load_multiple_data(
        symbols, source, start_date, end_date, interval
    )


def get_available_providers() -> Dict[str, bool]:
    """Get status of available providers.

    Returns:
        Dictionary mapping provider names to availability status
    """
    return {
        name: _provider_manager.is_provider_available(name)
        for name in ["alpha_vantage", "yfinance", "fallback"]
    }


def get_provider_status() -> Dict[str, Dict[str, Any]]:
    """Get detailed status of all providers.

    Returns:
        Dictionary with provider status information
    """
    return _provider_manager.get_provider_status()


def get_provider_manager() -> ProviderManager:
    """Get the global provider manager instance.

    Returns:
        Global provider manager instance
    """
    return _provider_manager


def get_data_loader() -> DataLoader:
    """Get the global data loader instance.

    Returns:
        Global data loader instance
    """
    return _data_loader


# --- Exports ---
__all__ = [
    # Core classes
    "ProviderManager",
    "DataLoader",
    "BaseDataProvider",
    "ProviderConfig",
    # Provider implementations
    "AlphaVantageProvider",
    "YFinanceProvider",
    "FallbackDataProvider",
    "get_fallback_provider",
    # Convenience functions
    "load_data",
    "load_multiple_data",
    "get_available_providers",
    "get_provider_status",
    "get_provider_manager",
    "get_data_loader",
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Data Providers for Market Data Fetching"
