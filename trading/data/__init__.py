"""
Trading Data Management and Processing Module

This module provides comprehensive data management capabilities for the trading system,
including data providers, preprocessing, validation, and real-time data listening.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_listener import DataListener, MarketDataStream, RealTimeDataFeed
from .data_loader import (
    DataLoader,
    get_latest_price,
    load_market_data,
    load_multiple_tickers,
)
from .data_provider import DataProvider, DataProviderConfig
from .macro_data_integration import EconomicIndicatorLoader, MacroDataIntegrator
from .preprocessing import (
    DataPreprocessor,
    DataScaler,
    DataValidator,
    FeatureEngineering,
)
from .providers.alpha_vantage_provider import AlphaVantageProvider
from .providers.base_provider import BaseDataProvider
from .providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)

# --- Provider Management ---


class DataProviderManager:
    """Manages data providers and provides unified access."""

    def __init__(self):
        """Initialize the provider manager."""
        self.providers: Dict[str, BaseDataProvider] = {}
        self.default_provider: Optional[str] = None
        self.provider_configs: Dict[str, Dict[str, Any]] = {}
        self._discover_providers()
        self._setup_default_provider()

    def _discover_providers(self):
        """Discover and register available data providers."""
        try:
            # Register known providers
            self.register_provider("yahoo", YFinanceProvider())
            self.register_provider("alpha_vantage", AlphaVantageProvider())

            # Discover additional providers in the providers directory
            providers_dir = Path(__file__).parent / "providers"
            if providers_dir.exists():
                for file_path in providers_dir.glob("*.py"):
                    if file_path.name.startswith("__"):
                        continue

                    try:
                        # Import the module
                        module_name = f"trading.data.providers.{file_path.stem}"
                        module = importlib.import_module(module_name)

                        # Look for provider classes
                        for name, obj in inspect.getmembers(module):
                            if (
                                inspect.isclass(obj)
                                and issubclass(obj, BaseDataProvider)
                                and obj != BaseDataProvider
                            ):
                                try:
                                    # Try to instantiate the provider
                                    provider_instance = obj()
                                    provider_name = (
                                        provider_instance.get_provider_name()
                                    )
                                    self.register_provider(
                                        provider_name, provider_instance
                                    )
                                    logger.info(f"Discovered provider: {provider_name}")

                                except Exception as e:
                                    logger.warning(
                                        f"Error instantiating provider {name}: {e}"
                                    )
                                    continue

                    except Exception as e:
                        logger.warning(f"Error loading provider from {file_path}: {e}")
                        continue

            logger.info(f"Discovered {len(self.providers)} data providers")

        except Exception as e:
            logger.error(f"Error discovering providers: {e}")

    def register_provider(self, name: str, provider: BaseDataProvider):
        """Register a data provider.

        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider
        logger.info(f"Registered provider: {name}")

    def get_provider(self, name: Optional[str] = None) -> Optional[BaseDataProvider]:
        """Get a data provider by name.

        Args:
            name: Provider name (uses default if None)

        Returns:
            Provider instance or None if not found
        """
        if name is None:
            name = self.default_provider

        if name is None:
            # Try to find any available provider
            if self.providers:
                name = list(self.providers.keys())[0]
                logger.warning(f"No default provider set, using: {name}")
            else:
                logger.error("No data providers available")
                return None

        provider = self.providers.get(name)
        if provider is None:
            logger.error(f"Provider '{name}' not found")
            return None

        return provider

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())

    def set_default_provider(self, name: str) -> bool:
        """Set the default data provider.

        Args:
            name: Provider name

        Returns:
            True if successful, False otherwise
        """
        if name in self.providers:
            self.default_provider = name
            logger.info(f"Set default provider to: {name}")
            return True
        else:
            logger.error(f"Provider '{name}' not found")
            return False

    def get_provider_config(self, name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider.

        Args:
            name: Provider name

        Returns:
            Provider configuration
        """
        return self.provider_configs.get(name, {})

    def set_provider_config(self, name: str, config: Dict[str, Any]) -> bool:
        """Set configuration for a specific provider.

        Args:
            name: Provider name
            config: Provider configuration

        Returns:
            True if successful, False otherwise
        """
        if name in self.providers:
            self.provider_configs[name] = config
            logger.info(f"Set configuration for provider: {name}")
            return True
        else:
            logger.error(f"Provider '{name}' not found")
            return False

    def _setup_default_provider(self):
        """Setup the default provider based on configuration."""
        try:
            # Try to load configuration
            import os

            default_provider = os.getenv("DEFAULT_DATA_PROVIDER", "yahoo")

            if default_provider in self.providers:
                self.default_provider = default_provider
                logger.info(
                    f"Set default provider from environment: {default_provider}"
                )
            elif self.providers:
                # Use first available provider
                self.default_provider = list(self.providers.keys())[0]
                logger.info(
                    f"No valid default provider in environment, using: {self.default_provider}"
                )
            else:
                logger.warning("No data providers available")

        except Exception as e:
            logger.error(f"Error setting up default provider: {e}")

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers.

        Returns:
            Dictionary containing provider status information
        """
        status = {
            "default_provider": self.default_provider,
            "available_providers": self.get_available_providers(),
            "provider_count": len(self.providers),
            "provider_details": {},
        }

        for name, provider in self.providers.items():
            try:
                provider_status = {
                    "name": name,
                    "class": provider.__class__.__name__,
                    "available": provider.is_available()
                    if hasattr(provider, "is_available")
                    else True,
                    "config": self.get_provider_config(name),
                }
                status["provider_details"][name] = provider_status
            except Exception as e:
                logger.warning(f"Error getting status for provider {name}: {e}")
                status["provider_details"][name] = {"name": name, "error": str(e)}

        return status


# --- Global Provider Manager ---
_provider_manager = None


def get_provider_manager() -> DataProviderManager:
    """Get the global provider manager instance."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = DataProviderManager()
    return _provider_manager


def get_data_provider(name: Optional[str] = None) -> Optional[BaseDataProvider]:
    """Universal function to get a data provider.

    Args:
        name: Provider name (uses default if None)

    Returns:
        Provider instance or None if not found
    """
    return get_provider_manager().get_provider(name)


def get_available_providers() -> List[str]:
    """Get list of available data providers."""
    return get_provider_manager().get_available_providers()


def set_default_provider(name: str) -> bool:
    """Set the default data provider.

    Args:
        name: Provider name

    Returns:
        True if successful, False otherwise
    """
    return get_provider_manager().set_default_provider(name)


def get_provider_status() -> Dict[str, Any]:
    """Get status of all data providers.

    Returns:
        Dictionary containing provider status information
    """
    return get_provider_manager().get_provider_status()


# --- Convenience Functions for Backward Compatibility ---


def get_default_provider() -> Optional[BaseDataProvider]:
    """Get the default data provider (backward compatibility).

    Returns:
        Default provider instance or None if not found
    """
    return get_data_provider()


def get_provider_by_name(name: str) -> Optional[BaseDataProvider]:
    """Get a specific data provider by name (backward compatibility).

    Args:
        name: Provider name

    Returns:
        Provider instance or None if not found
    """
    return get_data_provider(name)


__all__ = [
    # Core data components
    "DataProvider",
    "DataProviderConfig",
    "DataLoader",
    "DataListener",
    "RealTimeDataFeed",
    "MarketDataStream",
    # Preprocessing components
    "DataPreprocessor",
    "FeatureEngineering",
    "DataValidator",
    "DataScaler",
    # Data providers
    "BaseDataProvider",
    "AlphaVantageProvider",
    "YFinanceProvider",
    # Macro data integration
    "MacroDataIntegrator",
    "EconomicIndicatorLoader",
    # Provider management
    "DataProviderManager",
    "get_data_provider",
    "get_available_providers",
    "set_default_provider",
    "get_provider_status",
    "get_default_provider",
    "get_provider_by_name",
    # Utility functions
    "load_market_data",
    "load_multiple_tickers",
    "get_latest_price",
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Trading Data Management and Processing"
