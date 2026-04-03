"""
Trading Data Management and Processing Module

Import submodules directly for heavy types (e.g. ``trading.data.data_listener``).
Provider discovery lives in ``trading.data.provider_manager`` to avoid loading
providers on every ``import trading.data``.
"""

from .data_loader import (
    DataLoader,
    get_latest_price,
    load_market_data,
    load_multiple_tickers,
)
from .preprocessing import (
    DataPreprocessor,
    DataScaler,
    DataValidator,
    FeatureEngineering,
)
from .provider_manager import (
    DataProviderManager,
    get_available_providers,
    get_data_provider,
    get_default_provider,
    get_provider_by_name,
    get_provider_manager,
    get_provider_status,
    set_default_provider,
)

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "FeatureEngineering",
    "DataValidator",
    "DataScaler",
    "load_market_data",
    "load_multiple_tickers",
    "get_latest_price",
    "DataProviderManager",
    "get_data_provider",
    "get_available_providers",
    "set_default_provider",
    "get_provider_status",
    "get_default_provider",
    "get_provider_by_name",
    "get_provider_manager",
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Trading Data Management and Processing"
