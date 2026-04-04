"""Data Module for Evolve Trading Platform.

This module provides data loading, processing, and streaming capabilities.
"""

from .streaming_pipeline import (
    DataProvider,
    DataTrigger,
    InMemoryCache,
    MarketData,
    PolygonDataProvider,
    StreamingConfig,
    StreamingPipeline,
    YFinanceDataProvider,
    create_streaming_pipeline,
)

__all__ = [
    "StreamingPipeline",
    "InMemoryCache",
    "DataProvider",
    "PolygonDataProvider",
    "YFinanceDataProvider",
    "MarketData",
    "StreamingConfig",
    "DataTrigger",
    "create_streaming_pipeline",
]
