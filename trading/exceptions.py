"""
Trading system exceptions.

This module contains all custom exceptions used throughout the trading system.
"""


class TradingError(Exception):
    """Base class for all trading system errors."""
    pass


class ValidationError(TradingError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(TradingError):
    """Raised when configuration is invalid or missing."""
    pass


class DataError(TradingError):
    """Raised when there are issues with data loading or processing."""
    pass


class ModelError(TradingError):
    """Raised when there are issues with model training or prediction."""
    pass


class ExecutionError(TradingError):
    """Raised when trade execution fails."""
    pass


class MarketDataError(TradingError):
    """Raised when there are issues with market data."""
    pass


class StrategyError(TradingError):
    """Raised when there are issues with strategy execution."""
    pass


class StrategyNotFoundError(StrategyError):
    """Raised when a requested strategy is not found."""
    pass


class StrategyValidationError(StrategyError):
    """Raised when strategy validation fails."""
    pass


class OptimizationError(TradingError):
    """Raised when optimization processes fail."""
    pass


class PerformanceError(TradingError):
    """Raised when performance calculations fail."""
    pass


class GoalError(TradingError):
    """Raised when goal-related operations fail."""
    pass


class AgentError(TradingError):
    """Raised when agent operations fail."""
    pass


class MemoryError(TradingError):
    """Raised when memory operations fail."""
    pass


class LoggingError(TradingError):
    """Raised when logging operations fail."""
    pass


class VisualizationError(TradingError):
    """Raised when visualization operations fail."""
    pass


class ImportError(TradingError):
    """Raised when module imports fail."""
    pass


class NetworkError(TradingError):
    """Raised when network operations fail."""
    pass


class TimeoutError(TradingError):
    """Raised when operations timeout."""
    pass


class ResourceError(TradingError):
    """Raised when resource allocation fails."""
    pass


class SecurityError(TradingError):
    """Raised when security checks fail."""
    pass 