"""
Core Interfaces Module

This module defines the abstract interfaces that all system components must implement.
These interfaces ensure consistent behavior and enable dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# Base Data Types
# =============================================================================

@dataclass
class DataRequest:
    """Request for data from a data provider."""
    symbol: str
    period: str
    interval: str = "1d"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    fields: Optional[List[str]] = None

@dataclass
class DataResponse:
    """Response from a data provider."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""
    model_type: str
    parameters: Dict[str, Any]
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    random_state: int = 42

@dataclass
class TrainingResult:
    """Result of model training."""
    model: Any
    metrics: Dict[str, float]
    training_time: float
    validation_metrics: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None

@dataclass
class PredictionResult:
    """Result of model prediction."""
    predictions: np.ndarray
    metadata: Dict[str, Any]
    model_info: Dict[str, Any]
    confidence_intervals: Optional[np.ndarray] = None

@dataclass
class SignalData:
    """Trading signal data."""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class TradeSignal:
    """Complete trading signal."""
    signal: SignalData
    strategy_name: str
    parameters: Dict[str, Any]
    market_conditions: Dict[str, Any]

class EventType(Enum):
    """Types of system events."""
    DATA_LOADED = "data_loaded"
    MODEL_TRAINED = "model_trained"
    SIGNAL_GENERATED = "signal_generated"
    TRADE_EXECUTED = "trade_executed"
    RISK_ALERT = "risk_alert"
    SYSTEM_ERROR = "system_error"

@dataclass
class SystemEvent:
    """System event for event-driven architecture."""
    event_type: EventType
    timestamp: datetime
    data: Any
    source: str
    metadata: Optional[Dict[str, Any]] = None

# =============================================================================
# Data Provider Interface
# =============================================================================

class IDataProvider(ABC):
    """Abstract interface for data providers."""
    
    @abstractmethod
    def get_data(self, request: DataRequest) -> DataResponse:
        """Get market data for the specified request."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
    
    @abstractmethod
    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about available data for a symbol."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data provider is available."""
        pass

# =============================================================================
# Model Interface
# =============================================================================

class IModel(ABC):
    """Abstract interface for machine learning models."""
    
    @abstractmethod
    def train(self, data: pd.DataFrame, config: ModelConfig) -> TrainingResult:
        """Train the model with the given data and configuration."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make predictions on the given data."""
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """Save the model to the specified path."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """Load the model from the specified path."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

# =============================================================================
# Strategy Interface
# =============================================================================

class IStrategy(ABC):
    """Abstract interface for trading strategies."""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Set strategy parameters."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset strategy state."""
        pass

# =============================================================================
# Agent Interface
# =============================================================================

class IAgent(ABC):
    """Abstract interface for AI agents."""
    
    @abstractmethod
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute the agent's main logic."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        pass
    
    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update agent configuration."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        pass

# =============================================================================
# Execution Interface
# =============================================================================

class IExecutionEngine(ABC):
    """Abstract interface for trade execution engines."""
    
    @abstractmethod
    def execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade based on the signal."""
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for a symbol."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if execution engine is connected."""
        pass

# =============================================================================
# Risk Management Interface
# =============================================================================

class IRiskManager(ABC):
    """Abstract interface for risk management."""
    
    @abstractmethod
    def check_risk(self, signal: TradeSignal) -> Dict[str, Any]:
        """Check if a trade signal meets risk requirements."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradeSignal, capital: float) -> float:
        """Calculate appropriate position size for a trade."""
        pass
    
    @abstractmethod
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        pass
    
    @abstractmethod
    def update_risk_limits(self, limits: Dict[str, float]) -> bool:
        """Update risk limits."""
        pass

# =============================================================================
# Portfolio Management Interface
# =============================================================================

class IPortfolioManager(ABC):
    """Abstract interface for portfolio management."""
    
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        pass
    
    @abstractmethod
    def get_holdings(self) -> Dict[str, float]:
        """Get current holdings."""
        pass
    
    @abstractmethod
    def rebalance(self, target_weights: Dict[str, float]) -> bool:
        """Rebalance portfolio to target weights."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get portfolio performance metrics."""
        pass

# =============================================================================
# Event System Interface
# =============================================================================

class IEventBus(ABC):
    """Abstract interface for event bus."""
    
    @abstractmethod
    def subscribe(self, event_type: EventType, handler: Callable[[SystemEvent], None]) -> None:
        """Subscribe to an event type."""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: EventType, handler: Callable[[SystemEvent], None]) -> None:
        """Unsubscribe from an event type."""
        pass
    
    @abstractmethod
    def publish(self, event: SystemEvent) -> None:
        """Publish an event."""
        pass

# =============================================================================
# Configuration Interface
# =============================================================================

class IConfigManager(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value."""
        pass
    
    @abstractmethod
    def load_config(self, path: str) -> bool:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def save_config(self, path: str) -> bool:
        """Save configuration to file."""
        pass
    
    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        pass

# =============================================================================
# Logging Interface
# =============================================================================

class ILogger(ABC):
    """Abstract interface for logging."""
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        pass
    
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        pass
    
    @abstractmethod
    def log_event(self, event: SystemEvent) -> None:
        """Log system event."""
        pass

# =============================================================================
# Plugin Interface
# =============================================================================

class IPlugin(ABC):
    """Abstract interface for plugins."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

# =============================================================================
# Export all interfaces
# =============================================================================

__all__ = [
    # Data types
    'DataRequest', 'DataResponse', 'ModelConfig', 'TrainingResult',
    'PredictionResult', 'SignalData', 'TradeSignal', 'EventType', 'SystemEvent',
    
    # Core interfaces
    'IDataProvider', 'IModel', 'IStrategy', 'IAgent', 'IExecutionEngine',
    'IRiskManager', 'IPortfolioManager', 'IEventBus', 'IConfigManager',
    'ILogger', 'IPlugin'
] 