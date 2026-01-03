"""Execution Module for Evolve Trading Platform.

This module provides live trading execution capabilities.
"""

from .live_trading_interface import (
    AccountInfo,
    AlpacaTradingInterface,
    LiveTradingInterface,
    OrderRequest,
    OrderStatus,
    Position,
    SimulatedExecutionEngine,
    create_live_trading_interface,
)

# Import broker adapter if available
try:
    from .broker_adapter import BrokerAdapter, BrokerType
    _BROKER_ADAPTER_AVAILABLE = True
except ImportError:
    _BROKER_ADAPTER_AVAILABLE = False
    BrokerAdapter = None
    BrokerType = None

# Import execution agent if available
try:
    from .execution_agent import ExecutionAgent, ExecutionMode, OrderSide, OrderType
    _EXECUTION_AGENT_AVAILABLE = True
except ImportError:
    _EXECUTION_AGENT_AVAILABLE = False
    ExecutionAgent = None
    ExecutionMode = None
    OrderSide = None
    OrderType = None

__all__ = [
    "LiveTradingInterface",
    "SimulatedExecutionEngine",
    "AlpacaTradingInterface",
    "OrderRequest",
    "OrderStatus",
    "Position",
    "AccountInfo",
    "create_live_trading_interface",
]

# Add broker adapter exports if available
if _BROKER_ADAPTER_AVAILABLE:
    __all__.extend(["BrokerAdapter", "BrokerType"])

# Add execution agent exports if available
if _EXECUTION_AGENT_AVAILABLE:
    __all__.extend(["ExecutionAgent", "ExecutionMode", "OrderSide", "OrderType"])
