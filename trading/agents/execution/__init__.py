"""
Execution Agent Package

This package contains the modularized execution agent components:
- Core execution agent
- Risk controls
- Trade signals
- Execution providers
- Position management
"""

from .execution_agent import ExecutionAgent, create_execution_agent
from .execution_providers import (
    AlpacaProvider,
    ExecutionProvider,
    IBProvider,
    RobinhoodProvider,
    SimulationProvider,
)
from .position_manager import ExitEvent, ExitReason, PositionManager
from .risk_controls import RiskControls, RiskThreshold, RiskThresholdType
from .trade_signals import ExecutionRequest, ExecutionResult, TradeSignal

__all__ = [
    "ExecutionAgent",
    "create_execution_agent",
    "RiskControls",
    "RiskThreshold",
    "RiskThresholdType",
    "TradeSignal",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionProvider",
    "SimulationProvider",
    "AlpacaProvider",
    "IBProvider",
    "RobinhoodProvider",
    "PositionManager",
    "ExitEvent",
    "ExitReason",
]
