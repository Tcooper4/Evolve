"""
Execution Module

This module contains modularized execution components extracted from execution_agent.py.
"""

from .risk_controls import (
    RiskControls,
    RiskThreshold,
    RiskThresholdType,
    ExitReason,
    ExitEvent,
    create_default_risk_controls,
)
from .trade_signals import TradeSignal
from .execution_models import ExecutionRequest, ExecutionResult
from .risk_calculator import RiskCalculator
from .execution_providers import (
    ExecutionMode,
    ExecutionProvider,
    SimulationProvider,
    AlpacaProvider,
    InteractiveBrokersProvider,
    RobinhoodProvider,
    create_execution_provider,
)

__all__ = [
    # Risk Controls
    "RiskControls",
    "RiskThreshold", 
    "RiskThresholdType",
    "ExitReason",
    "ExitEvent",
    "create_default_risk_controls",
    
    # Trade Signals
    "TradeSignal",
    
    # Execution Models
    "ExecutionRequest",
    "ExecutionResult",
    
    # Risk Calculator
    "RiskCalculator",
    
    # Execution Providers
    "ExecutionMode",
    "ExecutionProvider",
    "SimulationProvider",
    "AlpacaProvider",
    "InteractiveBrokersProvider", 
    "RobinhoodProvider",
    "create_execution_provider",
] 