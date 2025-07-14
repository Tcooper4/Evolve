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
from .risk_controls import RiskControls, RiskThreshold, RiskThresholdType
from .trade_signals import TradeSignal, ExecutionRequest, ExecutionResult
from .execution_providers import ExecutionProvider, SimulationProvider, AlpacaProvider, IBProvider, RobinhoodProvider
from .position_manager import PositionManager, ExitEvent, ExitReason

__all__ = [
    'ExecutionAgent',
    'create_execution_agent',
    'RiskControls',
    'RiskThreshold',
    'RiskThresholdType',
    'TradeSignal',
    'ExecutionRequest',
    'ExecutionResult',
    'ExecutionProvider',
    'SimulationProvider',
    'AlpacaProvider',
    'IBProvider',
    'RobinhoodProvider',
    'PositionManager',
    'ExitEvent',
    'ExitReason'
] 