"""
Trade Signals Module

This module contains trade signal classes and execution request/result models.
Extracted from the original execution_agent.py for modularity.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from trading.portfolio.portfolio_manager import TradeDirection


@dataclass
class TradeSignal:
    """Trade signal data class."""

    symbol: str
    direction: TradeDirection
    strategy: str
    confidence: float
    entry_price: float
    size: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    max_holding_period: Optional[timedelta] = None
    market_data: Optional[Dict[str, Any]] = None
    risk_controls: Optional[Any] = None  # RiskControls type
    timestamp: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        signal_dict = asdict(self)
        signal_dict["timestamp"] = self.timestamp.isoformat()
        if self.max_holding_period:
            signal_dict["max_holding_period"] = self.max_holding_period.total_seconds()
        if self.risk_controls:
            signal_dict["risk_controls"] = self.risk_controls.to_dict()
        return signal_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeSignal":
        """Create from dictionary."""
        # Convert string enums back to enum values
        data["direction"] = TradeDirection(data["direction"])

        # Convert string timestamp to datetime
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        # Convert max_holding_period
        if "max_holding_period" in data and isinstance(
            data["max_holding_period"], (int, float)
        ):
            data["max_holding_period"] = timedelta(seconds=data["max_holding_period"])

        # Convert risk controls
        if "risk_controls" in data and data["risk_controls"]:
            from .risk_controls import RiskControls
            data["risk_controls"] = RiskControls.from_dict(data["risk_controls"])

        return cls(**data)


@dataclass
class ExecutionRequest:
    """Request for execution agent operations."""

    operation_type: str  # 'execute', 'exit', 'status', etc.
    signal: Optional[TradeSignal] = None
    market_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.utcnow()


@dataclass
class ExecutionResult:
    """Execution result data class."""

    success: bool
    signal: TradeSignal
    position: Optional[Any] = None  # Position type
    execution_price: Optional[float] = None
    slippage: float = 0.0
    fees: float = 0.0
    message: str = ""
    error: Optional[str] = None
    risk_metrics: Optional[Dict[str, float]] = None
    timestamp: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result_dict = asdict(self)
        result_dict["timestamp"] = self.timestamp.isoformat()
        result_dict["signal"] = self.signal.to_dict()
        if self.position:
            result_dict["position"] = self.position.to_dict()
        return result_dict
