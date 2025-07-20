"""Execution Models Module.

This module contains execution-related data models extracted from execution_agent.py.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from trading.portfolio.portfolio_manager import Position
from .trade_signals import TradeSignal


@dataclass
class ExecutionRequest:
    """Request for execution agent operations."""

    operation_type: str  # 'execute', 'exit', 'status', etc.
    signal: Optional[TradeSignal] = None
    market_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
        request_dict = asdict(self)
        request_dict["timestamp"] = self.timestamp.isoformat()
        if self.signal:
            request_dict["signal"] = self.signal.to_dict()
        return request_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRequest":
        """Create from dictionary."""
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "signal" in data and data["signal"]:
            data["signal"] = TradeSignal.from_dict(data["signal"])
        return cls(**data)


@dataclass
class ExecutionResult:
    """Execution result data class."""

    success: bool
    signal: TradeSignal
    position: Optional[Position] = None
    execution_price: Optional[float] = None
    slippage: float = 0.0
    fees: float = 0.0
    message: str = ""
    error: Optional[str] = None
    risk_metrics: Optional[Dict[str, float]] = None
    timestamp: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
        result_dict = asdict(self)
        result_dict["timestamp"] = self.timestamp.isoformat()
        result_dict["signal"] = self.signal.to_dict()
        if self.position:
            result_dict["position"] = self.position.to_dict()
        return result_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """Create from dictionary."""
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["signal"] = TradeSignal.from_dict(data["signal"])
        if "position" in data and data["position"]:
            from trading.portfolio.portfolio_manager import Position
            data["position"] = Position.from_dict(data["position"])
        return cls(**data)

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.success and self.error is None

    def get_total_cost(self) -> float:
        """Calculate total cost including fees and slippage."""
        if not self.execution_price or not self.position:
            return 0.0
        
        base_cost = self.execution_price * self.position.size
        return base_cost + self.fees + (self.slippage * self.position.size)
