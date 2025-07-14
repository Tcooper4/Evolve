"""Trade Signals Module.

This module contains trade signal classes and logic extracted from execution_agent.py.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from trading.portfolio.portfolio_manager import TradeDirection
from .risk_controls import RiskControls


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
    risk_controls: Optional[RiskControls] = None
    timestamp: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
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
            data["risk_controls"] = RiskControls.from_dict(data["risk_controls"])

        return cls(**data)

    def validate(self) -> bool:
        """Validate the trade signal."""
        if not self.symbol or not self.symbol.strip():
            return False
        
        if self.confidence < 0 or self.confidence > 1:
            return False
        
        if self.entry_price <= 0:
            return False
        
        if self.direction not in [TradeDirection.LONG, TradeDirection.SHORT]:
            return False
        
        return True

    def get_risk_adjusted_size(self, portfolio_value: float) -> float:
        """Calculate risk-adjusted position size."""
        if self.size is not None:
            return min(self.size, portfolio_value * 0.2)  # Max 20% of portfolio
        
        # Default to 5% of portfolio
        return portfolio_value * 0.05 