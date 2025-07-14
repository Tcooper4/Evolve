"""
Risk Controls Module

This module contains risk control classes and configurations for the execution agent.
Extracted from the original execution_agent.py for modularity.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional


class RiskThresholdType(Enum):
    """Risk threshold type enum."""

    PERCENTAGE = "percentage"
    ATR_BASED = "atr_based"
    FIXED = "fixed"


class ExitReason(Enum):
    """Exit reason enum."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    MAX_HOLDING_PERIOD = "max_holding_period"
    MANUAL = "manual"
    RISK_LIMIT = "risk_limit"
    VOLATILITY_LIMIT = "volatility_limit"
    CORRELATION_LIMIT = "correlation_limit"


@dataclass
class RiskThreshold:
    """Risk threshold configuration."""

    threshold_type: RiskThresholdType
    value: float
    atr_multiplier: Optional[float] = None
    atr_period: int = 14

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "threshold_type": self.threshold_type.value,
            "value": self.value,
            "atr_multiplier": self.atr_multiplier,
            "atr_period": self.atr_period,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskThreshold":
        """Create from dictionary."""
        data["threshold_type"] = RiskThresholdType(data["threshold_type"])
        return cls(**data)


@dataclass
class RiskControls:
    """Risk controls configuration."""

    stop_loss: RiskThreshold
    take_profit: RiskThreshold
    max_position_size: float = 0.2  # 20% of capital
    max_portfolio_risk: float = 0.05  # 5% of portfolio
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_correlation: float = 0.7  # Maximum correlation between positions
    volatility_limit: float = 0.5  # Maximum volatility for new positions
    trailing_stop: bool = False
    trailing_stop_distance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stop_loss": self.stop_loss.to_dict(),
            "take_profit": self.take_profit.to_dict(),
            "max_position_size": self.max_position_size,
            "max_portfolio_risk": self.max_portfolio_risk,
            "max_daily_loss": self.max_daily_loss,
            "max_correlation": self.max_correlation,
            "volatility_limit": self.volatility_limit,
            "trailing_stop": self.trailing_stop,
            "trailing_stop_distance": self.trailing_stop_distance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskControls":
        """Create from dictionary."""
        data["stop_loss"] = RiskThreshold.from_dict(data["stop_loss"])
        data["take_profit"] = RiskThreshold.from_dict(data["take_profit"])
        return cls(**data)


@dataclass
class ExitEvent:
    """Exit event data class."""

    timestamp: datetime
    symbol: str
    position_id: str
    exit_price: float
    exit_reason: ExitReason
    pnl: float
    holding_period: timedelta
    risk_metrics: Dict[str, float]
    market_conditions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        exit_dict = asdict(self)
        exit_dict["timestamp"] = self.timestamp.isoformat()
        exit_dict["exit_reason"] = self.exit_reason.value
        exit_dict["holding_period"] = self.holding_period.total_seconds()
        return exit_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExitEvent":
        """Create from dictionary."""
        data["exit_reason"] = ExitReason(data["exit_reason"])
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if isinstance(data["holding_period"], (int, float)):
            data["holding_period"] = timedelta(seconds=data["holding_period"])
        return cls(**data)


def create_default_risk_controls() -> RiskControls:
    """Create default risk controls configuration."""
    return RiskControls(
        stop_loss=RiskThreshold(
            threshold_type=RiskThresholdType.PERCENTAGE,
            value=0.02,  # 2% stop loss
        ),
        take_profit=RiskThreshold(
            threshold_type=RiskThresholdType.PERCENTAGE,
            value=0.06,  # 6% take profit
        ),
        max_position_size=0.2,
        max_portfolio_risk=0.05,
        max_daily_loss=0.02,
        max_correlation=0.7,
        volatility_limit=0.5,
        trailing_stop=False,
    ) 