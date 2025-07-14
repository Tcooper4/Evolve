#!/usr/bin/env python3
"""
Test Execution Agent Risk Controls

This module tests the risk controls functionality of the execution agent.
Updated for the new modular structure.
"""

import pytest
from datetime import datetime, timedelta

from trading.agents.execution.risk_controls import (
    RiskControls, RiskThreshold, RiskThresholdType, ExitReason, ExitEvent, create_default_risk_controls
)


class TestRiskControls:
    """Test risk controls functionality."""

    def test_risk_threshold_creation(self):
        """Test risk threshold creation."""
        threshold = RiskThreshold(
            threshold_type=RiskThresholdType.PERCENTAGE,
            value=0.02,
            atr_multiplier=None,
            atr_period=14
        )
        
        assert threshold.threshold_type == RiskThresholdType.PERCENTAGE
        assert threshold.value == 0.02
        assert threshold.atr_period == 14

    def test_risk_threshold_to_dict(self):
        """Test risk threshold serialization."""
        threshold = RiskThreshold(
            threshold_type=RiskThresholdType.ATR_BASED,
            value=2.0,
            atr_multiplier=2.0,
            atr_period=20
        )
        
        threshold_dict = threshold.to_dict()
        
        assert threshold_dict["threshold_type"] == "atr_based"
        assert threshold_dict["value"] == 2.0
        assert threshold_dict["atr_multiplier"] == 2.0
        assert threshold_dict["atr_period"] == 20

    def test_risk_threshold_from_dict(self):
        """Test risk threshold deserialization."""
        threshold_dict = {
            "threshold_type": "fixed",
            "value": 5.0,
            "atr_multiplier": None,
            "atr_period": 14
        }
        
        threshold = RiskThreshold.from_dict(threshold_dict)
        
        assert threshold.threshold_type == RiskThresholdType.FIXED
        assert threshold.value == 5.0
        assert threshold.atr_multiplier is None

    def test_risk_controls_creation(self):
        """Test risk controls creation."""
        stop_loss = RiskThreshold(
            threshold_type=RiskThresholdType.PERCENTAGE,
            value=0.02
        )
        take_profit = RiskThreshold(
            threshold_type=RiskThresholdType.PERCENTAGE,
            value=0.06
        )
        
        controls = RiskControls(
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_position_size=0.2,
            max_portfolio_risk=0.05
        )
        
        assert controls.max_position_size == 0.2
        assert controls.max_portfolio_risk == 0.05
        assert controls.stop_loss.value == 0.02
        assert controls.take_profit.value == 0.06

    def test_risk_controls_to_dict(self):
        """Test risk controls serialization."""
        controls = create_default_risk_controls()
        controls_dict = controls.to_dict()
        
        assert "stop_loss" in controls_dict
        assert "take_profit" in controls_dict
        assert controls_dict["max_position_size"] == 0.2
        assert controls_dict["max_portfolio_risk"] == 0.05

    def test_risk_controls_from_dict(self):
        """Test risk controls deserialization."""
        controls_dict = {
            "stop_loss": {
                "threshold_type": "percentage",
                "value": 0.03,
                "atr_multiplier": None,
                "atr_period": 14
            },
            "take_profit": {
                "threshold_type": "percentage",
                "value": 0.08,
                "atr_multiplier": None,
                "atr_period": 14
            },
            "max_position_size": 0.15,
            "max_portfolio_risk": 0.04,
            "max_daily_loss": 0.02,
            "max_correlation": 0.7,
            "volatility_limit": 0.5,
            "trailing_stop": False,
            "trailing_stop_distance": None
        }
        
        controls = RiskControls.from_dict(controls_dict)
        
        assert controls.max_position_size == 0.15
        assert controls.max_portfolio_risk == 0.04
        assert controls.stop_loss.value == 0.03
        assert controls.take_profit.value == 0.08

    def test_create_default_risk_controls(self):
        """Test default risk controls creation."""
        controls = create_default_risk_controls()
        
        assert controls.max_position_size == 0.2
        assert controls.max_portfolio_risk == 0.05
        assert controls.max_daily_loss == 0.02
        assert controls.max_correlation == 0.7
        assert controls.volatility_limit == 0.5
        assert controls.trailing_stop is False

    def test_exit_reason_enum(self):
        """Test exit reason enum."""
        assert ExitReason.TAKE_PROFIT.value == "take_profit"
        assert ExitReason.STOP_LOSS.value == "stop_loss"
        assert ExitReason.MAX_HOLDING_PERIOD.value == "max_holding_period"
        assert ExitReason.MANUAL.value == "manual"
        assert ExitReason.RISK_LIMIT.value == "risk_limit"

    def test_exit_event_creation(self):
        """Test exit event creation."""
        timestamp = datetime.utcnow()
        exit_event = ExitEvent(
            timestamp=timestamp,
            symbol="AAPL",
            position_id="pos_123",
            exit_price=150.00,
            exit_reason=ExitReason.TAKE_PROFIT,
            pnl=50.00,
            holding_period=timedelta(hours=2),
            risk_metrics={"volatility": 0.15, "var": 25.0},
            market_conditions={"volume": 1000000, "spread": 0.01}
        )
        
        assert exit_event.symbol == "AAPL"
        assert exit_event.exit_price == 150.00
        assert exit_event.exit_reason == ExitReason.TAKE_PROFIT
        assert exit_event.pnl == 50.00
        assert exit_event.holding_period == timedelta(hours=2)

    def test_exit_event_to_dict(self):
        """Test exit event serialization."""
        timestamp = datetime.utcnow()
        exit_event = ExitEvent(
            timestamp=timestamp,
            symbol="GOOGL",
            position_id="pos_456",
            exit_price=2800.00,
            exit_reason=ExitReason.STOP_LOSS,
            pnl=-100.00,
            holding_period=timedelta(hours=1),
            risk_metrics={"volatility": 0.12},
            market_conditions={"volume": 500000}
        )
        
        exit_dict = exit_event.to_dict()
        
        assert exit_dict["symbol"] == "GOOGL"
        assert exit_dict["exit_price"] == 2800.00
        assert exit_dict["exit_reason"] == "stop_loss"
        assert exit_dict["pnl"] == -100.00
        assert isinstance(exit_dict["timestamp"], str)
        assert isinstance(exit_dict["holding_period"], (int, float))

    def test_exit_event_from_dict(self):
        """Test exit event deserialization."""
        timestamp = datetime.utcnow()
        exit_dict = {
            "timestamp": timestamp.isoformat(),
            "symbol": "MSFT",
            "position_id": "pos_789",
            "exit_price": 300.00,
            "exit_reason": "manual",
            "pnl": 25.00,
            "holding_period": 3600,  # 1 hour in seconds
            "risk_metrics": {"volatility": 0.10},
            "market_conditions": {"volume": 750000}
        }
        
        exit_event = ExitEvent.from_dict(exit_dict)
        
        assert exit_event.symbol == "MSFT"
        assert exit_event.exit_price == 300.00
        assert exit_event.exit_reason == ExitReason.MANUAL
        assert exit_event.pnl == 25.00
        assert exit_event.holding_period == timedelta(seconds=3600)
        assert isinstance(exit_event.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__])
