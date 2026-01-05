"""
Risk Control Module

Provides risk control logic using rolling standard deviation and threshold comparison
instead of hardcoded values.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from trading.utils.safe_math import safe_drawdown

logger = logging.getLogger(__name__)


class RiskControl:
    """Risk control implementation using statistical measures."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize risk control.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.volatility_window = self.config.get("volatility_window", 20)
        self.volatility_threshold = self.config.get("volatility_threshold", 0.02)
        self.max_drawdown_threshold = self.config.get("max_drawdown_threshold", 0.15)
        self.position_size_limit = self.config.get("position_size_limit", 0.1)

    def calculate_rolling_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Calculate rolling standard deviation of returns.

        Args:
            returns: Series of returns

        Returns:
            Rolling volatility series
        """
        return returns.rolling(window=self.volatility_window).std()

    def check_volatility_risk(self, returns: pd.Series) -> bool:
        """
        Check if current volatility exceeds threshold.

        Args:
            returns: Series of returns

        Returns:
            True if volatility risk is high
        """
        if len(returns) < self.volatility_window:
            logger.warning("Insufficient data for volatility calculation")
            return False

        current_volatility = (
            returns.rolling(window=self.volatility_window).std().iloc[-1]
        )
        is_high_risk = current_volatility > self.volatility_threshold

        logger.info(
            f"Current volatility: {current_volatility:.4f}, threshold: {self.volatility_threshold:.4f}, risk: {is_high_risk}"
        )
        return is_high_risk

    def check_drawdown_risk(self, equity_curve: pd.Series) -> bool:
        """
        Check if maximum drawdown exceeds threshold.

        Args:
            equity_curve: Series of equity values

        Returns:
            True if drawdown risk is high
        """
        if len(equity_curve) < 2:
            return False

        # Calculate drawdown
        drawdown = safe_drawdown(equity_curve)
        max_drawdown = drawdown.min()

        is_high_risk = abs(max_drawdown) > self.max_drawdown_threshold

        logger.info(
            f"Max drawdown: {max_drawdown:.4f}, threshold: {self.max_drawdown_threshold:.4f}, risk: {is_high_risk}"
        )
        return is_high_risk

    def check_position_size_risk(self, position_size: float) -> bool:
        """
        Check if position size exceeds limit.

        Args:
            position_size: Current position size as fraction of portfolio

        Returns:
            True if position size risk is high
        """
        is_high_risk = abs(position_size) > self.position_size_limit

        logger.info(
            f"Position size: {position_size:.4f}, limit: {self.position_size_limit:.4f}, risk: {is_high_risk}"
        )
        return is_high_risk

    def get_risk_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive risk status.

        Args:
            data: Dictionary containing returns, equity_curve, position_size

        Returns:
            Dictionary with risk status for each metric
        """
        risk_status = {
            "volatility_risk": False,
            "drawdown_risk": False,
            "position_size_risk": False,
            "overall_risk": False,
        }

        # Check volatility risk
        if "returns" in data:
            risk_status["volatility_risk"] = self.check_volatility_risk(data["returns"])

        # Check drawdown risk
        if "equity_curve" in data:
            risk_status["drawdown_risk"] = self.check_drawdown_risk(
                data["equity_curve"]
            )

        # Check position size risk
        if "position_size" in data:
            risk_status["position_size_risk"] = self.check_position_size_risk(
                data["position_size"]
            )

        # Overall risk assessment
        risk_status["overall_risk"] = any(
            [
                risk_status["volatility_risk"],
                risk_status["drawdown_risk"],
                risk_status["position_size_risk"],
            ]
        )

        return risk_status

    def should_reduce_position(self, data: Dict[str, Any]) -> bool:
        """
        Determine if position should be reduced based on risk metrics.

        Args:
            data: Dictionary containing risk data

        Returns:
            True if position should be reduced
        """
        risk_status = self.get_risk_status(data)
        return risk_status["overall_risk"]

    def get_position_adjustment(self, data: Dict[str, Any]) -> float:
        """
        Get recommended position adjustment factor.

        Args:
            data: Dictionary containing risk data

        Returns:
            Adjustment factor (0.0 to 1.0)
        """
        risk_status = self.get_risk_status(data)

        # Count risk factors
        risk_count = sum(
            [
                risk_status["volatility_risk"],
                risk_status["drawdown_risk"],
                risk_status["position_size_risk"],
            ]
        )

        # Reduce position based on number of risk factors
        if risk_count == 0:
            return 1.0
        elif risk_count == 1:
            return 0.75
        elif risk_count == 2:
            return 0.5
        else:
            return 0.25
