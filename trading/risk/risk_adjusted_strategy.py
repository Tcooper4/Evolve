"""Risk-adjusted strategy module for dynamic position sizing."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from trading.risk_analyzer import RiskAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading/risk/logs/risk_strategy.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for risk-adjusted strategy."""

    base_position_size: float = 1.0
    max_position_size: float = 2.0
    min_position_size: float = 0.2
    volatility_weight: float = 0.4
    sharpe_weight: float = 0.3
    drawdown_weight: float = 0.3
    fallback_threshold: float = 0.7


@dataclass
class PositionAdjustment:
    """Container for position adjustment results."""

    position_size: float
    signal_threshold: float
    risk_score: float
    adjustments: Dict[str, float]
    fallback_mode: bool


class RiskAdjustedStrategy:
    """Strategy with dynamic risk-based adjustments."""

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        risk_analyzer: Optional[RiskAnalyzer] = None,
    ):
        """Initialize risk-adjusted strategy.

        Args:
            config: Strategy configuration
            risk_analyzer: Risk analyzer instance
        """
        self.config = config or StrategyConfig()
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()
        self.last_adjustment = None

    def adjust_position(
        self,
        returns: pd.Series,
        forecast_confidence: float,
        historical_error: float,
        current_position_size: float,
        current_signal: float,
    ) -> PositionAdjustment:
        """Adjust position based on risk metrics.

        Args:
            returns: Daily returns series
            forecast_confidence: Model forecast confidence
            historical_error: Historical forecast error
            current_position_size: Current position size
            current_signal: Current trading signal

        Returns:
            PositionAdjustment object
        """
        # Get risk assessment
        assessment = self.risk_analyzer.analyze_risk(
            returns, forecast_confidence, historical_error
        )

        # Calculate risk score
        risk_score = assessment.forecast_risk_score

        # Calculate adjustments
        adjustments = self._calculate_adjustments(
            assessment, current_position_size, current_signal
        )

        # Check for fallback mode
        fallback_mode = risk_score > self.config.fallback_threshold

        # Create adjustment
        adjustment = PositionAdjustment(
            position_size=adjustments["position_size"],
            signal_threshold=adjustments["signal_threshold"],
            risk_score=risk_score,
            adjustments=adjustments,
            fallback_mode=fallback_mode,
        )

        self.last_adjustment = adjustment
        return adjustment

    def _calculate_adjustments(
        self,
        assessment: "RiskAssessment",
        current_position_size: float,
        current_signal: float,
    ) -> Dict[str, float]:
        """Calculate position adjustments.

        Args:
            assessment: Risk assessment
            current_position_size: Current position size
            current_signal: Current trading signal

        Returns:
            Dictionary of adjustments
        """
        # Get metrics
        metrics = assessment.metrics

        # Calculate adjustment factors
        vol_factor = 1 - (metrics["volatility"] / 0.5)  # Normalize to 0.5
        sharpe_factor = metrics["sharpe_ratio"] / 2  # Normalize to 2
        dd_factor = 1 + metrics["max_drawdown"]  # Drawdown is negative

        # Calculate weighted adjustment
        adjustment = (
            vol_factor * self.config.volatility_weight
            + sharpe_factor * self.config.sharpe_weight
            + dd_factor * self.config.drawdown_weight
        )

        # Calculate new position size
        new_position_size = (
            current_position_size * adjustment * self.config.base_position_size
        )

        # Apply limits
        new_position_size = min(
            max(new_position_size, self.config.min_position_size),
            self.config.max_position_size,
        )

        # Adjust signal threshold based on risk
        signal_threshold = 0.5 * (1 + assessment.forecast_risk_score)

        return {
            "position_size": new_position_size,
            "signal_threshold": signal_threshold,
            "volatility_factor": vol_factor,
            "sharpe_factor": sharpe_factor,
            "drawdown_factor": dd_factor,
        }

    def get_fallback_rules(self) -> Dict[str, float]:
        """Get fallback rules for high-risk periods.

        Returns:
            Dictionary of fallback rules
        """
        return {
            "position_size": self.config.min_position_size,
            "signal_threshold": 0.8,
            "max_drawdown": -0.1,
            "volatility_limit": 0.3,
        }

    def should_enter_trade(self, signal: float, adjustment: PositionAdjustment) -> bool:
        """Check if trade should be entered.

        Args:
            signal: Trading signal
            adjustment: Position adjustment

        Returns:
            True if trade should be entered
        """
        if adjustment.fallback_mode:
            return (
                abs(signal) > adjustment.signal_threshold
                and adjustment.risk_score < self.config.fallback_threshold
            )
        else:
            return abs(signal) > adjustment.signal_threshold

    def get_last_adjustment(self) -> Optional[PositionAdjustment]:
        """Get the last position adjustment.

        Returns:
            Last PositionAdjustment object or None
        """
        return self.last_adjustment
