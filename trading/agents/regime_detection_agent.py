"""
Regime Detection Agent

This agent identifies market regimes and adapts trading strategies accordingly.
It uses multiple detection methods including:
- Volatility-based regime detection
- Trend-based regime classification
- Machine learning-based regime prediction
- Multi-timeframe analysis
- Historical regime pattern matching

Features:
- Real-time regime identification
- Strategy adaptation recommendations
- Regime transition detection
- Performance tracking by regime
- Automated strategy switching
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.memory.agent_memory import AgentMemory
from trading.utils.reasoning_logger import (
    ConfidenceLevel,
    DecisionType,
    ReasoningLogger,
)

from .base_agent_interface import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    NORMAL = "normal"
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Metrics for regime detection."""

    volatility: float
    trend_strength: float
    momentum: float
    volume_trend: float
    correlation: float
    skewness: float
    kurtosis: float


@dataclass
class RegimeResult:
    """Result of regime detection."""

    regime: MarketRegime
    confidence: float
    metrics: RegimeMetrics
    transition_probability: float
    recommended_strategies: List[str]
    reasoning: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class RegimeDetectionAgent(BaseAgent):
    """
    Market regime detection agent that identifies current market conditions
    and recommends appropriate trading strategies.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the regime detection agent."""
        if config is None:
            config = AgentConfig(
                name="RegimeDetectionAgent",
                enabled=True,
                priority=2,
                max_concurrent_runs=3,
                timeout_seconds=60,
                retry_attempts=2,
                custom_config={},
            )
        super().__init__(config)

        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.reasoning_logger = ReasoningLogger()

        # Configuration
        self.lookback_periods = self.config.custom_config.get(
            "lookback_periods", {"short": 20, "medium": 50, "long": 200}
        )

        self.volatility_thresholds = self.config.custom_config.get(
            "volatility_thresholds", {"low": 0.15, "high": 0.35}
        )

        self.trend_thresholds = self.config.custom_config.get(
            "trend_thresholds", {"weak": 0.1, "strong": 0.3}
        )

        # Regime-specific strategy recommendations
        self.regime_strategies = {
            MarketRegime.BULL: ["momentum", "trend_following", "breakout"],
            MarketRegime.BEAR: ["defensive", "mean_reversion", "short_selling"],
            MarketRegime.SIDEWAYS: ["mean_reversion", "range_trading", "options"],
            MarketRegime.VOLATILE: ["volatility_trading", "defensive", "options"],
            MarketRegime.NORMAL: ["balanced", "momentum", "mean_reversion"],
            MarketRegime.UNKNOWN: ["defensive", "balanced"],
        }

        # Historical regime data
        self.regime_history: List[RegimeResult] = []
        self.performance_by_regime: Dict[MarketRegime, Dict[str, float]] = {}

        # ML model for regime prediction (placeholder)
        self.regime_model = None

        self.logger.info("RegimeDetectionAgent initialized successfully")

    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the regime detection agent's main logic.

        Args:
            **kwargs: Expected to contain 'data' and optionally 'symbol'

        Returns:
            AgentResult: Result of the regime detection
        """
        try:
            data = kwargs.get("data")
            symbol = kwargs.get("symbol", "UNKNOWN")

            if data is None:
                return AgentResult(
                    success=False,
                    error_message="No data provided for regime detection",
                    error_type="MissingData",
                )

            # Detect regime
            result = self.detect_regime(data, symbol)

            return AgentResult(
                success=True,
                data={
                    "regime": result.regime.value,
                    "confidence": result.confidence,
                    "recommended_strategies": result.recommended_strategies,
                    "symbol": symbol,
                },
                metadata={
                    "detection_method": result.detection_method,
                    "features_used": result.features_used,
                    "timestamp": result.timestamp.isoformat(),
                },
            )

        except Exception as e:
            return self.handle_error(e)

    def detect_regime(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> RegimeResult:
        """
        Detect the current market regime.

        Args:
            data: Market data with OHLCV columns
            symbol: Symbol being analyzed

        Returns:
            Regime detection result
        """
        try:
            # Calculate regime metrics
            metrics = self._calculate_regime_metrics(data)

            # Detect regime using multiple methods
            regime_votes = []
            confidences = []

            # Method 1: Volatility-based detection
            vol_regime, vol_confidence = self._volatility_based_detection(metrics)
            regime_votes.append(vol_regime)
            confidences.append(vol_confidence)

            # Method 2: Trend-based detection
            trend_regime, trend_confidence = self._trend_based_detection(metrics)
            regime_votes.append(trend_regime)
            confidences.append(trend_confidence)

            # Method 3: Momentum-based detection
            momentum_regime, momentum_confidence = self._momentum_based_detection(
                metrics
            )
            regime_votes.append(momentum_regime)
            confidences.append(momentum_confidence)

            # Method 4: ML-based detection (if available)
            if self.regime_model is not None:
                ml_regime, ml_confidence = self._ml_based_detection(metrics)
                regime_votes.append(ml_regime)
                confidences.append(ml_confidence)

            # Combine results using weighted voting
            final_regime = self._combine_regime_votes(regime_votes, confidences)
            final_confidence = np.mean(confidences)

            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(final_regime)

            # Get recommended strategies
            recommended_strategies = self.regime_strategies.get(
                final_regime, ["defensive"]
            )

            # Generate reasoning
            reasoning = self._generate_regime_reasoning(
                final_regime, metrics, confidences
            )

            # Create result
            result = RegimeResult(
                regime=final_regime,
                confidence=final_confidence,
                metrics=metrics,
                transition_probability=transition_prob,
                recommended_strategies=recommended_strategies,
                reasoning=reasoning,
                timestamp=datetime.now(),
                metadata={
                    "symbol": symbol,
                    "method_votes": regime_votes,
                    "method_confidences": confidences,
                    "data_points": len(data),
                },
            )

            # Log the detection
            self._log_regime_detection(result)

            # Store in history
            self.regime_history.append(result)

            # Keep only last 1000 detections
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]

            self.logger.info(
                f"Detected {final_regime.value} regime for {symbol} with confidence {final_confidence:.2f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return self._create_fallback_result(symbol)

    def _calculate_regime_metrics(self, data: pd.DataFrame) -> RegimeMetrics:
        """Calculate metrics used for regime detection."""
        try:
            # Ensure we have required columns
            if "close" not in data.columns:
                raise ValueError("Data must contain 'close' column")

            close_prices = data["close"].values
            returns = np.diff(np.log(close_prices))

            # Include volatility bands in regime chart output
            df = data.copy()
            df["Volatility"] = df["close"].rolling(20).std()

            # Volatility (rolling standard deviation)
            volatility = np.std(returns[-self.lookback_periods["short"] :]) * np.sqrt(
                252
            )

            # Trend strength (linear regression slope)
            x = np.arange(len(close_prices[-self.lookback_periods["medium"] :]))
            y = close_prices[-self.lookback_periods["medium"] :]
            trend_slope = np.polyfit(x, y, 1)[0]
            trend_strength = abs(trend_slope) / np.mean(y)

            # Momentum (rate of change)
            momentum = (
                close_prices[-1] - close_prices[-self.lookback_periods["short"]]
            ) / close_prices[-self.lookback_periods["short"]]

            # Volume trend (if available)
            volume_trend = 0.0
            if "volume" in data.columns:
                volume = data["volume"].values
                if len(volume) >= self.lookback_periods["short"]:
                    recent_volume = np.mean(volume[-self.lookback_periods["short"] :])
                    historical_volume = np.mean(
                        volume[
                            -self.lookback_periods["medium"] : -self.lookback_periods[
                                "short"
                            ]
                        ]
                    )
                    volume_trend = (
                        (recent_volume - historical_volume) / historical_volume
                        if historical_volume > 0
                        else 0
                    )

            # Correlation (autocorrelation of returns)
            if len(returns) >= 10:
                correlation = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            else:
                correlation = 0.0

            # Skewness and kurtosis
            skewness = self._calculate_skewness(returns)
            kurtosis = self._calculate_kurtosis(returns)

            return RegimeMetrics(
                volatility=volatility,
                trend_strength=trend_strength,
                momentum=momentum,
                volume_trend=volume_trend,
                correlation=correlation,
                skewness=skewness,
                kurtosis=kurtosis,
            )

        except Exception as e:
            self.logger.error(f"Error calculating regime metrics: {e}")
            return RegimeMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _volatility_based_detection(
        self, metrics: RegimeMetrics
    ) -> Tuple[MarketRegime, float]:
        """Detect regime based on volatility."""
        vol = metrics.volatility

        if vol < self.volatility_thresholds["low"]:
            return MarketRegime.NORMAL, 0.8
        elif vol > self.volatility_thresholds["high"]:
            return MarketRegime.VOLATILE, 0.9
        else:
            return MarketRegime.NORMAL, 0.6

    def _trend_based_detection(
        self, metrics: RegimeMetrics
    ) -> Tuple[MarketRegime, float]:
        """Detect regime based on trend strength and momentum."""
        trend_strength = metrics.trend_strength
        momentum = metrics.momentum

        # Strong upward trend
        if trend_strength > self.trend_thresholds["strong"] and momentum > 0.05:
            return MarketRegime.BULL, 0.85
        # Strong downward trend
        elif trend_strength > self.trend_thresholds["strong"] and momentum < -0.05:
            return MarketRegime.BEAR, 0.85
        # Weak trend
        elif trend_strength < self.trend_thresholds["weak"]:
            return MarketRegime.SIDEWAYS, 0.7
        # Moderate trend
        else:
            if momentum > 0:
                return MarketRegime.BULL, 0.6
            else:
                return MarketRegime.BEAR, 0.6

    def _momentum_based_detection(
        self, metrics: RegimeMetrics
    ) -> Tuple[MarketRegime, float]:
        """Detect regime based on momentum and price action."""
        momentum = metrics.momentum
        correlation = metrics.correlation

        # High momentum with positive correlation
        if momentum > 0.1 and correlation > 0.3:
            return MarketRegime.BULL, 0.8
        # High momentum with negative correlation
        elif momentum < -0.1 and correlation > 0.3:
            return MarketRegime.BEAR, 0.8
        # Low momentum with low correlation (choppy)
        elif abs(momentum) < 0.05 and abs(correlation) < 0.2:
            return MarketRegime.SIDEWAYS, 0.7
        # High momentum with low correlation (volatile)
        elif abs(momentum) > 0.1 and abs(correlation) < 0.1:
            return MarketRegime.VOLATILE, 0.75
        else:
            return MarketRegime.NORMAL, 0.5

    def _ml_based_detection(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """Detect regime using machine learning model."""
        # Placeholder for ML-based detection
        # In practice, this would use a trained model
        return MarketRegime.UNKNOWN, 0.5

    def _combine_regime_votes(
        self, votes: List[MarketRegime], confidences: List[float]
    ) -> MarketRegime:
        """Combine regime votes using weighted voting."""
        if not votes:
            return MarketRegime.UNKNOWN

        # Count weighted votes for each regime
        regime_scores = {}
        for regime, confidence in zip(votes, confidences):
            if regime not in regime_scores:
                regime_scores[regime] = 0
            regime_scores[regime] += confidence

        # Return regime with highest score
        return max(regime_scores.items(), key=lambda x: x[1])[0]

    def _calculate_transition_probability(self, current_regime: MarketRegime) -> float:
        """Calculate probability of regime transition."""
        if len(self.regime_history) < 10:
            return 0.1  # Default low probability

        # Count recent regime changes
        recent_history = self.regime_history[-20:]
        transitions = 0

        for i in range(1, len(recent_history)):
            if recent_history[i].regime != recent_history[i - 1].regime:
                transitions += 1

        # Calculate transition probability
        transition_prob = transitions / (len(recent_history) - 1)

        # Adjust based on current regime stability
        if current_regime in [MarketRegime.VOLATILE, MarketRegime.SIDEWAYS]:
            transition_prob *= (
                1.5  # Higher transition probability for volatile/sideways
            )

        return min(1.0, transition_prob)

    def _generate_regime_reasoning(
        self, regime: MarketRegime, metrics: RegimeMetrics, confidences: List[float]
    ) -> str:
        """Generate reasoning for regime detection."""
        reasoning_parts = []

        reasoning_parts.append(f"Detected {regime.value} market regime")

        # Add metric-based reasoning
        if metrics.volatility > self.volatility_thresholds["high"]:
            reasoning_parts.append("High volatility indicates volatile regime")
        elif metrics.volatility < self.volatility_thresholds["low"]:
            reasoning_parts.append("Low volatility suggests stable conditions")

        if abs(metrics.momentum) > 0.1:
            direction = "bullish" if metrics.momentum > 0 else "bearish"
            reasoning_parts.append(f"Strong {direction} momentum detected")

        if metrics.trend_strength > self.trend_thresholds["strong"]:
            reasoning_parts.append("Strong trend indicates directional regime")
        elif metrics.trend_strength < self.trend_thresholds["weak"]:
            reasoning_parts.append("Weak trend suggests sideways movement")

        # Add confidence information
        avg_confidence = np.mean(confidences)
        reasoning_parts.append(f"Detection confidence: {avg_confidence:.1%}")

        return "; ".join(reasoning_parts)

    def _log_regime_detection(self, result: RegimeResult):
        """Log the regime detection decision."""
        self.reasoning_logger.log_decision(
            agent_name="RegimeDetectionAgent",
            decision_type=DecisionType.REGIME_DETECTION,
            action_taken=f"Detected {result.regime.value} regime",
            context={
                "regime": result.regime.value,
                "confidence": result.confidence,
                "transition_probability": result.transition_probability,
                "recommended_strategies": result.recommended_strategies,
            },
            reasoning={
                "primary_reason": result.reasoning,
                "supporting_factors": [
                    f"Volatility: {result.metrics.volatility:.3f}",
                    f"Trend strength: {result.metrics.trend_strength:.3f}",
                    f"Momentum: {result.metrics.momentum:.3f}",
                    f"Correlation: {result.metrics.correlation:.3f}",
                ],
                "alternatives_considered": [
                    f"Other possible regimes: {[r.value for r in MarketRegime if r != result.regime]}"
                ],
                "confidence_explanation": f"Confidence {result.confidence:.1%} based on multiple detection methods",
            },
            confidence_level=(
                ConfidenceLevel.HIGH
                if result.confidence > 0.8
                else ConfidenceLevel.MEDIUM
            ),
            metadata=result.metadata,
        )

    def _create_fallback_result(self, symbol: str) -> RegimeResult:
        """Create a fallback regime result when detection fails."""
        return RegimeResult(
            regime=MarketRegime.UNKNOWN,
            confidence=0.3,
            metrics=RegimeMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            transition_probability=0.1,
            recommended_strategies=["defensive"],
            reasoning="Fallback regime detection due to insufficient data or error",
            timestamp=datetime.now(),
            metadata={"symbol": symbol, "error": "Detection failed"},
        )

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics."""
        if not self.regime_history:
            return {"total_detections": 0}

        # Count regimes
        regime_counts = {}
        for result in self.regime_history:
            regime = result.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Calculate average confidence
        avg_confidence = np.mean([r.confidence for r in self.regime_history])

        # Calculate average transition probability
        avg_transition_prob = np.mean(
            [r.transition_probability for r in self.regime_history]
        )

        return {
            "total_detections": len(self.regime_history),
            "regime_distribution": regime_counts,
            "average_confidence": avg_confidence,
            "average_transition_probability": avg_transition_prob,
            "recent_regime": (
                self.regime_history[-1].regime.value
                if self.regime_history
                else "unknown"
            ),
        }

    def update_performance_by_regime(
        self, regime: MarketRegime, performance_metrics: Dict[str, float]
    ):
        """Update performance metrics for a specific regime."""
        if regime not in self.performance_by_regime:
            self.performance_by_regime[regime] = {}

        # Update with exponential moving average
        alpha = 0.1
        for metric, value in performance_metrics.items():
            if metric in self.performance_by_regime[regime]:
                self.performance_by_regime[regime][metric] = (
                    alpha * value
                    + (1 - alpha) * self.performance_by_regime[regime][metric]
                )
            else:
                self.performance_by_regime[regime][metric] = value

        self.logger.debug(
            f"Updated performance for {regime.value} regime: {performance_metrics}"
        )


# Convenience function for creating regime detection agent


def create_regime_detection_agent() -> RegimeDetectionAgent:
    """Create a configured regime detection agent."""
    return RegimeDetectionAgent()
