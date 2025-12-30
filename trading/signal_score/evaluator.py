"""
Signal Score Evaluator

This module provides comprehensive signal evaluation and scoring
for various trading strategies with confidence weighting and
composite scoring capabilities.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class StrategyType(Enum):
    """Supported strategy types."""

    RSI = "RSI"
    SMA = "SMA"
    MACD = "MACD"
    BB = "BB"  # Bollinger Bands
    CUSTOM = "custom"


@dataclass
class SignalScore:
    """Signal score with metadata."""

    signal_type: SignalType
    score: float
    confidence: float
    strategy: str
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of signal evaluation."""

    signal_scores: List[SignalScore]
    composite_score: float
    recommended_action: SignalType
    confidence: float
    evaluation_time: float
    warnings: List[str] = field(default_factory=list)


class SignalScoreEvaluator:
    """
    Enhanced signal score evaluator with multiple strategy support.

    Features:
    - Support for SMA, BB, and custom strategies
    - Numerical safety with np.nan_to_num()
    - Composite scoring across multiple signals
    - Confidence-weighted aggregation
    """

    def __init__(
        self,
        enable_nan_protection: bool = True,
        default_confidence_threshold: float = 0.3,
        max_score: float = 1.0,
        min_score: float = -1.0,
    ):
        """
        Initialize signal score evaluator.

        Args:
            enable_nan_protection: Enable NaN protection with np.nan_to_num()
            default_confidence_threshold: Minimum confidence for signal consideration
            max_score: Maximum allowed score
            min_score: Minimum allowed score
        """
        self.enable_nan_protection = enable_nan_protection
        self.default_confidence_threshold = default_confidence_threshold
        self.max_score = max_score
        self.min_score = min_score

        # Strategy evaluators
        self.strategy_evaluators = self._initialize_strategy_evaluators()

        # Custom strategy registry
        self.custom_strategies: Dict[str, Callable] = {}

        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []

        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "nan_detected_count": 0,
            "strategy_usage": {},
            "avg_composite_score": 0.0,
        }

        logger.info(
            f"SignalScoreEvaluator initialized with NaN protection: {enable_nan_protection}"
        )

    def _initialize_strategy_evaluators(self) -> Dict[str, Callable]:
        """Initialize built-in strategy evaluators."""
        return {
            StrategyType.RSI.value: self._evaluate_rsi_signal,
            StrategyType.SMA.value: self._evaluate_sma_signal,
            StrategyType.MACD.value: self._evaluate_macd_signal,
            StrategyType.BB.value: self._evaluate_bollinger_signal,
        }

    def evaluate_signal(
        self,
        signal_data: Dict[str, Any],
        strategy_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SignalScore:
        """
        Evaluate a single signal.

        Args:
            signal_data: Signal data dictionary
            strategy_type: Type of strategy
            parameters: Strategy parameters

        Returns:
            SignalScore object
        """
        try:
            # Get evaluator function
            if strategy_type in self.strategy_evaluators:
                evaluator = self.strategy_evaluators[strategy_type]
            elif strategy_type in self.custom_strategies:
                evaluator = self.custom_strategies[strategy_type]
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
                return self._create_default_score(strategy_type)

            # Evaluate signal
            score, confidence, signal_type = evaluator(signal_data, parameters or {})

            # Apply NaN protection
            if self.enable_nan_protection:
                score = self._apply_nan_protection(score)
                confidence = self._apply_nan_protection(confidence)

            # Create signal score
            signal_score = SignalScore(
                signal_type=signal_type,
                score=score,
                confidence=confidence,
                strategy=strategy_type,
                timestamp=datetime.now(),
                parameters=parameters or {},
            )

            # Update statistics
            self._update_strategy_usage(strategy_type)

            return signal_score

        except Exception as e:
            logger.error(f"Error evaluating {strategy_type} signal: {e}")
            return self._create_error_score(strategy_type, str(e))

    def _evaluate_rsi_signal(
        self, signal_data: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Tuple[float, float, SignalType]:
        """Evaluate RSI signal."""
        rsi_value = signal_data.get("rsi", 50.0)
        oversold_threshold = parameters.get("oversold", 30.0)
        overbought_threshold = parameters.get("overbought", 70.0)

        # Apply NaN protection
        if self.enable_nan_protection:
            rsi_value = self._apply_nan_protection(rsi_value)

        # Calculate score
        if rsi_value <= oversold_threshold:
            score = -0.8 + (oversold_threshold - rsi_value) / oversold_threshold * 0.2
            signal_type = SignalType.STRONG_BUY
            confidence = 0.9
        elif rsi_value >= overbought_threshold:
            score = (
                0.8
                + (rsi_value - overbought_threshold)
                / (100 - overbought_threshold)
                * 0.2
            )
            signal_type = SignalType.STRONG_SELL
            confidence = 0.9
        else:
            # Neutral zone
            neutral_center = (oversold_threshold + overbought_threshold) / 2
            distance_from_center = abs(rsi_value - neutral_center)
            max_distance = (overbought_threshold - oversold_threshold) / 2

            if rsi_value < neutral_center:
                score = -0.3 * (distance_from_center / max_distance)
                signal_type = SignalType.BUY
            else:
                score = 0.3 * (distance_from_center / max_distance)
                signal_type = SignalType.SELL

            confidence = 0.6

        return score, confidence, signal_type

    def _evaluate_sma_signal(
        self, signal_data: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Tuple[float, float, SignalType]:
        """Evaluate SMA signal."""
        current_price = signal_data.get("current_price", 100.0)
        sma_short = signal_data.get("sma_short", 100.0)
        sma_long = signal_data.get("sma_long", 100.0)

        # Apply NaN protection
        if self.enable_nan_protection:
            current_price = self._apply_nan_protection(current_price)
            sma_short = self._apply_nan_protection(sma_short)
            sma_long = self._apply_nan_protection(sma_long)

        # Calculate price position relative to SMAs
        short_ratio = current_price / sma_short if sma_short > 0 else 1.0
        long_ratio = current_price / sma_long if sma_long > 0 else 1.0

        # Determine signal
        if short_ratio > 1.02 and long_ratio > 1.02:
            # Strong uptrend
            score = 0.8
            signal_type = SignalType.STRONG_BUY
            confidence = 0.85
        elif short_ratio < 0.98 and long_ratio < 0.98:
            # Strong downtrend
            score = -0.8
            signal_type = SignalType.STRONG_SELL
            confidence = 0.85
        elif short_ratio > 1.01 and long_ratio > 1.0:
            # Weak uptrend
            score = 0.4
            signal_type = SignalType.BUY
            confidence = 0.7
        elif short_ratio < 0.99 and long_ratio < 1.0:
            # Weak downtrend
            score = -0.4
            signal_type = SignalType.SELL
            confidence = 0.7
        else:
            # No clear trend
            score = 0.0
            signal_type = SignalType.HOLD
            confidence = 0.5

        return score, confidence, signal_type

    def _evaluate_macd_signal(
        self, signal_data: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Tuple[float, float, SignalType]:
        """Evaluate MACD signal."""
        macd_line = signal_data.get("macd", 0.0)
        signal_line = signal_data.get("signal", 0.0)
        histogram = signal_data.get("histogram", 0.0)

        # Apply NaN protection
        if self.enable_nan_protection:
            macd_line = self._apply_nan_protection(macd_line)
            signal_line = self._apply_nan_protection(signal_line)
            histogram = self._apply_nan_protection(histogram)

        # Determine signal based on MACD crossover
        if macd_line > signal_line and histogram > 0:
            # Bullish crossover
            if histogram > 0.5:  # Strong bullish
                score = 0.8
                signal_type = SignalType.STRONG_BUY
                confidence = 0.85
            else:  # Weak bullish
                score = 0.4
                signal_type = SignalType.BUY
                confidence = 0.7
        elif macd_line < signal_line and histogram < 0:
            # Bearish crossover
            if histogram < -0.5:  # Strong bearish
                score = -0.8
                signal_type = SignalType.STRONG_SELL
                confidence = 0.85
            else:  # Weak bearish
                score = -0.4
                signal_type = SignalType.SELL
                confidence = 0.7
        else:
            # No clear signal
            score = 0.0
            signal_type = SignalType.HOLD
            confidence = 0.5

        return score, confidence, signal_type

    def _evaluate_bollinger_signal(
        self, signal_data: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Tuple[float, float, SignalType]:
        """Evaluate Bollinger Bands signal."""
        current_price = signal_data.get("current_price", 100.0)
        upper_band = signal_data.get("upper_band", 110.0)
        lower_band = signal_data.get("lower_band", 90.0)
        middle_band = signal_data.get("middle_band", 100.0)

        # Apply NaN protection
        if self.enable_nan_protection:
            current_price = self._apply_nan_protection(current_price)
            upper_band = self._apply_nan_protection(upper_band)
            lower_band = self._apply_nan_protection(lower_band)
            middle_band = self._apply_nan_protection(middle_band)

        # Calculate position within bands
        band_width = upper_band - lower_band
        if band_width > 0:
            position = (current_price - lower_band) / band_width
        else:
            position = 0.5

        # Determine signal
        if position <= 0.1:  # Near lower band
            score = 0.8
            signal_type = SignalType.STRONG_BUY
            confidence = 0.85
        elif position >= 0.9:  # Near upper band
            score = -0.8
            signal_type = SignalType.STRONG_SELL
            confidence = 0.85
        elif position <= 0.3:  # Below middle
            score = 0.4
            signal_type = SignalType.BUY
            confidence = 0.7
        elif position >= 0.7:  # Above middle
            score = -0.4
            signal_type = SignalType.SELL
            confidence = 0.7
        else:
            # Middle range
            score = 0.0
            signal_type = SignalType.HOLD
            confidence = 0.5

        return score, confidence, signal_type

    def _apply_nan_protection(
        self, value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Apply NaN protection to values."""
        if isinstance(value, np.ndarray):
            if np.any(np.isnan(value)):
                self.stats["nan_detected_count"] += 1
                return np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
            return value
        else:
            if np.isnan(value):
                self.stats["nan_detected_count"] += 1
                return 0.0
            return value

    def _create_default_score(self, strategy_type: str) -> SignalScore:
        """Create a default score for unknown strategies."""
        return SignalScore(
            signal_type=SignalType.HOLD,
            score=0.0,
            confidence=0.0,
            strategy=strategy_type,
            timestamp=datetime.now(),
            parameters={},
        )

    def _create_error_score(
        self, strategy_type: str, error_message: str
    ) -> SignalScore:
        """Create an error score when evaluation fails."""
        return SignalScore(
            signal_type=SignalType.HOLD,
            score=0.0,
            confidence=0.0,
            strategy=strategy_type,
            timestamp=datetime.now(),
            parameters={"error": error_message},
        )

    def evaluate_multiple_signals(
        self, signals: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate multiple signals and return composite result.

        Args:
            signals: List of signal dictionaries

        Returns:
            EvaluationResult with composite scoring
        """
        start_time = datetime.now()
        signal_scores = []
        warnings = []

        for signal in signals:
            try:
                strategy_type = signal.get("strategy_type", "unknown")
                signal_data = signal.get("signal_data", {})
                parameters = signal.get("parameters", {})

                score = self.evaluate_signal(signal_data, strategy_type, parameters)
                signal_scores.append(score)

            except Exception as e:
                warnings.append(f"Error evaluating signal: {e}")
                logger.error(f"Error in multiple signal evaluation: {e}")

        # Calculate composite score
        (
            composite_score,
            confidence,
            recommended_action,
        ) = self._calculate_composite_score(signal_scores)

        # Create result
        result = EvaluationResult(
            signal_scores=signal_scores,
            composite_score=composite_score,
            recommended_action=recommended_action,
            confidence=confidence,
            evaluation_time=(datetime.now() - start_time).total_seconds(),
            warnings=warnings,
        )

        # Update history and statistics
        self.evaluation_history.append(result)
        self.stats["total_evaluations"] += 1
        self._update_average_score(composite_score)

        return result

    def _calculate_composite_score(
        self, signal_scores: List[SignalScore]
    ) -> Tuple[float, float, SignalType]:
        """Calculate composite score from multiple signal scores."""
        if not signal_scores:
            return 0.0, 0.0, SignalType.HOLD

        # Filter by confidence threshold
        valid_scores = [
            s
            for s in signal_scores
            if s.confidence >= self.default_confidence_threshold
        ]

        if not valid_scores:
            return 0.0, 0.0, SignalType.HOLD

        # Calculate weighted average
        total_weight = sum(s.confidence for s in valid_scores)
        if total_weight == 0:
            return 0.0, 0.0, SignalType.HOLD

        weighted_score = (
            sum(s.score * s.confidence for s in valid_scores) / total_weight
        )
        avg_confidence = total_weight / len(valid_scores)

        # Determine recommended action
        if weighted_score >= 0.6:
            recommended_action = SignalType.STRONG_BUY
        elif weighted_score >= 0.2:
            recommended_action = SignalType.BUY
        elif weighted_score <= -0.6:
            recommended_action = SignalType.STRONG_SELL
        elif weighted_score <= -0.2:
            recommended_action = SignalType.SELL
        else:
            recommended_action = SignalType.HOLD

        return weighted_score, avg_confidence, recommended_action

    def register_custom_strategy(
        self, strategy_name: str, evaluator_function: Callable
    ) -> bool:
        """
        Register a custom strategy evaluator.

        Args:
            strategy_name: Name of the strategy
            evaluator_function: Function that evaluates the strategy

        Returns:
            True if registration successful
        """
        try:
            self.custom_strategies[strategy_name] = evaluator_function
            logger.info(f"Custom strategy '{strategy_name}' registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register custom strategy '{strategy_name}': {e}")
            return False

    def _update_strategy_usage(self, strategy_type: str):
        """Update strategy usage statistics."""
        self.stats["strategy_usage"][strategy_type] = (
            self.stats["strategy_usage"].get(strategy_type, 0) + 1
        )

    def _update_average_score(self, new_score: float):
        """Update average composite score."""
        current_avg = self.stats["avg_composite_score"]
        total_evaluations = self.stats["total_evaluations"]
        self.stats["avg_composite_score"] = (
            current_avg * (total_evaluations - 1) + new_score
        ) / total_evaluations

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "total_evaluations": self.stats["total_evaluations"],
            "nan_detected_count": self.stats["nan_detected_count"],
            "strategy_usage": self.stats["strategy_usage"],
            "avg_composite_score": self.stats["avg_composite_score"],
            "custom_strategies_count": len(self.custom_strategies),
            "evaluation_history_length": len(self.evaluation_history),
        }

    def enable_nan_protection(self, enable: bool = True):
        """Enable or disable NaN protection."""
        self.enable_nan_protection = enable
        logger.info(f"NaN protection {'enabled' if enable else 'disabled'}")


def create_signal_score_evaluator(
    enable_nan_protection: bool = True,
) -> SignalScoreEvaluator:
    """Factory function to create a signal score evaluator."""
    return SignalScoreEvaluator(enable_nan_protection=enable_nan_protection)
