"""
Signal Scorer - Batch 16
Rolling score decay with exponential weighting for shorter test windows
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DecayMethod(Enum):
    """Methods for score decay calculation."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"


@dataclass
class ScoreRecord:
    """Record of a score measurement."""

    timestamp: datetime
    scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecayConfig:
    """Configuration for score decay."""

    method: DecayMethod = DecayMethod.EXPONENTIAL
    span: int = 5  # ewm span parameter
    min_window_size: int = 3
    decay_factor: float = 0.9
    enable_trend_analysis: bool = True


class SignalScorer:
    """
    Signal scorer with rolling score decay capabilities.

    Features:
    - Rolling score decay with exponential weighting
    - Multiple decay methods (exponential, linear, step)
    - Score history tracking
    - Trend analysis
    - Configurable decay parameters
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        """
        Initialize signal scorer.

        Args:
            config: Decay configuration
        """
        self.config = config or DecayConfig()
        self.score_history: Dict[str, List[ScoreRecord]] = {}
        self.decay_weights: Dict[str, np.ndarray] = {}

        logger.info(f"SignalScorer initialized with {self.config.method.value} decay")

    def add_score(
        self,
        signal_name: str,
        scores: Dict[str, float],
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a score measurement.

        Args:
            signal_name: Name of the signal
            scores: Dictionary of metric scores
            timestamp: Timestamp for the measurement
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()

        if metadata is None:
            metadata = {}

        record = ScoreRecord(
            timestamp=timestamp, scores=scores.copy(), metadata=metadata
        )

        if signal_name not in self.score_history:
            self.score_history[signal_name] = []

        self.score_history[signal_name].append(record)

        # Sort by timestamp
        self.score_history[signal_name].sort(key=lambda x: x.timestamp)

        logger.debug(f"Added score for {signal_name}: {scores}")

    def get_decayed_score(
        self,
        signal_name: str,
        metric: str,
        current_score: float,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """
        Get decayed score using rolling decay.

        Args:
            signal_name: Name of the signal
            metric: Name of the metric
            current_score: Current score value
            timestamp: Current timestamp

        Returns:
            Decayed score value
        """
        if timestamp is None:
            timestamp = datetime.now()

        if signal_name not in self.score_history:
            return current_score

        history = self.score_history[signal_name]
        if len(history) < self.config.min_window_size:
            return current_score

        # Get historical scores for this metric
        metric_scores = []
        timestamps = []

        for record in history:
            if metric in record.scores:
                metric_scores.append(record.scores[metric])
                timestamps.append(record.timestamp)

        if len(metric_scores) < self.config.min_window_size:
            return current_score

        # Apply decay based on method
        if self.config.method == DecayMethod.EXPONENTIAL:
            return self._apply_exponential_decay(metric_scores, current_score)
        elif self.config.method == DecayMethod.LINEAR:
            return self._apply_linear_decay(metric_scores, current_score)
        elif self.config.method == DecayMethod.STEP:
            return self._apply_step_decay(metric_scores, current_score)
        else:
            return current_score

    def _apply_exponential_decay(
        self, historical_scores: List[float], current_score: float
    ) -> float:
        """
        Apply exponential decay using pandas ewm.

        Args:
            historical_scores: List of historical scores
            current_score: Current score

        Returns:
            Decayed score
        """
        try:
            # Create series with historical scores
            series = pd.Series(historical_scores + [current_score])

            # Apply exponential weighted mean
            ewm_series = series.ewm(span=self.config.span).mean()

            # Return the decayed current score
            return float(ewm_series.iloc[-1])

        except Exception as e:
            logger.error(f"Error applying exponential decay: {e}")
            return current_score

    def _apply_linear_decay(
        self, historical_scores: List[float], current_score: float
    ) -> float:
        """
        Apply linear decay.

        Args:
            historical_scores: List of historical scores
            current_score: Current score

        Returns:
            Decayed score
        """
        if not historical_scores:
            return current_score

        # Calculate weights based on position (newer = higher weight)
        n = len(historical_scores)
        weights = np.linspace(0.1, 1.0, n + 1)  # +1 for current score

        # Apply weights
        weighted_sum = sum(
            score * weight for score, weight in zip(historical_scores, weights[:-1])
        )
        weighted_sum += current_score * weights[-1]

        # Safely calculate weighted average with division-by-zero protection
        weights_total = weights.sum()
        if weights_total > 1e-10:
            return weighted_sum / weights_total
        else:
            # If no weights, return simple average or current score
            return current_score

    def _apply_step_decay(
        self, historical_scores: List[float], current_score: float
    ) -> float:
        """
        Apply step decay with configurable decay factor.

        Args:
            historical_scores: List of historical scores
            current_score: Current score

        Returns:
            Decayed score
        """
        if not historical_scores:
            return current_score

        # Apply decay factor to historical scores
        decayed_historical = [
            score * (self.config.decay_factor ** (i + 1))
            for i, score in enumerate(reversed(historical_scores))
        ]

        # Combine with current score
        all_scores = decayed_historical + [current_score]
        return np.mean(all_scores)

    def get_score_trend(
        self, signal_name: str, metric: str, window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get trend analysis for a signal metric.

        Args:
            signal_name: Name of the signal
            metric: Name of the metric
            window: Number of recent points to analyze

        Returns:
            Dictionary with trend information
        """
        if not self.config.enable_trend_analysis:
            return {}

        if signal_name not in self.score_history:
            return {}

        history = self.score_history[signal_name]
        if len(history) < 2:
            return {}

        # Get recent scores
        recent_scores = []
        recent_timestamps = []

        for record in history:
            if metric in record.scores:
                recent_scores.append(record.scores[metric])
                recent_timestamps.append(record.timestamp)

        if len(recent_scores) < 2:
            return {}

        # Apply window if specified
        if window and len(recent_scores) > window:
            recent_scores = recent_scores[-window:]
            recent_timestamps = recent_timestamps[-window:]

        # Calculate trend
        if len(recent_scores) >= 2:
            # Linear regression for trend
            x = np.arange(len(recent_scores))
            y = np.array(recent_scores)

            # Calculate slope and intercept
            slope, intercept = np.polyfit(x, y, 1)

            # Calculate R-squared with safe division
            y_pred = slope * x + intercept
            ss_total = np.sum((y - np.mean(y)) ** 2)
            if ss_total > 1e-10:
                r_squared = 1 - np.sum((y - y_pred) ** 2) / ss_total
            else:
                # Perfect fit if all y values are same
                r_squared = 1.0 if np.allclose(y, y_pred) else 0.0

            return {
                "trend_slope": float(slope),
                "trend_intercept": float(intercept),
                "r_squared": float(r_squared),
                "current_value": float(recent_scores[-1]),
                "data_points": len(recent_scores),
                "time_span": (
                    recent_timestamps[-1] - recent_timestamps[0]
                ).total_seconds(),
                "trend_direction": (
                    "increasing"
                    if slope > 0
                    else "decreasing"
                    if slope < 0
                    else "stable"
                ),
            }

        return {}

    def get_signal_summary(self, signal_name: str) -> Dict[str, Any]:
        """
        Get summary statistics for a signal.

        Args:
            signal_name: Name of the signal

        Returns:
            Dictionary with summary statistics
        """
        if signal_name not in self.score_history:
            return {}

        history = self.score_history[signal_name]
        if not history:
            return {}

        # Collect all metrics
        all_metrics = set()
        for record in history:
            all_metrics.update(record.scores.keys())

        summary = {
            "signal_name": signal_name,
            "total_measurements": len(history),
            "metrics": list(all_metrics),
            "first_measurement": history[0].timestamp.isoformat(),
            "last_measurement": history[-1].timestamp.isoformat(),
            "time_span_hours": (
                history[-1].timestamp - history[0].timestamp
            ).total_seconds()
            / 3600,
        }

        # Add statistics for each metric
        metric_stats = {}
        for metric in all_metrics:
            scores = [
                record.scores[metric] for record in history if metric in record.scores
            ]
            if scores:
                metric_stats[metric] = {
                    "count": len(scores),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "current": float(scores[-1]),
                }

        summary["metric_statistics"] = metric_stats
        return summary

    def clear_history(self, signal_name: Optional[str] = None):
        """
        Clear score history.

        Args:
            signal_name: Specific signal to clear, or None for all
        """
        if signal_name is None:
            self.score_history.clear()
            logger.info("Cleared all score history")
        elif signal_name in self.score_history:
            del self.score_history[signal_name]
            logger.info(f"Cleared score history for {signal_name}")

    def export_history(self, signal_name: str, format: str = "json") -> str:
        """
        Export score history.

        Args:
            signal_name: Name of the signal
            format: Export format ('json' or 'csv')

        Returns:
            Exported data as string
        """
        if signal_name not in self.score_history:
            return ""

        history = self.score_history[signal_name]

        if format == "json":
            import json

            data = []
            for record in history:
                data.append(
                    {
                        "timestamp": record.timestamp.isoformat(),
                        "scores": record.scores,
                        "metadata": record.metadata,
                    }
                )
            return json.dumps(data, indent=2)

        elif format == "csv":
            # Convert to DataFrame and export
            data = []
            for record in history:
                row = {"timestamp": record.timestamp}
                row.update(record.scores)
                data.append(row)

            df = pd.DataFrame(data)
            return df.to_csv(index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def update_config(self, new_config: DecayConfig):
        """Update decay configuration."""
        self.config = new_config
        logger.info(
            f"Updated decay config: {new_config.method.value}, span={new_config.span}"
        )


def create_signal_scorer(config: Optional[DecayConfig] = None) -> SignalScorer:
    """Factory function to create a signal scorer."""
    return SignalScorer(config)
