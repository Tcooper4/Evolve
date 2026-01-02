"""
Multi-Model Aggregator

This module provides functionality to aggregate forecasts from multiple models
with dynamic weighting based on rolling accuracy and performance metrics.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance tracking."""

    model_name: str
    mse_history: List[float]
    mae_history: List[float]
    accuracy_history: List[float]
    last_update: datetime
    total_predictions: int
    window_size: int = 30


@dataclass
class AggregatedForecast:
    """Aggregated forecast result."""

    forecast: np.ndarray
    weights: Dict[str, float]
    confidence: float
    model_contributions: Dict[str, np.ndarray]
    aggregation_method: str
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class MultiModelAggregator:
    """
    Multi-model aggregator with dynamic weighting based on rolling accuracy.

    Features:
    - Dynamic weighting based on rolling MSE (1/MSE)
    - Multiple aggregation methods
    - Performance tracking and updating
    - Confidence calculation
    - Model contribution analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-model aggregator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.performance_window = self.config.get("performance_window", 30)
        self.min_predictions = self.config.get("min_predictions", 5)

        # Weighting configuration
        self.weighting_method = self.config.get("weighting_method", "mse_inverse")
        self.smoothing_factor = self.config.get("smoothing_factor", 0.1)
        self.min_weight = self.config.get("min_weight", 0.01)

        # Aggregation configuration
        self.default_method = self.config.get("default_method", "weighted_average")
        self.confidence_calculation = self.config.get(
            "confidence_calculation", "weighted_variance"
        )

        # Storage
        self.aggregation_history: List[AggregatedForecast] = []
        self.forecast_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.actual_history: List[np.ndarray] = []

        self.logger.info("Multi-model aggregator initialized")

    def add_model(self, model_name: str, window_size: Optional[int] = None) -> None:
        """
        Add a model to the aggregator.

        Args:
            model_name: Name of the model
            window_size: Performance tracking window size
        """
        if model_name in self.model_performance:
            self.logger.warning(f"Model {model_name} already exists")
            return

        self.model_performance[model_name] = ModelPerformance(
            model_name=model_name,
            mse_history=[],
            mae_history=[],
            accuracy_history=[],
            last_update=datetime.now(),
            total_predictions=0,
            window_size=window_size or self.performance_window,
        )

        self.logger.info(f"Added model: {model_name}")

    def update_model_performance(
        self,
        model_name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Update model performance with new predictions and actuals.

        Args:
            model_name: Name of the model
            predictions: Model predictions
            actuals: Actual values
            timestamp: Timestamp for the update

        Returns:
            True if successful
        """
        try:
            if model_name not in self.model_performance:
                self.logger.error(f"Model {model_name} not found")
                return False

            if len(predictions) != len(actuals):
                raise ValueError("Predictions and actuals must have the same length")

            # Calculate performance metrics
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))

            # Calculate accuracy (percentage of predictions within 5% of actual) with safe division
            percentage_errors = np.where(
                np.abs(actuals) > 1e-10,
                np.abs((predictions - actuals) / actuals),
                np.inf  # Flag as error if actual is zero
            )
            accuracy = np.mean(percentage_errors < 0.05)

            # Update model performance
            model_perf = self.model_performance[model_name]
            model_perf.mse_history.append(mse)
            model_perf.mae_history.append(mae)
            model_perf.accuracy_history.append(accuracy)
            model_perf.total_predictions += len(predictions)
            model_perf.last_update = timestamp or datetime.now()

            # Maintain window size
            if len(model_perf.mse_history) > model_perf.window_size:
                model_perf.mse_history = model_perf.mse_history[
                    -model_perf.window_size :
                ]
                model_perf.mae_history = model_perf.mae_history[
                    -model_perf.window_size :
                ]
                model_perf.accuracy_history = model_perf.accuracy_history[
                    -model_perf.window_size :
                ]

            # Store for history
            self.forecast_history[model_name].append(predictions)
            if len(self.actual_history) == 0 or len(self.actual_history[-1]) != len(
                actuals
            ):
                self.actual_history.append(actuals)

            self.logger.info(
                f"Updated {model_name} performance: MSE={mse:.6f}, "
                f"MAE={mae:.6f}, Accuracy={accuracy:.2%}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error updating model performance for {model_name}: {e}")
            return False

    def calculate_weights(self, method: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate model weights based on performance.

        Args:
            method: Weighting method ('mse_inverse', 'accuracy', 'hybrid')

        Returns:
            Dictionary of model weights
        """
        method = method or self.weighting_method
        weights = {}

        try:
            for model_name, perf in self.model_performance.items():
                if perf.total_predictions < self.min_predictions:
                    weights[model_name] = self.min_weight
                    continue

                if method == "mse_inverse":
                    # Weight based on inverse MSE (1/MSE)
                    if perf.mse_history:
                        avg_mse = np.mean(perf.mse_history)
                        weight = 1.0 / (
                            avg_mse + 1e-8
                        )  # Add small epsilon to avoid division by zero
                    else:
                        weight = self.min_weight

                elif method == "accuracy":
                    # Weight based on accuracy
                    if perf.accuracy_history:
                        weight = np.mean(perf.accuracy_history)
                    else:
                        weight = self.min_weight

                elif method == "hybrid":
                    # Hybrid weighting: combination of MSE and accuracy
                    if perf.mse_history and perf.accuracy_history:
                        mse_weight = 1.0 / (np.mean(perf.mse_history) + 1e-8)
                        accuracy_weight = np.mean(perf.accuracy_history)
                        weight = 0.7 * mse_weight + 0.3 * accuracy_weight
                    else:
                        weight = self.min_weight

                elif method == "equal":
                    # Equal weighting
                    weight = 1.0

                else:
                    raise ValueError(f"Unknown weighting method: {method}")

                weights[model_name] = max(weight, self.min_weight)

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {
                    name: weight / total_weight for name, weight in weights.items()
                }

            # Apply smoothing
            if hasattr(self, "_previous_weights") and self._previous_weights:
                for name in weights:
                    if name in self._previous_weights:
                        weights[name] = (
                            self.smoothing_factor * weights[name]
                            + (1 - self.smoothing_factor) * self._previous_weights[name]
                        )

            self._previous_weights = weights.copy()

            self.logger.info(f"Calculated weights using {method}: {weights}")
            return weights

        except Exception as e:
            self.logger.error(f"Error calculating weights: {e}")
            # Return equal weights as fallback
            return {
                name: 1.0 / len(self.model_performance)
                for name in self.model_performance.keys()
            }

    def aggregate_forecasts(
        self,
        forecasts: Dict[str, np.ndarray],
        method: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> AggregatedForecast:
        """
        Aggregate forecasts from multiple models.

        Args:
            forecasts: Dictionary of model forecasts
            method: Aggregation method
            weights: Optional custom weights

        Returns:
            AggregatedForecast object
        """
        try:
            if not forecasts:
                raise ValueError("No forecasts provided")

            method = method or self.default_method
            weights = weights or self.calculate_weights()

            # Validate forecasts
            forecast_lengths = [len(f) for f in forecasts.values()]
            if len(set(forecast_lengths)) > 1:
                raise ValueError("All forecasts must have the same length")

            forecast_length = forecast_lengths[0]

            # Perform aggregation
            if method == "weighted_average":
                aggregated_forecast, model_contributions = self._weighted_average(
                    forecasts, weights, forecast_length
                )
            elif method == "median":
                aggregated_forecast, model_contributions = self._median_aggregation(
                    forecasts, weights, forecast_length
                )
            elif method == "trimmed_mean":
                (
                    aggregated_forecast,
                    model_contributions,
                ) = self._trimmed_mean_aggregation(forecasts, weights, forecast_length)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            # Calculate confidence
            confidence = self._calculate_confidence(forecasts, weights, method)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                forecasts, weights
            )

            # Create result
            result = AggregatedForecast(
                forecast=aggregated_forecast,
                weights=weights,
                confidence=confidence,
                model_contributions=model_contributions,
                aggregation_method=method,
                performance_metrics=performance_metrics,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "num_models": len(forecasts),
                    "forecast_length": forecast_length,
                },
            )

            # Store in history
            self.aggregation_history.append(result)

            self.logger.info(
                f"Aggregated forecasts from {len(forecasts)} models using {method}: "
                f"confidence={confidence:.3f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error aggregating forecasts: {e}")
            raise

    def _weighted_average(
        self,
        forecasts: Dict[str, np.ndarray],
        weights: Dict[str, float],
        forecast_length: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Perform weighted average aggregation."""
        aggregated = np.zeros(forecast_length)
        contributions = {}

        for model_name, forecast in forecasts.items():
            weight = weights.get(model_name, 0.0)
            contribution = weight * forecast
            aggregated += contribution
            contributions[model_name] = contribution

        return aggregated, contributions

    def _median_aggregation(
        self,
        forecasts: Dict[str, np.ndarray],
        weights: Dict[str, float],
        forecast_length: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Perform median aggregation."""
        # Stack all forecasts
        forecast_matrix = np.array(list(forecasts.values()))
        aggregated = np.median(forecast_matrix, axis=0)

        # Calculate contributions based on distance from median
        contributions = {}
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            weight = weights.get(model_name, 0.0)
            distance = np.abs(forecast - aggregated)
            contribution = weight * (
                aggregated + distance * np.sign(forecast - aggregated)
            )
            contributions[model_name] = contribution

        return aggregated, contributions

    def _trimmed_mean_aggregation(
        self,
        forecasts: Dict[str, np.ndarray],
        weights: Dict[str, float],
        forecast_length: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Perform trimmed mean aggregation."""
        # Stack all forecasts
        forecast_matrix = np.array(list(forecasts.values()))

        # Calculate trimmed mean (remove top and bottom 10%)
        trim_percent = 0.1
        aggregated = np.zeros(forecast_length)

        for i in range(forecast_length):
            values = forecast_matrix[:, i]
            sorted_indices = np.argsort(values)
            trim_count = int(len(values) * trim_percent)
            trimmed_values = values[sorted_indices[trim_count:-trim_count]]
            aggregated[i] = np.mean(trimmed_values)

        # Calculate contributions
        contributions = {}
        for model_name, forecast in forecasts.items():
            weight = weights.get(model_name, 0.0)
            contribution = weight * forecast
            contributions[model_name] = contribution

        return aggregated, contributions

    def _calculate_confidence(
        self, forecasts: Dict[str, np.ndarray], weights: Dict[str, float], method: str
    ) -> float:
        """Calculate confidence in the aggregated forecast."""
        try:
            if len(forecasts) < 2:
                return 0.5

            # Calculate weighted variance
            forecast_matrix = np.array(list(forecasts.values()))
            weighted_mean = np.average(
                forecast_matrix, axis=0, weights=list(weights.values())
            )

            # Calculate weighted variance
            weighted_variance = np.average(
                (forecast_matrix - weighted_mean) ** 2,
                axis=0,
                weights=list(weights.values()),
            )

            # Convert variance to confidence (inverse relationship)
            avg_variance = np.mean(weighted_variance)
            confidence = 1.0 / (1.0 + avg_variance)

            # Normalize to [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            return confidence

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _calculate_performance_metrics(
        self, forecasts: Dict[str, np.ndarray], weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance metrics for the aggregation."""
        try:
            metrics = {}

            # Calculate weighted average of individual model metrics
            total_mse = 0.0
            total_mae = 0.0
            total_accuracy = 0.0
            total_weight = 0.0

            for model_name, weight in weights.items():
                if model_name in self.model_performance:
                    perf = self.model_performance[model_name]
                    if perf.mse_history:
                        total_mse += weight * np.mean(perf.mse_history)
                        total_mae += weight * np.mean(perf.mae_history)
                        total_accuracy += weight * np.mean(perf.accuracy_history)
                        total_weight += weight

            if total_weight > 0:
                metrics["weighted_mse"] = total_mse / total_weight
                metrics["weighted_mae"] = total_mae / total_weight
                metrics["weighted_accuracy"] = total_accuracy / total_weight

            # Calculate forecast diversity
            if len(forecasts) > 1:
                forecast_matrix = np.array(list(forecasts.values()))
                diversity = np.mean(np.std(forecast_matrix, axis=0))
                metrics["forecast_diversity"] = diversity

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def get_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """Get performance data for a specific model."""
        return self.model_performance.get(model_name)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all model performances."""
        summary = {"total_models": len(self.model_performance), "models": {}}

        for model_name, perf in self.model_performance.items():
            summary["models"][model_name] = {
                "total_predictions": perf.total_predictions,
                "avg_mse": np.mean(perf.mse_history) if perf.mse_history else None,
                "avg_mae": np.mean(perf.mae_history) if perf.mae_history else None,
                "avg_accuracy": (
                    np.mean(perf.accuracy_history) if perf.accuracy_history else None
                ),
                "last_update": perf.last_update.isoformat(),
                "window_size": perf.window_size,
            }

        return summary

    def clear_history(self) -> None:
        """Clear aggregation history."""
        self.aggregation_history.clear()
        self.forecast_history.clear()
        self.actual_history.clear()
        self.logger.info("Aggregation history cleared")


def create_multi_model_aggregator(
    config: Optional[Dict[str, Any]] = None,
) -> MultiModelAggregator:
    """Create a multi-model aggregator instance."""
    return MultiModelAggregator(config)
