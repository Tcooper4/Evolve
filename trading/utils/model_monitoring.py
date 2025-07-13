"""
Model monitoring utilities for tracking model performance and detecting drift.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    rmse: float
    mae: float
    mape: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    timestamp: datetime
    data_size: int
    training_time: float
    inference_time: float


@dataclass
class DriftAlert:
    """Model drift alert."""

    model_id: str
    alert_type: str  # 'performance_degradation', 'data_drift', 'concept_drift'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    metrics: Dict[str, float]
    threshold: float
    current_value: float


class ModelMonitor:
    """Monitor model performance and detect drift."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model monitor."""
        self.config = config or {}
        self.metrics_history = {}
        self.drift_alerts = []
        self.performance_thresholds = {"accuracy": 0.6, "sharpe_ratio": 0.5, "max_drawdown": -0.2, "win_rate": 0.45}
        self.drift_thresholds = {"performance_degradation": 0.1, "data_drift": 0.15, "concept_drift": 0.2}

        # Ensure metrics directory exists
        self.metrics_dir = "metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)

        logger.info("ModelMonitor initialized successfully")

    def record_metrics(self, model_id: str, metrics: Dict[str, Any]) -> bool:
        """Record model performance metrics.

        Args:
            model_id: Model identifier
            metrics: Performance metrics dictionary

        Returns:
            True if metrics recorded successfully
        """
        try:
            # Create metrics object
            model_metrics = ModelMetrics(
                model_id=model_id,
                accuracy=metrics.get("accuracy", 0.0),
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                f1_score=metrics.get("f1_score", 0.0),
                rmse=metrics.get("rmse", 0.0),
                mae=metrics.get("mae", 0.0),
                mape=metrics.get("mape", 0.0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                max_drawdown=metrics.get("max_drawdown", 0.0),
                win_rate=metrics.get("win_rate", 0.0),
                profit_factor=metrics.get("profit_factor", 0.0),
                timestamp=datetime.now(),
                data_size=metrics.get("data_size", 0),
                training_time=metrics.get("training_time", 0.0),
                inference_time=metrics.get("inference_time", 0.0),
            )

            # Store in memory
            if model_id not in self.metrics_history:
                self.metrics_history[model_id] = []
            self.metrics_history[model_id].append(model_metrics)

            # Save to file
            self._save_metrics_to_file(model_id, model_metrics)

            # Check for drift
            drift_alerts = self._check_for_drift(model_id, model_metrics)
            self.drift_alerts.extend(drift_alerts)

            logger.info(f"Metrics recorded for model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Error recording metrics for model {model_id}: {e}")
            return False

    def get_model_metrics(self, model_id: str, days_back: int = 30) -> List[ModelMetrics]:
        """Get model metrics history.

        Args:
            model_id: Model identifier
            days_back: Number of days to look back

        Returns:
            List of model metrics
        """
        try:
            # Load from file if not in memory
            if model_id not in self.metrics_history:
                self._load_metrics_from_file(model_id)

            if model_id not in self.metrics_history:
                return []

            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_metrics = [m for m in self.metrics_history[model_id] if m.timestamp >= cutoff_date]

            return recent_metrics

        except Exception as e:
            logger.error(f"Error getting metrics for model {model_id}: {e}")
            return []

    def check_model_health(self, model_id: str) -> Dict[str, Any]:
        """Check overall model health.

        Args:
            model_id: Model identifier

        Returns:
            Health status dictionary
        """
        try:
            recent_metrics = self.get_model_metrics(model_id, days_back=7)

            if not recent_metrics:
                return {
                    "status": "unknown",
                    "message": "No recent metrics available",
                    "last_update": None,
                    "performance_score": 0.0,
                }

            latest_metrics = recent_metrics[-1]

            # Calculate performance score
            performance_score = self._calculate_performance_score(latest_metrics)

            # Determine health status
            if performance_score >= 0.8:
                status = "healthy"
                message = "Model performing well"
            elif performance_score >= 0.6:
                status = "warning"
                message = "Model performance degraded"
            else:
                status = "critical"
                message = "Model needs attention"

            return {
                "status": status,
                "message": message,
                "last_update": latest_metrics.timestamp,
                "performance_score": performance_score,
                "latest_metrics": {
                    "accuracy": latest_metrics.accuracy,
                    "sharpe_ratio": latest_metrics.sharpe_ratio,
                    "max_drawdown": latest_metrics.max_drawdown,
                    "win_rate": latest_metrics.win_rate,
                },
            }

        except Exception as e:
            logger.error(f"Error checking model health for {model_id}: {e}")
            return {
                "status": "error",
                "message": f"Error checking health: {e}",
                "last_update": None,
                "performance_score": 0.0,
            }

    def get_drift_alerts(self, model_id: Optional[str] = None, days_back: int = 7) -> List[DriftAlert]:
        """Get drift alerts.

        Args:
            model_id: Optional model identifier to filter
            days_back: Number of days to look back

        Returns:
            List of drift alerts
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            if model_id:
                alerts = [
                    alert
                    for alert in self.drift_alerts
                    if alert.model_id == model_id and alert.timestamp >= cutoff_date
                ]
            else:
                alerts = [alert for alert in self.drift_alerts if alert.timestamp >= cutoff_date]

            return alerts

        except Exception as e:
            logger.error(f"Error getting drift alerts: {e}")
            return []

    def _save_metrics_to_file(self, model_id: str, metrics: ModelMetrics) -> None:
        """Save metrics to file."""
        try:
            filepath = os.path.join(self.metrics_dir, f"{model_id}_metrics.json")

            # Convert to dictionary
            metrics_dict = {
                "model_id": metrics.model_id,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "mape": metrics.mape,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "timestamp": metrics.timestamp.isoformat(),
                "data_size": metrics.data_size,
                "training_time": metrics.training_time,
                "inference_time": metrics.inference_time,
            }

            # Load existing data
            existing_data = []
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    existing_data = json.load(f)

            # Add new metrics
            existing_data.append(metrics_dict)

            # Save back to file
            with open(filepath, "w") as f:
                json.dump(existing_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")

    def _load_metrics_from_file(self, model_id: str) -> None:
        """Load metrics from file."""
        try:
            filepath = os.path.join(self.metrics_dir, f"{model_id}_metrics.json")

            if not os.path.exists(filepath):
                return

            with open(filepath, "r") as f:
                data = json.load(f)

            # Convert back to ModelMetrics objects
            metrics_list = []
            for item in data:
                metrics = ModelMetrics(
                    model_id=item["model_id"],
                    accuracy=item["accuracy"],
                    precision=item["precision"],
                    recall=item["recall"],
                    f1_score=item["f1_score"],
                    rmse=item["rmse"],
                    mae=item["mae"],
                    mape=item["mape"],
                    sharpe_ratio=item["sharpe_ratio"],
                    max_drawdown=item["max_drawdown"],
                    win_rate=item["win_rate"],
                    profit_factor=item["profit_factor"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    data_size=item["data_size"],
                    training_time=item["training_time"],
                    inference_time=item["inference_time"],
                )
                metrics_list.append(metrics)

            self.metrics_history[model_id] = metrics_list

        except Exception as e:
            logger.error(f"Error loading metrics from file: {e}")

    def _check_for_drift(self, model_id: str, current_metrics: ModelMetrics) -> List[DriftAlert]:
        """Check for model drift."""
        try:
            alerts = []

            # Get historical metrics for comparison
            historical_metrics = self.get_model_metrics(model_id, days_back=30)

            if len(historical_metrics) < 2:
                return alerts

            # Calculate baseline metrics (average of last 10)
            baseline_metrics = historical_metrics[-10:-1]

            # Check performance degradation
            baseline_accuracy = np.mean([m.accuracy for m in baseline_metrics])
            accuracy_degradation = baseline_accuracy - current_metrics.accuracy

            if accuracy_degradation > self.drift_thresholds["performance_degradation"]:
                alerts.append(
                    DriftAlert(
                        model_id=model_id,
                        alert_type="performance_degradation",
                        severity="high" if accuracy_degradation > 0.2 else "medium",
                        message=f"Accuracy degraded by {accuracy_degradation:.3f}",
                        timestamp=datetime.now(),
                        metrics={"accuracy": current_metrics.accuracy},
                        threshold=self.drift_thresholds["performance_degradation"],
                        current_value=accuracy_degradation,
                    )
                )

            # Check for other drift indicators
            baseline_sharpe = np.mean([m.sharpe_ratio for m in baseline_metrics])
            sharpe_degradation = baseline_sharpe - current_metrics.sharpe_ratio

            if sharpe_degradation > 0.3:
                alerts.append(
                    DriftAlert(
                        model_id=model_id,
                        alert_type="performance_degradation",
                        severity="medium",
                        message=f"Sharpe ratio degraded by {sharpe_degradation:.3f}",
                        timestamp=datetime.now(),
                        metrics={"sharpe_ratio": current_metrics.sharpe_ratio},
                        threshold=0.3,
                        current_value=sharpe_degradation,
                    )
                )

            return alerts

        except Exception as e:
            logger.error(f"Error checking for drift: {e}")
            return []

    def _calculate_performance_score(self, metrics: ModelMetrics) -> float:
        """Calculate overall performance score."""
        try:
            # Weighted average of key metrics
            weights = {"accuracy": 0.3, "sharpe_ratio": 0.3, "win_rate": 0.2, "profit_factor": 0.2}

            score = (
                weights["accuracy"] * min(metrics.accuracy, 1.0)
                + weights["sharpe_ratio"] * min(max(metrics.sharpe_ratio / 2.0, 0.0), 1.0)
                + weights["win_rate"] * min(metrics.win_rate, 1.0)
                + weights["profit_factor"] * min(metrics.profit_factor / 3.0, 1.0)
            )

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0


def get_model_monitor(config: Optional[Dict[str, Any]] = None) -> ModelMonitor:
    """Get the model monitor instance."""
    return ModelMonitor(config)
