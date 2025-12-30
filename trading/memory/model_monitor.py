"""Model monitoring utilities for detecting drift and model performance issues.
Enhanced with parameter drift tracking and anomaly detection.
"""

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Try to import scipy
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError as e:
    print("⚠️ scipy not available. Disabling statistical drift detection.")
    print(f"   Missing: {e}")
    stats = None
    SCIPY_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("⚠️ scikit-learn not available. Disabling anomaly detection.")
    print(f"   Missing: {e}")
    IsolationForest = None
    SKLEARN_AVAILABLE = False

from utils.safe_json_saver import safe_save_historical_data

logger = logging.getLogger(__name__)


@dataclass
class ModelParameters:
    """Model parameters with drift tracking."""

    model_name: str
    timestamp: datetime
    parameters: Dict[str, Any]
    parameter_hash: str
    drift_score: float = 0.0
    anomaly_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAlert:
    """Drift detection alert."""

    timestamp: datetime
    model_name: str
    drift_type: str  # 'parameter', 'performance', 'data'
    drift_score: float
    threshold: float
    affected_parameters: List[str]
    severity: str  # 'low', 'medium', 'high'
    details: Dict[str, Any]


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""

    timestamp: datetime
    model_name: str
    anomaly_score: float
    anomaly_type: str  # 'performance', 'parameter', 'behavior'
    confidence: float
    features: List[str]
    details: Dict[str, Any]


class ModelMonitor:
    """Model monitoring class for tracking model performance and detecting issues.
    Enhanced with parameter drift and anomaly detection."""

    def __init__(self, drift_threshold: float = 0.1, anomaly_threshold: float = 0.8):
        """Initialize the model monitor.

        Args:
            drift_threshold: Threshold for drift detection
            anomaly_threshold: Threshold for anomaly detection
        """
        self.logger = logging.getLogger(__name__)
        self.drift_threshold = drift_threshold
        self.anomaly_threshold = anomaly_threshold

        # Trust levels for different models
        self.trust_levels = {
            "lstm": 0.85,
            "xgboost": 0.78,
            "prophet": 0.72,
            "ensemble": 0.91,
            "tcn": 0.68,
            "transformer": 0.82,
        }

        # Performance tracking
        self.performance_history = {}  # model_name -> list of performance records
        self.rolling_averages = {}  # model_name -> rolling average data
        self.performance_alerts = []  # List of performance alerts

        # Parameter tracking
        self.parameter_history = defaultdict(
            list
        )  # model_name -> list of ModelParameters
        self.baseline_parameters = {}  # model_name -> baseline parameters
        self.parameter_drift_scores = defaultdict(
            list
        )  # model_name -> list of drift scores

        # Anomaly detection
        if SKLEARN_AVAILABLE:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        else:
            self.anomaly_detector = None
        self.anomaly_scores = defaultdict(list)  # model_name -> list of anomaly scores
        self.behavior_features = defaultdict(
            list
        )  # model_name -> list of behavior features

        # Drift detection
        self.drift_alerts = []  # List of DriftAlert objects
        self.drift_history = defaultdict(list)  # model_name -> list of drift events

        # Thread safety
        self._lock = threading.RLock()

        # Load historical data
        self._load_monitoring_data()

        logger.info("Model monitor initialized with drift and anomaly detection")

    def _load_monitoring_data(self):
        """Load historical monitoring data from disk."""
        try:
            data_file = Path("data/model_monitoring_history.json")
            if data_file.exists():
                with open(data_file, "r") as f:
                    data = json.load(f)

                # Load parameter history
                for model_name, params_list in data.get(
                    "parameter_history", {}
                ).items():
                    for param_data in params_list:
                        param = ModelParameters(
                            model_name=param_data["model_name"],
                            timestamp=datetime.fromisoformat(param_data["timestamp"]),
                            parameters=param_data["parameters"],
                            parameter_hash=param_data["parameter_hash"],
                            drift_score=param_data.get("drift_score", 0.0),
                            anomaly_score=param_data.get("anomaly_score", 0.0),
                            metadata=param_data.get("metadata", {}),
                        )
                        self.parameter_history[model_name].append(param)

                # Load drift alerts
                for alert_data in data.get("drift_alerts", []):
                    alert = DriftAlert(
                        timestamp=datetime.fromisoformat(alert_data["timestamp"]),
                        model_name=alert_data["model_name"],
                        drift_type=alert_data["drift_type"],
                        drift_score=alert_data["drift_score"],
                        threshold=alert_data["threshold"],
                        affected_parameters=alert_data["affected_parameters"],
                        severity=alert_data["severity"],
                        details=alert_data["details"],
                    )
                    self.drift_alerts.append(alert)

                logger.info(
                    f"Loaded monitoring data: {len(self.parameter_history)} models, {len(self.drift_alerts)} drift alerts"
                )

        except Exception as e:
            logger.warning(f"Failed to load monitoring data: {e}")

    def _save_monitoring_data(self):
        """Save monitoring data to disk."""
        try:
            data_file = Path("data/model_monitoring_history.json")

            data = {
                "parameter_history": {
                    model_name: [
                        {
                            "model_name": param.model_name,
                            "timestamp": param.timestamp.isoformat(),
                            "parameters": param.parameters,
                            "parameter_hash": param.parameter_hash,
                            "drift_score": param.drift_score,
                            "anomaly_score": param.anomaly_score,
                            "metadata": param.metadata,
                        }
                        for param in params_list
                    ]
                    for model_name, params_list in self.parameter_history.items()
                },
                "drift_alerts": [
                    {
                        "timestamp": alert.timestamp.isoformat(),
                        "model_name": alert.model_name,
                        "drift_type": alert.drift_type,
                        "drift_score": alert.drift_score,
                        "threshold": alert.threshold,
                        "affected_parameters": alert.affected_parameters,
                        "severity": alert.severity,
                        "details": alert.details,
                    }
                    for alert in self.drift_alerts
                ],
            }

            # Use safe JSON saving to prevent data loss
            result = safe_save_historical_data(data, data_file)
            if not result["success"]:
                logger.error(f"Failed to save monitoring data: {result['error']}")
            else:
                logger.debug(
                    f"Successfully saved monitoring data: {result['filepath']}"
                )

        except Exception as e:
            logger.error(f"Failed to save monitoring data: {e}")

    def get_model_trust_levels(self) -> Dict[str, float]:
        """Get trust levels for different models.

        Returns:
            Dictionary mapping model names to trust levels (0-1)
        """
        try:
            self.logger.info(f"Model trust levels retrieved: {self.trust_levels}")
            return self.trust_levels
        except Exception as e:
            self.logger.error(f"Error getting model trust levels: {str(e)}")
            return {
                "lstm": 0.5,
                "xgboost": 0.5,
                "prophet": 0.5,
                "ensemble": 0.5,
                "tcn": 0.5,
                "transformer": 0.5,
            }

    def update_trust_level(self, model_name: str, new_trust: float):
        """Update trust level for a specific model.

        Args:
            model_name: Name of the model
            new_trust: New trust level (0-1)
        """
        try:
            self.trust_levels[model_name] = max(0.0, min(1.0, new_trust))
            self.logger.info(f"Updated trust level for {model_name}: {new_trust:.3f}")
        except Exception as e:
            self.logger.error(f"Error updating trust level: {str(e)}")

    def record_model_parameters(
        self,
        model_name: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record model parameters and detect drift.

        Args:
            model_name: Name of the model
            parameters: Model parameters dictionary
            metadata: Additional metadata

        Returns:
            Dictionary with drift detection results
        """
        try:
            with self._lock:
                timestamp = datetime.now()

                # Create parameter hash for comparison
                param_hash = self._hash_parameters(parameters)

                # Create parameter record
                param_record = ModelParameters(
                    model_name=model_name,
                    timestamp=timestamp,
                    parameters=parameters.copy(),
                    parameter_hash=param_hash,
                    metadata=metadata or {},
                )

                # Detect parameter drift
                drift_result = self._detect_parameter_drift(model_name, parameters)
                param_record.drift_score = drift_result.get("drift_score", 0.0)

                # Detect anomalies
                anomaly_result = self._detect_parameter_anomaly(model_name, parameters)
                param_record.anomaly_score = anomaly_result.get("anomaly_score", 0.0)

                # Store parameter record
                self.parameter_history[model_name].append(param_record)

                # Keep only last 100 parameter records
                if len(self.parameter_history[model_name]) > 100:
                    self.parameter_history[model_name] = self.parameter_history[
                        model_name
                    ][-100:]

                # Update drift scores
                self.parameter_drift_scores[model_name].append(param_record.drift_score)
                if len(self.parameter_drift_scores[model_name]) > 100:
                    self.parameter_drift_scores[
                        model_name
                    ] = self.parameter_drift_scores[model_name][-100:]

                # Check for drift alerts
                if param_record.drift_score > self.drift_threshold:
                    self._create_drift_alert(
                        model_name,
                        "parameter",
                        param_record.drift_score,
                        drift_result.get("affected_parameters", []),
                    )

                # Save periodically
                if len(self.parameter_history[model_name]) % 10 == 0:
                    self._save_monitoring_data()

                self.logger.info(
                    f"Recorded parameters for {model_name}, drift_score: {param_record.drift_score:.3f}"
                )

                return {
                    "success": True,
                    "drift_score": param_record.drift_score,
                    "anomaly_score": param_record.anomaly_score,
                    "drift_detected": param_record.drift_score > self.drift_threshold,
                    "anomaly_detected": param_record.anomaly_score
                    > self.anomaly_threshold,
                }

        except Exception as e:
            self.logger.error(f"Error recording model parameters: {e}")
            return {"success": False, "error": str(e)}

    def _hash_parameters(self, parameters: Dict[str, Any]) -> str:
        """Create a hash of parameters for comparison."""
        try:
            import hashlib

            # Sort parameters for consistent hashing
            sorted_params = json.dumps(parameters, sort_keys=True)
            return hashlib.sha256(sorted_params.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"

    def _detect_parameter_drift(
        self, model_name: str, current_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect parameter drift compared to baseline."""
        try:
            if model_name not in self.baseline_parameters:
                # Set baseline if not exists
                self.baseline_parameters[model_name] = current_parameters.copy()
                return {
                    "drift_score": 0.0,
                    "affected_parameters": [],
                    "baseline_set": True,
                }

            baseline = self.baseline_parameters[model_name]
            drift_scores = {}
            affected_parameters = []

            for param_name, current_value in current_parameters.items():
                if param_name in baseline:
                    baseline_value = baseline[param_name]

                    # Calculate drift for numeric parameters
                    if isinstance(current_value, (int, float)) and isinstance(
                        baseline_value, (int, float)
                    ):
                        if baseline_value != 0:
                            drift_pct = abs(current_value - baseline_value) / abs(
                                baseline_value
                            )
                            drift_scores[param_name] = drift_pct

                            if drift_pct > 0.1:  # 10% threshold
                                affected_parameters.append(param_name)
                        else:
                            drift_scores[param_name] = 0.0
                    else:
                        # For non-numeric parameters, check if they changed
                        drift_scores[param_name] = (
                            1.0 if current_value != baseline_value else 0.0
                        )
                        if current_value != baseline_value:
                            affected_parameters.append(param_name)

            # Calculate overall drift score
            overall_drift = (
                np.mean(list(drift_scores.values())) if drift_scores else 0.0
            )

            return {
                "drift_score": overall_drift,
                "affected_parameters": affected_parameters,
                "parameter_drifts": drift_scores,
            }

        except Exception as e:
            self.logger.error(f"Error detecting parameter drift: {e}")
            return {"drift_score": 0.0, "affected_parameters": [], "error": str(e)}

    def _detect_parameter_anomaly(
        self, model_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect parameter anomalies using isolation forest."""
        try:
            # Extract numeric parameters for anomaly detection
            numeric_params = []
            param_names = []

            for name, value in parameters.items():
                if isinstance(value, (int, float)):
                    numeric_params.append(value)
                    param_names.append(name)

            if len(numeric_params) < 2:
                return {"anomaly_score": 0.0, "anomaly_detected": False}

            # Add to behavior features
            self.behavior_features[model_name].append(numeric_params)
            if len(self.behavior_features[model_name]) > 100:
                self.behavior_features[model_name] = self.behavior_features[model_name][
                    -100:
                ]

            # Train anomaly detector if we have enough data
            if len(self.behavior_features[model_name]) >= 10:
                features_array = np.array(self.behavior_features[model_name])

                # Fit isolation forest
                self.anomaly_detector.fit(features_array)

                # Predict anomaly score for current parameters
                current_features = np.array([numeric_params])
                anomaly_score = self.anomaly_detector.decision_function(
                    current_features
                )[0]

                # Convert to 0-1 scale (higher = more anomalous)
                anomaly_score = 1 - (anomaly_score + 0.5)  # Normalize to 0-1

                # Store anomaly score
                self.anomaly_scores[model_name].append(anomaly_score)
                if len(self.anomaly_scores[model_name]) > 100:
                    self.anomaly_scores[model_name] = self.anomaly_scores[model_name][
                        -100:
                    ]

                return {
                    "anomaly_score": anomaly_score,
                    "anomaly_detected": anomaly_score > self.anomaly_threshold,
                    "features_used": param_names,
                }
            else:
                return {
                    "anomaly_score": 0.0,
                    "anomaly_detected": False,
                    "insufficient_data": True,
                }

        except Exception as e:
            self.logger.error(f"Error detecting parameter anomaly: {e}")
            return {"anomaly_score": 0.0, "anomaly_detected": False, "error": str(e)}

    def _create_drift_alert(
        self,
        model_name: str,
        drift_type: str,
        drift_score: float,
        affected_parameters: List[str],
    ):
        """Create a drift alert."""
        try:
            severity = (
                "high"
                if drift_score > 0.5
                else "medium"
                if drift_score > 0.2
                else "low"
            )

            alert = DriftAlert(
                timestamp=datetime.now(),
                model_name=model_name,
                drift_type=drift_type,
                drift_score=drift_score,
                threshold=self.drift_threshold,
                affected_parameters=affected_parameters,
                severity=severity,
                details={"drift_score": drift_score, "threshold": self.drift_threshold},
            )

            self.drift_alerts.append(alert)
            self.drift_history[model_name].append(alert)

            # Keep only last 100 alerts
            if len(self.drift_alerts) > 100:
                self.drift_alerts = self.drift_alerts[-100:]

            self.logger.warning(
                f"Drift alert for {model_name}: {drift_type} drift score {drift_score:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Error creating drift alert: {e}")

    def record_model_performance(
        self, model_name: str, performance_metrics: Dict[str, float]
    ):
        """Record performance metrics for a model and check for drops.

        Args:
            model_name: Name of the model
            performance_metrics: Dictionary of performance metrics
        """
        try:
            with self._lock:
                timestamp = datetime.now()

                # Initialize history if needed
                if model_name not in self.performance_history:
                    self.performance_history[model_name] = []

                # Add performance record
                record = {"timestamp": timestamp, "metrics": performance_metrics.copy()}
                self.performance_history[model_name].append(record)

                # Keep only last 100 records
                if len(self.performance_history[model_name]) > 100:
                    self.performance_history[model_name] = self.performance_history[
                        model_name
                    ][-100:]

                # Check for performance drops
                self._check_performance_drops(model_name, performance_metrics)

                # Check for performance drift
                self._check_performance_drift(model_name, performance_metrics)

                self.logger.info(f"Recorded performance for {model_name}")

        except Exception as e:
            self.logger.error(f"Error recording model performance: {e}")

    def _check_performance_drops(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        threshold_percent: float = 20.0,
        window_size: int = 10,
    ):
        """Check for performance drops compared to rolling average.

        Args:
            model_name: Name of the model
            current_metrics: Current performance metrics
            threshold_percent: Percentage drop threshold to flag
            window_size: Number of recent records to use for rolling average
        """
        try:
            history = self.performance_history.get(model_name, [])

            if len(history) < window_size:
                return  # Need more data for comparison

            # Get recent performance records
            recent_records = history[-window_size:-1]  # Exclude current record

            # Calculate rolling averages for each metric
            rolling_averages = {}
            for metric_name in current_metrics.keys():
                if metric_name in ["timestamp", "last_updated"]:
                    continue

                values = [
                    record["metrics"].get(metric_name, 0) for record in recent_records
                ]
                if values:
                    rolling_averages[metric_name] = np.mean(values)

            # Check for drops
            drops_detected = []
            for metric_name, current_value in current_metrics.items():
                if metric_name in ["timestamp", "last_updated"]:
                    continue

                if metric_name in rolling_averages:
                    rolling_avg = rolling_averages[metric_name]

                    if rolling_avg > 0:  # Avoid division by zero
                        # Calculate percentage change
                        if metric_name in ["mse", "mae", "max_drawdown", "volatility"]:
                            # For these metrics, higher is worse, so check for increases
                            percent_change = (
                                (current_value - rolling_avg) / rolling_avg
                            ) * 100
                            if percent_change > threshold_percent:
                                drops_detected.append(
                                    {
                                        "metric": metric_name,
                                        "current": current_value,
                                        "rolling_avg": rolling_avg,
                                        "percent_change": percent_change,
                                        "direction": "worse",
                                    }
                                )
                        else:
                            # For other metrics, lower is worse, so check for decreases
                            percent_change = (
                                (rolling_avg - current_value) / rolling_avg
                            ) * 100
                            if percent_change > threshold_percent:
                                drops_detected.append(
                                    {
                                        "metric": metric_name,
                                        "current": current_value,
                                        "rolling_avg": rolling_avg,
                                        "percent_change": percent_change,
                                        "direction": "worse",
                                    }
                                )

            # Create alert if drops detected
            if drops_detected:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "alert_type": "performance_drop",
                    "threshold_percent": threshold_percent,
                    "window_size": window_size,
                    "drops_detected": drops_detected,
                    "severity": (
                        "high"
                        if any(d["percent_change"] > 50 for d in drops_detected)
                        else "medium"
                    ),
                }

                self.performance_alerts.append(alert)

                # Log the alert
                drop_summary = ", ".join(
                    [
                        f"{d['metric']}: {d['percent_change']:.1f}%"
                        for d in drops_detected
                    ]
                )
                self.logger.warning(
                    f"Performance drop detected for {model_name}: {drop_summary}"
                )

        except Exception as e:
            self.logger.error(f"Error checking performance drops: {e}")

    def _check_performance_drift(
        self, model_name: str, current_metrics: Dict[str, float]
    ):
        """Check for performance drift over time."""
        try:
            history = self.performance_history.get(model_name, [])

            if len(history) < 20:
                return  # Need more data for drift detection

            # Get baseline performance (first 10 records)
            baseline_records = history[:10]
            baseline_metrics = {}

            for record in baseline_records:
                for metric_name, value in record["metrics"].items():
                    if metric_name not in baseline_metrics:
                        baseline_metrics[metric_name] = []
                    baseline_metrics[metric_name].append(value)

            # Calculate baseline averages
            baseline_averages = {}
            for metric_name, values in baseline_metrics.items():
                if values:
                    baseline_averages[metric_name] = np.mean(values)

            # Calculate drift for current metrics
            drift_scores = {}
            affected_metrics = []

            for metric_name, current_value in current_metrics.items():
                if metric_name in baseline_averages:
                    baseline_value = baseline_averages[metric_name]

                    if baseline_value != 0:
                        drift_pct = abs(current_value - baseline_value) / abs(
                            baseline_value
                        )
                        drift_scores[metric_name] = drift_pct

                        if drift_pct > 0.3:  # 30% threshold for performance drift
                            affected_metrics.append(metric_name)

            # Calculate overall performance drift
            overall_drift = (
                np.mean(list(drift_scores.values())) if drift_scores else 0.0
            )

            if overall_drift > self.drift_threshold:
                self._create_drift_alert(
                    model_name, "performance", overall_drift, affected_metrics
                )

        except Exception as e:
            self.logger.error(f"Error checking performance drift: {e}")

    def get_performance_alerts(
        self,
        model_name: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get performance alerts.

        Args:
            model_name: Filter by model name
            severity: Filter by severity ('high', 'medium', 'low')
            limit: Maximum number of alerts to return

        Returns:
            List of performance alerts
        """
        try:
            alerts = self.performance_alerts.copy()

            # Filter by model name
            if model_name:
                alerts = [a for a in alerts if a.get("model_name") == model_name]

            # Filter by severity
            if severity:
                alerts = [a for a in alerts if a.get("severity") == severity]

            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return alerts[:limit]

        except Exception as e:
            self.logger.error(f"Error getting performance alerts: {e}")
            return []

    def get_drift_alerts(
        self,
        model_name: Optional[str] = None,
        drift_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get drift alerts.

        Args:
            model_name: Filter by model name
            drift_type: Filter by drift type ('parameter', 'performance', 'data')
            severity: Filter by severity ('high', 'medium', 'low')
            limit: Maximum number of alerts to return

        Returns:
            List of drift alerts
        """
        try:
            alerts = []
            for alert in self.drift_alerts:
                alert_dict = {
                    "timestamp": alert.timestamp.isoformat(),
                    "model_name": alert.model_name,
                    "drift_type": alert.drift_type,
                    "drift_score": alert.drift_score,
                    "threshold": alert.threshold,
                    "affected_parameters": alert.affected_parameters,
                    "severity": alert.severity,
                    "details": alert.details,
                }
                alerts.append(alert_dict)

            # Apply filters
            if model_name:
                alerts = [a for a in alerts if a.get("model_name") == model_name]

            if drift_type:
                alerts = [a for a in alerts if a.get("drift_type") == drift_type]

            if severity:
                alerts = [a for a in alerts if a.get("severity") == severity]

            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return alerts[:limit]

        except Exception as e:
            self.logger.error(f"Error getting drift alerts: {e}")
            return []

    def get_rolling_performance_averages(
        self, model_name: str, window_size: int = 10
    ) -> Dict[str, float]:
        """Get rolling performance averages for a model.

        Args:
            model_name: Name of the model
            window_size: Rolling window size

        Returns:
            Dictionary of rolling averages by metric
        """
        try:
            history = self.performance_history.get(model_name, [])

            if len(history) < window_size:
                return {}

            # Get recent records
            recent_records = history[-window_size:]

            # Calculate averages for each metric
            averages = {}
            for record in recent_records:
                for metric_name, value in record["metrics"].items():
                    if metric_name not in averages:
                        averages[metric_name] = []
                    averages[metric_name].append(value)

            # Calculate means
            rolling_averages = {}
            for metric_name, values in averages.items():
                if values:
                    rolling_averages[metric_name] = np.mean(values)

            return rolling_averages

        except Exception as e:
            self.logger.error(f"Error getting rolling averages: {e}")
            return {}

    def get_parameter_drift_summary(self, model_name: str) -> Dict[str, Any]:
        """Get parameter drift summary for a model.

        Args:
            model_name: Name of the model

        Returns:
            Parameter drift summary
        """
        try:
            if model_name not in self.parameter_history:
                return {"message": "No parameter history available"}

            params_history = self.parameter_history[model_name]
            drift_scores = self.parameter_drift_scores.get(model_name, [])

            if not params_history:
                return {"message": "No parameter history available"}

            # Calculate drift statistics
            recent_drift_scores = (
                drift_scores[-10:] if len(drift_scores) >= 10 else drift_scores
            )

            summary = {
                "model_name": model_name,
                "total_parameter_records": len(params_history),
                "current_drift_score": (
                    params_history[-1].drift_score if params_history else 0.0
                ),
                "average_drift_score": np.mean(drift_scores) if drift_scores else 0.0,
                "max_drift_score": np.max(drift_scores) if drift_scores else 0.0,
                "recent_drift_trend": (
                    np.mean(recent_drift_scores) if recent_drift_scores else 0.0
                ),
                "drift_alerts": len(
                    [a for a in self.drift_alerts if a.model_name == model_name]
                ),
                "last_parameter_update": (
                    params_history[-1].timestamp.isoformat() if params_history else None
                ),
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting parameter drift summary: {e}")
            return {"error": str(e)}

    def clear_performance_alerts(self, model_name: Optional[str] = None):
        """Clear performance alerts.

        Args:
            model_name: Clear alerts for specific model (None for all)
        """
        try:
            if model_name:
                self.performance_alerts = [
                    a
                    for a in self.performance_alerts
                    if a.get("model_name") != model_name
                ]
                self.logger.info(f"Cleared performance alerts for {model_name}")
            else:
                self.performance_alerts.clear()
                self.logger.info("Cleared all performance alerts")
        except Exception as e:
            self.logger.error(f"Error clearing performance alerts: {e}")

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        threshold: float = 0.1,
        method: str = "ks_test",
    ) -> Dict[str, Any]:
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available. Cannot perform drift detection.")
            return {"drift_detected": False, "drift_score": 0.0, "method": method}
        """Detect data drift between current and historical data.

        Args:
            current_data: Current data DataFrame
            historical_data: Historical data DataFrame
            threshold: Drift detection threshold
            method: Drift detection method

        Returns:
            Drift detection results
        """
        try:
            if current_data.empty or historical_data.empty:
                return {"error": "Empty data provided"}

            # Ensure same columns
            common_columns = set(current_data.columns) & set(historical_data.columns)
            if not common_columns:
                return {"error": "No common columns between datasets"}

            drift_results = {}
            total_drift_score = 0.0
            drifted_features = []

            for column in common_columns:
                current_col = current_data[column].dropna()
                historical_col = historical_data[column].dropna()

                if len(current_col) < 10 or len(historical_col) < 10:
                    continue  # Need sufficient data

                # Detect drift based on method
                if method == "ks_test":
                    drift_score = self._calculate_ks_drift(current_col, historical_col)
                elif method == "chi_square":
                    drift_score = self._calculate_chi_square_drift(
                        current_col, historical_col
                    )
                elif method == "wasserstein":
                    drift_score = self._calculate_wasserstein_drift(
                        current_col, historical_col
                    )
                else:
                    drift_score = self._calculate_ks_drift(current_col, historical_col)

                drift_results[column] = {
                    "drift_score": drift_score,
                    "drift_detected": drift_score > threshold,
                    "method": method,
                }

                total_drift_score += drift_score

                if drift_score > threshold:
                    drifted_features.append(column)

            # Calculate overall drift
            overall_drift = (
                total_drift_score / len(common_columns) if common_columns else 0.0
            )

            return {
                "overall_drift_score": overall_drift,
                "drift_detected": overall_drift > threshold,
                "drifted_features": drifted_features,
                "feature_drift_scores": drift_results,
                "method": method,
                "threshold": threshold,
            }

        except Exception as e:
            self.logger.error(f"Error detecting drift: {e}")
            return {"error": str(e)}

    def generate_strategy_priority(
        self, performance_metrics: Dict[str, float], market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate strategy priority based on performance and market conditions.

        Args:
            performance_metrics: Model performance metrics
            market_conditions: Current market conditions

        Returns:
            Strategy priority information
        """
        try:
            # Calculate overall performance score
            performance_score = 0.0
            if performance_metrics:
                # Weight different metrics
                weights = {
                    "sharpe_ratio": 0.3,
                    "win_rate": 0.25,
                    "profit_factor": 0.2,
                    "max_drawdown": -0.15,  # Negative weight
                    "volatility": -0.1,  # Negative weight
                }

                for metric, weight in weights.items():
                    if metric in performance_metrics:
                        value = performance_metrics[metric]
                        if metric in ["max_drawdown", "volatility"]:
                            # For negative metrics, lower is better
                            normalized_value = max(0, 1 - abs(value))
                        else:
                            # For positive metrics, higher is better
                            normalized_value = min(1, max(0, value))

                        performance_score += weight * normalized_value

            # Adjust for market conditions
            market_adjustment = 0.0
            if market_conditions:
                volatility = market_conditions.get("volatility", 0.2)
                trend_strength = market_conditions.get("trend_strength", 0.5)

                # Higher volatility may require more conservative strategies
                volatility_penalty = min(0.2, volatility * 0.5)

                # Strong trends may favor trend-following strategies
                trend_bonus = max(-0.1, min(0.1, (trend_strength - 0.5) * 0.2))

                market_adjustment = trend_bonus - volatility_penalty

            # Final priority score
            priority_score = max(0, min(1, performance_score + market_adjustment))

            # Determine priority level
            if priority_score >= 0.8:
                priority_level = "high"
            elif priority_score >= 0.6:
                priority_level = "medium"
            else:
                priority_level = "low"

            return {
                "performance_score": performance_score,
                "market_adjustment": market_adjustment,
                "priority_score": priority_score,
                "priority_level": priority_level,
                "recommendation": f"Strategy priority: {priority_level} ({priority_score:.2f})",
            }

        except Exception as e:
            self.logger.error(f"Error generating strategy priority: {e}")
            return {"error": str(e)}

    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive model performance information.

        Args:
            model_name: Name of the model

        Returns:
            Model performance summary
        """
        try:
            performance_history = self.performance_history.get(model_name, [])
            parameter_history = self.parameter_history.get(model_name, [])

            if not performance_history:
                return {"message": f"No performance data for {model_name}"}

            # Calculate performance statistics
            all_metrics = []
            for record in performance_history:
                all_metrics.append(record["metrics"])

            # Aggregate metrics
            metric_names = set()
            for metrics in all_metrics:
                metric_names.update(metrics.keys())

            performance_stats = {}
            for metric_name in metric_names:
                values = [
                    m.get(metric_name, 0)
                    for m in all_metrics
                    if m.get(metric_name) is not None
                ]
                if values:
                    performance_stats[metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "latest": values[-1] if values else 0,
                    }

            # Get recent performance
            recent_performance = (
                performance_history[-1]["metrics"] if performance_history else {}
            )

            # Get parameter drift info
            parameter_drift = self.get_parameter_drift_summary(model_name)

            return {
                "model_name": model_name,
                "total_performance_records": len(performance_history),
                "total_parameter_records": len(parameter_history),
                "performance_statistics": performance_stats,
                "recent_performance": recent_performance,
                "parameter_drift": parameter_drift,
                "trust_level": self.trust_levels.get(model_name, 0.5),
                "last_updated": (
                    performance_history[-1]["timestamp"].isoformat()
                    if performance_history
                    else None
                ),
            }

        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return {"error": str(e)}


# Global instance
_model_monitor = None
_monitor_lock = threading.RLock()


def get_model_monitor() -> ModelMonitor:
    """Get the global model monitor instance."""
    global _model_monitor
    with _monitor_lock:
        if _model_monitor is None:
            _model_monitor = ModelMonitor()
        return _model_monitor


def detect_drift(
    current_data: pd.DataFrame,
    historical_data: pd.DataFrame,
    threshold: float = 0.1,
    method: str = "ks_test",
) -> Dict[str, Any]:
    """Detect data drift between current and historical data.

    Args:
        current_data: Current data DataFrame
        historical_data: Historical data DataFrame
        threshold: Drift detection threshold
        method: Drift detection method

    Returns:
        Drift detection results
    """
    return get_model_monitor().detect_drift(
        current_data, historical_data, threshold, method
    )


def _calculate_ks_drift(
    current_data: pd.DataFrame, historical_data: pd.DataFrame
) -> float:
    """Calculate Kolmogorov-Smirnov drift score."""
    try:
        statistic, p_value = stats.ks_2samp(current_data, historical_data)
        return statistic  # KS statistic is between 0 and 1
    except Exception:
        return 0.0


def _calculate_chi_square_drift(
    current_data: pd.DataFrame, historical_data: pd.DataFrame
) -> float:
    """Calculate Chi-square drift score."""
    try:
        # Create histograms for comparison
        bins = np.linspace(
            min(current_data.min(), historical_data.min()),
            max(current_data.max(), historical_data.max()),
            10,
        )

        current_hist, _ = np.histogram(current_data, bins=bins)
        historical_hist, _ = np.histogram(historical_data, bins=bins)

        # Normalize histograms
        current_hist = current_hist / current_hist.sum()
        historical_hist = historical_hist / historical_hist.sum()

        # Calculate chi-square statistic
        chi_square = np.sum(
            (current_hist - historical_hist) ** 2 / (historical_hist + 1e-8)
        )
        return min(1.0, chi_square / 100)  # Normalize to 0-1
    except Exception:
        return 0.0


def _calculate_wasserstein_drift(
    current_data: pd.DataFrame, historical_data: pd.DataFrame
) -> float:
    """Calculate Wasserstein distance drift score."""
    try:
        from scipy.stats import wasserstein_distance

        distance = wasserstein_distance(current_data, historical_data)

        # Normalize by data range
        data_range = max(current_data.max(), historical_data.max()) - min(
            current_data.min(), historical_data.min()
        )
        normalized_distance = distance / (data_range + 1e-8)

        return min(1.0, normalized_distance)
    except Exception:
        return 0.0


def generate_strategy_priority(
    performance_metrics: Dict[str, float], market_conditions: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate strategy priority based on performance and market conditions."""
    return get_model_monitor().generate_strategy_priority(
        performance_metrics, market_conditions
    )


def get_default_model_trust_levels() -> Dict[str, float]:
    """Get default model trust levels."""
    return {
        "lstm": 0.85,
        "xgboost": 0.78,
        "prophet": 0.72,
        "ensemble": 0.91,
        "tcn": 0.68,
        "transformer": 0.82,
    }
