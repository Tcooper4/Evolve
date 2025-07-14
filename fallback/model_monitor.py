"""
Fallback Model Monitor Implementation

Provides fallback functionality for model performance monitoring when
the primary model monitor is unavailable.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FallbackModelMonitor:
    """
    Fallback implementation of the Model Monitor.

    Provides basic model performance tracking and trust level assessment
    when the primary model monitor is unavailable.
    """

    def __init__(self) -> None:
        """
        Initialize the fallback model monitor.

        Sets up basic logging and initializes mock performance data for
        fallback operations.
        """
        self._status = "fallback"
        self._mock_performance = self._initialize_mock_performance()
        logger.info("FallbackModelMonitor initialized")

    def _initialize_mock_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize mock performance data for fallback models.

        Returns:
            Dict[str, Dict[str, float]]: Mock performance metrics
        """
        return {
            "lstm": {
                "mse": 0.15,
                "mae": 0.12,
                "rmse": 0.39,
                "accuracy": 0.65,
                "sharpe": 0.8,
                "trust_level": 0.7,
            },
            "xgboost": {
                "mse": 0.12,
                "mae": 0.10,
                "rmse": 0.35,
                "accuracy": 0.68,
                "sharpe": 0.85,
                "trust_level": 0.75,
            },
            "prophet": {
                "mse": 0.18,
                "mae": 0.14,
                "rmse": 0.42,
                "accuracy": 0.62,
                "sharpe": 0.75,
                "trust_level": 0.65,
            },
            "arima": {
                "mse": 0.20,
                "mae": 0.16,
                "rmse": 0.45,
                "accuracy": 0.60,
                "sharpe": 0.70,
                "trust_level": 0.60,
            },
            "ensemble": {
                "mse": 0.10,
                "mae": 0.08,
                "rmse": 0.32,
                "accuracy": 0.72,
                "sharpe": 0.90,
                "trust_level": 0.85,
            },
        }

    def get_model_trust_levels(self) -> Dict[str, float]:
        """
        Get trust levels for all models (fallback implementation).

        Returns:
            Dict[str, float]: Model trust levels
        """
        try:
            logger.debug("Getting model trust levels from fallback monitor")
            return {
                model: data["trust_level"]
                for model, data in self._mock_performance.items()
            }
        except Exception as e:
            logger.error(f"Error getting model trust levels: {e}")
            return {"fallback_model": 0.5}

    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific model (fallback implementation).

        Args:
            model_name: Name of the model

        Returns:
            Dict[str, Any]: Model performance metrics
        """
        try:
            logger.debug(f"Getting performance for model: {model_name}")

            if model_name.lower() in self._mock_performance:
                performance = self._mock_performance[model_name.lower()].copy()
                performance["model_name"] = model_name
                performance["timestamp"] = datetime.now().isoformat()
                performance["fallback_mode"] = True
                return performance
            else:
                # Return default performance for unknown models
                return {
                    "model_name": model_name,
                    "mse": 0.2,
                    "mae": 0.15,
                    "rmse": 0.45,
                    "accuracy": 0.55,
                    "sharpe": 0.6,
                    "trust_level": 0.5,
                    "timestamp": datetime.now().isoformat(),
                    "fallback_mode": True,
                    "message": "Unknown model - using default metrics",
                }

        except Exception as e:
            logger.error(f"Error getting model performance for {model_name}: {e}")
            return {
                "model_name": model_name,
                "mse": 0.2,
                "mae": 0.15,
                "rmse": 0.45,
                "accuracy": 0.55,
                "sharpe": 0.6,
                "trust_level": 0.5,
                "timestamp": datetime.now().isoformat(),
                "fallback_mode": True,
                "error": str(e),
            }

    def update_model_performance(
        self, model_name: str, metrics: Dict[str, float]
    ) -> bool:
        """
        Update model performance metrics (fallback implementation).

        Args:
            model_name: Name of the model
            metrics: Performance metrics to update

        Returns:
            bool: True if update was successful
        """
        try:
            logger.info(f"Updating performance for model: {model_name}")

            if model_name.lower() in self._mock_performance:
                self._mock_performance[model_name.lower()].update(metrics)
                logger.info(f"Updated performance for {model_name}")
                return True
            else:
                logger.warning(
                    f"Unknown model {model_name} - cannot update performance"
                )
                return False

        except Exception as e:
            logger.error(f"Error updating model performance for {model_name}: {e}")
            return False

    def get_best_model(self, metric: str = "sharpe") -> Optional[str]:
        """
        Get the best performing model based on a metric (fallback implementation).

        Args:
            metric: Metric to use for comparison (default: 'sharpe')

        Returns:
            Optional[str]: Name of the best model or None if no models available
        """
        try:
            logger.debug(f"Finding best model based on metric: {metric}")

            if not self._mock_performance:
                return None

            best_model = None
            best_value = float("-inf")

            for model_name, performance in self._mock_performance.items():
                if metric in performance:
                    value = performance[metric]
                    if value > best_value:
                        best_value = value
                        best_model = model_name

            logger.info(f"Best model by {metric}: {best_model} (value: {best_value})")
            return best_model

        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return None

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback model monitor.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            return {
                "status": self._status,
                "monitored_models": len(self._mock_performance),
                "models": list(self._mock_performance.keys()),
                "fallback_mode": True,
                "message": "Using fallback model monitor",
            }
        except Exception as e:
            logger.error(f"Error getting fallback model monitor health: {e}")
            return {
                "status": "error",
                "monitored_models": 0,
                "fallback_mode": True,
                "error": str(e),
            }
