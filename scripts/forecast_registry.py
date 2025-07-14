"""
Forecast Registry Module

This module handles forecast-related functionality and registry management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ForecastRegistry:
    """Registry for forecast models and configurations."""

    def __init__(self):
        """Initialize the forecast registry."""
        self.forecast_models = {}
        self.forecast_configs = {}
        self.performance_history = {}

    def register_model(self, model_name: str, model_config: Dict[str, Any]) -> bool:
        """Register a forecast model.

        Args:
            model_name: Name of the model
            model_config: Model configuration

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.forecast_models[model_name] = {
                "config": model_config,
                "registered_at": datetime.now().isoformat(),
                "status": "active",
            }
            logger.info(f"Registered forecast model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register forecast model {model_name}: {e}")
            return False

    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get a forecast model configuration.

        Args:
            model_name: Name of the model

        Returns:
            Dict: Model configuration or None if not found
        """
        return self.forecast_models.get(model_name)

    def list_models(self) -> List[str]:
        """List all registered forecast models.

        Returns:
            List: Names of registered models
        """
        return list(self.forecast_models.keys())

    def update_performance(
        self, model_name: str, performance_metrics: Dict[str, Any]
    ) -> bool:
        """Update performance metrics for a model.

        Args:
            model_name: Name of the model
            performance_metrics: Performance metrics

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []

            self.performance_history[model_name].append(
                {
                    "metrics": performance_metrics,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(f"Updated performance for model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update performance for model {model_name}: {e}")
            return False

    def get_best_model(self, metric: str = "sharpe_ratio") -> Optional[str]:
        """Get the best performing model based on a metric.

        Args:
            metric: Performance metric to use

        Returns:
            str: Name of best model or None
        """
        best_model = None
        best_score = float("-inf")

        for model_name, history in self.performance_history.items():
            if history:
                latest_metrics = history[-1]["metrics"]
                score = latest_metrics.get(metric, float("-inf"))

                if score > best_score:
                    best_score = score
                    best_model = model_name

        return best_model


# Global instance
forecast_registry = ForecastRegistry()
