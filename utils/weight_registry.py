"""
Weight Registry for Hybrid Models

This module provides centralized management of hybrid model weights:
- Weight updates and saving
- Performance history tracking
- Weight optimization and validation
- Cross-model weight consistency
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WeightRegistry:
    """
    Centralized registry for managing hybrid model weights and performance history.

    Features:
    - Weight storage and retrieval
    - Performance history tracking
    - Weight optimization and validation
    - Cross-model consistency checks
    - Automatic weight updates based on performance
    """

    def __init__(
        self,
        registry_file: str = "data/weight_registry.json",
        backup_dir: str = "backups/weights",
        max_history: int = 1000,
        auto_backup: bool = True,
    ):
        """
        Initialize weight registry.

        Args:
            registry_file: Path to registry JSON file
            backup_dir: Directory for weight backups
            max_history: Maximum number of history entries to keep
            auto_backup: Whether to automatically backup weights
        """
        self.registry_file = Path(registry_file)
        self.backup_dir = Path(backup_dir)
        self.max_history = max_history
        self.auto_backup = auto_backup

        # Create directories
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize registry data
        self.registry = self._load_registry()

        # Performance tracking
        self.performance_history = []

        logger.info(f"WeightRegistry initialized: {self.registry_file}")

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry = json.load(f)
                    logger.info(f"Loaded weight registry with {len(registry.get('models', {}))} models")
                    return registry
            except Exception as e:
                logger.error(f"Failed to load weight registry: {e}")

        # Return default registry structure
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "models": {},
            "performance_history": [],
            "optimization_history": [],
            "metadata": {},
        }

    def _save_registry(self):
        """Save registry to file."""
        try:
            self.registry["last_updated"] = datetime.now().isoformat()

            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)

            # Create backup if enabled
            if self.auto_backup:
                self._create_backup()

        except Exception as e:
            logger.error(f"Failed to save weight registry: {e}")

    def _create_backup(self):
        """Create backup of registry file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"weight_registry_{timestamp}.json"

            with open(backup_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)

            # Keep only recent backups (last 10)
            backup_files = sorted(self.backup_dir.glob("weight_registry_*.json"))
            if len(backup_files) > 10:
                for old_backup in backup_files[:-10]:
                    old_backup.unlink()

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def register_model(
        self,
        model_name: str,
        model_type: str,
        initial_weights: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a new model in the registry.

        Args:
            model_name: Name of the model
            model_type: Type of model (lstm, xgboost, prophet, etc.)
            initial_weights: Initial weights for the model
            metadata: Additional metadata

        Returns:
            True if successfully registered
        """
        try:
            if model_name in self.registry["models"]:
                logger.warning(f"Model {model_name} already registered")
                return False

            # Set default weights if not provided
            if initial_weights is None:
                initial_weights = {"base_weight": 1.0}

            # Validate weights
            if not self._validate_weights(initial_weights):
                logger.error(f"Invalid weights for model {model_name}")
                return False

            # Create model entry
            model_entry = {
                "name": model_name,
                "type": model_type,
                "weights": initial_weights,
                "performance": {
                    "accuracy": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "total_return": 0.0,
                    "volatility": 0.0,
                    "last_updated": datetime.now().isoformat(),
                },
                "history": [],
                "metadata": metadata or {},
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            }

            self.registry["models"][model_name] = model_entry
            self._save_registry()

            logger.info(f"Registered model: {model_name} ({model_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            return False

    def update_weights(
        self,
        model_name: str,
        new_weights: Dict[str, float],
        reason: str = "manual_update",
    ) -> bool:
        """
        Update weights for a model.

        Args:
            model_name: Name of the model
            new_weights: New weights
            reason: Reason for weight update

        Returns:
            True if successfully updated
        """
        try:
            if model_name not in self.registry["models"]:
                logger.error(f"Model {model_name} not found in registry")
                return False

            # Validate new weights
            if not self._validate_weights(new_weights):
                logger.error(f"Invalid weights for model {model_name}")
                return False

            # Get current weights
            model_entry = self.registry["models"][model_name]
            old_weights = model_entry["weights"].copy()

            # Update weights
            model_entry["weights"] = new_weights
            model_entry["last_updated"] = datetime.now().isoformat()

            # Record weight change in history
            weight_change = {
                "timestamp": datetime.now().isoformat(),
                "old_weights": old_weights,
                "new_weights": new_weights,
                "reason": reason,
                "change_magnitude": self._calculate_weight_change_magnitude(old_weights, new_weights),
            }

            model_entry["history"].append(weight_change)

            # Limit history size
            if len(model_entry["history"]) > self.max_history:
                model_entry["history"] = model_entry["history"][-self.max_history:]

            self._save_registry()

            logger.info(f"Updated weights for {model_name}: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to update weights for {model_name}: {e}")
            return False

    def update_performance(
        self,
        model_name: str,
        performance_metrics: Dict[str, float],
        evaluation_date: Optional[datetime] = None,
    ) -> bool:
        """
        Update performance metrics for a model.

        Args:
            model_name: Name of the model
            performance_metrics: Performance metrics
            evaluation_date: Date of evaluation

        Returns:
            True if successfully updated
        """
        try:
            if model_name not in self.registry["models"]:
                logger.error(f"Model {model_name} not found in registry")
                return False

            if evaluation_date is None:
                evaluation_date = datetime.now()

            model_entry = self.registry["models"][model_name]

            # Update performance metrics
            model_entry["performance"].update(performance_metrics)
            model_entry["performance"]["last_updated"] = evaluation_date.isoformat()

            # Record performance in history
            performance_record = {
                "timestamp": evaluation_date.isoformat(),
                "metrics": performance_metrics.copy(),
            }

            model_entry["history"].append(performance_record)

            # Limit history size
            if len(model_entry["history"]) > self.max_history:
                model_entry["history"] = model_entry["history"][-self.max_history:]

            # Add to global performance history
            global_record = {
                "timestamp": evaluation_date.isoformat(),
                "model_name": model_name,
                "model_type": model_entry["type"],
                "metrics": performance_metrics.copy(),
            }

            self.registry["performance_history"].append(global_record)

            # Limit global history size
            if len(self.registry["performance_history"]) > self.max_history:
                self.registry["performance_history"] = self.registry["performance_history"][-self.max_history:]

            self._save_registry()

            logger.info(f"Updated performance for {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update performance for {model_name}: {e}")
            return False

    def get_weights(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get current weights for a model.

        Args:
            model_name: Name of the model

        Returns:
            Current weights or None if model not found
        """
        if model_name in self.registry["models"]:
            return self.registry["models"][model_name]["weights"].copy()
        return None

    def get_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a model.

        Args:
            model_name: Name of the model

        Returns:
            Performance metrics or None if model not found
        """
        if model_name in self.registry["models"]:
            return self.registry["models"][model_name]["performance"].copy()
        return None

    def get_model_history(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get history for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of historical records
        """
        if model_name in self.registry["models"]:
            return self.registry["models"][model_name]["history"].copy()
        return []

    def optimize_weights(
        self,
        model_names: List[str],
        optimization_method: str = "performance_weighted",
        target_metric: str = "sharpe_ratio",
    ) -> Dict[str, float]:
        """
        Optimize weights for multiple models.

        Args:
            model_names: List of model names to optimize
            optimization_method: Optimization method
            target_metric: Target metric for optimization

        Returns:
            Optimized weights dictionary
        """
        try:
            # Get performance data for models
            model_performances = {}
            for model_name in model_names:
                if model_name in self.registry["models"]:
                    performance = self.registry["models"][model_name]["performance"]
                    model_performances[model_name] = performance

            if not model_performances:
                logger.warning("No valid models found for weight optimization")
                return {}

            # Apply optimization method
            if optimization_method == "performance_weighted":
                optimized_weights = self._optimize_performance_weighted(
                    model_performances, target_metric
                )
            elif optimization_method == "equal_weight":
                optimized_weights = self._optimize_equal_weight(model_performances)
            elif optimization_method == "risk_parity":
                optimized_weights = self._optimize_risk_parity(model_performances)
            else:
                logger.warning(f"Unknown optimization method: {optimization_method}")
                optimized_weights = self._optimize_equal_weight(model_performances)

            # Record optimization
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "method": optimization_method,
                "target_metric": target_metric,
                "models": model_names,
                "optimized_weights": optimized_weights,
            }

            self.registry["optimization_history"].append(optimization_record)

            # Limit optimization history
            if len(self.registry["optimization_history"]) > self.max_history:
                self.registry["optimization_history"] = self.registry["optimization_history"][-self.max_history:]

            self._save_registry()

            logger.info(f"Optimized weights using {optimization_method}")
            return optimized_weights

        except Exception as e:
            logger.error(f"Failed to optimize weights: {e}")
            return {}

    def _optimize_performance_weighted(
        self, model_performances: Dict[str, Dict[str, float]], target_metric: str
    ) -> Dict[str, float]:
        """Optimize weights based on performance."""
        weights = {}
        total_score = 0.0

        for model_name, performance in model_performances.items():
            score = performance.get(target_metric, 0.0)
            # Ensure non-negative scores
            score = max(0.0, score)
            weights[model_name] = score
            total_score += score

        # Normalize weights
        if total_score > 0:
            for model_name in weights:
                weights[model_name] /= total_score
        else:
            # Equal weights if no positive scores
            equal_weight = 1.0 / len(weights)
            for model_name in weights:
                weights[model_name] = equal_weight

        return weights

    def _optimize_equal_weight(self, model_performances: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Optimize weights using equal weighting."""
        equal_weight = 1.0 / len(model_performances)
        return {model_name: equal_weight for model_name in model_performances}

    def _optimize_risk_parity(self, model_performances: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Optimize weights using risk parity approach."""
        weights = {}
        total_risk = 0.0

        for model_name, performance in model_performances.items():
            # Use volatility as risk measure
            risk = performance.get("volatility", 1.0)
            if risk <= 0:
                risk = 1.0  # Default risk
            weights[model_name] = 1.0 / risk
            total_risk += 1.0 / risk

        # Normalize weights
        if total_risk > 0:
            for model_name in weights:
                weights[model_name] /= total_risk

        return weights

    def _validate_weights(self, weights: Dict[str, float]) -> bool:
        """Validate weight dictionary."""
        if not isinstance(weights, dict):
            return False

        if not weights:
            return False

        # Check for non-negative weights
        for weight in weights.values():
            if not isinstance(weight, (int, float)) or weight < 0:
                return False

        return True

    def _calculate_weight_change_magnitude(
        self, old_weights: Dict[str, float], new_weights: Dict[str, float]
    ) -> float:
        """Calculate magnitude of weight change."""
        all_keys = set(old_weights.keys()) | set(new_weights.keys())
        total_change = 0.0

        for key in all_keys:
            old_val = old_weights.get(key, 0.0)
            new_val = new_weights.get(key, 0.0)
            total_change += abs(new_val - old_val)

        return total_change

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of registry contents."""
        models = self.registry["models"]

        summary = {
            "total_models": len(models),
            "model_types": {},
            "total_performance_records": len(self.registry["performance_history"]),
            "total_optimizations": len(self.registry["optimization_history"]),
            "last_updated": self.registry["last_updated"],
        }

        # Count model types
        for model_entry in models.values():
            model_type = model_entry["type"]
            summary["model_types"][model_type] = summary["model_types"].get(model_type, 0) + 1

        return summary

    def export_weights(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export weights for specified models.

        Args:
            model_names: List of model names to export, or None for all

        Returns:
            Dictionary with exported weights and metadata
        """
        if model_names is None:
            model_names = list(self.registry["models"].keys())

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "models": {},
        }

        for model_name in model_names:
            if model_name in self.registry["models"]:
                model_entry = self.registry["models"][model_name]
                export_data["models"][model_name] = {
                    "weights": model_entry["weights"],
                    "performance": model_entry["performance"],
                    "type": model_entry["type"],
                    "metadata": model_entry["metadata"],
                }

        return export_data

    def import_weights(self, import_data: Dict[str, Any], overwrite: bool = False) -> bool:
        """
        Import weights from export data.

        Args:
            import_data: Export data to import
            overwrite: Whether to overwrite existing models

        Returns:
            True if successfully imported
        """
        try:
            for model_name, model_data in import_data.get("models", {}).items():
                if model_name in self.registry["models"] and not overwrite:
                    logger.warning(f"Model {model_name} already exists, skipping")
                    continue

                # Register or update model
                if model_name not in self.registry["models"]:
                    self.register_model(
                        model_name=model_name,
                        model_type=model_data.get("type", "unknown"),
                        initial_weights=model_data.get("weights", {}),
                        metadata=model_data.get("metadata", {}),
                    )
                else:
                    self.update_weights(
                        model_name=model_name,
                        new_weights=model_data.get("weights", {}),
                        reason="import",
                    )

                # Update performance if available
                if "performance" in model_data:
                    self.update_performance(model_name, model_data["performance"])

            logger.info("Successfully imported models")
            return True

        except Exception as e:
            logger.error(f"Failed to import weights: {e}")
            return False

    def cleanup_old_records(self, days_to_keep: int = 30):
        """
        Clean up old performance and optimization records.

        Args:
            days_to_keep: Number of days of records to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.isoformat()

        # Clean up performance history
        self.registry["performance_history"] = [
            record for record in self.registry["performance_history"]
            if record["timestamp"] >= cutoff_str
        ]

        # Clean up optimization history
        self.registry["optimization_history"] = [
            record for record in self.registry["optimization_history"]
            if record["timestamp"] >= cutoff_str
        ]

        # Clean up individual model histories
        for model_entry in self.registry["models"].values():
            model_entry["history"] = [
                record for record in model_entry["history"]
                if record["timestamp"] >= cutoff_str
            ]

        self._save_registry()

        logger.info(f"Cleaned up records older than {days_to_keep} days")


# Global registry instance
_weight_registry = None


def get_weight_registry() -> WeightRegistry:
    """Get the global weight registry instance."""
    global _weight_registry
    if _weight_registry is None:
        _weight_registry = WeightRegistry()
    return _weight_registry


# Convenience functions
def update_model_weights(model_name: str, new_weights: Dict[str, float], reason: str = "manual_update") -> bool:
    """Update weights for a model."""
    registry = get_weight_registry()
    return registry.update_weights(model_name, new_weights, reason)


def update_model_performance(model_name: str, performance_metrics: Dict[str, float]) -> bool:
    """Update performance for a model."""
    registry = get_weight_registry()
    return registry.update_performance(model_name, performance_metrics)


def get_model_weights(model_name: str) -> Optional[Dict[str, float]]:
    """Get weights for a model."""
    registry = get_weight_registry()
    return registry.get_weights(model_name)


def optimize_ensemble_weights(model_names: List[str], method: str = "performance_weighted") -> Dict[str, float]:
    """Optimize weights for an ensemble of models."""
    registry = get_weight_registry()
    return registry.optimize_weights(model_names, method)


def get_registry_summary() -> Dict[str, Any]:
    """Get registry summary."""
    registry = get_weight_registry()
    return registry.get_registry_summary()
