"""
Model Evaluator

Evaluates model performance with comprehensive metrics and schema validation
for strategy evaluation results.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print("⚠️ matplotlib not available. Disabling plotting capabilities.")
    print(f"   Missing: {e}")
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Try to import seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError as e:
    print("⚠️ seaborn not available. Disabling advanced plotting capabilities.")
    print(f"   Missing: {e}")
    sns = None
    SEABORN_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("⚠️ scikit-learn not available. Disabling evaluation metrics.")
    print(f"   Missing: {e}")
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status enum."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Validation result with details."""
    status: ValidationStatus
    message: str
    errors: List[str]
    warnings: List[str]
    timestamp: datetime


@dataclass
class StrategyEvaluationSchema:
    """Schema for strategy evaluation results."""
    
    # Required fields
    strategy_name: str
    evaluation_timestamp: datetime
    metrics: Dict[str, float]
    
    # Optional fields
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    predictions: Optional[np.ndarray] = None
    actual_values: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    # Validation rules
    required_metrics: List[str] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.required_metrics is None:
            self.required_metrics = ["mse", "mae", "rmse", "r2", "directional_accuracy"]


class ModelEvaluator:
    """Model evaluator with schema validation."""
    
    def __init__(self, validation_config: Optional[Dict[str, Any]] = None):
        """Initialize the model evaluator.
        
        Args:
            validation_config: Configuration for validation rules
        """
        self.metrics = {}
        self.predictions = {}
        self.actuals = {}
        self.validation_config = validation_config or {
            "strict_mode": True,
            "allow_missing_metrics": False,
            "metric_thresholds": {
                "mse": {"min": 0.0, "max": float('inf')},
                "mae": {"min": 0.0, "max": float('inf')},
                "rmse": {"min": 0.0, "max": float('inf')},
                "r2": {"min": -float('inf'), "max": 1.0},
                "directional_accuracy": {"min": 0.0, "max": 1.0},
            },
            "required_fields": ["strategy_name", "evaluation_timestamp", "metrics"],
        }
        
        # Schema registry for different evaluation types
        self.schemas = self._initialize_schemas()
        
    def _initialize_schemas(self) -> Dict[str, StrategyEvaluationSchema]:
        """Initialize validation schemas for different evaluation types."""
        return {
            "default": StrategyEvaluationSchema(
                strategy_name="",
                evaluation_timestamp=datetime.now(),
                metrics={},
                required_metrics=["mse", "mae", "rmse", "r2"]
            ),
            "trading_strategy": StrategyEvaluationSchema(
                strategy_name="",
                evaluation_timestamp=datetime.now(),
                metrics={},
                required_metrics=["mse", "mae", "rmse", "r2", "directional_accuracy", "sharpe_ratio", "max_drawdown"]
            ),
            "classification": StrategyEvaluationSchema(
                strategy_name="",
                evaluation_timestamp=datetime.now(),
                metrics={},
                required_metrics=["accuracy", "precision", "recall", "f1_score"]
            ),
        }

    def evaluate_model(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Cannot evaluate model metrics.")
            return {}
        """Evaluate model performance with validation."""
        try:
            metrics = {
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
            }

            # Calculate directional accuracy
            if len(y_true) > 1:
                direction_true = np.sign(np.diff(y_true))
                direction_pred = np.sign(np.diff(y_pred))
                metrics["directional_accuracy"] = np.mean(direction_true == direction_pred)
            else:
                metrics["directional_accuracy"] = 0.0

            # Store metrics and predictions
            self.metrics[model_name] = metrics
            self.predictions[model_name] = y_pred
            self.actuals[model_name] = y_true

            # Validate metrics
            validation_result = self._validate_metrics(metrics, model_name)
            if validation_result.status == ValidationStatus.INVALID:
                logger.warning(f"Model evaluation validation failed: {validation_result.message}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {}

    def _validate_metrics(self, metrics: Dict[str, float], model_name: str) -> ValidationResult:
        """Validate evaluation metrics against thresholds.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        thresholds = self.validation_config["metric_thresholds"]
        
        for metric_name, value in metrics.items():
            if metric_name in thresholds:
                threshold = thresholds[metric_name]
                
                # Check min/max bounds
                if value < threshold["min"]:
                    errors.append(f"{metric_name} ({value}) below minimum ({threshold['min']})")
                elif value > threshold["max"]:
                    errors.append(f"{metric_name} ({value}) above maximum ({threshold['max']})")
                    
                # Check for NaN or infinite values
                if np.isnan(value) or np.isinf(value):
                    errors.append(f"{metric_name} has invalid value: {value}")
                    
        # Check for missing required metrics
        if self.validation_config.get("allow_missing_metrics", False):
            for required_metric in self.schemas["default"].required_metrics:
                if required_metric not in metrics:
                    warnings.append(f"Missing required metric: {required_metric}")
        else:
            for required_metric in self.schemas["default"].required_metrics:
                if required_metric not in metrics:
                    errors.append(f"Missing required metric: {required_metric}")
                    
        # Determine status
        if errors:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID
            
        message = f"Validation {'failed' if errors else 'passed'} for {model_name}"
        if warnings:
            message += f" with {len(warnings)} warnings"
            
        return ValidationResult(
            status=status,
            message=message,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now()
        )

    def validate_evaluation_result(
        self, 
        result: Dict[str, Any], 
        schema_type: str = "default"
    ) -> ValidationResult:
        """Validate strategy evaluation result against schema.
        
        Args:
            result: Evaluation result dictionary
            schema_type: Type of schema to use for validation
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        if schema_type not in self.schemas:
            errors.append(f"Unknown schema type: {schema_type}")
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message=f"Unknown schema type: {schema_type}",
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
            
        schema = self.schemas[schema_type]
        required_fields = self.validation_config["required_fields"]
        
        # Check required fields
        for field in required_fields:
            if field not in result:
                errors.append(f"Missing required field: {field}")
                
        # Check strategy name
        if "strategy_name" in result:
            if not isinstance(result["strategy_name"], str) or not result["strategy_name"].strip():
                errors.append("strategy_name must be a non-empty string")
                
        # Check evaluation timestamp
        if "evaluation_timestamp" in result:
            timestamp = result["evaluation_timestamp"]
            if isinstance(timestamp, str):
                try:
                    datetime.fromisoformat(timestamp)
                except ValueError:
                    errors.append("evaluation_timestamp must be a valid ISO format datetime string")
            elif not isinstance(timestamp, datetime):
                errors.append("evaluation_timestamp must be a datetime object or ISO string")
                
        # Check metrics
        if "metrics" in result:
            metrics = result["metrics"]
            if not isinstance(metrics, dict):
                errors.append("metrics must be a dictionary")
            else:
                # Validate individual metrics
                for metric_name, metric_value in metrics.items():
                    if not isinstance(metric_name, str):
                        errors.append(f"Metric name must be string: {metric_name}")
                    if not isinstance(metric_value, (int, float, np.number)):
                        errors.append(f"Metric value must be numeric: {metric_name}={metric_value}")
                    if np.isnan(metric_value) or np.isinf(metric_value):
                        errors.append(f"Metric value is invalid: {metric_name}={metric_value}")
                        
                # Check required metrics
                for required_metric in schema.required_metrics:
                    if required_metric not in metrics:
                        if self.validation_config.get("allow_missing_metrics", False):
                            warnings.append(f"Missing recommended metric: {required_metric}")
                        else:
                            errors.append(f"Missing required metric: {required_metric}")
                            
        # Check parameters
        if "parameters" in result and result["parameters"] is not None:
            if not isinstance(result["parameters"], dict):
                errors.append("parameters must be a dictionary")
                
        # Check metadata
        if "metadata" in result and result["metadata"] is not None:
            if not isinstance(result["metadata"], dict):
                errors.append("metadata must be a dictionary")
                
        # Check predictions and actual values
        for field in ["predictions", "actual_values"]:
            if field in result and result[field] is not None:
                if not isinstance(result[field], (np.ndarray, list)):
                    errors.append(f"{field} must be a numpy array or list")
                elif len(result[field]) == 0:
                    warnings.append(f"{field} is empty")
                    
        # Check feature importance
        if "feature_importance" in result and result["feature_importance"] is not None:
            fi = result["feature_importance"]
            if not isinstance(fi, dict):
                errors.append("feature_importance must be a dictionary")
            else:
                for feature_name, importance in fi.items():
                    if not isinstance(feature_name, str):
                        errors.append(f"Feature name must be string: {feature_name}")
                    if not isinstance(importance, (int, float, np.number)):
                        errors.append(f"Feature importance must be numeric: {feature_name}={importance}")
                        
        # Determine status
        if errors:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID
            
        message = f"Schema validation {'failed' if errors else 'passed'} for {schema_type}"
        if warnings:
            message += f" with {len(warnings)} warnings"
            
        return ValidationResult(
            status=status,
            message=message,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now()
        )

    def create_evaluation_result(
        self,
        strategy_name: str,
        metrics: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        predictions: Optional[np.ndarray] = None,
        actual_values: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        schema_type: str = "default"
    ) -> Dict[str, Any]:
        """Create a validated evaluation result.
        
        Args:
            strategy_name: Name of the strategy
            metrics: Evaluation metrics
            parameters: Strategy parameters
            metadata: Additional metadata
            predictions: Model predictions
            actual_values: Actual values
            feature_importance: Feature importance scores
            schema_type: Schema type for validation
            
        Returns:
            Validated evaluation result dictionary
        """
        result = {
            "strategy_name": strategy_name,
            "evaluation_timestamp": datetime.now(),
            "metrics": metrics,
            "parameters": parameters,
            "metadata": metadata,
            "predictions": predictions.tolist() if predictions is not None else None,
            "actual_values": actual_values.tolist() if actual_values is not None else None,
            "feature_importance": feature_importance,
        }
        
        # Validate the result
        validation_result = self.validate_evaluation_result(result, schema_type)
        
        # Add validation info to result
        result["validation"] = {
            "status": validation_result.status.value,
            "message": validation_result.message,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "timestamp": validation_result.timestamp.isoformat(),
        }
        
        if validation_result.status == ValidationStatus.INVALID:
            logger.error(f"Evaluation result validation failed: {validation_result.message}")
            logger.error(f"Errors: {validation_result.errors}")
        elif validation_result.status == ValidationStatus.WARNING:
            logger.warning(f"Evaluation result validation warnings: {validation_result.warnings}")
        else:
            logger.info(f"Evaluation result validation passed for {strategy_name}")
            
        return result

    def export_evaluation_results(
        self,
        results: List[Dict[str, Any]],
        filepath: str,
        format: str = "json"
    ) -> bool:
        """Export evaluation results with validation.
        
        Args:
            results: List of evaluation results
            filepath: Output file path
            format: Export format ('json', 'csv')
            
        Returns:
            True if export successful
        """
        try:
            # Validate all results before export
            valid_results = []
            invalid_results = []
            
            for result in results:
                validation_result = self.validate_evaluation_result(result)
                if validation_result.status == ValidationStatus.VALID:
                    valid_results.append(result)
                else:
                    invalid_results.append(result)
                    logger.warning(f"Invalid result excluded from export: {validation_result.message}")
                    
            if not valid_results:
                logger.error("No valid results to export")
                return False
                
            # Export based on format
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(valid_results, f, indent=2)
            elif format == "csv":
                # Flatten results for CSV
                flat_data = []
                for result in valid_results:
                    base_record = {
                        "strategy_name": result.get("strategy_name", ""),
                        "evaluation_timestamp": result.get("evaluation_timestamp", ""),
                        "validation_status": result.get("validation", {}).get("status", ""),
                    }
                    
                    # Add metrics
                    for metric_name, metric_value in result.get("metrics", {}).items():
                        base_record[f"metric_{metric_name}"] = metric_value
                        
                    flat_data.append(base_record)
                    
                df = pd.DataFrame(flat_data)
                df.to_csv(filepath, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Exported {len(valid_results)} valid evaluation results to {filepath}")
            if invalid_results:
                logger.warning(f"Excluded {len(invalid_results)} invalid results from export")
                
            return True
            
        except Exception as e:
            logger.error(f"Error exporting evaluation results: {e}")
            return False

    def plot_predictions(
        self, model_name: str, save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available. Cannot plot predictions.")
            return {"success": False, "error": "matplotlib not available"}
        """Plot actual vs predicted values.

        Returns:
            Dictionary with plot status and details
        """
        try:
            if model_name not in self.predictions:
                return {
                    "success": False,
                    "message": f"No predictions found for model: {model_name}",
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                }

            plt.figure(figsize=(12, 6))
            plt.plot(self.actuals[model_name], label="Actual", color="blue")
            plt.plot(self.predictions[model_name], label="Predicted", color="red")
            plt.title(f"Actual vs Predicted Values - {model_name}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()

            if save_path:
                plt.savefig(save_path)
                plt.close()
                return {
                    "success": True,
                    "message": f"Predictions plot saved to {save_path}",
                    "model_name": model_name,
                    "save_path": save_path,
                    "data_points": len(self.actuals[model_name]),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                plt.show()
                plt.close()
                return {
                    "success": True,
                    "message": f"Predictions plot displayed for {model_name}",
                    "model_name": model_name,
                    "data_points": len(self.actuals[model_name]),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error plotting predictions: {str(e)}",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
            }

    def plot_residuals(
        self, model_name: str, save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Plot residuals.

        Returns:
            Dictionary with plot status and details
        """
        try:
            if model_name not in self.predictions:
                return {
                    "success": False,
                    "message": f"No predictions found for model: {model_name}",
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                }

            residuals = self.actuals[model_name] - self.predictions[model_name]

            plt.figure(figsize=(12, 6))
            plt.scatter(self.predictions[model_name], residuals)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.title(f"Residuals Plot - {model_name}")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")

            if save_path:
                plt.savefig(save_path)
                plt.close()
                return {
                    "success": True,
                    "message": f"Residuals plot saved to {save_path}",
                    "model_name": model_name,
                    "save_path": save_path,
                    "residuals_mean": np.mean(residuals),
                    "residuals_std": np.std(residuals),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                plt.show()
                plt.close()
                return {
                    "success": True,
                    "message": f"Residuals plot displayed for {model_name}",
                    "model_name": model_name,
                    "residuals_mean": np.mean(residuals),
                    "residuals_std": np.std(residuals),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error plotting residuals: {str(e)}",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
            }

    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        model_name: str,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plot feature importance.

        Returns:
            Dictionary with plot status and details
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x="importance", y="feature", data=feature_importance)
            plt.title(f"Feature Importance - {model_name}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")

            if save_path:
                plt.savefig(save_path)
                plt.close()
                return {
                    "success": True,
                    "message": f"Feature importance plot saved to {save_path}",
                    "model_name": model_name,
                    "save_path": save_path,
                    "num_features": len(feature_importance),
                    "max_importance": feature_importance["importance"].max(),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                plt.show()
                plt.close()
                return {
                    "success": True,
                    "message": f"Feature importance plot displayed for {model_name}",
                    "model_name": model_name,
                    "num_features": len(feature_importance),
                    "max_importance": feature_importance["importance"].max(),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error plotting feature importance: {str(e)}",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
            }

    def generate_report(self, model_name: str) -> Dict:
        """Generate comprehensive evaluation report."""
        if model_name not in self.metrics:
            return {
                "success": False,
                "message": f"No metrics found for model: {model_name}",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
            }

        metrics = self.metrics[model_name]
        summary = self._generate_summary(model_name)

        report = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "summary": summary,
            "validation": self._validate_metrics(metrics, model_name).__dict__,
        }

        return report

    def _generate_summary(self, model_name: str) -> str:
        """Generate summary of model performance."""
        metrics = self.metrics[model_name]
        
        summary_parts = [
            f"Model: {model_name}",
            f"MSE: {metrics.get('mse', 'N/A'):.6f}",
            f"RMSE: {metrics.get('rmse', 'N/A'):.6f}",
            f"MAE: {metrics.get('mae', 'N/A'):.6f}",
            f"R²: {metrics.get('r2', 'N/A'):.4f}",
        ]
        
        if "directional_accuracy" in metrics:
            summary_parts.append(f"Directional Accuracy: {metrics['directional_accuracy']:.4f}")
            
        return " | ".join(summary_parts)

    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare multiple models."""
        comparison_data = []
        
        for model_name in model_names:
            if model_name in self.metrics:
                metrics = self.metrics[model_name].copy()
                metrics["model_name"] = model_name
                comparison_data.append(metrics)
                
        if comparison_data:
            return pd.DataFrame(comparison_data)
        else:
            return pd.DataFrame()

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get all evaluation metrics."""
        return {
            "models": list(self.metrics.keys()),
            "metrics": self.metrics,
            "validation_config": self.validation_config,
            "timestamp": datetime.now().isoformat(),
        }
