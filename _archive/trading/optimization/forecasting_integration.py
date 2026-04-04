"""
Forecasting Integration Module

This module integrates the Sharpe Optuna tuner with the forecasting/model selection pipeline.
It provides a seamless interface for optimizing models during the forecasting process
and automatically selecting the best model based on Sharpe ratio performance.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.optimization.optuna_tuner import SharpeOptunaTuner, get_sharpe_optuna_tuner

logger = logging.getLogger(__name__)


class ForecastingOptimizer:
    """
    Integrated optimizer for forecasting models.

    This class combines the Sharpe Optuna tuner with the forecasting pipeline
    to automatically optimize and select the best model for a given dataset.
    """

    def __init__(
        self,
        tuner: Optional[SharpeOptunaTuner] = None,
        optimization_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the forecasting optimizer.

        Args:
            tuner: Pre-configured Sharpe Optuna tuner (optional)
            optimization_config: Configuration for optimization
        """
        self.tuner = tuner or get_sharpe_optuna_tuner()
        self.optimization_config = optimization_config or {}

        # Default configuration
        self.default_config = {
            "n_trials": 50,
            "timeout": 1800,  # 30 minutes
            "validation_split": 0.2,
            "model_types": ["lstm", "xgboost", "transformer"],
            "min_sharpe_threshold": 0.1,
            "auto_optimize": True,
            "save_results": True,
        }
        self.default_config.update(self.optimization_config)

        # Results storage
        self.optimization_results = {}
        self.model_recommendations = {}

        logger.info("ForecastingOptimizer initialized")

    def optimize_for_forecasting(
        self,
        data: pd.DataFrame,
        target_column: str,
        forecast_horizon: int = 30,
        model_types: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Optimize models for forecasting with automatic model selection.

        Args:
            data: Training data
            target_column: Target column for forecasting
            forecast_horizon: Forecast horizon in periods
            model_types: List of model types to optimize
            **kwargs: Additional optimization parameters

        Returns:
            Dictionary with optimization results and recommendations
        """
        logger.info(
            f"Starting forecasting optimization for {forecast_horizon}-period horizon"
        )

        # Update configuration
        config = self.default_config.copy()
        config.update(kwargs)
        model_types = model_types or config["model_types"]

        try:
            # Prepare data for optimization
            prepared_data = self._prepare_data_for_optimization(
                data, target_column, forecast_horizon
            )

            # Run optimization for all model types
            optimization_results = self.tuner.optimize_all_models(
                data=prepared_data, target_column=target_column, model_types=model_types
            )

            # Get model recommendation
            recommendation = self.tuner.get_model_recommendation(
                prepared_data, target_column
            )

            # Validate recommendation
            if recommendation["expected_sharpe"] < config["min_sharpe_threshold"]:
                logger.warning(
                    f"Best model Sharpe ratio ({recommendation['expected_sharpe']:.4f}) "
                    f"below threshold ({config['min_sharpe_threshold']})"
                )

            # Store results
            self.optimization_results[datetime.now().isoformat()] = {
                "optimization_results": optimization_results,
                "recommendation": recommendation,
                "config": config,
                "data_shape": data.shape,
                "forecast_horizon": forecast_horizon,
            }

            # Save results if configured
            if config["save_results"]:
                results_file = self.tuner.save_results()
                logger.info(f"Optimization results saved to: {results_file}")

            return {
                "success": True,
                "recommendation": recommendation,
                "all_results": optimization_results,
                "config": config,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Forecasting optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendation": None,
                "timestamp": datetime.now().isoformat(),
            }

    def _prepare_data_for_optimization(
        self, data: pd.DataFrame, target_column: str, forecast_horizon: int
    ) -> pd.DataFrame:
        """
        Prepare data for optimization by adding forecasting-specific features.

        Args:
            data: Raw data
            target_column: Target column name
            forecast_horizon: Forecast horizon

        Returns:
            Prepared data with additional features
        """
        prepared_data = data.copy()

        # Add lag features for time series
        for lag in [1, 2, 3, 5, 10]:
            prepared_data[f"{target_column}_lag_{lag}"] = prepared_data[
                target_column
            ].shift(lag)

        # Add rolling statistics
        for window in [5, 10, 20]:
            prepared_data[f"{target_column}_sma_{window}"] = (
                prepared_data[target_column].rolling(window).mean()
            )
            prepared_data[f"{target_column}_std_{window}"] = (
                prepared_data[target_column].rolling(window).std()
            )

        # Add returns
        prepared_data[f"{target_column}_returns"] = prepared_data[
            target_column
        ].pct_change()

        # Add volatility
        prepared_data[f"{target_column}_volatility"] = (
            prepared_data[f"{target_column}_returns"].rolling(20).std()
        )

        # Remove NaN values
        prepared_data = prepared_data.dropna()

        logger.info(f"Prepared data shape: {prepared_data.shape}")
        return prepared_data

    def get_optimized_model(
        self, model_type: str, data: pd.DataFrame, target_column: str, **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Get an optimized model instance with best parameters.

        Args:
            model_type: Type of model to create
            data: Training data
            target_column: Target column name
            **kwargs: Additional parameters

        Returns:
            Tuple of (model_instance, parameters)
        """
        try:
            # Get best parameters for the model type
            best_params = self.tuner.get_best_params(model_type)

            if best_params is None:
                logger.warning(
                    f"No optimized parameters found for {model_type}, using defaults"
                )
                best_params = self._get_default_params(model_type)

            # Create model instance
            model = self._create_model_instance(model_type, best_params, **kwargs)

            return model, best_params

        except Exception as e:
            logger.error(f"Failed to create optimized model {model_type}: {e}")
            # Return default model
            default_params = self._get_default_params(model_type)
            model = self._create_model_instance(model_type, default_params, **kwargs)
            return model, default_params

    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type."""
        default_params = {
            "lstm": {
                "num_layers": 2,
                "hidden_size": 64,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "lookback": 30,
                "batch_size": 32,
                "epochs": 100,
            },
            "xgboost": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            },
            "transformer": {
                "d_model": 128,
                "num_heads": 8,
                "ff_dim": 512,
                "dropout": 0.1,
                "num_layers": 4,
                "learning_rate": 0.0001,
                "batch_size": 64,
                "epochs": 100,
            },
        }

        return default_params.get(model_type, {})

    def _create_model_instance(
        self, model_type: str, params: Dict[str, Any], **kwargs
    ) -> Any:
        """Create a model instance with given parameters."""
        try:
            if model_type == "lstm":
                from trading.models.lstm_model import LSTMModel

                return LSTMModel(
                    input_dim=kwargs.get("input_dim", 1),
                    hidden_dim=params.get("hidden_size", 64),
                    output_dim=1,
                    num_layers=params.get("num_layers", 2),
                    dropout=params.get("dropout", 0.2),
                )

            elif model_type == "xgboost":
                from trading.models.xgboost_model import XGBoostModel

                return XGBoostModel(params)

            elif model_type == "transformer":
                from trading.models.advanced.transformer.time_series_transformer import (
                    TransformerForecaster,
                )

                return TransformerForecaster(params)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except ImportError as e:
            logger.error(f"Failed to import model {model_type}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            raise

    def evaluate_model_performance(
        self, model: Any, data: pd.DataFrame, target_column: str, model_type: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance using trading metrics.

        Args:
            model: Trained model instance
            data: Test data
            target_column: Target column name
            model_type: Type of model

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Prepare test data
            X_test = data.drop(columns=[target_column])
            y_test = data[target_column]

            # Make predictions
            if model_type == "lstm":
                # Prepare sequences for LSTM
                sequence_length = 30  # Default
                X_test_seq = self._prepare_sequences(X_test.values, sequence_length)
                _unused_var = y_test_seq  # Placeholder, flake8 ignore: F841
                y_pred = model.predict(X_test_seq)
            else:
                y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self.tuner._calculate_trading_metrics(y_test.values, y_pred)

            logger.info(
                f"Model performance - Sharpe: {metrics['sharpe_ratio']:.4f}, "
                f"Win Rate: {metrics['win_rate']:.4f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Failed to evaluate model performance: {e}")
            return {
                "sharpe_ratio": -1.0,
                "win_rate": 0.0,
                "max_drawdown": -1.0,
                "total_return": -1.0,
                "directional_accuracy": 0.0,
                "mse": float("inf"),
                "rmse": float("inf"),
                "mae": float("inf"),
            }

    def _prepare_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Prepare sequences for time series models."""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i : i + sequence_length])
        return np.array(sequences)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of all optimization results."""
        if not self.optimization_results:
            return {"message": "No optimization results available"}

        summary = {
            "total_optimizations": len(self.optimization_results),
            "best_models": {},
            "average_performance": {},
            "recent_recommendations": [],
        }

        # Collect best models and performance
        for timestamp, result in self.optimization_results.items():
            recommendation = result.get("recommendation", {})
            if recommendation:
                model_type = recommendation.get("recommended_model")
                sharpe = recommendation.get("expected_sharpe", 0)

                if model_type not in summary["best_models"]:
                    summary["best_models"][model_type] = {"count": 0, "best_sharpe": -1}

                summary["best_models"][model_type]["count"] += 1
                summary["best_models"][model_type]["best_sharpe"] = max(
                    summary["best_models"][model_type]["best_sharpe"], sharpe
                )

        # Get recent recommendations
        recent_results = sorted(
            self.optimization_results.items(), key=lambda x: x[0], reverse=True
        )[:5]

        summary["recent_recommendations"] = [
            {
                "timestamp": timestamp,
                "recommended_model": result.get("recommendation", {}).get(
                    "recommended_model"
                ),
                "expected_sharpe": result.get("recommendation", {}).get(
                    "expected_sharpe", 0
                ),
            }
            for timestamp, result in recent_results
        ]

        return summary


def get_forecasting_optimizer(
    optimization_config: Optional[Dict[str, Any]] = None,
) -> ForecastingOptimizer:
    """
    Get a configured forecasting optimizer instance.

    Args:
        optimization_config: Configuration for optimization

    Returns:
        Configured ForecastingOptimizer instance
    """
    return ForecastingOptimizer(optimization_config=optimization_config)


# Integration with existing forecasting pipeline
def integrate_with_forecasting_pipeline(
    data: pd.DataFrame,
    target_column: str,
    forecast_horizon: int = 30,
    auto_optimize: bool = True,
) -> Dict[str, Any]:
    """
    Integrate optimization with the existing forecasting pipeline.

    Args:
        data: Training data
        target_column: Target column name
        forecast_horizon: Forecast horizon
        auto_optimize: Whether to automatically optimize models

    Returns:
        Dictionary with optimized model and results
    """
    logger.info("Integrating optimization with forecasting pipeline")

    if auto_optimize:
        # Create optimizer and run optimization
        optimizer = get_forecasting_optimizer()
        optimization_result = optimizer.optimize_for_forecasting(
            data=data, target_column=target_column, forecast_horizon=forecast_horizon
        )

        if optimization_result["success"]:
            recommendation = optimization_result["recommendation"]
            model_type = recommendation["recommended_model"]

            # Get optimized model
            model, params = optimizer.get_optimized_model(
                model_type=model_type, data=data, target_column=target_column
            )

            return {
                "success": True,
                "model": model,
                "model_type": model_type,
                "parameters": params,
                "optimization_result": optimization_result,
                "recommendation": recommendation,
            }
        else:
            logger.error("Optimization failed, using default model")
            return {
                "success": False,
                "error": optimization_result.get("error"),
                "model": None,
            }
    else:
        # Use default model without optimization
        logger.info("Skipping optimization, using default model")
        return {
            "success": True,
            "model": None,
            "model_type": "default",
            "parameters": {},
            "optimization_result": None,
            "recommendation": None,
        }
