"""
Backtesting Evaluator

This module provides advanced evaluation functions for trading models and strategies,
including walk-forward backtesting for model stability analysis.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from trading.backtesting.performance_analysis import PerformanceAnalyzer
from trading.backtesting.risk_metrics import RiskMetricsEngine
from trading.models.base_model import BaseModel
from trading.strategies.registry import BaseStrategy
from utils.math_utils import calculate_sharpe_ratio

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Advanced model evaluator with walk-forward backtesting capabilities.

    This class provides comprehensive evaluation methods for trading models,
    including stability analysis through walk-forward backtesting.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100000.0,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        **kwargs,
    ):
        """
        Initialize the ModelEvaluator.

        Args:
            data: Historical price data
            initial_cash: Starting cash amount
            risk_free_rate: Risk-free rate for calculations
            trading_days_per_year: Trading days per year
            **kwargs: Additional backtester parameters
        """
        self.data = data
        self.initial_cash = initial_cash
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

        # Initialize components
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_metrics_engine = RiskMetricsEngine(
            risk_free_rate=risk_free_rate, period=trading_days_per_year
        )

        # Results storage
        self.evaluation_results: Dict[str, Any] = {}

        logger.info("ModelEvaluator initialized")

    def walk_forward_backtest(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        window_size: int = 100,
        step_size: int = 20,
        test_size: int = 20,
        target_column: str = "close",
        feature_columns: Optional[List[str]] = None,
        strategy: Optional[BaseStrategy] = None,
        confidence_threshold: float = 0.6,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform walk-forward backtesting for model stability analysis.

        This function implements a rolling window approach where:
        - Train on window i, test on window i+1
        - Average Sharpe, return, drawdown over all windows
        - Provides stability report for the model

        Args:
            model: Model to evaluate
            data: Historical data for backtesting
            window_size: Size of training window
            step_size: Step size for moving the window
            test_size: Size of test window
            target_column: Target column for prediction
            feature_columns: Feature columns for model input
            strategy: Trading strategy to use (optional)
            confidence_threshold: Minimum confidence for signal execution
            **kwargs: Additional parameters

        Returns:
            Dictionary containing stability report and detailed results
        """
        logger.info(
            f"Starting walk-forward backtest with window_size={window_size}, step_size={step_size}"
        )

        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]

        # Initialize results storage
        window_results = []
        all_metrics = []
        model_performance = []

        # Calculate number of windows
        total_data_points = len(data)
        num_windows = (total_data_points - window_size - test_size) // step_size + 1

        logger.info(
            f"Total data points: {total_data_points}, Number of windows: {num_windows}"
        )

        for window_idx in range(num_windows):
            try:
                # Calculate window boundaries
                train_start = window_idx * step_size
                train_end = train_start + window_size
                test_start = train_end
                test_end = min(test_start + test_size, total_data_points)

                # Skip if not enough data
                if test_end - test_start < test_size // 2:
                    logger.warning(
                        f"Window {window_idx}: Insufficient test data, skipping"
                    )
                    continue

                # Split data
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[test_start:test_end]

                logger.info(
                    f"Window {window_idx + 1}/{num_windows}: Train={len(train_data)}, Test={len(test_data)}"
                )

                # Train model
                model_result = self._train_model_on_window(
                    model, train_data, target_column, feature_columns, **kwargs
                )

                if not model_result["success"]:
                    logger.warning(
                        f"Window {window_idx}: Model training failed, skipping"
                    )
                    continue

                # Generate predictions
                predictions = self._generate_predictions(
                    model, test_data, target_column, feature_columns
                )

                if predictions is None:
                    logger.warning(
                        f"Window {window_idx}: Prediction generation failed, skipping"
                    )
                    continue

                # Calculate trading metrics
                window_metrics = self._calculate_window_metrics(
                    test_data,
                    predictions,
                    target_column,
                    strategy,
                    confidence_threshold,
                )

                # Store results
                window_result = {
                    "window_idx": window_idx,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "metrics": window_metrics,
                    "model_performance": model_result["model_performance"],
                    "predictions": predictions,
                    "actual_values": test_data[target_column].values,
                }

                window_results.append(window_result)
                all_metrics.append(window_metrics)
                model_performance.append(model_result["model_performance"])

                logger.info(
                    f"Window {window_idx + 1}: Sharpe={window_metrics['sharpe_ratio']:.4f}, "
                    f"Return={window_metrics['total_return']:.4f}"
                )

            except Exception as e:
                logger.error(f"Error in window {window_idx}: {e}")
                continue

        # Generate stability report
        stability_report = self._generate_stability_report(
            window_results, all_metrics, model_performance
        )

        # Store comprehensive results
        results = {
            "stability_report": stability_report,
            "window_results": window_results,
            "all_metrics": all_metrics,
            "model_performance": model_performance,
            "config": {
                "window_size": window_size,
                "step_size": step_size,
                "test_size": test_size,
                "num_windows": num_windows,
                "total_data_points": total_data_points,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "confidence_threshold": confidence_threshold,
            },
        }

        self.evaluation_results[
            f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ] = results

        logger.info(
            f"Walk-forward backtest completed. Processed {len(window_results)} windows"
        )
        return results

    def _train_model_on_window(
        self,
        model: BaseModel,
        train_data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train model on a specific window of data.

        Args:
            model: Model to train
            train_data: Training data
            target_column: Target column
            feature_columns: Feature columns
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training results
        """
        try:
            # Prepare training data
            X_train = train_data[feature_columns].values
            y_train = train_data[target_column].values

            # Train model
            if hasattr(model, "fit"):
                model.fit(X_train, y_train, **kwargs)
                training_success = True
            else:
                logger.warning("Model does not have fit method")
                training_success = False

            # Calculate training performance
            if training_success and hasattr(model, "predict"):
                y_pred_train = model.predict(X_train)
                train_metrics = self._calculate_basic_metrics(y_train, y_pred_train)
            else:
                train_metrics = {"mse": float("inf"), "mae": float("inf"), "r2": -1.0}

            return {
                "success": training_success,
                "model_performance": train_metrics,
                "train_size": len(train_data),
            }

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {
                "success": False,
                "model_performance": {
                    "mse": float("inf"),
                    "mae": float("inf"),
                    "r2": -1.0,
                },
                "error": str(e),
            }

    def _generate_predictions(
        self,
        model: BaseModel,
        test_data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
    ) -> Optional[np.ndarray]:
        """
        Generate predictions for test data.

        Args:
            model: Trained model
            test_data: Test data
            target_column: Target column
            feature_columns: Feature columns

        Returns:
            Array of predictions or None if failed
        """
        try:
            if not hasattr(model, "predict"):
                logger.warning("Model does not have predict method")
                return None

            X_test = test_data[feature_columns].values
            predictions = model.predict(X_test)

            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None

    def _calculate_window_metrics(
        self,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        target_column: str,
        strategy: Optional[BaseStrategy],
        confidence_threshold: float,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a test window.

        Args:
            test_data: Test data
            predictions: Model predictions
            target_column: Target column
            strategy: Trading strategy (optional)
            confidence_threshold: Confidence threshold

        Returns:
            Dictionary of metrics
        """
        try:
            actual_values = test_data[target_column].values

            # Basic prediction metrics
            basic_metrics = self._calculate_basic_metrics(actual_values, predictions)

            # Trading metrics
            trading_metrics = self._calculate_trading_metrics(
                actual_values, predictions
            )

            # Strategy-based metrics (if strategy provided)
            strategy_metrics = {}
            if strategy is not None:
                strategy_metrics = self._calculate_strategy_metrics(
                    test_data, predictions, strategy, confidence_threshold
                )

            # Combine all metrics
            all_metrics = {**basic_metrics, **trading_metrics, **strategy_metrics}

            return all_metrics

        except Exception as e:
            logger.error(f"Error calculating window metrics: {e}")
            return {
                "sharpe_ratio": -1.0,
                "total_return": -1.0,
                "max_drawdown": -1.0,
                "win_rate": 0.0,
                "mse": float("inf"),
                "mae": float("inf"),
                "r2": -1.0,
            }

    def _calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic prediction metrics."""
        try:
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(mse)

            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -1.0

            # Directional accuracy
            if len(y_true) > 1:
                actual_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction)
            else:
                directional_accuracy = 0.0

            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "directional_accuracy": directional_accuracy,
            }

        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {
                "mse": float("inf"),
                "mae": float("inf"),
                "rmse": float("inf"),
                "r2": -1.0,
                "directional_accuracy": 0.0,
            }

    def _calculate_trading_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        try:
            if len(y_true) < 2:
                return {
                    "sharpe_ratio": -1.0,
                    "total_return": -1.0,
                    "max_drawdown": -1.0,
                    "win_rate": 0.0,
                    "volatility": 0.0,
                }

            # Calculate returns with division-by-zero protection
            actual_returns = np.where(
                y_true[:-1] > 1e-10,
                np.diff(y_true) / y_true[:-1],
                0.0
            )
            pred_returns = np.where(
                y_pred[:-1] > 1e-10,
                np.diff(y_pred) / y_pred[:-1],
                0.0
            )

            # Trading signals (simple strategy)
            signals = np.where(pred_returns > 0, 1, -1)
            strategy_returns = actual_returns * signals

            # Remove NaN values
            strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

            if len(strategy_returns) == 0:
                return {
                    "sharpe_ratio": -1.0,
                    "total_return": -1.0,
                    "max_drawdown": -1.0,
                    "win_rate": 0.0,
                    "volatility": 0.0,
                }

            # Sharpe ratio
            sharpe_ratio = calculate_sharpe_ratio(strategy_returns)

            # Total return
            total_return = (
                (np.prod(1 + strategy_returns) - 1)
                if len(strategy_returns) > 0
                else -1.0
            )

            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            # Safely calculate drawdown with division-by-zero protection
            drawdown = np.where(
                running_max > 1e-10,
                (cumulative_returns - running_max) / running_max,
                0.0
            )
            max_drawdown = np.min(drawdown)

            # Win rate
            win_rate = np.mean(strategy_returns > 0)

            # Volatility
            volatility = np.std(strategy_returns) * np.sqrt(self.trading_days_per_year)

            return {
                "sharpe_ratio": sharpe_ratio,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "volatility": volatility,
            }

        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {
                "sharpe_ratio": -1.0,
                "total_return": -1.0,
                "max_drawdown": -1.0,
                "win_rate": 0.0,
                "volatility": 0.0,
            }

    def _calculate_strategy_metrics(
        self,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        strategy: BaseStrategy,
        confidence_threshold: float,
    ) -> Dict[str, float]:
        """Calculate strategy-specific metrics."""
        try:
            # This would integrate with the actual strategy implementation
            # For now, return basic metrics
            return {
                "strategy_sharpe": -1.0,
                "strategy_return": -1.0,
                "strategy_drawdown": -1.0,
            }
        except Exception as e:
            logger.error(f"Error calculating strategy metrics: {e}")
            return {
                "strategy_sharpe": -1.0,
                "strategy_return": -1.0,
                "strategy_drawdown": -1.0,
            }

    def _generate_stability_report(
        self,
        window_results: List[Dict[str, Any]],
        all_metrics: List[Dict[str, float]],
        model_performance: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive stability report.

        Args:
            window_results: Results from all windows
            all_metrics: Metrics from all windows
            model_performance: Model performance from all windows

        Returns:
            Comprehensive stability report
        """
        if not all_metrics:
            return {"error": "No valid metrics available"}

        try:
            # Extract key metrics
            sharpe_ratios = [m.get("sharpe_ratio", -1.0) for m in all_metrics]
            total_returns = [m.get("total_return", -1.0) for m in all_metrics]
            max_drawdowns = [m.get("max_drawdown", -1.0) for m in all_metrics]
            win_rates = [m.get("win_rate", 0.0) for m in all_metrics]
            directional_accuracies = [
                m.get("directional_accuracy", 0.0) for m in all_metrics
            ]

            # Calculate statistics
            stability_report = {
                "summary": {
                    "total_windows": len(window_results),
                    "successful_windows": len(
                        [m for m in all_metrics if m.get("sharpe_ratio", -1.0) > -1.0]
                    ),
                    "failed_windows": len(
                        [m for m in all_metrics if m.get("sharpe_ratio", -1.0) <= -1.0]
                    ),
                },
                "sharpe_ratio": {
                    "mean": np.mean(sharpe_ratios),
                    "std": np.std(sharpe_ratios),
                    "min": np.min(sharpe_ratios),
                    "max": np.max(sharpe_ratios),
                    "median": np.median(sharpe_ratios),
                    "stability_score": self._calculate_stability_score(sharpe_ratios),
                },
                "total_return": {
                    "mean": np.mean(total_returns),
                    "std": np.std(total_returns),
                    "min": np.min(total_returns),
                    "max": np.max(total_returns),
                    "median": np.median(total_returns),
                },
                "max_drawdown": {
                    "mean": np.mean(max_drawdowns),
                    "std": np.std(max_drawdowns),
                    "min": np.min(max_drawdowns),
                    "max": np.max(max_drawdowns),
                    "median": np.median(max_drawdowns),
                },
                "win_rate": {
                    "mean": np.mean(win_rates),
                    "std": np.std(win_rates),
                    "min": np.min(win_rates),
                    "max": np.max(win_rates),
                    "median": np.median(win_rates),
                },
                "directional_accuracy": {
                    "mean": np.mean(directional_accuracies),
                    "std": np.std(directional_accuracies),
                    "min": np.min(directional_accuracies),
                    "max": np.max(directional_accuracies),
                    "median": np.median(directional_accuracies),
                },
                "stability_analysis": {
                    "sharpe_consistency": self._calculate_consistency(sharpe_ratios),
                    "return_consistency": self._calculate_consistency(total_returns),
                    "drawdown_consistency": self._calculate_consistency(max_drawdowns),
                    "overall_stability": self._calculate_overall_stability(all_metrics),
                },
                "trend_analysis": {
                    "sharpe_trend": self._calculate_trend(sharpe_ratios),
                    "return_trend": self._calculate_trend(total_returns),
                    "performance_degradation": self._detect_performance_degradation(
                        all_metrics
                    ),
                },
            }

            return stability_report

        except Exception as e:
            logger.error(f"Error generating stability report: {e}")
            return {"error": f"Failed to generate stability report: {str(e)}"}

    def _calculate_stability_score(self, values: List[float]) -> float:
        """Calculate stability score based on coefficient of variation."""
        try:
            if not values or np.std(values) == 0:
                return 1.0

            cv = (
                np.std(values) / abs(np.mean(values))
                if np.mean(values) != 0
                else float("inf")
            )
            # Convert to 0-1 scale where 1 is most stable
            stability_score = 1.0 / (1.0 + cv)
            return min(1.0, max(0.0, stability_score))
        except Exception:
            return 0.0

    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency score."""
        try:
            if len(values) < 2:
                return 0.0

            # Calculate how many values are within 1 standard deviation of mean
            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val == 0:
                return 1.0

            within_std = sum(1 for v in values if abs(v - mean_val) <= std_val)
            consistency = within_std / len(values)

            return consistency
        except Exception:
            return 0.0

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values over time."""
        try:
            if len(values) < 2:
                return 0.0

            # Simple linear trend
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)

            return slope
        except Exception:
            return 0.0

    def _detect_performance_degradation(
        self, all_metrics: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Detect if model performance is degrading over time."""
        try:
            if len(all_metrics) < 4:
                return {"degradation_detected": False, "confidence": 0.0}

            # Split into early and late periods
            mid_point = len(all_metrics) // 2
            early_metrics = all_metrics[:mid_point]
            late_metrics = all_metrics[mid_point:]

            # Compare Sharpe ratios
            early_sharpe = [m.get("sharpe_ratio", -1.0) for m in early_metrics]
            late_sharpe = [m.get("sharpe_ratio", -1.0) for m in late_metrics]

            early_mean = np.mean(early_sharpe)
            late_mean = np.mean(late_sharpe)

            degradation = late_mean < early_mean
            magnitude = abs(late_mean - early_mean)

            return {
                "degradation_detected": degradation,
                "magnitude": magnitude,
                "early_period_mean": early_mean,
                "late_period_mean": late_mean,
                "confidence": min(1.0, magnitude / (abs(early_mean) + 1e-8)),
            }
        except Exception:
            return {"degradation_detected": False, "confidence": 0.0}

    def _calculate_overall_stability(
        self, all_metrics: List[Dict[str, float]]
    ) -> float:
        """Calculate overall stability score."""
        try:
            if not all_metrics:
                return 0.0

            # Calculate stability for key metrics
            sharpe_stability = self._calculate_stability_score(
                [m.get("sharpe_ratio", -1.0) for m in all_metrics]
            )
            return_stability = self._calculate_stability_score(
                [m.get("total_return", -1.0) for m in all_metrics]
            )
            drawdown_stability = self._calculate_stability_score(
                [m.get("max_drawdown", -1.0) for m in all_metrics]
            )

            # Weighted average
            overall_stability = (
                0.5 * sharpe_stability
                + 0.3 * return_stability
                + 0.2 * drawdown_stability
            )

            return overall_stability
        except Exception:
            return 0.0

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        return {
            "total_evaluations": len(self.evaluation_results),
            "evaluation_types": list(self.evaluation_results.keys()),
            "latest_evaluation": (
                max(self.evaluation_results.keys()) if self.evaluation_results else None
            ),
        }


def walk_forward_backtest(
    model: BaseModel,
    data: pd.DataFrame,
    window_size: int = 100,
    step_size: int = 20,
    test_size: int = 20,
    target_column: str = "close",
    feature_columns: Optional[List[str]] = None,
    strategy: Optional[BaseStrategy] = None,
    confidence_threshold: float = 0.6,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for walk-forward backtesting.

    Args:
        model: Model to evaluate
        data: Historical data for backtesting
        window_size: Size of training window
        step_size: Step size for moving the window
        test_size: Size of test window
        target_column: Target column for prediction
        feature_columns: Feature columns for model input
        strategy: Trading strategy to use (optional)
        confidence_threshold: Minimum confidence for signal execution
        **kwargs: Additional parameters

    Returns:
        Dictionary containing stability report and detailed results
    """
    evaluator = ModelEvaluator(data, **kwargs)
    return evaluator.walk_forward_backtest(
        model=model,
        data=data,
        window_size=window_size,
        step_size=step_size,
        test_size=test_size,
        target_column=target_column,
        feature_columns=feature_columns,
        strategy=strategy,
        confidence_threshold=confidence_threshold,
    )
