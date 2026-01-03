from datetime import datetime
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from trading.utils.safe_math import safe_mape


class RegressionMetrics:
    """Metrics for regression models."""

    def mean_squared_error(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Return Mean Squared Error."""
        return {
            "success": True,
            "result": float(mean_squared_error(actuals, predictions)),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def root_mean_squared_error(
        self, actuals: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Return Root Mean Squared Error."""
        return {
            "success": True,
            "result": float(np.sqrt(mean_squared_error(actuals, predictions))),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def mean_absolute_error(
        self, actuals: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Return Mean Absolute Error."""
        return {
            "success": True,
            "result": float(mean_absolute_error(actuals, predictions)),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def r2_score(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Return R squared score."""
        return {
            "success": True,
            "result": float(r2_score(actuals, predictions)),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }


class ClassificationMetrics:
    """Metrics for classification models."""

    def accuracy(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Classification accuracy."""
        actuals = np.asarray(actuals)
        predictions = np.asarray(predictions)
        return float(np.mean(actuals == predictions))

    def precision(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Precision score for binary labels."""
        actuals = np.asarray(actuals)
        predictions = np.asarray(predictions)
        tp = np.sum((predictions == 1) & (actuals == 1))
        fp = np.sum((predictions == 1) & (actuals == 0))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def recall(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Recall score for binary labels."""
        actuals = np.asarray(actuals)
        predictions = np.asarray(predictions)
        tp = np.sum((predictions == 1) & (actuals == 1))
        fn = np.sum((predictions == 0) & (actuals == 1))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def f1_score(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """F1 score for binary labels."""
        prec = self.precision(actuals, predictions)
        rec = self.recall(actuals, predictions)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


class TimeSeriesMetrics:
    """Class for calculating time series specific metrics."""

    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics = {}

    def mean_absolute_percentage_error(
        self, actuals: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Return MAPE."""
        return float(safe_mape(actuals, predictions))

    def symmetric_mean_absolute_percentage_error(
        self, actuals: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Return SMAPE."""
        actuals = np.asarray(actuals)
        predictions = np.asarray(predictions)
        denominator = (np.abs(actuals) + np.abs(predictions)) / 2
        mask = denominator != 0
        return float(
            np.mean(np.abs(actuals[mask] - predictions[mask]) / denominator[mask]) * 100
        )

    def mean_absolute_scaled_error(
        self, actuals: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Return MASE using naive forecast for scaling."""
        actuals = np.asarray(actuals)
        predictions = np.asarray(predictions)
        naive_forecast = actuals[:-1]
        denom = np.mean(np.abs(actuals[1:] - naive_forecast))
        if denom == 0:
            return float("inf")
        return float(np.mean(np.abs(actuals - predictions)) / denom)

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all time series metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names and values
        """
        self.metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": self.mean_absolute_percentage_error(y_true, y_pred),
            "smape": self.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            "mase": self.mean_absolute_scaled_error(y_true, y_pred),
        }
        return self.metrics

    def get_metric(self, metric_name: str) -> float:
        """Get a specific metric value.

        Args:
            metric_name: Name of the metric to retrieve

        Returns:
            Metric value
        """
        if not self.metrics:
            raise ValueError("No metrics have been calculated yet")
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not found")
        return self.metrics[metric_name]

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all calculated metrics.

        Returns:
            Dictionary of all metric names and values
        """
        if not self.metrics:
            raise ValueError("No metrics have been calculated yet")
        return self.metrics.copy()


class RiskMetrics:
    """Class for calculating risk metrics."""

    def __init__(self):
        """Initialize the risk metrics calculator."""
        self.metrics = {}

    def calculate_metrics(
        self, returns: np.ndarray, risk_free_rate: float = 0.0, window: int = 252
    ) -> Dict[str, float]:
        """Calculate risk metrics.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            window: Number of periods in a year (252 for daily data)

        Returns:
            Dictionary of risk metrics
        """
        annualized_return = (1 + np.mean(returns)) ** window - 1
        annualized_volatility = np.std(returns) * np.sqrt(window)

        sharpe_ratio = self.sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = self.sortino_ratio(returns, risk_free_rate)
        max_drawdown = self.maximum_drawdown(returns)
        var_95 = self.value_at_risk(returns)
        es_95 = np.mean(returns[returns <= var_95])

        self.metrics = {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "expected_shortfall_95": es_95,
        }

        return self.metrics

    def sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate the Sharpe ratio."""
        excess = returns - risk_free_rate / len(returns)
        return (
            float(np.sqrt(252) * excess.mean() / excess.std())
            if excess.std() != 0
            else 0.0
        )

    def sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate the Sortino ratio."""
        downside = returns[returns < 0]
        downside_std = downside.std()
        if downside_std == 0:
            return 0.0
        excess = returns.mean() - risk_free_rate / len(returns)
        return float(np.sqrt(252) * excess / downside_std)

    def maximum_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        
        drawdowns = np.where(
            running_max > 1e-10,
            cumulative / running_max - 1,
            0.0
        )
        
        return float(drawdowns.min())

    def value_at_risk(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (historical)."""
        return float(np.percentile(returns, (1 - confidence) * 100))

    def get_metric(self, metric_name: str) -> float:
        """Get a specific risk metric value.

        Args:
            metric_name: Name of the metric to retrieve

        Returns:
            Metric value
        """
        if not self.metrics:
            raise ValueError("No metrics have been calculated yet")
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not found")
        return self.metrics[metric_name]

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all calculated risk metrics.

        Returns:
            Dictionary of all metric names and values
        """
        if not self.metrics:
            raise ValueError("No metrics have been calculated yet")
        return self.metrics.copy()


# Standalone functions for compatibility with existing imports


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe ratio for a series of returns using safe division."""
    from trading.utils.safe_math import safe_sharpe_ratio
    if len(returns) == 0:
        return 0.0
    # Convert to pandas Series for safe_sharpe_ratio
    import pandas as pd
    returns_series = pd.Series(returns)
    return safe_sharpe_ratio(returns_series, risk_free_rate=risk_free_rate, periods_per_year=252)


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown for a series of returns."""
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(drawdowns.min())


def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate win rate (percentage of positive returns)."""
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))
