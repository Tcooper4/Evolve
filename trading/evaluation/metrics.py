import numpy as np
from typing import Union, List, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionMetrics:
    def __init__(self):
        pass

    def calculate_accuracy(self, predictions, actuals):
        """Calculate the accuracy of predictions."""
        return 0.0

    def calculate_precision(self, predictions, actuals):
        """Calculate the precision of predictions."""
        return 0.0

    def calculate_recall(self, predictions, actuals):
        """Calculate the recall of predictions."""
        return 0.0

class ClassificationMetrics:
    def __init__(self):
        pass

    def calculate_accuracy(self, predictions, actuals):
        """Calculate the accuracy of predictions."""
        return 0.0

    def calculate_precision(self, predictions, actuals):
        """Calculate the precision of predictions."""
        return 0.0

    def calculate_recall(self, predictions, actuals):
        """Calculate the recall of predictions."""
        return 0.0 

class TimeSeriesMetrics:
    """Class for calculating time series specific metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all time series metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        self.metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': self._calculate_mape(y_true, y_pred)
        }
        return self.metrics
        
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value
        """
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
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
        
    def calculate_metrics(self, returns: np.ndarray, 
                         risk_free_rate: float = 0.0,
                         window: int = 252) -> Dict[str, float]:
        """Calculate risk metrics.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            window: Number of periods in a year (252 for daily data)
            
        Returns:
            Dictionary of risk metrics
        """
        # Calculate basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Calculate annualized metrics
        annualized_return = (1 + mean_return) ** window - 1
        annualized_volatility = std_return * np.sqrt(window)
        
        # Calculate Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        
        # Calculate Expected Shortfall (ES)
        es_95 = np.mean(returns[returns <= var_95])
        
        self.metrics = {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'expected_shortfall_95': es_95
        }
        
        return self.metrics
        
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