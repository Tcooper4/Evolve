"""Walk-Forward Validation and Rolling Retraining Agent.

This agent implements walk-forward validation to prevent data leakage and simulate live deployment
by retraining models on rolling windows of data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardResult:
    """Walk-forward validation result."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    model_performance: Dict[str, float]
    predictions: pd.Series
    actual_values: pd.Series
    model_metadata: Dict[str, Any]

@dataclass
class RollingRetrainConfig:
    """Configuration for rolling retraining."""
    train_window_days: int = 252  # 1 year
    test_window_days: int = 63    # 3 months
    step_size_days: int = 21      # 1 month
    min_train_samples: int = 100
    max_train_samples: int = 1000
    validation_split: float = 0.2
    retrain_frequency: str = 'monthly'  # daily, weekly, monthly
    model_type: str = 'xgboost'  # xgboost, lstm, transformer, ensemble
    feature_engineering: bool = True
    hyperparameter_tuning: bool = True

class WalkForwardAgent:
    """Walk-forward validation and rolling retraining agent."""
    
    def __init__(self, config: Optional[RollingRetrainConfig] = None):
        """Initialize walk-forward agent.
        
        Args:
            config: Configuration for rolling retraining
        """
        self.config = config or RollingRetrainConfig()
        self.results_history = []
        self.model_history = []
        self.performance_tracker = {}
        
        logger.info("Walk-Forward Agent initialized")
    
    def run_walk_forward_validation(self, 
                                  data: pd.DataFrame,
                                  target_column: str,
                                  feature_columns: List[str],
                                  model_factory: Callable,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> List[WalkForwardResult]:
        """Run walk-forward validation on the data.
        
        Args:
            data: Input data with datetime index
            target_column: Target variable column
            feature_columns: Feature columns
            model_factory: Function to create model instances
            start_date: Start date for validation
            end_date: End date for validation
            
        Returns:
            List of walk-forward results
        """
        try:
            # Prepare data
            data = data.copy()
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            # Set date range
            if start_date is None:
                start_date = data.index.min()
            if end_date is None:
                end_date = data.index.max()
            
            # Calculate windows
            train_window = timedelta(days=self.config.train_window_days)
            test_window = timedelta(days=self.config.test_window_days)
            step_size = timedelta(days=self.config.step_size_days)
            
            results = []
            current_date = start_date + train_window
            
            while current_date + test_window <= end_date:
                # Define train and test periods
                train_start = current_date - train_window
                train_end = current_date
                test_start = current_date
                test_end = current_date + test_window
                
                # Extract train and test data
                train_data = data[(data.index >= train_start) & (data.index < train_end)]
                test_data = data[(data.index >= test_start) & (data.index < test_end)]
                
                # Check minimum data requirements
                if len(train_data) < self.config.min_train_samples:
                    logger.warning(f"Insufficient training data for period {train_start} to {train_end}")
                    current_date += step_size
                    continue
                
                # Train model
                model, train_metrics = self._train_model(
                    train_data, target_column, feature_columns, model_factory
                )
                
                # Make predictions
                predictions = self._make_predictions(model, test_data, feature_columns)
                
                # Calculate performance metrics
                performance = self._calculate_performance(
                    predictions, test_data[target_column]
                )
                
                # Store results
                result = WalkForwardResult(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    model_performance=performance,
                    predictions=predictions,
                    actual_values=test_data[target_column],
                    model_metadata={
                        'train_metrics': train_metrics,
                        'model_type': self.config.model_type,
                        'feature_count': len(feature_columns)
                    }
                )
                
                results.append(result)
                self.results_history.append(result)
                
                # Update performance tracker
                self._update_performance_tracker(performance, test_start)
                
                logger.info(f"Walk-forward step: {test_start} to {test_end} - "
                          f"Sharpe: {performance.get('sharpe_ratio', 0):.3f}")
                
                current_date += step_size
            
            logger.info(f"Walk-forward validation completed with {len(results)} steps")
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward validation: {e}")
            return []
    
    def _train_model(self, 
                    train_data: pd.DataFrame,
                    target_column: str,
                    feature_columns: List[str],
                    model_factory: Callable) -> Tuple[Any, Dict[str, float]]:
        """Train model on training data.
        
        Args:
            train_data: Training data
            target_column: Target variable
            feature_columns: Feature columns
            model_factory: Model factory function
            
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        try:
            # Prepare features and target
            X = train_data[feature_columns].fillna(0)
            y = train_data[target_column]
            
            # Split for validation
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Create and train model
            model = model_factory()
            
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
                
                # Validate on validation set
                val_pred = model.predict(X_val)
                train_metrics = self._calculate_performance(
                    pd.Series(val_pred, index=y_val.index), y_val
                )
            else:
                train_metrics = {}
            
            return model, train_metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None, {}
    
    def _make_predictions(self, model: Any, test_data: pd.DataFrame, feature_columns: List[str]) -> pd.Series:
        """Make predictions using trained model.
        
        Args:
            model: Trained model
            test_data: Test data
            feature_columns: Feature columns
            
        Returns:
            Predictions series
        """
        try:
            if model is None:
                return pd.Series(index=test_data.index, dtype=float)
            
            X_test = test_data[feature_columns].fillna(0)
            
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
                return pd.Series(predictions, index=test_data.index)
            else:
                return pd.Series(index=test_data.index, dtype=float)
                
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return pd.Series(index=test_data.index, dtype=float)
    
    def _calculate_performance(self, predictions: pd.Series, actual: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            predictions: Predicted values
            actual: Actual values
            
        Returns:
            Performance metrics dictionary
        """
        try:
            if len(predictions) == 0 or len(actual) == 0:
                return {}
            
            # Align series
            aligned_data = pd.DataFrame({
                'pred': predictions,
                'actual': actual
            }).dropna()
            
            if len(aligned_data) == 0:
                return {}
            
            pred = aligned_data['pred']
            actual = aligned_data['actual']
            
            # Calculate returns
            returns = (actual - pred.shift(1)) / pred.shift(1)
            returns = returns.dropna()
            
            if len(returns) == 0:
                return {}
            
            # Performance metrics
            metrics = {
                'mse': np.mean((actual - pred) ** 2),
                'mae': np.mean(np.abs(actual - pred)),
                'rmse': np.sqrt(np.mean((actual - pred) ** 2)),
                'mape': np.mean(np.abs((actual - pred) / actual)) * 100,
                'r2': 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2),
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'win_rate': (returns > 0).mean(),
                'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            returns: Return series
            
        Returns:
            Maximum drawdown
        """
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def _update_performance_tracker(self, performance: Dict[str, float], test_start: datetime):
        """Update performance tracking.
        
        Args:
            performance: Performance metrics
            test_start: Test period start date
        """
        for metric, value in performance.items():
            if metric not in self.performance_tracker:
                self.performance_tracker[metric] = []
            self.performance_tracker[metric].append({
                'date': test_start,
                'value': value
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of walk-forward performance.
        
        Returns:
            Performance summary
        """
        if not self.results_history:
            return {}
        
        summary = {}
        
        # Aggregate metrics across all periods
        for metric in ['sharpe_ratio', 'r2', 'mse', 'mae', 'rmse', 'mape']:
            values = [r.model_performance.get(metric, 0) for r in self.results_history]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        # Performance stability
        sharpe_values = [r.model_performance.get('sharpe_ratio', 0) for r in self.results_history]
        if sharpe_values:
            summary['performance_stability'] = np.std(sharpe_values)
            summary['positive_periods'] = sum(1 for s in sharpe_values if s > 0)
            summary['total_periods'] = len(sharpe_values)
            summary['success_rate'] = summary['positive_periods'] / summary['total_periods']
        
        # Model metadata
        summary['total_models_trained'] = len(self.results_history)
        summary['avg_training_samples'] = np.mean([
            r.model_metadata.get('feature_count', 0) for r in self.results_history
        ])
        
        return summary
    
    def get_performance_trends(self) -> Dict[str, pd.Series]:
        """Get performance trends over time.
        
        Returns:
            Dictionary of performance trends
        """
        trends = {}
        
        for metric in self.performance_tracker:
            data = pd.DataFrame(self.performance_tracker[metric])
            if len(data) > 0:
                data.set_index('date', inplace=True)
                trends[metric] = data['value']
        
        return trends
    
    def should_retrain(self, current_performance: Dict[str, float], threshold: float = 0.1) -> bool:
        """Determine if model should be retrained based on performance degradation.
        
        Args:
            current_performance: Current model performance
            threshold: Performance degradation threshold
            
        Returns:
            True if retraining is recommended
        """
        if not self.results_history:
            return True
        
        # Get recent performance
        recent_results = self.results_history[-5:]  # Last 5 periods
        if not recent_results:
            return True
        
        recent_sharpe = [r.model_performance.get('sharpe_ratio', 0) for r in recent_results]
        avg_recent_sharpe = np.mean(recent_sharpe)
        
        current_sharpe = current_performance.get('sharpe_ratio', 0)
        
        # Check for performance degradation
        if avg_recent_sharpe > 0 and current_sharpe < avg_recent_sharpe * (1 - threshold):
            logger.info(f"Performance degradation detected. Recent avg: {avg_recent_sharpe:.3f}, "
                       f"Current: {current_sharpe:.3f}. Retraining recommended.")
            return True
        
        return False
    
    def export_results(self, filepath: str) -> bool:
        """Export walk-forward results to file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export successful
        """
        try:
            results_data = []
            
            for result in self.results_history:
                row = {
                    'train_start': result.train_start,
                    'train_end': result.train_end,
                    'test_start': result.test_start,
                    'test_end': result.test_end,
                    **result.model_performance,
                    'model_type': result.model_metadata.get('model_type', ''),
                    'feature_count': result.model_metadata.get('feature_count', 0)
                }
                results_data.append(row)
            
            df = pd.DataFrame(results_data)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Walk-forward results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False

# Global walk-forward agent instance
walk_forward_agent = WalkForwardAgent()

def get_walk_forward_agent() -> WalkForwardAgent:
    """Get the global walk-forward agent instance."""
    return walk_forward_agent 