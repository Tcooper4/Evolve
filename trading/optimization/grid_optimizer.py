"""
Grid Search Optimizer.

This module implements a grid search optimizer that exhaustively searches through
a parameter space to find the optimal combination of parameters.
"""

from typing import Dict, List, Any, Union, Tuple
import itertools
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from .base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class GridOptimizer(BaseOptimizer):
    """Grid search optimizer implementation."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_type: str,
        verbose: bool = False,
        n_jobs: int = -1
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    ):
        """Initialize the grid optimizer.
        
        Args:
            data: DataFrame with OHLCV data
            strategy_type: Type of strategy to optimize
            verbose: Enable verbose logging
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__(data, strategy_type, verbose, n_jobs)
        self.best_params = None
        self.best_score = float('-inf')
        self.results = []
    
    def optimize(
        self,
        param_space: Dict[str, Union[List, Tuple]],
        objective: Union[str, List[str]],
        n_trials: int = None,  # Not used for grid search
        **kwargs
    ) -> List[OptimizationResult]:
        """Perform grid search optimization.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of values to try
            objective: Optimization objective(s)
            n_trials: Not used for grid search (all combinations are tried)
            **kwargs: Additional optimizer-specific arguments
            
        Returns:
            List of optimization results
        """
        logger.info(f"Starting grid search optimization for {self.strategy_type}")
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        logger.info(f"Grid search will evaluate {total_combinations} combinations")
        
        # Evaluate each combination
        for i, combination in enumerate(param_combinations):
            param_dict = dict(zip(param_names, combination))
            
            # Run strategy with parameters
            returns, signals, equity_curve = self._run_strategy(param_dict)
            
            # Calculate metrics
            metrics = self.calculate_metrics(returns, signals, equity_curve)
            
            # Create result
            result = OptimizationResult(
                parameters=param_dict,
                metrics=metrics,
                returns=returns,
                signals=signals,
                equity_curve=equity_curve,
                drawdown=(equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max(),
                timestamp=datetime.now(),
                optimization_type='grid_search',
                trial_id=i
            )
            
            # Log result
            self.log_result(result, i)
            
            # Update best parameters if better score found
            objective_value = metrics[objective] if isinstance(objective, str) else np.mean([metrics[obj] for obj in objective])
            if objective_value > self.best_score:
                self.best_score = objective_value
                self.best_params = param_dict
            
            # Store result
            self.results.append(result)
            
            if self.verbose and (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{total_combinations}, Best score: {self.best_score}")
        
        logger.info(f"Grid search completed. Best score: {self.best_score}")
        return {'success': True, 'result': self.results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _run_strategy(self, params: Dict[str, float]) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Run strategy with given parameters.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Tuple of (returns, signals, equity_curve)
        """
        # This should be implemented by strategy-specific optimizers
        raise NotImplementedError
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def get_best_params(self) -> Dict:
        """Get the best parameters found during optimization.
        
        Returns:
            Dictionary of best parameters
        """
        return {'success': True, 'result': self.best_params, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_results(
        self,
        plot_type: str = 'all',
        **kwargs
    ) -> None:
        """Plot optimization results.
        
        Args:
            plot_type: Type of plot ('all', 'history', 'importance', 'slice')
            **kwargs: Additional plot arguments
        """
        if not self.results:
            logger.warning("No results to plot")
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        # Create simple plots
        if plot_type in ['all', 'history']:
            self._plot_optimization_history()
        
        if plot_type in ['all', 'importance']:
            self._plot_parameter_importance()
    
    def _plot_optimization_history(self) -> None:
        """Plot optimization history."""
        import matplotlib.pyplot as plt
        
        scores = [result.metrics.get('sharpe_ratio', 0) for result in self.results]
        plt.figure(figsize=(10, 6))
        plt.plot(scores)
        plt.title('Grid Search Optimization History')
        plt.xlabel('Trial')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.show()
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def _plot_parameter_importance(self) -> None:
        """Plot parameter importance."""
        import matplotlib.pyplot as plt
        
        if not self.results:
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        # Calculate parameter importance based on correlation with performance
        param_importance = {}
        scores = [result.metrics.get('sharpe_ratio', 0) for result in self.results]
        
        for param_name in self.results[0].parameters.keys():
            param_values = [result.parameters[param_name] for result in self.results]
            correlation = np.corrcoef(param_values, scores)[0, 1]
            param_importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0
        
        # Plot importance
        plt.figure(figsize=(10, 6))
        param_names = list(param_importance.keys())
        importance_values = list(param_importance.values())
        
        plt.bar(param_names, importance_values)
        plt.title('Parameter Importance (Grid Search)')
        plt.xlabel('Parameter')
        plt.ylabel('Importance (|Correlation|)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 