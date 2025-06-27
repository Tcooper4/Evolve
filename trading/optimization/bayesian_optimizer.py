"""Bayesian optimizer using Optuna."""

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice
)
from typing import Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from .base_optimizer import BaseOptimizer, OptimizationResult

class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimizer using Optuna."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_type: str,
        verbose: bool = False,
        n_jobs: int = -1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        """Initialize Bayesian optimizer.
        
        Args:
            data: DataFrame with OHLCV data
            strategy_type: Type of strategy to optimize
            verbose: Enable verbose logging
            n_jobs: Number of parallel jobs (-1 for all cores)
            study_name: Name of the Optuna study
            storage: Storage URL for Optuna study
        """
        super().__init__(data, strategy_type, verbose, n_jobs)
        self.study_name = study_name or f"{strategy_type}_optimization"
        self.storage = storage
        self.study = None
    
    def optimize(
        self,
        param_space: Dict[str, Union[List, Tuple]],
        objective: Union[str, List[str]],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs
    ) -> List[OptimizationResult]:
        """Run Bayesian optimization.
        
        Args:
            param_space: Parameter space to search
            objective: Optimization objective(s)
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            early_stopping_rounds: Number of rounds for early stopping
            **kwargs: Additional optimizer-specific arguments
            
        Returns:
            List of optimization results
        """
        # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            directions=['maximize'] if isinstance(objective, str) else ['maximize'] * len(objective)
        )
        
        # Define objective function
        def objective_fn(trial):
            # Sample parameters
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, (list, tuple)):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, *param_range)
                    else:
                        params[param_name] = trial.suggest_float(param_name, *param_range)
            
            # Run strategy
            returns, signals, equity_curve = self._run_strategy(params)
            
            # Calculate metrics
            metrics = self.calculate_metrics(returns, signals, equity_curve)
            
            # Create result
            result = OptimizationResult(
                parameters=params,
                metrics=metrics,
                returns=returns,
                signals=signals,
                equity_curve=equity_curve,
                drawdown=(equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max(),
                timestamp=datetime.now(),
                optimization_type='bayesian',
                trial_id=trial.number
            )
            
            # Log result
            self.log_result(result, trial.number)
            
            # Return objective value(s)
            if isinstance(objective, str):
                return metrics[objective]
            else:
                return [metrics[obj] for obj in objective]
        
        # Run optimization
        self.study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            callbacks=[
                optuna.callbacks.EarlyStoppingCallback(
                    early_stopping_rounds=early_stopping_rounds
                ) if early_stopping_rounds else None
            ]
        )
        
        return self.get_all_results()
    
    def _run_strategy(self, params: Dict[str, float]) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Run strategy with given parameters.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Tuple of (returns, signals, equity_curve)
        """
        # This should be implemented by strategy-specific optimizers
        raise NotImplementedError
    
    def plot_results(
        self,
        plot_type: str = 'all',
        **kwargs
    ) -> Union[go.Figure, List[go.Figure]]:
        """Plot optimization results.
        
        Args:
            plot_type: Type of plot ('all', 'history', 'importance', 'slice')
            **kwargs: Additional plot arguments
            
        Returns:
            Plotly figure(s)
        """
        if not self.study:
            raise ValueError("No optimization study found")
        
        plots = []
        
        if plot_type in ['all', 'history']:
            plots.append(plot_optimization_history(self.study))
        
        if plot_type in ['all', 'importance']:
            plots.append(plot_param_importances(self.study))
        
        if plot_type in ['all', 'slice']:
            plots.append(plot_slice(self.study))
        
        return plots[0] if len(plots) == 1 else plots
    
    def get_best_trials(self, n_trials: int = 1) -> List[optuna.trial.FrozenTrial]:
        """Get best trials from study.
        
        Args:
            n_trials: Number of best trials to return
            
        Returns:
            List of best trials
        """
        if not self.study:
            raise ValueError("No optimization study found")
        
        return self.study.best_trials[:n_trials]
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance scores.
        
        Returns:
            Dictionary of parameter importance scores
        """
        if not self.study:
            raise ValueError("No optimization study found")
        
        importance = optuna.importance.get_param_importances(self.study)
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def suggest_parameters(self) -> Dict[str, float]:
        """Get suggested parameters from study.
        
        Returns:
            Dictionary of suggested parameters
        """
        if not self.study:
            raise ValueError("No optimization study found")
        
        return self.study.best_params 