"""Strategy Optimizer - Main Orchestrator.

This module orchestrates different optimization methods for trading strategies.
It has been refactored to use modular optimization components.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base_optimizer import BaseOptimizer, OptimizerConfig
from .grid_search_optimizer import GridSearch, OptimizationResult
from .bayesian_optimizer import BayesianOptimization
from .genetic_optimizer import GeneticAlgorithm
from .pso_optimizer import ParticleSwarmOptimization
from .ray_optimizer import RayTuneOptimization

logger = logging.getLogger(__name__)


class StrategyOptimizer(BaseOptimizer):
    """Main strategy optimizer that orchestrates different optimization methods."""

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize the strategy optimizer.

        Args:
            config: Optimizer configuration
        """
        super().__init__(config)
        self.optimization_methods = {
            "grid_search": GridSearch(),
            "bayesian": BayesianOptimization(),
            "genetic": GeneticAlgorithm(),
            "pso": ParticleSwarmOptimization(),
            "ray_tune": RayTuneOptimization(),
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Early stopping configuration
        self.early_stopping_config = {
            "enabled": self.config.get("early_stopping_enabled", True),
            "patience": self.config.get("early_stopping_patience", 10),
            "min_delta": self.config.get("early_stopping_min_delta", 0.001),
            "max_evaluations": self.config.get("max_evaluations", 1000),
            "timeout_seconds": self.config.get("timeout_seconds", 3600),  # 1 hour
        }

    def optimize(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        method: str = "grid_search",
        early_stopping: Optional[Dict[str, Any]] = None,
        max_evaluations: Optional[int] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Run strategy optimization using the specified method.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            method: Optimization method to use
            early_stopping: Early stopping configuration
            max_evaluations: Maximum number of evaluations
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult object
        """
        if method not in self.optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")

        self.logger.info(f"Starting {method} optimization")
        
        # Apply early stopping configuration
        if early_stopping is None:
            early_stopping = self.early_stopping_config.copy()
        else:
            # Merge with default config
            early_stopping = {**self.early_stopping_config, **early_stopping}
            
        # Apply max evaluations limit
        if max_evaluations is None:
            max_evaluations = early_stopping["max_evaluations"]
            
        # Create wrapped objective with early stopping
        wrapped_objective = self._create_early_stopping_objective(
            objective, early_stopping, max_evaluations
        )
        
        optimizer = self.optimization_methods[method]
        
        # Add early stopping parameters to kwargs
        kwargs.update({
            "early_stopping": early_stopping,
            "max_evaluations": max_evaluations,
            "timeout_seconds": early_stopping["timeout_seconds"]
        })
        
        return optimizer.optimize(wrapped_objective, param_space, data, **kwargs)

    def _create_early_stopping_objective(
        self, 
        objective: Callable, 
        early_stopping: Dict[str, Any],
        max_evaluations: int
    ) -> Callable:
        """Create an objective function with early stopping capabilities."""
        
        class EarlyStoppingObjective:
            def __init__(self, original_objective, config, max_evals):
                self.original_objective = original_objective
                self.config = config
                self.max_evaluations = max_evals
                self.evaluation_count = 0
                self.best_score = float('inf')
                self.patience_counter = 0
                self.start_time = time.time()
                self.scores_history = []
                
            def __call__(self, *args, **kwargs):
                # Check evaluation limit
                if self.evaluation_count >= self.max_evaluations:
                    self.logger.info(f"Reached maximum evaluations: {self.max_evaluations}")
                    return float('inf')
                    
                # Check timeout
                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.config["timeout_seconds"]:
                    self.logger.info(f"Optimization timeout after {elapsed_time:.1f} seconds")
                    return float('inf')
                    
                # Evaluate objective
                score = self.original_objective(*args, **kwargs)
                self.evaluation_count += 1
                self.scores_history.append(score)
                
                # Check for improvement
                if score < self.best_score - self.config["min_delta"]:
                    self.best_score = score
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                # Check early stopping
                if (self.config["enabled"] and 
                    self.patience_counter >= self.config["patience"] and
                    self.evaluation_count >= self.config["patience"]):
                    self.logger.info(f"Early stopping triggered after {self.evaluation_count} evaluations")
                    return float('inf')
                    
                return score
                
        return EarlyStoppingObjective(objective, early_stopping, max_evaluations)

    def optimize_with_early_stopping(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        method: str = "bayesian",
        patience: int = 10,
        min_delta: float = 0.001,
        max_evaluations: int = 500,
        timeout_seconds: int = 1800,
        **kwargs,
    ) -> OptimizationResult:
        """Run optimization with early stopping configuration.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            method: Optimization method to use
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum improvement required to reset patience
            max_evaluations: Maximum number of evaluations
            timeout_seconds: Maximum time in seconds
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult object
        """
        early_stopping_config = {
            "enabled": True,
            "patience": patience,
            "min_delta": min_delta,
            "max_evaluations": max_evaluations,
            "timeout_seconds": timeout_seconds,
        }
        
        return self.optimize(
            objective, param_space, data, method, 
            early_stopping=early_stopping_config,
            max_evaluations=max_evaluations,
            **kwargs
        )

    def optimize_multiple_methods(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        methods: List[str] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
        max_evaluations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, OptimizationResult]:
        """Run optimization using multiple methods and compare results.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            methods: List of optimization methods to use
            early_stopping: Early stopping configuration
            max_evaluations: Maximum number of evaluations
            **kwargs: Additional optimization parameters

        Returns:
            Dictionary mapping method names to results
        """
        if methods is None:
            methods = ["grid_search", "bayesian", "genetic"]

        results = {}
        
        for method in methods:
            if method in self.optimization_methods:
                try:
                    self.logger.info(f"Running {method} optimization")
                    result = self.optimize(
                        objective, param_space, data, method,
                        early_stopping=early_stopping,
                        max_evaluations=max_evaluations,
                        **kwargs
                    )
                    results[method] = result
                except Exception as e:
                    self.logger.error(f"Error in {method} optimization: {str(e)}")
                    continue

        return results

    def get_best_result(self, results: Dict[str, OptimizationResult]) -> tuple:
        """Get the best result from multiple optimization runs.

        Args:
            results: Dictionary of optimization results

        Returns:
            Tuple of (best_method, best_result)
        """
        if not results:
            return None, None

        best_method = min(results.keys(), key=lambda m: results[m].best_score)
        best_result = results[best_method]
        
        return best_method, best_result

    def compare_methods(self, results: Dict[str, OptimizationResult]) -> pd.DataFrame:
        """Compare results from different optimization methods.

        Args:
            results: Dictionary of optimization results

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for method, result in results.items():
            comparison_data.append({
                "method": method,
                "best_score": result.best_score,
                "optimization_time": result.optimization_time,
                "n_iterations": result.n_iterations,
                "convergence_rate": self._calculate_convergence_rate(result),
                "early_stopping_triggered": self._check_early_stopping(result),
            })

        return pd.DataFrame(comparison_data)

    def _calculate_convergence_rate(self, result: OptimizationResult) -> float:
        """Calculate convergence rate for an optimization result.

        Args:
            result: Optimization result

        Returns:
            Convergence rate (0-1)
        """
        if not result.convergence_history:
            return 0.0

        initial_score = result.convergence_history[0]
        final_score = result.convergence_history[-1]
        
        if initial_score == final_score:
            return 0.0

        improvement = initial_score - final_score
        total_possible_improvement = initial_score - min(result.convergence_history)
        
        if total_possible_improvement == 0:
            return 0.0

        return improvement / total_possible_improvement

    def _check_early_stopping(self, result: OptimizationResult) -> bool:
        """Check if early stopping was triggered for a result.

        Args:
            result: Optimization result

        Returns:
            True if early stopping was triggered
        """
        # This would need to be implemented based on the specific optimizer
        # For now, we'll check if the result has early stopping metadata
        if hasattr(result, 'metadata') and result.metadata:
            return result.metadata.get('early_stopping_triggered', False)
        return False

    def get_available_methods(self) -> List[str]:
        """Get list of available optimization methods.

        Returns:
            List of method names
        """
        return list(self.optimization_methods.keys())

    def get_method_info(self, method: str) -> Dict[str, Any]:
        """Get information about a specific optimization method.

        Args:
            method: Method name

        Returns:
            Dictionary with method information
        """
        if method not in self.optimization_methods:
            return {}

        optimizer = self.optimization_methods[method]
        
        return {
            "name": method,
            "class": optimizer.__class__.__name__,
            "description": optimizer.__doc__ or "",
            "config": optimizer.config,
            "supports_early_stopping": hasattr(optimizer, 'supports_early_stopping'),
        }

    def set_early_stopping_config(self, config: Dict[str, Any]):
        """Update early stopping configuration.

        Args:
            config: New early stopping configuration
        """
        self.early_stopping_config.update(config)
        self.logger.info(f"Updated early stopping config: {config}")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics.

        Returns:
            Dictionary with optimization statistics
        """
        return {
            "available_methods": self.get_available_methods(),
            "early_stopping_config": self.early_stopping_config,
            "total_optimizations": getattr(self, 'total_optimizations', 0),
        }
