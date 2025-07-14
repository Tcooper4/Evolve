"""Strategy Optimizer - Main Orchestrator.

This module orchestrates different optimization methods for trading strategies.
It has been refactored to use modular optimization components.
"""

import logging
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

    def optimize(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        method: str = "grid_search",
        **kwargs,
    ) -> OptimizationResult:
        """Run strategy optimization using the specified method.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            method: Optimization method to use
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult object
        """
        if method not in self.optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")

        self.logger.info(f"Starting {method} optimization")
        
        optimizer = self.optimization_methods[method]
        return optimizer.optimize(objective, param_space, data, **kwargs)

    def optimize_multiple_methods(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        methods: List[str] = None,
        **kwargs,
    ) -> Dict[str, OptimizationResult]:
        """Run optimization using multiple methods and compare results.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            methods: List of optimization methods to use
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
                    result = self.optimize(objective, param_space, data, method, **kwargs)
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
        }
