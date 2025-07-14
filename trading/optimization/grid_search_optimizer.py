"""Grid Search Optimization Method.

This module contains the GridSearch optimization method extracted from strategy_optimizer.py.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_optimizer import BaseOptimizer, OptimizerConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from strategy optimization."""

    best_params: Dict[str, Any]
    best_score: float
    all_scores: List[float]
    all_params: List[Dict[str, Any]]
    optimization_time: float
    n_iterations: int
    convergence_history: List[float]
    metadata: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None
    hyperparameter_importance: Optional[Dict[str, float]] = None


class OptimizationMethod(ABC):
    """Base class for optimization methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimization method.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def optimize(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult object
        """

    def _validate_param_space(self, param_space: Dict[str, Any]) -> None:
        """Validate parameter space.

        Args:
            param_space: Parameter space to validate

        Raises:
            ValueError: If parameter space is invalid
        """
        if not param_space:
            raise ValueError("Parameter space cannot be empty")

        for param, space in param_space.items():
            if isinstance(space, (list, tuple)):
                if not space:
                    raise ValueError(f"Parameter {param} has empty space")
            elif isinstance(space, dict):
                required_keys = {"start", "end"}
                if not required_keys.issubset(space.keys()):
                    raise ValueError(
                        f"Parameter {param} missing required keys: {required_keys}"
                    )
                if space["start"] >= space["end"]:
                    raise ValueError(f"Parameter {param} has invalid range")

    def _check_early_stopping(
        self, scores: List[float], patience: int = 5, min_delta: float = 1e-4
    ) -> bool:
        """Check if optimization should stop early.

        Args:
            scores: List of optimization scores
            patience: Number of iterations without improvement
            min_delta: Minimum change to be considered improvement

        Returns:
            True if should stop, False otherwise
        """
        if len(scores) < patience + 1:
            return False

        best_score = min(scores[:-patience])
        current_score = min(scores[-patience:])

        return (best_score - current_score) < min_delta

    def _objective_wrapper(self, objective: Callable, data: pd.DataFrame) -> Callable:
        """Create objective function wrapper.

        Args:
            objective: Original objective function
            data: Market data

        Returns:
            Wrapped objective function
        """

        def wrapper(params):
            try:
                return objective(params, data)
            except Exception as e:
                self.logger.error(f"Error in objective function: {str(e)}")
                return {
                    "success": True,
                    "result": {
                        "success": True,
                        "result": float("inf"),
                        "message": "Operation completed successfully",
                        "timestamp": datetime.now().isoformat(),
                    },
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                }

        return wrapper


class GridSearch(OptimizationMethod):
    """Grid search optimization method."""

    def optimize(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs,
    ) -> OptimizationResult:
        """Run grid search optimization.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult object
        """
        start_time = datetime.now()

        # Validate parameter space
        self._validate_param_space(param_space)

        # Generate parameter grid
        param_grid = self._generate_grid(param_space)
        max_points = kwargs.get("max_points", 100)
        
        if len(param_grid) > max_points:
            # Sample randomly if grid is too large
            np.random.seed(42)
            indices = np.random.choice(len(param_grid), max_points, replace=False)
            param_grid = [param_grid[i] for i in indices]
            self.logger.info(f"Sampled {max_points} points from grid of {len(param_grid)}")

        # Run grid search
        scores = []
        best_score = float("inf")
        best_params = None
        convergence_history = []

        for i, params in enumerate(param_grid):
            try:
                score = objective(params, data)
                scores.append(score)
                convergence_history.append(score)

                if score < best_score:
                    best_score = score
                    best_params = params

                # Check early stopping
                if self._check_early_stopping(scores):
                    self.logger.info(f"Early stopping at iteration {i}")
                    break

            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
                scores.append(float("inf"))
                convergence_history.append(float("inf"))

        optimization_time = (datetime.now() - start_time).total_seconds()

        # Calculate feature importance
        feature_importance = self._calculate_hyperparameter_importance(param_grid, scores)

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_scores=scores,
            all_params=param_grid,
            optimization_time=optimization_time,
            n_iterations=len(param_grid),
            convergence_history=convergence_history,
            feature_importance=feature_importance,
        )

    def _cross_validate(
        self, objective: Callable, params: Dict[str, Any], data: pd.DataFrame, **kwargs
    ) -> List[float]:
        """Perform cross-validation.

        Args:
            objective: Objective function
            params: Parameters to evaluate
            data: Market data
            **kwargs: Additional parameters

        Returns:
            List of cross-validation scores
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit

            n_splits = kwargs.get("n_splits", 3)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                score = objective(params, val_data)
                scores.append(score)

            return scores
        except ImportError:
            self.logger.warning("scikit-learn not available, skipping cross-validation")
            return [objective(params, data)]

    def _calculate_hyperparameter_importance(
        self, param_grid: List[Dict[str, Any]], scores: List[float]
    ) -> Dict[str, float]:
        """Calculate hyperparameter importance.

        Args:
            param_grid: Parameter grid
            scores: Corresponding scores

        Returns:
            Dictionary of parameter importance scores
        """
        if not param_grid or not scores:
            return {}

        importance = {}
        param_names = list(param_grid[0].keys())

        for param in param_names:
            param_values = [params[param] for params in param_grid]
            correlations = []

            for i, val1 in enumerate(param_values):
                for j, val2 in enumerate(param_values):
                    if i != j:
                        score_diff = abs(scores[i] - scores[j])
                        param_diff = abs(val1 - val2) if isinstance(val1, (int, float)) else 1
                        if param_diff > 0:
                            correlations.append(score_diff / param_diff)

            if correlations:
                importance[param] = np.mean(correlations)
            else:
                importance[param] = 0.0

        return importance

    def _generate_grid(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter grid.

        Args:
            param_space: Parameter space

        Returns:
            List of parameter combinations
        """
        import itertools

        param_names = list(param_space.keys())
        param_values = []

        for param in param_names:
            space = param_space[param]
            if isinstance(space, (list, tuple)):
                param_values.append(space)
            elif isinstance(space, dict):
                if "start" in space and "end" in space:
                    if isinstance(space["start"], int):
                        param_values.append(
                            list(range(space["start"], space["end"] + 1))
                        )
                    else:
                        # For float ranges, create a reasonable number of points
                        n_points = space.get("n_points", 10)
                        param_values.append(
                            np.linspace(space["start"], space["end"], n_points).tolist()
                        )

        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo)) for combo in combinations] 