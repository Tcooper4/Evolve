"""Bayesian Optimization Method.

This module contains the BayesianOptimization method extracted from strategy_optimizer.py.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .grid_search_optimizer import OptimizationMethod, OptimizationResult

logger = logging.getLogger(__name__)


class BayesianOptimization(OptimizationMethod):
    """Bayesian optimization method."""

    def optimize(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs,
    ) -> OptimizationResult:
        """Run Bayesian optimization.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult object
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            raise ImportError("scikit-optimize is required for Bayesian optimization")

        start_time = datetime.now()

        # Validate parameter space
        self._validate_param_space(param_space)

        # Convert parameter space to skopt format
        dimensions = []
        param_names = []
        
        for param, space in param_space.items():
            param_names.append(param)
            if isinstance(space, (list, tuple)):
                dimensions.append(Categorical(space, name=param))
            elif isinstance(space, dict):
                if "start" in space and "end" in space:
                    if isinstance(space["start"], int):
                        dimensions.append(Integer(space["start"], space["end"], name=param))
                    else:
                        dimensions.append(Real(space["start"], space["end"], name=param))

        # Define objective function
        @use_named_args(dimensions=dimensions)
        def objective_wrapper(**params):
            try:
                return objective(params, data)
            except Exception as e:
                self.logger.error(f"Error in objective function: {str(e)}")
                return float("inf")

        # Run optimization
        n_calls = kwargs.get("n_calls", 50)
        n_initial_points = kwargs.get("n_initial_points", 10)
        
        result = gp_minimize(
            objective_wrapper,
            dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=42,
        )

        optimization_time = (datetime.now() - start_time).total_seconds()

        # Convert results
        best_params = dict(zip(param_names, result.x))
        best_score = result.fun
        all_scores = [-score for score in result.func_vals]  # Convert back to minimization
        all_params = [dict(zip(param_names, x)) for x in result.x_iters]

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(result, param_names)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=all_scores,
            all_params=all_params,
            optimization_time=optimization_time,
            n_iterations=n_calls,
            convergence_history=all_scores,
            feature_importance=feature_importance,
        )

    def _calculate_feature_importance(self, result, param_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance from optimization results.

        Args:
            result: Optimization result from skopt
            param_names: List of parameter names

        Returns:
            Dictionary of parameter importance scores
        """
        try:
            # Use the GP model to estimate feature importance
            if hasattr(result, 'models') and result.models:
                model = result.models[-1]
                if hasattr(model, 'kernel_'):
                    # Extract length scales from the kernel
                    length_scales = model.kernel_.length_scale
                    if length_scales is not None:
                        importance = {}
                        for i, param in enumerate(param_names):
                            if i < len(length_scales):
                                # Inverse of length scale is importance
                                importance[param] = 1.0 / length_scales[i]
                            else:
                                importance[param] = 0.0
                        return importance
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {e}")

        # Fallback: return equal importance
        return {param: 1.0 / len(param_names) for param in param_names}
