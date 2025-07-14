"""Ray Tune Optimization Method.

This module contains the RayTuneOptimization method extracted from strategy_optimizer.py.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .grid_search_optimizer import OptimizationMethod, OptimizationResult

logger = logging.getLogger(__name__)


class RayTuneOptimization(OptimizationMethod):
    """Ray Tune optimization method."""

    def optimize(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs,
    ) -> OptimizationResult:
        """Run Ray Tune optimization.

        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult object
        """
        try:
            import ray
            from ray import tune
            from ray.tune.schedulers import ASHAScheduler
            from ray.tune.search.bayesopt import BayesOptSearch
        except ImportError:
            raise ImportError("Ray Tune is required for this optimization method")

        start_time = datetime.now()

        # Validate parameter space
        self._validate_param_space(param_space)

        # Convert parameter space to Ray Tune format
        tune_param_space = self._convert_to_tune_space(param_space)

        # Get Ray Tune parameters
        num_samples = kwargs.get("num_samples", 100)
        max_concurrent = kwargs.get("max_concurrent", 4)

        # Define objective function for Ray Tune
        def tune_objective(config):
            try:
                score = objective(config, data)
                tune.report(score=score)
                return score
            except Exception as e:
                logger.error(f"Error in objective function: {str(e)}")
                tune.report(score=float("inf"))
                return float("inf")

        # Configure scheduler and search algorithm
        scheduler = ASHAScheduler(
            metric="score",
            mode="min",
            max_t=num_samples,
            grace_period=10,
            reduction_factor=2,
        )

        search_alg = BayesOptSearch(
            metric="score",
            mode="min",
        )

        # Run optimization
        analysis = tune.run(
            tune_objective,
            config=tune_param_space,
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            max_concurrent_trials=max_concurrent,
            local_dir="./ray_results",
            name="strategy_optimization",
            verbose=1,
        )

        optimization_time = (datetime.now() - start_time).total_seconds()

        # Extract results
        best_trial = analysis.get_best_trial("score", "min")
        best_params = best_trial.config
        best_score = best_trial.last_result["score"]

        # Extract all results
        all_scores = []
        all_params = []
        for trial in analysis.trials:
            if trial.last_result:
                all_scores.append(trial.last_result["score"])
                all_params.append(trial.config)

        # Calculate convergence history
        convergence_history = []
        for trial in analysis.trials:
            if trial.last_result:
                convergence_history.append(trial.last_result["score"])

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=all_scores,
            all_params=all_params,
            optimization_time=optimization_time,
            n_iterations=num_samples,
            convergence_history=convergence_history,
        )

    def _convert_to_tune_space(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter space to Ray Tune format.

        Args:
            param_space: Original parameter space

        Returns:
            Ray Tune parameter space
        """
        try:
            from ray import tune
        except ImportError:
            raise ImportError("Ray Tune is required for this optimization method")

        tune_space = {}

        for param, space in param_space.items():
            if isinstance(space, (list, tuple)):
                tune_space[param] = tune.choice(space)
            elif isinstance(space, dict):
                if "start" in space and "end" in space:
                    if isinstance(space["start"], int):
                        tune_space[param] = tune.randint(space["start"], space["end"])
                    else:
                        tune_space[param] = tune.uniform(space["start"], space["end"])

        return tune_space

    def _setup_ray(self):
        """Setup Ray for distributed optimization."""
        try:
            import ray

            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True,
                    local_mode=False,
                    num_cpus=4,
                )
                logger.info("Ray initialized for distributed optimization")
        except Exception as e:
            logger.warning(f"Could not initialize Ray: {e}")
            raise 