"""Strategy Optimizer - Main Orchestrator.

This module orchestrates different optimization methods for trading strategies.
It has been refactored to use modular optimization components.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .base_optimizer import BaseOptimizer, OptimizerConfig
from .bayesian_optimizer import BayesianOptimization
from .genetic_optimizer import GeneticAlgorithm
from .grid_search_optimizer import GridSearch, OptimizationResult
from .pso_optimizer import ParticleSwarmOptimization
from .ray_optimizer import RayTuneOptimization

logger = logging.getLogger(__name__)


class StrategyOptimizer(BaseOptimizer):
    """Main strategy optimizer that orchestrates different optimization methods."""

    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
        data: Optional[pd.DataFrame] = None,
        strategy_type: str = "generic",
    ):
        """Initialize the strategy optimizer.

        BUG FIX: BaseOptimizer.__init__ requires `data` and `strategy_type`
        as required positional arguments with no defaults, but this
        constructor only accepted `config` and called
        `super().__init__(config)` - passing config into the `data` slot.
        StrategyOptimizer() could not be instantiated at all before this
        fix (TypeError: missing 1 required positional argument:
        'strategy_type'). Since this orchestrator's actual optimize()
        method already takes `data` as a per-call parameter (not bound at
        construction time - each call can optimize against different
        data), `data`/`strategy_type` are optional here with sensible
        defaults rather than forcing every caller to supply a real
        dataset just to construct the orchestrator.

        Args:
            config: Optimizer configuration
            data: Optional DataFrame to bind at construction (rarely
                needed - optimize() takes its own `data` argument per call)
            strategy_type: Optional strategy type label for BaseOptimizer's
                bookkeeping
        """
        if data is None:
            data = pd.DataFrame()
        super().__init__(data=data, strategy_type=strategy_type, config=config)
        self.optimization_methods = {
            "grid_search": GridSearch(),
            "bayesian": BayesianOptimization(),
            "genetic": GeneticAlgorithm(),
            "pso": ParticleSwarmOptimization(),
            "ray_tune": RayTuneOptimization(),
        }
        self.logger = logging.getLogger(self.__class__.__name__)

        # Early stopping configuration
        # BUG FIX: self.config is a strict Pydantic OptimizerConfig model
        # (set by BaseOptimizer.__init__), not a dict - it has no .get()
        # method at all, and most of these field names
        # (early_stopping_enabled, early_stopping_min_delta,
        # max_evaluations, timeout_seconds) aren't even declared on
        # OptimizerConfig's schema (only early_stopping_patience is).
        # This crashed with AttributeError on the very first
        # instantiation. Using getattr() with the original intended
        # defaults, which works safely whether or not a given field
        # exists on the model, without modifying OptimizerConfig's shared
        # schema (other optimizer classes also depend on it).
        self.early_stopping_config = {
            "enabled": getattr(self.config, "early_stopping_enabled", True),
            "patience": getattr(self.config, "early_stopping_patience", 10),
            "min_delta": getattr(self.config, "early_stopping_min_delta", 0.001),
            "max_evaluations": getattr(self.config, "max_evaluations", 1000),
            "timeout_seconds": getattr(self.config, "timeout_seconds", 3600),  # 1 hour
        }

    def log_results(
        self, results: List[OptimizationResult], **kwargs
    ) -> Dict[str, Any]:
        """Log optimization results with comprehensive analysis.

        BUG FIX: this was declared @abstractmethod on BaseOptimizer but
        never implemented here, which meant StrategyOptimizer - the main
        entry point for this entire optimization cluster - could not be
        instantiated at all (TypeError: Can't instantiate abstract class).
        Verified directly: `StrategyOptimizer()` raised immediately before
        this fix.

        Args:
            results: List of optimization results (as returned by optimize())
            **kwargs: Additional logging parameters

        Returns:
            Dictionary summarizing the results
        """
        if not results:
            return {"count": 0, "message": "No results to log"}

        best = min(results, key=lambda r: r.best_score)
        summary = {
            "count": len(results),
            "best_score": best.best_score,
            "best_params": best.best_params,
            "scores": [r.best_score for r in results],
        }
        self.logger.info(
            "Optimization results: %d run(s), best_score=%.6f, best_params=%s",
            summary["count"],
            summary["best_score"],
            summary["best_params"],
        )
        return summary

    def plot_results(self, **kwargs):
        """Plot optimization results.

        BUG FIX: same missing-abstract-method issue as log_results above.
        StrategyOptimizer orchestrates several underlying methods (grid
        search, Bayesian, genetic, PSO, Ray Tune) that each produce their
        own OptimizationResult; there's no single natural plot for the
        orchestrator itself without a plotting library dependency this
        module doesn't otherwise require. Logging a clear message rather
        than silently doing nothing or raising, consistent with how the
        rest of this codebase degrades gracefully when an optional
        visualization isn't available (see e.g. BacktestVisualizer).
        """
        self.logger.info(
            "plot_results() is not implemented for StrategyOptimizer; "
            "inspect the OptimizationResult objects returned by optimize() "
            "directly (best_params, all_scores, convergence_history)."
        )

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
        kwargs.update(
            {
                "early_stopping": early_stopping,
                "max_evaluations": max_evaluations,
                "timeout_seconds": early_stopping["timeout_seconds"],
            }
        )

        return optimizer.optimize(wrapped_objective, param_space, data, **kwargs)

    def _create_early_stopping_objective(
        self, objective: Callable, early_stopping: Dict[str, Any], max_evaluations: int
    ) -> Callable:
        """Create an objective function with early stopping capabilities."""

        class EarlyStoppingObjective:
            def __init__(self, original_objective, config, max_evals):
                self.original_objective = original_objective
                self.config = config
                self.max_evaluations = max_evals
                self.evaluation_count = 0
                self.best_score = float("inf")
                self.patience_counter = 0
                self.start_time = time.time()
                self.scores_history = []
                # BUG FIX: __call__ below references self.logger (for
                # max-evaluations/timeout/patience messages), but it was
                # never set here - every one of those log calls raised
                # AttributeError, silently caught somewhere upstream in
                # the genetic/pso evaluation loops and printed as
                # "Error evaluating individual: ...", flooding output
                # without ever actually reporting early-stopping status.
                self.logger = logging.getLogger(self.__class__.__name__)

            # BUG FIX: this class previously returned float("inf") from all
            # three early-stopping/limit paths below. That's fine for
            # grid_search/genetic/pso, which just treat it as "very bad"
            # and move on - but it crashes Bayesian optimization outright,
            # since gp_minimize fits a Gaussian Process regression to the
            # observed scores, and GP regression cannot fit non-finite
            # target values at all. Verified concretely: bayesian
            # optimization raised "ValueError: Input y contains infinity"
            # the moment patience-based early stopping triggered (after
            # just a few evaluations without improvement, which happens
            # quickly on a simple objective). A large finite sentinel
            # signals "very bad" to every method without breaking any of
            # them.
            _EARLY_STOP_PENALTY = 1e10

            def __call__(self, *args, **kwargs):
                # Check evaluation limit
                if self.evaluation_count >= self.max_evaluations:
                    self.logger.info(
                        f"Reached maximum evaluations: {self.max_evaluations}"
                    )
                    return self._EARLY_STOP_PENALTY

                # Check timeout
                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.config["timeout_seconds"]:
                    self.logger.info(
                        f"Optimization timeout after {elapsed_time:.1f} seconds"
                    )
                    return self._EARLY_STOP_PENALTY

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
                if (
                    self.config["enabled"]
                    and self.patience_counter >= self.config["patience"]
                    and self.evaluation_count >= self.config["patience"]
                ):
                    self.logger.info(
                        f"Early stopping triggered after {self.evaluation_count} evaluations"
                    )
                    return self._EARLY_STOP_PENALTY

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
            objective,
            param_space,
            data,
            method,
            early_stopping=early_stopping_config,
            max_evaluations=max_evaluations,
            **kwargs,
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
                        objective,
                        param_space,
                        data,
                        method,
                        early_stopping=early_stopping,
                        max_evaluations=max_evaluations,
                        **kwargs,
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
            comparison_data.append(
                {
                    "method": method,
                    "best_score": result.best_score,
                    "optimization_time": result.optimization_time,
                    "n_iterations": result.n_iterations,
                    "convergence_rate": self._calculate_convergence_rate(result),
                    "early_stopping_triggered": self._check_early_stopping(result),
                }
            )

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
        if hasattr(result, "metadata") and result.metadata:
            return result.metadata.get("early_stopping_triggered", False)
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
            "supports_early_stopping": hasattr(optimizer, "supports_early_stopping"),
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
            "total_optimizations": getattr(self, "total_optimizations", 0),
        }
