"""
Model Improver Agent for dynamic hyperparameter tuning and model optimization.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from trading.memory.agent_memory import AgentMemory
from trading.memory.model_monitor import ModelMonitor
from trading.models.model_registry import ModelRegistry
from trading.utils.performance_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)

from .base_agent_interface import AgentResult, BaseAgent

# Simple optimizer stubs to avoid import issues


class BayesianOptimizer:
    """Simple Bayesian optimizer stub."""

    def __init__(self, *args, **kwargs):
        pass

    def optimize(self, *args, **kwargs):
        return None


class GeneticOptimizer:
    """Simple genetic optimizer stub."""

    def __init__(self, *args, **kwargs):
        pass

    def optimize(self, *args, **kwargs):
        return None


logger = logging.getLogger(__name__)


@dataclass
class ModelImprovementRequest:
    """Request for model improvement."""

    model_name: str
    improvement_type: str  # 'hyperparameter', 'architecture', 'feature_engineering'
    performance_thresholds: Optional[Dict[str, float]] = None
    optimization_method: str = "bayesian"
    max_iterations: int = 50
    timeout: int = 3600
    priority: str = "normal"  # 'low', 'normal', 'high', 'urgent'
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelImprovementResult:
    """Result of model improvement process."""

    success: bool
    model_name: str
    improvement_type: str
    old_performance: Optional[Dict[str, float]] = None
    new_performance: Optional[Dict[str, float]] = None
    improvement_metrics: Optional[Dict[str, float]] = None
    changes_made: Optional[Dict[str, Any]] = None
    optimization_history: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ModelImproverAgent(BaseAgent):
    """
    Agent responsible for reviewing backtest performance and automatically updating model configs.

    This agent performs dynamic hyperparameter tuning based on recent performance metrics
    and market conditions to continuously improve model performance.
    """

    def __init__(
        self, name: str = "model_improver", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Model Improver Agent.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)

        # Initialize components
        self.model_registry = ModelRegistry()
        self.model_monitor = ModelMonitor()
        self.memory = AgentMemory()

        # Performance tracking
        self.improvement_history: List[Dict[str, Any]] = []
        self.last_improvement: Dict[str, datetime] = {}

        # Configuration
        self.improvement_interval = config.get(
            "improvement_interval", 86400
        )  # 24 hours
        self.performance_thresholds = config.get(
            "performance_thresholds",
            {
                "min_sharpe": 0.8,
                "max_drawdown": 0.25,
                "min_accuracy": 0.55,
                "max_mse": 0.1,
            },
        )

        # Optimization settings
        self.optimization_method = config.get("optimization_method", "bayesian")
        self.max_optimization_iterations = config.get("max_optimization_iterations", 50)
        self.optimization_timeout = config.get("optimization_timeout", 3600)  # 1 hour

        # Initialize optimizers
        self.bayesian_optimizer = BayesianOptimizer()
        self.genetic_optimizer = GeneticOptimizer()

        logger.info(
            f"Initialized ModelImproverAgent with {self.optimization_method} optimization"
        )

    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the model improvement logic.

        Args:
            **kwargs: action, model_name, performance_data, etc.

        Returns:
            AgentResult: Result of the improvement process
        """
        try:
            action = kwargs.get("action", "improve_models")

            if action == "improve_models":
                return await self._improve_all_models()
            elif action == "improve_specific_model":
                model_name = kwargs.get("model_name")
                if not model_name:
                    return AgentResult(
                        success=False, error_message="Missing model_name"
                    )
                return await self._improve_specific_model(model_name)
            elif action == "get_improvement_history":
                return AgentResult(
                    success=True,
                    data={"improvement_history": self.improvement_history[-10:]},
                )
            elif action == "force_improvement":
                return await self._force_improvement()
            else:
                return AgentResult(
                    success=False, error_message=f"Unknown action: {action}"
                )

        except Exception as e:
            return self.handle_error(e)

    async def _improve_all_models(self) -> AgentResult:
        """Improve all registered models based on performance."""
        try:
            models = self.model_registry.list_models()
            improvements = []

            for model_name in models:
                try:
                    # Check if improvement is needed
                    if self._should_improve_model(model_name):
                        improvement = await self._improve_specific_model(model_name)
                        if improvement.success:
                            improvements.append(improvement.data)
                        else:
                            logger.warning(
                                f"Failed to improve model {model_name}: {improvement.error_message}"
                            )
                except Exception as e:
                    logger.error(f"Error improving model {model_name}: {str(e)}")

            return AgentResult(
                success=True,
                data={
                    "improvements_made": len(improvements),
                    "improvements": improvements,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error in model improvement cycle: {str(e)}")
            return AgentResult(success=False, error_message=str(e))

    async def _improve_specific_model(self, model_name: str) -> AgentResult:
        """Improve a specific model through hyperparameter optimization."""
        try:
            # Get current model
            model = self.model_registry.get_model(model_name)
            if not model:
                return AgentResult(
                    success=False, error_message=f"Model {model_name} not found"
                )

            # Get recent performance
            performance = self._get_model_performance(model_name)
            if not performance:
                return AgentResult(
                    success=False, error_message=f"No performance data for {model_name}"
                )

            # Check if improvement is needed
            if not self._needs_improvement(performance):
                return AgentResult(
                    success=True,
                    data={
                        "message": f"Model {model_name} performing well, no improvement needed"
                    },
                )

            # Get current hyperparameters
            current_params = model.get_hyperparameters()

            # Define optimization objective
            def objective(hyperparams: Dict[str, Any]) -> float:
                """Optimization objective function."""
                try:
                    # Update model with new hyperparameters
                    model.update_hyperparameters(hyperparams)

                    # Estimate performance improvement
                    estimated_improvement = self._estimate_performance_improvement(
                        model_name, hyperparams, performance
                    )

                    # Return negative score (minimize)
                    return -estimated_improvement

                except Exception as e:
                    logger.error(f"Error in optimization objective: {str(e)}")
                    return 0.0

            # Define hyperparameter space
            param_space = self._get_hyperparameter_space(model_name)

            # Run optimization
            if self.optimization_method == "bayesian":
                best_params = await self._run_bayesian_optimization(
                    objective, param_space, current_params
                )
            else:
                best_params = await self._run_genetic_optimization(
                    objective, param_space, current_params
                )

            # Apply improvements
            if best_params:
                model.update_hyperparameters(best_params)

                # Log improvement
                improvement_record = {
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "old_params": current_params,
                    "new_params": best_params,
                    "performance_before": performance,
                    "estimated_improvement": self._estimate_performance_improvement(
                        model_name, best_params, performance
                    ),
                }

                self.improvement_history.append(improvement_record)
                self.last_improvement[model_name] = datetime.now()

                # Store in memory
                self.memory.log_outcome(
                    agent=self.name,
                    run_type="model_improvement",
                    outcome=improvement_record,
                )

                logger.info(f"Improved model {model_name} with new hyperparameters")

                return AgentResult(
                    success=True,
                    data={
                        "model_name": model_name,
                        "improvement": improvement_record,
                        "message": f"Successfully improved {model_name}",
                    },
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=f"Failed to find better hyperparameters for {model_name}",
                )

        except Exception as e:
            logger.error(f"Error improving model {model_name}: {str(e)}")
            return AgentResult(success=False, error_message=str(e))

    def _should_improve_model(self, model_name: str) -> bool:
        """Check if a model should be improved."""
        try:
            # Check if enough time has passed since last improvement
            if model_name in self.last_improvement:
                time_since_improvement = (
                    datetime.now() - self.last_improvement[model_name]
                )
                if time_since_improvement.total_seconds() < self.improvement_interval:
                    return False

            # Check performance
            performance = self._get_model_performance(model_name)
            if not performance:
                return True  # No performance data, needs improvement

            return self._needs_improvement(performance)

        except Exception as e:
            logger.error(f"Error checking if model should be improved: {str(e)}")
            return False

    def _get_model_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get recent performance metrics for a model."""
        try:
            # Get performance from model monitor
            trust_level = self.model_monitor.get_model_trust(model_name)

            # Get recent predictions and outcomes
            recent_data = self.memory.get_recent_outcomes(
                agent=model_name, run_type="prediction", limit=20
            )

            if not recent_data:
                return None

            # Calculate metrics
            predictions = [entry.get("prediction", 0) for entry in recent_data]
            actuals = [entry.get("actual", 0) for entry in recent_data]

            if len(predictions) < 5:
                return None

            # Calculate performance metrics
            mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
            accuracy = np.mean(
                np.sign(np.array(predictions)) == np.sign(np.array(actuals))
            )

            # Calculate returns-based metrics
            returns = np.diff(actuals)
            if len(returns) > 0:
                sharpe = calculate_sharpe_ratio(returns)
                max_dd = calculate_max_drawdown(returns)
            else:
                sharpe = 0.0
                max_dd = 0.0

            return {
                "mse": mse,
                "accuracy": accuracy,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "trust_level": trust_level,
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return None

    def _needs_improvement(self, performance: Dict[str, Any]) -> bool:
        """Check if performance indicates need for improvement."""
        try:
            if not performance:
                return True

            # Check against thresholds
            if (
                performance.get("sharpe_ratio", 0)
                < self.performance_thresholds["min_sharpe"]
            ):
                return True

            if (
                performance.get("max_drawdown", 1)
                > self.performance_thresholds["max_drawdown"]
            ):
                return True

            if (
                performance.get("accuracy", 0)
                < self.performance_thresholds["min_accuracy"]
            ):
                return True

            if performance.get("mse", 1) > self.performance_thresholds["max_mse"]:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking improvement need: {str(e)}")
            return True

    def _get_hyperparameter_space(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter search space for a model."""
        try:
            # Define search spaces for different model types
            if "lstm" in model_name.lower():
                return {
                    "hidden_size": (32, 256),
                    "num_layers": (1, 4),
                    "dropout": (0.1, 0.5),
                    "learning_rate": (0.001, 0.01),
                }
            elif "xgboost" in model_name.lower():
                return {
                    "n_estimators": (50, 300),
                    "max_depth": (3, 10),
                    "learning_rate": (0.01, 0.3),
                    "subsample": (0.6, 1.0),
                }
            elif "transformer" in model_name.lower():
                return {
                    "d_model": (64, 512),
                    "nhead": (4, 16),
                    "num_layers": (2, 8),
                    "dropout": (0.1, 0.5),
                }
            else:
                # Default space
                return {"learning_rate": (0.001, 0.1), "regularization": (0.01, 0.1)}

        except Exception as e:
            logger.error(f"Error getting hyperparameter space: {str(e)}")
            return {}

    def _estimate_performance_improvement(
        self,
        model_name: str,
        hyperparams: Dict[str, Any],
        current_performance: Dict[str, Any],
    ) -> float:
        """Estimate performance improvement from hyperparameter changes."""
        try:
            # This is a simplified estimation - in practice, you'd use more sophisticated methods
            # like meta-learning or surrogate models

            # Base improvement score
            improvement_score = 0.0

            # Check if hyperparameters are reasonable
            if "learning_rate" in hyperparams:
                lr = hyperparams["learning_rate"]
                if 0.001 <= lr <= 0.01:
                    improvement_score += 0.2
                elif 0.01 < lr <= 0.1:
                    improvement_score += 0.1

            if "dropout" in hyperparams:
                dropout = hyperparams["dropout"]
                if 0.1 <= dropout <= 0.3:
                    improvement_score += 0.15
                elif 0.3 < dropout <= 0.5:
                    improvement_score += 0.1

            # Consider current performance
            if current_performance.get("sharpe_ratio", 0) < 0.5:
                improvement_score += 0.3  # High potential for improvement

            if current_performance.get("accuracy", 0) < 0.5:
                improvement_score += 0.2

            return min(1.0, improvement_score)

        except Exception as e:
            logger.error(f"Error estimating performance improvement: {str(e)}")
            return 0.0

    async def _run_bayesian_optimization(
        self,
        objective: callable,
        param_space: Dict[str, Any],
        current_params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run Bayesian optimization for hyperparameter tuning."""
        try:
            # Initialize with current parameters
            initial_params = [current_params]

            # Run optimization
            best_params = self.bayesian_optimizer.optimize(
                objective=objective,
                param_space=param_space,
                initial_points=initial_params,
                n_iterations=self.max_optimization_iterations,
                timeout=self.optimization_timeout,
            )

            return best_params

        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}")
            return None

    async def _run_genetic_optimization(
        self,
        objective: callable,
        param_space: Dict[str, Any],
        current_params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run genetic optimization for hyperparameter tuning."""
        try:
            # Initialize with current parameters
            initial_population = [current_params]

            # Run optimization
            best_params = self.genetic_optimizer.optimize(
                objective=objective,
                param_space=param_space,
                initial_population=initial_population,
                generations=10,
                population_size=20,
            )

            return best_params

        except Exception as e:
            logger.error(f"Error in genetic optimization: {str(e)}")
            return None

    async def _force_improvement(self) -> AgentResult:
        """Force improvement cycle for all models."""
        try:
            # Reset last improvement times
            self.last_improvement = {}

            # Run improvement
            return await self._improve_all_models()

        except Exception as e:
            logger.error(f"Error in forced improvement: {str(e)}")
            return AgentResult(success=False, error_message=str(e))

    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of recent improvements."""
        try:
            recent_improvements = self.improvement_history[-10:]

            return {
                "total_improvements": len(self.improvement_history),
                "recent_improvements": len(recent_improvements),
                "models_improved": list(
                    set(imp["model_name"] for imp in recent_improvements)
                ),
                "last_improvement": recent_improvements[-1]["timestamp"]
                if recent_improvements
                else None,
                "average_improvement_score": np.mean(
                    [imp.get("estimated_improvement", 0) for imp in recent_improvements]
                )
                if recent_improvements
                else 0.0,
            }

        except Exception as e:
            logger.error(f"Error getting improvement summary: {str(e)}")
            return {}

    def update_performance_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update performance thresholds."""
        self.performance_thresholds.update(new_thresholds)
        logger.info(f"Updated performance thresholds: {new_thresholds}")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        base_status = super().get_status()
        base_status.update(
            {
                "improvement_interval": self.improvement_interval,
                "optimization_method": self.optimization_method,
                "performance_thresholds": self.performance_thresholds,
                "improvement_summary": self.get_improvement_summary(),
                "models_tracked": len(self.model_registry.list_models()),
            }
        )
        return base_status
