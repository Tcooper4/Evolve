"""Base optimizer class with common functionality.

Enhanced with proper abstract method definitions, parameter schema validation,
and comprehensive logging capabilities.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, validator

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
try:
    os.makedirs("trading/optimization/logs", exist_ok=True)
    debug_handler = logging.FileHandler(
        "trading/optimization/logs/optimization_debug.log"
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)
except Exception as e:
    logger.warning(f"Could not setup debug logging: {e}")


class ParameterSchema(BaseModel):
    """Schema for parameter validation."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (int, float, str, bool)")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value")
    default: Optional[Any] = Field(None, description="Default value")
    choices: Optional[List[Any]] = Field(
        None, description="Valid choices for categorical parameters"
    )
    description: Optional[str] = Field(None, description="Parameter description")

    @validator("type")
    def validate_type(cls, v):
        """Validate parameter type."""
        valid_types = ["int", "float", "str", "bool"]
        if v not in valid_types:
            raise ValueError(f"Parameter type must be one of {valid_types}")
        return v

    @validator("min_value", "max_value")
    def validate_range(cls, v, values):
        """Validate min/max values."""
        if v is not None and "type" in values:
            if values["type"] in ["int", "float"] and not isinstance(v, (int, float)):
                raise ValueError(
                    f"min_value/max_value must be numeric for {values['type']} parameters"
                )
        return v


class OptimizerConfig(BaseModel):
    """Configuration for optimizer."""

    # General settings
    name: str = Field(..., description="Name of the optimizer")
    max_iterations: int = Field(
        100, ge=1, description="Maximum number of optimization iterations"
    )
    early_stopping_patience: int = Field(
        5, ge=1, description="Number of iterations to wait before early stopping"
    )
    learning_rate: float = Field(0.01, gt=0, description="Initial learning rate")
    batch_size: int = Field(32, ge=1, description="Batch size for optimization")

    # Multi-objective settings
    is_multi_objective: bool = Field(
        False, description="Whether to use multi-objective optimization"
    )
    objectives: List[str] = Field(
        ["sharpe_ratio", "win_rate"], description="List of objectives to optimize"
    )
    objective_weights: Dict[str, float] = Field(
        {"sharpe_ratio": 0.6, "win_rate": 0.4}, description="Weights for each objective"
    )

    # Learning rate scheduler settings
    use_lr_scheduler: bool = Field(
        True, description="Whether to use learning rate scheduling"
    )
    scheduler_type: str = Field("cosine", description="Type of learning rate scheduler")
    min_lr: float = Field(0.0001, gt=0, description="Minimum learning rate")
    warmup_steps: int = Field(0, ge=0, description="Number of warmup steps")

    # Checkpoint settings
    save_checkpoints: bool = Field(True, description="Whether to save checkpoints")
    checkpoint_dir: str = Field(
        "checkpoints", description="Directory to save checkpoints"
    )
    checkpoint_frequency: int = Field(
        5, ge=1, description="Save checkpoint every N iterations"
    )

    # Validation settings
    validation_split: float = Field(
        0.2, gt=0, lt=1, description="Validation split ratio"
    )
    cross_validation_folds: int = Field(
        3, ge=2, description="Number of cross-validation folds"
    )

    # Logging settings
    log_level: str = Field("INFO", description="Logging level")
    log_to_file: bool = Field(True, description="Whether to log to file")
    log_file: str = Field("optimization.log", description="Log file path")

    @validator("scheduler_type", allow_reuse=True)
    def validate_scheduler_type(cls, v):
        """Validate scheduler type."""
        valid_types = ["cosine", "step", "performance"]
        if v not in valid_types:
            raise ValueError(f"scheduler_type must be one of {valid_types}")
        return v

    @validator("objectives", allow_reuse=True)
    def validate_objectives(cls, v):
        """Validate objectives."""
        valid_objectives = [
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
            "mse",
            "alpha",
            "calmar_ratio",
            "sortino_ratio",
        ]
        for obj in v:
            if obj not in valid_objectives:
                raise ValueError(
                    f"Invalid objective: {obj}. Must be one of {valid_objectives}"
                )
        return v

    @validator("log_level", allow_reuse=True)
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    parameters: Dict[str, float]
    metrics: Dict[str, float]
    returns: pd.Series
    signals: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    timestamp: datetime
    optimization_type: str
    trial_id: Optional[int] = None
    validation_passed: bool = True
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class BaseOptimizer(ABC):
    """Base class for all optimizers with enhanced validation and logging."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_type: str,
        config: Optional[OptimizerConfig] = None,
        parameter_schema: Optional[List[ParameterSchema]] = None,
        verbose: bool = False,
        n_jobs: int = -1,
    ):
        """Initialize base optimizer.

        Args:
            data: DataFrame with OHLCV data
            strategy_type: Type of strategy to optimize
            config: Optimizer configuration
            parameter_schema: Schema for parameter validation
            verbose: Enable verbose logging
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.data = data
        self.strategy_type = strategy_type
        self.config = config or OptimizerConfig(name=self.__class__.__name__)
        self.parameter_schema = parameter_schema or []
        self.verbose = verbose
        self.n_jobs = n_jobs

        # Initialize state
        self.results = []
        self.best_result = None
        self.current_iteration = 0
        self.best_metric = float("-inf")
        self.best_params = None
        self.metrics_history = []
        self.early_stopping_counter = 0
        self.optimization_start_time = None
        self.optimization_end_time = None

        # Setup logging
        self._setup_logging()

        # Setup checkpointing
        try:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create checkpoint_dir: {e}")

        # Validate parameter schema
        self._validate_parameter_schema()

        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level)
        logger.setLevel(log_level)

        if self.config.log_to_file:
            try:
                log_path = Path(self.config.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path)
                file_handler.setLevel(log_level)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")

    def _validate_parameter_schema(self):
        """Validate parameter schema."""
        if not self.parameter_schema:
            return

        for param in self.parameter_schema:
            try:
                # Validate parameter schema
                ParameterSchema(**param.dict())
            except ValidationError as e:
                logger.error(f"Invalid parameter schema for {param.name}: {e}")
                raise

    @abstractmethod
    def optimize(
        self,
        param_space: Dict[str, Union[List, Tuple]],
        objective: Union[str, List[str]],
        n_trials: int = 100,
        **kwargs,
    ) -> List[OptimizationResult]:
        """Run optimization.

        Args:
            param_space: Parameter space to search
            objective: Optimization objective(s)
            n_trials: Number of trials to run
            **kwargs: Additional optimizer-specific arguments

        Returns:
            List of optimization results
        """

    @abstractmethod
    def log_results(
        self, results: List[OptimizationResult], **kwargs
    ) -> Dict[str, Any]:
        """Log optimization results with comprehensive analysis.

        Args:
            results: List of optimization results
            **kwargs: Additional logging parameters

        Returns:
            Dictionary containing logged information
        """

    def validate_parameters(
        self, parameters: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate parameters against schema.

        Args:
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.parameter_schema:
            return True, None

        try:
            for param_schema in self.parameter_schema:
                param_name = param_schema.name
                param_value = parameters.get(param_name)

                if param_value is None:
                    if param_schema.default is not None:
                        continue
                    else:
                        return False, f"Required parameter '{param_name}' is missing"

                # Type validation
                if param_schema.type == "int" and not isinstance(param_value, int):
                    return False, f"Parameter '{param_name}' must be an integer"
                elif param_schema.type == "float" and not isinstance(
                    param_value, (int, float)
                ):
                    return False, f"Parameter '{param_name}' must be a number"
                elif param_schema.type == "str" and not isinstance(param_value, str):
                    return False, f"Parameter '{param_name}' must be a string"
                elif param_schema.type == "bool" and not isinstance(param_value, bool):
                    return False, f"Parameter '{param_name}' must be a boolean"

                # Range validation
                if (
                    param_schema.min_value is not None
                    and param_value < param_schema.min_value
                ):
                    return (
                        False,
                        f"Parameter '{param_name}' must be >= {param_schema.min_value}",
                    )
                if (
                    param_schema.max_value is not None
                    and param_value > param_schema.max_value
                ):
                    return (
                        False,
                        f"Parameter '{param_name}' must be <= {param_schema.max_value}",
                    )

                # Choices validation
                if (
                    param_schema.choices is not None
                    and param_value not in param_schema.choices
                ):
                    return (
                        False,
                        f"Parameter '{param_name}' must be one of {param_schema.choices}",
                    )

            return True, None

        except Exception as e:
            return False, f"Parameter validation error: {str(e)}"

    def calculate_metrics(
        self, returns: pd.Series, signals: pd.Series, equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics.

        Args:
            returns: Strategy returns
            signals: Trading signals
            equity_curve: Equity curve

        Returns:
            Dictionary of metrics
        """
        try:
            # Basic metrics
            total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0
            annual_return = (
                (1 + total_return) ** (252 / len(returns)) - 1
                if len(returns) > 0
                else 0
            )
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0

            # Win rate
            winning_trades = returns[returns > 0]
            total_trades = returns[returns != 0]
            win_rate = (
                len(winning_trades) / len(total_trades) if len(total_trades) > 0 else 0
            )

            # Drawdown
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

            # Additional metrics
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            sortino_ratio = (
                annual_return / (returns[returns < 0].std() * np.sqrt(252))
                if len(returns[returns < 0]) > 0
                else 0
            )

            # Alpha and beta (if market data available)
            alpha = 0.0
            beta = 1.0
            if "market_returns" in self.data.columns:
                market_returns = self.data["market_returns"].dropna()
                if len(market_returns) > 0 and len(returns) > 0:
                    # Align returns
                    aligned_returns = returns.reindex(market_returns.index).dropna()
                    aligned_market = market_returns.reindex(
                        aligned_returns.index
                    ).dropna()

                    if len(aligned_returns) > 10:
                        # Calculate beta
                        covariance = np.cov(aligned_returns, aligned_market)[0, 1]
                        market_variance = np.var(aligned_market)
                        beta = (
                            covariance / market_variance
                            if market_variance != 0
                            else 1.0
                        )

                        # Calculate alpha
                        alpha = aligned_returns.mean() - beta * aligned_market.mean()

            return {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "calmar_ratio": calmar_ratio,
                "sortino_ratio": sortino_ratio,
                "alpha": alpha,
                "beta": beta,
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "sortino_ratio": 0.0,
                "alpha": 0.0,
                "beta": 1.0,
            }

    def log_result(self, result: OptimizationResult, trial_id: Optional[int] = None):
        """Log optimization result.

        Args:
            result: Optimization result
            trial_id: Trial ID (if applicable)
        """
        try:
            if self.verbose:
                logger.info(
                    f"Trial {trial_id if trial_id is not None else 'N/A'}: "
                    f"Parameters: {result.parameters}, "
                    f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.3f}, "
                    f"Win Rate: {result.metrics.get('win_rate', 0):.3f}, "
                    f"Valid: {result.validation_passed}"
                )

            self.results.append(result)

            # Update best result
            if self.best_result is None or (
                result.metrics.get("sharpe_ratio", 0)
                > self.best_result.metrics.get("sharpe_ratio", 0)
            ):
                self.best_result = result
                logger.info(
                    f"New best result found: Sharpe={result.metrics.get('sharpe_ratio', 0):.3f}"
                )

        except Exception as e:
            logger.error(f"Error logging result: {e}")

    def get_best_result(self) -> Optional[OptimizationResult]:
        """Get the best optimization result.

        Returns:
            Best optimization result or None
        """
        return self.best_result

    def get_all_results(self) -> List[OptimizationResult]:
        """Get all optimization results.

        Returns:
            List of all optimization results
        """
        return self.results

    def get_top_results(
        self, n: int = 10, metric: str = "sharpe_ratio"
    ) -> List[OptimizationResult]:
        """Get top N results by specified metric.

        Args:
            n: Number of top results to return
            metric: Metric to sort by

        Returns:
            List of top N results
        """
        if not self.results:
            return []

        # Sort by metric
        sorted_results = sorted(
            self.results, key=lambda x: x.metrics.get(metric, 0), reverse=True
        )

        return sorted_results[:n]

    def export_results(self, filepath: str, format: str = "json") -> Dict[str, Any]:
        """Export optimization results.

        Args:
            filepath: Path to save results
            format: Export format ('json', 'csv', 'pickle')

        Returns:
            Export status
        """
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                # Convert results to JSON-serializable format
                export_data = {
                    "metadata": {
                        "optimizer_type": self.__class__.__name__,
                        "strategy_type": self.strategy_type,
                        "timestamp": datetime.now().isoformat(),
                        "total_results": len(self.results),
                        "best_sharpe": (
                            self.best_result.metrics.get("sharpe_ratio", 0)
                            if self.best_result
                            else 0
                        ),
                    },
                    "results": [],
                }

                for result in self.results:
                    result_dict = {
                        "parameters": result.parameters,
                        "metrics": result.metrics,
                        "timestamp": result.timestamp.isoformat(),
                        "optimization_type": result.optimization_type,
                        "trial_id": result.trial_id,
                        "validation_passed": result.validation_passed,
                        "error_message": result.error_message,
                        "execution_time": result.execution_time,
                    }
                    export_data["results"].append(result_dict)

                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2)

            elif format.lower() == "csv":
                # Export as CSV
                results_df = pd.DataFrame(
                    [
                        {
                            "trial_id": r.trial_id,
                            "sharpe_ratio": r.metrics.get("sharpe_ratio", 0),
                            "win_rate": r.metrics.get("win_rate", 0),
                            "total_return": r.metrics.get("total_return", 0),
                            "max_drawdown": r.metrics.get("max_drawdown", 0),
                            "validation_passed": r.validation_passed,
                            **r.parameters,
                        }
                        for r in self.results
                    ]
                )
                results_df.to_csv(output_path, index=False)

            elif format.lower() == "pickle":
                # Export as pickle
                import pickle

                with open(output_path, "wb") as f:
                    pickle.dump(self.results, f)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Results exported to {output_path}")
            return {"success": True, "filepath": str(output_path)}

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return {"success": False, "error": str(e)}

    @abstractmethod
    def plot_results(self, **kwargs):
        """Plot optimization results.

        Args:
            **kwargs: Plotting parameters
        """

    def log_metrics(
        self, metrics: Dict[str, float], iteration: Optional[int] = None
    ) -> None:
        """Log metrics for current iteration.

        Args:
            metrics: Metrics to log
            iteration: Current iteration number
        """
        try:
            iteration = iteration or self.current_iteration

            log_entry = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
            }

            self.metrics_history.append(log_entry)

            if self.verbose:
                logger.info(
                    f"Iteration {iteration}: "
                    f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                    f"Win Rate={metrics.get('win_rate', 0):.3f}, "
                    f"Total Return={metrics.get('total_return', 0):.3f}"
                )

            # Check for early stopping
            if self.should_stop_early(metrics.get("sharpe_ratio", 0)):
                logger.info(f"Early stopping triggered at iteration {iteration}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
            return False

    def save_checkpoint(
        self, params: Dict[str, Any], metrics: Dict[str, float]
    ) -> None:
        """Save optimization checkpoint.

        Args:
            params: Current parameters
            metrics: Current metrics
        """
        try:
            if not self.config.save_checkpoints:
                return

            checkpoint_data = {
                "iteration": self.current_iteration,
                "parameters": params,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "best_metric": self.best_metric,
            }

            checkpoint_path = (
                Path(self.config.checkpoint_dir)
                / f"checkpoint_{self.current_iteration}.json"
            )

            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.debug(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Load optimization checkpoint.

        Args:
            iteration: Iteration number to load

        Returns:
            Checkpoint data or None
        """
        try:
            checkpoint_path = (
                Path(self.config.checkpoint_dir) / f"checkpoint_{iteration}.json"
            )

            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return None

            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def should_stop_early(self, current_metric: float) -> bool:
        """Check if optimization should stop early.

        Args:
            current_metric: Current optimization metric

        Returns:
            True if should stop early
        """
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.config.early_stopping_patience

    def get_learning_rate(self) -> float:
        """Get current learning rate.

        Returns:
            Current learning rate
        """
        if not self.config.use_lr_scheduler:
            return self.config.learning_rate

        # Simple cosine annealing
        if self.config.scheduler_type == "cosine":
            progress = min(self.current_iteration / self.config.max_iterations, 1.0)
            lr = self.config.min_lr + 0.5 * (
                self.config.learning_rate - self.config.min_lr
            ) * (1 + np.cos(np.pi * progress))
            return lr

        # Step decay
        elif self.config.scheduler_type == "step":
            decay_factor = 0.1
            decay_steps = self.config.max_iterations // 3
            lr = self.config.learning_rate * (
                decay_factor ** (self.current_iteration // decay_steps)
            )
            return max(lr, self.config.min_lr)

        # Performance-based decay
        elif self.config.scheduler_type == "performance":
            if len(self.metrics_history) > 10:
                recent_metrics = [
                    m["metrics"].get("sharpe_ratio", 0)
                    for m in self.metrics_history[-10:]
                ]
                if np.std(recent_metrics) < 0.01:  # Low improvement
                    return self.config.learning_rate * 0.5
            return self.config.learning_rate

        return self.config.learning_rate

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary.

        Returns:
            Dictionary with optimization summary
        """
        if not self.results:
            return {"error": "No results available"}

        # Calculate summary statistics
        sharpe_ratios = [r.metrics.get("sharpe_ratio", 0) for r in self.results]
        win_rates = [r.metrics.get("win_rate", 0) for r in self.results]
        total_returns = [r.metrics.get("total_return", 0) for r in self.results]

        summary = {
            "total_trials": len(self.results),
            "successful_trials": sum(1 for r in self.results if r.validation_passed),
            "best_sharpe": max(sharpe_ratios),
            "avg_sharpe": np.mean(sharpe_ratios),
            "std_sharpe": np.std(sharpe_ratios),
            "best_win_rate": max(win_rates),
            "avg_win_rate": np.mean(win_rates),
            "best_total_return": max(total_returns),
            "avg_total_return": np.mean(total_returns),
            "optimization_duration": None,
        }

        # Calculate duration if available
        if self.optimization_start_time and self.optimization_end_time:
            duration = self.optimization_end_time - self.optimization_start_time
            summary["optimization_duration"] = str(duration)

        return summary
