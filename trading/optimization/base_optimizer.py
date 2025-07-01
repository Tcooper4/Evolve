"""Base optimizer class with common functionality."""

import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/optimization/logs/optimization_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

class OptimizerConfig(BaseModel):
    """Configuration for optimizer."""
    
    # General settings
    name: str = Field(..., description="Name of the optimizer")
    max_iterations: int = Field(100, ge=1, description="Maximum number of optimization iterations")
    early_stopping_patience: int = Field(5, ge=1, description="Number of iterations to wait before early stopping")
    learning_rate: float = Field(0.01, gt=0, description="Initial learning rate")
    batch_size: int = Field(32, ge=1, description="Batch size for optimization")
    
    # Multi-objective settings
    is_multi_objective: bool = Field(False, description="Whether to use multi-objective optimization")
    objectives: List[str] = Field(
        ["sharpe_ratio", "win_rate"],
        description="List of objectives to optimize"
    )
    objective_weights: Dict[str, float] = Field(
        {"sharpe_ratio": 0.6, "win_rate": 0.4},
        description="Weights for each objective"
    )
    
    # Learning rate scheduler settings
    use_lr_scheduler: bool = Field(True, description="Whether to use learning rate scheduling")
    scheduler_type: str = Field("cosine", description="Type of learning rate scheduler")
    min_lr: float = Field(0.0001, gt=0, description="Minimum learning rate")
    warmup_steps: int = Field(0, ge=0, description="Number of warmup steps")
    
    # Checkpoint settings
    save_checkpoints: bool = Field(True, description="Whether to save checkpoints")
    checkpoint_dir: str = Field("checkpoints", description="Directory to save checkpoints")
    checkpoint_frequency: int = Field(5, ge=1, description="Save checkpoint every N iterations")
    
    # Validation settings
    validation_split: float = Field(0.2, gt=0, lt=1, description="Validation split ratio")
    cross_validation_folds: int = Field(3, ge=2, description="Number of cross-validation folds")
    
    @validator('scheduler_type', allow_reuse=True)
    def validate_scheduler_type(cls, v):
        """Validate scheduler type."""
        valid_types = ["cosine", "step", "performance"]
        if v not in valid_types:
            raise ValueError(f"scheduler_type must be one of {valid_types}")
        return v
    
    @validator('objectives', allow_reuse=True)
    def validate_objectives(cls, v):
        """Validate objectives."""
        valid_objectives = ["sharpe_ratio", "win_rate", "max_drawdown", "mse", "alpha"]
        for obj in v:
            if obj not in valid_objectives:
                raise ValueError(f"Invalid objective: {obj}. Must be one of {valid_objectives}")
        return v

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

class BaseOptimizer(ABC):
    """Base class for all optimizers."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_type: str,
        verbose: bool = False,
        n_jobs: int = -1
    ):
        """Initialize base optimizer.
        
        Args:
            data: DataFrame with OHLCV data
            strategy_type: Type of strategy to optimize
            verbose: Enable verbose logging
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.data = data
        self.strategy_type = strategy_type
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.results = []
        self.best_result = None
        
        # Initialize state
        self.current_iteration = 0
        self.best_metric = float('-inf')
        self.best_params = None
        self.metrics_history = []
        self.early_stopping_counter = 0
        
        # Setup checkpointing
        if self.config.save_checkpoints:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def optimize(
        self,
        param_space: Dict[str, Union[List, Tuple]],
        objective: Union[str, List[str]],
        n_trials: int = 100,
        **kwargs
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
        pass
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        signals: pd.Series,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            returns: Strategy returns
            signals: Trading signals
            equity_curve: Equity curve
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        total_return = equity_curve.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Win rate
        winning_trades = returns[returns > 0]
        total_trades = returns[returns != 0]
        win_rate = len(winning_trades) / len(total_trades) if len(total_trades) > 0 else 0
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = annual_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def log_result(
        self,
        result: OptimizationResult,
        trial_id: Optional[int] = None
    ):
        """Log optimization result.
        
        Args:
            result: Optimization result
            trial_id: Trial ID (if applicable)
        """
        if self.verbose:
            logger.info(
                f"Trial {trial_id if trial_id is not None else 'N/A'}: "
                f"Parameters: {result.parameters}, "
                f"Sharpe: {result.metrics['sharpe_ratio']:.2f}, "
                f"Win Rate: {result.metrics['win_rate']:.2f}"
            )
        
        self.results.append(result)
        
        # Update best result
        if self.best_result is None or (
            result.metrics['sharpe_ratio'] > self.best_result.metrics['sharpe_ratio']
        ):
            self.best_result = result
    
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
    
    def export_results(
        self,
        filepath: str,
        format: str = 'json'
    ):
        """Export optimization results.
        
        Args:
            filepath: Path to save results
            format: Export format ('json' or 'csv')
        """
        if not self.results:
            logger.warning("No results to export")
            return
        
        try:
            if format == 'json':
                results_dict = [
                    {
                        'parameters': r.parameters,
                        'metrics': r.metrics,
                        'timestamp': r.timestamp.isoformat(),
                        'optimization_type': r.optimization_type,
                        'trial_id': r.trial_id
                    }
                    for r in self.results
                ]
                
                with open(filepath, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                    
            elif format == 'csv':
                results_df = pd.DataFrame([
                    {
                        **r.parameters,
                        **r.metrics,
                        'timestamp': r.timestamp,
                        'optimization_type': r.optimization_type,
                        'trial_id': r.trial_id
                    }
                    for r in self.results
                ])
                
                results_df.to_csv(filepath, index=False)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported results to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
    
    @abstractmethod
    def plot_results(self, **kwargs):
        """Plot optimization results.
        
        Args:
            **kwargs: Plot-specific arguments
        """
        pass
    
    def log_metrics(self, metrics: Dict[str, float], iteration: Optional[int] = None) -> None:
        """Log optimization metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            iteration: Current iteration number (defaults to self.current_iteration)
        """
        if iteration is None:
            iteration = self.current_iteration
            
        # Add timestamp and iteration
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": iteration,
            **metrics
        }
        
        # Append to metrics history
        self.metrics_history.append(log_entry)
        
        # Save to JSONL file
        log_path = "trading/optimization/logs/optimization_metrics.jsonl"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        logger.debug(f"Logged metrics for iteration {iteration}: {metrics}")
    
    def save_checkpoint(self, params: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Save optimization checkpoint.
        
        Args:
            params: Current parameters
            metrics: Current metrics
        """
        if not self.config.save_checkpoints:
            return
            
        checkpoint = {
            "iteration": self.current_iteration,
            "params": params,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{self.current_iteration}.json"
        )
        
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
            
        logger.info(f"Saved checkpoint at iteration {self.current_iteration}")
    
    def load_checkpoint(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Load optimization checkpoint.
        
        Args:
            iteration: Iteration number to load
            
        Returns:
            Dictionary containing checkpoint data or None if not found
        """
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{iteration}.json"
        )
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found for iteration {iteration}")
            return
            
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            
        logger.info(f"Loaded checkpoint from iteration {iteration}")
        return checkpoint
    
    def should_stop_early(self, current_metric: float) -> bool:
        """Check if optimization should stop early.
        
        Args:
            current_metric: Current optimization metric value
            
        Returns:
            True if optimization should stop, False otherwise
        """
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
            return False
            
        self.early_stopping_counter += 1
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def get_learning_rate(self) -> float:
        """Get current learning rate based on scheduler settings.
        
        Returns:
            Current learning rate
        """
        if not self.config.use_lr_scheduler:
            return self.config.learning_rate
            
        if self.config.scheduler_type == "cosine":
            # Cosine decay
            progress = self.current_iteration / self.config.max_iterations
            return self.config.min_lr + 0.5 * (self.config.learning_rate - self.config.min_lr) * \
                   (1 + np.cos(np.pi * progress))
                   
        elif self.config.scheduler_type == "step":
            # Step decay
            return self.config.learning_rate * (0.1 ** (self.current_iteration // 30))
            
        elif self.config.scheduler_type == "performance":
            # Performance-based decay
            if len(self.metrics_history) < 2:
                return self.config.learning_rate
                
            current_metric = self.metrics_history[-1].get("objective", 0)
            prev_metric = self.metrics_history[-2].get("objective", 0)
            
            if current_metric < prev_metric:
                return {'success': True, 'result': max(self.config.learning_rate * 0.5, self.config.min_lr), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            return self.config.learning_rate
            
        return self.config.learning_rate 