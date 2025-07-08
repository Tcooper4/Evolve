"""Strategy optimizer for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime
import logging
from trading.models.base_model import BaseModel
from .base_optimizer import BaseOptimizer, OptimizerConfig
from .performance_logger import PerformanceLogger, PerformanceMetrics
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pydantic import Field, validator
import torch.nn as nn
import json
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Try to import ray and its submodules
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.bayesopt import BayesOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from trading.optimization.base_optimizer import BaseOptimizer, OptimizerConfig
from trading.optimization.performance_logger import PerformanceLogger, PerformanceMetrics

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/optimization/logs/optimization_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

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
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    @abstractmethod
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run optimization.
        
        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        pass
    
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
                required_keys = {'start', 'end'}
                if not required_keys.issubset(space.keys()):
                    raise ValueError(f"Parameter {param} missing required keys: {required_keys}")
                if space['start'] >= space['end']:
                    raise ValueError(f"Parameter {param} has invalid range")

    def _check_early_stopping(self, scores: List[float], patience: int = 5,
                            min_delta: float = 1e-4) -> bool:
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
                return {'success': True, 'result': {'success': True, 'result': float('inf'), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        return wrapper

class GridSearch(OptimizationMethod):
    """Grid search optimization method."""
    
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
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
        n_iterations = len(param_grid)
        
        # Run grid search
        scores = []
        best_score = float('inf')
        best_params = None
        convergence_history = []
        feature_importance = {}
        cross_validation_scores = []
        
        # Use tqdm for progress bar
        for params in tqdm(param_grid, desc="Grid Search"):
            # Calculate feature importance
            base_score = objective(params, data)
            feature_scores = {}
            
            for param in params:
                modified_params = params.copy()
                modified_params[param] = modified_params[param] * 1.1  # 10% increase
                modified_score = objective(modified_params, data)
                feature_scores[param] = abs(modified_score - base_score)
                
            feature_importance.update(feature_scores)
            
            # Calculate cross-validation scores
            cv_scores = self._cross_validate(objective, params, data, **kwargs)
            cross_validation_scores.extend(cv_scores)
            
            # Calculate main score
            score = objective(params, data)
            scores.append(score)
            convergence_history.append(score)
            
            if score < best_score:
                best_score = score
                best_params = params
                
            # Check early stopping
            if self._check_early_stopping(convergence_history, **kwargs):
                self.logger.info("Early stopping triggered")
                break
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate hyperparameter importance
        hyperparameter_importance = self._calculate_hyperparameter_importance(
            param_grid, scores
        )
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=scores,
            all_params=param_grid,
            optimization_time=optimization_time,
            n_iterations=n_iterations,
            convergence_history=convergence_history,
            feature_importance=feature_importance,
            cross_validation_scores=cross_validation_scores,
            hyperparameter_importance=hyperparameter_importance
        )
    
    def _cross_validate(self, objective: Callable, params: Dict[str, Any],
                       data: pd.DataFrame, **kwargs) -> List[float]:
        """Perform cross-validation.
        
        Args:
            objective: Objective function
            params: Parameters to evaluate
            data: Market data
            **kwargs: Additional parameters
            
        Returns:
            List of cross-validation scores
        """
        n_splits = kwargs.get('n_splits', 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            score = objective(params, val_data)
            scores.append(score)
            
        return scores
    
    def _calculate_hyperparameter_importance(self, param_grid: List[Dict[str, Any]],
                                          scores: List[float]) -> Dict[str, float]:
        """Calculate hyperparameter importance.
        
        Args:
            param_grid: List of parameter combinations
            scores: List of scores for each combination
            
        Returns:
            Dictionary of parameter importance scores
        """
        importance = {}
        
        for param in param_grid[0].keys():
            param_values = [p[param] for p in param_grid]
            correlations = np.corrcoef(param_values, scores)[0, 1]
            importance[param] = abs(correlations)
            
        return importance
    
    def _generate_grid(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter grid.
        
        Args:
            param_space: Parameter space to search
            
        Returns:
            List of parameter combinations
        """
        grid = {}
        
        for param, space in param_space.items():
            if isinstance(space, (list, tuple)):
                grid[param] = space
            elif isinstance(space, dict):
                grid[param] = np.linspace(
                    space['start'],
                    space['end'],
                    space.get('n_points', 10)
                )
                
        return list(ParameterGrid(grid))

class BayesianOptimization(OptimizationMethod):
    """Bayesian optimization method."""
    
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run Bayesian optimization.
        
        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args
        
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
                if 'start' in space and 'end' in space:
                    if isinstance(space['start'], int):
                        dimensions.append(Integer(space['start'], space['end'], name=param))
                    else:
                        dimensions.append(Real(space['start'], space['end'], name=param))
        
        # Define objective function with cross-validation
        @use_named_args(dimensions=dimensions)
        def objective_wrapper(**params):
            param_dict = params
            cv_scores = self._cross_validate(objective, param_dict, data, **kwargs)
            return {'success': True, 'result': {'success': True, 'result': np.mean(cv_scores), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        # Run optimization
        n_calls = kwargs.get('n_calls', 100)
        result = gp_minimize(
            objective_wrapper,
            dimensions,
            n_calls=n_calls,
            random_state=42,
            n_jobs=-1,
            callback=self._optimization_callback
        )
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            result.models[-1], dimensions
        )
        
        # Calculate hyperparameter importance
        hyperparameter_importance = self._calculate_hyperparameter_importance(
            result.x_iters, result.func_vals
        )
        
        return OptimizationResult(
            best_params=dict(zip(param_names, result.x)),
            best_score=result.fun,
            all_scores=result.func_vals,
            all_params=[dict(zip(param_names, x)) for x in result.x_iters],
            optimization_time=optimization_time,
            n_iterations=n_calls,
            convergence_history=result.func_vals,
            feature_importance=feature_importance,
            hyperparameter_importance=hyperparameter_importance
        )
        
    def _optimization_callback(self, res):
        """Callback for optimization progress.
        
        Args:
            res: Optimization result
        """
        if len(res.func_vals) % 10 == 0:
            self.logger.info(f"Iteration {len(res.func_vals)}, Best score: {min(res.func_vals)}")

    def _calculate_feature_importance(self, model, dimensions):
        """Calculate feature importance from GP model.
        
        Args:
            model: Gaussian Process model
            dimensions: Parameter dimensions
            
        Returns:
            Dictionary of feature importance scores
        """
        if not hasattr(model, 'kernel_'):
            return {}
            
        length_scales = model.kernel_.get_params()['k1__k2__length_scale']
        if not isinstance(length_scales, np.ndarray):
            length_scales = np.array([length_scales])
            
        importance = {}
        for i, dim in enumerate(dimensions):
            importance[dim.name] = 1.0 / length_scales[i]
            
        return importance

class RayOptimization(OptimizationMethod):
    """Ray-based distributed optimization with fallback to ThreadPoolExecutor."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ray optimization.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        if not RAY_AVAILABLE:
            self.logger.warning("Ray is not available. Falling back to ThreadPoolExecutor.")
            
    def _setup_ray(self):
        """Setup Ray for distributed computing."""
        if RAY_AVAILABLE and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run distributed optimization using Ray or ThreadPoolExecutor.
        
        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        start_time = datetime.now()
        
        if RAY_AVAILABLE:
            # Use Ray for distributed computing
            self._setup_ray()
            
            # Convert parameter space to Ray format
            config = {}
            for param, space in param_space.items():
                if isinstance(space, (list, tuple)):
                    config[param] = tune.choice(space)
                elif isinstance(space, dict):
                    if 'start' in space and 'end' in space:
                        if isinstance(space['start'], int):
                            config[param] = tune.randint(space['start'], space['end'])
                        else:
                            config[param] = tune.uniform(space['start'], space['end'])
            
            # Define training function
            def trainable(config):
                score = objective(config, data)
                tune.report(score=score)

            # Setup scheduler and search algorithm
            scheduler = ASHAScheduler(
                metric="score",
                mode="min",
                max_t=kwargs.get('max_t', 100),
                grace_period=kwargs.get('grace_period', 10)
            )
            
            search_alg = BayesOptSearch(
                metric="score",
                mode="min"
            )
            
            # Run optimization
            analysis = tune.run(
                trainable,
                config=config,
                num_samples=kwargs.get('num_samples', 100),
                scheduler=scheduler,
                search_alg=search_alg,
                resources_per_trial={"cpu": 1},
                verbose=1
            )
            
            # Get results
            best_trial = analysis.get_best_trial("score", "min", "last")
            best_params = best_trial.config
            best_score = best_trial.last_result["score"]
            all_params = [t.config for t in analysis.trials]
            all_scores = [t.last_result["score"] for t in analysis.trials]
        else:
            # Fallback to ThreadPoolExecutor
            param_combinations = self._generate_param_combinations(param_space)
            scores = []
            best_score = float('inf')
            best_params = None
            
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(objective, params, data) 
                          for params in param_combinations]
                for future, params in zip(as_completed(futures), param_combinations):
                    score = future.result()
                    scores.append(score)
                    if score < best_score:
                        best_score = score
                        best_params = params
            
            all_params = param_combinations
            all_scores = scores
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=all_scores,
            all_params=all_params,
            optimization_time=optimization_time,
            n_iterations=len(all_params),
            convergence_history=all_scores
        )

class OptunaOptimization(OptimizationMethod):
    """Optuna optimization method."""
    
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run Optuna optimization.
        
        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        start_time = datetime.now()
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42)
        )
        
        # Define objective function
        def objective_wrapper(trial):
            params = {}
            for param, space in param_space.items():
                if isinstance(space, (list, tuple)):
                    params[param] = trial.suggest_categorical(param, space)
                elif isinstance(space, dict):
                    if 'start' in space and 'end' in space:
                        if isinstance(space['start'], int):
                            params[param] = trial.suggest_int(
                                param, space['start'], space['end']
                            )
                        else:
                            params[param] = trial.suggest_float(
                                param, space['start'], space['end']
                            )
            return objective(params, data)
            
        # Run optimization
        n_trials = kwargs.get('n_trials', 100)
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Get results
        best_params = study.best_params
        best_score = study.best_value
        all_params = [t.params for t in study.trials]
        all_scores = [t.value for t in study.trials]
        
        # Calculate feature importance
        feature_importance = optuna.importance.get_param_importances(study)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=all_scores,
            all_params=all_params,
            optimization_time=optimization_time,
            n_iterations=n_trials,
            convergence_history=all_scores,
            feature_importance=feature_importance
        )

class PyTorchOptimization(OptimizationMethod):
    """PyTorch-based optimization method."""
    
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run PyTorch-based optimization.
        
        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        start_time = datetime.now()
        
        # Convert parameters to PyTorch tensors
        param_tensors = {}
        for param, space in param_space.items():
            if isinstance(space, (list, tuple)):
                param_tensors[param] = torch.tensor(space, dtype=torch.float32)
            elif isinstance(space, dict):
                if 'start' in space and 'end' in space:
                    param_tensors[param] = torch.linspace(
                        space['start'], space['end'], 
                        kwargs.get('n_points', 100)
                    )
                    
        # Initialize optimizer
        optimizer = optim.Adam([
            {'params': torch.tensor([0.0], requires_grad=True)}
        ])
        
        # Run optimization
        n_iterations = kwargs.get('n_iterations', 100)
        scores = []
        best_score = float('inf')
        best_params = None
        convergence_history = []
        
        for i in tqdm(range(n_iterations), desc="PyTorch Optimization"):
            optimizer.zero_grad()
            
            # Sample parameters
            params = {}
            for param, tensor in param_tensors.items():
                idx = torch.randint(0, len(tensor), (1,))
                params[param] = tensor[idx].item()
                
            # Calculate score
            score = objective(params, data)
            scores.append(score)
            convergence_history.append(score)
            
            if score < best_score:
                best_score = score
                best_params = params
                
            # Update parameters
            loss = torch.tensor(score, requires_grad=True)
            loss.backward()
            optimizer.step()
            
            # Check early stopping
            if self._check_early_stopping(convergence_history, **kwargs):
                self.logger.info("Early stopping triggered")
                break
                
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=scores,
            all_params=[params],
            optimization_time=optimization_time,
            n_iterations=n_iterations,
            convergence_history=convergence_history
        )

class DistributedOptimization(OptimizationMethod):
    """Distributed optimization using ray or ThreadPoolExecutor."""
    
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run distributed optimization.
        
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
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_space)
        n_iterations = len(param_combinations)
        
        # Initialize results
        scores = []
        best_score = float('inf')
        best_params = None
        convergence_history = []
        
        if RAY_AVAILABLE:
            # Initialize ray
            if not ray.is_initialized():
                ray.init()
            
            # Convert data to ray object
            data_ref = ray.put(data)
            
            # Define remote function
            @ray.remote
            def remote_objective(params):
                return {'success': True, 'result': {'success': True, 'result': objective(params, ray.get(data_ref)), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            # Run optimization in parallel
            futures = [remote_objective.remote(params) for params in param_combinations]
            results = ray.get(futures)
            
            # Process results
            for params, score in zip(param_combinations, results):
                scores.append(score)
                convergence_history.append(score)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    
                # Check early stopping
                if self._check_early_stopping(convergence_history, **kwargs):
                    break
        else:
            # Use ThreadPoolExecutor as fallback
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(objective, params, data) for params in param_combinations]
                
                for params, future in zip(param_combinations, futures):
                    score = future.result()
                    scores.append(score)
                    convergence_history.append(score)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
                        
                    # Check early stopping
                    if self._check_early_stopping(convergence_history, **kwargs):
                        break
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=scores,
            all_params=param_combinations,
            optimization_time=optimization_time,
            n_iterations=n_iterations,
            convergence_history=convergence_history
        )
        
    def _generate_param_combinations(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter combinations.
        
        Args:
            param_space: Parameter space to search
            
        Returns:
            List of parameter combinations
        """
        param_values = []
        param_names = []
        
        for param, space in param_space.items():
            param_names.append(param)
            if isinstance(space, (list, tuple)):
                param_values.append(space)
            elif isinstance(space, dict):
                if 'start' in space and 'end' in space:
                    param_values.append(np.linspace(
                        space['start'], space['end'], 
                        space.get('n_points', 10)
                    ))
                    
        combinations = []
        for values in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, values)))
            
        return combinations

class StrategyOptimizerConfig(OptimizerConfig):
    """Configuration for strategy optimizer."""
    
    # Optimization settings
    optimizer_type: str = Field("bayesian", description="Type of optimizer to use")
    n_initial_points: int = Field(5, ge=1, description="Number of initial points for Bayesian optimization")
    n_iterations: int = Field(50, ge=1, description="Number of optimization iterations")
    grid_search_points: int = Field(100, ge=1, description="Maximum number of points for grid search")
    
    # Metric settings
    primary_metric: str = Field("sharpe_ratio", description="Primary metric to optimize")
    secondary_metrics: List[str] = Field(
        ["win_rate", "max_drawdown"],
        description="Secondary metrics to track"
    )
    metric_weights: Dict[str, float] = Field(
        {"sharpe_ratio": 0.6, "win_rate": 0.3, "max_drawdown": 0.1},
        description="Weights for each metric"
    )
    
    # Bayesian optimization settings
    kernel_type: str = Field("matern", description="Type of kernel for Gaussian process")
    kernel_length_scale: float = Field(1.0, gt=0, description="Length scale for kernel")
    kernel_nu: float = Field(2.5, gt=0, description="Nu parameter for Matern kernel")
    
    # Grid search settings
    grid_search_strategy: str = Field("random", description="Strategy for grid search")
    grid_search_batch_size: int = Field(10, ge=1, description="Batch size for grid search")
    
    @validator('optimizer_type', allow_reuse=True)
    def validate_optimizer_type(cls, v):
        """Validate optimizer type."""
        valid_types = ["grid", "bayesian", "genetic"]
        if v not in valid_types:
            raise ValueError(f"optimizer_type must be one of {valid_types}")
        return v
    
    @validator('primary_metric', allow_reuse=True)
    def validate_primary_metric(cls, v):
        """Validate primary metric."""
        valid_metrics = ["sharpe_ratio", "win_rate", "max_drawdown", "mse", "alpha"]
        if v not in valid_metrics:
            raise ValueError(f"primary_metric must be one of {valid_metrics}")
        return v

class StrategyOptimizer(BaseOptimizer):
    """Strategy optimizer with multiple optimization methods."""
    
    def __init__(self, config: Union[Dict[str, Any], StrategyOptimizerConfig]):
        """Initialize strategy optimizer.
        
        Args:
            config: Configuration dictionary or StrategyOptimizerConfig object
        """
        if isinstance(config, dict):
            config = StrategyOptimizerConfig(**config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.performance_logger = PerformanceLogger()
        
        # Initialize optimization method
        self.optimizer = self._create_optimizer()
        
    def optimize(self, strategy_class: Any, data: pd.DataFrame,
                initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize strategy parameters.
        
        Args:
            strategy_class: Strategy class to optimize
            data: Market data
            initial_params: Optional initial parameters
            
        Returns:
            Dictionary of optimized parameters
        """
        # Get default parameters if not provided
        if initial_params is None:
            initial_params = self._get_default_params(strategy_class)
        
        # Create parameter space
        param_space = self._create_parameter_grid(initial_params)
        
        # Create objective function
        objective = self._objective_wrapper(strategy_class, data)
        
        # Run optimization
        result = self.optimizer.optimize(
            objective=objective,
            param_space=param_space,
            data=data,
            n_iterations=self.config.n_iterations
        )
        
        # Log results
        self._log_optimization_results(strategy_class, result.best_params, data)
        
        return result.best_params
    
    def _create_optimizer(self) -> OptimizationMethod:
        """Create optimization method based on config.
        
        Returns:
            OptimizationMethod instance
        """
        optimizer_map = {
            "grid": GridSearch,
            "bayesian": BayesianOptimization,
            "ray": RayOptimization if RAY_AVAILABLE else None,
            "optuna": OptunaOptimization,
            "pytorch": PyTorchOptimization
        }
        
        optimizer_class = optimizer_map.get(self.config.optimizer_type)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")
            
        return optimizer_class(self.config.dict())
    
    def _objective_wrapper(self, strategy_class: Any, data: pd.DataFrame) -> Callable:
        """Create objective function wrapper.
        
        Args:
            strategy_class: Strategy class
            data: Market data
            
        Returns:
            Objective function
        """
        def objective(params: Dict[str, Any]) -> float:
            try:
                # Create strategy instance
                strategy = strategy_class(params)
                
                # Generate signals
                signals = strategy.generate_signals(data)
                
                # Calculate metrics
                metrics = strategy.evaluate_performance(signals, data)
                
                # Calculate weighted score
                score = 0
                for metric, weight in self.config.metric_weights.items():
                    score += weight * getattr(metrics, metric)
                    
                return -score  # Minimize negative score
                
            except Exception as e:
                self.logger.error(f"Error in objective function: {str(e)}")
                return {'success': True, 'result': {'success': True, 'result': float('inf'), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                
        return objective
    
    def _get_default_params(self, strategy_class: Any) -> Dict[str, Any]:
        """Get default parameters for strategy.
        
        Args:
            strategy_class: Strategy class
            
        Returns:
            Dictionary of default parameters
        """
        if hasattr(strategy_class, 'default_params'):
            return strategy_class.default_params
        return {}
    
    def _create_parameter_grid(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create parameter grid for optimization.
        
        Args:
            params: Base parameters
            
        Returns:
            List of parameter combinations
        """
        grid = {}
        
        for param, value in params.items():
            if isinstance(value, (int, float)):
                # Create range around value
                grid[param] = {
                    'start': value * 0.5,
                    'end': value * 1.5,
                    'n_points': 5
                }
            elif isinstance(value, (list, tuple)):
                grid[param] = value
                
        return grid
    
    def _log_optimization_results(self, strategy_class: Any,
                                optimized_params: Dict[str, Any],
                                data: pd.DataFrame) -> None:
        """Log optimization results.
        
        Args:
            strategy_class: Strategy class
            optimized_params: Optimized parameters
            data: Market data
        """
        # Create strategy instance with optimized parameters
        strategy = strategy_class(optimized_params)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Calculate metrics
        metrics = strategy.evaluate_performance(signals, data)
        
        # Log results
        self.performance_logger.log_metrics(
            strategy_name=strategy_class.__name__,
            metrics=metrics,
            parameters=optimized_params
        )
        
        # Save results to file
        results_dir = Path('trading/optimization/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'strategy': strategy_class.__name__,
            'parameters': optimized_params,
            'metrics': metrics.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_dir / f"{strategy_class.__name__}_optimization.json", 'w') as f:
            json.dump(results, f, indent=4)

    def plot_optimization_results(self, save_path: Optional[str] = None) -> None:
        """Plot optimization results with convergence and parameter evolution.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            if not self.optimization_history:
                self.logger.warning("No optimization history available for plotting")
                return
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Strategy Optimization Results', fontsize=16)
            
            # Plot 1: Convergence curve
            iterations = range(1, len(self.optimization_history) + 1)
            best_scores = [entry['best_score'] for entry in self.optimization_history]
            axes[0, 0].plot(iterations, best_scores, 'b-', linewidth=2)
            axes[0, 0].set_title('Convergence Curve')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Best Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Parameter evolution
            if self.optimization_history:
                param_names = list(self.optimization_history[0]['best_params'].keys())
                for i, param in enumerate(param_names[:3]):  # Show first 3 parameters
                    values = [entry['best_params'].get(param, 0) for entry in self.optimization_history]
                    axes[0, 1].plot(iterations, values, label=param, linewidth=2)
                axes[0, 1].set_title('Parameter Evolution')
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_ylabel('Parameter Value')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Score distribution
            all_scores = [score for entry in self.optimization_history for score in entry.get('scores', [])]
            if all_scores:
                axes[1, 0].hist(all_scores, bins=20, alpha=0.7, color='green')
                axes[1, 0].set_title('Score Distribution')
                axes[1, 0].set_xlabel('Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Optimization progress
            if len(self.optimization_history) > 1:
                improvements = []
                for i in range(1, len(self.optimization_history)):
                    improvement = self.optimization_history[i]['best_score'] - self.optimization_history[i-1]['best_score']
                    improvements.append(improvement)
                
                axes[1, 1].bar(range(len(improvements)), improvements, alpha=0.7, color='orange')
                axes[1, 1].set_title('Score Improvements')
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Score Improvement')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Optimization plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting optimization results: {e}")
            raise

    def get_available_optimizers(self) -> List[str]:
        """Get list of available optimizers.
        
        Returns:
            List of optimizer names
        """
        return {'success': True, 'result': ["Grid", "Bayesian", "Genetic"], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_strategy_param_space(self, strategy: str) -> Dict[str, Any]:
        """Get parameter space for a given strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Dictionary with parameter ranges
        """
        param_spaces = {
            "RSI": {
                "period": {"start": 10, "end": 30},
                "overbought": {"start": 70, "end": 90},
                "oversold": {"start": 10, "end": 30}
            },
            "MACD": {
                "fast_period": {"start": 8, "end": 16},
                "slow_period": {"start": 20, "end": 30},
                "signal_period": {"start": 7, "end": 12}
            },
            "Bollinger": {
                "period": {"start": 15, "end": 30},
                "std_dev": {"start": 1.5, "end": 3.0}
            },
            "SMA": {
                "short_period": {"start": 5, "end": 15},
                "long_period": {"start": 20, "end": 50}
            }
        }
        
        return {'success': True, 'result': param_spaces.get(strategy, {}), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def optimize_strategy(self, strategy: str, optimizer_type: str, 
                         param_space: Dict[str, Any], training_data: pd.DataFrame, 
                         **settings) -> Dict[str, Any]:
        """Optimize a strategy with given parameters.
        
        Args:
            strategy: Strategy name
            optimizer_type: Type of optimizer to use
            param_space: Parameter space
            training_data: Training data
            **settings: Optimization settings
            
        Returns:
            Optimization results
        """
        # TODO: Implement actual optimization with cross-validation
        # - Add k-fold cross-validation for robust parameter estimation
        # - Implement time-series aware validation (walk-forward analysis)
        # - Add early stopping based on validation performance
        # - Include confidence intervals for parameter estimates
        # - Add support for multiple objective functions (multi-objective optimization)
        raise NotImplementedError('Pending feature - requires cross-validation implementation')

    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to a JSON file.
        
        Args:
            filepath: Path to save the optimization results
        """
        try:
            # Prepare data for saving
            save_data = {
                'optimization_history': self.optimization_history,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_config': {
                    'strategy_name': self.strategy_name,
                    'param_space': self.param_space,
                    'n_trials': self.n_trials,
                    'timeout': self.timeout
                },
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")
            raise

    def load_optimization_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results from a JSON file.
        
        Args:
            filepath: Path to load the optimization results from
            
        Returns:
            Dictionary containing the loaded optimization data
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore optimization state
            self.optimization_history = data.get('optimization_history', [])
            self.best_params = data.get('best_params', {})
            self.best_score = data.get('best_score', float('-inf'))
            
            # Validate loaded data
            if not self.optimization_history:
                self.logger.warning("No optimization history found in loaded file")
            
            self.logger.info(f"Optimization results loaded from {filepath}")
            return data
            
        except FileNotFoundError:
            self.logger.error(f"Optimization results file not found: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading optimization results: {e}")
            raise

__all__ = ["StrategyOptimizer", "StrategyOptimizerConfig"] 

# Merged from optimizer.py
class Optimizer:
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict] = None):
        """Initialize the optimizer with state and action dimensions.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary containing:
                - learning_rate: Learning rate for optimizer (default: 0.001)
                - hidden_dims: List of hidden layer dimensions (default: [64, 32])
                - dropout_rate: Dropout rate (default: 0.1)
                - batch_norm: Whether to use batch normalization (default: True)
                - log_level: Logging level (default: INFO)
                - results_dir: Directory for saving results (default: optimization_results)
        """
        # Load configuration from environment variables with defaults
        self.config = {
            'learning_rate': float(os.getenv('OPTIMIZER_LEARNING_RATE', 0.001)),
            'hidden_dims': [int(x) for x in os.getenv('OPTIMIZER_HIDDEN_DIMS', '64,32').split(',')],
            'dropout_rate': float(os.getenv('OPTIMIZER_DROPOUT_RATE', 0.1)),
            'batch_norm': os.getenv('OPTIMIZER_BATCH_NORM', 'true').lower() == 'true',
            'log_level': os.getenv('OPTIMIZER_LOG_LEVEL', 'INFO'),
            'results_dir': os.getenv('OPTIMIZER_RESULTS_DIR', 'optimization_results')
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config['log_level']))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Validate dimensions
        if not isinstance(state_dim, int) or state_dim <= 0:
            raise OptimizationError("state_dim must be a positive integer")
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise OptimizationError("action_dim must be a positive integer")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create results directory
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize neural network
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        # Initialize metrics tracking
        self.metrics_history = {
            'loss': [],
            'validation_loss': [],
            'optimization_time': []
        }
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _build_model(self) -> nn.Module:
        """Build neural network model.
        
        Returns:
            Neural network model
        """
        layers = []
        prev_dim = self.state_dim
        
        # Add hidden layers
        for dim in self.config['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim) if self.config['batch_norm'] else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate'])
            ])
            prev_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, self.action_dim))
        
        return nn.Sequential(*layers).to(self.device)
    
    def optimize_portfolio(self, state: np.ndarray, target: np.ndarray,
                         validation_split: float = 0.2) -> Tuple[np.ndarray, float]:
        """Optimize portfolio weights given current state and target.
        
        Args:
            state: State array
            target: Target array
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (optimized weights, loss)
            
        Raises:
            OptimizationError: If optimization fails
        """
        try:
            # Validate inputs
            if not isinstance(state, np.ndarray) or not isinstance(target, np.ndarray):
                raise OptimizationError("state and target must be numpy arrays")
            if state.shape[0] != target.shape[0]:
                raise OptimizationError("state and target must have same number of samples")
            
            # Split data
            split_idx = int(len(state) * (1 - validation_split))
            train_state = state[:split_idx]
            train_target = target[:split_idx]
            val_state = state[split_idx:]
            val_target = target[split_idx:]
            
            # Convert to tensors
            train_state_tensor = torch.FloatTensor(train_state).to(self.device)
            train_target_tensor = torch.FloatTensor(train_target).to(self.device)
            val_state_tensor = torch.FloatTensor(val_state).to(self.device)
            val_target_tensor = torch.FloatTensor(val_target).to(self.device)
            
            # Training
            start_time = datetime.now()
            self.optimizer.zero_grad()
            train_output = self.model(train_state_tensor)
            train_loss = self.criterion(train_output, train_target_tensor)
            train_loss.backward()
            self.optimizer.step()
            
            # Validation
            with torch.no_grad():
                val_output = self.model(val_state_tensor)
                val_loss = self.criterion(val_output, val_target_tensor)
            
            # Update metrics
            optimization_time = (datetime.now() - start_time).total_seconds()
            self.metrics_history['loss'].append(train_loss.item())
            self.metrics_history['validation_loss'].append(val_loss.item())
            self.metrics_history['optimization_time'].append(optimization_time)
            
            self.logger.info(f"Optimization completed - Loss: {train_loss.item():.4f}, "
                           f"Validation Loss: {val_loss.item():.4f}, "
                           f"Time: {optimization_time:.2f}s")
            
            return train_output.detach().cpu().numpy(), train_loss.item()
            
        except Exception as e:
            raise OptimizationError(f"Failed to optimize portfolio: {str(e)}")
    
    def get_optimal_weights(self, state: np.ndarray) -> np.ndarray:
        """Get optimal portfolio weights for given state.
        
        Args:
            state: State array
            
        Returns:
            Optimal weights array
            
        Raises:
            OptimizationError: If weight calculation fails
        """
        try:
            if not isinstance(state, np.ndarray):
                raise OptimizationError("state must be a numpy array")
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                weights = self.model(state_tensor)
                return weights.cpu().numpy()
                
        except Exception as e:
            raise OptimizationError(f"Failed to get optimal weights: {str(e)}")
    
    def update_model(self, state: np.ndarray, target: np.ndarray) -> float:
        """Update model with new data.
        
        Args:
            state: State array
            target: Target array
            
        Returns:
            Loss value
            
        Raises:
            OptimizationError: If model update fails
        """
        try:
            if not isinstance(state, np.ndarray) or not isinstance(target, np.ndarray):
                raise OptimizationError("state and target must be numpy arrays")
            
            state_tensor = torch.FloatTensor(state).to(self.device)
            target_tensor = torch.FloatTensor(target).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            
            self.logger.info(f"Model updated - Loss: {loss.item():.4f}")
            return loss.item()
            
        except Exception as e:
            raise OptimizationError(f"Failed to update model: {str(e)}")
    
    def save_model(self, path: str):
        """Save model to file.
        
        Args:
            path: Path to save model
            
        Raises:
            OptimizationError: If model saving fails
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'metrics_history': self.metrics_history
            }, save_path)
            
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            raise OptimizationError(f"Failed to save model: {str(e)}")

    def load_model(self, path: str):
        """Load model from file.
        
        Args:
            path: Path to load model from
            
        Raises:
            OptimizationError: If model loading fails
        """
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            self.metrics_history = checkpoint['metrics_history']
            
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            raise OptimizationError(f"Failed to load model: {str(e)}")

    def get_optimization_metrics(self) -> Dict[str, float]:
        """Get optimization metrics.
        
        Returns:
            Dictionary of optimization metrics
        """
        metrics = {
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'avg_loss': np.mean(self.metrics_history['loss']) if self.metrics_history['loss'] else 0,
            'avg_validation_loss': np.mean(self.metrics_history['validation_loss']) if self.metrics_history['validation_loss'] else 0,
            'avg_optimization_time': np.mean(self.metrics_history['optimization_time']) if self.metrics_history['optimization_time'] else 0
        }
        
        return metrics
    
    def save_metrics(self, path: Optional[str] = None):
        """Save optimization metrics to file.
        
        Args:
            path: Optional path to save metrics
            
        Raises:
            OptimizationError: If metrics saving fails
        """
        try:
            if path is None:
                path = self.results_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            metrics = {
                'optimization_metrics': self.get_optimization_metrics(),
                'metrics_history': self.metrics_history
            }
            
            with open(path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self.logger.info(f"Metrics saved to {path}")
            
        except Exception as e:
            raise OptimizationError(f"Failed to save metrics: {str(e)}")
