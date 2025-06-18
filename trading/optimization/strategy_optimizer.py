"""Strategy optimization module with multiple optimization methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from collections import deque
import random
from trading.models.base_model import BaseModel
import torch.optim as optim
import pandas as pd
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime
import json
import os
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
from optuna.samplers import TPESampler
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
from functools import partial
import asyncio
import aiohttp
from typing_extensions import TypedDict
import itertools
from pydantic import BaseModel, Field, validator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import ParameterGrid

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
                return float('inf')
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
            return np.mean(cv_scores)
        
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
                return objective(params, ray.get(data_ref))
            
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
    
    @validator('optimizer_type')
    def validate_optimizer_type(cls, v):
        """Validate optimizer type."""
        valid_types = ["bayesian", "grid", "random", "ray", "optuna", "pytorch"]
        if v not in valid_types:
            raise ValueError(f"Invalid optimizer type. Must be one of: {valid_types}")
        return v
    
    @validator('primary_metric')
    def validate_primary_metric(cls, v):
        """Validate primary metric."""
        valid_metrics = ["sharpe_ratio", "win_rate", "max_drawdown", "profit_factor"]
        if v not in valid_metrics:
            raise ValueError(f"Invalid primary metric. Must be one of: {valid_metrics}")
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
                return float('inf')
                
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

__all__ = ["StrategyOptimizer", "StrategyOptimizerConfig"] 