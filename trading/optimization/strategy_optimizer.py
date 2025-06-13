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

# Try to import ray and its submodules
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.bayesopt import BayesOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

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
            scores: List of corresponding scores
            
        Returns:
            Dictionary of hyperparameter importance scores
        """
        importance = {}
        
        for param in param_grid[0].keys():
            param_values = [p[param] for p in param_grid]
            correlation = np.corrcoef(param_values, scores)[0, 1]
            importance[param] = abs(correlation)
            
        return importance

    def _generate_grid(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter grid from parameter space.
        
        Args:
            param_space: Parameter space to generate grid from
            
        Returns:
            List of parameter combinations
        """
        # Convert parameter space to lists
        param_lists = {}
        for param, space in param_space.items():
            if isinstance(space, (list, tuple)):
                param_lists[param] = space
            elif isinstance(space, dict):
                start = space['start']
                end = space['end']
                step = space.get('step', (end - start) / 10)
                param_lists[param] = np.arange(start, end + step, step)
            else:
                param_lists[param] = [space]
                
        # Generate all combinations
        keys = param_lists.keys()
        values = param_lists.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]

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

class StrategyOptimizer:
    """Strategy optimization class."""
    
    def __init__(self, config: Optional[Dict] = None, state_dim: Optional[int] = None, action_dim: Optional[int] = None):
        """Initialize the StrategyOptimizer.

        Args:
            config: Optional configuration dictionary.
            state_dim: Optional dimension of the state space.
            action_dim: Optional dimension of the action space.
        """
        self.config = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization methods
        self.methods = {
            'grid': GridSearch(),
            'bayesian': BayesianOptimization(),
            'optuna': OptunaOptimization(),
            'pytorch': PyTorchOptimization(),
            'distributed': DistributedOptimization()
        }
        
        # Initialize data scaler
        self.scaler = StandardScaler()
        
        # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        # Set memory management parameters
        self.max_memory = self.config.get('max_memory', 1e9)  # 1GB default
        self.chunk_size = self.config.get('chunk_size', 1000)  # 1000 rows default

    def optimize(self, strategy: Any, data: pd.DataFrame, param_space: Dict[str, Any],
                method: str = 'bayesian', n_splits: int = 5, 
                metric: Optional[Callable] = None, constraints: Optional[List[Callable]] = None,
                **kwargs) -> OptimizationResult:
        """Optimize strategy parameters.
        
        Args:
            strategy: Strategy to optimize
            data: Market data
            param_space: Parameter space to search
            method: Optimization method to use
            n_splits: Number of time series splits for cross-validation
            metric: Custom metric function to use
            constraints: List of constraint functions
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        if method not in self.methods:
            raise ValueError(f"Unknown optimization method: {method}")
            
        # Normalize data
        data = self._normalize_data(data)
        
        # Define objective function
        def objective(params: Dict[str, Any], data: pd.DataFrame) -> float:
            # Check constraints
            if constraints:
                for constraint in constraints:
                    if not constraint(params):
                        return float('inf')
                        
            # Set strategy parameters
            strategy.set_params(**params)
            
            # Perform time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                # Train strategy
                strategy.fit(train_data)
                
                # Evaluate strategy
                if metric:
                    score = metric(strategy, val_data)
                else:
                    score = strategy.evaluate(val_data)
                scores.append(score)
                
            return np.mean(scores)
            
        # Run optimization with memory management
        if data.memory_usage().sum() > self.max_memory:
            self.logger.info("Using memory-optimized optimization")
            return self._memory_optimized_optimize(
                objective=objective,
                data=data,
                param_space=param_space,
                method=method,
                **kwargs
            )
            
        # Run optimization
        result = self.methods[method].optimize(
            objective=objective,
            param_space=param_space,
            data=data,
            n_splits=n_splits,
            **kwargs
        )
        
        # Save results
        self.save_results(result, f"{strategy.__class__.__name__}_{method}")
        
        return result
        
    def _memory_optimized_optimize(self, objective: Callable, data: pd.DataFrame,
                                 param_space: Dict[str, Any], method: str,
                                 **kwargs) -> OptimizationResult:
        """Run memory-optimized optimization.
        
        Args:
            objective: Objective function
            data: Market data
            param_space: Parameter space to search
            method: Optimization method to use
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        # Split data into chunks
        chunks = np.array_split(data, len(data) // self.chunk_size + 1)
        
        # Run optimization on each chunk
        results = []
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            result = self.methods[method].optimize(
                objective=objective,
                param_space=param_space,
                data=chunk,
                **kwargs
            )
            results.append(result)
            
        # Combine results
        best_result = min(results, key=lambda x: x.best_score)
        all_scores = []
        all_params = []
        convergence_history = []
        
        for result in results:
            all_scores.extend(result.all_scores)
            all_params.extend(result.all_params)
            convergence_history.extend(result.convergence_history)
            
        return OptimizationResult(
            best_params=best_result.best_params,
            best_score=best_result.best_score,
            all_scores=all_scores,
            all_params=all_params,
            optimization_time=sum(r.optimization_time for r in results),
            n_iterations=sum(r.n_iterations for r in results),
            convergence_history=convergence_history
        )
        
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize market data.
        
        Args:
            data: Market data to normalize
            
        Returns:
            Normalized data
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        return data
        
    def parallel_optimize(self, strategies: List[Any], data: pd.DataFrame,
                         param_spaces: List[Dict[str, Any]], method: str = 'bayesian',
                         n_splits: int = 5, **kwargs) -> Dict[str, OptimizationResult]:
        """Optimize multiple strategies in parallel.
        
        Args:
            strategies: List of strategies to optimize
            data: Market data
            param_spaces: List of parameter spaces
            method: Optimization method to use
            n_splits: Number of time series splits
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary mapping strategy names to optimization results
        """
        if len(strategies) != len(param_spaces):
            raise ValueError("Number of strategies must match number of parameter spaces")
            
        results = {}
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for strategy, param_space in zip(strategies, param_spaces):
                future = executor.submit(
                    self.optimize,
                    strategy=strategy,
                    data=data,
                    param_space=param_space,
                    method=method,
                    n_splits=n_splits,
                    **kwargs
                )
                futures.append((strategy.__class__.__name__, future))
                
            for name, future in futures:
                try:
                    results[name] = future.result()
                except Exception as e:
                    self.logger.error(f"Error optimizing {name}: {str(e)}")
                    
        return results
        
    def save_results(self, result: OptimizationResult, filename: str) -> None:
        """Save optimization results.
        
        Args:
            result: Optimization result to save
            filename: Output filename
        """
        # Convert result to dictionary
        result_dict = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'all_scores': result.all_scores,
            'all_params': result.all_params,
            'optimization_time': result.optimization_time,
            'n_iterations': result.n_iterations,
            'convergence_history': result.convergence_history,
            'metadata': result.metadata,
            'feature_importance': result.feature_importance,
            'cross_validation_scores': result.cross_validation_scores,
            'hyperparameter_importance': result.hyperparameter_importance
        }
        
        # Save to file
        output_path = self.results_dir / f"{filename}.json"
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=4)
            
        # Save plots
        self._plot_results(result, filename)
        
    def _plot_results(self, result: OptimizationResult, filename: str) -> None:
        """Plot optimization results.
        
        Args:
            result: Optimization result to plot
            filename: Output filename
        """
        # Create plots directory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot convergence history
        plt.figure(figsize=(10, 6))
        plt.plot(result.convergence_history)
        plt.title("Optimization Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.savefig(plots_dir / f"{filename}_convergence.png")
        plt.close()
        
        # Plot feature importance
        if result.feature_importance:
            plt.figure(figsize=(10, 6))
            importance = pd.Series(result.feature_importance)
            importance.sort_values().plot(kind='barh')
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{filename}_feature_importance.png")
            plt.close()
            
        # Plot hyperparameter importance
        if result.hyperparameter_importance:
            plt.figure(figsize=(10, 6))
            importance = pd.Series(result.hyperparameter_importance)
            importance.sort_values().plot(kind='barh')
            plt.title("Hyperparameter Importance")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{filename}_hyperparameter_importance.png")
            plt.close()
            
        # Plot cross-validation scores
        if result.cross_validation_scores:
            plt.figure(figsize=(10, 6))
            plt.boxplot(result.cross_validation_scores)
            plt.title("Cross-validation Scores")
            plt.ylabel("Score")
            plt.savefig(plots_dir / f"{filename}_cv_scores.png")
            plt.close()
            
    def load_results(self, filename: str) -> OptimizationResult:
        """Load optimization results.
        
        Args:
            filename: Input filename
            
        Returns:
            OptimizationResult object
        """
        input_path = self.results_dir / f"{filename}.json"
        with open(input_path, 'r') as f:
            result_dict = json.load(f)
            
        return OptimizationResult(
            best_params=result_dict['best_params'],
            best_score=result_dict['best_score'],
            all_scores=result_dict['all_scores'],
            all_params=result_dict['all_params'],
            optimization_time=result_dict['optimization_time'],
            n_iterations=result_dict['n_iterations'],
            convergence_history=result_dict['convergence_history'],
            metadata=result_dict.get('metadata'),
            feature_importance=result_dict.get('feature_importance'),
            cross_validation_scores=result_dict.get('cross_validation_scores'),
            hyperparameter_importance=result_dict.get('hyperparameter_importance')
        ) 