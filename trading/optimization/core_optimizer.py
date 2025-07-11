"""
Core Optimizer Module

This module consolidates all optimization functionality into a single, unified interface.
It provides a clean abstraction over different optimization algorithms and strategies.
Enhanced with parallelization support for improved performance.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import time
import traceback
from pathlib import Path

# Parallelization imports
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    CONCURRENT_FUTURES_AVAILABLE = True
except ImportError:
    CONCURRENT_FUTURES_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime
    optimizer_type: str
    strategy_name: str
    parallel_info: Optional[Dict[str, Any]] = None

class OptimizerConfig(BaseModel):
    """Configuration for optimizers."""
    optimizer_type: str = Field(..., description="Type of optimizer (bayesian, genetic, grid, etc.)")
    max_iterations: int = Field(100, description="Maximum optimization iterations")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    n_jobs: int = Field(1, description="Number of parallel jobs")
    random_state: Optional[int] = Field(None, description="Random seed")
    verbose: bool = Field(True, description="Verbose output")
    
    # Parallelization settings
    use_parallel: bool = Field(True, description="Whether to use parallel processing")
    parallel_backend: str = Field("joblib", description="Parallel backend (joblib, concurrent.futures)")
    parallel_chunk_size: int = Field(10, description="Chunk size for parallel processing")
    parallel_timeout: Optional[int] = Field(300, description="Timeout for parallel jobs (seconds)")
    
    # Bayesian optimization specific
    n_initial_points: int = Field(10, description="Number of initial random points")
    
    # Genetic optimization specific
    population_size: int = Field(50, description="Population size")
    n_generations: int = Field(50, description="Number of generations")
    mutation_rate: float = Field(0.1, description="Mutation rate")
    crossover_rate: float = Field(0.8, description="Crossover rate")
    
    # Grid optimization specific
    grid_resolution: int = Field(10, description="Grid resolution")
    
    @validator('parallel_backend')
    def validate_parallel_backend(cls, v):
        """Validate parallel backend."""
        valid_backends = ["joblib", "concurrent.futures"]
        if v not in valid_backends:
            raise ValueError(f"parallel_backend must be one of {valid_backends}")
        return v

class ParallelProcessor:
    """Handles parallel processing for optimization tasks."""
    
    def __init__(self, config: OptimizerConfig):
        """Initialize parallel processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check availability
        if config.use_parallel:
            if config.parallel_backend == "joblib" and not JOBLIB_AVAILABLE:
                self.logger.warning("joblib not available, falling back to sequential processing")
                config.use_parallel = False
            elif config.parallel_backend == "concurrent.futures" and not CONCURRENT_FUTURES_AVAILABLE:
                self.logger.warning("concurrent.futures not available, falling back to sequential processing")
                config.use_parallel = False
    
    def parallel_map(self, func: Callable, iterable: List, **kwargs) -> List:
        """Execute function in parallel over iterable.
        
        Args:
            func: Function to execute
            iterable: Items to process
            **kwargs: Additional arguments for func
            
        Returns:
            List of results
        """
        if not self.config.use_parallel or len(iterable) == 1:
            return [func(item, **kwargs) for item in iterable]
        
        if self.config.parallel_backend == "joblib":
            return self._joblib_parallel_map(func, iterable, **kwargs)
        elif self.config.parallel_backend == "concurrent.futures":
            return self._concurrent_futures_parallel_map(func, iterable, **kwargs)
        else:
            return [func(item, **kwargs) for item in iterable]
    
    def _joblib_parallel_map(self, func: Callable, iterable: List, **kwargs) -> List:
        """Use joblib for parallel processing."""
        try:
            results = Parallel(
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                timeout=self.config.parallel_timeout,
                batch_size=self.config.parallel_chunk_size
            )(
                delayed(func)(item, **kwargs) for item in iterable
            )
            return results
        except Exception as e:
            self.logger.error(f"Joblib parallel processing failed: {e}")
            self.logger.info("Falling back to sequential processing")
            return [func(item, **kwargs) for item in iterable]
    
    def _concurrent_futures_parallel_map(self, func: Callable, iterable: List, **kwargs) -> List:
        """Use concurrent.futures for parallel processing."""
        try:
            # Choose executor based on function type
            if self._is_cpu_bound(func):
                executor_class = ProcessPoolExecutor
            else:
                executor_class = ThreadPoolExecutor
            
            results = []
            with executor_class(max_workers=self.config.n_jobs) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(func, item, **kwargs): item 
                    for item in iterable
                }
                
                # Collect results
                for future in as_completed(future_to_item, timeout=self.config.parallel_timeout):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        item = future_to_item[future]
                        self.logger.error(f"Error processing item {item}: {e}")
                        results.append(None)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Concurrent.futures parallel processing failed: {e}")
            self.logger.info("Falling back to sequential processing")
            return [func(item, **kwargs) for item in iterable]
    
    def _is_cpu_bound(self, func: Callable) -> bool:
        """Determine if function is CPU-bound."""
        # Simple heuristic - could be enhanced
        func_name = func.__name__.lower()
        cpu_bound_keywords = ['compute', 'calculate', 'optimize', 'train', 'fit']
        return any(keyword in func_name for keyword in cpu_bound_keywords)

class BaseOptimizer(ABC):
    """Base class for all optimizers."""
    
    def __init__(self, config: OptimizerConfig):
        """Initialize optimizer with configuration."""
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.best_result: Optional[OptimizationResult] = None
        self.parallel_processor = ParallelProcessor(config)
        self.optimization_start_time = None
        self.optimization_end_time = None
        
    @abstractmethod
    def optimize(self, objective_function, param_space: Dict[str, Any], 
                data: pd.DataFrame) -> OptimizationResult:
        """Optimize the objective function over the parameter space."""
        pass
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found so far."""
        if self.best_result:
            return self.best_result.best_params
        return None
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the optimization history."""
        return self.history
    
    def reset(self):
        """Reset the optimizer state."""
        self.history = []
        self.best_result = None
        self.optimization_start_time = None
        self.optimization_end_time = None
    
    def _log_parallel_info(self) -> Dict[str, Any]:
        """Log parallel processing information."""
        return {
            'use_parallel': self.config.use_parallel,
            'parallel_backend': self.config.parallel_backend,
            'n_jobs': self.config.n_jobs,
            'chunk_size': self.config.parallel_chunk_size,
            'joblib_available': JOBLIB_AVAILABLE,
            'concurrent_futures_available': CONCURRENT_FUTURES_AVAILABLE
        }

class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using scikit-optimize."""
    
    def optimize(self, objective_function, param_space: Dict[str, Any], 
                data: pd.DataFrame) -> OptimizationResult:
        """Run Bayesian optimization."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            
            self.optimization_start_time = time.time()
            
            # Convert param_space to skopt format
            dimensions = []
            param_names = []
            
            for name, bounds in param_space.items():
                if isinstance(bounds, (list, tuple)):
                    if len(bounds) == 2 and all(isinstance(x, (int, float)) for x in bounds):
                        if all(isinstance(x, int) for x in bounds):
                            dimensions.append(Integer(bounds[0], bounds[1], name=name))
                        else:
                            dimensions.append(Real(bounds[0], bounds[1], name=name))
                        param_names.append(name)
                    elif all(isinstance(x, str) for x in bounds):
                        dimensions.append(Categorical(bounds, name=name))
                        param_names.append(name)
            
            # Define objective function for skopt
            def skopt_objective(params):
                param_dict = dict(zip(param_names, params))
                try:
                    score = objective_function(param_dict, data)
                    self.history.append({
                        'params': param_dict,
                        'score': score,
                        'timestamp': datetime.now()
                    })
                    return -score  # Minimize negative score (maximize score)
                except Exception as e:
                    logger.error(f"Objective function error: {e}")
                    return float('inf')
            
            # Run optimization
            result = gp_minimize(
                skopt_objective,
                dimensions,
                n_calls=self.config.max_iterations,
                n_initial_points=self.config.n_initial_points,
                random_state=self.config.random_state,
                verbose=self.config.verbose
            )
            
            self.optimization_end_time = time.time()
            
            # Create result
            best_params = dict(zip(param_names, result.x))
            self.best_result = OptimizationResult(
                best_params=best_params,
                best_score=-result.fun,  # Convert back to positive
                optimization_history=self.history,
                metadata={
                    'n_iterations': len(result.x_iters),
                    'optimization_time': self.optimization_end_time - self.optimization_start_time
                },
                timestamp=datetime.now(),
                optimizer_type='bayesian',
                strategy_name='unknown',
                parallel_info=self._log_parallel_info()
            )
            
            return self.best_result
            
        except ImportError:
            logger.error("scikit-optimize not available for Bayesian optimization")
            raise ImportError("scikit-optimize is required for Bayesian optimization")

class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm optimization."""
    
    def optimize(self, objective_function, param_space: Dict[str, Any], 
                data: pd.DataFrame) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        try:
            from deap import base, creator, tools, algorithms
            import random
            
            self.optimization_start_time = time.time()
            
            # Setup DEAP
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Define parameter bounds
            param_bounds = []
            param_names = []
            for name, bounds in param_space.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    param_bounds.append(bounds)
                    param_names.append(name)
            
            # Create individual and population
            def create_individual():
                return [random.uniform(bounds[0], bounds[1]) for bounds in param_bounds]
            
            toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Define evaluation function
            def evaluate(individual):
                param_dict = dict(zip(param_names, individual))
                try:
                    score = objective_function(param_dict, data)
                    self.history.append({
                        'params': param_dict,
                        'score': score,
                        'timestamp': datetime.now()
                    })
                    return (score,)
                except Exception as e:
                    logger.error(f"Objective function error: {e}")
                    return (float('-inf'),)
            
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Create initial population
            population = toolbox.population(n=self.config.population_size)
            
            # Run genetic algorithm
            algorithms.eaSimple(
                population, 
                toolbox, 
                cxpb=self.config.crossover_rate,
                mutpb=self.config.mutation_rate,
                ngen=self.config.n_generations,
                verbose=self.config.verbose
            )
            
            self.optimization_end_time = time.time()
            
            # Get best individual
            best_individual = tools.selBest(population, 1)[0]
            best_params = dict(zip(param_names, best_individual))
            
            self.best_result = OptimizationResult(
                best_params=best_params,
                best_score=best_individual.fitness.values[0],
                optimization_history=self.history,
                metadata={
                    'n_generations': self.config.n_generations,
                    'population_size': self.config.population_size,
                    'optimization_time': self.optimization_end_time - self.optimization_start_time
                },
                timestamp=datetime.now(),
                optimizer_type='genetic',
                strategy_name='unknown',
                parallel_info=self._log_parallel_info()
            )
            
            return self.best_result
            
        except ImportError:
            logger.error("DEAP not available for genetic optimization")
            raise ImportError("DEAP is required for genetic optimization")

class GridOptimizer(BaseOptimizer):
    """Grid search optimization with parallel processing."""
    
    def optimize(self, objective_function, param_space: Dict[str, Any], 
                data: pd.DataFrame) -> OptimizationResult:
        """Run grid search optimization with parallel processing."""
        try:
            from itertools import product
            
            self.optimization_start_time = time.time()
            
            # Generate parameter combinations
            param_names = list(param_space.keys())
            param_values = list(param_space.values())
            
            # Create grid
            grid_combinations = list(product(*param_values))
            
            if self.config.verbose:
                logger.info(f"Grid search: {len(grid_combinations)} combinations to evaluate")
            
            # Define evaluation function for parallel processing
            def evaluate_combination(combination):
                param_dict = dict(zip(param_names, combination))
                try:
                    score = objective_function(param_dict, data)
                    return {
                        'params': param_dict,
                        'score': score,
                        'timestamp': datetime.now()
                    }
                except Exception as e:
                    logger.error(f"Error evaluating combination {param_dict}: {e}")
                    return {
                        'params': param_dict,
                        'score': float('-inf'),
                        'timestamp': datetime.now(),
                        'error': str(e)
                    }
            
            # Run evaluations in parallel
            results = self.parallel_processor.parallel_map(
                evaluate_combination, 
                grid_combinations
            )
            
            # Filter out None results and update history
            valid_results = []
            for result in results:
                if result is not None:
                    self.history.append(result)
                    valid_results.append(result)
            
            if not valid_results:
                raise ValueError("No valid results from grid search")
            
            # Find best result
            best_result = max(valid_results, key=lambda x: x['score'])
            best_params = best_result['params']
            best_score = best_result['score']
            
            self.optimization_end_time = time.time()
            
            # Calculate parallel processing statistics
            parallel_stats = self._log_parallel_info()
            parallel_stats.update({
                'total_combinations': len(grid_combinations),
                'valid_results': len(valid_results),
                'failed_results': len(grid_combinations) - len(valid_results)
            })
            
            self.best_result = OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                optimization_history=self.history,
                metadata={
                    'n_combinations': len(grid_combinations),
                    'optimization_time': self.optimization_end_time - self.optimization_start_time,
                    'parallel_efficiency': len(valid_results) / len(grid_combinations)
                },
                timestamp=datetime.now(),
                optimizer_type='grid',
                strategy_name='unknown',
                parallel_info=parallel_stats
            )
            
            if self.config.verbose:
                logger.info(f"Grid search completed: {len(valid_results)}/{len(grid_combinations)} valid results")
                logger.info(f"Best score: {best_score:.4f}")
                logger.info(f"Optimization time: {self.optimization_end_time - self.optimization_start_time:.2f}s")
            
            return self.best_result
            
        except Exception as e:
            logger.error(f"Grid search optimization failed: {e}")
            raise

class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimization with parallel processing."""
    
    def optimize(self, objective_function, param_space: Dict[str, Any], 
                data: pd.DataFrame) -> OptimizationResult:
        """Run random search optimization."""
        try:
            import random
            
            self.optimization_start_time = time.time()
            
            # Set random seed if provided
            if self.config.random_state is not None:
                random.seed(self.config.random_state)
                np.random.seed(self.config.random_state)
            
            # Generate random parameter combinations
            param_combinations = []
            for _ in range(self.config.max_iterations):
                combination = {}
                for name, bounds in param_space.items():
                    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                        if all(isinstance(x, int) for x in bounds):
                            combination[name] = random.randint(bounds[0], bounds[1])
                        else:
                            combination[name] = random.uniform(bounds[0], bounds[1])
                    elif isinstance(bounds, (list, tuple)):
                        combination[name] = random.choice(bounds)
                param_combinations.append(combination)
            
            if self.config.verbose:
                logger.info(f"Random search: {len(param_combinations)} iterations")
            
            # Define evaluation function for parallel processing
            def evaluate_combination(combination):
                try:
                    score = objective_function(combination, data)
                    return {
                        'params': combination,
                        'score': score,
                        'timestamp': datetime.now()
                    }
                except Exception as e:
                    logger.error(f"Error evaluating combination {combination}: {e}")
                    return {
                        'params': combination,
                        'score': float('-inf'),
                        'timestamp': datetime.now(),
                        'error': str(e)
                    }
            
            # Run evaluations in parallel
            results = self.parallel_processor.parallel_map(
                evaluate_combination, 
                param_combinations
            )
            
            # Filter out None results and update history
            valid_results = []
            for result in results:
                if result is not None:
                    self.history.append(result)
                    valid_results.append(result)
            
            if not valid_results:
                raise ValueError("No valid results from random search")
            
            # Find best result
            best_result = max(valid_results, key=lambda x: x['score'])
            best_params = best_result['params']
            best_score = best_result['score']
            
            self.optimization_end_time = time.time()
            
            # Calculate parallel processing statistics
            parallel_stats = self._log_parallel_info()
            parallel_stats.update({
                'total_iterations': len(param_combinations),
                'valid_results': len(valid_results),
                'failed_results': len(param_combinations) - len(valid_results)
            })
            
            self.best_result = OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                optimization_history=self.history,
                metadata={
                    'n_iterations': len(param_combinations),
                    'optimization_time': self.optimization_end_time - self.optimization_start_time,
                    'success_rate': len(valid_results) / len(param_combinations)
                },
                timestamp=datetime.now(),
                optimizer_type='random_search',
                strategy_name='unknown',
                parallel_info=parallel_stats
            )
            
            if self.config.verbose:
                logger.info(f"Random search completed: {len(valid_results)}/{len(param_combinations)} valid results")
                logger.info(f"Best score: {best_score:.4f}")
                logger.info(f"Optimization time: {self.optimization_end_time - self.optimization_start_time:.2f}s")
            
            return self.best_result
            
        except Exception as e:
            logger.error(f"Random search optimization failed: {e}")
            raise

class OptimizerFactory:
    """Factory for creating optimizers."""
    
    _optimizers = {
        'bayesian': BayesianOptimizer,
        'genetic': GeneticOptimizer,
        'grid': GridOptimizer,
        'random_search': RandomSearchOptimizer
    }
    
    @classmethod
    def create(cls, optimizer_type: str, config: Optional[OptimizerConfig] = None) -> BaseOptimizer:
        """Create an optimizer instance."""
        if optimizer_type not in cls._optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        if config is None:
            config = OptimizerConfig(optimizer_type=optimizer_type)
        
        return cls._optimizers[optimizer_type](config)
    
    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        """Get list of available optimizer types."""
        return list(cls._optimizers.keys())
    
    @classmethod
    def register_optimizer(cls, name: str, optimizer_class: type):
        """Register a new optimizer type."""
        cls._optimizers[name] = optimizer_class

class StrategyOptimizer:
    """High-level strategy optimizer with parallel processing support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy optimizer."""
        self.config = config or {}
        self.strategies = self._load_strategies()
        self.logger = logging.getLogger(__name__)
    
    def _load_strategies(self) -> Dict[str, Any]:
        """Load available strategies."""
        # This would load from strategy registry
        return {
            'moving_average': {
                'param_space': {
                    'short_window': [5, 10, 15, 20],
                    'long_window': [20, 30, 40, 50],
                    'threshold': [0.01, 0.02, 0.05]
                }
            },
            'rsi': {
                'param_space': {
                    'period': [14, 20, 30],
                    'overbought': [70, 75, 80],
                    'oversold': [20, 25, 30]
                }
            }
        }
    
    def optimize_strategy(self, strategy_name: str, data: pd.DataFrame, 
                         optimizer_type: str = 'bayesian', **kwargs) -> OptimizationResult:
        """Optimize a specific strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        # Create optimizer config
        optimizer_config = OptimizerConfig(
            optimizer_type=optimizer_type,
            **{**self.config, **kwargs}
        )
        
        # Create optimizer
        optimizer = OptimizerFactory.create(optimizer_type, optimizer_config)
        
        # Define objective function
        def objective_function(params, data):
            return self._evaluate_strategy(strategy_name, params, data)
        
        # Run optimization
        param_space = self.strategies[strategy_name]['param_space']
        result = optimizer.optimize(objective_function, param_space, data)
        
        # Update strategy name
        result.strategy_name = strategy_name
        
        return result
    
    def _evaluate_strategy(self, strategy_name: str, params: Dict[str, Any], 
                          data: pd.DataFrame) -> float:
        """Evaluate strategy performance."""
        try:
            # This would integrate with actual strategy evaluation
            # For now, return a mock score
            import random
            random.seed(hash(str(params)) % 2**32)
            return random.uniform(0.5, 2.0)
        except Exception as e:
            self.logger.error(f"Strategy evaluation error: {e}")
            return float('-inf')
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        return list(self.strategies.keys())
    
    def get_strategy_param_space(self, strategy_name: str) -> Dict[str, Any]:
        """Get parameter space for a strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
        return self.strategies[strategy_name]['param_space']

def create_genetic_optimizer(config: Optional[Dict[str, Any]] = None) -> GeneticOptimizer:
    """Create a genetic optimizer with default configuration."""
    optimizer_config = OptimizerConfig(
        optimizer_type='genetic',
        **(config or {})
    )
    return GeneticOptimizer(optimizer_config)

def create_grid_optimizer(config: Optional[Dict[str, Any]] = None) -> GridOptimizer:
    """Create a grid optimizer with default configuration."""
    optimizer_config = OptimizerConfig(
        optimizer_type='grid',
        **(config or {})
    )
    return GridOptimizer(optimizer_config)

def create_bayesian_optimizer(config: Optional[Dict[str, Any]] = None) -> BayesianOptimizer:
    """Create a Bayesian optimizer with default configuration."""
    optimizer_config = OptimizerConfig(
        optimizer_type='bayesian',
        **(config or {})
    )
    return BayesianOptimizer(optimizer_config)

def create_random_search_optimizer(config: Optional[Dict[str, Any]] = None) -> RandomSearchOptimizer:
    """Create a random search optimizer with default configuration."""
    optimizer_config = OptimizerConfig(
        optimizer_type='random_search',
        **(config or {})
    )
    return RandomSearchOptimizer(optimizer_config) 