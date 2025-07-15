"""
Parameter Tuner

Tunes hyperparameters with support for user-defined parameter ranges
from configuration files or command line arguments.
"""

import logging
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParameterRange:
    """Parameter range configuration."""
    name: str
    min_value: Union[float, int]
    max_value: Union[float, int]
    step: Optional[Union[float, int]] = None
    parameter_type: str = "float"  # 'float', 'int', 'categorical'
    categories: Optional[List[Any]] = None
    distribution: str = "uniform"  # 'uniform', 'log', 'normal'
    description: str = ""


@dataclass
class TuningConfig:
    """Configuration for parameter tuning."""
    algorithm: str
    objective: str
    max_trials: int
    timeout_seconds: int
    n_jobs: int
    random_state: int
    early_stopping: bool
    early_stopping_patience: int
    parameter_ranges: Dict[str, ParameterRange]
    constraints: List[Dict[str, Any]]


class ParameterTuner:
    """Parameter tuner with configurable parameter ranges."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the parameter tuner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.tuning_config = self._initialize_tuning_config()
        self.logger = logging.getLogger(__name__)
        self.tuning_history = []
        
    def _initialize_tuning_config(self) -> TuningConfig:
        """Initialize tuning configuration."""
        return TuningConfig(
            algorithm=self.config.get("algorithm", "bayesian"),
            objective=self.config.get("objective", "minimize"),
            max_trials=self.config.get("max_trials", 100),
            timeout_seconds=self.config.get("timeout_seconds", 3600),
            n_jobs=self.config.get("n_jobs", -1),
            random_state=self.config.get("random_state", 42),
            early_stopping=self.config.get("early_stopping", True),
            early_stopping_patience=self.config.get("early_stopping_patience", 10),
            parameter_ranges=self._load_parameter_ranges(),
            constraints=self.config.get("constraints", [])
        )
        
    def _load_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Load parameter ranges from configuration."""
        ranges = {}
        
        # Load from config if provided
        if "parameter_ranges" in self.config:
            for name, range_config in self.config["parameter_ranges"].items():
                ranges[name] = ParameterRange(
                    name=name,
                    min_value=range_config["min_value"],
                    max_value=range_config["max_value"],
                    step=range_config.get("step"),
                    parameter_type=range_config.get("parameter_type", "float"),
                    categories=range_config.get("categories"),
                    distribution=range_config.get("distribution", "uniform"),
                    description=range_config.get("description", "")
                )
                
        return ranges
        
    def load_config_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {config_path.suffix}")
            
        self.config.update(config)
        self.tuning_config = self._initialize_tuning_config()
        self.logger.info(f"Loaded configuration from {config_path}")
        
    def load_config_from_cli_args(self, args: Optional[List[str]] = None) -> None:
        """Load configuration from command line arguments.
        
        Args:
            args: Command line arguments (None for sys.argv)
        """
        parser = self._create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Convert parsed args to config
        config = {}
        
        if parsed_args.config_file:
            self.load_config_from_file(parsed_args.config_file)
            return
            
        # Algorithm settings
        if parsed_args.algorithm:
            config["algorithm"] = parsed_args.algorithm
        if parsed_args.max_trials:
            config["max_trials"] = parsed_args.max_trials
        if parsed_args.timeout:
            config["timeout_seconds"] = parsed_args.timeout
        if parsed_args.n_jobs:
            config["n_jobs"] = parsed_args.n_jobs
        if parsed_args.random_state:
            config["random_state"] = parsed_args.random_state
            
        # Parameter ranges
        if parsed_args.parameter_ranges:
            config["parameter_ranges"] = self._parse_parameter_ranges(parsed_args.parameter_ranges)
            
        # Early stopping
        if parsed_args.early_stopping is not None:
            config["early_stopping"] = parsed_args.early_stopping
        if parsed_args.early_stopping_patience:
            config["early_stopping_patience"] = parsed_args.early_stopping_patience
            
        self.config.update(config)
        self.tuning_config = self._initialize_tuning_config()
        self.logger.info("Loaded configuration from command line arguments")
        
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI configuration."""
        parser = argparse.ArgumentParser(description="Parameter Tuner CLI")
        
        # Configuration file
        parser.add_argument(
            "--config-file", "-c",
            type=str,
            help="Path to configuration file (YAML or JSON)"
        )
        
        # Algorithm settings
        parser.add_argument(
            "--algorithm", "-a",
            type=str,
            choices=["bayesian", "grid_search", "random_search", "genetic"],
            help="Tuning algorithm to use"
        )
        
        parser.add_argument(
            "--max-trials", "-t",
            type=int,
            help="Maximum number of trials"
        )
        
        parser.add_argument(
            "--timeout", "-o",
            type=int,
            help="Timeout in seconds"
        )
        
        parser.add_argument(
            "--n-jobs", "-j",
            type=int,
            help="Number of parallel jobs"
        )
        
        parser.add_argument(
            "--random-state", "-r",
            type=int,
            help="Random state for reproducibility"
        )
        
        # Parameter ranges
        parser.add_argument(
            "--parameter-ranges", "-p",
            type=str,
            nargs="+",
            help="Parameter ranges in format 'name:min:max:type' or 'name:min:max:step:type'"
        )
        
        # Early stopping
        parser.add_argument(
            "--early-stopping",
            action="store_true",
            help="Enable early stopping"
        )
        
        parser.add_argument(
            "--no-early-stopping",
            dest="early_stopping",
            action="store_false",
            help="Disable early stopping"
        )
        
        parser.add_argument(
            "--early-stopping-patience",
            type=int,
            help="Early stopping patience"
        )
        
        return parser
        
    def _parse_parameter_ranges(self, range_strings: List[str]) -> Dict[str, Dict[str, Any]]:
        """Parse parameter ranges from string format.
        
        Format: 'name:min:max:type' or 'name:min:max:step:type'
        """
        ranges = {}
        
        for range_str in range_strings:
            parts = range_str.split(':')
            
            if len(parts) < 4:
                raise ValueError(f"Invalid parameter range format: {range_str}")
                
            name = parts[0]
            min_value = float(parts[1])
            max_value = float(parts[2])
            
            if len(parts) == 4:
                param_type = parts[3]
                step = None
            else:
                step = float(parts[3])
                param_type = parts[4]
                
            ranges[name] = {
                "min_value": min_value,
                "max_value": max_value,
                "step": step,
                "parameter_type": param_type
            }
            
        return ranges
        
    def add_parameter_range(
        self,
        name: str,
        min_value: Union[float, int],
        max_value: Union[float, int],
        step: Optional[Union[float, int]] = None,
        parameter_type: str = "float",
        categories: Optional[List[Any]] = None,
        distribution: str = "uniform",
        description: str = ""
    ) -> None:
        """Add a parameter range.
        
        Args:
            name: Parameter name
            min_value: Minimum value
            max_value: Maximum value
            step: Step size (for grid search)
            parameter_type: Parameter type ('float', 'int', 'categorical')
            categories: Categories for categorical parameters
            distribution: Distribution type ('uniform', 'log', 'normal')
            description: Parameter description
        """
        param_range = ParameterRange(
            name=name,
            min_value=min_value,
            max_value=max_value,
            step=step,
            parameter_type=parameter_type,
            categories=categories,
            distribution=distribution,
            description=description
        )
        
        self.tuning_config.parameter_ranges[name] = param_range
        self.logger.info(f"Added parameter range: {name}")
        
    def remove_parameter_range(self, name: str) -> None:
        """Remove a parameter range.
        
        Args:
            name: Parameter name to remove
        """
        if name in self.tuning_config.parameter_ranges:
            del self.tuning_config.parameter_ranges[name]
            self.logger.info(f"Removed parameter range: {name}")
        else:
            self.logger.warning(f"Parameter range not found: {name}")
            
    def get_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Get all parameter ranges."""
        return self.tuning_config.parameter_ranges.copy()
        
    def tune_parameters(
        self,
        objective_function: callable,
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Tune parameters using the configured algorithm.
        
        Args:
            objective_function: Function to optimize
            data: Data for evaluation
            **kwargs: Additional arguments for objective function
            
        Returns:
            Dictionary with tuning results
        """
        if not self.tuning_config.parameter_ranges:
            raise ValueError("No parameter ranges configured")
            
        self.logger.info(f"Starting parameter tuning with {self.tuning_config.algorithm}")
        self.logger.info(f"Parameter ranges: {list(self.tuning_config.parameter_ranges.keys())}")
        
        # Create parameter space
        param_space = self._create_parameter_space()
        
        # Run tuning based on algorithm
        if self.tuning_config.algorithm == "bayesian":
            result = self._bayesian_optimization(objective_function, param_space, data, **kwargs)
        elif self.tuning_config.algorithm == "grid_search":
            result = self._grid_search(objective_function, param_space, data, **kwargs)
        elif self.tuning_config.algorithm == "random_search":
            result = self._random_search(objective_function, param_space, data, **kwargs)
        elif self.tuning_config.algorithm == "genetic":
            result = self._genetic_optimization(objective_function, param_space, data, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.tuning_config.algorithm}")
            
        # Store in history
        self.tuning_history.append(result)
        
        return result
        
    def _create_parameter_space(self) -> Dict[str, Any]:
        """Create parameter space for optimization."""
        param_space = {}
        
        for name, param_range in self.tuning_config.parameter_ranges.items():
            if param_range.parameter_type == "categorical":
                param_space[name] = param_range.categories
            elif param_range.parameter_type == "int":
                if param_range.step:
                    param_space[name] = np.arange(
                        param_range.min_value, 
                        param_range.max_value + param_range.step, 
                        param_range.step
                    ).astype(int)
                else:
                    param_space[name] = (param_range.min_value, param_range.max_value)
            else:  # float
                if param_range.step:
                    param_space[name] = np.arange(
                        param_range.min_value, 
                        param_range.max_value + param_range.step, 
                        param_range.step
                    )
                else:
                    param_space[name] = (param_range.min_value, param_range.max_value)
                    
        return param_space
        
    def _bayesian_optimization(
        self, 
        objective_function: callable, 
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            
            # Convert parameter space to skopt format
            dimensions = []
            param_names = []
            
            for name, space in param_space.items():
                param_range = self.tuning_config.parameter_ranges[name]
                param_names.append(name)
                
                if param_range.parameter_type == "categorical":
                    dimensions.append(Categorical(space, name=name))
                elif param_range.parameter_type == "int":
                    if isinstance(space, tuple):
                        dimensions.append(Integer(space[0], space[1], name=name))
                    else:
                        dimensions.append(Categorical(space, name=name))
                else:  # float
                    if isinstance(space, tuple):
                        dimensions.append(Real(space[0], space[1], name=name))
                    else:
                        dimensions.append(Categorical(space, name=name))
                        
            # Define objective function
            def objective(params):
                param_dict = dict(zip(param_names, params))
                return objective_function(param_dict, data, **kwargs)
                
            # Run optimization
            result = gp_minimize(
                objective,
                dimensions,
                n_calls=self.tuning_config.max_trials,
                random_state=self.tuning_config.random_state,
                n_jobs=self.tuning_config.n_jobs
            )
            
            return {
                "algorithm": "bayesian",
                "best_params": dict(zip(param_names, result.x)),
                "best_score": result.fun,
                "n_trials": len(result.func_vals),
                "convergence": result.converged,
                "all_scores": result.func_vals,
                "all_params": [dict(zip(param_names, x)) for x in result.x_iters]
            }
            
        except ImportError:
            self.logger.error("scikit-optimize not available for Bayesian optimization")
            raise
            
    def _grid_search(
        self, 
        objective_function: callable, 
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Run grid search optimization."""
        import itertools
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Convert tuple ranges to lists
        param_lists = []
        for i, values in enumerate(param_values):
            if isinstance(values, tuple):
                param_range = self.tuning_config.parameter_ranges[param_names[i]]
                if param_range.parameter_type == "int":
                    param_lists.append(range(int(values[0]), int(values[1]) + 1))
                else:
                    param_lists.append(np.linspace(values[0], values[1], 10))
            else:
                param_lists.append(values)
                
        # Generate combinations
        combinations = list(itertools.product(*param_lists))
        
        # Limit combinations if too many
        if len(combinations) > self.tuning_config.max_trials:
            indices = np.random.choice(
                len(combinations), 
                self.tuning_config.max_trials, 
                replace=False
            )
            combinations = [combinations[i] for i in indices]
            
        # Evaluate all combinations
        scores = []
        all_params = []
        
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            score = objective_function(param_dict, data, **kwargs)
            scores.append(score)
            all_params.append(param_dict)
            
        # Find best result
        best_idx = np.argmin(scores) if self.tuning_config.objective == "minimize" else np.argmax(scores)
        
        return {
            "algorithm": "grid_search",
            "best_params": all_params[best_idx],
            "best_score": scores[best_idx],
            "n_trials": len(scores),
            "convergence": True,
            "all_scores": scores,
            "all_params": all_params
        }
        
    def _random_search(
        self, 
        objective_function: callable, 
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Run random search optimization."""
        scores = []
        all_params = []
        
        for trial in range(self.tuning_config.max_trials):
            # Sample random parameters
            params = {}
            for name, space in param_space.items():
                param_range = self.tuning_config.parameter_ranges[name]
                
                if param_range.parameter_type == "categorical":
                    params[name] = np.random.choice(space)
                elif isinstance(space, tuple):
                    if param_range.parameter_type == "int":
                        params[name] = np.random.randint(space[0], space[1] + 1)
                    else:
                        params[name] = np.random.uniform(space[0], space[1])
                else:
                    params[name] = np.random.choice(space)
                    
            # Evaluate
            score = objective_function(params, data, **kwargs)
            scores.append(score)
            all_params.append(params)
            
        # Find best result
        best_idx = np.argmin(scores) if self.tuning_config.objective == "minimize" else np.argmax(scores)
        
        return {
            "algorithm": "random_search",
            "best_params": all_params[best_idx],
            "best_score": scores[best_idx],
            "n_trials": len(scores),
            "convergence": True,
            "all_scores": scores,
            "all_params": all_params
        }
        
    def _genetic_optimization(
        self, 
        objective_function: callable, 
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Run genetic algorithm optimization."""
        try:
            from deap import base, creator, tools, algorithms
            import random
            
            # Setup genetic algorithm
            if self.tuning_config.objective == "minimize":
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMin)
            else:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
                
            toolbox = base.Toolbox()
            
            # Parameter generation
            param_names = list(param_space.keys())
            
            def generate_param():
                params = []
                for name, space in param_space.items():
                    param_range = self.tuning_config.parameter_ranges[name]
                    
                    if param_range.parameter_type == "categorical":
                        params.append(np.random.choice(space))
                    elif isinstance(space, tuple):
                        if param_range.parameter_type == "int":
                            params.append(np.random.randint(space[0], space[1] + 1))
                        else:
                            params.append(np.random.uniform(space[0], space[1]))
                    else:
                        params.append(np.random.choice(space))
                return params
                
            toolbox.register("individual", tools.initIterate, creator.Individual, generate_param)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Evaluation function
            def evaluate(individual):
                param_dict = dict(zip(param_names, individual))
                return objective_function(param_dict, data, **kwargs),
                
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Run genetic algorithm
            population = toolbox.population(n=50)
            algorithms.eaSimple(
                population, toolbox, 
                cxpb=0.7, mutpb=0.3, 
                ngen=self.tuning_config.max_trials // 50,
                verbose=False
            )
            
            # Get best individual
            best_individual = tools.selBest(population, 1)[0]
            best_params = dict(zip(param_names, best_individual))
            best_score = best_individual.fitness.values[0]
            
            return {
                "algorithm": "genetic",
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": self.tuning_config.max_trials,
                "convergence": True,
                "all_scores": [ind.fitness.values[0] for ind in population],
                "all_params": [dict(zip(param_names, ind)) for ind in population]
            }
            
        except ImportError:
            self.logger.error("DEAP not available for genetic optimization")
            raise
            
    def get_tuning_history(self) -> List[Dict[str, Any]]:
        """Get tuning history."""
        return self.tuning_history.copy()
        
    def save_config(self, file_path: Union[str, Path]) -> None:
        """Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
        """
        config_data = {
            "algorithm": self.tuning_config.algorithm,
            "objective": self.tuning_config.objective,
            "max_trials": self.tuning_config.max_trials,
            "timeout_seconds": self.tuning_config.timeout_seconds,
            "n_jobs": self.tuning_config.n_jobs,
            "random_state": self.tuning_config.random_state,
            "early_stopping": self.tuning_config.early_stopping,
            "early_stopping_patience": self.tuning_config.early_stopping_patience,
            "parameter_ranges": {
                name: {
                    "min_value": pr.min_value,
                    "max_value": pr.max_value,
                    "step": pr.step,
                    "parameter_type": pr.parameter_type,
                    "categories": pr.categories,
                    "distribution": pr.distribution,
                    "description": pr.description
                }
                for name, pr in self.tuning_config.parameter_ranges.items()
            },
            "constraints": self.tuning_config.constraints
        }
        
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        self.logger.info(f"Saved configuration to {file_path}") 