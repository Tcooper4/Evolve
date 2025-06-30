"""Genetic Algorithm Optimizer for Trading Strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import DEAP with fallback
try:
    import deap
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    deap = base = creator = tools = algorithms = None

from .base_optimizer import BaseOptimizer, OptimizationResult
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm optimizer for trading strategies."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_type: str,
        verbose: bool = False,
        n_jobs: int = -1,
        population_size: int = 100,
        generations: int = 50,
        mutation_prob: float = 0.2,
        crossover_prob: float = 0.7
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    ):
        """Initialize genetic optimizer.
        
        Args:
            data: Market data
            strategy_type: Type of strategy to optimize
            verbose: Verbose output
            n_jobs: Number of parallel jobs
            population_size: Population size
            generations: Number of generations
            mutation_prob: Mutation probability
            crossover_prob: Crossover probability
        """
        super().__init__(data, strategy_type, verbose, n_jobs)
        
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP not available. Please install deap.")
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        
        # Initialize DEAP creator
        self._initialize_deap()
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.stats = None
        self.hof = None
    
    def _initialize_deap(self):
        """Initialize DEAP creator classes."""
        # Clear existing creator classes to avoid conflicts
        if hasattr(creator, 'Individual'):
            del creator.Individual
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        
        # Create fitness class
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Create individual class
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def optimize(
        self,
        param_space: Dict[str, Union[List, Tuple]],
        objective: Union[str, List[str]],
        n_trials: int = 100,
        **kwargs
    ) -> List[OptimizationResult]:
        """Run genetic algorithm optimization.
        
        Args:
            param_space: Parameter space definition
            objective: Optimization objective(s)
            n_trials: Number of trials (ignored for GA)
            **kwargs: Additional arguments
            
        Returns:
            List of optimization results
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP not available")
        
        self.param_space = param_space
        
        # Register parameter generators
        import random
        for param_name, param_range in param_space.items():
            if isinstance(param_range[0], int):
                self.toolbox.register(
                    f"attr_{param_name}",
                    random.randint,
                    param_range[0],
                    param_range[1]
                )
            else:
                self.toolbox.register(
                    f"attr_{param_name}",
                    random.uniform,
                    param_range[0],
                    param_range[1]
                )
        
        # Register individual and population
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (getattr(self.toolbox, f"attr_{param}") for param in param_space.keys()),
            n=1
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Initialize statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        # Initialize hall of fame
        self.hof = tools.HallOfFame(maxsize=10)
        
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Run evolution
        pop, logbook = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=self.stats,
            halloffame=self.hof,
            verbose=self.verbose
        )
        
        return {'success': True, 'result': self.get_all_results(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _evaluate_individual(
        self,
        individual: List[float]
    ) -> Tuple[float]:
        """Evaluate individual fitness.
        
        Args:
            individual: Individual parameters
            
        Returns:
            Tuple of fitness value
        """
        # Convert individual to parameter dictionary
        params = dict(zip(self.param_space.keys(), individual))
        
        # Run strategy
        returns, signals, equity_curve = self._run_strategy(params)
        
        # Calculate metrics
        metrics = self.calculate_metrics(returns, signals, equity_curve)
        
        # Create result
        result = OptimizationResult(
            parameters=params,
            metrics=metrics,
            returns=returns,
            signals=signals,
            equity_curve=equity_curve,
            drawdown=(equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max(),
            timestamp=datetime.now(),
            optimization_type='genetic'
        )
        
        # Log result
        self.log_result(result)
        
        # Return fitness (Sharpe ratio)
        return {'success': True, 'result': (metrics['sharpe_ratio'],), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _run_strategy(self, params: Dict[str, float]) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Run strategy with given parameters.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Tuple of (returns, signals, equity_curve)
        """
        # This should be implemented by strategy-specific optimizers
        raise NotImplementedError
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def plot_results(
        self,
        plot_type: str = 'all',
        **kwargs
    ) -> Union[go.Figure, List[go.Figure]]:
        """Plot optimization results.
        
        Args:
            plot_type: Type of plot ('all', 'evolution', 'fitness')
            **kwargs: Additional plot arguments
            
        Returns:
            Plotly figure(s)
        """
        if not self.stats:
            raise ValueError("No optimization statistics found")
        
        plots = []
        
        if plot_type in ['all', 'evolution']:
            # Plot evolution of fitness
            fig = go.Figure()
            
            for stat in ['avg', 'min', 'max']:
                values = [entry[stat] for entry in self.stats]
                fig.add_trace(go.Scatter(
                    y=values,
                    name=stat.capitalize()
                ))
            
            fig.update_layout(
                title="Evolution of Fitness",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                template="plotly_white"
            )
            
            plots.append(fig)
        
        if plot_type in ['all', 'fitness']:
            # Plot fitness distribution
            fig = go.Figure()
            
            fitness_values = [ind.fitness.values[0] for ind in self.hof]
            fig.add_trace(go.Histogram(
                x=fitness_values,
                name="Fitness Distribution"
            ))
            
            fig.update_layout(
                title="Fitness Distribution",
                xaxis_title="Fitness",
                yaxis_title="Count",
                template="plotly_white"
            )
            
            plots.append(fig)
        
        return {'success': True, 'result': plots[0] if len(plots) == 1 else plots, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_best_individuals(self, n: int = 1) -> List[Any]:
        """Get best individuals from hall of fame.
        
        Args:
            n: Number of best individuals to return
            
        Returns:
            List of best individuals
        """
        if not self.hof:
            raise ValueError("No hall of fame found")
        
        return {'success': True, 'result': self.hof[:n], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_best_parameters(self) -> Dict[str, float]:
        """Get best parameters from hall of fame.
        
        Returns:
            Dictionary of best parameters
        """
        if not self.hof:
            raise ValueError("No hall of fame found")
        
        best_individual = self.hof[0]
        return dict(zip(self.param_space.keys(), best_individual))

def create_genetic_optimizer(data: pd.DataFrame, 
                           strategy_type: str,
                           config: Optional[Dict[str, Any]] = None) -> GeneticOptimizer:
    """Create genetic optimizer.
    
    Args:
        data: Market data
        strategy_type: Strategy type
        config: Configuration dictionary
        
    Returns:
        GeneticOptimizer instance
    """
    if not DEAP_AVAILABLE:
        raise ImportError("DEAP not available. Please install deap.")
    
    config = config or {}
    population_size = config.get('population_size', 100)
    generations = config.get('generations', 50)
    mutation_prob = config.get('mutation_prob', 0.2)
    crossover_prob = config.get('crossover_prob', 0.7)
    verbose = config.get('verbose', False)
    
    return {'success': True, 'result': GeneticOptimizer(, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        data=data,
        strategy_type=strategy_type,
        population_size=population_size,
        generations=generations,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        verbose=verbose
    ) 