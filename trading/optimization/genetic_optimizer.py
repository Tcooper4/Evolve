"""Genetic Algorithm Optimization Method.

This module contains the GeneticAlgorithm optimization method extracted from strategy_optimizer.py.
"""

import logging
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .grid_search_optimizer import OptimizationMethod, OptimizationResult

logger = logging.getLogger(__name__)


class GeneticAlgorithm(OptimizationMethod):
    """Genetic algorithm optimization method."""

    def optimize(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs,
    ) -> OptimizationResult:
        """Run genetic algorithm optimization.

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

        # Get genetic algorithm parameters
        population_size = kwargs.get("population_size", 50)
        n_generations = kwargs.get("n_generations", 100)
        mutation_rate = kwargs.get("mutation_rate", 0.1)
        crossover_rate = kwargs.get("crossover_rate", 0.8)
        elite_size = kwargs.get("elite_size", 5)

        # Initialize population
        population = self._initialize_population(param_space, population_size)

        # Run genetic algorithm
        best_score = float("inf")
        best_params = None
        all_scores = []
        all_params = []
        convergence_history = []

        for generation in range(n_generations):
            # Evaluate population
            scores = []
            for individual in population:
                try:
                    score = objective(individual, data)
                    scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error evaluating individual: {str(e)}")
                    scores.append(float("inf"))

            # Update best solution
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < best_score:
                best_score = scores[min_score_idx]
                best_params = population[min_score_idx].copy()

            # Store results
            all_scores.extend(scores)
            all_params.extend(population)
            convergence_history.append(best_score)

            # Selection
            selected = self._selection(population, scores, elite_size)

            # Crossover
            offspring = self._crossover(selected, crossover_rate, population_size - elite_size)

            # Mutation
            offspring = self._mutation(offspring, mutation_rate, param_space)

            # Update population
            population = selected + offspring

            # Log progress
            if generation % 10 == 0:
                self.logger.info(f"Generation {generation}: Best score = {best_score}")

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_scores=all_scores,
            all_params=all_params,
            optimization_time=optimization_time,
            n_iterations=n_generations,
            convergence_history=convergence_history,
        )

    def _initialize_population(
        self, param_space: Dict[str, Any], population_size: int
    ) -> List[Dict[str, Any]]:
        """Initialize population with random individuals.

        Args:
            param_space: Parameter space
            population_size: Size of population

        Returns:
            List of random parameter combinations
        """
        population = []
        for _ in range(population_size):
            individual = {}
            for param, space in param_space.items():
                if isinstance(space, (list, tuple)):
                    individual[param] = random.choice(space)
                elif isinstance(space, dict):
                    if "start" in space and "end" in space:
                        if isinstance(space["start"], int):
                            individual[param] = random.randint(space["start"], space["end"])
                        else:
                            individual[param] = random.uniform(space["start"], space["end"])
            population.append(individual)
        return population

    def _selection(
        self, population: List[Dict[str, Any]], scores: List[float], elite_size: int
    ) -> List[Dict[str, Any]]:
        """Select individuals using tournament selection.

        Args:
            population: Current population
            scores: Fitness scores
            elite_size: Number of elite individuals to keep

        Returns:
            Selected individuals
        """
        # Keep elite individuals
        elite_indices = np.argsort(scores)[:elite_size]
        selected = [population[i] for i in elite_indices]

        # Tournament selection for the rest
        tournament_size = 3
        while len(selected) < len(population):
            # Select tournament participants
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_scores = [scores[i] for i in tournament_indices]
            
            # Select winner (minimize score)
            winner_idx = tournament_indices[np.argmin(tournament_scores)]
            selected.append(population[winner_idx])

        return selected

    def _crossover(
        self, selected: List[Dict[str, Any]], crossover_rate: float, offspring_size: int
    ) -> List[Dict[str, Any]]:
        """Perform crossover between selected individuals.

        Args:
            selected: Selected individuals
            crossover_rate: Probability of crossover
            offspring_size: Number of offspring to generate

        Returns:
            List of offspring
        """
        offspring = []
        for _ in range(offspring_size):
            if random.random() < crossover_rate and len(selected) >= 2:
                # Select two parents
                parent1, parent2 = random.sample(selected, 2)
                
                # Create child by combining parameters
                child = {}
                for param in parent1.keys():
                    if random.random() < 0.5:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
                
                offspring.append(child)
            else:
                # No crossover, copy parent
                offspring.append(random.choice(selected).copy())
        
        return offspring

    def _mutation(
        self,
        offspring: List[Dict[str, Any]],
        mutation_rate: float,
        param_space: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Perform mutation on offspring.

        Args:
            offspring: Offspring individuals
            mutation_rate: Probability of mutation
            param_space: Parameter space

        Returns:
            Mutated offspring
        """
        for individual in offspring:
            for param, space in param_space.items():
                if random.random() < mutation_rate:
                    if isinstance(space, (list, tuple)):
                        individual[param] = random.choice(space)
                    elif isinstance(space, dict):
                        if "start" in space and "end" in space:
                            if isinstance(space["start"], int):
                                individual[param] = random.randint(space["start"], space["end"])
                            else:
                                individual[param] = random.uniform(space["start"], space["end"])
        return offspring
