"""Particle Swarm Optimization Method.

This module contains the ParticleSwarmOptimization method extracted from strategy_optimizer.py.
"""

import logging
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .grid_search_optimizer import OptimizationMethod, OptimizationResult

logger = logging.getLogger(__name__)


class ParticleSwarmOptimization(OptimizationMethod):
    """Particle swarm optimization method."""

    def optimize(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        **kwargs,
    ) -> OptimizationResult:
        """Run particle swarm optimization.

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

        # Get PSO parameters
        n_particles = kwargs.get("n_particles", 30)
        n_iterations = kwargs.get("n_iterations", 100)
        w = kwargs.get("w", 0.7)  # Inertia weight
        c1 = kwargs.get("c1", 1.5)  # Cognitive parameter
        c2 = kwargs.get("c2", 1.5)  # Social parameter

        # Initialize particles
        particles, velocities = self._initialize_particles(param_space, n_particles)

        # Initialize best positions and scores
        best_positions = particles.copy()
        best_scores = [float("inf")] * n_particles
        global_best_position = None
        global_best_score = float("inf")

        # Run PSO
        all_scores = []
        all_params = []
        convergence_history = []

        for iteration in range(n_iterations):
            # Evaluate particles
            scores = []
            for i, particle in enumerate(particles):
                try:
                    score = objective(particle, data)
                    scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error evaluating particle: {str(e)}")
                    score = float("inf")
                    scores.append(score)

                # Update personal best
                if score < best_scores[i]:
                    best_scores[i] = score
                    best_positions[i] = particle.copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particle.copy()

            # Store results
            all_scores.extend(scores)
            all_params.extend(particles)
            convergence_history.append(global_best_score)

            # Update velocities and positions
            for i in range(n_particles):
                for param in param_space.keys():
                    # Update velocity
                    r1, r2 = random.random(), random.random()
                    cognitive_velocity = c1 * r1 * (best_positions[i][param] - particles[i][param])
                    social_velocity = c2 * r2 * (global_best_position[param] - particles[i][param])
                    velocities[i][param] = w * velocities[i][param] + cognitive_velocity + social_velocity

                    # Update position
                    particles[i][param] += velocities[i][param]

                    # Clamp to bounds
                    space = param_space[param]
                    if isinstance(space, (list, tuple)):
                        particles[i][param] = random.choice(space)
                    elif isinstance(space, dict):
                        if "start" in space and "end" in space:
                            particles[i][param] = np.clip(
                                particles[i][param], space["start"], space["end"]
                            )

            # Log progress
            if iteration % 10 == 0:
                self.logger.info(f"Iteration {iteration}: Best score = {global_best_score}")

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=global_best_position or {},
            best_score=global_best_score,
            all_scores=all_scores,
            all_params=all_params,
            optimization_time=optimization_time,
            n_iterations=n_iterations,
            convergence_history=convergence_history,
        )

    def _initialize_particles(
        self, param_space: Dict[str, Any], n_particles: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Initialize particles and velocities.

        Args:
            param_space: Parameter space
            n_particles: Number of particles

        Returns:
            Tuple of (particles, velocities)
        """
        particles = []
        velocities = []

        for _ in range(n_particles):
            particle = {}
            velocity = {}

            for param, space in param_space.items():
                if isinstance(space, (list, tuple)):
                    particle[param] = random.choice(space)
                    velocity[param] = 0.0
                elif isinstance(space, dict):
                    if "start" in space and "end" in space:
                        particle[param] = random.uniform(space["start"], space["end"])
                        # Initialize velocity as a fraction of the parameter range
                        param_range = space["end"] - space["start"]
                        velocity[param] = random.uniform(-0.1 * param_range, 0.1 * param_range)

            particles.append(particle)
            velocities.append(velocity)

        return particles, velocities 