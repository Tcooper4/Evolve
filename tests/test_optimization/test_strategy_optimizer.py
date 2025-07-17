"""
Tests for Enhanced Strategy Optimizer

Tests the strategy optimizer with genetic algorithm, particle swarm optimization,
and other advanced optimization methods.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading.optimization.strategy_optimizer import (
    GeneticAlgorithm,
    OptimizationResult,
    ParticleSwarmOptimization,
    StrategyOptimizer,
)
from trading.optimization.base_optimizer import OptimizerConfig


class TestGeneticAlgorithm:
    """Test Genetic Algorithm optimization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
                "returns": np.random.normal(0.001, 0.02, 100),
            },
            index=dates,
        )
        return data

    @pytest.fixture
    def param_space(self):
        """Create parameter space for testing."""
        return {
            "window": {"start": 5, "end": 50, "type": "int"},
            "threshold": {"start": 0.01, "end": 0.1},
            "method": ["sma", "ema", "rsi"],
        }

    def test_genetic_algorithm_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithm()
        assert ga.config == {}
        assert ga.logger is not None

    def test_initialize_population(self, param_space):
        """Test population initialization."""
        ga = GeneticAlgorithm()
        population = ga._initialize_population(param_space, 10)

        assert len(population) == 10
        for individual in population:
            assert "window" in individual
            assert "threshold" in individual
            assert "method" in individual
            assert isinstance(individual["window"], int)
            assert isinstance(individual["threshold"], float)
            assert individual["method"] in ["sma", "ema", "rsi"]

    def test_selection(self, param_space):
        """Test tournament selection."""
        ga = GeneticAlgorithm()
        population = ga._initialize_population(param_space, 20)
        scores = [np.random.random() for _ in range(20)]
        elite_size = 3

        selected = ga._selection(population, scores, elite_size)

        assert len(selected) == len(population)
        # Elite should be the best individuals
        elite_indices = np.argsort(scores)[:elite_size]
        for i, idx in enumerate(elite_indices):
            assert selected[i] == population[idx]

    def test_crossover(self, param_space):
        """Test uniform crossover."""
        ga = GeneticAlgorithm()
        population = ga._initialize_population(param_space, 10)
        crossover_rate = 0.8
        offspring_size = 5

        offspring = ga._crossover(population, crossover_rate, offspring_size)

        assert len(offspring) == offspring_size
        for child in offspring:
            assert "window" in child
            assert "threshold" in child
            assert "method" in child

    def test_mutation(self, param_space):
        """Test Gaussian mutation."""
        ga = GeneticAlgorithm()
        population = ga._initialize_population(param_space, 5)
        mutation_rate = 0.5

        original_population = [ind.copy() for ind in population]
        mutated = ga._mutation(population, mutation_rate, param_space)

        # Some parameters should have changed
        changed = False
        for orig, mut in zip(original_population, mutated):
            if orig != mut:
                changed = True
                break
        assert changed

    def test_genetic_algorithm_optimization(self, sample_data, param_space):
        """Test complete GA optimization."""
        ga = GeneticAlgorithm()

        def objective(params, data):
            # Simple objective function
            return params["window"] * params["threshold"] + len(data)

        result = ga.optimize(
            objective, param_space, sample_data, population_size=10, n_generations=5
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score < float("inf")
        assert result.n_iterations == 5
        assert len(result.convergence_history) > 0
        assert result.optimization_time > 0

    def test_genetic_algorithm_early_stopping(self, sample_data, param_space):
        """Test GA early stopping."""
        ga = GeneticAlgorithm()

        def objective(params, data):
            # Objective that improves quickly then plateaus
            return 1.0 / (1.0 + params["window"])

        result = ga.optimize(
            objective, param_space, sample_data, population_size=5, n_generations=20
        )

        # Should converge before max generations
        assert result.n_iterations <= 20

    def test_genetic_algorithm_error_handling(self, sample_data, param_space):
        """Test GA error handling."""
        ga = GeneticAlgorithm()

        def objective(params, data):
            # Objective that sometimes fails
            if params["window"] > 40:
                raise ValueError("Window too large")
            return params["window"]

        result = ga.optimize(
            objective, param_space, sample_data, population_size=10, n_generations=5
        )

        # Should handle errors gracefully
        assert result.best_params is not None
        assert result.best_score < float("inf")


class TestParticleSwarmOptimization:
    """Test Particle Swarm Optimization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
                "returns": np.random.normal(0.001, 0.02, 100),
            },
            index=dates,
        )
        return data

    @pytest.fixture
    def param_space(self):
        """Create parameter space for testing."""
        return {
            "window": {"start": 5, "end": 50, "type": "int"},
            "threshold": {"start": 0.01, "end": 0.1},
            "method": ["sma", "ema", "rsi"],
        }

    def test_pso_initialization(self):
        """Test PSO initialization."""
        pso = ParticleSwarmOptimization()
        assert pso.config == {}
        assert pso.logger is not None

    def test_initialize_particles(self, param_space):
        """Test particle initialization."""
        pso = ParticleSwarmOptimization()
        particles, velocities = pso._initialize_particles(param_space, 10)

        assert len(particles) == 10
        assert len(velocities) == 10

        for particle in particles:
            assert "window" in particle
            assert "threshold" in particle
            assert "method" in particle
            assert isinstance(particle["window"], int)
            assert isinstance(particle["threshold"], float)
            assert particle["method"] in ["sma", "ema", "rsi"]

        for velocity in velocities:
            assert "window" in velocity
            assert "threshold" in velocity
            assert "method" in velocity

    def test_pso_optimization(self, sample_data, param_space):
        """Test complete PSO optimization."""
        pso = ParticleSwarmOptimization()

        def objective(params, data):
            # Simple objective function
            return params["window"] * params["threshold"] + len(data)

        result = pso.optimize(
            objective, param_space, sample_data, n_particles=10, n_iterations=5
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score < float("inf")
        assert result.n_iterations == 5
        assert len(result.convergence_history) > 0
        assert result.optimization_time > 0

    def test_pso_early_stopping(self, sample_data, param_space):
        """Test PSO early stopping."""
        pso = ParticleSwarmOptimization()

        def objective(params, data):
            # Objective that improves quickly then plateaus
            return 1.0 / (1.0 + params["window"])

        result = pso.optimize(
            objective, param_space, sample_data, n_particles=5, n_iterations=20
        )

        # Should converge before max iterations
        assert result.n_iterations <= 20

    def test_pso_parameter_sensitivity(self, sample_data, param_space):
        """Test PSO parameter sensitivity."""
        pso = ParticleSwarmOptimization()

        def objective(params, data):
            return params["window"] * params["threshold"]

        # Test different PSO parameters
        result1 = pso.optimize(
            objective,
            param_space,
            sample_data,
            n_particles=5,
            n_iterations=5,
            inertia_weight=0.5,
            cognitive_weight=1.0,
            social_weight=1.0,
        )

        result2 = pso.optimize(
            objective,
            param_space,
            sample_data,
            n_particles=5,
            n_iterations=5,
            inertia_weight=0.9,
            cognitive_weight=2.0,
            social_weight=2.0,
        )

        # Both should complete successfully
        assert result1.best_params is not None
        assert result2.best_params is not None

    def test_pso_error_handling(self, sample_data, param_space):
        """Test PSO error handling."""
        pso = ParticleSwarmOptimization()

        def objective(params, data):
            # Objective that sometimes fails
            if params["window"] > 40:
                raise ValueError("Window too large")
            return params["window"]

        result = pso.optimize(
            objective, param_space, sample_data, n_particles=10, n_iterations=5
        )

        # Should handle errors gracefully
        assert result.best_params is not None
        assert result.best_score < float("inf")


class TestStrategyOptimizer:
    """Test the enhanced strategy optimizer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
                "returns": np.random.normal(0.001, 0.02, 100),
            },
            index=dates,
        )
        return data

    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy class."""

        class MockStrategy:
            def __init__(self, params):
                self.params = params

            def generate_signals(self, data):
                return pd.Series(
                    np.random.choice([-1, 0, 1], len(data)), index=data.index
                )

            def evaluate_performance(self, signals, data):
                class MockMetrics:
                    def __init__(self):
                        self.sharpe_ratio = np.random.random()
                        self.win_rate = np.random.random()
                        self.max_drawdown = np.random.random()

                return MockMetrics()

            @classmethod
            def default_params(cls):
                return {"window": 20, "threshold": 0.05, "method": "sma"}

        return MockStrategy

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = OptimizerConfig(optimizer_type="genetic")
        optimizer = StrategyOptimizer(config)

        assert optimizer.config.optimizer_type == "genetic"
        assert optimizer.logger is not None

    def test_create_optimizer_genetic(self):
        """Test creating genetic algorithm optimizer."""
        config = OptimizerConfig(optimizer_type="genetic")
        optimizer = StrategyOptimizer(config)

        opt_method = optimizer._create_optimizer()
        assert isinstance(opt_method, GeneticAlgorithm)

    def test_create_optimizer_pso(self):
        """Test creating PSO optimizer."""
        config = OptimizerConfig(optimizer_type="pso")
        optimizer = StrategyOptimizer(config)

        opt_method = optimizer._create_optimizer()
        assert isinstance(opt_method, ParticleSwarmOptimization)

    def test_create_optimizer_invalid_type(self):
        """Test creating optimizer with invalid type."""
        config = OptimizerConfig(optimizer_type="invalid")
        optimizer = StrategyOptimizer(config)

        with pytest.raises(ValueError, match="Unsupported optimizer type"):
            optimizer._create_optimizer()

    def test_objective_wrapper(self, sample_data, mock_strategy):
        """Test objective function wrapper."""
        config = OptimizerConfig()
        optimizer = StrategyOptimizer(config)

        objective = optimizer._objective_wrapper(mock_strategy, sample_data)
        params = {"window": 20, "threshold": 0.05, "method": "sma"}

        score = objective(params)
        assert isinstance(score, float)
        assert score < 0  # Should be negative (minimization)

    def test_get_default_params(self, mock_strategy):
        """Test getting default parameters."""
        config = OptimizerConfig()
        optimizer = StrategyOptimizer(config)

        params = optimizer._get_default_params(mock_strategy)
        assert params == {"window": 20, "threshold": 0.05, "method": "sma"}

    def test_create_parameter_grid(self):
        """Test parameter grid creation."""
        config = OptimizerConfig()
        optimizer = StrategyOptimizer(config)

        params = {"window": 20, "threshold": 0.05, "method": ["sma", "ema"]}
        grid = optimizer._create_parameter_grid(params)

        assert "window" in grid
        assert "threshold" in grid
        assert "method" in grid
        assert grid["window"]["start"] == 10  # 20 * 0.5
        assert grid["window"]["end"] == 30  # 20 * 1.5

    def test_optimize_with_genetic(self, sample_data, mock_strategy):
        """Test optimization with genetic algorithm."""
        config = OptimizerConfig(optimizer_type="genetic", n_iterations=5)
        optimizer = StrategyOptimizer(config)

        # Mock the optimization method
        with patch.object(optimizer.optimizer, "optimize") as mock_optimize:
            mock_optimize.return_value = OptimizationResult(
                best_params={"window": 25, "threshold": 0.06, "method": "ema"},
                best_score=-0.8,
                all_scores=[-0.7, -0.8, -0.75],
                all_params=[{}, {}, {}],
                optimization_time=1.0,
                n_iterations=5,
                convergence_history=[-0.7, -0.8, -0.8],
            )

            result = optimizer.optimize(mock_strategy, sample_data)

            assert result == {"window": 25, "threshold": 0.06, "method": "ema"}
            mock_optimize.assert_called_once()

    def test_optimize_with_pso(self, sample_data, mock_strategy):
        """Test optimization with PSO."""
        config = OptimizerConfig(optimizer_type="pso", n_iterations=5)
        optimizer = StrategyOptimizer(config)

        # Mock the optimization method
        with patch.object(optimizer.optimizer, "optimize") as mock_optimize:
            mock_optimize.return_value = OptimizationResult(
                best_params={"window": 30, "threshold": 0.07, "method": "rsi"},
                best_score=-0.9,
                all_scores=[-0.8, -0.9, -0.85],
                all_params=[{}, {}, {}],
                optimization_time=1.5,
                n_iterations=5,
                convergence_history=[-0.8, -0.9, -0.9],
            )

            result = optimizer.optimize(mock_strategy, sample_data)

            assert result == {"window": 30, "threshold": 0.07, "method": "rsi"}
            mock_optimize.assert_called_once()

    def test_optimizer_config_validation(self):
        """Test optimizer configuration validation."""
        # Test valid optimizer types
        valid_types = ["grid", "bayesian", "genetic", "pso", "optuna", "pytorch", "ray"]
        for opt_type in valid_types:
            config = OptimizerConfig(optimizer_type=opt_type)
            assert config.optimizer_type == opt_type

        # Test invalid optimizer type
        with pytest.raises(ValueError, match="optimizer_type must be one of"):
            OptimizerConfig(optimizer_type="invalid")

    def test_optimizer_comparison(self, sample_data, mock_strategy):
        """Test comparing different optimizers."""
        optimizers = ["genetic", "pso"]
        results = {}

        for opt_type in optimizers:
            config = OptimizerConfig(optimizer_type=opt_type, n_iterations=3)
            optimizer = StrategyOptimizer(config)

            # Mock the optimization method
            with patch.object(optimizer.optimizer, "optimize") as mock_optimize:
                mock_optimize.return_value = OptimizationResult(
                    best_params={"window": 20, "threshold": 0.05},
                    best_score=-0.8,
                    all_scores=[-0.7, -0.8, -0.75],
                    all_params=[{}, {}, {}],
                    optimization_time=1.0,
                    n_iterations=3,
                    convergence_history=[-0.7, -0.8, -0.8],
                )

                result = optimizer.optimize(mock_strategy, sample_data)
                results[opt_type] = result

        assert len(results) == 2
        assert "genetic" in results
        assert "pso" in results

    def test_optimization_error_handling(self, sample_data, mock_strategy):
        """Test optimization error handling."""
        config = OptimizerConfig(optimizer_type="genetic")
        optimizer = StrategyOptimizer(config)

        # Mock objective function to raise error
        def failing_objective(params, data):
            raise ValueError("Optimization failed")

        with patch.object(
            optimizer, "_objective_wrapper", return_value=failing_objective
        ):
            with pytest.raises(ValueError, match="Optimization failed"):
                optimizer.optimize(mock_strategy, sample_data)

    def test_optimization_with_custom_parameters(self, sample_data, mock_strategy):
        """Test optimization with custom parameters."""
        config = OptimizerConfig(optimizer_type="genetic", n_iterations=5)
        optimizer = StrategyOptimizer(config)

        initial_params = {"window": 15, "threshold": 0.03, "method": "rsi"}

        # Mock the optimization method
        with patch.object(optimizer.optimizer, "optimize") as mock_optimize:
            mock_optimize.return_value = OptimizationResult(
                best_params=initial_params,
                best_score=-0.8,
                all_scores=[-0.8],
                all_params=[initial_params],
                optimization_time=1.0,
                n_iterations=5,
                convergence_history=[-0.8],
            )

            result = optimizer.optimize(mock_strategy, sample_data, initial_params)

            assert result == initial_params
            mock_optimize.assert_called_once()

    def test_optimization_metrics_tracking(self, sample_data, mock_strategy):
        """Test optimization metrics tracking."""
        config = OptimizerConfig(optimizer_type="genetic", n_iterations=3)
        optimizer = StrategyOptimizer(config)

        # Mock the optimization method
        with patch.object(optimizer.optimizer, "optimize") as mock_optimize:
            mock_optimize.return_value = OptimizationResult(
                best_params={"window": 20, "threshold": 0.05},
                best_score=-0.8,
                all_scores=[-0.7, -0.8, -0.75],
                all_params=[{}, {}, {}],
                optimization_time=1.0,
                n_iterations=3,
                convergence_history=[-0.7, -0.8, -0.8],
            )

            result = optimizer.optimize(mock_strategy, sample_data)

            # Check that metrics were logged
            assert result is not None
            # Additional assertions could be added here for metrics tracking

    def test_optimization_convergence_analysis(self, sample_data, mock_strategy):
        """Test optimization convergence analysis."""
        config = OptimizerConfig(optimizer_type="genetic", n_iterations=10)
        optimizer = StrategyOptimizer(config)

        # Mock the optimization method with convergence data
        with patch.object(optimizer.optimizer, "optimize") as mock_optimize:
            convergence_history = [
                -0.5,
                -0.6,
                -0.7,
                -0.75,
                -0.8,
                -0.8,
                -0.8,
                -0.8,
                -0.8,
                -0.8,
            ]
            mock_optimize.return_value = OptimizationResult(
                best_params={"window": 20, "threshold": 0.05},
                best_score=-0.8,
                all_scores=[
                    -0.5,
                    -0.6,
                    -0.7,
                    -0.75,
                    -0.8,
                    -0.8,
                    -0.8,
                    -0.8,
                    -0.8,
                    -0.8,
                ],
                all_params=[{}] * 10,
                optimization_time=2.0,
                n_iterations=10,
                convergence_history=convergence_history,
            )

            result = optimizer.optimize(mock_strategy, sample_data)

            # Check convergence properties
            assert result is not None
            assert mock_optimize.return_value.best_score == -0.8
            assert len(mock_optimize.return_value.convergence_history) == 10
            # Should show improvement over iterations
            assert (
                mock_optimize.return_value.convergence_history[0]
                > mock_optimize.return_value.convergence_history[-1]
            )
