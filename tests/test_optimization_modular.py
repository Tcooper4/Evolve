"""Tests for modularized optimization components."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from trading.optimization import (
    StrategyOptimizer,
    GridSearch,
    BayesianOptimization,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    RayTuneOptimization,
    OptimizationResult,
)


class TestGridSearch:
    """Test GridSearch optimization method."""

    def setup_method(self):
        """Setup test method."""
        self.optimizer = GridSearch()
        self.param_space = {
            "param1": [1, 2, 3],
            "param2": {"start": 0.1, "end": 0.5, "n_points": 3},
        }
        self.data = pd.DataFrame({
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000, 10000, 100),
        })

    def test_optimize_basic(self):
        """Test basic optimization."""
        def objective(params, data):
            return params["param1"] + params["param2"]

        result = self.optimizer.optimize(objective, self.param_space, self.data)

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score is not None
        assert result.optimization_time > 0
        assert result.n_iterations > 0

    def test_optimize_with_max_points(self):
        """Test optimization with max points limit."""
        def objective(params, data):
            return params["param1"] + params["param2"]

        result = self.optimizer.optimize(
            objective, self.param_space, self.data, max_points=5
        )

        assert result.n_iterations <= 5

    def test_validate_param_space(self):
        """Test parameter space validation."""
        # Test empty param space
        with pytest.raises(ValueError):
            self.optimizer._validate_param_space({})

        # Test invalid range
        invalid_space = {"param": {"start": 10, "end": 5}}
        with pytest.raises(ValueError):
            self.optimizer._validate_param_space(invalid_space)

    def test_early_stopping(self):
        """Test early stopping functionality."""
        scores = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
        assert self.optimizer._check_early_stopping(scores, patience=3, min_delta=0.01)


class TestBayesianOptimization:
    """Test BayesianOptimization method."""

    def setup_method(self):
        """Setup test method."""
        self.optimizer = BayesianOptimization()
        self.param_space = {
            "param1": {"start": 0.1, "end": 1.0},
            "param2": {"start": 1, "end": 10},
        }
        self.data = pd.DataFrame({
            "Close": np.random.randn(100).cumsum() + 100,
        })

    @patch("trading.optimization.bayesian_optimizer.gp_minimize")
    def test_optimize_basic(self, mock_gp_minimize):
        """Test basic Bayesian optimization."""
        # Mock the optimization result
        mock_result = Mock()
        mock_result.x = [0.5, 5]
        mock_result.fun = 0.1
        mock_result.func_vals = [-0.1, -0.2, -0.15]
        mock_result.x_iters = [[0.3, 3], [0.7, 7], [0.5, 5]]
        mock_gp_minimize.return_value = mock_result

        def objective(params, data):
            return params["param1"] + params["param2"]

        result = self.optimizer.optimize(objective, self.param_space, self.data)

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score is not None

    def test_optimize_missing_dependency(self):
        """Test optimization without scikit-optimize."""
        with patch("trading.optimization.bayesian_optimizer.gp_minimize", side_effect=ImportError):
            def objective(params, data):
                return params["param1"] + params["param2"]

            with pytest.raises(ImportError):
                self.optimizer.optimize(objective, self.param_space, self.data)


class TestGeneticAlgorithm:
    """Test GeneticAlgorithm method."""

    def setup_method(self):
        """Setup test method."""
        self.optimizer = GeneticAlgorithm()
        self.param_space = {
            "param1": [1, 2, 3, 4, 5],
            "param2": {"start": 0.1, "end": 1.0},
        }
        self.data = pd.DataFrame({
            "Close": np.random.randn(100).cumsum() + 100,
        })

    def test_optimize_basic(self):
        """Test basic genetic algorithm optimization."""
        def objective(params, data):
            return params["param1"] + params["param2"]

        result = self.optimizer.optimize(
            objective, 
            self.param_space, 
            self.data,
            population_size=10,
            n_generations=5,
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score is not None
        assert result.n_iterations == 5

    def test_initialize_population(self):
        """Test population initialization."""
        population = self.optimizer._initialize_population(self.param_space, 5)
        
        assert len(population) == 5
        for individual in population:
            assert "param1" in individual
            assert "param2" in individual
            assert individual["param1"] in [1, 2, 3, 4, 5]
            assert 0.1 <= individual["param2"] <= 1.0

    def test_selection(self):
        """Test selection process."""
        population = [
            {"param1": 1, "param2": 0.1},
            {"param1": 2, "param2": 0.2},
            {"param1": 3, "param2": 0.3},
        ]
        scores = [0.3, 0.2, 0.1]  # Lower is better
        
        selected = self.optimizer._selection(population, scores, elite_size=1)
        
        assert len(selected) == len(population)
        # Best individual should be in elite
        assert selected[0]["param1"] == 3


class TestParticleSwarmOptimization:
    """Test ParticleSwarmOptimization method."""

    def setup_method(self):
        """Setup test method."""
        self.optimizer = ParticleSwarmOptimization()
        self.param_space = {
            "param1": {"start": 0.1, "end": 1.0},
            "param2": {"start": 1, "end": 10},
        }
        self.data = pd.DataFrame({
            "Close": np.random.randn(100).cumsum() + 100,
        })

    def test_optimize_basic(self):
        """Test basic PSO optimization."""
        def objective(params, data):
            return params["param1"] + params["param2"]

        result = self.optimizer.optimize(
            objective,
            self.param_space,
            self.data,
            n_particles=5,
            n_iterations=3,
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score is not None
        assert result.n_iterations == 3

    def test_initialize_particles(self):
        """Test particle initialization."""
        particles, velocities = self.optimizer._initialize_particles(self.param_space, 3)
        
        assert len(particles) == 3
        assert len(velocities) == 3
        
        for particle in particles:
            assert 0.1 <= particle["param1"] <= 1.0
            assert 1 <= particle["param2"] <= 10


class TestRayTuneOptimization:
    """Test RayTuneOptimization method."""

    def setup_method(self):
        """Setup test method."""
        self.optimizer = RayTuneOptimization()
        self.param_space = {
            "param1": {"start": 0.1, "end": 1.0},
            "param2": {"start": 1, "end": 10},
        }
        self.data = pd.DataFrame({
            "Close": np.random.randn(100).cumsum() + 100,
        })

    @patch("trading.optimization.ray_optimizer.tune.run")
    def test_optimize_basic(self, mock_tune_run):
        """Test basic Ray Tune optimization."""
        # Mock the analysis result
        mock_analysis = Mock()
        mock_best_trial = Mock()
        mock_best_trial.config = {"param1": 0.5, "param2": 5}
        mock_best_trial.last_result = {"score": 0.1}
        mock_analysis.get_best_trial.return_value = mock_best_trial
        mock_analysis.trials = [mock_best_trial]
        mock_tune_run.return_value = mock_analysis

        def objective(params, data):
            return params["param1"] + params["param2"]

        result = self.optimizer.optimize(objective, self.param_space, self.data)

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score is not None

    def test_optimize_missing_dependency(self):
        """Test optimization without Ray Tune."""
        with patch("trading.optimization.ray_optimizer.tune", side_effect=ImportError):
            def objective(params, data):
                return params["param1"] + params["param2"]

            with pytest.raises(ImportError):
                self.optimizer.optimize(objective, self.param_space, self.data)


class TestStrategyOptimizer:
    """Test StrategyOptimizer orchestrator."""

    def setup_method(self):
        """Setup test method."""
        self.optimizer = StrategyOptimizer()
        self.param_space = {
            "param1": [1, 2, 3],
            "param2": {"start": 0.1, "end": 0.5},
        }
        self.data = pd.DataFrame({
            "Close": np.random.randn(100).cumsum() + 100,
        })

    def test_optimize_single_method(self):
        """Test optimization with single method."""
        def objective(params, data):
            return params["param1"] + params["param2"]

        result = self.optimizer.optimize(
            objective, self.param_space, self.data, method="grid_search"
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None

    def test_optimize_invalid_method(self):
        """Test optimization with invalid method."""
        def objective(params, data):
            return params["param1"] + params["param2"]

        with pytest.raises(ValueError):
            self.optimizer.optimize(
                objective, self.param_space, self.data, method="invalid_method"
            )

    def test_optimize_multiple_methods(self):
        """Test optimization with multiple methods."""
        def objective(params, data):
            return params["param1"] + params["param2"]

        results = self.optimizer.optimize_multiple_methods(
            objective, self.param_space, self.data, methods=["grid_search", "genetic"]
        )

        assert isinstance(results, dict)
        assert "grid_search" in results
        assert "genetic" in results

    def test_get_best_result(self):
        """Test getting best result from multiple methods."""
        # Create mock results
        result1 = OptimizationResult(
            best_params={"param1": 1},
            best_score=0.5,
            all_scores=[0.5, 0.6],
            all_params=[{"param1": 1}, {"param1": 2}],
            optimization_time=1.0,
            n_iterations=2,
            convergence_history=[0.6, 0.5],
        )
        
        result2 = OptimizationResult(
            best_params={"param1": 2},
            best_score=0.3,
            all_scores=[0.3, 0.4],
            all_params=[{"param1": 2}, {"param1": 3}],
            optimization_time=1.0,
            n_iterations=2,
            convergence_history=[0.4, 0.3],
        )

        results = {"method1": result1, "method2": result2}
        best_method, best_result = self.optimizer.get_best_result(results)

        assert best_method == "method2"  # Lower score is better
        assert best_result == result2

    def test_compare_methods(self):
        """Test method comparison."""
        # Create mock results
        result1 = OptimizationResult(
            best_params={"param1": 1},
            best_score=0.5,
            all_scores=[0.5, 0.6],
            all_params=[{"param1": 1}, {"param1": 2}],
            optimization_time=1.0,
            n_iterations=2,
            convergence_history=[0.6, 0.5],
        )
        
        result2 = OptimizationResult(
            best_params={"param1": 2},
            best_score=0.3,
            all_scores=[0.3, 0.4],
            all_params=[{"param1": 2}, {"param1": 3}],
            optimization_time=2.0,
            n_iterations=2,
            convergence_history=[0.4, 0.3],
        )

        results = {"method1": result1, "method2": result2}
        comparison = self.optimizer.compare_methods(results)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "method" in comparison.columns
        assert "best_score" in comparison.columns

    def test_get_available_methods(self):
        """Test getting available methods."""
        methods = self.optimizer.get_available_methods()
        
        expected_methods = ["grid_search", "bayesian", "genetic", "pso", "ray_tune"]
        for method in expected_methods:
            assert method in methods

    def test_get_method_info(self):
        """Test getting method information."""
        info = self.optimizer.get_method_info("grid_search")
        
        assert "name" in info
        assert "class" in info
        assert info["name"] == "grid_search"
        assert info["class"] == "GridSearch"


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            best_params={"param1": 1, "param2": 0.5},
            best_score=0.1,
            all_scores=[0.2, 0.1, 0.3],
            all_params=[{"param1": 2}, {"param1": 1}, {"param1": 3}],
            optimization_time=1.5,
            n_iterations=3,
            convergence_history=[0.2, 0.1, 0.1],
        )

        assert result.best_params == {"param1": 1, "param2": 0.5}
        assert result.best_score == 0.1
        assert len(result.all_scores) == 3
        assert len(result.all_params) == 3
        assert result.optimization_time == 1.5
        assert result.n_iterations == 3
        assert len(result.convergence_history) == 3 