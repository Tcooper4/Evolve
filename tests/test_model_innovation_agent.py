"""
Tests for ModelInnovationAgent

This module contains comprehensive tests for the ModelInnovationAgent,
including unit tests, integration tests, and performance tests.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
from datetime import datetime

# Import the agent
from agents.model_innovation_agent import (
    ModelInnovationAgent,
    InnovationConfig,
    ModelCandidate,
    ModelEvaluation,
    create_model_innovation_agent
)


class TestModelInnovationAgent(unittest.TestCase):
    """Test cases for ModelInnovationAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test configuration
        self.config = InnovationConfig(
            models_dir=str(self.models_dir),
            cache_dir=str(self.cache_dir),
            automl_time_budget=10,  # Short for testing
            max_models_per_search=3,
            min_improvement_threshold=0.01,
        )
        
        # Create test data
        self.test_data = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.randn(100),
        })
        
        # Mock weight registry
        self.mock_registry = Mock()
        self.mock_registry.registry = {
            "models": {
                "existing_model": {
                    "type": "linear",
                    "performance": {
                        "mse": 1.0,
                        "sharpe_ratio": 0.5,
                        "r2_score": 0.3,
                    }
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ModelInnovationAgent(self.config)
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.config, self.config)
        self.assertEqual(len(agent.discovered_models), 0)
        self.assertEqual(len(agent.evaluations), 0)
    
    def test_create_model_innovation_agent(self):
        """Test convenience function for creating agent."""
        agent = create_model_innovation_agent(self.config)
        
        self.assertIsInstance(agent, ModelInnovationAgent)
        self.assertEqual(agent.config, self.config)
    
    def test_prepare_data(self):
        """Test data preparation."""
        agent = ModelInnovationAgent(self.config)
        
        # Test with valid data
        X, y = agent._prepare_data(self.test_data, "target")
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertNotIn("target", X.columns)
        self.assertEqual(y.name, "target")
    
    def test_prepare_data_with_missing_values(self):
        """Test data preparation with missing values."""
        agent = ModelInnovationAgent(self.config)
        
        # Create data with missing values
        data_with_nans = self.test_data.copy()
        data_with_nans.loc[0, "feature1"] = np.nan
        data_with_nans.loc[1, "target"] = np.nan
        
        X, y = agent._prepare_data(data_with_nans, "target")
        
        self.assertFalse(X.isna().any().any())
        self.assertFalse(y.isna().any())
    
    def test_classify_model_type(self):
        """Test model type classification."""
        agent = ModelInnovationAgent(self.config)
        
        # Test linear models
        self.assertEqual(agent._classify_model_type("linear"), "linear")
        self.assertEqual(agent._classify_model_type("ridge"), "linear")
        self.assertEqual(agent._classify_model_type("lasso"), "linear")
        
        # Test tree models
        self.assertEqual(agent._classify_model_type("rf"), "tree")
        self.assertEqual(agent._classify_model_type("xgboost"), "tree")
        self.assertEqual(agent._classify_model_type("lgbm"), "tree")
        
        # Test neural models
        self.assertEqual(agent._classify_model_type("neural"), "neural")
        self.assertEqual(agent._classify_model_type("mlp"), "neural")
        self.assertEqual(agent._classify_model_type("lstm"), "neural")
        
        # Test unknown
        self.assertEqual(agent._classify_model_type("unknown_model"), "unknown")
    
    def test_calculate_model_size(self):
        """Test model size calculation."""
        agent = ModelInnovationAgent(self.config)
        
        # Create a simple model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(self.test_data[["feature1", "feature2"]], self.test_data["target"])
        
        size = agent._calculate_model_size(model)
        
        self.assertGreater(size, 0)
        self.assertIsInstance(size, float)
    
    @patch('agents.model_innovation_agent.FLAML_AVAILABLE', True)
    @patch('agents.model_innovation_agent.AutoML')
    def test_discover_with_flaml(self, mock_automl):
        """Test FLAML model discovery."""
        agent = ModelInnovationAgent(self.config)
        
        # Mock FLAML AutoML
        mock_automl_instance = Mock()
        mock_automl_instance.best_loss = 0.5
        mock_automl_instance.best_config = {"estimator": "rf", "n_estimators": 100}
        mock_automl_instance.model = Mock()
        mock_automl.return_value = mock_automl_instance
        
        X, y = agent._prepare_data(self.test_data, "target")
        candidates = agent._discover_with_flaml(X, y)
        
        self.assertGreater(len(candidates), 0)
        self.assertIsInstance(candidates[0], ModelCandidate)
        self.assertEqual(candidates[0].model_type, "tree")
    
    @patch('agents.model_innovation_agent.OPTUNA_AVAILABLE', True)
    @patch('agents.model_innovation_agent.create_study')
    def test_discover_with_optuna(self, mock_create_study):
        """Test Optuna model discovery."""
        agent = ModelInnovationAgent(self.config)
        
        # Mock Optuna study
        mock_study = Mock()
        mock_study.best_trial.value = 0.4
        mock_study.best_trial.params = {
            "model_type": "linear",
            "alpha": 1.0,
            "fit_intercept": True
        }
        mock_create_study.return_value = mock_study
        
        X, y = agent._prepare_data(self.test_data, "target")
        candidates = agent._discover_with_optuna(X, y)
        
        self.assertGreater(len(candidates), 0)
        self.assertIsInstance(candidates[0], ModelCandidate)
        self.assertEqual(candidates[0].model_type, "linear")
    
    @patch('agents.model_innovation_agent.SKLEARN_AVAILABLE', True)
    def test_discover_manual_models(self):
        """Test manual model discovery."""
        agent = ModelInnovationAgent(self.config)
        
        X, y = agent._prepare_data(self.test_data, "target")
        candidates = agent._discover_manual_models(X, y)
        
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertIsInstance(candidate, ModelCandidate)
            self.assertIn(candidate.model_type, ["linear", "tree"])
    
    def test_evaluate_candidate(self):
        """Test candidate evaluation."""
        agent = ModelInnovationAgent(self.config)
        
        # Create a simple candidate
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        candidate = ModelCandidate(
            name="test_model",
            model_type="linear",
            model=model,
            hyperparameters={},
            training_time=1.0
        )
        
        evaluation = agent.evaluate_candidate(candidate, self.test_data, "target")
        
        self.assertIsInstance(evaluation, ModelEvaluation)
        self.assertEqual(evaluation.model_name, "test_model")
        self.assertGreater(evaluation.training_time, 0)
        self.assertIsInstance(evaluation.mse, float)
        self.assertIsInstance(evaluation.r2_score, float)
    
    def test_compare_with_ensemble(self):
        """Test ensemble comparison."""
        agent = ModelInnovationAgent(self.config)
        
        # Mock weight registry
        agent.weight_registry = self.mock_registry
        
        # Create evaluation
        evaluation = ModelEvaluation(
            model_name="test_model",
            mse=0.5,  # Better than existing (1.0)
            mae=0.3,
            r2_score=0.6,  # Better than existing (0.3)
            sharpe_ratio=0.8,  # Better than existing (0.5)
            max_drawdown=-0.1,
            total_return=0.1,
            volatility=0.2,
            training_time=1.0,
            inference_time=0.01,
            model_size_mb=0.1
        )
        
        comparison = agent.compare_with_ensemble(evaluation)
        
        self.assertIsInstance(comparison, dict)
        self.assertIn("improvement", comparison)
        self.assertIn("improvements", comparison)
        self.assertIn("current_ensemble", comparison)
        self.assertIn("candidate_metrics", comparison)
    
    @patch('agents.model_innovation_agent.get_weight_registry')
    def test_integrate_model(self, mock_get_registry):
        """Test model integration."""
        agent = ModelInnovationAgent(self.config)
        
        # Mock weight registry
        mock_registry = Mock()
        mock_registry.register_model.return_value = True
        mock_registry.update_performance.return_value = True
        mock_registry.update_weights.return_value = True
        agent.weight_registry = mock_registry
        
        # Create candidate and evaluation
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        candidate = ModelCandidate(
            name="test_model",
            model_type="linear",
            model=model,
            hyperparameters={},
            training_time=1.0
        )
        
        evaluation = ModelEvaluation(
            model_name="test_model",
            mse=0.5,
            mae=0.3,
            r2_score=0.6,
            sharpe_ratio=0.8,
            max_drawdown=-0.1,
            total_return=0.1,
            volatility=0.2,
            training_time=1.0,
            inference_time=0.01,
            model_size_mb=0.1
        )
        
        success = agent.integrate_model(candidate, evaluation)
        
        self.assertTrue(success)
        mock_registry.register_model.assert_called_once()
        mock_registry.update_performance.assert_called_once()
    
    @patch('agents.model_innovation_agent.optimize_ensemble_weights')
    def test_run_innovation_cycle(self, mock_optimize):
        """Test complete innovation cycle."""
        agent = ModelInnovationAgent(self.config)
        
        # Mock dependencies
        agent.weight_registry = self.mock_registry
        mock_optimize.return_value = {"existing_model": 0.5, "new_model": 0.5}
        
        # Mock discovery to return a simple model
        with patch.object(agent, 'discover_models') as mock_discover:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(self.test_data[["feature1"]], self.test_data["target"])
            
            candidate = ModelCandidate(
                name="test_model",
                model_type="linear",
                model=model,
                hyperparameters={},
                training_time=1.0
            )
            
            mock_discover.return_value = [candidate]
            
            # Mock weight registry methods
            agent.weight_registry.register_model.return_value = True
            agent.weight_registry.update_performance.return_value = True
            agent.weight_registry.update_weights.return_value = True
            
            results = agent.run_innovation_cycle(self.test_data, "target")
        
        self.assertIsInstance(results, dict)
        self.assertIn("cycle_start", results)
        self.assertIn("candidates_discovered", results)
        self.assertIn("candidates_evaluated", results)
        self.assertIn("models_integrated", results)
        self.assertIn("improvements_found", results)
    
    def test_get_innovation_statistics(self):
        """Test innovation statistics."""
        agent = ModelInnovationAgent(self.config)
        
        # Mock weight registry
        agent.weight_registry = self.mock_registry
        
        # Add some innovation history
        agent.innovation_history = [
            {
                "timestamp": datetime.now().isoformat(),
                "model_name": "test_model",
                "model_type": "linear",
                "integration_success": True,
                "improvement_metrics": {"mse": 0.5, "sharpe_ratio": 0.8, "r2_score": 0.6}
            }
        ]
        
        stats = agent.get_innovation_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_cycles", stats)
        self.assertIn("total_models_integrated", stats)
        self.assertIn("total_evaluations", stats)
        self.assertIn("recent_innovations", stats)
        self.assertIn("model_type_distribution", stats)
        self.assertIn("performance_improvements", stats)
    
    def test_get_model_type_distribution(self):
        """Test model type distribution calculation."""
        agent = ModelInnovationAgent(self.config)
        
        # Mock weight registry with different model types
        agent.weight_registry.registry = {
            "models": {
                "model1": {"type": "linear"},
                "model2": {"type": "tree"},
                "model3": {"type": "linear"},
                "model4": {"type": "neural"},
            }
        }
        
        distribution = agent._get_model_type_distribution()
        
        self.assertEqual(distribution["linear"], 2)
        self.assertEqual(distribution["tree"], 1)
        self.assertEqual(distribution["neural"], 1)
    
    def test_get_performance_improvements(self):
        """Test performance improvements extraction."""
        agent = ModelInnovationAgent(self.config)
        
        # Add innovation history
        agent.innovation_history = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "model_name": "test_model",
                "model_type": "linear",
                "integration_success": True,
                "improvement_metrics": {"mse": 0.5, "sharpe_ratio": 0.8, "r2_score": 0.6}
            },
            {
                "timestamp": "2024-01-02T00:00:00",
                "model_name": "test_model2",
                "model_type": "tree",
                "integration_success": False,  # Should be filtered out
                "improvement_metrics": {"mse": 0.6, "sharpe_ratio": 0.7, "r2_score": 0.5}
            }
        ]
        
        improvements = agent._get_performance_improvements()
        
        self.assertEqual(len(improvements), 1)
        self.assertEqual(improvements[0]["model_name"], "test_model")
        self.assertEqual(improvements[0]["model_type"], "linear")


class TestModelInnovationAgentIntegration(unittest.TestCase):
    """Integration tests for ModelInnovationAgent."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = InnovationConfig(
            models_dir=str(Path(self.temp_dir) / "models"),
            cache_dir=str(Path(self.temp_dir) / "cache"),
            automl_time_budget=5,  # Very short for testing
            max_models_per_search=2,
        )
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow_with_real_data(self):
        """Test full workflow with realistic data."""
        # Create realistic time series data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # Create features with some correlation to target
        feature1 = np.random.randn(200)
        feature2 = feature1 * 0.7 + np.random.randn(200) * 0.3
        feature3 = np.random.randn(200)
        
        # Create target with trend and seasonality
        trend = np.linspace(0, 10, 200)
        seasonality = 2 * np.sin(2 * np.pi * np.arange(200) / 30)
        noise = np.random.randn(200) * 0.5
        target = trend + seasonality + feature1 * 0.3 + feature2 * 0.2 + noise
        
        data = pd.DataFrame({
            "date": dates,
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "target": target,
        })
        
        # Create agent
        agent = ModelInnovationAgent(self.config)
        
        # Mock weight registry
        agent.weight_registry = Mock()
        agent.weight_registry.registry = {"models": {}}
        agent.weight_registry.register_model.return_value = True
        agent.weight_registry.update_performance.return_value = True
        agent.weight_registry.update_weights.return_value = True
        
        # Run innovation cycle
        results = agent.run_innovation_cycle(data, "target")
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn("cycle_start", results)
        self.assertIn("cycle_end", results)
        self.assertIn("candidates_discovered", results)
        self.assertIn("candidates_evaluated", results)
        self.assertIn("models_integrated", results)
        self.assertIn("improvements_found", results)
        self.assertIn("errors", results)
        
        # Verify cycle completed
        self.assertGreater(results["candidates_discovered"], 0)
        self.assertGreater(results["candidates_evaluated"], 0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2) 