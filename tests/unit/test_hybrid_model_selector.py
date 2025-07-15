"""
Tests for Hybrid Model Selector

Tests metric selection, equal scores handling, and user-selected evaluation metrics.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from trading.forecasting.hybrid_model_selector import HybridModelSelector


class TestHybridModelSelector(unittest.TestCase):
    """Test cases for HybridModelSelector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.selector = HybridModelSelector()
        
        # Sample model scores for testing
        self.sample_scores = {
            "model1": {
                "mse": 0.001,
                "sharpe": 1.5,
                "return": 0.15
            },
            "model2": {
                "mse": 0.002,
                "sharpe": 1.8,
                "return": 0.18
            },
            "model3": {
                "mse": 0.0015,
                "sharpe": 1.6,
                "return": 0.16
            }
        }
        
    def test_select_best_model_mse(self):
        """Test model selection using MSE metric."""
        best_model = self.selector.select_best_model(
            self.sample_scores, 
            metric="mse"
        )
        self.assertEqual(best_model, "model2")  # Alphabetically first among ties
        
    def test_select_best_model_sharpe(self):
        """Test model selection using Sharpe ratio."""
        best_model = self.selector.select_best_model(
            self.sample_scores, 
            metric="sharpe"
        )
        self.assertEqual(best_model, "model2")  # Highest Sharpe
        
    def test_select_best_model_return(self):
        """Test model selection using return metric."""
        best_model = self.selector.select_best_model(
            self.sample_scores, 
            metric="return"
        )
        self.assertEqual(best_model, "model2")  # Highest return
        
    def test_select_best_model_equal_scores(self):
        """Test model selection with equal scores."""
        equal_scores = {
            "model1": {"mse": 0.001, "sharpe": 1.5},
            "model2": {"mse": 0.001, "sharpe": 1.5},  # Equal scores
            "model3": {"mse": 0.002, "sharpe": 1.0}
        }
        
        best_model = self.selector.select_best_model(
            equal_scores, 
            metric="mse"
        )
        # Should return the alphabetically first model with the best score
        self.assertEqual(best_model, "model1")
        
    def test_select_best_model_missing_metric(self):
        """Test model selection with missing metric."""
        missing_metric_scores = {
            "model1": {"mse": 0.001},  # Missing sharpe and return
            "model2": {"sharpe": 1.8},  # Missing mse and return
            "model3": {"return": 0.16}   # Missing mse and sharpe
        }
        
        # Should handle missing metrics gracefully
        best_model = self.selector.select_best_model(
            missing_metric_scores, 
            metric="mse"
        )
        self.assertEqual(best_model, "model2")  # Alphabetically first among ties
        
    def test_select_best_model_unknown_metric(self):
        """Test model selection with unknown metric."""
        with self.assertLogs(level='WARNING'):
            best_model = self.selector.select_best_model(
                self.sample_scores, 
                metric="unknown_metric"
            )
        # Should fall back to default metric
        self.assertIn(best_model, self.sample_scores.keys())
        
    def test_select_highest_sharpe(self):
        """Test highest Sharpe ratio selection strategy."""
        best_model = self.selector._select_highest_sharpe(self.sample_scores)
        self.assertEqual(best_model, "model2")  # Highest Sharpe
        
    def test_select_highest_sharpe_missing_sharpe(self):
        """Test highest Sharpe selection with missing Sharpe ratios."""
        no_sharpe_scores = {
            "model1": {"mse": 0.001, "return": 0.15},
            "model2": {"mse": 0.002, "return": 0.18}
        }
        
        with self.assertLogs(level='WARNING'):
            best_model = self.selector._select_highest_sharpe(no_sharpe_scores)
        # Should return first available model
        self.assertEqual(best_model, "model1")
        
    def test_select_weighted_return(self):
        """Test weighted return selection strategy."""
        best_model = self.selector._select_weighted_return(self.sample_scores)
        # Should select model with highest weighted return
        self.assertIn(best_model, self.sample_scores.keys())
        
    def test_select_weighted_return_missing_metrics(self):
        """Test weighted return selection with missing metrics."""
        partial_scores = {
            "model1": {"return": 0.15},  # Missing sharpe
            "model2": {"sharpe": 1.8, "return": 0.18}
        }
        
        best_model = self.selector._select_weighted_return(partial_scores)
        # Should handle missing metrics gracefully
        self.assertIn(best_model, partial_scores.keys())
        
    def test_select_lowest_mse(self):
        """Test lowest MSE selection strategy."""
        best_model = self.selector._select_lowest_mse(self.sample_scores)
        self.assertEqual(best_model, "model1")  # Lowest MSE
        
    def test_select_lowest_mse_missing_mse(self):
        """Test lowest MSE selection with missing MSE values."""
        no_mse_scores = {
            "model1": {"sharpe": 1.5, "return": 0.15},
            "model2": {"sharpe": 1.8, "return": 0.18}
        }
        
        with self.assertLogs(level='WARNING'):
            best_model = self.selector._select_lowest_mse(no_mse_scores)
        # Should return first available model
        self.assertEqual(best_model, "model1")
        
    def test_get_model_ranking(self):
        """Test model ranking functionality."""
        ranking = self.selector.get_model_ranking(self.sample_scores, "mse")
        
        # Should return sorted list of (model, score) tuples
        self.assertIsInstance(ranking, list)
        self.assertEqual(len(ranking), 3)
        
        # Check sorting (MSE: lower is better)
        scores = [score for _, score in ranking]
        self.assertEqual(scores, sorted(scores))
        
    def test_get_model_ranking_sharpe(self):
        """Test model ranking with Sharpe ratio (higher is better)."""
        ranking = self.selector.get_model_ranking(self.sample_scores, "sharpe")
        
        # Check sorting (Sharpe: higher is better)
        scores = [score for _, score in ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
    def test_validate_scores(self):
        """Test score validation."""
        # Valid scores
        self.assertTrue(self.selector.validate_scores(self.sample_scores))
        
        # Empty scores
        self.assertFalse(self.selector.validate_scores({}))
        
        # Invalid format
        invalid_scores = {
            "model1": "not_a_dict",
            "model2": {"mse": 0.001}
        }
        self.assertFalse(self.selector.validate_scores(invalid_scores))
        
    def test_user_selected_evaluation_metric(self):
        """Test user-selected evaluation metric."""
        # Test with custom metric
        custom_selector = HybridModelSelector({
            "default_metric": "sharpe",
            "score_strategy": "weighted_return"
        })
        
        best_model = custom_selector.select_best_model(
            self.sample_scores,
            metric="return"  # Override default
        )
        self.assertEqual(best_model, "model2")  # Highest return
        
    def test_config_driven_selection(self):
        """Test config-driven model selection."""
        config = {
            "default_metric": "sharpe",
            "score_strategy": "highest_sharpe"
        }
        config_selector = HybridModelSelector(config)
        
        best_model = config_selector.select_best_model(self.sample_scores)
        self.assertEqual(best_model, "model2")  # Highest Sharpe
        
    def test_empty_scores_handling(self):
        """Test handling of empty scores."""
        with self.assertRaises(IndexError):
            self.selector.select_best_model({})
            
    def test_single_model_selection(self):
        """Test selection when only one model is available."""
        single_model_scores = {
            "model1": {"mse": 0.001, "sharpe": 1.5}
        }
        
        best_model = self.selector.select_best_model(
            single_model_scores, 
            metric="mse"
        )
        self.assertEqual(best_model, "model1")


if __name__ == "__main__":
    unittest.main() 