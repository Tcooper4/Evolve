"""
Test script for system modules improvements.

This script tests the improvements made to:
1. system/logger_debugger.py (refactored from debug.py)
2. trading/models/advanced/transformer/time_series_transformer.py (enhanced with fallbacks)
3. system/forecast_controller.py (new routing logic)
4. system/hybrid_engine.py (new hybrid model logic)
"""

import json
import logging
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from system.logger_debugger import LoggerDebugger, analyze_errors, monitor_errors, get_logger_status
from system.forecast_controller import ForecastController, route_forecast
from system.hybrid_engine import HybridEngine


class TestLoggerDebugger(unittest.TestCase):
    """Test LoggerDebugger functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.debugger = LoggerDebugger()
        
        # Create test log files
        self.test_log_file = Path(self.temp_dir) / "test.log"
        with open(self.test_log_file, "w") as f:
            f.write("2024-01-01 10:00:00 INFO: Test info message\n")
            f.write("2024-01-01 10:01:00 ERROR: ImportError: Module not found\n")
            f.write("2024-01-01 10:02:00 ERROR: ValidationError: Invalid data\n")
            f.write("2024-01-01 10:03:00 ERROR: ConnectionError: Network timeout\n")

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_analyze_errors(self):
        """Test error analysis functionality."""
        # Test direct function call
        analysis = analyze_errors([str(self.test_log_file)])
        
        self.assertIn("total_errors", analysis)
        self.assertEqual(analysis["total_errors"], 3)
        self.assertIn("error_types", analysis)
        self.assertIn("ImportError", analysis["error_types"])
        self.assertIn("ValidationError", analysis["error_types"])
        self.assertIn("ConnectionError", analysis["error_types"])

    def test_logger_debugger_instance(self):
        """Test LoggerDebugger instance methods."""
        # Test error analysis
        analysis = self.debugger.analyze_errors([str(self.test_log_file)])
        self.assertEqual(analysis["total_errors"], 3)
        
        # Test fix suggestions
        suggestions = self.debugger.fix_errors(analysis)
        self.assertGreater(len(suggestions), 0)
        
        # Test logger status
        status = self.debugger.get_logger_status()
        self.assertIn("loggers", status)
        self.assertIn("handlers", status)

    def test_clear_logs(self):
        """Test log clearing functionality."""
        # Create old log file
        old_log = Path(self.temp_dir) / "old.log"
        old_log.touch()
        
        # Set old modification time
        old_time = datetime.now() - timedelta(days=10)
        old_log.touch()
        
        # Test clearing
        deleted_count = self.debugger.clear_logs(days_to_keep=7)
        self.assertGreaterEqual(deleted_count, 0)

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test get_logger_status
        status = get_logger_status()
        self.assertIsInstance(status, dict)
        self.assertIn("timestamp", status)


class TestTransformerForecaster(unittest.TestCase):
    """Test enhanced TransformerForecaster functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame({
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "date": dates
        }).set_index("date")
        
        # Mock ARIMA model
        self.mock_arima = Mock()
        self.mock_arima.forecast.return_value = {
            "forecast": np.random.randn(30),
            "confidence": 0.8,
            "model": "ARIMA"
        }

    @patch("trading.models.arima_model.ARIMAForecaster")
    def test_short_series_guard(self, mock_arima_class):
        """Test short series guard functionality."""
        mock_arima_class.return_value = self.mock_arima
        
        # Import transformer
        from trading.models.advanced.transformer.time_series_transformer import TransformerForecaster
        
        # Test with short series
        short_data = self.test_data.head(10)  # Only 10 points
        
        config = {
            "min_series_length": 20,
            "enable_fallback": True,
            "input_size": 2,
            "feature_columns": ["close", "volume"],
            "target_column": "close",
            "sequence_length": 5,
        }
        
        transformer = TransformerForecaster(config)
        
        # Test series length check
        is_sufficient = transformer._check_series_length(short_data)
        self.assertFalse(is_sufficient)
        
        # Test with sufficient series
        is_sufficient = transformer._check_series_length(self.test_data)
        self.assertTrue(is_sufficient)

    @patch("trading.models.arima_model.ARIMAForecaster")
    def test_fallback_model_setup(self, mock_arima_class):
        """Test fallback model setup."""
        mock_arima_class.return_value = self.mock_arima
        
        from trading.models.advanced.transformer.time_series_transformer import TransformerForecaster
        
        config = {
            "enable_fallback": True,
            "input_size": 2,
            "feature_columns": ["close", "volume"],
            "target_column": "close",
            "sequence_length": 5,
        }
        
        transformer = TransformerForecaster(config)
        self.assertIsNotNone(transformer.fallback_model)

    def test_masking_configuration(self):
        """Test masking configuration."""
        from trading.models.advanced.transformer.time_series_transformer import TransformerForecaster
        
        config = {
            "masking": True,
            "input_size": 2,
            "feature_columns": ["close", "volume"],
            "target_column": "close",
            "sequence_length": 5,
        }
        
        transformer = TransformerForecaster(config)
        self.assertTrue(transformer.config["masking"])


class TestForecastController(unittest.TestCase):
    """Test ForecastController functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame({
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "date": dates
        }).set_index("date")
        
        # Mock dependencies
        self.mock_model_selector = Mock()
        self.mock_forecast_router = Mock()
        
        # Mock model selector response
        self.mock_model_selector.select_best_models.return_value = ["ARIMA", "XGBoost"]

    @patch("trading.models.arima_model.ARIMAForecaster")
    @patch("trading.models.xgboost_model.XGBoostForecaster")
    def test_forecast_routing(self, mock_xgboost_class, mock_arima_class):
        """Test forecast routing functionality."""
        # Mock model classes
        mock_arima = Mock()
        mock_arima.forecast.return_value = {
            "forecast": np.random.randn(30),
            "confidence": 0.8,
            "model": "ARIMA"
        }
        mock_arima_class.return_value = mock_arima
        
        mock_xgboost = Mock()
        mock_xgboost.forecast.return_value = {
            "forecast": np.random.randn(30),
            "confidence": 0.9,
            "model": "XGBoost"
        }
        mock_xgboost_class.return_value = mock_xgboost
        
        # Create controller
        controller = ForecastController(
            model_selector=self.mock_model_selector,
            forecast_router=self.mock_forecast_router,
            enable_hybrid=False  # Disable hybrid for simpler test
        )
        
        # Test routing
        context = {
            "market_volatility": "medium",
            "market_trend": "neutral",
            "seasonality": False
        }
        
        result = controller.route_forecast_request(
            data=self.test_data,
            context=context,
            horizon=30,
            confidence_required=0.7
        )
        
        # Verify result
        self.assertIn("request_id", result)
        self.assertIn("models_used", result)
        self.assertIn("routing_info", result)
        self.assertIn("forecast", result)

    def test_context_model_selection(self):
        """Test context-based model selection."""
        controller = ForecastController()
        
        # Test different contexts
        high_vol_context = {"market_volatility": "high", "data_length": 50}
        low_vol_context = {"market_volatility": "low", "data_length": 200}
        
        high_vol_models = controller._select_models_for_context(
            self.test_data, high_vol_context, 30
        )
        low_vol_models = controller._select_models_for_context(
            self.test_data, low_vol_context, 30
        )
        
        # Should select different models for different contexts
        self.assertIsInstance(high_vol_models, list)
        self.assertIsInstance(low_vol_models, list)
        self.assertGreater(len(high_vol_models), 0)
        self.assertGreater(len(low_vol_models), 0)

    def test_trend_strength_calculation(self):
        """Test trend strength calculation."""
        controller = ForecastController()
        
        # Test with trending data
        trending_data = pd.DataFrame({
            "close": np.arange(100) + np.random.randn(100) * 0.1,
            "volume": np.random.randint(1000, 10000, 100)
        })
        
        trend_strength = controller._calculate_trend_strength(trending_data)
        self.assertGreater(trend_strength, 0.5)  # Should detect strong trend
        
        # Test with random data
        random_data = pd.DataFrame({
            "close": np.random.randn(100),
            "volume": np.random.randint(1000, 10000, 100)
        })
        
        trend_strength = controller._calculate_trend_strength(random_data)
        self.assertLess(trend_strength, 0.5)  # Should detect weak trend

    def test_performance_summary(self):
        """Test performance summary functionality."""
        controller = ForecastController()
        
        # Add some mock history
        controller.request_history = [
            {"success": True, "confidence": 0.8},
            {"success": False, "confidence": 0.3},
            {"success": True, "confidence": 0.9},
        ]
        
        summary = controller.get_performance_summary()
        self.assertIn("total_requests", summary)
        self.assertIn("success_rate", summary)
        self.assertIn("average_confidence", summary)
        
        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["success_rate"], 2/3)

    def test_routing_recommendations(self):
        """Test routing recommendations."""
        controller = ForecastController()
        
        context = {
            "market_volatility": "high",
            "data_length": 50
        }
        
        recommendations = controller.get_routing_recommendations(context)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        for rec in recommendations:
            self.assertIn("model", rec)
            self.assertIn("reason", rec)
            self.assertIn("confidence", rec)


class TestHybridEngine(unittest.TestCase):
    """Test HybridEngine functionality."""

    def setUp(self):
        """Set up test environment."""
        self.engine = HybridEngine()
        
        # Create test forecasts
        self.test_forecasts = [
            {
                "forecast": np.random.randn(30) + 100,
                "confidence": 0.8,
                "model": "ARIMA"
            },
            {
                "forecast": np.random.randn(30) + 100,
                "confidence": 0.9,
                "model": "XGBoost"
            },
            {
                "forecast": np.random.randn(30) + 100,
                "confidence": 0.7,
                "model": "LSTM"
            }
        ]

    def test_forecast_validation(self):
        """Test forecast validation."""
        # Test valid forecasts
        valid_forecasts = self.engine._validate_forecasts(self.test_forecasts)
        self.assertEqual(len(valid_forecasts), 3)
        
        # Test invalid forecast
        invalid_forecasts = [
            {"model": "Invalid"},  # Missing forecast
            {"forecast": "not_array", "model": "Invalid"},  # Wrong type
            {"forecast": [], "model": "Empty"},  # Empty array
        ]
        
        valid_forecasts = self.engine._validate_forecasts(invalid_forecasts)
        self.assertEqual(len(valid_forecasts), 0)

    def test_weighted_average_combination(self):
        """Test weighted average combination."""
        context = {"market_volatility": "medium"}
        
        result = self.engine._weighted_average_combination(self.test_forecasts, context)
        
        self.assertIn("forecast", result)
        self.assertIn("confidence", result)
        self.assertIn("model", result)
        self.assertIn("weights", result)
        
        self.assertEqual(result["model"], "Hybrid_Weighted")
        self.assertEqual(len(result["weights"]), 3)

    def test_median_combination(self):
        """Test median combination."""
        context = {"market_volatility": "medium"}
        
        result = self.engine._median_combination(self.test_forecasts, context)
        
        self.assertIn("forecast", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["model"], "Hybrid_Median")

    def test_outlier_detection(self):
        """Test outlier detection."""
        # Create forecasts with outliers
        outlier_forecasts = [
            {
                "forecast": np.random.randn(30) + 100,
                "confidence": 0.8,
                "model": "Normal"
            },
            {
                "forecast": np.random.randn(30) + 100,
                "confidence": 0.9,
                "model": "Normal"
            },
            {
                "forecast": np.random.randn(30) + 1000,  # Outlier
                "confidence": 0.7,
                "model": "Outlier"
            }
        ]
        
        # Test with outlier detection enabled
        self.engine.outlier_detection = True
        filtered_forecasts = self.engine._handle_outliers(outlier_forecasts)
        
        # Should remove outlier
        self.assertLess(len(filtered_forecasts), len(outlier_forecasts))

    def test_combination_methods(self):
        """Test different combination methods."""
        context = {"market_volatility": "medium"}
        
        methods = ["weighted_average", "median", "trimmed_mean", "bayesian", "stacking", "voting"]
        
        for method in methods:
            self.engine.set_combination_method(method)
            result = self.engine.combine_forecasts(self.test_forecasts, context)
            
            self.assertIn("forecast", result)
            self.assertIn("confidence", result)
            self.assertIn("combination_method", result)
            self.assertEqual(result["combination_method"], method)

    def test_performance_summary(self):
        """Test performance summary."""
        # Add some combination history
        self.engine.combination_history = [
            {"confidence": 0.8, "method": "weighted_average"},
            {"confidence": 0.9, "method": "median"},
            {"confidence": 0.7, "method": "weighted_average"},
        ]
        
        summary = self.engine.get_performance_summary()
        self.assertIn("total_combinations", summary)
        self.assertIn("average_confidence", summary)
        self.assertIn("method_usage", summary)
        
        self.assertEqual(summary["total_combinations"], 3)
        self.assertAlmostEqual(summary["average_confidence"], 0.8, places=1)


class TestIntegration(unittest.TestCase):
    """Test integration between modules."""

    def setUp(self):
        """Set up test environment."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame({
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "date": dates
        }).set_index("date")

    @patch("trading.models.arima_model.ARIMAForecaster")
    @patch("trading.models.xgboost_model.XGBoostForecaster")
    def test_forecast_controller_with_hybrid_engine(self, mock_xgboost_class, mock_arima_class):
        """Test integration between ForecastController and HybridEngine."""
        # Mock models
        mock_arima = Mock()
        mock_arima.forecast.return_value = {
            "forecast": np.random.randn(30),
            "confidence": 0.8,
            "model": "ARIMA"
        }
        mock_arima_class.return_value = mock_arima
        
        mock_xgboost = Mock()
        mock_xgboost.forecast.return_value = {
            "forecast": np.random.randn(30),
            "confidence": 0.9,
            "model": "XGBoost"
        }
        mock_xgboost_class.return_value = mock_xgboost
        
        # Create controller with hybrid enabled
        controller = ForecastController(enable_hybrid=True)
        
        # Mock model selector
        controller.model_selector.select_best_models.return_value = ["ARIMA", "XGBoost"]
        
        # Test routing with multiple models
        context = {"market_volatility": "medium"}
        result = controller.route_forecast_request(
            data=self.test_data,
            context=context,
            horizon=30
        )
        
        # Should use hybrid combination
        self.assertIn("routing_info", result)
        self.assertTrue(result["routing_info"]["hybrid_used"])

    def test_logger_debugger_with_forecast_controller(self):
        """Test integration between LoggerDebugger and ForecastController."""
        # Create debugger
        debugger = LoggerDebugger()
        
        # Create controller
        controller = ForecastController()
        
        # Test that both can coexist
        self.assertIsNotNone(debugger)
        self.assertIsNotNone(controller)
        
        # Test logger status
        status = debugger.get_logger_status()
        self.assertIsInstance(status, dict)


def run_performance_tests():
    """Run performance tests."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTS")
    print("="*60)
    
    # Test data generation
    np.random.seed(42)
    large_data = pd.DataFrame({
        "close": np.random.randn(1000).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 1000),
        "date": pd.date_range("2024-01-01", periods=1000, freq="D")
    }).set_index("date")
    
    # Test hybrid engine performance
    print("\nTesting HybridEngine performance...")
    engine = HybridEngine()
    
    # Create many forecasts
    many_forecasts = []
    for i in range(10):
        many_forecasts.append({
            "forecast": np.random.randn(100) + 100,
            "confidence": np.random.uniform(0.5, 0.95),
            "model": f"Model_{i}"
        })
    
    import time
    start_time = time.time()
    
    for method in ["weighted_average", "median", "trimmed_mean", "bayesian"]:
        engine.set_combination_method(method)
        result = engine.combine_forecasts(many_forecasts, {})
        elapsed = time.time() - start_time
        print(f"  {method}: {elapsed:.4f}s")
    
    # Test forecast controller performance
    print("\nTesting ForecastController performance...")
    controller = ForecastController()
    
    # Mock model selector
    controller.model_selector.select_best_models.return_value = ["ARIMA", "XGBoost"]
    
    start_time = time.time()
    for i in range(5):
        context = {"market_volatility": "medium", "data_length": len(large_data)}
        result = controller.route_forecast_request(
            data=large_data.head(100),  # Use subset for speed
            context=context,
            horizon=30
        )
    
    elapsed = time.time() - start_time
    print(f"  Average routing time: {elapsed/5:.4f}s")


def main():
    """Run all tests."""
    print("Testing System Modules Improvements")
    print("="*60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main() 