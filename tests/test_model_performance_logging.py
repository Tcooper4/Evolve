"""
Tests for Model Performance Logging Module

This module tests the functionality of the model performance logging system.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory.model_log import (
    log_model_performance,
    get_model_performance_history,
    get_best_models,
    get_available_tickers,
    get_available_models,
    clear_model_performance_log,
    ensure_log_directory
)


class TestModelPerformanceLogging(unittest.TestCase):
    """Test cases for model performance logging functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.original_log_dir = Path("memory/logs")
        
        # Backup original log directory if it exists
        if self.original_log_dir.exists():
            self.backup_dir = tempfile.mkdtemp()
            shutil.copytree(self.original_log_dir, self.backup_dir, dirs_exist_ok=True)
        
        # Clear any existing test data
        clear_model_performance_log()
    
    def tearDown(self):
        """Clean up test environment."""
        # Clear test data
        clear_model_performance_log()
        
        # Restore original log directory if it existed
        if hasattr(self, 'backup_dir') and Path(self.backup_dir).exists():
            if self.original_log_dir.exists():
                shutil.rmtree(self.original_log_dir)
            shutil.copytree(self.backup_dir, self.original_log_dir)
            shutil.rmtree(self.backup_dir)
        
        # Clean up test directory
        shutil.rmtree(self.test_dir)
    
    def test_log_model_performance(self):
        """Test logging model performance data."""
        # Test data
        test_data = {
            "model_name": "LSTM_v1",
            "ticker": "AAPL",
            "sharpe": 1.85,
            "mse": 0.0234,
            "drawdown": -0.12,
            "total_return": 0.25,
            "win_rate": 0.68,
            "accuracy": 0.72,
            "notes": "Test LSTM model"
        }
        
        # Log performance
        result = log_model_performance(**test_data)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("timestamp", result)
        self.assertEqual(result["model_name"], "LSTM_v1")
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["sharpe"], 1.85)
        
        # Verify data was saved
        history = get_model_performance_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["model_name"], "LSTM_v1")
    
    def test_get_model_performance_history(self):
        """Test retrieving model performance history."""
        # Add multiple test records
        test_records = [
            {
                "model_name": "LSTM_v1",
                "ticker": "AAPL",
                "sharpe": 1.85,
                "mse": 0.0234,
                "drawdown": -0.12,
                "total_return": 0.25,
                "win_rate": 0.68,
                "accuracy": 0.72,
                "notes": "Test LSTM model"
            },
            {
                "model_name": "XGBoost_v1",
                "ticker": "AAPL",
                "sharpe": 2.1,
                "mse": 0.0189,
                "drawdown": -0.08,
                "total_return": 0.31,
                "win_rate": 0.75,
                "accuracy": 0.78,
                "notes": "Test XGBoost model"
            },
            {
                "model_name": "LSTM_v1",
                "ticker": "GOOGL",
                "sharpe": 1.92,
                "mse": 0.0198,
                "drawdown": -0.11,
                "total_return": 0.26,
                "win_rate": 0.70,
                "accuracy": 0.73,
                "notes": "Test LSTM model for GOOGL"
            }
        ]
        
        for record in test_records:
            log_model_performance(**record)
        
        # Test getting all history
        all_history = get_model_performance_history()
        self.assertEqual(len(all_history), 3)
        
        # Test filtering by ticker
        aapl_history = get_model_performance_history(ticker="AAPL")
        self.assertEqual(len(aapl_history), 2)
        
        # Test filtering by model
        lstm_history = get_model_performance_history(model_name="LSTM_v1")
        self.assertEqual(len(lstm_history), 2)
        
        # Test filtering by both ticker and model
        aapl_lstm_history = get_model_performance_history(ticker="AAPL", model_name="LSTM_v1")
        self.assertEqual(len(aapl_lstm_history), 1)
    
    def test_get_best_models(self):
        """Test getting best models for each metric."""
        # Add test records with different performance levels
        test_records = [
            {
                "model_name": "LSTM_v1",
                "ticker": "AAPL",
                "sharpe": 1.5,
                "mse": 0.03,
                "drawdown": -0.15,
                "total_return": 0.20,
                "win_rate": 0.60,
                "accuracy": 0.65,
                "notes": "Lower performance model"
            },
            {
                "model_name": "XGBoost_v1",
                "ticker": "AAPL",
                "sharpe": 2.5,
                "mse": 0.01,
                "drawdown": -0.05,
                "total_return": 0.40,
                "win_rate": 0.80,
                "accuracy": 0.85,
                "notes": "Higher performance model"
            }
        ]
        
        for record in test_records:
            log_model_performance(**record)
        
        # Get best models
        best_models = get_best_models("AAPL")
        
        # Verify best models are correctly identified
        self.assertEqual(best_models["best_sharpe"]["model"], "XGBoost_v1")
        self.assertEqual(best_models["best_sharpe"]["value"], 2.5)
        
        self.assertEqual(best_models["best_mse"]["model"], "XGBoost_v1")
        self.assertEqual(best_models["best_mse"]["value"], 0.01)
        
        self.assertEqual(best_models["best_total_return"]["model"], "XGBoost_v1")
        self.assertEqual(best_models["best_total_return"]["value"], 0.40)
    
    def test_get_available_tickers(self):
        """Test getting available tickers."""
        # Add test records for different tickers
        test_records = [
            {"model_name": "LSTM_v1", "ticker": "AAPL", "sharpe": 1.5, "mse": 0.03},
            {"model_name": "XGBoost_v1", "ticker": "GOOGL", "sharpe": 2.0, "mse": 0.02},
            {"model_name": "Transformer_v1", "ticker": "MSFT", "sharpe": 1.8, "mse": 0.025}
        ]
        
        for record in test_records:
            log_model_performance(**record)
        
        # Get available tickers
        tickers = get_available_tickers()
        
        # Verify all tickers are returned
        expected_tickers = ["AAPL", "GOOGL", "MSFT"]
        self.assertEqual(set(tickers), set(expected_tickers))
    
    def test_get_available_models(self):
        """Test getting available models."""
        # Add test records for different models
        test_records = [
            {"model_name": "LSTM_v1", "ticker": "AAPL", "sharpe": 1.5, "mse": 0.03},
            {"model_name": "XGBoost_v1", "ticker": "AAPL", "sharpe": 2.0, "mse": 0.02},
            {"model_name": "LSTM_v1", "ticker": "GOOGL", "sharpe": 1.8, "mse": 0.025}
        ]
        
        for record in test_records:
            log_model_performance(**record)
        
        # Get available models for AAPL
        aapl_models = get_available_models("AAPL")
        expected_models = ["LSTM_v1", "XGBoost_v1"]
        self.assertEqual(set(aapl_models), set(expected_models))
        
        # Get all available models
        all_models = get_available_models()
        expected_all_models = ["LSTM_v1", "XGBoost_v1"]
        self.assertEqual(set(all_models), set(expected_all_models))
    
    def test_clear_model_performance_log(self):
        """Test clearing model performance logs."""
        # Add test data
        test_data = {
            "model_name": "LSTM_v1",
            "ticker": "AAPL",
            "sharpe": 1.85,
            "mse": 0.0234,
            "drawdown": -0.12,
            "total_return": 0.25,
            "win_rate": 0.68,
            "accuracy": 0.72,
            "notes": "Test model"
        }
        
        log_model_performance(**test_data)
        
        # Verify data exists
        history = get_model_performance_history()
        self.assertEqual(len(history), 1)
        
        # Clear logs
        clear_model_performance_log()
        
        # Verify data is cleared
        history = get_model_performance_history()
        self.assertEqual(len(history), 0)
        
        # Verify best models are also cleared
        best_models = get_best_models("AAPL")
        self.assertEqual(best_models, {})
    
    def test_optional_parameters(self):
        """Test logging with optional parameters."""
        # Test with minimal required parameters
        minimal_data = {
            "model_name": "LSTM_v1",
            "ticker": "AAPL"
        }
        
        result = log_model_performance(**minimal_data)
        
        # Verify result contains None for optional parameters
        self.assertIsNone(result["sharpe"])
        self.assertIsNone(result["mse"])
        self.assertIsNone(result["drawdown"])
        self.assertIsNone(result["total_return"])
        self.assertIsNone(result["win_rate"])
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["notes"], "")
    
    def test_data_persistence(self):
        """Test that data persists between function calls."""
        # Add test data
        test_data = {
            "model_name": "LSTM_v1",
            "ticker": "AAPL",
            "sharpe": 1.85,
            "mse": 0.0234,
            "drawdown": -0.12,
            "total_return": 0.25,
            "win_rate": 0.68,
            "accuracy": 0.72,
            "notes": "Test model"
        }
        
        log_model_performance(**test_data)
        
        # Add more data
        test_data2 = {
            "model_name": "XGBoost_v1",
            "ticker": "AAPL",
            "sharpe": 2.1,
            "mse": 0.0189,
            "drawdown": -0.08,
            "total_return": 0.31,
            "win_rate": 0.75,
            "accuracy": 0.78,
            "notes": "Test model 2"
        }
        
        log_model_performance(**test_data2)
        
        # Verify both records exist
        history = get_model_performance_history()
        self.assertEqual(len(history), 2)
        
        # Verify both models are in the best models
        best_models = get_best_models("AAPL")
        self.assertIsNotNone(best_models["best_sharpe"]["model"])
        self.assertIsNotNone(best_models["best_mse"]["model"])


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
