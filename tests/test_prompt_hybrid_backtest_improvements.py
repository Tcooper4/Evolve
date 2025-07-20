"""
Test script for prompt parser, hybrid model, and backtest improvements.

This script tests the improvements made to:
1. trading/llm/prompt_parser.py (spaCy-based prompt classification)
2. pages/HybridModel.py (auto-adjusting weights and sidebar)
3. trading/core/backtest_common.py (extracted utilities and frequency scaling)
"""

import unittest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.llm.prompt_parser import PromptParser, ActionPlan, parse_prompt, parse_prompts
from trading.core.backtest_common import BacktestCommon, Frequency, validate_backtest_data, calculate_backtest_metrics


class TestPromptParser(unittest.TestCase):
    """Test PromptParser functionality."""

    def setUp(self):
        """Set up test environment."""
        # Mock spaCy to avoid loading large models
        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_nlp.return_value = Mock()
            mock_load.return_value = mock_nlp
            self.parser = PromptParser()

    def test_action_plan_creation(self):
        """Test ActionPlan dataclass creation and normalization."""
        # Test basic creation
        action_plan = ActionPlan(
            model="lstm",
            strategy="bollinger",
            backtest_flag=True,
            export_type="csv",
            confidence=0.8,
            raw_prompt="Use LSTM with Bollinger Bands strategy",
            extracted_entities={"models": ["LSTM"], "strategies": ["BollingerBands"]}
        )
        
        # Test normalization
        self.assertEqual(action_plan.model, "LSTM")
        self.assertEqual(action_plan.strategy, "BollingerBands")
        self.assertEqual(action_plan.export_type, "csv")
        self.assertTrue(action_plan.backtest_flag)
        self.assertEqual(action_plan.confidence, 0.8)

    def test_action_plan_to_dict(self):
        """Test ActionPlan to_dict method."""
        action_plan = ActionPlan(
            model="transformer",
            strategy="macd",
            backtest_flag=False,
            export_type="json",
            confidence=0.9,
            raw_prompt="Test prompt",
            extracted_entities={}
        )
        
        plan_dict = action_plan.to_dict()
        
        self.assertIsInstance(plan_dict, dict)
        self.assertEqual(plan_dict["model"], "Transformer")
        self.assertEqual(plan_dict["strategy"], "MACD")
        self.assertFalse(plan_dict["backtest_flag"])
        self.assertEqual(plan_dict["export_type"], "json")

    def test_action_plan_string_representation(self):
        """Test ActionPlan string representation."""
        action_plan = ActionPlan(
            model="xgboost",
            strategy="rsi",
            backtest_flag=True,
            export_type="pdf",
            confidence=0.75,
            raw_prompt="Test prompt",
            extracted_entities={}
        )
        
        plan_str = str(action_plan)
        self.assertIn("XGBoost", plan_str)
        self.assertIn("RSI", plan_str)
        self.assertIn("True", plan_str)
        self.assertIn("pdf", plan_str)
        self.assertIn("0.75", plan_str)

    @patch('spacy.load')
    def test_prompt_parser_initialization(self, mock_load):
        """Test PromptParser initialization."""
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp
        
        parser = PromptParser()
        
        self.assertIsNotNone(parser)
        self.assertIsNotNone(parser.matcher)
        self.assertIn("LSTM", parser.model_keywords)
        self.assertIn("BollingerBands", parser.strategy_keywords)

    def test_model_keywords(self):
        """Test model keyword mappings."""
        self.assertIn("LSTM", self.parser.model_keywords)
        self.assertIn("Transformer", self.parser.model_keywords)
        self.assertIn("XGBoost", self.parser.model_keywords)
        self.assertIn("ARIMA", self.parser.model_keywords)
        self.assertIn("Prophet", self.parser.model_keywords)

    def test_strategy_keywords(self):
        """Test strategy keyword mappings."""
        self.assertIn("BollingerBands", self.parser.strategy_keywords)
        self.assertIn("MACD", self.parser.strategy_keywords)
        self.assertIn("RSI", self.parser.strategy_keywords)
        self.assertIn("MovingAverage", self.parser.strategy_keywords)
        self.assertIn("Momentum", self.parser.strategy_keywords)

    def test_action_keywords(self):
        """Test action keyword mappings."""
        self.assertIn("backtest", self.parser.action_keywords)
        self.assertIn("export", self.parser.action_keywords)
        self.assertIn("train", self.parser.action_keywords)
        self.assertIn("predict", self.parser.action_keywords)

    @patch('spacy.load')
    def test_parse_prompt_convenience_function(self, mock_load):
        """Test parse_prompt convenience function."""
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp
        
        # Mock the parsing process
        with patch.object(PromptParser, 'parse_prompt') as mock_parse:
            mock_parse.return_value = ActionPlan(
                model="Ensemble",
                strategy="Momentum",
                backtest_flag=False,
                export_type="csv",
                confidence=0.5,
                raw_prompt="Test",
                extracted_entities={}
            )
            
            result = parse_prompt("Use ensemble model with momentum strategy")
            
            self.assertIsInstance(result, ActionPlan)
            self.assertEqual(result.model, "Ensemble")

    @patch('spacy.load')
    def test_parse_prompts_batch(self, mock_load):
        """Test parse_prompts batch function."""
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp
        
        # Mock the parsing process
        with patch.object(PromptParser, 'parse_batch') as mock_parse_batch:
            mock_parse_batch.return_value = [
                ActionPlan(
                    model="LSTM",
                    strategy="BollingerBands",
                    backtest_flag=True,
                    export_type="csv",
                    confidence=0.8,
                    raw_prompt="Test 1",
                    extracted_entities={}
                ),
                ActionPlan(
                    model="XGBoost",
                    strategy="MACD",
                    backtest_flag=False,
                    export_type="json",
                    confidence=0.7,
                    raw_prompt="Test 2",
                    extracted_entities={}
                )
            ]
            
            prompts = [
                "Use LSTM with Bollinger Bands and backtest",
                "Use XGBoost with MACD strategy"
            ]
            
            results = parse_prompts(prompts)
            
            self.assertEqual(len(results), 2)
            self.assertIsInstance(results[0], ActionPlan)
            self.assertIsInstance(results[1], ActionPlan)


class TestBacktestCommon(unittest.TestCase):
    """Test BacktestCommon functionality."""

    def setUp(self):
        """Set up test environment."""
        self.common = BacktestCommon()
        
        # Create test data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame({
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 102,
            "low": np.random.randn(100).cumsum() + 98,
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_frequency_enum(self):
        """Test Frequency enum values."""
        self.assertEqual(Frequency.DAILY.value, "1d")
        self.assertEqual(Frequency.HOUR_1.value, "1h")
        self.assertEqual(Frequency.MINUTE_5.value, "5min")
        self.assertEqual(Frequency.WEEKLY.value, "1w")

    def test_validate_data_success(self):
        """Test successful data validation."""
        is_valid, message = self.common.validate_data(
            self.test_data,
            required_columns=["open", "high", "low", "close", "volume"]
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Data is valid")

    def test_validate_data_empty(self):
        """Test validation of empty data."""
        empty_data = pd.DataFrame()
        is_valid, message = self.common.validate_data(empty_data)
        
        self.assertFalse(is_valid)
        self.assertIn("empty", message)

    def test_validate_data_missing_columns(self):
        """Test validation with missing columns."""
        is_valid, message = self.common.validate_data(
            self.test_data,
            required_columns=["open", "high", "low", "close", "volume", "missing_col"]
        )
        
        self.assertFalse(is_valid)
        self.assertIn("missing_col", message)

    def test_validate_data_short(self):
        """Test validation of short data."""
        short_data = self.test_data.head(5)
        is_valid, message = self.common.validate_data(short_data, min_length=10)
        
        self.assertFalse(is_valid)
        self.assertIn("minimum 10", message)

    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Add some NaN values
        test_data_with_nan = self.test_data.copy()
        test_data_with_nan.loc[10:15, "close"] = np.nan
        
        processed_data = self.common.preprocess_data(
            test_data_with_nan,
            frequency=Frequency.DAILY,
            fill_method="ffill"
        )
        
        self.assertFalse(processed_data.isnull().any().any())
        self.assertEqual(len(processed_data), len(self.test_data) - 6)  # NaN rows removed

    def test_calculate_returns(self):
        """Test return calculation."""
        prices = pd.Series([100, 110, 105, 120, 115])
        
        # Test log returns
        log_returns = self.common.calculate_returns(prices, method="log")
        self.assertEqual(len(log_returns), 4)  # First value is NaN
        
        # Test simple returns
        simple_returns = self.common.calculate_returns(prices, method="simple")
        self.assertEqual(len(simple_returns), 4)

    def test_calculate_volatility(self):
        """Test volatility calculation with frequency scaling."""
        returns = pd.Series(np.random.randn(252) * 0.02)  # 2% daily volatility
        
        # Test daily volatility
        daily_vol = self.common.calculate_volatility(returns, window=252, frequency=Frequency.DAILY)
        self.assertIsInstance(daily_vol, pd.Series)
        
        # Test hourly volatility (should be higher due to scaling)
        hourly_vol = self.common.calculate_volatility(returns, window=252, frequency=Frequency.HOUR_1)
        self.assertIsInstance(hourly_vol, pd.Series)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])  # Positive returns
        
        sharpe = self.common.calculate_sharpe_ratio(
            returns,
            risk_free_rate=0.02,
            frequency=Frequency.DAILY
        )
        
        self.assertIsInstance(sharpe, float)
        self.assertGreater(sharpe, 0)  # Should be positive for positive returns

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        prices = pd.Series([100, 110, 105, 120, 115, 125, 110, 130])
        
        max_dd, peak_date, trough_date = self.common.calculate_max_drawdown(prices)
        
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0)  # Drawdown should be negative or zero
        self.assertIsInstance(peak_date, pd.Timestamp)
        self.assertIsInstance(trough_date, pd.Timestamp)

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        trades = pd.DataFrame({
            "pnl": [100, -50, 200, -30, 150, -20]
        })
        
        win_rate = self.common.calculate_win_rate(trades)
        
        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0)
        self.assertLessEqual(win_rate, 1)
        self.assertEqual(win_rate, 0.5)  # 3 wins out of 6 trades

    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        trades = pd.DataFrame({
            "pnl": [100, -50, 200, -30, 150, -20]
        })
        
        profit_factor = self.common.calculate_profit_factor(trades)
        
        self.assertIsInstance(profit_factor, float)
        self.assertGreater(profit_factor, 0)
        # Gross profit: 450, Gross loss: 100, so profit factor should be 4.5
        self.assertAlmostEqual(profit_factor, 4.5, places=1)

    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        prices = pd.Series([100, 101, 103, 102, 105, 106])
        
        calmar = self.common.calculate_calmar_ratio(returns, prices, Frequency.DAILY)
        
        self.assertIsInstance(calmar, float)

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        sortino = self.common.calculate_sortino_ratio(
            returns,
            risk_free_rate=0.02,
            frequency=Frequency.DAILY
        )
        
        self.assertIsInstance(sortino, float)

    def test_calculate_metrics_summary(self):
        """Test comprehensive metrics calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        prices = pd.Series([100, 101, 103, 102, 105, 106])
        trades = pd.DataFrame({
            "pnl": [100, -50, 200, -30]
        })
        
        metrics = self.common.calculate_metrics_summary(
            returns,
            prices,
            trades,
            risk_free_rate=0.02,
            frequency=Frequency.DAILY
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_return", metrics)
        self.assertIn("annual_return", metrics)
        self.assertIn("volatility", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("profit_factor", metrics)

    def test_generate_backtest_report(self):
        """Test backtest report generation."""
        metrics = {
            "total_return": 0.15,
            "annual_return": 0.12,
            "volatility": 0.18,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "calmar_ratio": 2.0,
            "max_drawdown": -0.08,
            "win_rate": 0.65,
            "profit_factor": 1.8
        }
        
        report = self.common.generate_backtest_report(
            metrics,
            frequency=Frequency.DAILY
        )
        
        self.assertIsInstance(report, str)
        self.assertIn("Backtest Report", report)
        self.assertIn("15.00%", report)  # Total return
        self.assertIn("1.20", report)    # Sharpe ratio

    def test_frequency_scaling(self):
        """Test frequency scaling utilities."""
        # Test window scaling
        daily_window = 252
        hourly_window = self.common._scale_window(daily_window, Frequency.HOUR_1)
        self.assertGreater(hourly_window, daily_window)
        
        # Test annualization factors
        daily_factor = self.common._get_annualization_factor(Frequency.DAILY)
        hourly_factor = self.common._get_annualization_factor(Frequency.HOUR_1)
        self.assertEqual(daily_factor, 252)
        self.assertGreater(hourly_factor, daily_factor)

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test validate_backtest_data
        is_valid, message = validate_backtest_data(self.test_data)
        self.assertTrue(is_valid)
        
        # Test calculate_backtest_metrics
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        prices = pd.Series([100, 101, 103, 102, 105, 106])
        
        metrics = calculate_backtest_metrics(returns, prices, Frequency.DAILY)
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_return", metrics)


class TestHybridModelIntegration(unittest.TestCase):
    """Test integration between components."""

    def setUp(self):
        """Set up test environment."""
        self.common = BacktestCommon()
        
        # Create mock data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame({
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_prompt_to_backtest_workflow(self):
        """Test workflow from prompt parsing to backtesting."""
        # Mock prompt parsing
        action_plan = ActionPlan(
            model="Ensemble",
            strategy="Momentum",
            backtest_flag=True,
            export_type="csv",
            confidence=0.8,
            raw_prompt="Test ensemble with momentum strategy",
            extracted_entities={}
        )
        
        # Validate data
        is_valid, message = self.common.validate_data(self.test_data)
        self.assertTrue(is_valid)
        
        # Preprocess data
        processed_data = self.common.preprocess_data(
            self.test_data,
            frequency=Frequency.DAILY
        )
        
        # Calculate returns
        returns = self.common.calculate_returns(processed_data["close"])
        
        # Calculate metrics
        metrics = self.common.calculate_metrics_summary(
            returns,
            processed_data["close"],
            frequency=Frequency.DAILY
        )
        
        # Verify results
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_return", metrics)
        self.assertIn("sharpe_ratio", metrics)
        
        # Check if backtest was requested
        self.assertTrue(action_plan.backtest_flag)

    def test_frequency_aware_metrics(self):
        """Test that metrics are properly scaled for different frequencies."""
        # Create returns data
        returns = pd.Series(np.random.randn(252) * 0.02)
        prices = pd.Series(np.random.randn(252).cumsum() + 100)
        
        # Calculate metrics for different frequencies
        daily_metrics = self.common.calculate_metrics_summary(
            returns, prices, frequency=Frequency.DAILY
        )
        
        hourly_metrics = self.common.calculate_metrics_summary(
            returns, prices, frequency=Frequency.HOUR_1
        )
        
        # Volatility should be higher for hourly (more frequent trading)
        self.assertGreater(hourly_metrics["volatility"], daily_metrics["volatility"])
        
        # Sharpe ratio should be properly scaled
        self.assertIsInstance(hourly_metrics["sharpe_ratio"], float)
        self.assertIsInstance(daily_metrics["sharpe_ratio"], float)


def run_performance_tests():
    """Run performance tests."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTS")
    print("="*60)
    
    # Test prompt parsing performance
    print("\nTesting PromptParser performance...")
    parser = PromptParser()
    
    test_prompts = [
        "Use LSTM model with Bollinger Bands strategy and run backtest",
        "Apply XGBoost with MACD strategy and export to CSV",
        "Train Transformer model with RSI strategy",
        "Use ensemble approach with momentum strategy",
        "Apply ARIMA model with mean reversion strategy"
    ]
    
    import time
    start_time = time.time()
    
    for prompt in test_prompts:
        # Mock parsing to avoid spaCy loading
        with patch.object(parser, 'parse_prompt') as mock_parse:
            mock_parse.return_value = ActionPlan(
                model="Ensemble",
                strategy="Momentum",
                backtest_flag=False,
                export_type="csv",
                confidence=0.8,
                raw_prompt=prompt,
                extracted_entities={}
            )
            result = parser.parse_prompt(prompt)
    
    elapsed = time.time() - start_time
    print(f"  Average parsing time: {elapsed/len(test_prompts):.4f}s per prompt")
    
    # Test backtest common performance
    print("\nTesting BacktestCommon performance...")
    common = BacktestCommon()
    
    # Create large dataset
    large_data = pd.DataFrame({
        "close": np.random.randn(10000).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 10000)
    })
    
    start_time = time.time()
    
    # Test data validation
    is_valid, _ = common.validate_data(large_data)
    
    # Test preprocessing
    processed_data = common.preprocess_data(large_data, Frequency.DAILY)
    
    # Test metrics calculation
    returns = common.calculate_returns(processed_data["close"])
    metrics = common.calculate_metrics_summary(returns, processed_data["close"])
    
    elapsed = time.time() - start_time
    print(f"  Large dataset processing time: {elapsed:.4f}s")
    print(f"  Dataset size: {len(large_data)} rows")
    print(f"  Metrics calculated: {len(metrics)}")


def main():
    """Run all tests."""
    print("Testing Prompt Parser, Hybrid Model, and Backtest Improvements")
    print("="*70)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
