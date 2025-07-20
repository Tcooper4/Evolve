"""
Batch 11 Tests

Comprehensive test suite for Batch 11 implementations:
- Strategy composer with runtime toggles
- Enhanced logging with file output and rotating handlers
- Risk manager with dynamic volatility models
- Visualizer with input validation
- Math helpers without NumPy wrappers
"""

import unittest
import tempfile
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import Batch 11 modules
try:
    from trading.strategies.strategy_composer import (
        StrategyComposer, SubStrategyConfig, StrategyToggle
    )
    STRATEGY_COMPOSER_AVAILABLE = True
except ImportError:
    STRATEGY_COMPOSER_AVAILABLE = False

try:
    from utils.logging import EnhancedLogManager
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False

try:
    from trading.risk.risk_manager import (
        RiskManager, DynamicVolatilityModel, VolatilityModel,
        VolatilityForecast, PositionSizeRecommendation
    )
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

try:
    from trading.visualization.visualizer import (
        EnhancedVisualizer, VisualizationError, DataValidationError
    )
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

try:
    from utils.math_helpers import (
        calculate_rolling_statistics,
        calculate_percentile_ranks,
        calculate_zscore,
        calculate_momentum_score,
        calculate_regime_probability,
        calculate_tail_risk_metrics,
        calculate_volatility_regime,
        calculate_correlation_regime,
        calculate_entropy,
        calculate_information_ratio,
        calculate_treynor_ratio,
        calculate_jensen_alpha,
        calculate_sortino_ratio,
        calculate_calmar_ratio
    )
    MATH_HELPERS_AVAILABLE = True
except ImportError:
    MATH_HELPERS_AVAILABLE = False


class TestStrategyComposer(unittest.TestCase):
    """Test strategy composer with runtime toggles."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STRATEGY_COMPOSER_AVAILABLE:
            self.skipTest("Strategy composer not available")
        
        self.composer = StrategyComposer()
        
        # Sample data
        self.sample_data = {
            "strategy_1": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
                "signal_strength": np.random.randn(100),
                "confidence": np.random.uniform(0.5, 0.9, 100)
            }),
            "strategy_2": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
                "signal_strength": np.random.randn(100),
                "confidence": np.random.uniform(0.5, 0.9, 100)
            })
        }
    
    def test_add_sub_strategy(self):
        """Test adding sub-strategies."""
        result = self.composer.add_sub_strategy(
            name="test_strategy",
            enabled=True,
            weight=1.5,
            priority=2,
            conditions={"volatility_threshold": 0.2},
            parameters={"window": 20},
            performance_threshold=0.1
        )
        
        self.assertTrue(result)
        self.assertIn("test_strategy", self.composer.sub_strategies)
        
        strategy = self.composer.sub_strategies["test_strategy"]
        self.assertEqual(strategy.name, "test_strategy")
        self.assertTrue(strategy.enabled)
        self.assertEqual(strategy.weight, 1.5)
        self.assertEqual(strategy.priority, 2)
        self.assertEqual(strategy.conditions["volatility_threshold"], 0.2)
        self.assertEqual(strategy.parameters["window"], 20)
        self.assertEqual(strategy.performance_threshold, 0.1)
    
    def test_toggle_sub_strategy(self):
        """Test toggling sub-strategies."""
        # Add strategy first
        self.composer.add_sub_strategy("test_strategy", enabled=True)
        
        # Toggle off
        result = self.composer.toggle_sub_strategy(
            "test_strategy", 
            False, 
            "test reason"
        )
        
        self.assertTrue(result)
        self.assertFalse(self.composer.sub_strategies["test_strategy"].enabled)
        self.assertEqual(
            self.composer.sub_strategies["test_strategy"].toggle_state,
            StrategyToggle.DISABLED
        )
        
        # Check history
        self.assertEqual(len(self.composer.strategy_history), 1)
        history_entry = self.composer.strategy_history[0]
        self.assertEqual(history_entry["strategy"], "test_strategy")
        self.assertEqual(history_entry["old_state"], True)
        self.assertEqual(history_entry["new_state"], False)
        self.assertEqual(history_entry["reason"], "test reason")
    
    def test_conditional_toggle(self):
        """Test conditional toggle setting."""
        # Add strategy first
        self.composer.add_sub_strategy("test_strategy", enabled=True)
        
        # Set conditional toggle
        conditions = {
            "volatility_threshold": 0.2,
            "market_regime": ["bull", "neutral"]
        }
        
        result = self.composer.set_conditional_toggle(
            "test_strategy",
            conditions,
            "conditional test"
        )
        
        self.assertTrue(result)
        self.assertEqual(
            self.composer.sub_strategies["test_strategy"].toggle_state,
            StrategyToggle.CONDITIONAL
        )
        self.assertEqual(
            self.composer.sub_strategies["test_strategy"].conditions,
            conditions
        )
    
    def test_evaluate_conditions(self):
        """Test condition evaluation."""
        # Add strategy with conditions
        self.composer.add_sub_strategy(
            "test_strategy",
            conditions={"volatility_threshold": 0.2}
        )
        self.composer.set_conditional_toggle("test_strategy", {"volatility_threshold": 0.2})
        
        # Test market data
        market_data = {"volatility": 0.15}  # Below threshold
        results = self.composer.evaluate_conditions(market_data)
        
        self.assertIn("test_strategy", results)
        self.assertTrue(results["test_strategy"])  # Should be enabled
    
    def test_compose_signals(self):
        """Test signal composition."""
        # Add strategies
        self.composer.add_sub_strategy("strategy_1", weight=1.0)
        self.composer.add_sub_strategy("strategy_2", weight=2.0)
        
        # Compose signals
        result = self.composer.compose_signals(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        
        # Check that weights are applied
        if "weight" in result.columns:
            weights = result["weight"].unique()
            self.assertIn(1.0, weights)
            self.assertIn(2.0, weights)
    
    def test_get_active_strategies(self):
        """Test getting active strategies."""
        # Add strategies with different states
        self.composer.add_sub_strategy("strategy_1", enabled=True)
        self.composer.add_sub_strategy("strategy_2", enabled=False)
        self.composer.add_sub_strategy("strategy_3", enabled=True)
        
        active = self.composer.get_active_strategies()
        
        self.assertIn("strategy_1", active)
        self.assertIn("strategy_3", active)
        self.assertNotIn("strategy_2", active)
    
    def test_get_strategy_weights(self):
        """Test getting strategy weights."""
        # Add strategies with different weights
        self.composer.add_sub_strategy("strategy_1", weight=1.0)
        self.composer.add_sub_strategy("strategy_2", weight=2.0)
        
        weights = self.composer.get_strategy_weights()
        
        self.assertEqual(weights["strategy_1"], 1.0 / 3.0)
        self.assertEqual(weights["strategy_2"], 2.0 / 3.0)


class TestEnhancedLogging(unittest.TestCase):
    """Test enhanced logging with file output and rotating handlers."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not ENHANCED_LOGGING_AVAILABLE:
            self.skipTest("Enhanced logging not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.log_manager = EnhancedLogManager(
            log_dir=self.temp_dir,
            enable_file_output=True,
            enable_rotating_handlers=True
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_enhanced_logger_creation(self):
        """Test creating enhanced logger."""
        logger = self.log_manager.get_enhanced_logger(
            "test_logger",
            log_file="test.log",
            enable_console=True,
            enable_file=True
        )
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_logger")
        self.assertGreater(len(logger.handlers), 0)
    
    def test_timed_rotating_handler(self):
        """Test timed rotating handler."""
        logger = self.log_manager.setup_timed_rotating_handler(
            "timed_logger",
            "timed.log",
            when="midnight",
            interval=1,
            backup_count=5
        )
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "timed_logger")
        
        # Check handlers
        handler_types = [type(h) for h in logger.handlers]
        self.assertIn(logging.handlers.TimedRotatingFileHandler, handler_types)
    
    def test_memory_handler(self):
        """Test memory handler."""
        logger = self.log_manager.setup_memory_handler(
            "memory_logger",
            capacity=100,
            flushLevel=logging.ERROR
        )
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "memory_logger")
        
        # Check handlers
        handler_types = [type(h) for h in logger.handlers]
        self.assertIn(logging.handlers.MemoryHandler, handler_types)
    
    def test_queue_handler(self):
        """Test queue handler."""
        logger = self.log_manager.setup_queue_handler(
            "queue_logger",
            queue_size=100
        )
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "queue_logger")
    
    def test_log_level_setting(self):
        """Test log level setting."""
        logger = self.log_manager.get_enhanced_logger("level_test")
        
        # Set level
        self.log_manager.set_log_level(logging.DEBUG)
        self.assertEqual(logger.level, logging.DEBUG)
        
        self.log_manager.set_log_level(logging.INFO)
        self.assertEqual(logger.level, logging.INFO)
    
    def test_log_files_info(self):
        """Test getting log files information."""
        # Create some log files
        logger = self.log_manager.get_enhanced_logger("file_test")
        logger.info("Test message")
        
        log_files = self.log_manager.get_log_files()
        
        self.assertIsInstance(log_files, dict)
        self.assertGreater(len(log_files), 0)
        
        # Check file info structure
        for filename, info in log_files.items():
            self.assertIn("size_kb", info)
            self.assertIn("size_bytes", info)
            self.assertIn("modified", info)
            self.assertIn("created", info)
    
    def test_cleanup_old_logs(self):
        """Test cleanup of old logs."""
        # Create some log files
        logger = self.log_manager.get_enhanced_logger("cleanup_test")
        logger.info("Test message")
        
        # Test cleanup
        self.log_manager.cleanup_old_logs(days=1)
        
        # Should not raise any errors
        self.assertTrue(True)
    
    def test_logger_stats(self):
        """Test getting logger statistics."""
        # Create some loggers
        self.log_manager.get_enhanced_logger("stats_test_1")
        self.log_manager.get_enhanced_logger("stats_test_2")
        
        stats = self.log_manager.get_logger_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_loggers", stats)
        self.assertIn("log_files", stats)
        self.assertIn("handlers", stats)
        
        self.assertEqual(stats["total_loggers"], 2)
    
    def test_flush_and_close_handlers(self):
        """Test flushing and closing handlers."""
        logger = self.log_manager.get_enhanced_logger("flush_test")
        logger.info("Test message")
        
        # Test flush
        self.log_manager.flush_all_handlers()
        
        # Test close
        self.log_manager.close_all_handlers()
        
        # Should not raise any errors
        self.assertTrue(True)


class TestRiskManager(unittest.TestCase):
    """Test risk manager with dynamic volatility models."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not RISK_MANAGER_AVAILABLE:
            self.skipTest("Risk manager not available")
        
        self.risk_manager = RiskManager()
        
        # Sample returns data
        np.random.seed(42)
        self.returns = pd.Series(
            np.random.randn(252) * 0.02,
            index=pd.date_range("2023-01-01", periods=252, freq="D")
        )
        
        self.risk_manager.update_returns(self.returns)
    
    def test_dynamic_volatility_model_initialization(self):
        """Test dynamic volatility model initialization."""
        self.assertIsInstance(self.risk_manager.volatility_model, DynamicVolatilityModel)
        
        # Test different model types
        for model_type in VolatilityModel:
            model = DynamicVolatilityModel(model_type=model_type)
            self.assertEqual(model.model_type, model_type)
    
    def test_rolling_volatility_calculation(self):
        """Test rolling volatility calculation."""
        model = DynamicVolatilityModel(model_type=VolatilityModel.ROLLING_STD)
        vol_series = model.calculate_rolling_volatility(self.returns)
        
        self.assertIsInstance(vol_series, pd.Series)
        self.assertEqual(len(vol_series), len(self.returns))
        self.assertTrue((vol_series >= 0).all())
    
    def test_ewma_volatility_calculation(self):
        """Test EWMA volatility calculation."""
        model = DynamicVolatilityModel(model_type=VolatilityModel.EWMA)
        vol_series = model.calculate_ewma_volatility(self.returns)
        
        self.assertIsInstance(vol_series, pd.Series)
        self.assertEqual(len(vol_series), len(self.returns))
        self.assertTrue((vol_series >= 0).all())
    
    def test_hybrid_volatility_calculation(self):
        """Test hybrid volatility calculation."""
        model = DynamicVolatilityModel(model_type=VolatilityModel.HYBRID)
        vol_series = model.calculate_hybrid_volatility(self.returns)
        
        self.assertIsInstance(vol_series, pd.Series)
        self.assertEqual(len(vol_series), len(self.returns))
        self.assertTrue((vol_series >= 0).all())
    
    def test_volatility_forecasting(self):
        """Test volatility forecasting."""
        model = DynamicVolatilityModel(model_type=VolatilityModel.ROLLING_STD)
        forecast = model.forecast_volatility(self.returns, horizon=5)
        
        self.assertIsInstance(forecast, VolatilityForecast)
        self.assertIsInstance(forecast.timestamp, str)
        self.assertIsInstance(forecast.current_volatility, float)
        self.assertIsInstance(forecast.forecasted_volatility, float)
        self.assertIsInstance(forecast.confidence_interval, tuple)
        self.assertEqual(len(forecast.confidence_interval), 2)
        self.assertIsInstance(forecast.model_type, str)
        self.assertIsInstance(forecast.model_parameters, dict)
        self.assertEqual(forecast.forecast_horizon, 5)
    
    def test_dynamic_position_sizing(self):
        """Test dynamic position sizing."""
        recommendation = self.risk_manager.calculate_dynamic_position_size(
            symbol="AAPL",
            base_position_size=0.05,
            confidence_score=0.8
        )
        
        self.assertIsInstance(recommendation, PositionSizeRecommendation)
        self.assertEqual(recommendation.symbol, "AAPL")
        self.assertEqual(recommendation.base_position_size, 0.05)
        self.assertEqual(recommendation.confidence_score, 0.8)
        self.assertIsInstance(recommendation.final_recommendation, float)
        self.assertIsInstance(recommendation.volatility_model, str)
        self.assertIsInstance(recommendation.risk_factors, dict)
        
        # Check that final recommendation is within bounds
        self.assertGreaterEqual(recommendation.final_recommendation, 0)
        self.assertLessEqual(recommendation.final_recommendation, 0.1)  # max position size
    
    def test_volatility_forecast_retrieval(self):
        """Test volatility forecast retrieval."""
        forecast = self.risk_manager.get_volatility_forecast()
        
        self.assertIsInstance(forecast, VolatilityForecast)
        self.assertGreater(forecast.current_volatility, 0)
        self.assertGreater(forecast.forecasted_volatility, 0)
    
    def test_volatility_model_update(self):
        """Test volatility model update."""
        result = self.risk_manager.update_volatility_model(
            VolatilityModel.EWMA,
            window=100,
            alpha=0.95
        )
        
        self.assertTrue(result)
        self.assertEqual(self.risk_manager.volatility_model.model_type, VolatilityModel.EWMA)
        self.assertEqual(self.risk_manager.volatility_model.window, 100)
        self.assertEqual(self.risk_manager.volatility_model.alpha, 0.95)


class TestVisualizer(unittest.TestCase):
    """Test visualizer with input validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not VISUALIZER_AVAILABLE:
            self.skipTest("Visualizer not available")
        
        self.visualizer = EnhancedVisualizer()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "volume": np.random.randint(1000000, 10000000, 100)
        })
        
        self.price_data = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            "open": 100 + np.random.randn(100) * 0.5,
            "high": 101 + np.random.randn(100) * 0.5,
            "low": 99 + np.random.randn(100) * 0.5,
            "close": 100 + np.random.randn(100) * 0.5,
            "volume": np.random.randint(1000000, 10000000, 100)
        })
    
    def test_dataframe_validation_success(self):
        """Test successful DataFrame validation."""
        result = self.visualizer.validate_dataframe(
            self.sample_data,
            required_columns=["timestamp", "close"],
            numeric_columns=["close", "volume"]
        )
        
        self.assertTrue(result)
    
    def test_dataframe_validation_failure(self):
        """Test DataFrame validation failure."""
        # Test with missing required column
        with self.assertRaises(DataValidationError):
            self.visualizer.validate_dataframe(
                self.sample_data,
                required_columns=["timestamp", "missing_column"]
            )
        
        # Test with empty DataFrame
        with self.assertRaises(DataValidationError):
            self.visualizer.validate_dataframe(pd.DataFrame())
        
        # Test with None
        with self.assertRaises(DataValidationError):
            self.visualizer.validate_dataframe(None)
    
    def test_price_data_validation(self):
        """Test price data validation."""
        result = self.visualizer.validate_price_data(self.price_data)
        self.assertTrue(result)
    
    def test_volume_data_validation(self):
        """Test volume data validation."""
        volume_data = self.sample_data[["timestamp", "volume"]]
        result = self.visualizer.validate_volume_data(volume_data)
        self.assertTrue(result)
    
    def test_indicator_data_validation(self):
        """Test indicator data validation."""
        indicator_data = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            "rsi": np.random.uniform(0, 100, 100),
            "macd": np.random.randn(100)
        })
        
        result = self.visualizer.validate_indicator_data(indicator_data, "RSI")
        self.assertTrue(result)
    
    def test_candlestick_chart_creation(self):
        """Test candlestick chart creation."""
        fig = self.visualizer.plot_candlestick_chart(
            self.price_data,
            title="Test Candlestick"
        )
        
        self.assertIsInstance(fig, go.Figure)
    
    def test_line_chart_creation(self):
        """Test line chart creation."""
        fig = self.visualizer.plot_line_chart(
            self.sample_data,
            y_column="close",
            title="Test Line Chart"
        )
        
        self.assertIsInstance(fig, go.Figure)
    
    def test_histogram_creation(self):
        """Test histogram creation."""
        fig = self.visualizer.plot_histogram(
            self.sample_data,
            column="close",
            title="Test Histogram"
        )
        
        self.assertIsInstance(fig, go.Figure)
    
    def test_scatter_plot_creation(self):
        """Test scatter plot creation."""
        fig = self.visualizer.plot_scatter(
            self.sample_data,
            x_column="volume",
            y_column="close",
            title="Test Scatter"
        )
        
        self.assertIsInstance(fig, go.Figure)
    
    def test_heatmap_creation(self):
        """Test heatmap creation."""
        # Create correlation matrix
        correlation_matrix = pd.DataFrame({
            "A": [1.0, 0.5, 0.3],
            "B": [0.5, 1.0, 0.7],
            "C": [0.3, 0.7, 1.0]
        }, index=["A", "B", "C"])
        
        fig = self.visualizer.plot_heatmap(
            correlation_matrix,
            title="Test Heatmap"
        )
        
        self.assertIsInstance(fig, go.Figure)
    
    def test_chart_export(self):
        """Test chart export."""
        fig = self.visualizer.plot_line_chart(
            self.sample_data,
            y_column="close"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            result = self.visualizer.export_chart(fig, tmp.name, "html")
            self.assertTrue(result)
            os.unlink(tmp.name)


class TestMathHelpers(unittest.TestCase):
    """Test math helpers without NumPy wrappers."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MATH_HELPERS_AVAILABLE:
            self.skipTest("Math helpers not available")
        
        # Sample data
        np.random.seed(42)
        self.returns = pd.Series(np.random.randn(252) * 0.02)
        self.prices = pd.Series(100 + np.cumsum(self.returns))
        self.market_returns = pd.Series(np.random.randn(252) * 0.015)
    
    def test_rolling_statistics(self):
        """Test rolling statistics calculation."""
        stats = calculate_rolling_statistics(
            self.returns,
            window=20,
            statistics=["mean", "std", "min", "max", "median"]
        )
        
        self.assertIsInstance(stats, dict)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("median", stats)
        
        for stat_name, stat_series in stats.items():
            self.assertIsInstance(stat_series, pd.Series)
            self.assertEqual(len(stat_series), len(self.returns))
    
    def test_percentile_ranks(self):
        """Test percentile ranks calculation."""
        ranks = calculate_percentile_ranks(self.returns, window=60)
        
        self.assertIsInstance(ranks, pd.Series)
        self.assertEqual(len(ranks), len(self.returns))
        
        # Check that ranks are between 0 and 1
        valid_ranks = ranks.dropna()
        self.assertTrue((valid_ranks >= 0).all())
        self.assertTrue((valid_ranks <= 1).all())
    
    def test_zscore_calculation(self):
        """Test z-score calculation."""
        zscores = calculate_zscore(self.returns, window=20)
        
        self.assertIsInstance(zscores, pd.Series)
        self.assertEqual(len(zscores), len(self.returns))
        
        # Check that z-scores have reasonable values
        valid_zscores = zscores.dropna()
        self.assertTrue((abs(valid_zscores) < 10).all())  # Reasonable range
    
    def test_momentum_score(self):
        """Test momentum score calculation."""
        momentum = calculate_momentum_score(
            self.prices,
            short_window=10,
            long_window=50
        )
        
        self.assertIsInstance(momentum, pd.Series)
        self.assertEqual(len(momentum), len(self.prices))
    
    def test_regime_probability(self):
        """Test regime probability calculation."""
        probabilities = calculate_regime_probability(
            self.returns,
            regimes=["bull", "bear", "sideways"],
            window=60
        )
        
        self.assertIsInstance(probabilities, dict)
        self.assertIn("bull", probabilities)
        self.assertIn("bear", probabilities)
        self.assertIn("sideways", probabilities)
        
        for regime, prob_series in probabilities.items():
            self.assertIsInstance(prob_series, pd.Series)
            self.assertEqual(len(prob_series), len(self.returns))
    
    def test_tail_risk_metrics(self):
        """Test tail risk metrics calculation."""
        metrics = calculate_tail_risk_metrics(self.returns, confidence_level=0.05)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("var", metrics)
        self.assertIn("cvar", metrics)
        self.assertIn("tail_dependence", metrics)
        self.assertIn("expected_shortfall", metrics)
        self.assertIn("tail_risk_ratio", metrics)
        
        # Check that metrics are reasonable
        self.assertLess(metrics["var"], 0)  # VaR should be negative
        self.assertLess(metrics["cvar"], metrics["var"])  # CVaR should be more negative
        self.assertGreaterEqual(metrics["tail_dependence"], 0)
        self.assertLessEqual(metrics["tail_dependence"], 1)
    
    def test_volatility_regime(self):
        """Test volatility regime calculation."""
        regimes = calculate_volatility_regime(self.returns, window=60, regimes=3)
        
        self.assertIsInstance(regimes, pd.Series)
        self.assertEqual(len(regimes), len(self.returns))
        
        # Check that regimes are strings
        valid_regimes = regimes.dropna()
        self.assertTrue(all(isinstance(r, str) for r in valid_regimes))
    
    def test_correlation_regime(self):
        """Test correlation regime calculation."""
        regimes = calculate_correlation_regime(
            self.returns,
            self.market_returns,
            window=60
        )
        
        self.assertIsInstance(regimes, pd.Series)
        self.assertEqual(len(regimes), len(self.returns))
        
        # Check that regimes are strings
        valid_regimes = regimes.dropna()
        self.assertTrue(all(isinstance(r, str) for r in valid_regimes))
    
    def test_entropy_calculation(self):
        """Test entropy calculation."""
        entropy = calculate_entropy(self.returns, bins=20)
        
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0)
    
    def test_information_ratio(self):
        """Test information ratio calculation."""
        ir = calculate_information_ratio(
            self.returns,
            self.market_returns,
            window=60
        )
        
        self.assertIsInstance(ir, pd.Series)
        self.assertEqual(len(ir), len(self.returns))
    
    def test_treynor_ratio(self):
        """Test Treynor ratio calculation."""
        tr = calculate_treynor_ratio(
            self.returns,
            self.market_returns,
            risk_free_rate=0.02,
            window=60
        )
        
        self.assertIsInstance(tr, pd.Series)
        self.assertEqual(len(tr), len(self.returns))
    
    def test_jensen_alpha(self):
        """Test Jensen's alpha calculation."""
        alpha = calculate_jensen_alpha(
            self.returns,
            self.market_returns,
            risk_free_rate=0.02,
            window=60
        )
        
        self.assertIsInstance(alpha, pd.Series)
        self.assertEqual(len(alpha), len(self.returns))
    
    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        sr = calculate_sortino_ratio(
            self.returns,
            risk_free_rate=0.02,
            window=60
        )
        
        self.assertIsInstance(sr, pd.Series)
        self.assertEqual(len(sr), len(self.returns))
    
    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        cr = calculate_calmar_ratio(
            self.prices,
            window=60
        )
        
        self.assertIsInstance(cr, pd.Series)
        self.assertEqual(len(cr), len(self.prices))


def run_batch_11_tests():
    """Run all Batch 11 tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestStrategyComposer,
        TestEnhancedLogging,
        TestRiskManager,
        TestVisualizer,
        TestMathHelpers
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_batch_11_tests()
    print(f"\nBatch 11 tests {'PASSED' if success else 'FAILED'}")
