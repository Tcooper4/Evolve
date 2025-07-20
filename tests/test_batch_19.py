"""
Batch 19 Tests
Tests for fault tolerance, safety, and generalization improvements
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import Batch 19 modules
from trading.nlp.prompt_bridge import PromptBridge, IntentType
from trading.context_manager.trading_context import TradingContextManager, SessionStatus, StrategyType
from trading.signal_score.evaluator import SignalScoreEvaluator, SignalType, StrategyType as EvalStrategyType
from trading.meta_learning.model_swapper import ModelSwapper, ValidationResult, ModelMetrics


class TestPromptBridge(unittest.TestCase):
    """Test PromptBridge compound intent handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bridge = PromptBridge(enable_regex_fallback=True)
    
    def test_compound_prompt_detection(self):
        """Test detection of compound prompts."""
        # Test conjunction-based compound
        compound_prompt = "run forecast and apply RSI+MACD strategy"
        self.assertTrue(self.bridge._is_compound_prompt(compound_prompt))
        
        # Test comma-separated compound
        comma_prompt = "analyze market, generate signals, execute trades"
        self.assertTrue(self.bridge._is_compound_prompt(comma_prompt))
        
        # Test single intent
        single_prompt = "run RSI strategy"
        self.assertFalse(self.bridge._is_compound_prompt(single_prompt))
    
    def test_split_intents_conjunction(self):
        """Test splitting intents by conjunctions."""
        compound_prompt = "forecast prices and apply RSI strategy"
        intents = self.bridge.split_intents(compound_prompt)
        
        self.assertEqual(len(intents), 1)  # Should create compound intent
        self.assertEqual(intents[0].intent_type, IntentType.COMPOUND)
        self.assertEqual(len(intents[0].sub_intents), 2)
    
    def test_split_intents_comma(self):
        """Test splitting intents by commas."""
        compound_prompt = "analyze market, generate signals, execute trades"
        intents = self.bridge.split_intents(compound_prompt)
        
        self.assertEqual(len(intents), 1)  # Should create compound intent
        self.assertEqual(intents[0].intent_type, IntentType.COMPOUND)
        self.assertGreaterEqual(len(intents[0].sub_intents), 3)
    
    def test_regex_fallback(self):
        """Test regex fallback for unknown intents."""
        unknown_prompt = "do something with XYZ stock"
        result = self.bridge.parse_prompt(unknown_prompt)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.intents), 0)
    
    def test_dependency_resolution(self):
        """Test intent dependency resolution."""
        intents = [
            Mock(intent_type=IntentType.EXECUTION),
            Mock(intent_type=IntentType.ANALYSIS),
            Mock(intent_type=IntentType.FORECAST)
        ]
        
        ordered = self.bridge.resolve_dependencies(intents)
        
        # Analysis should come first, execution last
        self.assertEqual(ordered[0].intent_type, IntentType.ANALYSIS)
        self.assertEqual(ordered[-1].intent_type, IntentType.EXECUTION)


class TestTradingContextManager(unittest.TestCase):
    """Test TradingContextManager session management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context_manager = TradingContextManager(max_active_strategies=5)
    
    def test_session_creation(self):
        """Test session creation with max_age."""
        session_id = self.context_manager.create_session(
            user_id="test_user",
            max_age=timedelta(hours=2)
        )
        
        session = self.context_manager.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, "test_user")
        self.assertEqual(session.max_age, timedelta(hours=2))
    
    def test_session_expiration(self):
        """Test session auto-expiration."""
        # Create session with short max_age
        session_id = self.context_manager.create_session(
            user_id="test_user",
            max_age=timedelta(seconds=1)
        )
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Trigger cleanup
        expired_count = self.context_manager.cleanup_expired_sessions()
        self.assertGreater(expired_count, 0)
        
        # Session should be expired
        session = self.context_manager.get_session(session_id)
        self.assertIsNone(session)
    
    def test_strategy_limit(self):
        """Test maximum active strategies limit."""
        session_id = self.context_manager.create_session("test_user")
        
        # Try to register more strategies than limit
        for i in range(7):  # More than max_active_strategies (5)
            success = self.context_manager.register_strategy(
                session_id=session_id,
                strategy_id=f"strategy_{i}",
                strategy_type=StrategyType.FORECAST
            )
            
            if i < 5:
                self.assertTrue(success)
            else:
                self.assertFalse(success)  # Should fail after limit
    
    def test_strategy_unregistration(self):
        """Test strategy unregistration."""
        session_id = self.context_manager.create_session("test_user")
        
        # Register strategy
        success = self.context_manager.register_strategy(
            session_id=session_id,
            strategy_id="test_strategy",
            strategy_type=StrategyType.SIGNAL
        )
        self.assertTrue(success)
        
        # Unregister strategy
        success = self.context_manager.unregister_strategy("test_strategy")
        self.assertTrue(success)
        
        # Check that strategy is removed
        self.assertEqual(len(self.context_manager.active_strategies), 0)
    
    def test_session_statistics(self):
        """Test session statistics generation."""
        session_id = self.context_manager.create_session("test_user")
        
        # Register a strategy
        self.context_manager.register_strategy(
            session_id=session_id,
            strategy_id="test_strategy",
            strategy_type=StrategyType.EXECUTION
        )
        
        stats = self.context_manager.get_session_statistics(session_id)
        
        self.assertEqual(stats['session_id'], session_id)
        self.assertEqual(stats['strategy_count'], 1)
        self.assertIn('age_seconds', stats)
        self.assertIn('inactive_seconds', stats)


class TestSignalScoreEvaluator(unittest.TestCase):
    """Test SignalScoreEvaluator with NaN protection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = SignalScoreEvaluator(enable_nan_protection=True)
    
    def test_nan_protection(self):
        """Test NaN protection in signal evaluation."""
        # Test with NaN values
        signal_data = {
            'rsi': np.nan,
            'current_price': 100.0,
            'sma_short': np.inf
        }
        
        score = self.evaluator.evaluate_signal(
            signal_data=signal_data,
            strategy_type="RSI"
        )
        
        # Should handle NaN gracefully
        self.assertIsNotNone(score)
        self.assertNotEqual(score.score, np.nan)
        self.assertNotEqual(score.score, np.inf)
    
    def test_sma_signal_evaluation(self):
        """Test SMA signal evaluation."""
        signal_data = {
            'current_price': 105.0,
            'sma_short': 102.0,
            'sma_long': 100.0
        }
        
        score = self.evaluator.evaluate_signal(
            signal_data=signal_data,
            strategy_type="SMA"
        )
        
        self.assertIsNotNone(score)
        self.assertGreater(score.score, 0)  # Should be bullish
        self.assertEqual(score.signal_type, SignalType.BUY)
    
    def test_bollinger_signal_evaluation(self):
        """Test Bollinger Bands signal evaluation."""
        signal_data = {
            'current_price': 95.0,  # Near lower band
            'upper_band': 105.0,
            'lower_band': 95.0,
            'middle_band': 100.0
        }
        
        score = self.evaluator.evaluate_signal(
            signal_data=signal_data,
            strategy_type="BB"
        )
        
        self.assertIsNotNone(score)
        self.assertLess(score.score, 0)  # Should be buy signal (negative score for strong buy)
        self.assertEqual(score.signal_type, SignalType.STRONG_BUY)
    
    def test_multiple_signals_evaluation(self):
        """Test evaluation of multiple signals."""
        signals = [
            {
                'strategy_type': 'RSI',
                'signal_data': {'rsi': 25.0},
                'parameters': {'oversold': 30.0, 'overbought': 70.0}
            },
            {
                'strategy_type': 'SMA',
                'signal_data': {'current_price': 105.0, 'sma_short': 102.0, 'sma_long': 100.0},
                'parameters': {}
            }
        ]
        
        result = self.evaluator.evaluate_multiple_signals(signals)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.signal_scores), 2)
        self.assertIsNotNone(result.composite_score)
        self.assertIsNotNone(result.recommended_action)
    
    def test_custom_strategy_registration(self):
        """Test custom strategy registration."""
        def custom_evaluator(signal_data, parameters):
            return 0.5, 0.8, SignalType.BUY
        
        success = self.evaluator.register_custom_strategy("CUSTOM_RSI", custom_evaluator)
        self.assertTrue(success)
        
        # Test custom strategy evaluation
        score = self.evaluator.evaluate_signal(
            signal_data={'test': 1.0},
            strategy_type="CUSTOM_RSI"
        )
        
        self.assertIsNotNone(score)
        self.assertEqual(score.score, 0.5)
        self.assertEqual(score.confidence, 0.8)


class TestModelSwapper(unittest.TestCase):
    """Test ModelSwapper validation and safety checks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.swapper = ModelSwapper(min_sharpe_ratio=0.8, max_mse_increase=0.1)
    
    def test_sharpe_validation(self):
        """Test Sharpe ratio validation."""
        # Test candidate with low Sharpe
        metrics = ModelMetrics(
            model_id="test_model",
            sharpe_ratio=0.5,  # Below threshold
            mse=0.01,
            win_rate=0.6,
            total_return=0.1,
            max_drawdown=0.1,
            volatility=0.2,
            last_updated=datetime.now()
        )
        
        result = self.swapper._validate_candidate_model(metrics)
        self.assertEqual(result, ValidationResult.FAILED_SHARPE)
    
    def test_mse_validation(self):
        """Test MSE validation."""
        # Add a current model to compare against
        current_metrics = ModelMetrics(
            model_id="current_model",
            sharpe_ratio=1.0,
            mse=0.01,  # Current MSE
            win_rate=0.6,
            total_return=0.1,
            max_drawdown=0.1,
            volatility=0.2,
            last_updated=datetime.now()
        )
        self.swapper.active_models["current_model"] = current_metrics
        
        # Test candidate with high MSE increase
        candidate_metrics = ModelMetrics(
            model_id="candidate_model",
            sharpe_ratio=1.2,
            mse=0.015,  # 50% increase (above 10% threshold)
            win_rate=0.6,
            total_return=0.1,
            max_drawdown=0.1,
            volatility=0.2,
            last_updated=datetime.now()
        )
        
        result = self.swapper._validate_candidate_model(candidate_metrics)
        self.assertEqual(result, ValidationResult.FAILED_MSE)
    
    def test_safety_checks(self):
        """Test additional safety checks."""
        # Test candidate with unreasonable metrics
        metrics = ModelMetrics(
            model_id="test_model",
            sharpe_ratio=1.0,
            mse=0.01,
            win_rate=0.95,  # Too high
            total_return=0.1,
            max_drawdown=0.6,  # Too high
            volatility=0.4,  # Too high
            last_updated=datetime.now()
        )
        
        result = self.swapper._validate_candidate_model(metrics)
        self.assertEqual(result, ValidationResult.FAILED_OTHER)
    
    def test_valid_candidate_acceptance(self):
        """Test acceptance of valid candidate."""
        metrics = ModelMetrics(
            model_id="valid_model",
            sharpe_ratio=1.2,  # Above threshold
            mse=0.008,  # Better than current
            win_rate=0.65,
            total_return=0.15,
            max_drawdown=0.1,
            volatility=0.2,
            last_updated=datetime.now()
        )
        
        result = self.swapper._validate_candidate_model(metrics)
        self.assertEqual(result, ValidationResult.PASSED)
    
    def test_candidate_addition(self):
        """Test adding valid candidate model."""
        metrics = {
            'sharpe_ratio': 1.2,
            'mse': 0.008,
            'win_rate': 0.65,
            'total_return': 0.15,
            'max_drawdown': 0.1,
            'volatility': 0.2
        }
        
        success = self.swapper.add_candidate_model(
            model_id="test_candidate",
            model_type="LSTM",
            metrics=metrics,
            confidence_score=0.8
        )
        
        self.assertTrue(success)
        self.assertIn("test_candidate", self.swapper.candidate_models)
    
    def test_model_swapping(self):
        """Test complete model swapping process."""
        # Add current model
        current_metrics = ModelMetrics(
            model_id="current_model",
            sharpe_ratio=0.9,
            mse=0.01,
            win_rate=0.6,
            total_return=0.1,
            max_drawdown=0.1,
            volatility=0.2,
            last_updated=datetime.now()
        )
        self.swapper.active_models["current_model"] = current_metrics
        
        # Add candidate model
        candidate_metrics = {
            'sharpe_ratio': 1.3,
            'mse': 0.008,
            'win_rate': 0.7,
            'total_return': 0.2,
            'max_drawdown': 0.08,
            'volatility': 0.18
        }
        
        self.swapper.add_candidate_model(
            model_id="candidate_model",
            model_type="LSTM",
            metrics=candidate_metrics,
            confidence_score=0.9
        )
        
        # Perform swap
        result = self.swapper.swap_model("candidate_model", "current_model")
        
        self.assertTrue(result.success)
        self.assertEqual(result.old_model_id, "current_model")
        self.assertEqual(result.new_model_id, "candidate_model")
        self.assertGreater(result.performance_improvement, 0)


class TestBatch19Integration(unittest.TestCase):
    """Integration tests for Batch 19 modules."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow with all Batch 19 modules."""
        # Create instances
        prompt_bridge = PromptBridge()
        context_manager = TradingContextManager(max_active_strategies=3)
        signal_evaluator = SignalScoreEvaluator()
        model_swapper = ModelSwapper()
        
        # Test compound prompt parsing
        compound_prompt = "analyze market data and apply RSI strategy with MACD confirmation"
        parse_result = prompt_bridge.parse_prompt(compound_prompt)
        
        self.assertTrue(parse_result.success)
        self.assertTrue(parse_result.compound_detected)
        
        # Test session creation and strategy registration
        session_id = context_manager.create_session("test_user")
        success = context_manager.register_strategy(
            session_id=session_id,
            strategy_id="test_strategy",
            strategy_type=StrategyType.SIGNAL
        )
        self.assertTrue(success)
        
        # Test signal evaluation with NaN protection
        signal_data = {'rsi': 25.0, 'current_price': 100.0}
        signal_score = signal_evaluator.evaluate_signal(
            signal_data=signal_data,
            strategy_type="RSI"
        )
        self.assertIsNotNone(signal_score)
        
        # Test model swapping validation
        candidate_metrics = {
            'sharpe_ratio': 1.2,
            'mse': 0.008,
            'win_rate': 0.65,
            'total_return': 0.15,
            'max_drawdown': 0.1,
            'volatility': 0.2
        }
        
        success = model_swapper.add_candidate_model(
            model_id="test_candidate",
            model_type="LSTM",
            metrics=candidate_metrics
        )
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
