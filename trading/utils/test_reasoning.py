"""
Test Reasoning System

Comprehensive tests for the reasoning logger and display components.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from utils.reasoning_logger import (
    ReasoningLogger, AgentDecision, DecisionType, ConfidenceLevel,
    DecisionContext, DecisionReasoning, log_forecast_decision, log_strategy_decision
)
from utils.reasoning_display import ReasoningDisplay

class TestReasoningLogger(unittest.TestCase):
    """Test the ReasoningLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ReasoningLogger(
            redis_host='localhost',
            redis_port=6379,
            redis_db=1,  # Use different DB for testing
            log_dir=self.temp_dir,
            enable_gpt_explanations=False  # Disable GPT for testing
        )
        
        # Sample decision data
        self.sample_decision_data = {
            'agent_name': 'TestAgent',
            'decision_type': DecisionType.FORECAST,
            'action_taken': 'Predicted AAPL will reach $185.50',
            'context': {
                'symbol': 'AAPL',
                'timeframe': '1h',
                'market_conditions': {'trend': 'bullish'},
                'available_data': ['price', 'volume'],
                'constraints': {},
                'user_preferences': {}
            },
            'reasoning': {
                'primary_reason': 'Strong technical indicators',
                'supporting_factors': ['RSI oversold', 'MACD positive'],
                'alternatives_considered': ['Wait', 'Sell'],
                'risks_assessed': ['Market volatility'],
                'confidence_explanation': 'High confidence due to clear signals',
                'expected_outcome': 'Expected 5% upside'
            },
            'confidence_level': ConfidenceLevel.HIGH,
            'metadata': {'test': True}
        }
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_log_decision(self):
        """Test decision logging."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        
        self.assertIsInstance(decision_id, str)
        self.assertTrue(decision_id.startswith('TestAgent_forecast_'))
        
        # Verify decision was stored
        decision = self.logger.get_decision(decision_id)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent_name, 'TestAgent')
        self.assertEqual(decision.decision_type, DecisionType.FORECAST)

    def test_get_decision(self):
        """Test retrieving a decision."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        decision = self.logger.get_decision(decision_id)
        
        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(decision.decision_id, decision_id)
        self.assertEqual(decision.agent_name, 'TestAgent')
        self.assertEqual(decision.context.symbol, 'AAPL')

    def test_get_agent_decisions(self):
        """Test retrieving decisions by agent."""
        # Log multiple decisions
        self.logger.log_decision(**self.sample_decision_data)
        self.logger.log_decision(**self.sample_decision_data)
        
        decisions = self.logger.get_agent_decisions('TestAgent', limit=10)
        
        self.assertIsInstance(decisions, list)
        self.assertGreaterEqual(len(decisions), 2)
        
        for decision in decisions:
            self.assertEqual(decision.agent_name, 'TestAgent')

    def test_get_decisions_by_type(self):
        """Test retrieving decisions by type."""
        # Log decisions of different types
        self.logger.log_decision(**self.sample_decision_data)
        
        strategy_data = self.sample_decision_data.copy()
        strategy_data['decision_type'] = DecisionType.STRATEGY
        self.logger.log_decision(**strategy_data)
        
        forecast_decisions = self.logger.get_decisions_by_type(DecisionType.FORECAST)
        strategy_decisions = self.logger.get_decisions_by_type(DecisionType.STRATEGY)
        
        self.assertGreaterEqual(len(forecast_decisions), 1)
        self.assertGreaterEqual(len(strategy_decisions), 1)
        
        for decision in forecast_decisions:
            self.assertEqual(decision.decision_type, DecisionType.FORECAST)

    def test_get_summary(self):
        """Test retrieving decision summary."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        summary = self.logger.get_summary(decision_id)
        
        self.assertIsInstance(summary, str)
        self.assertIn('TestAgent', summary)
        self.assertIn('AAPL', summary)
        self.assertIn('Strong technical indicators', summary)

    def test_get_explanation(self):
        """Test retrieving decision explanation."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        explanation = self.logger.get_explanation(decision_id)
        
        self.assertIsInstance(explanation, str)
        self.assertIn('TestAgent', explanation)
        self.assertIn('AAPL', explanation)

    def test_get_statistics(self):
        """Test getting statistics."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)
        self.logger.log_decision(**self.sample_decision_data)
        
        stats = self.logger.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_decisions', stats)
        self.assertIn('decisions_by_agent', stats)
        self.assertIn('decisions_by_type', stats)
        self.assertIn('confidence_distribution', stats)
        self.assertIn('recent_activity', stats)
        
        self.assertGreaterEqual(stats['total_decisions'], 2)
        self.assertIn('TestAgent', stats['decisions_by_agent'])

    def test_clear_old_decisions(self):
        """Test clearing old decisions."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        
        # Verify decision exists
        decision = self.logger.get_decision(decision_id)
        self.assertIsNotNone(decision)
        
        # Clear old decisions (should clear our test decision)
        self.logger.clear_old_decisions(days=0)
        
        # Verify decision was cleared
        decision = self.logger.get_decision(decision_id)
        self.assertIsNone(decision)

class TestReasoningDisplay(unittest.TestCase):
    """Test the ReasoningDisplay class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ReasoningLogger(
            redis_host='localhost',
            redis_port=6379,
            redis_db=2,  # Use different DB for testing
            log_dir=self.temp_dir,
            enable_gpt_explanations=False
        )
        self.display = ReasoningDisplay(self.logger)
        
        # Create sample decision
        self.sample_decision_data = {
            'agent_name': 'TestAgent',
            'decision_type': DecisionType.FORECAST,
            'action_taken': 'Predicted AAPL will reach $185.50',
            'context': {
                'symbol': 'AAPL',
                'timeframe': '1h',
                'market_conditions': {'trend': 'bullish'},
                'available_data': ['price', 'volume'],
                'constraints': {},
                'user_preferences': {}
            },
            'reasoning': {
                'primary_reason': 'Strong technical indicators',
                'supporting_factors': ['RSI oversold', 'MACD positive'],
                'alternatives_considered': ['Wait', 'Sell'],
                'risks_assessed': ['Market volatility'],
                'confidence_explanation': 'High confidence due to clear signals',
                'expected_outcome': 'Expected 5% upside'
            },
            'confidence_level': ConfidenceLevel.HIGH,
            'metadata': {'test': True}
        }
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_display_decision_terminal(self):
        """Test terminal decision display."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        decision = self.logger.get_decision(decision_id)
        
        # This should not raise an exception
        self.display.display_decision_terminal(decision)

    def test_display_recent_decisions_terminal(self):
        """Test terminal recent decisions display."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)
        self.logger.log_decision(**self.sample_decision_data)
        
        # This should not raise an exception
        self.display.display_recent_decisions_terminal(limit=5)

    def test_display_statistics_terminal(self):
        """Test terminal statistics display."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)
        
        # This should not raise an exception
        self.display.display_statistics_terminal()

    def test_display_decision_streamlit(self):
        """Test Streamlit decision display."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        decision = self.logger.get_decision(decision_id)
        
        # This should not raise an exception
        self.display.display_decision_streamlit(decision)

    def test_display_recent_decisions_streamlit(self):
        """Test Streamlit recent decisions display."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)
        self.logger.log_decision(**self.sample_decision_data)
        
        # This should not raise an exception
        self.display.display_recent_decisions_streamlit(limit=5)

    def test_display_statistics_streamlit(self):
        """Test Streamlit statistics display."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)
        
        # This should not raise an exception
        self.display.display_statistics_streamlit()

    def test_create_streamlit_sidebar(self):
        """Test Streamlit sidebar creation."""
        # Log some decisions first
        self.logger.log_decision(**self.sample_decision_data)
        
        # This should not raise an exception
        filters = self.display.create_streamlit_sidebar()
        
        self.assertIsInstance(filters, dict)
        self.assertIn('agent', filters)
        self.assertIn('type', filters)
        self.assertIn('confidence', filters)
        self.assertIn('limit', filters)

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ReasoningLogger(
            redis_host='localhost',
            redis_port=6379,
            redis_db=3,  # Use different DB for testing
            log_dir=self.temp_dir,
            enable_gpt_explanations=False
        )
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_log_forecast_decision(self):
        """Test forecast decision logging convenience function."""
        decision_id = log_forecast_decision(
            agent_name='TestForecaster',
            symbol='AAPL',
            timeframe='1h',
            forecast_value=185.50,
            confidence=0.85,
            reasoning={
                'primary_reason': 'Technical analysis',
                'supporting_factors': ['RSI', 'MACD'],
                'alternatives_considered': ['Wait'],
                'risks_assessed': ['Volatility'],
                'confidence_explanation': 'High confidence',
                'expected_outcome': '5% upside'
            }
        )
        
        self.assertIsInstance(decision_id, str)
        self.assertTrue(decision_id.startswith('TestForecaster_forecast_'))

    def test_log_strategy_decision(self):
        """Test strategy decision logging convenience function."""
        decision_id = log_strategy_decision(
            agent_name='TestStrategy',
            symbol='AAPL',
            action='BUY 100 shares',
            strategy_name='RSI Strategy',
            reasoning={
                'primary_reason': 'RSI oversold',
                'supporting_factors': ['Support level'],
                'alternatives_considered': ['Wait'],
                'risks_assessed': ['Market risk'],
                'confidence_explanation': 'Medium confidence',
                'expected_outcome': '3% gain'
            }
        )
        
        self.assertIsInstance(decision_id, str)
        self.assertTrue(decision_id.startswith('TestStrategy_strategy_'))

class TestDataStructures(unittest.TestCase):
    """Test data structures."""
    
    def test_decision_context(self):
        """Test DecisionContext dataclass."""
        context = DecisionContext(
            symbol='AAPL',
            timeframe='1h',
            timestamp='2024-01-01T12:00:00',
            market_conditions={'trend': 'bullish'},
            available_data=['price', 'volume'],
            constraints={},
            user_preferences={}
        )
        
        self.assertEqual(context.symbol, 'AAPL')
        self.assertEqual(context.timeframe, '1h')
        self.assertEqual(context.market_conditions['trend'], 'bullish')

    def test_decision_reasoning(self):
        """Test DecisionReasoning dataclass."""
        reasoning = DecisionReasoning(
            primary_reason='Technical analysis',
            supporting_factors=['RSI', 'MACD'],
            alternatives_considered=['Wait'],
            risks_assessed=['Volatility'],
            confidence_explanation='High confidence',
            expected_outcome='5% upside'
        )
        
        self.assertEqual(reasoning.primary_reason, 'Technical analysis')
        self.assertEqual(len(reasoning.supporting_factors), 2)
        self.assertEqual(reasoning.confidence_explanation, 'High confidence')

    def test_agent_decision(self):
        """Test AgentDecision dataclass."""
        context = DecisionContext(
            symbol='AAPL',
            timeframe='1h',
            timestamp='2024-01-01T12:00:00',
            market_conditions={},
            available_data=[],
            constraints={},
            user_preferences={}
        )
        
        reasoning = DecisionReasoning(
            primary_reason='Test',
            supporting_factors=[],
            alternatives_considered=[],
            risks_assessed=[],
            confidence_explanation='Test',
            expected_outcome='Test'
        )
        
        decision = AgentDecision(
            decision_id='test_123',
            agent_name='TestAgent',
            decision_type=DecisionType.FORECAST,
            action_taken='Test action',
            context=context,
            reasoning=reasoning,
            confidence_level=ConfidenceLevel.HIGH,
            timestamp='2024-01-01T12:00:00',
            metadata={}
        )
        
        self.assertEqual(decision.decision_id, 'test_123')
        self.assertEqual(decision.agent_name, 'TestAgent')
        self.assertEqual(decision.decision_type, DecisionType.FORECAST)
        self.assertEqual(decision.confidence_level, ConfidenceLevel.HIGH)

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestReasoningLogger,
        TestReasoningDisplay,
        TestConvenienceFunctions,
        TestDataStructures
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 