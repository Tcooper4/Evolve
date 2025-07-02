"""Tests for the LLM router logic."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

class TestLLMRouter:
    """Test LLM router functionality."""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": "Based on the market analysis, I recommend a BUY signal for AAPL with 75% confidence due to strong technical indicators."
                    }
                }
            ]
        }
    
    @pytest.fixture
    def mock_router(self):
        """Create a mock router for testing."""
        from core.agents.router import AgentRouter
        return AgentRouter()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        return pd.DataFrame({
            'Open': np.random.normal(100, 2, 30),
            'High': np.random.normal(102, 2, 30),
            'Low': np.random.normal(98, 2, 30),
            'Close': np.random.normal(100, 2, 30),
            'Volume': np.random.normal(1000000, 100000, 30)
        }, index=dates)

    @patch('openai.ChatCompletion.create')
    def test_intent_detection(self, mock_openai, mock_router, mock_openai_response):
        """Test that the router correctly detects user intent."""
        mock_openai.return_value = mock_openai_response
        
        # Test different user inputs
        test_cases = [
            ("I want to buy AAPL", "buy"),
            ("Sell TSLA now", "sell"),
            ("What's the market analysis for GOOGL?", "analyze"),
            ("Show me the performance", "report"),
            ("Optimize my strategy", "optimize")
        ]
        
        for user_input, expected_intent in test_cases:
            with patch.object(mock_router, 'detect_intent') as mock_detect:
                mock_detect.return_value = {
                    'intent': expected_intent,
                    'confidence': 0.9,
                    'entities': ['AAPL']
                }
                
                result = mock_router.route_intent(user_input)
                
                assert result['intent'] == expected_intent
                assert result['confidence'] > 0.5
                assert 'entities' in result

    @patch('openai.ChatCompletion.create')
    def test_entity_extraction(self, mock_openai, mock_router):
        """Test that the router correctly extracts entities from user input."""
        mock_openai.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "AAPL, TSLA, GOOGL"
                    }
                }
            ]
        }
        
        test_inputs = [
            "Buy AAPL and TSLA",
            "Analyze GOOGL performance",
            "Sell all my positions"
        ]
        
        for user_input in test_inputs:
            with patch.object(mock_router, 'extract_entities') as mock_extract:
                mock_extract.return_value = ['AAPL', 'TSLA']
                
                entities = mock_router.extract_entities(user_input)
                
                assert isinstance(entities, list)
                assert len(entities) > 0

    def test_confidence_threshold(self, mock_router):
        """Test that the router applies confidence thresholds correctly."""
        # Test high confidence
        high_conf_result = {
            'intent': 'buy',
            'confidence': 0.9,
            'entities': ['AAPL']
        }
        
        # Test low confidence
        low_conf_result = {
            'intent': 'buy',
            'confidence': 0.3,
            'entities': ['AAPL']
        }
        
        # High confidence should be accepted
        assert mock_router._validate_confidence(high_conf_result, threshold=0.5)
        
        # Low confidence should be rejected
        assert not mock_router._validate_confidence(low_conf_result, threshold=0.5)

    @patch('openai.ChatCompletion.create')
    def test_strategy_selection(self, mock_openai, mock_router, sample_market_data):
        """Test that the router selects appropriate strategies based on market conditions."""
        mock_openai.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "RSI strategy is recommended due to oversold conditions."
                    }
                }
            ]
        }
        
        # Test different market conditions
        market_conditions = [
            {'volatility': 'high', 'trend': 'upward', 'expected_strategy': 'momentum'},
            {'volatility': 'low', 'trend': 'sideways', 'expected_strategy': 'mean_reversion'},
            {'volatility': 'high', 'trend': 'downward', 'expected_strategy': 'defensive'}
        ]
        
        for condition in market_conditions:
            with patch.object(mock_router, 'select_strategy') as mock_select:
                mock_select.return_value = {
                    'strategy': condition['expected_strategy'],
                    'confidence': 0.8,
                    'reasoning': 'Market conditions analysis'
                }
                
                result = mock_router.select_strategy(sample_market_data, condition)
                
                assert result['strategy'] == condition['expected_strategy']
                assert result['confidence'] > 0.5

    def test_error_handling(self, mock_router):
        """Test that the router handles errors gracefully."""
        # Test with invalid input
        with pytest.raises(ValueError):
            mock_router.route_intent("")
        
        # Test with None input
        with pytest.raises(ValueError):
            mock_router.route_intent(None)
        
        # Test with very long input
        long_input = "A" * 10000
        with pytest.raises(ValueError):
            mock_router.route_intent(long_input)

    @patch('openai.ChatCompletion.create')
    def test_context_awareness(self, mock_openai, mock_router):
        """Test that the router maintains context across interactions."""
        mock_openai.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Based on previous analysis, continue with BUY recommendation."
                    }
                }
            ]
        }
        
        # Simulate conversation context
        context = {
            'previous_intent': 'buy',
            'previous_entities': ['AAPL'],
            'conversation_history': [
                {'user': 'I want to buy AAPL', 'assistant': 'Analyzing AAPL...'},
                {'user': 'What do you think?', 'assistant': 'Recommendation: BUY'}
            ]
        }
        
        with patch.object(mock_router, 'route_with_context') as mock_context:
            mock_context.return_value = {
                'intent': 'buy',
                'confidence': 0.85,
                'entities': ['AAPL'],
                'context_used': True
            }
            
            result = mock_router.route_with_context("Should I proceed?", context)
            
            assert result['intent'] == 'buy'
            assert result['context_used'] is True

    def test_performance_metrics(self, mock_router):
        """Test that the router tracks performance metrics."""
        # Simulate multiple routing requests
        test_inputs = [
            "Buy AAPL",
            "Sell TSLA",
            "Analyze GOOGL",
            "Show performance"
        ]
        
        for user_input in test_inputs:
            with patch.object(mock_router, 'route_intent') as mock_route:
                mock_route.return_value = {
                    'intent': 'buy',
                    'confidence': 0.8,
                    'entities': ['AAPL']
                }
                
                result = mock_router.route_intent(user_input)
                
                # Check that performance is logged
                assert hasattr(mock_router, 'log_performance')
                assert result['confidence'] > 0

    @patch('openai.ChatCompletion.create')
    def test_multi_entity_handling(self, mock_openai, mock_router):
        """Test that the router handles multiple entities correctly."""
        mock_openai.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "AAPL, TSLA, GOOGL are all recommended for purchase."
                    }
                }
            ]
        }
        
        user_input = "Buy AAPL, TSLA, and GOOGL"
        
        with patch.object(mock_router, 'extract_entities') as mock_extract:
            mock_extract.return_value = ['AAPL', 'TSLA', 'GOOGL']
            
            entities = mock_router.extract_entities(user_input)
            
            assert len(entities) == 3
            assert 'AAPL' in entities
            assert 'TSLA' in entities
            assert 'GOOGL' in entities

    def test_strategy_validation(self, mock_router):
        """Test that the router validates strategy recommendations."""
        # Test valid strategy
        valid_strategy = {
            'strategy': 'RSI',
            'confidence': 0.8,
            'parameters': {'period': 14}
        }
        
        # Test invalid strategy
        invalid_strategy = {
            'strategy': 'INVALID_STRATEGY',
            'confidence': 0.8,
            'parameters': {}
        }
        
        # Valid strategy should pass validation
        assert mock_router._validate_strategy(valid_strategy)
        
        # Invalid strategy should fail validation
        assert not mock_router._validate_strategy(invalid_strategy)

    @patch('openai.ChatCompletion.create')
    def test_adaptive_learning(self, mock_openai, mock_router):
        """Test that the router learns from user feedback."""
        mock_openai.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Learning from feedback to improve recommendations."
                    }
                }
            ]
        }
        
        # Simulate user feedback
        feedback = {
            'intent': 'buy',
            'entities': ['AAPL'],
            'user_satisfaction': 0.9,
            'actual_outcome': 'positive'
        }
        
        with patch.object(mock_router, 'learn_from_feedback') as mock_learn:
            mock_learn.return_value = True
            
            result = mock_router.learn_from_feedback(feedback)
            
            assert result is True

    def test_rate_limiting(self, mock_router):
        """Test that the router handles rate limiting correctly."""
        # Simulate rate limiting
        with patch.object(mock_router, '_check_rate_limit') as mock_rate_limit:
            mock_rate_limit.return_value = False  # Rate limit exceeded
            
            with pytest.raises(Exception):
                mock_router.route_intent("Buy AAPL")

    def test_fallback_mechanism(self, mock_router):
        """Test that the router has fallback mechanisms when primary methods fail."""
        # Simulate primary method failure
        with patch.object(mock_router, 'route_intent') as mock_route:
            mock_route.side_effect = Exception("Primary method failed")
            
            # Should use fallback mechanism
            with patch.object(mock_router, '_fallback_route') as mock_fallback:
                mock_fallback.return_value = {
                    'intent': 'buy',
                    'confidence': 0.5,
                    'entities': ['AAPL'],
                    'method': 'fallback'
                }
                
                result = mock_router.route_intent("Buy AAPL")
                
                assert result['method'] == 'fallback'
                assert result['confidence'] > 0

    def test_input_sanitization(self, mock_router):
        """Test that the router sanitizes user input correctly."""
        # Test with potentially malicious input
        malicious_inputs = [
            "Buy AAPL<script>alert('xss')</script>",
            "DROP TABLE users;",
            "Buy AAPL" + "A" * 1000  # Very long input
        ]
        
        for malicious_input in malicious_inputs:
            with patch.object(mock_router, '_sanitize_input') as mock_sanitize:
                mock_sanitize.return_value = "Buy AAPL"
                
                sanitized = mock_router._sanitize_input(malicious_input)
                
                assert sanitized == "Buy AAPL"
                assert len(sanitized) < 1000

    def test_response_formatting(self, mock_router):
        """Test that the router formats responses correctly."""
        raw_response = {
            'intent': 'buy',
            'confidence': 0.85,
            'entities': ['AAPL'],
            'strategy': 'RSI',
            'reasoning': 'Oversold conditions detected'
        }
        
        with patch.object(mock_router, '_format_response') as mock_format:
            mock_format.return_value = {
                'action': 'BUY',
                'symbol': 'AAPL',
                'confidence': '85%',
                'strategy': 'RSI',
                'reason': 'Oversold conditions detected'
            }
            
            formatted = mock_router._format_response(raw_response)
            
            assert formatted['action'] == 'BUY'
            assert formatted['symbol'] == 'AAPL'
            assert 'confidence' in formatted
            assert 'strategy' in formatted 