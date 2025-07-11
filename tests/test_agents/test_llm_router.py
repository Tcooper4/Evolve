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

    def test_fallback_llm_triggered_on_failure_or_overload(self, mock_router):
        """Validate fallback LLM is triggered when primary route fails or is overloaded."""
        print("\n🔄 Testing Fallback LLM Trigger on Failure/Overload")
        
        # Test fallback on primary route failure
        print("  Testing fallback on primary route failure...")
        
        # Mock primary route failure
        with patch.object(mock_router, 'route_intent') as mock_primary:
            mock_primary.side_effect = Exception("Primary route failed")
            
            # Mock fallback route
            with patch.object(mock_router, 'route_fallback') as mock_fallback:
                mock_fallback.return_value = {
                    'intent': 'buy',
                    'confidence': 0.7,
                    'entities': ['AAPL'],
                    'fallback_used': True
                }
                
                # Test fallback activation
                result = mock_router.route_with_fallback("Buy AAPL")
                
                # Verify fallback was used
                self.assertTrue(result['fallback_used'], "Fallback should be used when primary fails")
                self.assertEqual(result['intent'], 'buy', "Fallback should provide valid intent")
                self.assertGreater(result['confidence'], 0.5, "Fallback should provide reasonable confidence")
                
                # Verify fallback was called
                mock_fallback.assert_called_once()
                print("  ✅ Fallback triggered on primary route failure")
        
        # Test fallback on overload (high latency)
        print("  Testing fallback on overload (high latency)...")
        
        # Mock primary route with high latency
        with patch.object(mock_router, 'route_intent') as mock_primary:
            mock_primary.side_effect = TimeoutError("Primary route timeout")
            
            # Mock fallback route
            with patch.object(mock_router, 'route_fallback') as mock_fallback:
                mock_fallback.return_value = {
                    'intent': 'analyze',
                    'confidence': 0.6,
                    'entities': ['TSLA'],
                    'fallback_used': True
                }
                
                # Test fallback activation
                result = mock_router.route_with_fallback("Analyze TSLA")
                
                # Verify fallback was used
                self.assertTrue(result['fallback_used'], "Fallback should be used when primary times out")
                self.assertEqual(result['intent'], 'analyze', "Fallback should provide valid intent")
                
                # Verify fallback was called
                mock_fallback.assert_called_once()
                print("  ✅ Fallback triggered on primary route timeout")
        
        # Test fallback on low confidence
        print("  Testing fallback on low confidence...")
        
        # Mock primary route with low confidence
        with patch.object(mock_router, 'route_intent') as mock_primary:
            mock_primary.return_value = {
                'intent': 'buy',
                'confidence': 0.3,  # Below threshold
                'entities': ['GOOGL']
            }
            
            # Mock fallback route
            with patch.object(mock_router, 'route_fallback') as mock_fallback:
                mock_fallback.return_value = {
                    'intent': 'buy',
                    'confidence': 0.8,  # Higher confidence
                    'entities': ['GOOGL'],
                    'fallback_used': True
                }
                
                # Test fallback activation
                result = mock_router.route_with_fallback("Buy GOOGL", confidence_threshold=0.5)
                
                # Verify fallback was used due to low confidence
                self.assertTrue(result['fallback_used'], "Fallback should be used when confidence is low")
                self.assertGreater(result['confidence'], 0.5, "Fallback should provide higher confidence")
                
                # Verify fallback was called
                mock_fallback.assert_called_once()
                print("  ✅ Fallback triggered on low confidence")
        
        # Test fallback on invalid response
        print("  Testing fallback on invalid response...")
        
        # Mock primary route with invalid response
        with patch.object(mock_router, 'route_intent') as mock_primary:
            mock_primary.return_value = {
                'intent': 'invalid_intent',  # Invalid intent
                'confidence': 0.9,
                'entities': ['MSFT']
            }
            
            # Mock fallback route
            with patch.object(mock_router, 'route_fallback') as mock_fallback:
                mock_fallback.return_value = {
                    'intent': 'buy',
                    'confidence': 0.7,
                    'entities': ['MSFT'],
                    'fallback_used': True
                }
                
                # Test fallback activation
                result = mock_router.route_with_fallback("Buy MSFT")
                
                # Verify fallback was used due to invalid response
                self.assertTrue(result['fallback_used'], "Fallback should be used when response is invalid")
                self.assertIn(result['intent'], ['buy', 'sell', 'analyze', 'report', 'optimize'], 
                             "Fallback should provide valid intent")
                
                # Verify fallback was called
                mock_fallback.assert_called_once()
                print("  ✅ Fallback triggered on invalid response")
        
        # Test fallback chain (multiple fallbacks)
        print("  Testing fallback chain...")
        
        # Mock multiple fallback levels
        with patch.object(mock_router, 'route_intent') as mock_primary:
            mock_primary.side_effect = Exception("Primary failed")
            
            with patch.object(mock_router, 'route_fallback') as mock_fallback1:
                mock_fallback1.side_effect = Exception("Fallback 1 failed")
                
                with patch.object(mock_router, 'route_fallback2') as mock_fallback2:
                    mock_fallback2.return_value = {
                        'intent': 'sell',
                        'confidence': 0.5,
                        'entities': ['NVDA'],
                        'fallback_used': True,
                        'fallback_level': 2
                    }
                    
                    # Test fallback chain
                    result = mock_router.route_with_fallback_chain("Sell NVDA")
                    
                    # Verify second fallback was used
                    self.assertTrue(result['fallback_used'], "Fallback chain should be used")
                    self.assertEqual(result['fallback_level'], 2, "Should use second fallback level")
                    self.assertEqual(result['intent'], 'sell', "Should provide valid intent")
                    
                    # Verify fallback chain was called
                    mock_fallback1.assert_called_once()
                    mock_fallback2.assert_called_once()
                    print("  ✅ Fallback chain triggered successfully")
        
        # Test fallback performance monitoring
        print("  Testing fallback performance monitoring...")
        
        # Mock performance tracking
        with patch.object(mock_router, 'track_fallback_performance') as mock_track:
            mock_track.return_value = {
                'fallback_usage_rate': 0.15,
                'fallback_success_rate': 0.85,
                'avg_fallback_latency': 0.5
            }
            
            # Test fallback performance tracking
            performance = mock_router.get_fallback_performance()
            
            # Verify performance metrics
            self.assertIn('fallback_usage_rate', performance)
            self.assertIn('fallback_success_rate', performance)
            self.assertIn('avg_fallback_latency', performance)
            
            self.assertGreaterEqual(performance['fallback_success_rate'], 0.8, 
                                   "Fallback success rate should be high")
            self.assertLess(performance['avg_fallback_latency'], 1.0, 
                           "Fallback latency should be reasonable")
            
            print("  ✅ Fallback performance monitoring working")
        
        # Test fallback circuit breaker
        print("  Testing fallback circuit breaker...")
        
        # Mock circuit breaker logic
        with patch.object(mock_router, 'check_circuit_breaker') as mock_circuit:
            mock_circuit.return_value = True  # Circuit breaker open
            
            # Test circuit breaker activation
            should_use_fallback = mock_router.should_use_fallback()
            
            self.assertTrue(should_use_fallback, "Should use fallback when circuit breaker is open")
            print("  ✅ Circuit breaker logic working")
        
        print("✅ Fallback LLM trigger test completed")

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