"""
Unit tests for prompt_router.py agent.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

def test_invest_prompt():
    """Test that investment prompts are correctly identified and routed."""
    try:
        from prompt_router import PromptRouterAgent, RequestType
        
        agent = PromptRouterAgent()
        result = agent.handle_prompt("What stocks should I invest in today?")
        
        assert result['success'] is True
        assert result['request_type'] == 'investment'
        assert 'TopRankedForecastAgent' in result['routing_suggestions']
        
    except ImportError as e:
        pytest.skip(f"PromptRouterAgent not available: {e}")

def test_forecast_prompt():
    """Test that forecast prompts are correctly identified."""
    try:
        from prompt_router import PromptRouterAgent
        
        agent = PromptRouterAgent()
        result = agent.handle_prompt("Forecast AAPL for next week")
        
        assert result['success'] is True
        assert result['request_type'] == 'forecast'
        assert 'ModelSelectorAgent' in result['routing_suggestions']
        
    except ImportError as e:
        pytest.skip(f"PromptRouterAgent not available: {e}")

def test_strategy_prompt():
    """Test that strategy prompts are correctly identified."""
    try:
        from prompt_router import PromptRouterAgent
        
        agent = PromptRouterAgent()
        result = agent.handle_prompt("What's the best RSI strategy?")
        
        assert result['success'] is True
        assert result['request_type'] == 'strategy'
        assert 'StrategySelectorAgent' in result['routing_suggestions']
        
    except ImportError as e:
        pytest.skip(f"PromptRouterAgent not available: {e}")

def test_prompt_normalization():
    """Test that prompts are properly normalized."""
    try:
        from prompt_router import PromptProcessor
        
        processor = PromptProcessor()
        
        # Test normalization
        normalized = processor._normalize_prompt("  What STOCKS should I INVEST in TODAY?  ")
        assert normalized == "what stocks should i invest in today?"
        
    except ImportError as e:
        pytest.skip(f"PromptProcessor not available: {e}")

def test_investment_query_detection():
    """Test investment query detection logic."""
    try:
        from prompt_router import PromptProcessor
        
        processor = PromptProcessor()
        
        # Test various investment-related queries
        investment_queries = [
            "What stocks should I invest in today?",
            "Which stocks are the best to buy?",
            "Recommend some stocks for investment",
            "What should I invest in?",
            "Top stocks to buy now"
        ]
        
        for query in investment_queries:
            normalized = processor._normalize_prompt(query)
            assert processor._is_investment_query(normalized), f"Failed to detect investment query: {query}"
        
    except ImportError as e:
        pytest.skip(f"PromptProcessor not available: {e}")

def test_fuzzy_matching():
    """Test fuzzy matching functionality."""
    try:
        from prompt_router import PromptProcessor
        
        processor = PromptProcessor()
        
        # Test fuzzy matching
        assert processor._fuzzy_match("invest", "investment", threshold=0.8)
        assert processor._fuzzy_match("stocks", "stock", threshold=0.8)
        assert not processor._fuzzy_match("forecast", "strategy", threshold=0.8)
        
    except ImportError as e:
        pytest.skip(f"PromptProcessor not available: {e}")

def test_error_handling():
    """Test error handling in prompt router."""
    try:
        from prompt_router import PromptRouterAgent
        
        agent = PromptRouterAgent()
        
        # Test with None input
        result = agent.handle_prompt(None)
        assert result['success'] is False
        assert 'error' in result['message'].lower()
        
    except ImportError as e:
        pytest.skip(f"PromptRouterAgent not available: {e}")

def test_parameter_extraction():
    """Test parameter extraction from prompts."""
    try:
        from prompt_router import PromptProcessor
        
        processor = PromptProcessor()
        
        # Test symbol extraction
        result = processor._extract_parameters("Forecast AAPL for 30 days")
        assert 'symbol' in result
        assert result['symbol'] == 'AAPL'
        
        # Test timeframe extraction
        result = processor._extract_parameters("Analyze TSLA with 1d timeframe")
        assert 'timeframe' in result
        assert result['timeframe'] == '1d'
        
    except ImportError as e:
        pytest.skip(f"PromptProcessor not available: {e}")

def test_confidence_calculation():
    """Test confidence calculation for prompt classification."""
    try:
        from prompt_router import PromptProcessor, RequestType
        
        processor = PromptProcessor()
        
        # Test high confidence for clear investment query
        confidence = processor._calculate_confidence(
            "What stocks should I invest in today?", 
            RequestType.INVESTMENT
        )
        assert confidence > 0.5
        
        # Test low confidence for unclear query
        confidence = processor._calculate_confidence(
            "Hello world", 
            RequestType.UNKNOWN
        )
        assert confidence == 0.0
        
    except ImportError as e:
        pytest.skip(f"PromptProcessor not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 