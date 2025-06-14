"""Tests for the LLMProcessor class."""

import pytest
from unittest.mock import Mock, patch
from trading.nlp.llm_processor import LLMProcessor

@pytest.fixture
def processor():
    return LLMProcessor()

def test_extract_ticker():
    """Test ticker extraction from various prompts."""
    processor = LLMProcessor()
    
    # Test cases
    test_cases = [
        ("Analyze AAPL stock", "AAPL"),
        ("What's the forecast for MSFT?", "MSFT"),
        ("Show me GOOGL's performance", "GOOGL"),
        ("TSLA is looking bullish", "TSLA"),
        ("No ticker mentioned here", None)
    ]
    
    for prompt, expected in test_cases:
        result = processor.extract_ticker(prompt)
        assert result == expected

def test_extract_strategy():
    """Test strategy extraction from various prompts."""
    processor = LLMProcessor()
    
    # Test cases
    test_cases = [
        ("Use RSI strategy", "RSI"),
        ("Apply MACD to the chart", "MACD"),
        ("Show me SMA crossover", "SMA"),
        ("No strategy mentioned", None)
    ]
    
    for prompt, expected in test_cases:
        result = processor.extract_strategy(prompt)
        assert result == expected

def test_extract_timeframe():
    """Test timeframe extraction from various prompts."""
    processor = LLMProcessor()
    
    # Test cases
    test_cases = [
        ("Show 1h chart", "1h"),
        ("Daily timeframe analysis", "1d"),
        ("Weekly forecast", "1w"),
        ("Monthly view", "1m"),
        ("No timeframe specified", None)
    ]
    
    for prompt, expected in test_cases:
        result = processor.extract_timeframe(prompt)
        assert result == expected

def test_extract_multiple_entities():
    """Test extraction of multiple entities from a single prompt."""
    processor = LLMProcessor()
    
    prompt = "Analyze AAPL using RSI on 1h timeframe"
    entities = processor.extract_entities(prompt)
    
    assert entities["ticker"] == "AAPL"
    assert entities["strategy"] == "RSI"
    assert entities["timeframe"] == "1h"

def test_entity_validation():
    """Test validation of extracted entities."""
    processor = LLMProcessor()
    
    # Test valid ticker
    assert processor.validate_ticker("AAPL")
    assert not processor.validate_ticker("INVALID")
    
    # Test valid strategy
    assert processor.validate_strategy("RSI")
    assert not processor.validate_strategy("INVALID")
    
    # Test valid timeframe
    assert processor.validate_timeframe("1h")
    assert not processor.validate_timeframe("INVALID")

def test_entity_normalization():
    """Test normalization of extracted entities."""
    processor = LLMProcessor()
    
    # Test ticker normalization
    assert processor.normalize_ticker("aapl") == "AAPL"
    assert processor.normalize_ticker("AAPL") == "AAPL"
    
    # Test strategy normalization
    assert processor.normalize_strategy("rsi") == "RSI"
    assert processor.normalize_strategy("RSI") == "RSI"
    
    # Test timeframe normalization
    assert processor.normalize_timeframe("1H") == "1h"
    assert processor.normalize_timeframe("1h") == "1h"

def test_complex_prompt_parsing():
    """Test parsing of complex prompts with multiple entities."""
    processor = LLMProcessor()
    
    prompt = "Show me AAPL's RSI on 1h chart and MSFT's MACD on daily"
    entities = processor.extract_entities(prompt)
    
    assert "AAPL" in entities["tickers"]
    assert "MSFT" in entities["tickers"]
    assert "RSI" in entities["strategies"]
    assert "MACD" in entities["strategies"]
    assert "1h" in entities["timeframes"]
    assert "1d" in entities["timeframes"]

def test_error_handling():
    """Test error handling in entity extraction."""
    processor = LLMProcessor()
    
    # Test with invalid input
    with pytest.raises(ValueError):
        processor.extract_entities(None)
    
    with pytest.raises(ValueError):
        processor.extract_entities(123)
    
    # Test with empty input
    entities = processor.extract_entities("")
    assert all(v is None for v in entities.values()) 