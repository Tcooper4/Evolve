import pytest
import logging
from trading.nlp.llm_processor import LLMProcessor

def test_llm_processor():
    """Test the LLM processor functionality."""
    llm = LLMProcessor()
    
    # Test sentiment analysis
    sentiment_result = llm.analyze_sentiment("This is a great day for trading!")
    assert 'label' in sentiment_result
    assert 'score' in sentiment_result
    
    # Test text summarization
    text = "The stock market experienced significant volatility today. Many investors are concerned about the future of the economy. Analysts predict a potential downturn in the coming months."
    summary = llm.summarize_text(text, max_length=30)
    assert len(summary.split()) <= 30
    
    # Test entity extraction
    entities = llm.extract_entities("Apple Inc. announced a new product today.")
    assert len(entities) > 0
    assert any(entity['entity'] == 'ORG' for entity in entities) 