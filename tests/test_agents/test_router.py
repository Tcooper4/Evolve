"""Tests for the Router Intent Detection."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.agents.prompt_router_agent import PromptRouterAgent as Router

class TestRouter:
    @pytest.fixture
    def router(self):
        """Create a Router instance for testing."""
        return Router(
            confidence_threshold=0.7,
            max_intents=3
        )

    @pytest.fixture
    def sample_queries(self):
        """Create sample queries for testing."""
        return [
            "What is the current price of AAPL?",
            "Show me the performance of my portfolio",
            "Analyze the market trends for tech stocks",
            "Place a buy order for 100 shares of MSFT",
            "What are the best performing strategies?",
            "Generate a report for my trading activity"
        ]

    def test_router_initialization(self, router):
        """Test that router initializes with correct parameters."""
        assert router.confidence_threshold == 0.7
        assert router.max_intents == 3
        assert router.name == 'Router'

    def test_intent_detection(self, router, sample_queries):
        """Test that intents are detected correctly."""
        for query in sample_queries:
            intents = router.detect_intent(query)
            
            assert isinstance(intents, list)
            assert len(intents) <= router.max_intents
            assert all(isinstance(i, dict) for i in intents)
            assert all('intent' in i for i in intents)
            assert all('confidence' in i for i in intents)
            assert all(0 <= i['confidence'] <= 1 for i in intents)

    def test_confidence_threshold(self, router, sample_queries):
        """Test that confidence threshold is applied correctly."""
        for query in sample_queries:
            intents = router.detect_intent(query)
            
            # Check that all returned intents meet the confidence threshold
            assert all(i['confidence'] >= router.confidence_threshold for i in intents)

    def test_intent_ranking(self, router, sample_queries):
        """Test that intents are ranked by confidence."""
        for query in sample_queries:
            intents = router.detect_intent(query)
            
            # Check that intents are sorted by confidence in descending order
            confidences = [i['confidence'] for i in intents]
            assert confidences == sorted(confidences, reverse=True)

    def test_entity_extraction(self, router, sample_queries):
        """Test that entities are extracted correctly."""
        for query in sample_queries:
            entities = router.extract_entities(query)
            
            assert isinstance(entities, dict)
            assert all(isinstance(v, list) for v in entities.values())

    def test_context_management(self, router, sample_queries):
        """Test that context is managed correctly."""
        # Test context initialization
        assert router.context is not None
        
        # Test context update
        for query in sample_queries:
            router.update_context(query)
            assert router.context['last_query'] == query
            assert 'timestamp' in router.context

    def test_intent_validation(self, router):
        """Test that intents are validated correctly."""
        # Test invalid intent
        with pytest.raises(ValueError):
            router.validate_intent({'intent': 'invalid_intent', 'confidence': 0.8})
        
        # Test invalid confidence
        with pytest.raises(ValueError):
            router.validate_intent({'intent': 'price_query', 'confidence': 1.5})

    def test_query_preprocessing(self, router, sample_queries):
        """Test that queries are preprocessed correctly."""
        for query in sample_queries:
            processed = router.preprocess_query(query)
            
            assert isinstance(processed, str)
            assert len(processed) <= len(query)
            assert processed.lower() == processed  # Should be lowercase

    def test_intent_mapping(self, router):
        """Test that intents are mapped to actions correctly."""
        intent = 'price_query'
        action = router.map_intent_to_action(intent)
        
        assert isinstance(action, dict)
        assert 'action' in action
        assert 'parameters' in action

    def test_error_handling(self, router):
        """Test that errors are handled correctly."""
        # Test empty query
        with pytest.raises(ValueError):
            router.detect_intent("")
        
        # Test None query
        with pytest.raises(ValueError):
            router.detect_intent(None)

    def test_intent_combinations(self, router):
        """Test that multiple intents are handled correctly."""
        query = "Show me the price of AAPL and analyze its performance"
        intents = router.detect_intent(query)
        
        assert len(intents) > 1
        assert any(i['intent'] == 'price_query' for i in intents)
        assert any(i['intent'] == 'performance_analysis' for i in intents)

    def test_context_aware_routing(self, router, sample_queries):
        """Test that routing is context-aware."""
        # Set up context
        router.context['last_intent'] = 'portfolio_query'
        router.context['last_entities'] = {'symbol': ['AAPL']}
        
        # Test routing with context
        query = "What about MSFT?"
        intents = router.detect_intent(query)
        
        assert len(intents) > 0
        assert intents[0]['intent'] == 'portfolio_query'

    def test_intent_history(self, router, sample_queries):
        """Test that intent history is maintained correctly."""
        for query in sample_queries:
            router.detect_intent(query)
        
        history = router.get_intent_history()
        
        assert isinstance(history, list)
        assert len(history) == len(sample_queries)
        assert all(isinstance(h, dict) for h in history)
        assert all('query' in h for h in history)
        assert all('intents' in h for h in history) 