"""Tests for the router functionality."""

import os
import sys
from unittest.mock import patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.agents.intent_detector import IntentDetector
from trading.agents.prompt_router_agent import PromptRouterAgent as AgentRouter


class TestRouter:
    @pytest.fixture
    def router(self):
        """Create a router instance for testing."""
        return AgentRouter()

    @pytest.fixture
    def intent_detector(self):
        """Create an intent detector for testing."""
        return IntentDetector()

    def test_router_initialization(self, router):
        """Test that the router initializes correctly."""
        assert router is not None
        assert hasattr(router, "intent_detector")
        assert hasattr(router, "agent_registry")

    def test_intent_detection(self, router, intent_detector):
        """Test that the router correctly detects intents."""
        test_prompts = [
            "What's the best stock to buy today?",
            "Should I hold AAPL?",
            "Generate a forecast for MSFT",
            "What's the current market trend?",
            "Show me the performance of my portfolio",
        ]

        for prompt in test_prompts:
            intent = router.detect_intent(prompt)
            assert intent is not None
            assert "type" in intent
            assert "confidence" in intent
            assert intent["confidence"] >= 0.0
            assert intent["confidence"] <= 1.0

    def test_agent_routing(self, router):
        """Test that the router correctly routes to appropriate agents."""
        test_cases = [
            {
                "intent": {"type": "forecast", "entity": "AAPL"},
                "expected_agent": "forecast_agent",
            },
            {
                "intent": {"type": "trade", "entity": "MSFT"},
                "expected_agent": "trading_agent",
            },
            {
                "intent": {"type": "analysis", "entity": "GOOGL"},
                "expected_agent": "analysis_agent",
            },
        ]

        for case in test_cases:
            agent = router.route_to_agent(case["intent"])
            assert agent is not None
            assert agent.__class__.__name__.lower() == case["expected_agent"]

    def test_error_handling(self, router):
        """Test that the router handles errors gracefully."""
        # Test with invalid intent
        with pytest.raises(ValueError):
            router.route_to_agent({"type": "invalid"})

        # Test with missing entity
        with pytest.raises(ValueError):
            router.route_to_agent({"type": "forecast"})

    def test_agent_registry(self, router):
        """Test that the agent registry contains all required agents."""
        required_agents = [
            "forecast_agent",
            "trading_agent",
            "analysis_agent",
            "self_improving_agent",
        ]

        for agent_name in required_agents:
            assert agent_name in router.agent_registry
            agent = router.agent_registry[agent_name]
            assert agent is not None
            assert hasattr(agent, "process")

    @patch("trading.agents.router.AgentRouter.detect_intent")
    def test_end_to_end_routing(self, mock_detect_intent, router):
        """Test end-to-end routing process."""
        # Mock intent detection
        mock_detect_intent.return_value = {
            "type": "forecast",
            "entity": "AAPL",
            "confidence": 0.9,
        }

        # Test routing
        result = router.process_request("What's the forecast for AAPL?")
        assert result is not None
        assert "response" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0
