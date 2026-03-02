"""Tests for the router functionality."""

import os
import sys
from unittest.mock import patch

import pytest

from trading.agents.intent_detector import IntentDetector
from trading.agents.enhanced_prompt_router import EnhancedPromptRouterAgent as AgentRouter

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRouter:
    @pytest.fixture
    def router(self):
        """Create a router instance for testing (EnhancedPromptRouterAgent)."""
        return AgentRouter()

    @pytest.fixture
    def intent_detector(self):
        """Create an intent detector instance for testing."""
        return IntentDetector()

    def test_router_initialization(self, router):
        """Test that the router initializes correctly (EnhancedPromptRouterAgent API)."""
        assert router is not None
        assert hasattr(router, "parse_intent")
        assert hasattr(router, "route")

    def test_intent_detection(self, router, intent_detector):
        """Test that the router correctly detects intents via parse_intent."""
        test_prompts = [
            "What's the best stock to buy today?",
            "Should I hold AAPL?",
            "Generate a forecast for MSFT",
            "What's the current market trend?",
            "Show me the performance of my portfolio",
        ]

        for prompt in test_prompts:
            parsed = router.parse_intent(prompt)
            assert parsed is not None
            assert hasattr(parsed, "intent")
            assert hasattr(parsed, "confidence")
            assert parsed.confidence >= 0.0
            assert parsed.confidence <= 1.0

    def test_agent_routing(self, router):
        """Test that the router routes via route(prompt, agents) and returns intent/routed_agent."""
        agents = {
            "forecast": None,
            "trade": None,
            "analysis": None,
            "general_agent": None,
        }
        result = router.route("Generate a forecast for MSFT", agents)
        assert result is not None
        assert "intent" in result
        assert "confidence" in result
        assert "routed_agent" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0

    def test_error_handling(self, router):
        """EnhancedPromptRouterAgent does not raise on invalid intent; returns parsed result."""
        # route() always returns a dict; parse_intent never raises for unknown
        result = router.parse_intent("xyz invalid qwerty")
        assert result is not None
        assert result.intent is not None
        assert result.confidence >= 0.0

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no agent_registry; routing is via route(prompt, agents)")
    def test_agent_registry(self, router):
        """Legacy test: agent_registry not present on EnhancedPromptRouterAgent."""
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

    @patch("trading.agents.enhanced_prompt_router.EnhancedPromptRouterAgent.parse_intent")
    def test_end_to_end_routing(self, mock_parse_intent, router):
        """Test end-to-end routing via route() using EnhancedPromptRouterAgent API."""
        from trading.agents.enhanced_prompt_router import ParsedIntent

        mock_parse_intent.return_value = ParsedIntent(
            intent="forecast",
            confidence=0.9,
            args={"symbol": "AAPL"},
            provider="regex",
            raw_response="mocked",
        )
        agents = {"forecast": None, "general_agent": None}
        result = router.route("What's the forecast for AAPL?", agents)
        assert result is not None
        assert "intent" in result
        assert result["intent"] == "forecast"
        assert "confidence" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0
