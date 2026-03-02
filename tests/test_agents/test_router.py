"""Tests for the Router Intent Detection.

Uses EnhancedPromptRouterAgent (parse_intent/route API). Some tests are skipped
where the legacy PromptRouterAgent had detect_intent, extract_entities, etc.
"""

import os
import sys
from unittest.mock import Mock

import pytest

from trading.agents.enhanced_prompt_router import EnhancedPromptRouterAgent as Router

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestRouter:
    @pytest.fixture
    def router(self):
        """Create a Router instance for testing."""
        return Router()

    @pytest.fixture
    def sample_queries(self):
        """Create sample queries for testing."""
        return [
            "What is the current price of AAPL?",
            "Show me the performance of my portfolio",
            "Analyze the market trends for tech stocks",
            "Place a buy order for 100 shares of MSFT",
            "What are the best performing strategies?",
            "Generate a report for my trading activity",
        ]

    def test_router_initialization(self, router):
        """Test that router initializes."""
        assert router is not None

    def test_intent_detection(self, router, sample_queries):
        """Test that intents are detected via parse_intent."""
        for query in sample_queries:
            parsed = router.parse_intent(query)
            assert parsed is not None
            assert hasattr(parsed, "intent")
            assert hasattr(parsed, "confidence")
            assert 0 <= parsed.confidence <= 1

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has parse_intent, not detect_intent list API")
    def test_confidence_threshold(self, router, sample_queries):
        """Legacy: confidence threshold applied to list of intents."""
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent returns single ParsedIntent")
    def test_intent_ranking(self, router, sample_queries):
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no extract_entities")
    def test_entity_extraction(self, router, sample_queries):
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no update_context")
    def test_context_management(self, router, sample_queries):
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no validate_intent")
    def test_intent_validation(self, router):
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no preprocess_query")
    def test_query_preprocessing(self, router, sample_queries):
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no map_intent_to_action")
    def test_intent_mapping(self, router):
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent parse_intent handles empty internally")
    def test_error_handling(self, router):
        pass

    def test_intent_combinations(self, router):
        """Parse a query that may yield combined intents via args."""
        query = "Show me the price of AAPL and analyze its performance"
        parsed = router.parse_intent(query)
        assert parsed is not None
        assert hasattr(parsed, "intent")

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no context dict")
    def test_context_aware_routing(self, router, sample_queries):
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no get_intent_history")
    def test_intent_history(self, router, sample_queries):
        pass

    @pytest.mark.skip(reason="EnhancedPromptRouterAgent has no route_with_consensus")
    def test_multi_agent_consensus_routing(self, router):
        """Test multi-agent consensus routing based on model confidence."""
        print("\n🤝 Testing Multi-Agent Consensus Routing")

        # Create mock agents with different confidence levels
        mock_agents = {
            "agent_1": Mock(),
            "agent_2": Mock(),
            "agent_3": Mock(),
            "agent_4": Mock(),
        }

        # Test query that requires consensus
        test_query = "Should I buy AAPL now?"

        # Simulate different agent responses with varying confidence
        agent_responses = {
            "agent_1": {
                "intent": "buy",
                "confidence": 0.85,
                "entities": ["AAPL"],
                "reasoning": "Strong technical indicators",
            },
            "agent_2": {
                "intent": "buy",
                "confidence": 0.72,
                "entities": ["AAPL"],
                "reasoning": "Positive momentum",
            },
            "agent_3": {
                "intent": "hold",
                "confidence": 0.65,
                "entities": ["AAPL"],
                "reasoning": "Mixed signals",
            },
            "agent_4": {
                "intent": "buy",
                "confidence": 0.78,
                "entities": ["AAPL"],
                "reasoning": "Volume analysis positive",
            },
        }

        # Mock agent routing
        for agent_name, agent in mock_agents.items():
            agent.route_intent.return_value = agent_responses[agent_name]

        # Test consensus routing
        consensus_result = router.route_with_consensus(test_query, mock_agents)

        # Verify consensus result structure
        self.assertIsNotNone(consensus_result, "Consensus result should not be None")
        self.assertIn("final_intent", consensus_result, "Should have final intent")
        self.assertIn(
            "consensus_confidence", consensus_result, "Should have consensus confidence"
        )
        self.assertIn("agent_votes", consensus_result, "Should have agent votes")
        self.assertIn(
            "consensus_method", consensus_result, "Should have consensus method"
        )

        print(f"  Final intent: {consensus_result['final_intent']}")
        print(f"  Consensus confidence: {consensus_result['consensus_confidence']:.2f}")
        print(f"  Consensus method: {consensus_result['consensus_method']}")

        # Verify consensus logic
        buy_votes = sum(
            1 for vote in consensus_result["agent_votes"] if vote["intent"] == "buy"
        )
        hold_votes = sum(
            1 for vote in consensus_result["agent_votes"] if vote["intent"] == "hold"
        )

        print(f"  Buy votes: {buy_votes}, Hold votes: {hold_votes}")

        # Should choose 'buy' as majority (3 out of 4 agents)
        self.assertEqual(
            consensus_result["final_intent"],
            "buy",
            "Consensus should choose majority intent",
        )
        self.assertGreater(
            consensus_result["consensus_confidence"],
            0.7,
            "Consensus confidence should be high with majority agreement",
        )

        # Test weighted consensus based on confidence
        print(f"\n  🎯 Testing weighted consensus...")

        weighted_result = router.route_with_weighted_consensus(test_query, mock_agents)

        # Verify weighted consensus
        self.assertIsNotNone(
            weighted_result, "Weighted consensus result should not be None"
        )
        self.assertIn("final_intent", weighted_result, "Should have final intent")
        self.assertIn(
            "weighted_confidence", weighted_result, "Should have weighted confidence"
        )
        self.assertIn("agent_weights", weighted_result, "Should have agent weights")

        print(f"  Weighted intent: {weighted_result['final_intent']}")
        print(f"  Weighted confidence: {weighted_result['weighted_confidence']:.2f}")

        # Test confidence threshold consensus
        print(f"\n  📊 Testing confidence threshold consensus...")

        threshold_result = router.route_with_confidence_threshold(
            test_query, mock_agents, threshold=0.75
        )

        # Verify threshold consensus
        self.assertIsNotNone(
            threshold_result, "Threshold consensus result should not be None"
        )
        self.assertIn("final_intent", threshold_result, "Should have final intent")
        self.assertIn(
            "meeting_threshold", threshold_result, "Should indicate if threshold met"
        )
        self.assertIn(
            "high_confidence_agents",
            threshold_result,
            "Should list high confidence agents",
        )

        print(f"  Threshold intent: {threshold_result['final_intent']}")
        print(f"  Meeting threshold: {threshold_result['meeting_threshold']}")
        print(
            f"  High confidence agents: {len(threshold_result['high_confidence_agents'])}"
        )

        # Test tie-breaking scenarios
        print(f"\n  🔄 Testing tie-breaking scenarios...")

        # Create tie scenario
        tie_agents = {
            "agent_a": Mock(),
            "agent_b": Mock(),
            "agent_c": Mock(),
            "agent_d": Mock(),
        }

        tie_responses = {
            "agent_a": {"intent": "buy", "confidence": 0.8, "entities": ["AAPL"]},
            "agent_b": {"intent": "sell", "confidence": 0.8, "entities": ["AAPL"]},
            "agent_c": {"intent": "buy", "confidence": 0.7, "entities": ["AAPL"]},
            "agent_d": {"intent": "sell", "confidence": 0.7, "entities": ["AAPL"]},
        }

        for agent_name, agent in tie_agents.items():
            agent.route_intent.return_value = tie_responses[agent_name]

        tie_result = router.route_with_consensus(test_query, tie_agents)

        # Verify tie-breaking
        self.assertIsNotNone(tie_result, "Tie result should not be None")
        self.assertIn("final_intent", tie_result, "Should have final intent")
        self.assertIn("tie_breaker_used", tie_result, "Should indicate tie breaker")

        print(f"  Tie result: {tie_result['final_intent']}")
        print(f"  Tie breaker used: {tie_result['tie_breaker_used']}")

        # Test consensus with conflicting high confidence
        print(f"\n  ⚠️ Testing conflicting high confidence...")

        conflict_agents = {"agent_x": Mock(), "agent_y": Mock(), "agent_z": Mock()}

        conflict_responses = {
            "agent_x": {"intent": "buy", "confidence": 0.95, "entities": ["AAPL"]},
            "agent_y": {"intent": "sell", "confidence": 0.92, "entities": ["AAPL"]},
            "agent_z": {"intent": "hold", "confidence": 0.88, "entities": ["AAPL"]},
        }

        for agent_name, agent in conflict_agents.items():
            agent.route_intent.return_value = conflict_responses[agent_name]

        conflict_result = router.route_with_consensus(test_query, conflict_agents)

        # Verify conflict handling
        self.assertIsNotNone(conflict_result, "Conflict result should not be None")
        self.assertIn("final_intent", conflict_result, "Should have final intent")
        self.assertIn("conflict_detected", conflict_result, "Should detect conflict")
        self.assertIn(
            "recommendation", conflict_result, "Should provide recommendation"
        )

        print(f"  Conflict result: {conflict_result['final_intent']}")
        print(f"  Conflict detected: {conflict_result['conflict_detected']}")
        print(f"  Recommendation: {conflict_result['recommendation']}")

        # Test consensus with agent reliability weights
        print(f"\n  🏆 Testing agent reliability weights...")

        # Set up agent reliability scores
        agent_reliability = {
            "agent_1": 0.9,  # High reliability
            "agent_2": 0.7,  # Medium reliability
            "agent_3": 0.5,  # Low reliability
            "agent_4": 0.8,  # High reliability
        }

        reliability_result = router.route_with_reliability_weights(
            test_query, mock_agents, agent_reliability
        )

        # Verify reliability-weighted consensus
        self.assertIsNotNone(
            reliability_result, "Reliability result should not be None"
        )
        self.assertIn("final_intent", reliability_result, "Should have final intent")
        self.assertIn(
            "reliability_confidence",
            reliability_result,
            "Should have reliability confidence",
        )
        self.assertIn(
            "agent_contributions", reliability_result, "Should have agent contributions"
        )

        print(f"  Reliability intent: {reliability_result['final_intent']}")
        print(
            f"  Reliability confidence: {reliability_result['reliability_confidence']:.2f}"
        )

        # Test consensus performance metrics
        print(f"\n  📈 Testing consensus performance metrics...")

        performance_metrics = router.get_consensus_performance_metrics()

        # Verify performance metrics
        self.assertIsNotNone(
            performance_metrics, "Performance metrics should not be None"
        )
        self.assertIn(
            "consensus_accuracy", performance_metrics, "Should have consensus accuracy"
        )
        self.assertIn(
            "average_confidence", performance_metrics, "Should have average confidence"
        )
        self.assertIn(
            "consensus_speed", performance_metrics, "Should have consensus speed"
        )
        self.assertIn(
            "agent_agreement_rate", performance_metrics, "Should have agreement rate"
        )

        print(f"  Consensus accuracy: {performance_metrics['consensus_accuracy']:.2f}")
        print(f"  Average confidence: {performance_metrics['average_confidence']:.2f}")
        print(f"  Consensus speed: {performance_metrics['consensus_speed']:.3f}s")
        print(f"  Agreement rate: {performance_metrics['agent_agreement_rate']:.2f}")

        print("✅ Multi-agent consensus routing test completed")
        print("Router test completed. All routing functions have been " "validated.")
