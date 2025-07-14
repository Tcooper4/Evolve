"""
Test Reasoning System

Comprehensive tests for the reasoning logger and display components.
"""

import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from trading.utils.reasoning_display import ReasoningDisplay
from trading.utils.reasoning_logger import (
    AgentDecision,
    ConfidenceLevel,
    DecisionContext,
    DecisionReasoning,
    DecisionType,
    ReasoningLogger,
    log_forecast_decision,
    log_strategy_decision,
)


class TestReasoningLogger(unittest.TestCase):
    """Test the ReasoningLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ReasoningLogger(
            redis_host="localhost",
            redis_port=6379,
            redis_db=1,  # Use different DB for testing
            log_dir=self.temp_dir,
            enable_gpt_explanations=False,  # Disable GPT for testing
        )

        # Sample decision data
        self.sample_decision_data = {
            "agent_name": "TestAgent",
            "decision_type": DecisionType.FORECAST,
            "action_taken": "Predicted AAPL will reach $185.50",
            "context": {
                "symbol": "AAPL",
                "timeframe": "1h",
                "market_conditions": {"trend": "bullish"},
                "available_data": ["price", "volume"],
                "constraints": {},
                "user_preferences": {},
            },
            "reasoning": {
                "primary_reason": "Strong technical indicators",
                "supporting_factors": ["RSI oversold", "MACD positive"],
                "alternatives_considered": ["Wait", "Sell"],
                "risks_assessed": ["Market volatility"],
                "confidence_explanation": "High confidence due to clear signals",
                "expected_outcome": "Expected 5% upside",
            },
            "confidence_level": ConfidenceLevel.HIGH,
            "metadata": {"test": True},
        }

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_log_decision(self):
        """Test decision logging."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)

        self.assertIsInstance(decision_id, str)
        self.assertTrue(decision_id.startswith("TestAgent_forecast_"))

        # Verify decision was stored
        decision = self.logger.get_decision(decision_id)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent_name, "TestAgent")
        self.assertEqual(decision.decision_type, DecisionType.FORECAST)

    def test_get_decision(self):
        """Test retrieving a decision."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        decision = self.logger.get_decision(decision_id)

        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(decision.decision_id, decision_id)
        self.assertEqual(decision.agent_name, "TestAgent")
        self.assertEqual(decision.context.symbol, "AAPL")

    def test_get_agent_decisions(self):
        """Test retrieving decisions by agent."""
        # Log multiple decisions
        self.logger.log_decision(**self.sample_decision_data)
        self.logger.log_decision(**self.sample_decision_data)

        decisions = self.logger.get_agent_decisions("TestAgent", limit=10)

        self.assertIsInstance(decisions, list)
        self.assertGreaterEqual(len(decisions), 2)

        for decision in decisions:
            self.assertEqual(decision.agent_name, "TestAgent")

    def test_get_decisions_by_type(self):
        """Test retrieving decisions by type."""
        # Log decisions of different types
        self.logger.log_decision(**self.sample_decision_data)

        strategy_data = self.sample_decision_data.copy()
        strategy_data["decision_type"] = DecisionType.STRATEGY
        self.logger.log_decision(**strategy_data)

        forecast_decisions = self.logger.get_decisions_by_type(DecisionType.FORECAST)
        strategy_decisions = self.logger.get_decisions_by_type(DecisionType.STRATEGY)

        self.assertGreaterEqual(len(forecast_decisions), 1)
        self.assertGreaterEqual(len(strategy_decisions), 1)

        for decision in forecast_decisions:
            self.assertEqual(decision.decision_type, DecisionType.FORECAST)

    def test_get_summary(self):
        """Test retrieving decision summary."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        summary = self.logger.get_summary(decision_id)

        self.assertIsInstance(summary, str)
        self.assertIn("TestAgent", summary)
        self.assertIn("AAPL", summary)
        self.assertIn("Strong technical indicators", summary)

    def test_get_explanation(self):
        """Test retrieving decision explanation."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        explanation = self.logger.get_explanation(decision_id)

        self.assertIsInstance(explanation, str)
        self.assertIn("TestAgent", explanation)
        self.assertIn("AAPL", explanation)

    def test_get_statistics(self):
        """Test getting statistics."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)
        self.logger.log_decision(**self.sample_decision_data)

        stats = self.logger.get_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_decisions", stats)
        self.assertIn("decisions_by_agent", stats)
        self.assertIn("decisions_by_type", stats)
        self.assertIn("confidence_distribution", stats)
        self.assertIn("recent_activity", stats)

        self.assertGreaterEqual(stats["total_decisions"], 2)
        self.assertIn("TestAgent", stats["decisions_by_agent"])

    def test_clear_old_decisions(self):
        """Test clearing old decisions."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)

        # Verify decision exists
        decision = self.logger.get_decision(decision_id)
        self.assertIsNotNone(decision)

        # Clear old decisions (should clear our test decision)
        self.logger.clear_old_decisions(days=0)

        # Verify decision was cleared
        decision = self.logger.get_decision(decision_id)
        self.assertIsNone(decision)

    def test_invalid_decision_tree(self):
        """Test handling of invalid decision trees."""
        # Test with missing required fields
        invalid_data = self.sample_decision_data.copy()
        del invalid_data["agent_name"]

        with self.assertRaises(ValueError):
            self.logger.log_decision(**invalid_data)

        # Test with invalid decision type
        invalid_data = self.sample_decision_data.copy()
        invalid_data["decision_type"] = "INVALID_TYPE"

        with self.assertRaises(ValueError):
            self.logger.log_decision(**invalid_data)

        # Test with empty reasoning
        invalid_data = self.sample_decision_data.copy()
        invalid_data["reasoning"] = {}

        with self.assertRaises(ValueError):
            self.logger.log_decision(**invalid_data)

    def test_fallback_routes(self):
        """Test fallback routes when primary operations fail."""
        # Test fallback when Redis is unavailable
        logger_no_redis = ReasoningLogger(
            redis_host="invalid_host",
            redis_port=9999,
            redis_db=1,
            log_dir=self.temp_dir,
            enable_gpt_explanations=False,
        )

        # Should still work with file-based fallback
        decision_id = logger_no_redis.log_decision(**self.sample_decision_data)
        self.assertIsInstance(decision_id, str)

        # Test fallback when log directory is read-only
        import os

        os.chmod(self.temp_dir, 0o444)  # Read-only

        try:
            logger_readonly = ReasoningLogger(
                redis_host="localhost",
                redis_port=6379,
                redis_db=1,
                log_dir=self.temp_dir,
                enable_gpt_explanations=False,
            )

            # Should still work with memory fallback
            decision_id = logger_readonly.log_decision(**self.sample_decision_data)
            self.assertIsInstance(decision_id, str)
        finally:
            os.chmod(self.temp_dir, 0o755)  # Restore permissions

    def test_decision_tree_validation(self):
        """Test validation of decision tree structure."""
        # Test decision tree with circular references
        circular_data = self.sample_decision_data.copy()
        circular_data["reasoning"]["supporting_factors"] = ["self_reference"]
        circular_data["metadata"] = {"circular_ref": circular_data}

        # Should handle gracefully
        decision_id = self.logger.log_decision(**circular_data)
        self.assertIsInstance(decision_id, str)

        # Test decision tree with deep nesting
        deep_data = self.sample_decision_data.copy()
        deep_data["context"]["nested"] = {
            "level1": {"level2": {"level3": {"level4": {"level5": "deep"}}}}
        }

        decision_id = self.logger.log_decision(**deep_data)
        self.assertIsInstance(decision_id, str)

        # Test decision tree with large data
        large_data = self.sample_decision_data.copy()
        large_data["reasoning"]["supporting_factors"] = ["factor"] * 1000

        decision_id = self.logger.log_decision(**large_data)
        self.assertIsInstance(decision_id, str)

    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test recovery from corrupted decision data
        corrupted_data = self.sample_decision_data.copy()
        corrupted_data["context"] = None

        with self.assertRaises(ValueError):
            self.logger.log_decision(**corrupted_data)

        # Test recovery from invalid confidence level
        invalid_confidence_data = self.sample_decision_data.copy()
        invalid_confidence_data["confidence_level"] = "INVALID"

        with self.assertRaises(ValueError):
            self.logger.log_decision(**invalid_confidence_data)

        # Test recovery from missing context fields
        incomplete_data = self.sample_decision_data.copy()
        incomplete_data["context"] = {"symbol": "AAPL"}  # Missing required fields

        # Should work with default values
        decision_id = self.logger.log_decision(**incomplete_data)
        self.assertIsInstance(decision_id, str)


class TestReasoningDisplay(unittest.TestCase):
    """Test the ReasoningDisplay class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ReasoningLogger(
            redis_host="localhost",
            redis_port=6379,
            redis_db=2,  # Use different DB for testing
            log_dir=self.temp_dir,
            enable_gpt_explanations=False,
        )
        self.display = ReasoningDisplay(self.logger)

        # Create sample decision
        self.sample_decision_data = {
            "agent_name": "TestAgent",
            "decision_type": DecisionType.FORECAST,
            "action_taken": "Predicted AAPL will reach $185.50",
            "context": {
                "symbol": "AAPL",
                "timeframe": "1h",
                "market_conditions": {"trend": "bullish"},
                "available_data": ["price", "volume"],
                "constraints": {},
                "user_preferences": {},
            },
            "reasoning": {
                "primary_reason": "Strong technical indicators",
                "supporting_factors": ["RSI oversold", "MACD positive"],
                "alternatives_considered": ["Wait", "Sell"],
                "risks_assessed": ["Market volatility"],
                "confidence_explanation": "High confidence due to clear signals",
                "expected_outcome": "Expected 5% upside",
            },
            "confidence_level": ConfidenceLevel.HIGH,
            "metadata": {"test": True},
        }

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_display_decision_terminal(self):
        """Test terminal decision display."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        decision = self.logger.get_decision(decision_id)

        # This should not raise an exception
        self.display.display_decision_terminal(decision)

    def test_display_recent_decisions_terminal(self):
        """Test terminal recent decisions display."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)
        self.logger.log_decision(**self.sample_decision_data)

        # This should not raise an exception
        self.display.display_recent_decisions_terminal(limit=5)

    def test_display_statistics_terminal(self):
        """Test terminal statistics display."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)

        # This should not raise an exception
        self.display.display_statistics_terminal()

    def test_display_decision_streamlit(self):
        """Test Streamlit decision display."""
        decision_id = self.logger.log_decision(**self.sample_decision_data)
        decision = self.logger.get_decision(decision_id)

        # This should not raise an exception
        self.display.display_decision_streamlit(decision)

    def test_display_recent_decisions_streamlit(self):
        """Test Streamlit recent decisions display."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)
        self.logger.log_decision(**self.sample_decision_data)

        # This should not raise an exception
        self.display.display_recent_decisions_streamlit(limit=5)

    def test_display_statistics_streamlit(self):
        """Test Streamlit statistics display."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)

        # This should not raise an exception
        self.display.display_statistics_streamlit()

    def test_create_streamlit_sidebar(self):
        """Test Streamlit sidebar creation."""
        # Log some decisions
        self.logger.log_decision(**self.sample_decision_data)

        # This should not raise an exception
        self.display.create_streamlit_sidebar()

    def test_display_invalid_decision(self):
        """Test display handling of invalid decisions."""
        # Test with None decision
        with self.assertRaises(ValueError):
            self.display.display_decision_terminal(None)

        # Test with invalid decision object
        invalid_decision = "not_a_decision"
        with self.assertRaises(ValueError):
            self.display.display_decision_terminal(invalid_decision)

    def test_display_fallback_routes(self):
        """Test display fallback routes when primary display fails."""
        # Test fallback when decision has missing fields
        incomplete_data = self.sample_decision_data.copy()
        del incomplete_data["reasoning"]["primary_reason"]

        decision_id = self.logger.log_decision(**incomplete_data)
        decision = self.logger.get_decision(decision_id)

        # Should handle gracefully
        self.display.display_decision_terminal(decision)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ReasoningLogger(
            redis_host="localhost",
            redis_port=6379,
            redis_db=3,  # Use different DB for testing
            log_dir=self.temp_dir,
            enable_gpt_explanations=False,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_log_forecast_decision(self):
        """Test forecast decision logging convenience function."""
        decision_id = log_forecast_decision(
            agent_name="TestAgent",
            symbol="AAPL",
            prediction="$185.50",
            confidence=ConfidenceLevel.HIGH,
            reasoning="Strong technical indicators",
            logger=self.logger,
        )

        self.assertIsInstance(decision_id, str)
        decision = self.logger.get_decision(decision_id)
        self.assertEqual(decision.decision_type, DecisionType.FORECAST)
        self.assertEqual(decision.context.symbol, "AAPL")

    def test_log_strategy_decision(self):
        """Test strategy decision logging convenience function."""
        decision_id = log_strategy_decision(
            agent_name="TestAgent",
            symbol="AAPL",
            action="BUY",
            confidence=ConfidenceLevel.MEDIUM,
            reasoning="RSI oversold condition",
            logger=self.logger,
        )

        self.assertIsInstance(decision_id, str)
        decision = self.logger.get_decision(decision_id)
        self.assertEqual(decision.decision_type, DecisionType.STRATEGY)
        self.assertEqual(decision.context.symbol, "AAPL")


class TestDataStructures(unittest.TestCase):
    """Test data structures."""

    def test_decision_context(self):
        """Test DecisionContext data structure."""
        context = DecisionContext(
            symbol="AAPL",
            timeframe="1h",
            market_conditions={"trend": "bullish"},
            available_data=["price", "volume"],
            constraints={},
            user_preferences={},
        )

        self.assertEqual(context.symbol, "AAPL")
        self.assertEqual(context.timeframe, "1h")
        self.assertIn("trend", context.market_conditions)

    def test_decision_reasoning(self):
        """Test DecisionReasoning data structure."""
        reasoning = DecisionReasoning(
            primary_reason="Strong technical indicators",
            supporting_factors=["RSI oversold", "MACD positive"],
            alternatives_considered=["Wait", "Sell"],
            risks_assessed=["Market volatility"],
            confidence_explanation="High confidence due to clear signals",
            expected_outcome="Expected 5% upside",
        )

        self.assertEqual(reasoning.primary_reason, "Strong technical indicators")
        self.assertIn("RSI oversold", reasoning.supporting_factors)

    def test_agent_decision(self):
        """Test AgentDecision data structure."""
        context = DecisionContext(
            symbol="AAPL",
            timeframe="1h",
            market_conditions={"trend": "bullish"},
            available_data=["price", "volume"],
            constraints={},
            user_preferences={},
        )

        reasoning = DecisionReasoning(
            primary_reason="Strong technical indicators",
            supporting_factors=["RSI oversold", "MACD positive"],
            alternatives_considered=["Wait", "Sell"],
            risks_assessed=["Market volatility"],
            confidence_explanation="High confidence due to clear signals",
            expected_outcome="Expected 5% upside",
        )

        decision = AgentDecision(
            decision_id="test_id",
            agent_name="TestAgent",
            decision_type=DecisionType.FORECAST,
            action_taken="Predicted AAPL will reach $185.50",
            context=context,
            reasoning=reasoning,
            confidence_level=ConfidenceLevel.HIGH,
            timestamp=datetime.now(),
            metadata={"test": True},
        )

        self.assertEqual(decision.decision_id, "test_id")
        self.assertEqual(decision.agent_name, "TestAgent")
        self.assertEqual(decision.decision_type, DecisionType.FORECAST)

    def test_invalid_data_structures(self):
        """Test handling of invalid data structures."""
        # Test DecisionContext with invalid symbol
        with self.assertRaises(ValueError):
            DecisionContext(
                symbol="",  # Empty symbol
                timeframe="1h",
                market_conditions={},
                available_data=[],
                constraints={},
                user_preferences={},
            )

        # Test DecisionReasoning with empty primary reason
        with self.assertRaises(ValueError):
            DecisionReasoning(
                primary_reason="",  # Empty primary reason
                supporting_factors=[],
                alternatives_considered=[],
                risks_assessed=[],
                confidence_explanation="",
                expected_outcome="",
            )

        # Test AgentDecision with invalid decision type
        with self.assertRaises(ValueError):
            AgentDecision(
                decision_id="test_id",
                agent_name="TestAgent",
                decision_type="INVALID_TYPE",  # Invalid decision type
                action_taken="Test action",
                context=DecisionContext("AAPL", "1h", {}, [], {}, {}),
                reasoning=DecisionReasoning("Test reason", [], [], [], "", ""),
                confidence_level=ConfidenceLevel.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
