"""
Test script for Batch 5 fixes.

This script tests various fixes and improvements made in Batch 5,
including fallback logic, strategy routing, and error handling.
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_prompt_agent_fallback():
    """Test prompt agent fallback logic."""
    logger.info("Testing prompt agent fallback...")

    try:
        from agents.prompt_agent import PromptAgent, RequestType

        agent = PromptAgent(
            use_regex_first=True, use_local_llm=False, use_openai_fallback=False
        )

        # Test empty prompt fallback
        result = agent.route_request("")
        assert result.request_type == RequestType.GENERAL
        assert result.primary_agent == "GeneralAgent"
        assert "No strategy detected" in result.metadata["message"]

        logger.info("✅ Prompt agent fallback test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Prompt agent fallback test failed: {e}")
        return False


def test_strategy_router():
    """Test strategy router functionality."""
    logger.info("Testing strategy router...")

    try:
        from trading.strategies.strategy_router import StrategyRouter

        router = StrategyRouter()

        # Test strategy matching
        matches = router.find_strategy_matches("use RSI and MACD strategy")
        assert len(matches) >= 2  # Should find both RSI and MACD

        # Test best strategy selection
        best_match = router.select_best_strategy("RSI strategy for AAPL")
        assert best_match is not None
        assert "RSI" in best_match.strategy_name

        logger.info("✅ Strategy router test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy router test failed: {e}")
        return False


def test_meta_agent_orchestrator():
    """Test meta agent orchestrator."""
    logger.info("Testing meta agent orchestrator...")

    try:
        from trading.agents.meta_agent_orchestrator import (
            AgentCall,
            MetaAgentOrchestrator,
        )

        orchestrator = MetaAgentOrchestrator()

        # Test agent registration
        mock_agent = type(
            "MockAgent", (), {"handle_fallback": lambda: "fallback response"}
        )()
        orchestrator.register_agent("TestAgent", mock_agent)

        # Test orchestration with error handling
        agent_calls = [
            AgentCall(
                agent_name="TestAgent",
                method="handle_fallback",
                args=(),
                kwargs={},
                timeout=5.0,
            )
        ]

        result = orchestrator.orchestrate_agents(agent_calls)
        assert result.success
        assert result.agent_used == "TestAgent"

        logger.info("✅ Meta agent orchestrator test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Meta agent orchestrator test failed: {e}")
        return False


def test_backtest_utils():
    """Test backtest utils guard clauses."""
    logger.info("Testing backtest utils...")

    try:
        import pandas as pd

        from trading.backtesting.backtest_utils import BacktestUtils

        utils = BacktestUtils()

        # Test with empty data
        empty_df = pd.DataFrame()
        result = utils.validate_backtest_data(empty_df)
        assert not result["valid"]
        assert "empty" in result["error"].lower()

        # Test with missing required columns
        invalid_df = pd.DataFrame({"price": [100, 101, 102]})
        result = utils.validate_backtest_data(invalid_df)
        assert not result["valid"]
        assert "required" in result["error"].lower()

        # Test with valid data
        valid_df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [103, 104, 105],
                "volume": [1000, 1100, 1200],
            }
        )
        result = utils.validate_backtest_data(valid_df)
        assert result["valid"]

        logger.info("✅ Backtest utils test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Backtest utils test failed: {e}")
        return False


def test_prompt_formatter():
    """Test prompt formatter improvements."""
    logger.info("Testing prompt formatter...")

    try:
        from trading.nlp.prompt_formatter import PromptFormatter

        formatter = PromptFormatter()

        # Test context extraction
        prompt = "Use RSI strategy with period 14 for AAPL stock"
        context = formatter.extract_context(prompt)
        assert context["strategy"] == "RSI"
        assert context["symbol"] == "AAPL"
        assert context["parameters"]["period"] == 14

        # Test prompt enhancement
        enhanced = formatter.enhance_prompt(prompt)
        assert "RSI" in enhanced
        assert "AAPL" in enhanced
        assert "period" in enhanced

        logger.info("✅ Prompt formatter test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Prompt formatter test failed: {e}")
        return False


def test_prompt_clarification_agent():
    """Test prompt clarification agent."""
    logger.info("Testing prompt clarification agent...")

    try:
        from trading.agents.prompt_clarification_agent import PromptClarificationAgent

        agent = PromptClarificationAgent()

        # Test ambiguous prompt detection
        ambiguous_prompt = "use strategy"
        is_ambiguous = agent.is_ambiguous(ambiguous_prompt)
        assert is_ambiguous

        # Test clarification generation
        clarifications = agent.generate_clarifications(ambiguous_prompt)
        assert len(clarifications) > 0
        assert any("strategy" in c.lower() for c in clarifications)

        # Test clear prompt
        clear_prompt = "Use RSI strategy with period 14 for AAPL"
        is_ambiguous = agent.is_ambiguous(clear_prompt)
        assert not is_ambiguous

        logger.info("✅ Prompt clarification agent test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Prompt clarification agent test failed: {e}")
        return False


def test_strategy_ranking():
    """Test strategy ranking improvements."""
    logger.info("Testing strategy ranking...")

    try:
        from trading.strategies.strategy_ranking import StrategyRanker

        ranker = StrategyRanker()

        # Test strategy scoring
        strategies = ["RSI", "MACD", "Bollinger Bands"]
        scores = ranker.score_strategies(strategies, "AAPL")
        assert len(scores) == len(strategies)
        assert all(0 <= score <= 1 for score in scores.values())

        # Test strategy ranking
        ranked = ranker.rank_strategies(strategies, "AAPL")
        assert len(ranked) == len(strategies)
        assert ranked[0]["score"] >= ranked[1]["score"]  # Should be sorted

        logger.info("✅ Strategy ranking test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy ranking test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🚀 Starting Batch 5 Fixes Test Suite")
    logger.info("=" * 50)

    test_results = []

    # Run all tests
    tests = [
        ("Prompt Agent Fallback", test_prompt_agent_fallback),
        ("Strategy Router", test_strategy_router),
        ("Meta Agent Orchestrator", test_meta_agent_orchestrator),
        ("Backtest Utils", test_backtest_utils),
        ("Prompt Formatter", test_prompt_formatter),
        ("Prompt Clarification Agent", test_prompt_clarification_agent),
        ("Strategy Ranking", test_strategy_ranking),
    ]

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            test_results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All Batch 5 fixes working correctly!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
