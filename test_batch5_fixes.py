#!/usr/bin/env python3
"""
Simple test script for Batch 5 fixes
"""

import sys
import os
import logging
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prompt_agent_fallback():
    """Test prompt agent fallback logic."""
    logger.info("Testing prompt agent fallback...")
    
    try:
        from agents.prompt_agent import PromptAgent, RequestType
        
        agent = PromptAgent(
            use_regex_first=True,
            use_local_llm=False,
            use_openai_fallback=False
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
        from trading.strategies.strategy_router import StrategyRouter, StrategyMatch
        
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
        from trading.agents.meta_agent_orchestrator import MetaAgentOrchestrator, AgentCall
        
        orchestrator = MetaAgentOrchestrator()
        
        # Test agent registration
        mock_agent = type('MockAgent', (), {'handle_fallback': lambda: "fallback response"})()
        orchestrator.register_agent("TestAgent", mock_agent)
        
        # Test orchestration with error handling
        agent_calls = [
            AgentCall(
                agent_name="TestAgent",
                method="handle_fallback",
                args=(),
                kwargs={},
                timeout=5.0
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
        
        # Test missing 'Buy' column guard clause
        df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Sell': [0, 0, 1]
        })
        
        report = utils.generate_backtest_report(df)
        assert report is not None
        assert not report.metadata["validation_passed"]
        assert "Missing 'Buy' column" in report.metadata["error"]
        
        logger.info("✅ Backtest utils test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtest utils test failed: {e}")
        return False

def test_prompt_formatter():
    """Test prompt formatter JSON handling."""
    logger.info("Testing prompt formatter...")
    
    try:
        from trading.utils.prompt_formatter import PromptFormatter
        
        formatter = PromptFormatter()
        
        # Test malformed JSON fallback
        malformed_json = '{"prompt": "forecast AAPL",'
        result = formatter.format_prompt(malformed_json)
        
        assert result.format_type == "fallback"
        assert not result.validation_passed
        assert "JSON decode error" in result.errors[0]
        
        # Test valid JSON
        valid_json = '{"prompt": "forecast AAPL", "timeframe": "7d"}'
        result = formatter.format_prompt(valid_json)
        
        assert result.format_type == "json"
        assert result.validation_passed
        assert "forecast AAPL" in result.formatted_prompt
        
        logger.info("✅ Prompt formatter test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prompt formatter test failed: {e}")
        return False

def test_prompt_clarification_agent():
    """Test prompt clarification agent."""
    logger.info("Testing prompt clarification agent...")
    
    try:
        from trading.agents.prompt_clarification_agent import PromptClarificationAgent, AmbiguityType
        
        agent = PromptClarificationAgent()
        
        # Test multiple strategy ambiguity detection
        clarification = agent.analyze_prompt("use RSI and MACD strategy")
        assert clarification is not None
        assert clarification.ambiguity_type == AmbiguityType.MULTIPLE_STRATEGIES
        assert len(clarification.options) >= 2
        
        # Test vague request detection
        clarification = agent.analyze_prompt("analyze check examine market")
        assert clarification is not None
        assert clarification.ambiguity_type == AmbiguityType.VAGUE_REQUEST
        
        logger.info("✅ Prompt clarification agent test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prompt clarification agent test failed: {e}")
        return False

def test_strategy_ranking():
    """Test strategy ranking logic."""
    logger.info("Testing strategy ranking...")
    
    try:
        from trading.strategies.strategy_ranking import StrategyRanker
        
        ranker = StrategyRanker()
        
        # Test strategy usage recording
        ranker.record_strategy_usage(
            strategy_name="RSI_Strategy",
            prompt="use RSI strategy",
            success=True,
            confidence=0.8,
            performance_metrics={"returns": 0.05}
        )
        
        # Test ranking
        rankings = ranker.rank_strategies()
        assert len(rankings) >= 1
        
        # Test recommendations
        recommendations = ranker.get_strategy_recommendations("forecast AAPL")
        assert isinstance(recommendations, list)
        
        logger.info("✅ Strategy ranking test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy ranking test failed: {e}")
        return False

def main():
    """Run all Batch 5 fix tests."""
    logger.info("🧪 Running Batch 5 Fix Tests\n")
    
    tests = [
        test_prompt_agent_fallback,
        test_strategy_router,
        test_meta_agent_orchestrator,
        test_backtest_utils,
        test_prompt_formatter,
        test_prompt_clarification_agent,
        test_strategy_ranking
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed with exception: {e}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Batch 5 fixes are working correctly!")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 