#!/usr/bin/env python3
"""
Test Execution Agent

This script tests the execution agent functionality using the new modular structure.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading.agents.base_agent_interface import AgentConfig
from trading.agents.execution import ExecutionAgent, create_execution_agent
from trading.portfolio.portfolio_manager import TradeDirection
from trading.agents.execution.trade_signals import TradeSignal


async def test_execution_agent():
    """Test execution agent functionality."""
    logger.info("ðŸ§ª Testing Execution Agent (Modular)")
    logger.info("=" * 50)

    try:
        # Create execution agent using factory function
        config = {
            "execution_mode": "simulation",
            "max_positions": 3,
            "min_confidence": 0.6,
            "max_slippage": 0.002,
        }

        agent = create_execution_agent(config)
        logger.info("âœ… ExecutionAgent created successfully using factory function")

        # Test signal validation
        logger.info("\nðŸ“‹ Testing signal validation...")

        valid_signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="test",
            confidence=0.8,
            entry_price=150.00,
        )

        invalid_signal = TradeSignal(
            symbol="INVALID",
            direction=TradeDirection.LONG,
            strategy="test",
            confidence=0.3,  # Too low
            entry_price=0.00,  # Invalid price
        )

        logger.info(f"  Valid signal: {agent._validate_signal(valid_signal)}")
        logger.info(f"  Invalid signal: {agent._validate_signal(invalid_signal)}")

        # Test position limits
        logger.info("\nðŸ“Š Testing position limits...")
        logger.info(
            f"  Position limit check: {agent._check_position_limits(valid_signal)}"
        )

        # Test execution price calculation
        logger.info("\nðŸ’° Testing execution price calculation...")
        market_data = {"AAPL_price": 150.50, "AAPL_volume": 1000000}
        execution_price = agent._calculate_execution_price(valid_signal, market_data)

        logger.info(f"  Entry price: ${valid_signal.entry_price:.2f}")
        logger.info(f"  Execution price: ${execution_price:.2f}")
        logger.info(
            f"  Slippage: {abs(execution_price - valid_signal.entry_price) / valid_signal.entry_price:.4f}"
        )

        # Test trade execution
        logger.info("\nðŸ”„ Testing trade execution...")

        result = await agent.execute(signal=valid_signal, market_data=market_data)

        if result.success:
            logger.info("âœ… Trade execution successful")
            logger.info(f"  Message: {result.message}")

            # Test portfolio status
            portfolio_status = agent.get_portfolio_status()
            logger.info(f"  Portfolio status: {portfolio_status}")
        else:
            logger.error(f"âŒ Trade execution failed: {result.message}")

        # Test execution history
        logger.info("\nðŸ“œ Testing execution history...")
        history = agent.get_execution_history()
        logger.info(f"  History entries: {len(history)}")

        # Test trade log
        logger.info("\nðŸ“„ Testing trade log...")
        trade_log = agent.get_trade_log()
        logger.info(f"  Trade log entries: {len(trade_log)}")

        # Test modular components
        logger.info("\nðŸ”§ Testing modular components...")
        
        # Test risk controls
        from trading.agents.execution.risk_controls import create_default_risk_controls
        risk_controls = create_default_risk_controls()
        logger.info(f"  Risk controls created: {risk_controls.max_position_size}")
        
        # Test position manager
        position_manager = agent.position_manager
        risk_summary = position_manager.get_risk_summary()
        logger.info(f"  Risk summary: {risk_summary}")

        logger.info("\nâœ… All tests completed!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_execution_agent())
