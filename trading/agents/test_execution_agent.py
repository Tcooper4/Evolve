#!/usr/bin/env python3
"""
Test Execution Agent

This script tests the execution agent functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading.agents.execution_agent import ExecutionAgent, TradeSignal, TradeDirection
from trading.agents.base_agent_interface import AgentConfig

async def test_execution_agent():
    """Test execution agent functionality."""
    logger.info("🧪 Testing Execution Agent")
    logger.info("=" * 50)
    
    try:
        # Create execution agent
        config = {
            'name': 'test_execution_agent',
            'enabled': True,
            'custom_config': {
                'execution_mode': 'simulation',
                'max_positions': 3,
                'min_confidence': 0.6,
                'max_slippage': 0.002
            }
        }
        
        agent_config = AgentConfig(**config)
        agent = ExecutionAgent(agent_config)
        
        logger.info("✅ ExecutionAgent created successfully")
        
        # Test signal validation
        logger.info("\n📋 Testing signal validation...")
        
        valid_signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="test",
            confidence=0.8,
            entry_price=150.00
        )
        
        invalid_signal = TradeSignal(
            symbol="INVALID",
            direction=TradeDirection.LONG,
            strategy="test",
            confidence=0.3,  # Too low
            entry_price=0.00  # Invalid price
        )
        
        logger.info(f"  Valid signal: {agent._validate_signal(valid_signal)}")
        logger.info(f"  Invalid signal: {agent._validate_signal(invalid_signal)}")
        
        # Test position limits
        logger.info("\n📊 Testing position limits...")
        logger.info(f"  Position limit check: {agent._check_position_limits(valid_signal)}")
        
        # Test execution price calculation
        logger.info("\n💰 Testing execution price calculation...")
        market_data = {'AAPL': {'price': 150.50, 'volume': 1000000}}
        execution_price = agent._calculate_execution_price(valid_signal, market_data)
        
        logger.info(f"  Entry price: ${valid_signal.entry_price:.2f}")
        logger.info(f"  Execution price: ${execution_price:.2f}")
        logger.info(f"  Slippage: {abs(execution_price - valid_signal.entry_price) / valid_signal.entry_price:.4f}")
        
        # Test trade execution
        logger.info("\n🔄 Testing trade execution...")
        
        result = await agent.execute(
            signals=[valid_signal],
            market_data=market_data
        )
        
        if result.success:
            logger.info("✅ Trade execution successful")
            logger.info(f"  Message: {result.message}")
            
            # Test execution summary
            summary = agent.get_execution_summary()
            logger.info(f"  Success rate: {summary['success_rate']:.2%}")
            logger.info(f"  Total slippage: ${summary['total_slippage']:.2f}")
            logger.info(f"  Total fees: ${summary['total_fees']:.2f}")
        else:
            logger.error(f"❌ Trade execution failed: {result.message}")
        
        # Test portfolio status
        logger.info("\n📈 Testing portfolio status...")
        portfolio_status = await agent.portfolio_manager.get_portfolio_status()
        logger.info(f"  Cash: ${portfolio_status['cash']:.2f}")
        logger.info(f"  Equity: ${portfolio_status['equity']:.2f}")
        logger.info(f"  Open positions: {len(portfolio_status['open_positions'])}")
        
        # Test execution history
        logger.info("\n📜 Testing execution history...")
        history = agent.get_execution_history()
        logger.info(f"  History entries: {len(history)}")
        
        # Test trade log
        logger.info("\n📄 Testing trade log...")
        trade_log = agent.get_trade_log()
        logger.info(f"  Trade log entries: {len(trade_log)}")
        
        logger.info("\n✅ All tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_execution_agent()) 