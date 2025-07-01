#!/usr/bin/env python3
"""
Test Execution Agent

Test the ExecutionAgent functionality.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from trading.agents.execution_agent import ExecutionAgent, TradeSignal, TradeDirection
from trading.agents.base_agent_interface import AgentConfig

async def test_execution_agent():
    """Test the execution agent."""
    print("🧪 Testing Execution Agent")
    print("=" * 50)
    
    # Create agent
    config = {
        'name': 'test_execution_agent',
        'enabled': True,
        'custom_config': {
            'execution_mode': 'simulation',
            'max_positions': 3,
            'min_confidence': 0.5
        }
    }
    
    agent_config = AgentConfig(**config)
    agent = ExecutionAgent(agent_config)
    
    print("✅ ExecutionAgent created successfully")
    
    # Test signal validation
    print("\n📋 Testing signal validation...")
    
    valid_signal = TradeSignal(
        symbol="AAPL",
        direction=TradeDirection.LONG,
        strategy="test",
        confidence=0.8,
        entry_price=150.00
    )
    
    invalid_signal = TradeSignal(
        symbol="",
        direction=TradeDirection.LONG,
        strategy="test",
        confidence=1.5,  # Invalid confidence
        entry_price=-10.0  # Invalid price
    )
    
    print(f"  Valid signal: {agent._validate_signal(valid_signal)}")
    print(f"  Invalid signal: {agent._validate_signal(invalid_signal)}")
    
    # Test position limits
    print("\n📊 Testing position limits...")
    print(f"  Position limit check: {agent._check_position_limits(valid_signal)}")
    
    # Test execution price calculation
    print("\n💰 Testing execution price calculation...")
    market_data = {'AAPL': {'price': 150.00, 'volatility': 0.15}}
    execution_price = agent._calculate_execution_price(valid_signal, market_data)
    print(f"  Entry price: ${valid_signal.entry_price:.2f}")
    print(f"  Execution price: ${execution_price:.2f}")
    print(f"  Slippage: {abs(execution_price - valid_signal.entry_price) / valid_signal.entry_price:.4f}")
    
    # Test trade execution
    print("\n🔄 Testing trade execution...")
    signals = [valid_signal]
    result = await agent.execute(signals=signals, market_data=market_data)
    
    if result.success:
        print("✅ Trade execution successful")
        print(f"  Message: {result.message}")
        
        # Show results
        execution_results = result.data['execution_results']
        summary = result.data['summary']
        
        print(f"  Success rate: {summary['success_rate']:.2%}")
        print(f"  Total slippage: ${summary['total_slippage']:.2f}")
        print(f"  Total fees: ${summary['total_fees']:.2f}")
        
    else:
        print(f"❌ Trade execution failed: {result.message}")
    
    # Test portfolio status
    print("\n📈 Testing portfolio status...")
    portfolio_status = agent.get_portfolio_status()
    print(f"  Cash: ${portfolio_status['cash']:.2f}")
    print(f"  Equity: ${portfolio_status['equity']:.2f}")
    print(f"  Open positions: {len(portfolio_status['open_positions'])}")
    
    # Test execution history
    print("\n📜 Testing execution history...")
    history = agent.get_execution_history(limit=5)
    print(f"  History entries: {len(history)}")
    
    # Test trade log
    print("\n📄 Testing trade log...")
    trade_log = agent.get_trade_log()
    print(f"  Trade log entries: {len(trade_log)}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_execution_agent()) 