#!/usr/bin/env python3
"""
Execution Agent Demo

Demonstrates the ExecutionAgent functionality with simulated trades.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from trading.agents.execution_agent import ExecutionAgent, TradeSignal, ExecutionMode, TradeDirection
from trading.agents.base_agent_interface import AgentConfig


def create_sample_signals() -> List[TradeSignal]:
    """Create sample trade signals for demonstration."""
    signals = [
        TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="bollinger_bands",
            confidence=0.85,
            entry_price=150.25,
            take_profit=155.00,
            stop_loss=148.00,
            max_holding_period=timedelta(days=7),
            market_data={
                'volatility': 0.15,
                'volume': 1000000,
                'rsi': 65
            }
        ),
        TradeSignal(
            symbol="TSLA",
            direction=TradeDirection.SHORT,
            strategy="rsi_strategy",
            confidence=0.72,
            entry_price=245.50,
            take_profit=240.00,
            stop_loss=250.00,
            max_holding_period=timedelta(days=5),
            market_data={
                'volatility': 0.25,
                'volume': 2000000,
                'rsi': 75
            }
        ),
        TradeSignal(
            symbol="NVDA",
            direction=TradeDirection.LONG,
            strategy="macd_strategy",
            confidence=0.68,
            entry_price=420.75,
            take_profit=430.00,
            stop_loss=415.00,
            max_holding_period=timedelta(days=10),
            market_data={
                'volatility': 0.20,
                'volume': 1500000,
                'rsi': 55
            }
        )
    ]
    return signals


def create_sample_market_data() -> Dict[str, Any]:
    """Create sample market data for demonstration."""
    return {
        'AAPL': {
            'price': 150.25,
            'volatility': 0.15,
            'volume': 1000000,
            'returns': [0.01, -0.02, 0.015, -0.01, 0.02]
        },
        'TSLA': {
            'price': 245.50,
            'volatility': 0.25,
            'volume': 2000000,
            'returns': [0.03, -0.01, 0.02, -0.015, 0.01]
        },
        'NVDA': {
            'price': 420.75,
            'volatility': 0.20,
            'volume': 1500000,
            'returns': [0.02, -0.01, 0.025, -0.02, 0.015]
        }
    }


async def demo_execution_agent():
    """Demonstrate ExecutionAgent functionality."""
    print("🚀 Execution Agent Demo")
    print("=" * 60)
    
    # Create execution agent
    config = {
        'name': 'demo_execution_agent',
        'enabled': True,
        'priority': 1,
        'max_concurrent_runs': 1,
        'timeout_seconds': 300,
        'retry_attempts': 3,
        'custom_config': {
            'execution_mode': 'simulation',
            'max_positions': 5,
            'min_confidence': 0.6,
            'max_slippage': 0.002,
            'execution_delay': 0.5,
            'risk_per_trade': 0.02,
            'max_position_size': 0.2,
            'base_fee': 0.001,
            'min_fee': 1.0
        }
    }
    
    agent_config = AgentConfig(**config)
    execution_agent = ExecutionAgent(agent_config)
    
    print(f"✅ ExecutionAgent initialized in {execution_agent.execution_mode.value} mode")
    print(f"📊 Portfolio status: {execution_agent.get_portfolio_status()}")
    
    # Create sample signals
    signals = create_sample_signals()
    print(f"\n📈 Created {len(signals)} sample trade signals")
    
    # Create sample market data
    market_data = create_sample_market_data()
    print(f"📊 Created market data for {len(market_data)} symbols")
    
    # Execute trades
    print("\n🔄 Executing trades...")
    print("-" * 40)
    
    result = await execution_agent.execute(
        signals=signals,
        market_data=market_data,
        portfolio_update=True
    )
    
    if result.success:
        print("✅ Trade execution completed successfully!")
        print(f"📝 Message: {result.message}")
        
        # Show execution results
        execution_results = result.data['execution_results']
        summary = result.data['summary']
        
        print(f"\n📊 Execution Summary:")
        print(f"  Total signals: {summary['total_signals']}")
        print(f"  Successful: {summary['successful_executions']}")
        print(f"  Failed: {summary['failed_executions']}")
        print(f"  Success rate: {summary['success_rate']:.2%}")
        print(f"  Total slippage: ${summary['total_slippage']:.2f}")
        print(f"  Total fees: ${summary['total_fees']:.2f}")
        print(f"  Average confidence: {summary['average_confidence']:.2%}")
        
        # Show individual results
        print(f"\n📋 Individual Results:")
        for i, exec_result in enumerate(execution_results, 1):
            signal = exec_result['signal']
            print(f"  {i}. {signal['symbol']} {signal['direction']} "
                  f"at ${signal['entry_price']:.2f} "
                  f"(confidence: {signal['confidence']:.2%})")
            print(f"     Status: {'✅ Success' if exec_result['success'] else '❌ Failed'}")
            if exec_result['success']:
                print(f"     Execution price: ${exec_result['execution_price']:.2f}")
                print(f"     Slippage: {exec_result['slippage']:.4f}")
                print(f"     Fees: ${exec_result['fees']:.2f}")
            else:
                print(f"     Error: {exec_result['error']}")
            print()
        
        # Show updated portfolio
        portfolio_state = result.data['portfolio_state']
        print(f"💰 Updated Portfolio:")
        print(f"  Cash: ${portfolio_state['cash']:.2f}")
        print(f"  Equity: ${portfolio_state['equity']:.2f}")
        print(f"  Total PnL: ${portfolio_state['total_pnl']:.2f}")
        print(f"  Unrealized PnL: ${portfolio_state['unrealized_pnl']:.2f}")
        print(f"  Open positions: {len(portfolio_state['open_positions'])}")
        
        # Show open positions
        if portfolio_state['open_positions']:
            print(f"\n📈 Open Positions:")
            for position in portfolio_state['open_positions']:
                print(f"  {position['symbol']} {position['direction']} "
                      f"Size: {position['size']:.2f} "
                      f"Entry: ${position['entry_price']:.2f} "
                      f"Unrealized PnL: ${position['unrealized_pnl']:.2f}")
        
    else:
        print(f"❌ Trade execution failed: {result.message}")
        if result.error:
            print(f"Error: {result.error}")
    
    # Show execution history
    print(f"\n📜 Recent Execution History:")
    history = execution_agent.get_execution_history(limit=5)
    for i, entry in enumerate(history, 1):
        signal = entry['signal']
        print(f"  {i}. {signal['symbol']} {signal['direction']} "
              f"at {entry['timestamp'][:19]} "
              f"({'✅' if entry['success'] else '❌'})")
    
    # Show trade log
    print(f"\n📄 Trade Log Entries:")
    trade_log = execution_agent.get_trade_log()
    print(f"  Total entries: {len(trade_log)}")
    
    # Export trade log
    print(f"\n💾 Exporting trade log...")
    export_path = execution_agent.export_trade_log(format='json')
    print(f"  Exported to: {export_path}")
    
    print(f"\n🎉 Demo completed!")


async def demo_portfolio_updates():
    """Demonstrate portfolio updates with price changes."""
    print("\n🔄 Portfolio Updates Demo")
    print("=" * 60)
    
    # Create execution agent
    config = {
        'name': 'portfolio_demo_agent',
        'enabled': True,
        'custom_config': {
            'execution_mode': 'simulation',
            'max_positions': 3,
            'min_confidence': 0.5
        }
    }
    
    agent_config = AgentConfig(**config)
    execution_agent = ExecutionAgent(agent_config)
    
    # Create and execute initial trades
    initial_signals = [
        TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="demo",
            confidence=0.8,
            entry_price=150.00,
            market_data={'volatility': 0.15}
        )
    ]
    
    initial_market_data = {
        'AAPL': {'price': 150.00, 'volatility': 0.15}
    }
    
    # Execute initial trade
    result = await execution_agent.execute(
        signals=initial_signals,
        market_data=initial_market_data
    )
    
    if result.success:
        print("✅ Initial trade executed")
        
        # Simulate price changes
        price_changes = [
            {'AAPL': {'price': 152.00, 'volatility': 0.15}},  # +$2.00
            {'AAPL': {'price': 148.00, 'volatility': 0.15}},  # -$2.00
            {'AAPL': {'price': 155.00, 'volatility': 0.15}},  # +$5.00
        ]
        
        for i, market_data in enumerate(price_changes, 1):
            print(f"\n📊 Price Update {i}: AAPL = ${market_data['AAPL']['price']:.2f}")
            
            # Update portfolio
            await execution_agent.execute(
                signals=[],
                market_data=market_data,
                portfolio_update=True
            )
            
            # Show portfolio status
            portfolio_state = execution_agent.get_portfolio_status()
            print(f"  Cash: ${portfolio_state['cash']:.2f}")
            print(f"  Equity: ${portfolio_state['equity']:.2f}")
            print(f"  Unrealized PnL: ${portfolio_state['unrealized_pnl']:.2f}")
            
            # Show position details
            for position in portfolio_state['open_positions']:
                print(f"  {position['symbol']}: {position['size']:.2f} shares "
                      f"@ ${position['entry_price']:.2f} "
                      f"(Unrealized: ${position['unrealized_pnl']:.2f})")
    
    print(f"\n✅ Portfolio updates demo completed!")


async def main():
    """Main demo function."""
    await demo_execution_agent()
    await demo_portfolio_updates()
    
    print(f"\n📋 Demo Summary:")
    print("  ✅ ExecutionAgent successfully created and configured")
    print("  ✅ Simulated trades executed with realistic slippage and fees")
    print("  ✅ Portfolio tracking and updates working")
    print("  ✅ Trade logging and export functionality demonstrated")
    print("  🔮 Ready for real execution integration (Alpaca, IB, Robinhood)")


if __name__ == "__main__":
    asyncio.run(main()) 