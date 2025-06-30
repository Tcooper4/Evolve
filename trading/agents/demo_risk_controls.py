#!/usr/bin/env python3
"""
Risk Controls Demo

Demonstrates the comprehensive risk management features of the ExecutionAgent,
including stop-loss, take-profit, automatic exits, and detailed logging.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from trading.agents.execution_agent import (
    ExecutionAgent, TradeSignal, ExecutionMode, TradeDirection,
    RiskControls, RiskThreshold, RiskThresholdType, ExitReason
)
from trading.agents.base_agent_interface import AgentConfig


def create_risk_aware_signals() -> List[TradeSignal]:
    """Create trade signals with risk controls for demonstration."""
    
    # Create risk controls with different threshold types
    percentage_risk = RiskControls(
        stop_loss=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.02),  # 2% stop loss
        take_profit=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.06),  # 6% take profit
        max_position_size=0.15,
        max_daily_loss=0.03,
        volatility_limit=0.4
    )
    
    atr_risk = RiskControls(
        stop_loss=RiskThreshold(RiskThresholdType.ATR_BASED, 0.0, atr_multiplier=2.0),
        take_profit=RiskThreshold(RiskThresholdType.ATR_BASED, 0.0, atr_multiplier=3.0),
        max_position_size=0.2,
        max_daily_loss=0.02,
        volatility_limit=0.5
    )
    
    fixed_risk = RiskControls(
        stop_loss=RiskThreshold(RiskThresholdType.FIXED, 5.0),  # $5 stop loss
        take_profit=RiskThreshold(RiskThresholdType.FIXED, 15.0),  # $15 take profit
        max_position_size=0.1,
        max_daily_loss=0.025,
        volatility_limit=0.3
    )
    
    signals = [
        TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="bollinger_bands",
            confidence=0.85,
            entry_price=150.25,
            max_holding_period=timedelta(days=7),
            risk_controls=percentage_risk,
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
            max_holding_period=timedelta(days=5),
            risk_controls=atr_risk,
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
            max_holding_period=timedelta(days=10),
            risk_controls=fixed_risk,
            market_data={
                'volatility': 0.20,
                'volume': 1500000,
                'rsi': 55
            }
        )
    ]
    return {'success': True, 'result': signals, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def create_market_scenarios() -> List[Dict[str, Any]]:
    """Create different market scenarios to test risk controls."""
    scenarios = [
        # Scenario 1: Normal market conditions
        {
            'name': 'Normal Market',
            'description': 'Prices move within normal ranges',
            'market_data': {
                'AAPL': {'price': 150.25, 'volatility': 0.15, 'volume': 1000000},
                'TSLA': {'price': 245.50, 'volatility': 0.25, 'volume': 2000000},
                'NVDA': {'price': 420.75, 'volatility': 0.20, 'volume': 1500000}
            }
        },
        # Scenario 2: Stop loss triggered
        {
            'name': 'Stop Loss Triggered',
            'description': 'AAPL price drops to trigger stop loss',
            'market_data': {
                'AAPL': {'price': 147.00, 'volatility': 0.15, 'volume': 1000000},  # -2.2%
                'TSLA': {'price': 245.50, 'volatility': 0.25, 'volume': 2000000},
                'NVDA': {'price': 420.75, 'volatility': 0.20, 'volume': 1500000}
            }
        },
        # Scenario 3: Take profit triggered
        {
            'name': 'Take Profit Triggered',
            'description': 'TSLA price rises to trigger take profit',
            'market_data': {
                'AAPL': {'price': 150.25, 'volatility': 0.15, 'volume': 1000000},
                'TSLA': {'price': 260.00, 'volatility': 0.25, 'volume': 2000000},  # +5.9%
                'NVDA': {'price': 420.75, 'volatility': 0.20, 'volume': 1500000}
            }
        },
        # Scenario 4: High volatility
        {
            'name': 'High Volatility',
            'description': 'High volatility triggers volatility limit exit',
            'market_data': {
                'AAPL': {'price': 150.25, 'volatility': 0.45, 'volume': 1000000},  # High vol
                'TSLA': {'price': 245.50, 'volatility': 0.25, 'volume': 2000000},
                'NVDA': {'price': 420.75, 'volatility': 0.20, 'volume': 1500000}
            }
        },
        # Scenario 5: Max holding period exceeded
        {
            'name': 'Max Holding Period',
            'description': 'Simulate time passing to test max holding period',
            'market_data': {
                'AAPL': {'price': 150.25, 'volatility': 0.15, 'volume': 1000000},
                'TSLA': {'price': 245.50, 'volatility': 0.25, 'volume': 2000000},
                'NVDA': {'price': 420.75, 'volatility': 0.20, 'volume': 1500000}
            }
        }
    ]
    return {'success': True, 'result': scenarios, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


async def demo_risk_controls():
    """Demonstrate comprehensive risk controls functionality."""
    print("🛡️ Risk Controls Demo")
    print("=" * 60)
    
    # Create execution agent with risk controls enabled
    config = {
        'name': 'risk_controlled_execution_agent',
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
            'risk_monitoring_enabled': True,
            'auto_exit_enabled': True,
            'risk_controls': {
                'stop_loss': {
                    'threshold_type': 'percentage',
                    'value': 0.02,
                    'atr_multiplier': 2.0,
                    'atr_period': 14
                },
                'take_profit': {
                    'threshold_type': 'percentage',
                    'value': 0.06,
                    'atr_multiplier': 3.0,
                    'atr_period': 14
                },
                'max_position_size': 0.2,
                'max_portfolio_risk': 0.05,
                'max_daily_loss': 0.02,
                'max_correlation': 0.7,
                'volatility_limit': 0.5,
                'trailing_stop': False
            }
        }
    }
    
    agent_config = AgentConfig(**config)
    execution_agent = ExecutionAgent(agent_config)
    
    print(f"✅ ExecutionAgent initialized with risk controls")
    print(f"📊 Default risk controls: {execution_agent.default_risk_controls.to_dict()}")
    
    # Create risk-aware signals
    signals = create_risk_aware_signals()
    print(f"\n📈 Created {len(signals)} signals with custom risk controls")
    
    for i, signal in enumerate(signals, 1):
        print(f"  Signal {i}: {signal.symbol} - {signal.strategy}")
        print(f"    Risk Controls: {signal.risk_controls.to_dict()}")
    
    # Execute initial trades
    print("\n🔄 Executing initial trades...")
    print("-" * 40)
    
    initial_market_data = {
        'AAPL': {'price': 150.25, 'volatility': 0.15, 'volume': 1000000},
        'TSLA': {'price': 245.50, 'volatility': 0.25, 'volume': 2000000},
        'NVDA': {'price': 420.75, 'volatility': 0.20, 'volume': 1500000}
    }
    
    result = await execution_agent.execute(
        signals=signals,
        market_data=initial_market_data,
        portfolio_update=True,
        risk_check=True
    )
    
    if result.success:
        print("✅ Initial trades executed successfully")
        print(f"  Portfolio status: {result.data['portfolio_state']}")
        print(f"  Risk metrics: {result.data['risk_metrics']}")
    else:
        print(f"❌ Initial trades failed: {result.message}")
        return
    
    # Test different market scenarios
    scenarios = create_market_scenarios()
    
    print(f"\n🧪 Testing {len(scenarios)} market scenarios...")
    print("=" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print("-" * 40)
        
        # Update market data and run risk monitoring
        result = await execution_agent.execute(
            signals=[],
            market_data=scenario['market_data'],
            portfolio_update=True,
            risk_check=True
        )
        
        if result.success:
            print("✅ Risk monitoring completed")
            
            # Show portfolio changes
            portfolio_state = result.data['portfolio_state']
            risk_metrics = result.data['risk_metrics']
            
            print(f"  Cash: ${portfolio_state['cash']:.2f}")
            print(f"  Equity: ${portfolio_state['equity']:.2f}")
            print(f"  Unrealized PnL: ${portfolio_state['unrealized_pnl']:.2f}")
            print(f"  Daily PnL: ${risk_metrics['daily_pnl']:.2f}")
            print(f"  Open positions: {len(portfolio_state['open_positions'])}")
            
            # Show position details
            for position in portfolio_state['open_positions']:
                print(f"    {position['symbol']}: {position['size']:.2f} shares "
                      f"@ ${position['entry_price']:.2f} "
                      f"(Unrealized: ${position['unrealized_pnl']:.2f})")
        else:
            print(f"❌ Risk monitoring failed: {result.message}")
    
    # Show exit events
    print(f"\n📋 Exit Events Summary")
    print("=" * 60)
    
    exit_events = execution_agent.get_exit_events()
    print(f"Total exit events: {len(exit_events)}")
    
    for event in exit_events:
        exit_data = event['exit_event']
        print(f"  {exit_data['symbol']}: {exit_data['exit_reason']} "
              f"@ ${exit_data['exit_price']:.2f} "
              f"(PnL: ${exit_data['pnl']:.2f})")
    
    # Show risk summary
    print(f"\n📊 Risk Management Summary")
    print("=" * 60)
    
    risk_summary = execution_agent.get_risk_summary()
    print(f"Total exits: {risk_summary['total_exits']}")
    print(f"Total PnL: ${risk_summary['total_pnl']:.2f}")
    print(f"Daily PnL: ${risk_summary['daily_pnl']:.2f}")
    
    print("\nExit reasons breakdown:")
    for reason, count in risk_summary['exit_reasons'].items():
        print(f"  {reason}: {count} exits")
    
    print("\nPnL by exit reason:")
    for reason, data in risk_summary['pnl_by_reason'].items():
        print(f"  {reason}: ${data['total']:.2f} (avg: ${data['avg']:.2f})")
    
    print(f"\n✅ Risk controls demo completed!")


async def demo_risk_threshold_types():
    """Demonstrate different risk threshold types."""
    print("\n🎯 Risk Threshold Types Demo")
    print("=" * 60)
    
    # Create agent
    config = {
        'name': 'threshold_demo_agent',
        'enabled': True,
        'custom_config': {
            'execution_mode': 'simulation',
            'max_positions': 3,
            'risk_monitoring_enabled': True
        }
    }
    
    agent_config = AgentConfig(**config)
    agent = ExecutionAgent(agent_config)
    
    # Test different threshold types
    threshold_types = [
        ("Percentage", RiskThresholdType.PERCENTAGE, 0.02),
        ("ATR-based", RiskThresholdType.ATR_BASED, 0.0, 2.0),
        ("Fixed", RiskThresholdType.FIXED, 5.0)
    ]
    
    for threshold_name, threshold_type, value, *args in threshold_types:
        print(f"\n📊 Testing {threshold_name} threshold:")
        
        # Create risk controls
        if threshold_type == RiskThresholdType.ATR_BASED:
            threshold = RiskThreshold(threshold_type, value, atr_multiplier=args[0])
        else:
            threshold = RiskThreshold(threshold_type, value)
        
        risk_controls = RiskControls(
            stop_loss=threshold,
            take_profit=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.06)
        )
        
        # Create signal
        signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="demo",
            confidence=0.8,
            entry_price=150.00,
            risk_controls=risk_controls
        )
        
        # Calculate stop loss price
        market_data = {'AAPL': {'price': 150.00, 'volatility': 0.15}}
        
        # Update price history for ATR calculation
        agent.price_history['AAPL'] = [145, 148, 152, 149, 151, 150, 153, 147, 150, 155, 148, 152, 149, 150]
        
        stop_loss_price = agent._calculate_stop_loss_price(
            type('Position', (), {
                'symbol': 'AAPL',
                'direction': TradeDirection.LONG,
                'entry_price': 150.00
            })(),
            risk_controls,
            market_data
        )
        
        print(f"  Entry price: ${signal.entry_price:.2f}")
        print(f"  Stop loss price: ${stop_loss_price:.2f}")
        print(f"  Stop loss distance: ${abs(stop_loss_price - signal.entry_price):.2f}")
    
    print(f"\n✅ Threshold types demo completed!")


async def demo_emergency_exits():
    """Demonstrate emergency exit scenarios."""
    print("\n🚨 Emergency Exits Demo")
    print("=" * 60)
    
    # Create agent
    config = {
        'name': 'emergency_demo_agent',
        'enabled': True,
        'custom_config': {
            'execution_mode': 'simulation',
            'max_positions': 3,
            'risk_monitoring_enabled': True,
            'risk_controls': {
                'max_daily_loss': 0.01,  # 1% daily loss limit
                'max_correlation': 0.5,  # Lower correlation limit
                'max_portfolio_risk': 0.03  # 3% portfolio risk
            }
        }
    }
    
    agent_config = AgentConfig(**config)
    agent = ExecutionAgent(agent_config)
    
    # Create positions
    signals = [
        TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="demo",
            confidence=0.8,
            entry_price=150.00
        ),
        TradeSignal(
            symbol="TSLA",
            direction=TradeDirection.LONG,
            strategy="demo",
            confidence=0.8,
            entry_price=245.00
        )
    ]
    
    # Execute trades
    market_data = {
        'AAPL': {'price': 150.00, 'volatility': 0.15},
        'TSLA': {'price': 245.00, 'volatility': 0.25}
    }
    
    result = await agent.execute(signals=signals, market_data=market_data)
    
    if result.success:
        print("✅ Initial positions opened")
        
        # Simulate daily loss limit breach
        print("\n📉 Simulating daily loss limit breach...")
        
        # Update daily PnL to breach limit
        agent.daily_pnl = -0.015  # -1.5% (above 1% limit)
        
        # Run risk monitoring
        result = await agent.execute(
            signals=[],
            market_data=market_data,
            risk_check=True
        )
        
        if result.success:
            print("✅ Emergency exit triggered")
            print(f"  Daily PnL: {agent.daily_pnl:.2%}")
            print(f"  Open positions: {len(agent.portfolio_manager.state.open_positions)}")
        else:
            print(f"❌ Emergency exit failed: {result.message}")
    
    print(f"\n✅ Emergency exits demo completed!")


async def main():
    """Run all risk controls demos."""
    print("🛡️ Comprehensive Risk Controls Demo")
    print("=" * 80)
    
    try:
        # Run main risk controls demo
        await demo_risk_controls()
        
        # Run threshold types demo
        await demo_risk_threshold_types()
        
        # Run emergency exits demo
        await demo_emergency_exits()
        
        print(f"\n🎉 All risk controls demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 