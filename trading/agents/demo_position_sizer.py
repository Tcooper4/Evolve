"""
Position Sizer Demo

This script demonstrates the PositionSizer functionality with different
sizing strategies and scenarios. Shows how position sizing adapts to
risk tolerance, confidence scores, and forecast certainty.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from trading.agents.execution_agent import (
    ExecutionAgent, TradeSignal, RiskControls, RiskThreshold, 
    RiskThresholdType, TradeDirection
)
from trading.agents.base_agent_interface import AgentConfig
from trading.portfolio.position_sizer import (
    PositionSizer, SizingStrategy, SizingParameters,
    MarketContext, SignalContext, PortfolioContext
)
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.memory.agent_memory import AgentMemory


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_demo_config() -> AgentConfig:
    """Create demo configuration for ExecutionAgent."""
    return AgentConfig(
        name="demo_execution_agent",
        agent_type="execution",
        enabled=True,
        custom_config={
            'execution_mode': 'simulation',
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
                'volatility_limit': 0.5
            },
            'position_sizing_config': {
                'default_strategy': 'confidence_based',
                'risk_per_trade': 0.02,
                'max_position_size': 0.2,
                'confidence_multiplier': 1.5,
                'volatility_multiplier': 1.0,
                'kelly_fraction': 0.25,
                'enable_risk_adjustment': True,
                'enable_correlation_adjustment': True,
                'enable_volatility_adjustment': True
            },
            'base_fee': 0.001,
            'min_fee': 1.0
        }
    )


def create_demo_market_data() -> Dict[str, Any]:
    """Create demo market data."""
    return {
        'AAPL': {
            'price': 150.0,
            'volatility': 0.18,
            'volume': 50000000,
            'liquidity_score': 0.9,
            'bid_ask_spread': 0.0005
        },
        'TSLA': {
            'price': 250.0,
            'volatility': 0.35,
            'volume': 30000000,
            'liquidity_score': 0.8,
            'bid_ask_spread': 0.001
        },
        'SPY': {
            'price': 450.0,
            'volatility': 0.15,
            'volume': 80000000,
            'liquidity_score': 0.95,
            'bid_ask_spread': 0.0002
        },
        'QQQ': {
            'price': 380.0,
            'volatility': 0.22,
            'volume': 60000000,
            'liquidity_score': 0.92,
            'bid_ask_spread': 0.0003
        }
    }


def create_demo_signals() -> List[TradeSignal]:
    """Create demo trade signals with different characteristics."""
    signals = []
    
    # High confidence, low volatility signal
    signals.append(TradeSignal(
        symbol='AAPL',
        direction=TradeDirection.LONG,
        strategy='momentum_strategy',
        confidence=0.85,
        entry_price=150.0,
        market_data={
            'forecast_certainty': 0.8,
            'signal_strength': 0.9,
            'sizing_strategy': 'confidence_based',
            'risk_per_trade': 0.025,
            'confidence_multiplier': 1.2
        }
    ))
    
    # Medium confidence, high volatility signal
    signals.append(TradeSignal(
        symbol='TSLA',
        direction=TradeDirection.LONG,
        strategy='mean_reversion_strategy',
        confidence=0.65,
        entry_price=250.0,
        market_data={
            'forecast_certainty': 0.6,
            'signal_strength': 0.7,
            'sizing_strategy': 'volatility_based',
            'risk_per_trade': 0.015,
            'volatility_multiplier': 0.8
        }
    ))
    
    # Low confidence, conservative signal
    signals.append(TradeSignal(
        symbol='SPY',
        direction=TradeDirection.SHORT,
        strategy='trend_following_strategy',
        confidence=0.45,
        entry_price=450.0,
        market_data={
            'forecast_certainty': 0.4,
            'signal_strength': 0.5,
            'sizing_strategy': 'fixed_percentage',
            'risk_per_trade': 0.01
        }
    ))
    
    # Kelly Criterion signal with good win rate
    signals.append(TradeSignal(
        symbol='QQQ',
        direction=TradeDirection.LONG,
        strategy='kelly_strategy',
        confidence=0.75,
        entry_price=380.0,
        market_data={
            'forecast_certainty': 0.7,
            'signal_strength': 0.8,
            'sizing_strategy': 'kelly_criterion',
            'risk_per_trade': 0.03,
            'kelly_fraction': 0.3
        }
    ))
    
    return signals


def create_demo_portfolio() -> PortfolioManager:
    """Create demo portfolio with initial capital."""
    portfolio = PortfolioManager(initial_capital=100000.0)
    
    # Add some existing positions to test correlation
    portfolio.open_position(
        symbol='MSFT',
        direction=TradeDirection.LONG,
        size=100,
        entry_price=300.0,
        strategy='existing_position'
    )
    
    portfolio.open_position(
        symbol='GOOGL',
        direction=TradeDirection.LONG,
        size=50,
        entry_price=2800.0,
        strategy='existing_position'
    )
    
    return portfolio


async def demo_position_sizing_strategies():
    """Demonstrate different position sizing strategies."""
    print("\n" + "="*60)
    print("POSITION SIZER DEMO - SIZING STRATEGIES")
    print("="*60)
    
    # Create position sizer
    position_sizer = PositionSizer()
    
    # Create demo contexts
    market_context = MarketContext(
        symbol='AAPL',
        current_price=150.0,
        volatility=0.18,
        volume=50000000,
        market_regime='normal',
        correlation=0.3,
        liquidity_score=0.9,
        bid_ask_spread=0.0005
    )
    
    signal_context = SignalContext(
        confidence=0.8,
        forecast_certainty=0.75,
        strategy_performance=0.12,
        win_rate=0.65,
        avg_win=0.03,
        avg_loss=-0.015,
        sharpe_ratio=1.2,
        max_drawdown=0.08,
        signal_strength=0.85
    )
    
    portfolio_context = PortfolioContext(
        total_capital=100000.0,
        available_capital=50000.0,
        current_exposure=0.5,
        open_positions=2,
        daily_pnl=0.02,
        portfolio_volatility=0.15
    )
    
    # Test different strategies
    strategies = [
        SizingStrategy.FIXED_PERCENTAGE,
        SizingStrategy.KELLY_CRITERION,
        SizingStrategy.VOLATILITY_BASED,
        SizingStrategy.CONFIDENCE_BASED,
        SizingStrategy.FORECAST_CERTAINTY,
        SizingStrategy.OPTIMAL_F,
        SizingStrategy.RISK_PARITY
    ]
    
    entry_price = 150.0
    stop_loss_price = 147.0
    
    print(f"\nEntry Price: ${entry_price:.2f}")
    print(f"Stop Loss: ${stop_loss_price:.2f}")
    print(f"Risk per Share: ${entry_price - stop_loss_price:.2f}")
    print(f"Portfolio Capital: ${portfolio_context.total_capital:,.2f}")
    
    print(f"\n{'Strategy':<20} {'Position Size':<15} {'Risk %':<10} {'Value':<12}")
    print("-" * 60)
    
    for strategy in strategies:
        sizing_params = SizingParameters(strategy=strategy)
        
        position_size, details = position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            market_context=market_context,
            signal_context=signal_context,
            portfolio_context=portfolio_context,
            sizing_params=sizing_params
        )
        
        risk_percentage = details['risk_percentage']
        position_value = details['position_value']
        
        print(f"{strategy.value:<20} {position_size:<15.4f} {risk_percentage:<10.2%} ${position_value:<11,.0f}")
    
    # Show sizing summary
    print(f"\nSizing Summary:")
    summary = position_sizer.get_sizing_summary()
    print(f"Total Decisions: {summary.get('total_sizing_decisions', 0)}")
    print(f"Average Position Size: {summary.get('average_position_size', 0):.4f}")
    print(f"Average Risk: {summary.get('average_risk_percentage', 0):.2%}")


async def demo_execution_agent_integration():
    """Demonstrate PositionSizer integration with ExecutionAgent."""
    print("\n" + "="*60)
    print("POSITION SIZER DEMO - EXECUTION AGENT INTEGRATION")
    print("="*60)
    
    # Create execution agent
    config = create_demo_config()
    portfolio = create_demo_portfolio()
    memory = AgentMemory()
    
    execution_agent = ExecutionAgent(config)
    execution_agent.portfolio_manager = portfolio
    execution_agent.memory = memory
    
    # Create demo signals
    signals = create_demo_signals()
    market_data = create_demo_market_data()
    
    print(f"\nPortfolio Status:")
    portfolio_status = portfolio.get_status()
    print(f"Total Capital: ${portfolio_status['equity']:,.2f}")
    print(f"Available Cash: ${portfolio_status['cash']:,.2f}")
    print(f"Current Exposure: {portfolio_status['total_exposure']:.2%}")
    print(f"Open Positions: {len(portfolio_status['open_positions'])}")
    
    print(f"\nProcessing {len(signals)} trade signals...")
    print(f"\n{'Symbol':<8} {'Strategy':<20} {'Confidence':<12} {'Size':<12} {'Risk %':<10} {'Value':<12}")
    print("-" * 75)
    
    for signal in signals:
        # Calculate position size
        position_size, sizing_details = execution_agent._calculate_position_size(
            signal, signal.entry_price, market_data
        )
        
        risk_percentage = sizing_details['risk_percentage']
        position_value = sizing_details['position_value']
        strategy_used = sizing_details['strategy']
        
        print(f"{signal.symbol:<8} {strategy_used:<20} {signal.confidence:<12.2f} "
              f"{position_size:<12.2f} {risk_percentage:<10.2%} ${position_value:<11,.0f}")
        
        # Log sizing decision
        execution_agent.memory.log_decision(
            agent_name=execution_agent.config.name,
            decision_type='position_sizing',
            details={
                'signal': signal.to_dict(),
                'sizing_details': sizing_details,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    # Show position sizer history
    print(f"\nPosition Sizer History:")
    history = execution_agent.position_sizer.get_sizing_history(limit=5)
    for entry in history:
        print(f"  {entry['timestamp']}: {entry['strategy']} -> {entry['final_size']:.4f} "
              f"(risk: {entry['risk_percentage']:.2%})")


async def demo_risk_scenarios():
    """Demonstrate position sizing under different risk scenarios."""
    print("\n" + "="*60)
    print("POSITION SIZER DEMO - RISK SCENARIOS")
    print("="*60)
    
    position_sizer = PositionSizer()
    
    # Base contexts
    base_market_context = MarketContext(
        symbol='AAPL',
        current_price=150.0,
        volatility=0.18,
        volume=50000000,
        market_regime='normal',
        correlation=0.3,
        liquidity_score=0.9,
        bid_ask_spread=0.0005
    )
    
    base_signal_context = SignalContext(
        confidence=0.7,
        forecast_certainty=0.6,
        strategy_performance=0.08,
        win_rate=0.6,
        avg_win=0.025,
        avg_loss=-0.015,
        sharpe_ratio=0.8,
        max_drawdown=0.12,
        signal_strength=0.7
    )
    
    base_portfolio_context = PortfolioContext(
        total_capital=100000.0,
        available_capital=50000.0,
        current_exposure=0.5,
        open_positions=2,
        daily_pnl=0.01,
        portfolio_volatility=0.15
    )
    
    entry_price = 150.0
    stop_loss_price = 147.0
    
    scenarios = [
        {
            'name': 'High Confidence',
            'signal_context': SignalContext(
                confidence=0.9,
                forecast_certainty=0.85,
                strategy_performance=0.15,
                win_rate=0.75,
                avg_win=0.035,
                avg_loss=-0.012,
                sharpe_ratio=1.5,
                max_drawdown=0.06,
                signal_strength=0.9
            )
        },
        {
            'name': 'Low Confidence',
            'signal_context': SignalContext(
                confidence=0.4,
                forecast_certainty=0.35,
                strategy_performance=0.02,
                win_rate=0.45,
                avg_win=0.02,
                avg_loss=-0.02,
                sharpe_ratio=0.3,
                max_drawdown=0.2,
                signal_strength=0.4
            )
        },
        {
            'name': 'High Volatility',
            'market_context': MarketContext(
                symbol='AAPL',
                current_price=150.0,
                volatility=0.35,
                volume=50000000,
                market_regime='volatile',
                correlation=0.3,
                liquidity_score=0.9,
                bid_ask_spread=0.001
            )
        },
        {
            'name': 'Portfolio Loss',
            'portfolio_context': PortfolioContext(
                total_capital=100000.0,
                available_capital=50000.0,
                current_exposure=0.5,
                open_positions=2,
                daily_pnl=-0.03,
                portfolio_volatility=0.15
            )
        },
        {
            'name': 'High Correlation',
            'market_context': MarketContext(
                symbol='AAPL',
                current_price=150.0,
                volatility=0.18,
                volume=50000000,
                market_regime='normal',
                correlation=0.85,
                liquidity_score=0.9,
                bid_ask_spread=0.0005
            )
        }
    ]
    
    print(f"\n{'Scenario':<15} {'Position Size':<15} {'Risk %':<10} {'Strategy':<20}")
    print("-" * 65)
    
    for scenario in scenarios:
        # Use scenario-specific contexts or defaults
        market_context = scenario.get('market_context', base_market_context)
        signal_context = scenario.get('signal_context', base_signal_context)
        portfolio_context = scenario.get('portfolio_context', base_portfolio_context)
        
        position_size, details = position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            market_context=market_context,
            signal_context=signal_context,
            portfolio_context=portfolio_context
        )
        
        risk_percentage = details['risk_percentage']
        strategy = details['strategy']
        
        print(f"{scenario['name']:<15} {position_size:<15.4f} {risk_percentage:<10.2%} {strategy:<20}")


async def demo_custom_sizing_parameters():
    """Demonstrate custom sizing parameters."""
    print("\n" + "="*60)
    print("POSITION SIZER DEMO - CUSTOM SIZING PARAMETERS")
    print("="*60)
    
    position_sizer = PositionSizer()
    
    # Base contexts
    market_context = MarketContext(
        symbol='AAPL',
        current_price=150.0,
        volatility=0.18,
        volume=50000000,
        market_regime='normal',
        correlation=0.3,
        liquidity_score=0.9,
        bid_ask_spread=0.0005
    )
    
    signal_context = SignalContext(
        confidence=0.7,
        forecast_certainty=0.6,
        strategy_performance=0.08,
        win_rate=0.6,
        avg_win=0.025,
        avg_loss=-0.015,
        sharpe_ratio=0.8,
        max_drawdown=0.12,
        signal_strength=0.7
    )
    
    portfolio_context = PortfolioContext(
        total_capital=100000.0,
        available_capital=50000.0,
        current_exposure=0.5,
        open_positions=2,
        daily_pnl=0.01,
        portfolio_volatility=0.15
    )
    
    entry_price = 150.0
    stop_loss_price = 147.0
    
    # Custom parameters
    custom_params = [
        {
            'name': 'Conservative',
            'params': SizingParameters(
                strategy=SizingStrategy.FIXED_PERCENTAGE,
                risk_per_trade=0.01,
                max_position_size=0.1,
                confidence_multiplier=0.8
            )
        },
        {
            'name': 'Aggressive',
            'params': SizingParameters(
                strategy=SizingStrategy.CONFIDENCE_BASED,
                risk_per_trade=0.04,
                max_position_size=0.3,
                confidence_multiplier=2.0
            )
        },
        {
            'name': 'Kelly Conservative',
            'params': SizingParameters(
                strategy=SizingStrategy.KELLY_CRITERION,
                risk_per_trade=0.02,
                kelly_fraction=0.15,
                max_position_size=0.15
            )
        },
        {
            'name': 'Volatility Adjusted',
            'params': SizingParameters(
                strategy=SizingStrategy.VOLATILITY_BASED,
                risk_per_trade=0.025,
                volatility_multiplier=1.5,
                max_position_size=0.25
            )
        }
    ]
    
    print(f"\n{'Parameter Set':<20} {'Position Size':<15} {'Risk %':<10} {'Strategy':<20}")
    print("-" * 70)
    
    for param_set in custom_params:
        position_size, details = position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            market_context=market_context,
            signal_context=signal_context,
            portfolio_context=portfolio_context,
            sizing_params=param_set['params']
        )
        
        risk_percentage = details['risk_percentage']
        strategy = details['strategy']
        
        print(f"{param_set['name']:<20} {position_size:<15.4f} {risk_percentage:<10.2%} {strategy:<20}")


async def main():
    """Run the complete position sizer demo."""
    setup_logging()
    
    print("POSITION SIZER DEMONSTRATION")
    print("="*60)
    print("This demo showcases the PositionSizer functionality with:")
    print("- Multiple sizing strategies (Fixed %, Kelly, Volatility-based, etc.)")
    print("- Dynamic sizing based on confidence and forecast certainty")
    print("- Risk-adjusted position sizing")
    print("- Integration with ExecutionAgent")
    print("- Different market and portfolio scenarios")
    
    try:
        # Demo 1: Different sizing strategies
        await demo_position_sizing_strategies()
        
        # Demo 2: Execution agent integration
        await demo_execution_agent_integration()
        
        # Demo 3: Risk scenarios
        await demo_risk_scenarios()
        
        # Demo 4: Custom sizing parameters
        await demo_custom_sizing_parameters()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Multiple sizing strategies (9 different approaches)")
        print("✓ Dynamic sizing based on confidence and certainty")
        print("✓ Risk-adjusted position sizing")
        print("✓ Market condition adjustments")
        print("✓ Portfolio correlation handling")
        print("✓ Integration with ExecutionAgent")
        print("✓ Comprehensive logging and history tracking")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        logging.error(f"Demo error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main()) 