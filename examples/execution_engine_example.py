"""
Execution Engine Example

This example demonstrates the trade execution engine with realistic scenarios:
- Market and limit order execution
- Risk management and position tracking
- Performance monitoring and analytics
- Integration with different brokers
- Real-time order book management
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Import execution modules
from execution.execution_agent import (
    ExecutionAgent,
    ExecutionMode,
    OrderType,
    OrderSide,
    OrderStatus,
    create_execution_agent
)
from execution.broker_adapter import (
    BrokerAdapter,
    BrokerType,
    create_broker_adapter,
    test_broker_connection
)

# Import utility functions
from utils.common_helpers import safe_json_save
from utils.cache_utils import cache_result


class ExecutionEngineExample:
    """
    Comprehensive execution engine example
    """
    
    def __init__(self):
        """Initialize the example"""
        # Create execution agent
        self.agent = create_execution_agent()
        
        # Trading parameters
        self.tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
        self.initial_capital = 100000.0
        
        # Results storage
        self.trade_history = []
        self.performance_metrics = []
        self.risk_analysis = {}
        
        # Trading strategy parameters
        self.position_sizes = {
            "AAPL": 0.2,  # 20% of portfolio
            "TSLA": 0.15, # 15% of portfolio
            "NVDA": 0.15, # 15% of portfolio
            "MSFT": 0.2,  # 20% of portfolio
            "GOOGL": 0.2  # 20% of portfolio
        }
        
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
    
    async def run_basic_execution_demo(self):
        """Run basic execution demonstration"""
        print("="*60)
        print("BASIC EXECUTION DEMONSTRATION")
        print("="*60)
        
        # Start execution agent
        await self.agent.start()
        
        print(f"Execution Agent started in {self.agent.execution_mode.value} mode")
        
        # Submit various order types
        orders = [
            {
                'ticker': 'AAPL',
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'quantity': 100,
                'description': 'Market Buy AAPL'
            },
            {
                'ticker': 'TSLA',
                'side': OrderSide.BUY,
                'order_type': OrderType.LIMIT,
                'quantity': 50,
                'price': 200.0,
                'description': 'Limit Buy TSLA @ $200'
            },
            {
                'ticker': 'NVDA',
                'side': OrderSide.SELL,
                'order_type': OrderType.STOP,
                'quantity': 75,
                'stop_price': 400.0,
                'description': 'Stop Sell NVDA @ $400'
            }
        ]
        
        order_ids = []
        
        for order_spec in orders:
            print(f"\nSubmitting: {order_spec['description']}")
            
            order_id = await self.agent.submit_order(
                ticker=order_spec['ticker'],
                side=order_spec['side'],
                order_type=order_spec['order_type'],
                quantity=order_spec['quantity'],
                price=order_spec.get('price'),
                stop_price=order_spec.get('stop_price')
            )
            
            order_ids.append(order_id)
            print(f"  Order ID: {order_id}")
        
        # Wait for execution
        print("\nWaiting for order execution...")
        await asyncio.sleep(2)
        
        # Check order statuses
        print("\nOrder Statuses:")
        for order_id in order_ids:
            execution = self.agent.get_order_status(order_id)
            if execution:
                print(f"  {order_id}: {execution.status.value} @ ${execution.average_price:.2f}")
                print(f"    Commission: ${execution.commission:.2f}")
                print(f"    Metadata: {execution.metadata}")
        
        # Check positions
        print("\nCurrent Positions:")
        positions = self.agent.get_all_positions()
        for ticker, position in positions.items():
            print(f"  {ticker}: {position.quantity} shares @ ${position.average_price:.2f}")
            print(f"    Market Value: ${position.market_value:.2f}")
            print(f"    Unrealized P&L: ${position.unrealized_pnl:.2f}")
        
        # Stop agent
        await self.agent.stop()
        print("\nExecution Agent stopped")
    
    async def run_risk_management_demo(self):
        """Run risk management demonstration"""
        print("\n" + "="*60)
        print("RISK MANAGEMENT DEMONSTRATION")
        print("="*60)
        
        # Start agent with strict risk limits
        self.agent.max_order_size = 5000  # $5K max order
        self.agent.max_daily_trades = 10   # 10 trades per day
        self.agent.max_position_size = 0.1 # 10% max position
        
        await self.agent.start()
        
        print("Risk Limits:")
        print(f"  Max Order Size: ${self.agent.max_order_size}")
        print(f"  Max Daily Trades: {self.agent.max_daily_trades}")
        print(f"  Max Position Size: {self.agent.max_position_size:.1%}")
        
        # Test risk limit violations
        print("\nTesting Risk Limits:")
        
        # Test order size limit
        try:
            large_order_id = await self.agent.submit_order(
                ticker="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=10000  # Very large order
            )
            print("  Large order submitted successfully")
        except ValueError as e:
            print(f"  Large order rejected: {e}")
        
        # Test daily trade limit
        print("\nSubmitting multiple orders to test daily limit:")
        for i in range(15):
            try:
                order_id = await self.agent.submit_order(
                    ticker="AAPL",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=10
                )
                print(f"  Order {i+1}: {order_id}")
            except ValueError as e:
                print(f"  Order {i+1} rejected: {e}")
                break
        
        await asyncio.sleep(1)
        
        # Check performance metrics
        metrics = self.agent.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        print(f"  Daily Trades: {metrics['daily_trades']}")
        print(f"  Daily Volume: ${metrics['daily_volume']:.2f}")
        print(f"  Daily Commission: ${metrics['daily_commission']:.2f}")
        
        await self.agent.stop()
    
    async def run_portfolio_rebalancing_demo(self):
        """Run portfolio rebalancing demonstration"""
        print("\n" + "="*60)
        print("PORTFOLIO REBALANCING DEMONSTRATION")
        print("="*60)
        
        await self.agent.start()
        
        # Initial portfolio allocation
        print("Initial Portfolio Allocation:")
        total_value = self.initial_capital
        
        for ticker, target_pct in self.position_sizes.items():
            target_value = total_value * target_pct
            market_data = self.agent.get_market_data(ticker)
            target_quantity = int(target_value / market_data.last)
            
            print(f"  {ticker}: {target_quantity} shares (${target_value:.2f})")
            
            # Submit order
            order_id = await self.agent.submit_order(
                ticker=ticker,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=target_quantity
            )
        
        await asyncio.sleep(2)
        
        # Check initial positions
        print("\nInitial Positions:")
        positions = self.agent.get_all_positions()
        total_portfolio_value = 0
        
        for ticker, position in positions.items():
            print(f"  {ticker}: {position.quantity} shares @ ${position.average_price:.2f}")
            print(f"    Value: ${position.market_value:.2f}")
            total_portfolio_value += position.market_value
        
        print(f"  Total Portfolio Value: ${total_portfolio_value:.2f}")
        
        # Simulate price changes and rebalance
        print("\nSimulating price changes and rebalancing...")
        
        # Simulate some price movements
        for ticker in self.tickers:
            if ticker in positions:
                position = positions[ticker]
                target_pct = self.position_sizes[ticker]
                target_value = total_portfolio_value * target_pct
                current_value = position.market_value
                
                # Check if rebalancing needed
                if abs(current_value - target_value) > total_portfolio_value * 0.02:  # 2% threshold
                    if current_value > target_value:
                        # Need to sell
                        excess_value = current_value - target_value
                        market_data = self.agent.get_market_data(ticker)
                        sell_quantity = int(excess_value / market_data.last)
                        
                        if sell_quantity > 0:
                            print(f"  Rebalancing {ticker}: Selling {sell_quantity} shares")
                            await self.agent.submit_order(
                                ticker=ticker,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=sell_quantity
                            )
                    else:
                        # Need to buy
                        deficit_value = target_value - current_value
                        market_data = self.agent.get_market_data(ticker)
                        buy_quantity = int(deficit_value / market_data.last)
                        
                        if buy_quantity > 0:
                            print(f"  Rebalancing {ticker}: Buying {buy_quantity} shares")
                            await self.agent.submit_order(
                                ticker=ticker,
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=buy_quantity
                            )
        
        await asyncio.sleep(2)
        
        # Check final positions
        print("\nFinal Positions:")
        final_positions = self.agent.get_all_positions()
        final_total_value = 0
        
        for ticker, position in final_positions.items():
            print(f"  {ticker}: {position.quantity} shares @ ${position.average_price:.2f}")
            print(f"    Value: ${position.market_value:.2f}")
            final_total_value += position.market_value
        
        print(f"  Final Portfolio Value: ${final_total_value:.2f}")
        print(f"  Portfolio Return: {((final_total_value - total_portfolio_value) / total_portfolio_value):.2%}")
        
        await self.agent.stop()
    
    async def run_broker_adapter_demo(self):
        """Run broker adapter demonstration"""
        print("\n" + "="*60)
        print("BROKER ADAPTER DEMONSTRATION")
        print("="*60)
        
        # Test different broker types
        broker_types = ["simulation", "alpaca", "ibkr", "polygon"]
        
        for broker_type in broker_types:
            print(f"\nTesting {broker_type.upper()} broker:")
            
            try:
                # Test connection
                connected = await test_broker_connection(broker_type)
                print(f"  Connection: {'âœ… Connected' if connected else 'âŒ Failed'}")
                
                if connected:
                    # Create adapter
                    adapter = create_broker_adapter(broker_type)
                    await adapter.connect()
                    
                    # Test market data
                    try:
                        market_data = await adapter.get_market_data("AAPL")
                        print(f"  Market Data: âœ… {market_data.ticker} @ ${market_data.last:.2f}")
                    except Exception as e:
                        print(f"  Market Data: âŒ {e}")
                    
                    # Test order submission (if supported)
                    if broker_type in ["simulation", "alpaca", "ibkr"]:
                        try:
                            from execution.broker_adapter import OrderRequest
                            order = OrderRequest(
                                order_id="test_order",
                                ticker="AAPL",
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=1
                            )
                            execution = await adapter.submit_order(order)
                            print(f"  Order Submission: âœ… {execution.status.value}")
                        except Exception as e:
                            print(f"  Order Submission: âŒ {e}")
                    
                    await adapter.disconnect()
                
            except Exception as e:
                print(f"  Error: {e}")
    
    async def run_performance_analysis_demo(self):
        """Run performance analysis demonstration"""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS DEMONSTRATION")
        print("="*60)
        
        await self.agent.start()
        
        # Execute a series of trades
        print("Executing trade series...")
        
        trades = [
            {"ticker": "AAPL", "side": OrderSide.BUY, "quantity": 50, "price": 150.0},
            {"ticker": "TSLA", "side": OrderSide.BUY, "quantity": 25, "price": 200.0},
            {"ticker": "AAPL", "side": OrderSide.SELL, "quantity": 30, "price": 155.0},
            {"ticker": "NVDA", "side": OrderSide.BUY, "quantity": 40, "price": 300.0},
            {"ticker": "TSLA", "side": OrderSide.SELL, "quantity": 15, "price": 210.0}
        ]
        
        for i, trade in enumerate(trades):
            print(f"  Trade {i+1}: {trade['side'].value} {trade['quantity']} {trade['ticker']}")
            
            order_id = await self.agent.submit_order(
                ticker=trade['ticker'],
                side=trade['side'],
                order_type=OrderType.LIMIT,
                quantity=trade['quantity'],
                price=trade['price']
            )
            
            # Store trade info
            self.trade_history.append({
                'trade_id': i + 1,
                'order_id': order_id,
                'ticker': trade['ticker'],
                'side': trade['side'].value,
                'quantity': trade['quantity'],
                'price': trade['price'],
                'timestamp': datetime.now().isoformat()
            })
        
        await asyncio.sleep(2)
        
        # Analyze performance
        print("\nPerformance Analysis:")
        
        # Get all executions
        executions = self.agent.get_execution_history()
        
        # Calculate metrics
        total_trades = len(executions)
        total_volume = sum(e.executed_quantity * e.average_price for e in executions)
        total_commission = sum(e.commission for e in executions)
        
        # Calculate P&L
        buy_trades = [e for e in executions if e.side == OrderSide.BUY]
        sell_trades = [e for e in executions if e.side == OrderSide.SELL]
        
        buy_value = sum(e.executed_quantity * e.average_price for e in buy_trades)
        sell_value = sum(e.executed_quantity * e.average_price for e in sell_trades)
        
        gross_pnl = sell_value - buy_value
        net_pnl = gross_pnl - total_commission
        
        print(f"  Total Trades: {total_trades}")
        print(f"  Total Volume: ${total_volume:.2f}")
        print(f"  Total Commission: ${total_commission:.2f}")
        print(f"  Gross P&L: ${gross_pnl:.2f}")
        print(f"  Net P&L: ${net_pnl:.2f}")
        print(f"  Return on Volume: {(net_pnl / total_volume):.2%}")
        
        # Position analysis
        print("\nPosition Analysis:")
        positions = self.agent.get_all_positions()
        
        for ticker, position in positions.items():
            if position.quantity != 0:
                print(f"  {ticker}: {position.quantity} shares")
                print(f"    Average Price: ${position.average_price:.2f}")
                print(f"    Market Value: ${position.market_value:.2f}")
                print(f"    Unrealized P&L: ${position.unrealized_pnl:.2f}")
                print(f"    Realized P&L: ${position.realized_pnl:.2f}")
        
        # Risk metrics
        print("\nRisk Metrics:")
        metrics = self.agent.get_performance_metrics()
        
        print(f"  Daily Trades: {metrics['daily_trades']}")
        print(f"  Daily Volume: ${metrics['daily_volume']:.2f}")
        print(f"  Total Positions: {metrics['total_positions']}")
        print(f"  Total P&L: ${metrics['total_pnl']:.2f}")
        
        await self.agent.stop()
    
    async def run_complete_demo(self):
        """Run complete execution engine demonstration"""
        print("Execution Engine Complete Demonstration")
        print("="*60)
        
        # Run all demos
        await self.run_basic_execution_demo()
        await self.run_risk_management_demo()
        await self.run_portfolio_rebalancing_demo()
        await self.run_broker_adapter_demo()
        await self.run_performance_analysis_demo()
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
    
    def generate_summary_report(self):
        """Generate summary report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'execution_mode': self.agent.execution_mode.value,
            'trade_history': self.trade_history,
            'performance_metrics': self.agent.get_performance_metrics(),
            'risk_analysis': self.risk_analysis,
            'summary': {
                'total_trades': len(self.trade_history),
                'demo_completed': True,
                'execution_successful': True
            }
        }
        
        # Save report
        safe_json_save('execution_engine_demo_report.json', report)
        print("\nSummary report saved as 'execution_engine_demo_report.json'")


def main():
    """Main function to run the execution engine example"""
    print("Execution Engine Example")
    print("="*60)
    
    # Create example instance
    example = ExecutionEngineExample()
    
    # Run complete demo
    asyncio.run(example.run_complete_demo())
    
    return example


if __name__ == "__main__":
    main()
