"""
Trade Execution Agent

This module provides realistic trade execution simulation and live trading capabilities:
- Market/limit order execution with spread, slippage, and delay simulation
- Comprehensive order book logging and trade tracking
- Integration with broker adapters for live trading
- Risk management and position tracking
- Performance monitoring and analytics
"""

import os
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
import uuid
from decimal import Decimal, ROUND_HALF_UP

# Local imports
from utils.cache_utils import cache_result
from utils.common_helpers import safe_json_save, load_config
from .broker_adapter import BrokerAdapter, OrderType, OrderSide, OrderStatus


class ExecutionMode(Enum):
    """Execution modes"""
    SIMULATION = "simulation"
    LIVE = "live"
    PAPER = "paper"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderRequest:
    """Order request structure"""
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    client_order_id: Optional[str] = None
    timestamp: str = None


@dataclass
class OrderExecution:
    """Order execution result"""
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    executed_quantity: float
    average_price: float
    commission: float
    timestamp: str
    status: OrderStatus
    fills: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class MarketData:
    """Market data snapshot"""
    ticker: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: str
    spread: float
    volatility: float


@dataclass
class Position:
    """Position information"""
    ticker: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: str


class ExecutionAgent:
    """
    Trade execution agent with simulation and live trading capabilities
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.execution_config = self.config.get('execution', {})
        
        # Execution mode
        self.execution_mode = ExecutionMode(self.execution_config.get('mode', 'simulation'))
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create log directory
        self.log_dir = Path("logs/execution")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Order book and trade tracking
        self.order_book: Dict[str, OrderRequest] = {}
        self.executed_orders: Dict[str, OrderExecution] = {}
        self.positions: Dict[str, Position] = {}
        
        # Market data cache
        self.market_data_cache: Dict[str, MarketData] = {}
        self.market_data_history: Dict[str, List[MarketData]] = {}
        
        # Execution parameters
        self.spread_multiplier = self.execution_config.get('spread_multiplier', 1.0)
        self.slippage_basis_points = self.execution_config.get('slippage_bps', 5)
        self.execution_delay_ms = self.execution_config.get('execution_delay_ms', 100)
        self.commission_rate = self.execution_config.get('commission_rate', 0.001)
        self.min_commission = self.execution_config.get('min_commission', 1.0)
        
        # Risk management
        self.max_position_size = self.execution_config.get('max_position_size', 0.1)
        self.max_order_size = self.execution_config.get('max_order_size', 10000)
        self.max_daily_trades = self.execution_config.get('max_daily_trades', 100)
        
        # Performance tracking
        self.daily_trades = 0
        self.daily_volume = 0.0
        self.daily_commission = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Broker adapter
        self.broker_adapter = None
        self._initialize_broker_adapter()
        
        # Threading
        self.order_queue = asyncio.Queue()
        self.execution_thread = None
        self.is_running = False
        
        # Market simulation parameters
        self.market_volatility = self.execution_config.get('market_volatility', 0.02)
        self.price_impact_factor = self.execution_config.get('price_impact_factor', 0.0001)
        
        # Load historical data for simulation
        self._load_historical_data()
    
    def _initialize_broker_adapter(self):
        """Initialize broker adapter for live trading"""
        if self.execution_mode in [ExecutionMode.LIVE, ExecutionMode.PAPER]:
            try:
                from .broker_adapter import BrokerAdapter
                broker_config = self.execution_config.get('broker', {})
                self.broker_adapter = BrokerAdapter(
                    broker_type=broker_config.get('type', 'alpaca'),
                    config=broker_config
                )
                self.logger.info(f"Broker adapter initialized: {broker_config.get('type', 'alpaca')}")
            except Exception as e:
                self.logger.error(f"Failed to initialize broker adapter: {e}")
                self.execution_mode = ExecutionMode.SIMULATION
                self.logger.info("Falling back to simulation mode")
    
    def _load_historical_data(self):
        """Load historical market data for simulation"""
        try:
            # This would typically load from your data system
            # For now, create sample historical data
            dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
            
            for ticker in ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']:
                # Generate realistic price data
                base_price = 100 + np.random.randint(0, 900)
                returns = np.random.normal(0.001, self.market_volatility, len(dates))
                prices = [base_price]
                
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                # Create market data history
                market_data_list = []
                for i, date in enumerate(dates):
                    price = prices[i]
                    spread = price * 0.001  # 0.1% spread
                    
                    market_data = MarketData(
                        ticker=ticker,
                        bid=price - spread/2,
                        ask=price + spread/2,
                        last=price,
                        volume=np.random.randint(1000000, 10000000),
                        timestamp=date.isoformat(),
                        spread=spread,
                        volatility=self.market_volatility
                    )
                    market_data_list.append(market_data)
                
                self.market_data_history[ticker] = market_data_list
                
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
    
    def _get_current_market_data(self, ticker: str) -> MarketData:
        """Get current market data for a ticker"""
        if self.execution_mode == ExecutionMode.SIMULATION:
            # Use simulated market data
            if ticker in self.market_data_history:
                # Get most recent data and add some noise
                latest_data = self.market_data_history[ticker][-1]
                
                # Add realistic price movement
                price_change = np.random.normal(0, latest_data.volatility)
                new_price = latest_data.last * (1 + price_change)
                spread = new_price * 0.001
                
                return MarketData(
                    ticker=ticker,
                    bid=new_price - spread/2,
                    ask=new_price + spread/2,
                    last=new_price,
                    volume=np.random.randint(1000000, 10000000),
                    timestamp=datetime.now().isoformat(),
                    spread=spread,
                    volatility=latest_data.volatility
                )
            else:
                # Create default market data
                return MarketData(
                    ticker=ticker,
                    bid=100.0,
                    ask=100.1,
                    last=100.05,
                    volume=1000000,
                    timestamp=datetime.now().isoformat(),
                    spread=0.1,
                    volatility=0.02
                )
        else:
            # Get live market data from broker
            if self.broker_adapter:
                return self.broker_adapter.get_market_data(ticker)
            else:
                raise Exception("No broker adapter available for live market data")
    
    def _simulate_order_execution(self, order: OrderRequest) -> OrderExecution:
        """Simulate realistic order execution"""
        # Get current market data
        market_data = self._get_current_market_data(order.ticker)
        
        # Calculate execution parameters
        if order.order_type == OrderType.MARKET:
            # Market order execution
            if order.side == OrderSide.BUY:
                execution_price = market_data.ask
            else:
                execution_price = market_data.bid
        else:
            # Limit order execution
            execution_price = order.price or market_data.last
        
        # Apply slippage
        slippage_bps = self.slippage_basis_points / 10000
        if order.side == OrderSide.BUY:
            execution_price *= (1 + slippage_bps)
        else:
            execution_price *= (1 - slippage_bps)
        
        # Apply price impact for large orders
        order_value = order.quantity * execution_price
        price_impact = (order_value / 1000000) * self.price_impact_factor  # Impact per $1M
        
        if order.side == OrderSide.BUY:
            execution_price *= (1 + price_impact)
        else:
            execution_price *= (1 - price_impact)
        
        # Simulate execution delay
        time.sleep(self.execution_delay_ms / 1000)
        
        # Calculate commission
        commission = max(
            order_value * self.commission_rate,
            self.min_commission
        )
        
        # Create fills
        fills = [{
            'fill_id': str(uuid.uuid4()),
            'quantity': order.quantity,
            'price': execution_price,
            'timestamp': datetime.now().isoformat(),
            'commission': commission
        }]
        
        # Create execution result
        execution = OrderExecution(
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price or execution_price,
            executed_quantity=order.quantity,
            average_price=execution_price,
            commission=commission,
            timestamp=datetime.now().isoformat(),
            status=OrderStatus.FILLED,
            fills=fills,
            metadata={
                'slippage_bps': slippage_bps * 10000,
                'price_impact': price_impact,
                'market_spread': market_data.spread,
                'execution_delay_ms': self.execution_delay_ms
            }
        )
        
        return execution
    
    def _update_position(self, execution: OrderExecution):
        """Update position after order execution"""
        ticker = execution.ticker
        
        if ticker not in self.positions:
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=0.0,
                average_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        position = self.positions[ticker]
        
        if execution.side == OrderSide.BUY:
            # Buying
            if position.quantity >= 0:
                # Adding to long position
                total_cost = position.quantity * position.average_price + execution.executed_quantity * execution.average_price
                position.quantity += execution.executed_quantity
                position.average_price = total_cost / position.quantity
            else:
                # Covering short position
                if abs(position.quantity) <= execution.executed_quantity:
                    # Full cover
                    realized_pnl = (position.average_price - execution.average_price) * abs(position.quantity)
                    position.realized_pnl += realized_pnl
                    position.quantity += execution.executed_quantity
                    if position.quantity > 0:
                        position.average_price = execution.average_price
                    else:
                        position.average_price = 0.0
                else:
                    # Partial cover
                    realized_pnl = (position.average_price - execution.average_price) * execution.executed_quantity
                    position.realized_pnl += realized_pnl
                    position.quantity += execution.executed_quantity
        else:
            # Selling
            if position.quantity <= 0:
                # Adding to short position
                total_cost = abs(position.quantity) * position.average_price + execution.executed_quantity * execution.average_price
                position.quantity -= execution.executed_quantity
                position.average_price = total_cost / abs(position.quantity)
            else:
                # Reducing long position
                if position.quantity <= execution.executed_quantity:
                    # Full sell
                    realized_pnl = (execution.average_price - position.average_price) * position.quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= execution.executed_quantity
                    if position.quantity < 0:
                        position.average_price = execution.average_price
                    else:
                        position.average_price = 0.0
                else:
                    # Partial sell
                    realized_pnl = (execution.average_price - position.average_price) * execution.executed_quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= execution.executed_quantity
        
        # Update market value and unrealized PnL
        market_data = self._get_current_market_data(ticker)
        position.market_value = position.quantity * market_data.last
        position.unrealized_pnl = position.market_value - (position.quantity * position.average_price)
        position.timestamp = datetime.now().isoformat()
    
    def _log_order(self, order: OrderRequest):
        """Log order to order book file"""
        try:
            order_book_file = self.log_dir / "order_book.json"
            
            order_data = asdict(order)
            order_data['timestamp'] = datetime.now().isoformat()
            
            # Load existing orders
            orders = []
            if order_book_file.exists():
                with open(order_book_file, 'r') as f:
                    orders = json.load(f)
            
            # Add new order
            orders.append(order_data)
            
            # Save updated order book
            safe_json_save(str(order_book_file), orders)
            
        except Exception as e:
            self.logger.error(f"Failed to log order: {e}")
    
    def _log_execution(self, execution: OrderExecution):
        """Log execution to trade log file"""
        try:
            trade_log_file = self.log_dir / "trade_log.json"
            
            execution_data = asdict(execution)
            execution_data['timestamp'] = datetime.now().isoformat()
            
            # Load existing executions
            executions = []
            if trade_log_file.exists():
                with open(trade_log_file, 'r') as f:
                    executions = json.load(f)
            
            # Add new execution
            executions.append(execution_data)
            
            # Save updated trade log
            safe_json_save(str(trade_log_file), executions)
            
        except Exception as e:
            self.logger.error(f"Failed to log execution: {e}")
    
    def _check_risk_limits(self, order: OrderRequest) -> Tuple[bool, str]:
        """Check risk limits before order execution"""
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit exceeded: {self.max_daily_trades}"
        
        # Check order size limit
        market_data = self._get_current_market_data(order.ticker)
        order_value = order.quantity * market_data.last
        
        if order_value > self.max_order_size:
            return False, f"Order size limit exceeded: ${order_value:.2f} > ${self.max_order_size:.2f}"
        
        # Check position size limit
        current_position = self.positions.get(order.ticker, Position(
            ticker=order.ticker, quantity=0, average_price=0, market_value=0,
            unrealized_pnl=0, realized_pnl=0, timestamp=datetime.now().isoformat()
        ))
        
        new_quantity = current_position.quantity
        if order.side == OrderSide.BUY:
            new_quantity += order.quantity
        else:
            new_quantity -= order.quantity
        
        new_position_value = abs(new_quantity) * market_data.last
        total_portfolio_value = sum(pos.market_value for pos in self.positions.values()) + order_value
        
        if total_portfolio_value > 0 and new_position_value / total_portfolio_value > self.max_position_size:
            return False, f"Position size limit exceeded: {new_position_value/total_portfolio_value:.2%} > {self.max_position_size:.2%}"
        
        return True, "Risk checks passed"
    
    async def submit_order(self, 
                          ticker: str,
                          side: OrderSide,
                          order_type: OrderType,
                          quantity: float,
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          time_in_force: str = "day",
                          client_order_id: Optional[str] = None) -> str:
        """
        Submit an order for execution
        """
        # Generate order ID
        order_id = client_order_id or str(uuid.uuid4())
        
        # Create order request
        order = OrderRequest(
            order_id=order_id,
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Check risk limits
        risk_ok, risk_message = self._check_risk_limits(order)
        if not risk_ok:
            self.logger.warning(f"Risk limit violation: {risk_message}")
            raise ValueError(f"Risk limit violation: {risk_message}")
        
        # Log order
        self._log_order(order)
        
        # Add to order book
        self.order_book[order_id] = order
        
        # Submit for execution
        await self.order_queue.put(order)
        
        self.logger.info(f"Order submitted: {order_id} - {side.value} {quantity} {ticker} @ {price or 'MARKET'}")
        
        return order_id
    
    async def _execute_order(self, order: OrderRequest) -> OrderExecution:
        """Execute an order"""
        try:
            if self.execution_mode == ExecutionMode.SIMULATION:
                # Simulate execution
                execution = self._simulate_order_execution(order)
            else:
                # Live execution through broker
                if self.broker_adapter:
                    execution = await self.broker_adapter.submit_order(order)
                else:
                    raise Exception("No broker adapter available for live execution")
            
            # Update position
            self._update_position(execution)
            
            # Log execution
            self._log_execution(execution)
            
            # Update performance metrics
            self.daily_trades += 1
            self.daily_volume += execution.executed_quantity * execution.average_price
            self.daily_commission += execution.commission
            
            # Store execution
            self.executed_orders[order.order_id] = execution
            
            self.logger.info(f"Order executed: {order.order_id} - {execution.executed_quantity} @ {execution.average_price:.2f}")
            
            return execution
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {order.order_id} - {e}")
            
            # Create failed execution record
            failed_execution = OrderExecution(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price or 0.0,
                executed_quantity=0.0,
                average_price=0.0,
                commission=0.0,
                timestamp=datetime.now().isoformat(),
                status=OrderStatus.REJECTED,
                fills=[],
                metadata={'error': str(e)}
            )
            
            return failed_execution
    
    async def _execution_worker(self):
        """Background worker for order execution"""
        while self.is_running:
            try:
                # Get order from queue
                order = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                
                # Execute order
                execution = await self._execute_order(order)
                
                # Mark task as done
                self.order_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Execution worker error: {e}")
    
    async def start(self):
        """Start the execution agent"""
        if self.is_running:
            return
        
        self.is_running = True
        self.execution_thread = asyncio.create_task(self._execution_worker())
        
        self.logger.info(f"Execution agent started in {self.execution_mode.value} mode")
    
    async def stop(self):
        """Stop the execution agent"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.execution_thread:
            self.execution_thread.cancel()
            try:
                await self.execution_thread
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Execution agent stopped")
    
    def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        """Get order execution status"""
        return self.executed_orders.get(order_id)
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get current position for a ticker"""
        return self.positions.get(ticker)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()
    
    def get_order_book(self) -> Dict[str, OrderRequest]:
        """Get current order book"""
        return self.order_book.copy()
    
    def get_execution_history(self, ticker: Optional[str] = None) -> List[OrderExecution]:
        """Get execution history"""
        executions = list(self.executed_orders.values())
        
        if ticker:
            executions = [e for e in executions if e.ticker == ticker]
        
        return sorted(executions, key=lambda x: x.timestamp)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        # Reset daily metrics if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.daily_volume = 0.0
            self.daily_commission = 0.0
            self.last_reset_date = current_date
        
        total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in self.positions.values())
        total_commission = sum(pos.realized_pnl for pos in self.positions.values())  # This should track commission separately
        
        return {
            'daily_trades': self.daily_trades,
            'daily_volume': self.daily_volume,
            'daily_commission': self.daily_commission,
            'total_positions': len(self.positions),
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'execution_mode': self.execution_mode.value,
            'last_reset_date': self.last_reset_date.isoformat()
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id in self.order_book:
            order = self.order_book[order_id]
            
            # Remove from order book
            del self.order_book[order_id]
            
            # Log cancellation
            self.logger.info(f"Order cancelled: {order_id}")
            
            return True
        else:
            self.logger.warning(f"Order not found for cancellation: {order_id}")
            return False
    
    def get_market_data(self, ticker: str) -> MarketData:
        """Get current market data"""
        return self._get_current_market_data(ticker)
    
    def set_execution_parameters(self, **kwargs):
        """Update execution parameters"""
        if 'spread_multiplier' in kwargs:
            self.spread_multiplier = kwargs['spread_multiplier']
        if 'slippage_bps' in kwargs:
            self.slippage_basis_points = kwargs['slippage_bps']
        if 'execution_delay_ms' in kwargs:
            self.execution_delay_ms = kwargs['execution_delay_ms']
        if 'commission_rate' in kwargs:
            self.commission_rate = kwargs['commission_rate']
        
        self.logger.info(f"Execution parameters updated: {kwargs}")


# Convenience functions
def create_execution_agent(config_path: str = "config/app_config.yaml") -> ExecutionAgent:
    """Create an execution agent instance"""
    return ExecutionAgent(config_path)


async def submit_order(agent: ExecutionAgent,
                      ticker: str,
                      side: OrderSide,
                      quantity: float,
                      price: Optional[float] = None,
                      order_type: OrderType = OrderType.MARKET) -> str:
    """Quick function to submit an order"""
    return await agent.submit_order(ticker, side, order_type, quantity, price)


if __name__ == "__main__":
    # Example usage
    async def main():
        agent = ExecutionAgent()
        await agent.start()
        
        # Submit some orders
        order_id1 = await agent.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        order_id2 = await agent.submit_order(
            ticker="TSLA",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=200.0
        )
        
        # Wait for execution
        await asyncio.sleep(2)
        
        # Check results
        print("Order Status:")
        print(f"  {order_id1}: {agent.get_order_status(order_id1).status.value}")
        print(f"  {order_id2}: {agent.get_order_status(order_id2).status.value}")
        
        print("\nPositions:")
        for ticker, position in agent.get_all_positions().items():
            print(f"  {ticker}: {position.quantity} @ {position.average_price:.2f}")
        
        print("\nPerformance:")
        metrics = agent.get_performance_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        await agent.stop()
    
    asyncio.run(main()) 