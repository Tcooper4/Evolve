"""Live Trade Execution Engine for Evolve Trading Platform.

This module provides live trade execution capabilities with simulation
and optional real trading via Alpaca/Robinhood APIs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import broker APIs
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    tradeapi = None

logger = logging.getLogger(__name__)

@dataclass
class TradeOrder:
    """Trade order structure."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit', 'stop'
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'day'
    order_id: Optional[str] = None
    status: str = 'pending'
    created_at: Optional[str] = None
    filled_at: Optional[str] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    commission: Optional[float] = None

@dataclass
class ExecutionConfig:
    """Execution configuration."""
    live_trading: bool = False
    broker: str = 'alpaca'  # 'alpaca', 'robinhood', 'simulation'
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    paper_trading: bool = True
    max_position_size: float = 0.1  # Max 10% of portfolio per position
    max_daily_trades: int = 50
    slippage_model: str = 'fixed'  # 'fixed', 'proportional', 'realistic'
    commission_model: str = 'fixed'  # 'fixed', 'proportional'
    fixed_commission: float = 1.0
    commission_rate: float = 0.001  # 0.1%
    min_order_size: float = 100.0
    max_slippage: float = 0.002  # 0.2%

class SlippageModel:
    """Slippage modeling for realistic trade execution."""
    
    def __init__(self, model_type: str = 'realistic'):
        """Initialize slippage model."""
        self.model_type = model_type
        
    def calculate_slippage(self, order_size: float, 
                          current_price: float,
                          market_volatility: float = 0.02) -> float:
        """Calculate slippage for order."""
        
        if self.model_type == 'fixed':
            return self.max_slippage
        
        elif self.model_type == 'proportional':
            # Proportional to order size
            return min(self.max_slippage, order_size / 10000 * 0.001)
        
        elif self.model_type == 'realistic':
            # Realistic model considering market impact
            base_slippage = 0.0005  # 0.05% base slippage
            
            # Size impact
            size_impact = min(0.001, order_size / 100000 * 0.002)
            
            # Volatility impact
            volatility_impact = market_volatility * 0.1
            
            # Time of day impact (simulate market hours)
            hour = datetime.now().hour
            if 9 <= hour <= 11 or 14 <= hour <= 16:  # Market open/close
                time_impact = 0.0002
            else:
                time_impact = 0.0005
            
            total_slippage = base_slippage + size_impact + volatility_impact + time_impact
            return min(self.max_slippage, total_slippage)
        
        return 0.0

class CommissionModel:
    """Commission modeling for trade execution."""
    
    def __init__(self, model_type: str = 'fixed', 
                 fixed_amount: float = 1.0,
                 rate: float = 0.001):
        """Initialize commission model."""
        self.model_type = model_type
        self.fixed_amount = fixed_amount
        self.rate = rate
    
    def calculate_commission(self, order_value: float) -> float:
        """Calculate commission for order."""
        
        if self.model_type == 'fixed':
            return self.fixed_amount
        
        elif self.model_type == 'proportional':
            return order_value * self.rate
        
        elif self.model_type == 'tiered':
            # Tiered commission structure
            if order_value < 1000:
                return max(1.0, order_value * 0.002)
            elif order_value < 10000:
                return max(2.0, order_value * 0.001)
            else:
                return max(5.0, order_value * 0.0005)
        
        return 0.0

class TradeExecutor:
    """Main trade execution engine."""
    
    def __init__(self, config: ExecutionConfig):
        """Initialize trade executor."""
        self.config = config
        self.slippage_model = SlippageModel(config.slippage_model)
        self.commission_model = CommissionModel(
            config.commission_model,
            config.fixed_commission,
            config.commission_rate
        )
        
        # Initialize broker connection
        self.broker_api = None
        self.initialize_broker()
        
        # Trading state
        self.portfolio = {}
        self.order_history = []
        self.daily_trades = 0
        self.last_trade_reset = datetime.now().date()
        
        # Performance tracking
        self.execution_metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'avg_fill_time': 0.0
        }
    
    def initialize_broker(self) -> bool:
        """Initialize broker connection."""
        if not self.config.live_trading:
            logger.info("Running in simulation mode")
            return True
        
        try:
            if self.config.broker == 'alpaca' and ALPACA_AVAILABLE:
                self.broker_api = tradeapi.REST(
                    key=self.config.api_key,
                    secret=self.config.api_secret,
                    base_url='https://paper-api.alpaca.markets' if self.config.paper_trading else 'https://api.alpaca.markets'
                )
                logger.info("Connected to Alpaca API")
                return True
            
            elif self.config.broker == 'robinhood':
                # Robinhood integration would go here
                logger.warning("Robinhood integration not implemented")
                return False
            
            else:
                logger.error(f"Unsupported broker: {self.config.broker}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing broker: {e}")
            return False
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for symbol."""
        try:
            if self.config.live_trading and self.broker_api:
                # Get real market data
                if self.config.broker == 'alpaca':
                    ticker = self.broker_api.get_latest_trade(symbol)
                    return {
                        'symbol': symbol,
                        'price': float(ticker.price),
                        'volume': int(ticker.size),
                        'timestamp': ticker.timestamp.isoformat()
                    }
            else:
                # Simulated market data
                return self._simulate_market_data(symbol)
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _simulate_market_data(self, symbol: str) -> Dict:
        """Simulate market data for testing."""
        # Simple price simulation with some noise
        base_price = 100.0  # Could be loaded from historical data
        noise = np.random.normal(0, 0.01)
        price = base_price * (1 + noise)
        
        return {
            'symbol': symbol,
            'price': price,
            'volume': np.random.randint(1000, 10000),
            'timestamp': datetime.now().isoformat()
        }
    
    def place_order(self, order: TradeOrder) -> bool:
        """Place trade order."""
        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False
        
        # Validate order
        if not self._validate_order(order):
            return False
        
        try:
            if self.config.live_trading and self.broker_api:
                return self._place_live_order(order)
            else:
                return self._place_simulated_order(order)
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False
    
    def _validate_order(self, order: TradeOrder) -> bool:
        """Validate order parameters."""
        if order.quantity <= 0:
            logger.error("Invalid quantity")
            return False
        
        if order.side not in ['buy', 'sell']:
            logger.error("Invalid order side")
            return False
        
        if order.order_type not in ['market', 'limit', 'stop']:
            logger.error("Invalid order type")
            return False
        
        # Check minimum order size
        market_data = self.get_market_data(order.symbol)
        if market_data:
            order_value = order.quantity * market_data['price']
            if order_value < self.config.min_order_size:
                logger.error(f"Order value {order_value} below minimum {self.config.min_order_size}")
                return False
        
        return True
    
    def _place_live_order(self, order: TradeOrder) -> bool:
        """Place live order with broker."""
        try:
            if self.config.broker == 'alpaca':
                # Place order with Alpaca
                api_order = self.broker_api.submit_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=order.side,
                    type=order.order_type,
                    time_in_force=order.time_in_force,
                    limit_price=order.price if order.order_type == 'limit' else None,
                    stop_price=order.stop_price if order.order_type == 'stop' else None
                )
                
                order.order_id = api_order.id
                order.status = api_order.status
                order.created_at = api_order.created_at.isoformat()
                
                logger.info(f"Placed live order: {order.order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error placing live order: {e}")
            return False
    
    def _place_simulated_order(self, order: TradeOrder) -> bool:
        """Place simulated order."""
        # Simulate order execution
        order.order_id = f"sim_{int(time.time())}"
        order.created_at = datetime.now().isoformat()
        
        # Simulate fill
        market_data = self.get_market_data(order.symbol)
        if market_data:
            # Apply slippage
            slippage = self.slippage_model.calculate_slippage(
                order.quantity * market_data['price'],
                market_data['price']
            )
            
            if order.side == 'buy':
                fill_price = market_data['price'] * (1 + slippage)
            else:
                fill_price = market_data['price'] * (1 - slippage)
            
            # Calculate commission
            commission = self.commission_model.calculate_commission(
                order.quantity * fill_price
            )
            
            # Update order
            order.filled_at = datetime.now().isoformat()
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.commission = commission
            order.status = 'filled'
            
            # Update portfolio
            self._update_portfolio(order)
            
            # Update metrics
            self.execution_metrics['total_orders'] += 1
            self.execution_metrics['filled_orders'] += 1
            self.execution_metrics['total_commission'] += commission
            self.execution_metrics['total_slippage'] += slippage * order.quantity * market_data['price']
            
            self.daily_trades += 1
            self.order_history.append(order)
            
            logger.info(f"Simulated order filled: {order.order_id} at {fill_price:.2f}")
            return True
        
        return False
    
    def _update_portfolio(self, order: TradeOrder):
        """Update portfolio after order fill."""
        symbol = order.symbol
        
        if symbol not in self.portfolio:
            self.portfolio[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'total_cost': 0
            }
        
        if order.side == 'buy':
            # Add to position
            new_quantity = self.portfolio[symbol]['quantity'] + order.filled_quantity
            new_cost = self.portfolio[symbol]['total_cost'] + (order.filled_quantity * order.filled_price)
            
            self.portfolio[symbol]['quantity'] = new_quantity
            self.portfolio[symbol]['total_cost'] = new_cost
            self.portfolio[symbol]['avg_price'] = new_cost / new_quantity if new_quantity > 0 else 0
        
        else:  # sell
            # Reduce position
            self.portfolio[symbol]['quantity'] -= order.filled_quantity
            if self.portfolio[symbol]['quantity'] <= 0:
                self.portfolio[symbol]['quantity'] = 0
                self.portfolio[symbol]['avg_price'] = 0
                self.portfolio[symbol]['total_cost'] = 0
    
    def get_order_status(self, order_id: str) -> Optional[TradeOrder]:
        """Get order status."""
        if self.config.live_trading and self.broker_api:
            try:
                if self.config.broker == 'alpaca':
                    api_order = self.broker_api.get_order(order_id)
                    # Convert to TradeOrder format
                    return self._convert_api_order(api_order)
            except Exception as e:
                logger.error(f"Error getting order status: {e}")
        
        # Check local history
        for order in self.order_history:
            if order.order_id == order_id:
                return order
        
        return None
    
    def _convert_api_order(self, api_order) -> TradeOrder:
        """Convert broker API order to TradeOrder."""
        return TradeOrder(
            symbol=api_order.symbol,
            side=api_order.side,
            quantity=float(api_order.qty),
            order_type=api_order.type,
            price=float(api_order.limit_price) if api_order.limit_price else None,
            stop_price=float(api_order.stop_price) if api_order.stop_price else None,
            order_id=api_order.id,
            status=api_order.status,
            created_at=api_order.created_at.isoformat(),
            filled_at=api_order.filled_at.isoformat() if api_order.filled_at else None,
            filled_price=float(api_order.filled_avg_price) if api_order.filled_avg_price else None,
            filled_quantity=float(api_order.filled_qty) if api_order.filled_qty else None
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            if self.config.live_trading and self.broker_api:
                if self.config.broker == 'alpaca':
                    self.broker_api.cancel_order(order_id)
                    logger.info(f"Cancelled live order: {order_id}")
                    return True
            
            # For simulated orders, mark as cancelled
            for order in self.order_history:
                if order.order_id == order_id and order.status == 'pending':
                    order.status = 'cancelled'
                    self.execution_metrics['cancelled_orders'] += 1
                    logger.info(f"Cancelled simulated order: {order_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        total_value = 0.0
        positions = []
        
        for symbol, position in self.portfolio.items():
            if position['quantity'] > 0:
                market_data = self.get_market_data(symbol)
                if market_data:
                    current_value = position['quantity'] * market_data['price']
                    total_value += current_value
                    
                    positions.append({
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'avg_price': position['avg_price'],
                        'current_price': market_data['price'],
                        'current_value': current_value,
                        'unrealized_pnl': current_value - position['total_cost']
                    })
        
        return {
            'total_value': total_value,
            'positions': positions,
            'num_positions': len(positions),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics."""
        metrics = self.execution_metrics.copy()
        
        if metrics['filled_orders'] > 0:
            metrics['fill_rate'] = metrics['filled_orders'] / metrics['total_orders']
            metrics['avg_commission'] = metrics['total_commission'] / metrics['filled_orders']
            metrics['avg_slippage'] = metrics['total_slippage'] / metrics['filled_orders']
        
        metrics['daily_trades'] = self.daily_trades
        metrics['max_daily_trades'] = self.config.max_daily_trades
        
        return metrics
    
    def reset_daily_limits(self):
        """Reset daily trading limits."""
        current_date = datetime.now().date()
        if current_date > self.last_trade_reset:
            self.daily_trades = 0
            self.last_trade_reset = current_date
            logger.info("Reset daily trading limits")

# Global trade executor instance
trade_executor = None

def get_trade_executor(config: Optional[ExecutionConfig] = None) -> TradeExecutor:
    """Get the global trade executor instance."""
    global trade_executor
    if trade_executor is None:
        if config is None:
            config = ExecutionConfig()
        trade_executor = TradeExecutor(config)
    return trade_executor 