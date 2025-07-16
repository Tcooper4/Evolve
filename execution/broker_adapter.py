"""
Broker Adapter

This module provides a unified interface for multiple brokers:
- Alpaca (paper and live trading)
- Interactive Brokers (IBKR)
- Polygon (market data)
- Extensible for other brokers

Features:
- Unified API for order submission, position management, and market data
- Automatic connection management and error handling
- Rate limiting and retry logic
- Real-time market data streaming
- Position and account monitoring
"""

import os
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Local imports


class BrokerType(Enum):
    """Supported broker types"""
    ALPACA = "alpaca"
    IBKR = "ibkr"
    POLYGON = "polygon"
    SIMULATION = "simulation"


class OrderType(Enum):
    """Order types supported by brokers"""
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


@dataclass
class AccountInfo:
    """Account information"""
    account_id: str
    cash: float
    buying_power: float
    equity: float
    margin_used: float
    timestamp: str


class BaseBrokerAdapter(ABC):
    """Abstract base class for broker adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_connected = False
        self.rate_limits = {}
        self.last_request_time = {}
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderExecution:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for ticker"""
        pass
    
    @abstractmethod
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_market_data(self, ticker: str) -> MarketData:
        """Get market data for ticker"""
        pass
    
    def _check_rate_limit(self, endpoint: str) -> bool:
        """Check rate limit for endpoint"""
        if endpoint not in self.rate_limits:
            return True
        
        limit = self.rate_limits[endpoint]
        current_time = time.time()
        
        if endpoint not in self.last_request_time:
            self.last_request_time[endpoint] = []
        
        # Remove old requests outside window
        window_start = current_time - limit['window']
        self.last_request_time[endpoint] = [
            t for t in self.last_request_time[endpoint] 
            if t > window_start
        ]
        
        # Check if limit exceeded
        if len(self.last_request_time[endpoint]) >= limit['max_requests']:
            return False
        
        # Add current request
        self.last_request_time[endpoint].append(current_time)
        return True
    
    async def _retry_request(self, func, *args, max_retries: int = 3, **kwargs):
        """Retry request with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = 2 ** attempt
                self.logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)


class AlpacaBrokerAdapter(BaseBrokerAdapter):
    """Alpaca broker adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        self.client = None
        
        # Rate limits
        self.rate_limits = {
            'orders': {'max_requests': 200, 'window': 60},
            'positions': {'max_requests': 200, 'window': 60},
            'market_data': {'max_requests': 1000, 'window': 60}
        }
    
    async def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            # Import alpaca-py
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            
            # Initialize trading client
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=True if 'paper' in self.base_url else False
            )
            
            # Test connection
            account = self.client.get_account()
            self.is_connected = True
            
            self.logger.info(f"Connected to Alpaca: {account.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca"""
        self.is_connected = False
        self.client = None
        self.logger.info("Disconnected from Alpaca")
    
    async def submit_order(self, order: OrderRequest) -> OrderExecution:
        """Submit order to Alpaca"""
        if not self._check_rate_limit('orders'):
            raise Exception("Rate limit exceeded for orders")
        
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
            
            # Convert order side
            side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
            
            # Create order request based on type
            if order.order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=order.ticker,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            elif order.order_type == OrderType.LIMIT:
                order_request = LimitOrderRequest(
                    symbol=order.ticker,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.price
                )
            elif order.order_type == OrderType.STOP:
                order_request = StopOrderRequest(
                    symbol=order.ticker,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=order.stop_price
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
            
            # Submit order
            alpaca_order = self.client.submit_order(order_request)
            
            # Wait for order to be processed
            await asyncio.sleep(1)
            
            # Get order details
            order_details = self.client.get_order_by_id(alpaca_order.id)
            
            # Create execution result
            execution = OrderExecution(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price or 0.0,
                executed_quantity=float(order_details.filled_qty),
                average_price=float(order_details.filled_avg_price) if order_details.filled_avg_price else 0.0,
                commission=0.0,  # Alpaca doesn't charge commissions
                timestamp=order_details.submitted_at.isoformat(),
                status=OrderStatus(order_details.status),
                fills=[],  # Would need to get fills separately
                metadata={'alpaca_order_id': alpaca_order.id}
            )
            
            return execution
            
        except Exception as e:
            self.logger.error(f"Failed to submit order to Alpaca: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca"""
        if not self._check_rate_limit('orders'):
            raise Exception("Rate limit exceeded for orders")
        
        try:
            # Find Alpaca order ID from metadata
            # This is simplified - in practice you'd need to track order mappings
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order on Alpaca: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        """Get order status from Alpaca"""
        if not self._check_rate_limit('orders'):
            raise Exception("Rate limit exceeded for orders")
        
        try:
            order_details = self.client.get_order_by_id(order_id)
            
            # Convert to our format
            execution = OrderExecution(
                order_id=order_id,
                ticker=order_details.symbol,
                side=OrderSide.BUY if order_details.side.value == 'buy' else OrderSide.SELL,
                order_type=OrderType(order_details.order_type.value),
                quantity=float(order_details.qty),
                price=float(order_details.limit_price) if order_details.limit_price else 0.0,
                executed_quantity=float(order_details.filled_qty),
                average_price=float(order_details.filled_avg_price) if order_details.filled_avg_price else 0.0,
                commission=0.0,
                timestamp=order_details.submitted_at.isoformat(),
                status=OrderStatus(order_details.status.value),
                fills=[],
                metadata={}
            )
            
            return execution
            
        except Exception as e:
            self.logger.error(f"Failed to get order status from Alpaca: {e}")
            return None
    
    async def get_position(self, ticker: str) -> Optional[Position]:
        """Get position from Alpaca"""
        if not self._check_rate_limit('positions'):
            raise Exception("Rate limit exceeded for positions")
        
        try:
            position = self.client.get_position(ticker)
            
            return Position(
                ticker=ticker,
                quantity=float(position.qty),
                average_price=float(position.avg_entry_price),
                market_value=float(position.market_value),
                unrealized_pnl=float(position.unrealized_pl),
                realized_pnl=0.0,  # Would need to track separately
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            # Position might not exist
            return None
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions from Alpaca"""
        if not self._check_rate_limit('positions'):
            raise Exception("Rate limit exceeded for positions")
        
        try:
            positions = self.client.get_all_positions()
            
            result = {}
            for position in positions:
                result[position.symbol] = Position(
                    ticker=position.symbol,
                    quantity=float(position.qty),
                    average_price=float(position.avg_entry_price),
                    market_value=float(position.market_value),
                    unrealized_pnl=float(position.unrealized_pl),
                    realized_pnl=0.0,
                    timestamp=datetime.now().isoformat()
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get positions from Alpaca: {e}")
            return {}
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information from Alpaca"""
        try:
            account = self.client.get_account()
            
            return AccountInfo(
                account_id=account.id,
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                equity=float(account.equity),
                margin_used=float(account.margin_used),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get account info from Alpaca: {e}")
            raise
    
    async def get_market_data(self, ticker: str) -> MarketData:
        """Get market data from Alpaca"""
        if not self._check_rate_limit('market_data'):
            raise Exception("Rate limit exceeded for market data")
        
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            
            # Get latest quote
            data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = data_client.get_stock_latest_quote(request)
            
            if ticker in quote:
                q = quote[ticker]
                spread = q.ask_price - q.bid_price
                
                return MarketData(
                    ticker=ticker,
                    bid=float(q.bid_price),
                    ask=float(q.ask_price),
                    last=float(q.bid_price + spread/2),  # Approximate last price
                    volume=0,  # Would need separate request
                    timestamp=q.timestamp.isoformat(),
                    spread=float(spread),
                    volatility=0.02  # Would need to calculate
                )
            else:
                raise Exception(f"No market data available for {ticker}")
                
        except Exception as e:
            self.logger.error(f"Failed to get market data from Alpaca: {e}")
            raise


class IBKRBrokerAdapter(BaseBrokerAdapter):
    """Interactive Brokers (IBKR) adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7497)  # 7497 for TWS, 4001 for IB Gateway
        self.client_id = config.get('client_id', 1)
        self.client = None
        
        # Rate limits
        self.rate_limits = {
            'orders': {'max_requests': 50, 'window': 60},
            'positions': {'max_requests': 50, 'window': 60},
            'market_data': {'max_requests': 100, 'window': 60}
        }
    
    async def connect(self) -> bool:
        """Connect to IBKR"""
        try:
            # Import ib_insync
            from ib_insync import IB
            
            self.client = IB()
            self.client.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id
            )
            
            # Wait for connection
            await asyncio.sleep(2)
            
            if self.client.isConnected():
                self.is_connected = True
                self.logger.info(f"Connected to IBKR on {self.host}:{self.port}")
                return True
            else:
                self.logger.error("Failed to connect to IBKR")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from IBKR"""
        if self.client:
            self.client.disconnect()
        self.is_connected = False
        self.logger.info("Disconnected from IBKR")
    
    async def submit_order(self, order: OrderRequest) -> OrderExecution:
        """Submit order to IBKR"""
        if not self._check_rate_limit('orders'):
            raise Exception("Rate limit exceeded for orders")
        
        try:
            from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
            
            # Create contract
            contract = Stock(order.ticker, 'SMART', 'USD')
            
            # Create order based on type
            if order.order_type == OrderType.MARKET:
                ib_order = MarketOrder(order.side.value.upper(), order.quantity)
            elif order.order_type == OrderType.LIMIT:
                ib_order = LimitOrder(order.side.value.upper(), order.quantity, order.price)
            elif order.order_type == OrderType.STOP:
                ib_order = StopOrder(order.side.value.upper(), order.quantity, order.stop_price)
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
            
            # Submit order
            trade = self.client.placeOrder(contract, ib_order)
            
            # Wait for order to be processed
            await asyncio.sleep(2)
            
            # Get order status
            order_status = trade.orderStatus
            
            # Create execution result
            execution = OrderExecution(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price or 0.0,
                executed_quantity=order_status.filled,
                average_price=order_status.avgFillPrice,
                commission=order_status.commission,
                timestamp=datetime.now().isoformat(),
                status=OrderStatus(order_status.status.lower()),
                fills=[],
                metadata={'ib_order_id': trade.order.orderId}
            )
            
            return execution
            
        except Exception as e:
            self.logger.error(f"Failed to submit order to IBKR: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on IBKR"""
        if not self._check_rate_limit('orders'):
            raise Exception("Rate limit exceeded for orders")
        
        try:
            # Find order by ID and cancel
            # This is simplified - in practice you'd need to track order mappings
            self.client.cancelOrder(order_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order on IBKR: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        """Get order status from IBKR"""
        # Implementation would depend on how you track orders
        # This is a placeholder
        return None
    
    async def get_position(self, ticker: str) -> Optional[Position]:
        """Get position from IBKR"""
        if not self._check_rate_limit('positions'):
            raise Exception("Rate limit exceeded for positions")
        
        try:
            positions = self.client.positions()
            
            for position in positions:
                if position.contract.symbol == ticker:
                    return Position(
                        ticker=ticker,
                        quantity=position.position,
                        average_price=0.0,  # Would need to calculate
                        market_value=position.marketValue,
                        unrealized_pnl=position.unrealizedPNL,
                        realized_pnl=0.0,
                        timestamp=datetime.now().isoformat()
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get position from IBKR: {e}")
            return None
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions from IBKR"""
        if not self._check_rate_limit('positions'):
            raise Exception("Rate limit exceeded for positions")
        
        try:
            positions = self.client.positions()
            
            result = {}
            for position in positions:
                ticker = position.contract.symbol
                result[ticker] = Position(
                    ticker=ticker,
                    quantity=position.position,
                    average_price=0.0,
                    market_value=position.marketValue,
                    unrealized_pnl=position.unrealizedPNL,
                    realized_pnl=0.0,
                    timestamp=datetime.now().isoformat()
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get positions from IBKR: {e}")
            return {}
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information from IBKR"""
        try:
            account_values = self.client.accountSummary()
            
            # Parse account values
            cash = 0.0
            buying_power = 0.0
            equity = 0.0
            margin_used = 0.0
            
            for value in account_values:
                if value.tag == 'AvailableFunds':
                    cash = float(value.value)
                elif value.tag == 'BuyingPower':
                    buying_power = float(value.value)
                elif value.tag == 'NetLiquidation':
                    equity = float(value.value)
                elif value.tag == 'GrossPositionValue':
                    margin_used = float(value.value)
            
            return AccountInfo(
                account_id=self.client.clientId,
                cash=cash,
                buying_power=buying_power,
                equity=equity,
                margin_used=margin_used,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get account info from IBKR: {e}")
            raise
    
    async def get_market_data(self, ticker: str) -> MarketData:
        """Get market data from IBKR"""
        if not self._check_rate_limit('market_data'):
            raise Exception("Rate limit exceeded for market data")
        
        try:
            from ib_insync import Stock
            
            contract = Stock(ticker, 'SMART', 'USD')
            
            # Request market data
            self.client.reqMktData(contract)
            
            # Wait for data
            await asyncio.sleep(1)
            
            # Get ticker
            ticker_data = self.client.ticker(contract)
            
            if ticker_data:
                spread = ticker_data.ask - ticker_data.bid
                
                return MarketData(
                    ticker=ticker,
                    bid=ticker_data.bid,
                    ask=ticker_data.ask,
                    last=ticker_data.last,
                    volume=ticker_data.volume,
                    timestamp=datetime.now().isoformat(),
                    spread=spread,
                    volatility=0.02
                )
            else:
                raise Exception(f"No market data available for {ticker}")
                
        except Exception as e:
            self.logger.error(f"Failed to get market data from IBKR: {e}")
            raise


class PolygonBrokerAdapter(BaseBrokerAdapter):
    """Polygon.io market data adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = "https://api.polygon.io"
        
        # Rate limits
        self.rate_limits = {
            'market_data': {'max_requests': 1000, 'window': 60}
        }
    
    async def connect(self) -> bool:
        """Connect to Polygon (API key validation)"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/aggs/ticker/AAPL/prev"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        self.is_connected = True
                        self.logger.info("Connected to Polygon.io")
                        return True
                    else:
                        self.logger.error(f"Failed to connect to Polygon: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to connect to Polygon: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Polygon"""
        self.is_connected = False
        self.logger.info("Disconnected from Polygon")
    
    async def get_market_data(self, ticker: str) -> MarketData:
        """Get market data from Polygon"""
        if not self._check_rate_limit('market_data'):
            raise Exception("Rate limit exceeded for market data")
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Get previous day's data for bid/ask
                url = f"{self.base_url}/v2/aggs/ticker/{ticker}/prev"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data['results']:
                            result = data['results'][0]
                            
                            # Estimate bid/ask from OHLC
                            high = result['h']
                            low = result['l']
                            close = result['c']
                            
                            # Simple bid/ask estimation
                            spread = (high - low) * 0.001
                            bid = close - spread/2
                            ask = close + spread/2
                            
                            return MarketData(
                                ticker=ticker,
                                bid=bid,
                                ask=ask,
                                last=close,
                                volume=result['v'],
                                timestamp=datetime.now().isoformat(),
                                spread=spread,
                                volatility=0.02
                            )
                        else:
                            raise Exception(f"No data available for {ticker}")
                    else:
                        raise Exception(f"Failed to get data for {ticker}: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to get market data from Polygon: {e}")
            raise
    
    # Polygon is primarily for market data, so other methods are not implemented
    async def submit_order(self, order: OrderRequest) -> OrderExecution:
        raise NotImplementedError("Polygon adapter does not support order submission")
    
    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("Polygon adapter does not support order cancellation")
    
    async def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        raise NotImplementedError("Polygon adapter does not support order status")
    
    async def get_position(self, ticker: str) -> Optional[Position]:
        raise NotImplementedError("Polygon adapter does not support position management")
    
    async def get_all_positions(self) -> Dict[str, Position]:
        raise NotImplementedError("Polygon adapter does not support position management")
    
    async def get_account_info(self) -> AccountInfo:
        raise NotImplementedError("Polygon adapter does not support account management")


class SimulationBrokerAdapter(BaseBrokerAdapter):
    """Simulation broker adapter for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.orders = {}
        self.positions = {}
        self.account = AccountInfo(
            account_id="SIM_ACCOUNT",
            cash=100000.0,
            buying_power=100000.0,
            equity=100000.0,
            margin_used=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    async def connect(self) -> bool:
        """Connect to simulation"""
        self.is_connected = True
        self.logger.info("Connected to simulation broker")
        return True
    
    async def disconnect(self):
        """Disconnect from simulation"""
        self.is_connected = False
        self.logger.info("Disconnected from simulation broker")
    
    async def submit_order(self, order: OrderRequest) -> OrderExecution:
        """Simulate order submission"""
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        # Generate execution price
        if order.order_type == OrderType.MARKET:
            execution_price = 100.0 + np.random.normal(0, 1)  # Simulated price
        else:
            execution_price = order.price
        
        # Create execution
        execution = OrderExecution(
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price or execution_price,
            executed_quantity=order.quantity,
            average_price=execution_price,
            commission=1.0,
            timestamp=datetime.now().isoformat(),
            status=OrderStatus.FILLED,
            fills=[{
                'fill_id': str(uuid.uuid4()),
                'quantity': order.quantity,
                'price': execution_price,
                'timestamp': datetime.now().isoformat(),
                'commission': 1.0
            }],
            metadata={'simulation': True}
        )
        
        # Store order
        self.orders[order.order_id] = execution
        
        return execution
    
    async def cancel_order(self, order_id: str) -> bool:
        """Simulate order cancellation"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        """Get simulated order status"""
        return self.orders.get(order_id)
    
    async def get_position(self, ticker: str) -> Optional[Position]:
        """Get simulated position"""
        return self.positions.get(ticker)
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all simulated positions"""
        return self.positions.copy()
    
    async def get_account_info(self) -> AccountInfo:
        """Get simulated account info"""
        return self.account
    
    async def get_market_data(self, ticker: str) -> MarketData:
        """Get simulated market data"""
        price = 100.0 + np.random.normal(0, 2)
        spread = price * 0.001
        
        return MarketData(
            ticker=ticker,
            bid=price - spread/2,
            ask=price + spread/2,
            last=price,
            volume=np.random.randint(1000000, 10000000),
            timestamp=datetime.now().isoformat(),
            spread=spread,
            volatility=0.02
        )


class BrokerAdapter:
    """
    Unified broker adapter that supports multiple brokers
    """
    
    def __init__(self, broker_type: str = "simulation", config: Dict[str, Any] = None):
        self.broker_type = BrokerType(broker_type.lower())
        self.config = config or {}
        
        # Create specific adapter
        if self.broker_type == BrokerType.ALPACA:
            self.adapter = AlpacaBrokerAdapter(self.config)
        elif self.broker_type == BrokerType.IBKR:
            self.adapter = IBKRBrokerAdapter(self.config)
        elif self.broker_type == BrokerType.POLYGON:
            self.adapter = PolygonBrokerAdapter(self.config)
        elif self.broker_type == BrokerType.SIMULATION:
            self.adapter = SimulationBrokerAdapter(self.config)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
        
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """Connect to broker"""
        return await self.adapter.connect()
    
    async def disconnect(self):
        """Disconnect from broker"""
        await self.adapter.disconnect()
    
    async def submit_order(self, order: OrderRequest) -> OrderExecution:
        """Submit order"""
        return await self.adapter.submit_order(order)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        return await self.adapter.cancel_order(order_id)
    
    async def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        """Get order status"""
        return await self.adapter.get_order_status(order_id)
    
    async def get_position(self, ticker: str) -> Optional[Position]:
        """Get position"""
        return await self.adapter.get_position(ticker)
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return await self.adapter.get_all_positions()
    
    async def get_account_info(self) -> AccountInfo:
        """Get account info"""
        return await self.adapter.get_account_info()
    
    async def get_market_data(self, ticker: str) -> MarketData:
        """Get market data"""
        return await self.adapter.get_market_data(ticker)


# Convenience functions
def create_broker_adapter(broker_type: str = "simulation", config: Dict[str, Any] = None) -> BrokerAdapter:
    """Create a broker adapter instance"""
    return BrokerAdapter(broker_type, config)


async def test_broker_connection(broker_type: str = "simulation", config: Dict[str, Any] = None) -> bool:
    """Test broker connection"""
    adapter = create_broker_adapter(broker_type, config)
    try:
        connected = await adapter.connect()
        await adapter.disconnect()
        return connected
    except Exception as e:
        logging.error(f"Broker connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    async def main():
        # Test simulation broker
        adapter = create_broker_adapter("simulation")
        await adapter.connect()
        
        # Get market data
        market_data = await adapter.get_market_data("AAPL")
        print(f"AAPL Market Data: {market_data}")
        
        # Submit order
        order = OrderRequest(
            order_id="test_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        execution = await adapter.submit_order(order)
        print(f"Order Execution: {execution}")
        
        await adapter.disconnect()
    
    asyncio.run(main()) 