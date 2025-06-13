import logging
import redis
import json
import os
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
from functools import lru_cache
import time

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    FILL_OR_KILL = "fill_or_kill"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionError(Exception):
    """Custom exception for execution errors."""
    pass

class MarketDataError(Exception):
    """Custom exception for market data errors."""
    pass

class ExecutionEngine:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize execution engine.
        
        Args:
            config: Configuration dictionary containing:
                - redis_host: Redis host (default: localhost)
                - redis_port: Redis port (default: 6379)
                - redis_db: Redis database (default: 0)
                - redis_password: Redis password
                - redis_ssl: Whether to use SSL (default: false)
                - log_level: Logging level (default: INFO)
                - max_retries: Maximum number of execution retries (default: 3)
                - retry_delay: Delay between retries in seconds (default: 1)
                - price_cache_ttl: Price cache TTL in seconds (default: 60)
                - market_data_provider: Market data provider class
                - market_data_config: Market data provider configuration
        """
        # Load configuration from environment variables with defaults
        self.config = {
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', 6379)),
            'redis_db': int(os.getenv('REDIS_DB', 0)),
            'redis_password': os.getenv('REDIS_PASSWORD'),
            'redis_ssl': os.getenv('REDIS_SSL', 'false').lower() == 'true',
            'log_level': os.getenv('EXECUTION_LOG_LEVEL', 'INFO'),
            'max_retries': int(os.getenv('EXECUTION_MAX_RETRIES', 3)),
            'retry_delay': int(os.getenv('EXECUTION_RETRY_DELAY', 1)),
            'price_cache_ttl': int(os.getenv('PRICE_CACHE_TTL', 60)),
            'market_data_provider': os.getenv('MARKET_DATA_PROVIDER', 'AlphaVantageProvider'),
            'market_data_config': json.loads(os.getenv('MARKET_DATA_CONFIG', '{}'))
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config['log_level']))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.config['redis_host'],
            port=self.config['redis_port'],
            db=self.config['redis_db'],
            password=self.config['redis_password'],
            ssl=self.config['redis_ssl']
        )
        
        # Create trades directory
        self.trades_dir = Path("trades")
        self.trades_dir.mkdir(exist_ok=True)
        
        # Initialize market data provider
        self._init_market_data_provider()
        
        # Initialize price cache
        self.price_cache = {}
        self.price_cache_timestamps = {}
        
        # Initialize async session
        self.session = None

        self.trades = []

    def _init_market_data_provider(self):
        """Initialize market data provider."""
        try:
            provider_class = self.config['market_data_provider']
            if provider_class == 'AlphaVantageProvider':
                from trading.data.providers.alpha_vantage_provider import AlphaVantageProvider
                self.market_data = AlphaVantageProvider(**self.config['market_data_config'])
            elif provider_class == 'YFinanceProvider':
                from trading.data.providers.yfinance_provider import YFinanceProvider
                self.market_data = YFinanceProvider(**self.config['market_data_config'])
            else:
                raise ValueError(f"Unknown market data provider: {provider_class}")
        except Exception as e:
            self.logger.error(f"Failed to initialize market data provider: {str(e)}")
            raise ExecutionError(f"Market data provider initialization failed: {str(e)}")
            
    async def _init_session(self):
        """Initialize aiohttp session if not exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def _close_session(self):
        """Close aiohttp session if exists."""
        if self.session:
            await self.session.close()
            self.session = None
            
    @lru_cache(maxsize=1000)
    def _get_cached_price(self, asset: str) -> Optional[float]:
        """Get cached price for an asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Cached price if valid, None otherwise
        """
        if asset in self.price_cache:
            timestamp = self.price_cache_timestamps.get(asset, 0)
            if time.time() - timestamp < self.config['price_cache_ttl']:
                return self.price_cache[asset]
        return None
        
    def _update_price_cache(self, asset: str, price: float):
        """Update price cache for an asset.
        
        Args:
            asset: Asset symbol
            price: Current price
        """
        self.price_cache[asset] = price
        self.price_cache_timestamps[asset] = time.time()
        
    async def _fetch_price_async(self, asset: str) -> float:
        """Fetch current price asynchronously.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Current price
            
        Raises:
            MarketDataError: If price fetching fails
        """
        try:
            await self._init_session()
            
            # Try cache first
            cached_price = self._get_cached_price(asset)
            if cached_price is not None:
                return cached_price
                
            # Fetch from market data provider
            price = await self.market_data.get_current_price(asset)
            self._update_price_cache(asset, price)
            return price
            
        except Exception as e:
            raise MarketDataError(f"Failed to fetch price for {asset}: {str(e)}")
            
    def _get_current_price(self, asset: str) -> float:
        """Get current price for an asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Current price
            
        Raises:
            MarketDataError: If price fetching fails
        """
        try:
            # Try cache first
            cached_price = self._get_cached_price(asset)
            if cached_price is not None:
                return cached_price
                
            # Fetch synchronously
            price = self.market_data.get_current_price(asset)
            self._update_price_cache(asset, price)
            return price
            
        except Exception as e:
            raise MarketDataError(f"Failed to get price for {asset}: {str(e)}")
            
    async def _execute_order_async(self, order_type: OrderType, asset: str, 
                                 quantity: float, **kwargs) -> Optional[Dict]:
        """Execute order asynchronously.
        
        Args:
            order_type: Type of order to execute
            asset: Asset symbol
            quantity: Order quantity
            **kwargs: Additional order parameters
            
        Returns:
            Trade details if executed, None otherwise
            
        Raises:
            ExecutionError: If order execution fails
        """
        try:
            if order_type == OrderType.MARKET:
                price = await self._fetch_price_async(asset)
                return self.execute_market_order(asset, quantity, price, kwargs.get('metadata'))
            elif order_type == OrderType.LIMIT:
                return self.execute_limit_order(asset, quantity, kwargs['limit_price'], kwargs.get('metadata'))
            elif order_type == OrderType.STOP:
                return self.execute_stop_order(asset, quantity, kwargs['stop_price'], 
                                             kwargs.get('limit_price'), kwargs.get('metadata'))
            elif order_type == OrderType.TRAILING_STOP:
                return self.execute_trailing_stop(asset, quantity, kwargs['trail_percent'],
                                                kwargs.get('metadata'))
            elif order_type == OrderType.FILL_OR_KILL:
                return self.execute_fill_or_kill(asset, quantity, kwargs['limit_price'],
                                               kwargs.get('metadata'))
            else:
                raise ExecutionError(f"Unsupported order type: {order_type}")
                
        except Exception as e:
            raise ExecutionError(f"Async order execution failed: {str(e)}")
            
    async def execute_orders_batch(self, orders: List[Dict]) -> List[Optional[Dict]]:
        """Execute multiple orders in batch asynchronously.
        
        Args:
            orders: List of order dictionaries containing:
                - type: OrderType
                - asset: Asset symbol
                - quantity: Order quantity
                - Additional parameters based on order type
                
        Returns:
            List of trade details for executed orders
            
        Raises:
            ExecutionError: If batch execution fails
        """
        try:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for order in orders:
                    task = self._execute_order_async(
                        OrderType(order['type']),
                        order['asset'],
                        order['quantity'],
                        **{k: v for k, v in order.items() if k not in ['type', 'asset', 'quantity']}
                    )
                    tasks.append(task)
                return await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            raise ExecutionError(f"Batch order execution failed: {str(e)}")
            
    def _save_trade(self, trade: Dict) -> None:
        """Save trade details to file.
        
        Args:
            trade: Trade details dictionary
        """
        try:
            # Create filename with timestamp
            timestamp = datetime.fromisoformat(trade['timestamp'])
            filename = self.trades_dir / f"trades_{timestamp.strftime('%Y%m%d')}.json"
            
            # Load existing trades
            trades = []
            if filename.exists():
                with open(filename, 'r') as f:
                    trades = json.load(f)
                    
            # Append new trade
            trades.append(trade)
            
            # Save updated trades
            with open(filename, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save trade: {str(e)}")
            
    def get_trade_history(self, asset: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict]:
        """Get trade history with optional filters.
        
        Args:
            asset: Filter by asset symbol
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of trade details
        """
        try:
            trades = []
            
            # Get all trade files
            trade_files = sorted(self.trades_dir.glob('trades_*.json'))
            
            for file in trade_files:
                with open(file, 'r') as f:
                    file_trades = json.load(f)
                    
                # Apply filters
                for trade in file_trades:
                    trade_time = datetime.fromisoformat(trade['timestamp'])
                    
                    if asset and trade['asset'] != asset:
                        continue
                    if start_time and trade_time < start_time:
                        continue
                    if end_time and trade_time > end_time:
                        continue
                        
                    trades.append(trade)
                    
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to get trade history: {str(e)}")
            return []
            
    def __del__(self):
        """Cleanup on object destruction."""
        if self.session:
            asyncio.create_task(self._close_session())

    def execute_market_order(self, asset: str, quantity: float, price: float,
                           metadata: Optional[Dict] = None) -> Dict:
        """Execute a market order and log the details.
        
        Args:
            asset: Asset symbol
            quantity: Order quantity
            price: Current market price
            metadata: Additional order metadata
            
        Returns:
            Trade details dictionary
            
        Raises:
            ExecutionError: If order execution fails
        """
        try:
            trade = {
                'asset': asset,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.utcnow().isoformat(),
                'type': OrderType.MARKET.value,
                'status': OrderStatus.FILLED.value,
                'metadata': metadata or {}
            }
            
            # Store trade in Redis
            trade_id = f"{asset}_{trade['timestamp']}"
            self.redis_client.hset('trades', trade_id, json.dumps(trade))
            
            # Save trade to file
            self._save_trade(trade)
            
            self.logger.info(f"Executed market order: {trade}")
            return trade
            
        except Exception as e:
            raise ExecutionError(f"Failed to execute market order: {str(e)}")

    def execute_limit_order(self, asset: str, quantity: float, limit_price: float,
                          metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Execute a limit order if the current price is favorable.
        
        Args:
            asset: Asset symbol
            quantity: Order quantity
            limit_price: Limit price
            metadata: Additional order metadata
            
        Returns:
            Trade details dictionary if executed, None otherwise
            
        Raises:
            ExecutionError: If order execution fails
        """
        try:
            # Get current market price (replace with actual market data)
            current_price = self._get_current_price(asset)
            
            if current_price <= limit_price:
                trade = {
                    'asset': asset,
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': OrderType.LIMIT.value,
                    'status': OrderStatus.FILLED.value,
                    'limit_price': limit_price,
                    'metadata': metadata or {}
                }
                
                # Store trade in Redis
                trade_id = f"{asset}_{trade['timestamp']}"
                self.redis_client.hset('trades', trade_id, json.dumps(trade))
                
                # Save trade to file
                self._save_trade(trade)
                
                self.logger.info(f"Executed limit order: {trade}")
                return trade
            else:
                self.logger.info(f"Limit order not executed: current price {current_price} > limit price {limit_price}")
                return None
                
        except Exception as e:
            raise ExecutionError(f"Failed to execute limit order: {str(e)}")

    def execute_stop_order(self, asset: str, quantity: float, stop_price: float,
                         limit_price: Optional[float] = None,
                         metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Execute a stop order when price reaches stop level.
        
        Args:
            asset: Asset symbol
            quantity: Order quantity
            stop_price: Stop price level
            limit_price: Optional limit price for stop-limit order
            metadata: Additional order metadata
            
        Returns:
            Trade details dictionary if executed, None otherwise
            
        Raises:
            ExecutionError: If order execution fails
        """
        try:
            # Get current market price (replace with actual market data)
            current_price = self._get_current_price(asset)
            
            if current_price <= stop_price:
                if limit_price is not None:
                    # Stop-limit order
                    if current_price <= limit_price:
                        return self.execute_limit_order(asset, quantity, limit_price, metadata)
                    else:
                        self.logger.info(f"Stop-limit order not executed: current price {current_price} > limit price {limit_price}")
                        return None
                else:
                    # Stop order
                    return self.execute_market_order(asset, quantity, current_price, metadata)
            else:
                self.logger.info(f"Stop order not triggered: current price {current_price} > stop price {stop_price}")
                return None
                
        except Exception as e:
            raise ExecutionError(f"Failed to execute stop order: {str(e)}")

    def execute_trailing_stop(self, asset: str, quantity: float, trail_percent: float,
                            metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Execute a trailing stop order.
        
        Args:
            asset: Asset symbol
            quantity: Order quantity
            trail_percent: Trailing stop percentage
            metadata: Additional order metadata
            
        Returns:
            Trade details dictionary if executed, None otherwise
            
        Raises:
            ExecutionError: If order execution fails
        """
        try:
            # Get current market price (replace with actual market data)
            current_price = self._get_current_price(asset)
            
            # Calculate trailing stop price
            stop_price = current_price * (1 - trail_percent)
            
            return self.execute_stop_order(asset, quantity, stop_price, metadata=metadata)

        except Exception as e:
            raise ExecutionError(f"Failed to execute trailing stop order: {str(e)}")

    def execute_fill_or_kill(self, asset: str, quantity: float, limit_price: float,
                             metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Execute a fill-or-kill order.

        The order is executed immediately at or below the limit price. If the
        price is above the limit, the order is cancelled.

        Args:
            asset: Asset symbol
            quantity: Order quantity
            limit_price: Maximum acceptable price
            metadata: Additional order metadata

        Returns:
            Trade details dictionary if filled, None if cancelled
        """
        try:
            current_price = self._get_current_price(asset)

            trade = {
                'asset': asset,
                'quantity': quantity,
                'price': current_price,
                'timestamp': datetime.utcnow().isoformat(),
                'type': OrderType.FILL_OR_KILL.value,
                'limit_price': limit_price,
                'metadata': metadata or {}
            }

            if current_price <= limit_price:
                trade['status'] = OrderStatus.FILLED.value
            else:
                trade['status'] = OrderStatus.CANCELLED.value

            trade_id = f"{asset}_{trade['timestamp']}"
            self.redis_client.hset('trades', trade_id, json.dumps(trade))
            self._save_trade(trade)

            if trade['status'] == OrderStatus.FILLED.value:
                self.logger.info(f"Executed fill-or-kill order: {trade}")
                return trade

            self.logger.info(f"Fill-or-kill order cancelled: {trade}")
            return None

        except Exception as e:
            raise ExecutionError(f"Failed to execute fill-or-kill order: {str(e)}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order was cancelled, False otherwise
            
        Raises:
            ExecutionError: If order cancellation fails
        """
        try:
            order_data = self.redis_client.hget('trades', order_id)
            if order_data:
                order = json.loads(order_data)
                if order['status'] == OrderStatus.PENDING.value:
                    order['status'] = OrderStatus.CANCELLED.value
                    self.redis_client.hset('trades', order_id, json.dumps(order))
                    self.logger.info(f"Cancelled order: {order_id}")
                    return True
            return False
            
        except Exception as e:
            raise ExecutionError(f"Failed to cancel order: {str(e)}")

    def execute_trade(self, symbol: str, quantity: int, order_type: str) -> None:
        """Execute a trade.

        Args:
            symbol (str): The stock symbol to trade.
            quantity (int): The quantity of shares to trade.
            order_type (str): The type of order (e.g., 'buy', 'sell').
        """
        # Placeholder for trade execution logic
        self.trades.append({'symbol': symbol, 'quantity': quantity, 'order_type': order_type})
        print(f"Executed {order_type} order for {quantity} shares of {symbol}")

    def get_trades(self) -> List[Dict[str, Any]]:
        """Get a list of executed trades.

        Returns:
            List[Dict[str, Any]]: A list of executed trades.
        """
        return self.trades 