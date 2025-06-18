"""Execution engine for trading operations with robust error handling and caching."""

import logging
import redis
import json
import os
import uuid
import importlib
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Union, Tuple, Any, Type
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from contextlib import asynccontextmanager

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
                - slippage_percent: Expected slippage percentage (default: 0.1)
                - fee_percent: Trading fee percentage (default: 0.1)
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
            'market_data_config': json.loads(os.getenv('MARKET_DATA_CONFIG', '{}')),
            'slippage_percent': float(os.getenv('SLIPPAGE_PERCENT', 0.1)),
            'fee_percent': float(os.getenv('FEE_PERCENT', 0.1))
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
        
        # Initialize trades list
        self.trades = []

    def _init_market_data_provider(self):
        """Initialize market data provider using dynamic class loading."""
        try:
            provider_class = self.config['market_data_provider']
            provider_module = f"trading.data.providers.{provider_class.lower()}_provider"
            
            # Dynamically import provider class
            module = importlib.import_module(provider_module)
            provider_class = getattr(module, provider_class)
            
            # Initialize provider
            self.market_data = provider_class(**self.config['market_data_config'])
            self.logger.info(f"Initialized market data provider: {provider_class.__name__}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize market data provider: {str(e)}")
            raise ExecutionError(f"Market data provider initialization failed: {str(e)}")

    @asynccontextmanager
    async def _get_session(self):
        """Async context manager for aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        try:
            yield self.session
        except Exception as e:
            self.logger.error(f"Session error: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    def _calculate_transaction_costs(self, price: float, quantity: float) -> Dict[str, float]:
        """Calculate transaction costs including slippage and fees.
        
        Args:
            price: Base price
            quantity: Order quantity
            
        Returns:
            Dictionary with cost breakdown
        """
        slippage = price * (self.config['slippage_percent'] / 100)
        fee = price * quantity * (self.config['fee_percent'] / 100)
        total_cost = price * quantity + fee
        
        return {
            'base_price': price,
            'slippage': slippage,
            'fee': fee,
            'total_cost': total_cost
        }

    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        return str(uuid.uuid4())

    def _save_trade(self, trade: Dict) -> None:
        """Save trade with robust error recovery.
        
        Args:
            trade: Trade dictionary to save
        """
        # Add order ID if missing
        if 'order_id' not in trade:
            trade['order_id'] = self._generate_order_id()
            
        # Add timestamp if missing
        if 'timestamp' not in trade:
            trade['timestamp'] = datetime.now().isoformat()
            
        # Add transaction costs
        if 'price' in trade and 'quantity' in trade:
            costs = self._calculate_transaction_costs(trade['price'], trade['quantity'])
            trade.update(costs)
            
        # Save to primary file
        try:
            trade_file = self.trades_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
            with open(trade_file, 'a') as f:
                json.dump(trade, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save trade to primary file: {str(e)}")
            
            # Try backup file
            try:
                backup_file = self.trades_dir / f"trades_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(trade, f)
                    f.write('\n')
                self.logger.info(f"Saved trade to backup file: {backup_file}")
            except Exception as backup_error:
                self.logger.error(f"Failed to save trade to backup file: {str(backup_error)}")
                raise ExecutionError("Failed to save trade to any file")

    async def execute_trade(self, symbol: str, quantity: float, order_type: OrderType, **kwargs) -> Dict:
        """Execute a trade with full error handling and logging.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            order_type: Type of order
            **kwargs: Additional order parameters
            
        Returns:
            Trade execution result
            
        Raises:
            ExecutionError: If trade execution fails
        """
        try:
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Get current price
            price = await self._fetch_price_async(symbol)
            
            # Calculate transaction costs
            costs = self._calculate_transaction_costs(price, quantity)
            
            # Create trade record
            trade = {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'order_type': order_type.value,
                'price': price,
                'timestamp': datetime.now().isoformat(),
                'status': OrderStatus.PENDING.value,
                **costs,
                **kwargs
            }
            
            # Execute based on order type
            if order_type == OrderType.MARKET:
                trade['status'] = OrderStatus.FILLED.value
            elif order_type == OrderType.LIMIT:
                if 'limit_price' not in kwargs:
                    raise ExecutionError("Limit price required for limit orders")
                if price <= kwargs['limit_price']:
                    trade['status'] = OrderStatus.FILLED.value
                else:
                    trade['status'] = OrderStatus.PENDING.value
            # Add other order types as needed
            
            # Save trade
            self._save_trade(trade)
            
            # Update trades list
            self.trades.append(trade)
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            raise ExecutionError(f"Trade execution failed: {str(e)}")

    def get_trades(self) -> List[Dict[str, Any]]:
        """Get all trades with optional filtering.
        
        Returns:
            List of trade dictionaries
        """
        return self.trades

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None

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
            await self._get_session()
            
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