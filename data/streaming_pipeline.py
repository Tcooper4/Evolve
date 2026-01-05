"""Real-Time Streaming Data Pipeline for Evolve Trading Platform.

This module provides real-time data streaming with multi-timeframe support,
websocket connections, and in-memory caching for low-latency trading.
"""

import asyncio
import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd
import websockets

# Data provider imports
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"yfinance not available: {e}")
    YFINANCE_AVAILABLE = False

try:
    pass

    ALPHA_VANTAGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"alpha_vantage not available: {e}")
    ALPHA_VANTAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data structure."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    source: str


@dataclass
class StreamingConfig:
    """Streaming configuration."""

    symbols: List[str]
    timeframes: List[str]  # ["1m", "5m", "15m", "1h", "1d"]
    providers: List[str]  # ["polygon", "finnhub", "alpaca", "yfinance"]
    cache_size: int = 1000
    update_frequency: float = 1.0  # seconds
    retry_attempts: int = 3
    retry_delay: float = 5.0


@dataclass
class DataTrigger:
    """Data trigger for real-time events."""

    symbol: str
    condition: str  # "price_above", "price_below", "volume_spike", "volatility_spike"
    threshold: float
    timeframe: str
    callback: Callable
    is_active: bool = True
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None


class InMemoryCache:
    """In-memory cache for real-time data."""

    def __init__(self, max_size: int = 1000, compression: bool = True):
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of data points per symbol/timeframe
            compression: Enable data compression
        """
        self.max_size = max_size
        self.compression = compression

        # Cache structure: {symbol: {timeframe: deque}}
        self.cache = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_size)))

        # Metadata
        self.metadata = defaultdict(dict)
        self.last_update = defaultdict(dict)

        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0

        logger.info(f"Initialized InMemoryCache with max_size={max_size}")

        # Removed return statement - __init__ should not return values

    def add_data(self, symbol: str, timeframe: str, data: MarketData):
        """Add data to cache.

        Args:
            symbol: Symbol
            timeframe: Timeframe
            data: Market data
        """
        cache_key = f"{symbol}_{timeframe}"

        # Add to cache
        self.cache[symbol][timeframe].append(data)

        # Update metadata
        self.metadata[cache_key] = {
            "last_update": datetime.now(),
            "data_count": len(self.cache[symbol][timeframe]),
            "oldest_timestamp": data.timestamp,
            "newest_timestamp": data.timestamp,
        }

        # Update last update time
        self.last_update[symbol][timeframe] = datetime.now()

    def get_data(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[MarketData]:
        """Get data from cache.

        Args:
            symbol: Symbol
            timeframe: Timeframe
            limit: Maximum number of data points
            start_time: Start time filter
            end_time: End time filter

        Returns:
            List of market data
        """
        self.total_requests += 1

        if symbol not in self.cache or timeframe not in self.cache[symbol]:
            self.miss_count += 1
            return []

        data_list = list(self.cache[symbol][timeframe])

        # Apply time filters
        if start_time:
            data_list = [d for d in data_list if d.timestamp >= start_time]

        if end_time:
            data_list = [d for d in data_list if d.timestamp <= end_time]

        # Apply limit
        if limit:
            data_list = data_list[-limit:]

        self.hit_count += 1
        return data_list

    def get_latest_data(self, symbol: str, timeframe: str) -> Optional[MarketData]:
        """Get latest data point for symbol/timeframe."""
        # noqa: F841 - Placeholder implementation
        return (
            self.get_data(symbol, timeframe, limit=1)[0]
            if self.get_data(symbol, timeframe, limit=1)
            else None
        )

    def get_dataframe(
        self, symbol: str, timeframe: str, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get data as pandas DataFrame."""
        data_list = self.get_data(symbol, timeframe, limit=limit)

        if not data_list:
            return pd.DataFrame()

        # Convert to DataFrame
        df_data = []
        for data in data_list:
            df_data.append(
                {
                    "timestamp": data.timestamp,
                    "open": data.open,
                    "high": data.high,
                    "low": data.low,
                    "close": data.close,
                    "volume": data.volume,
                    "timeframe": data.timeframe,
                    "source": data.source,
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)

        return df

    def clear_cache(
        self, symbol: Optional[str] = None, timeframe: Optional[str] = None
    ):
        """Clear cache for specific symbol/timeframe or all."""
        if symbol is None:
            self.cache.clear()
            self.metadata.clear()
            self.last_update.clear()
        elif timeframe is None:
            if symbol in self.cache:
                del self.cache[symbol]
            for key in list(self.metadata.keys()):
                if key.startswith(f"{symbol}_"):
                    del self.metadata[key]
            if symbol in self.last_update:
                del self.last_update[symbol]
        else:
            if symbol in self.cache and timeframe in self.cache[symbol]:
                self.cache[symbol][timeframe].clear()
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.metadata:
                del self.metadata[cache_key]
            if symbol in self.last_update and timeframe in self.last_update[symbol]:
                del self.last_update[symbol][timeframe]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_data_points = sum(
            len(timeframe_data)
            for symbol_data in self.cache.values()
            for timeframe_data in symbol_data.values()
        )

        return {
            "total_symbols": len(self.cache),
            "total_data_points": total_data_points,
            "hit_rate": self.hit_count / max(self.total_requests, 1),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": self.total_requests,
            "cache_size": self.max_size,
        }


class DataProvider:
    """Base class for data providers."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize data provider.

        Args:
            api_key: API key for the provider
        """
        self.api_key = api_key
        self.session = None

    async def connect(self):
        """Connect to data provider (stub)."""
        import logging

        logging.getLogger(__name__).warning(
            "connect() not implemented for base DataProvider; override in subclass."
        )
        return None

    async def disconnect(self):
        """Disconnect from data provider."""
        if self.session:
            await self.session.close()

    async def subscribe(self, symbols: List[str], timeframes: List[str]):
        """Subscribe to data streams (stub)."""
        import logging

        logging.getLogger(__name__).warning(
            "subscribe() not implemented for base DataProvider; override in subclass."
        )
        return None

    async def get_historical_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> List[MarketData]:
        """Get historical data (stub)."""
        import logging

        logging.getLogger(__name__).warning(
            "get_historical_data() not implemented for base DataProvider; override in subclass."
        )
        return []


class PolygonDataProvider(DataProvider):
    """Polygon.io data provider with real WebSocket streaming."""

    def __init__(self, api_key: str):
        """Initialize Polygon provider.

        Args:
            api_key: Polygon API key
        """
        super().__init__(api_key)
        self.base_url = "wss://delayed.polygon.io"
        self.rest_url = "https://api.polygon.io"
        self.websocket = None
        self.is_connected = False

    async def connect(self):
        """Connect to Polygon websocket with retry logic."""
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                self.websocket = await websockets.connect(
                    f"{self.base_url}/stocks",
                    extra_headers={"Authorization": f"Bearer {self.api_key}"},
                    ping_interval=20,
                    ping_timeout=10,
                )
                self.is_connected = True
                logger.info("Connected to Polygon websocket")
                return
            except Exception as e:
                logger.warning(f"Polygon connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to connect to Polygon after {max_retries} attempts")
                    self.is_connected = False
                    raise

    async def subscribe(self, symbols: List[str], timeframes: List[str]):
        """Subscribe to Polygon data streams."""
        if not self.websocket or not self.is_connected:
            await self.connect()

        # Subscribe to trades (T) and aggregates (A) for real-time bars
        # Format: T.SYMBOL for trades, A.SYMBOL for aggregates
        trade_params = ",".join([f"T.{s}" for s in symbols])
        aggregate_params = ",".join([f"A.{s}" for s in symbols])
        
        subscribe_message = {
            "action": "subscribe",
            "params": f"{trade_params},{aggregate_params}",
        }

        await self.websocket.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to {len(symbols)} symbols on Polygon (trades and aggregates)")

    async def disconnect(self):
        """Disconnect from Polygon websocket."""
        if self.websocket:
            try:
                await self.websocket.close()
                self.is_connected = False
                logger.info("Disconnected from Polygon websocket")
            except Exception as e:
                logger.error(f"Error disconnecting from Polygon: {e}")

    async def get_historical_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> List[MarketData]:
        """Get historical data from Polygon."""
        try:
            # Convert timeframe to Polygon format
            timeframe_map = {
                "1m": "1",
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "1h": "60",
                "1d": "D",
                "1w": "W",
                "1M": "M",
            }

            polygon_timeframe = timeframe_map.get(timeframe, "1")

            # Build URL
            url = (
                f"{self.rest_url}/v2/aggs/ticker/{symbol}/range/{polygon_timeframe}/day/"
                f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            )

            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                url, params={"apiKey": self.api_key}
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    market_data = []
                    for bar in data.get("results", []):
                        market_data.append(
                            MarketData(
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(bar["t"] / 1000),
                                open=bar["o"],
                                high=bar["h"],
                                low=bar["l"],
                                close=bar["c"],
                                volume=bar["v"],
                                timeframe=timeframe,
                                source="polygon",
                            )
                        )

                    return market_data
                else:
                    logger.error(f"Error fetching historical data: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error getting historical data from Polygon: {e}")
            return []


class YFinanceDataProvider(DataProvider):
    """Yahoo Finance data provider."""

    def __init__(self):
        """Initialize YFinance provider."""
        super().__init__()

    async def connect(self):
        """Connect to YFinance (no connection needed)."""

    async def subscribe(self, symbols: List[str], timeframes: List[str]):
        """Subscribe to YFinance data (simulated)."""
        logger.info(f"Subscribed to {len(symbols)} symbols on YFinance")

    async def get_historical_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> List[MarketData]:
        """Get historical data from YFinance."""
        if not YFINANCE_AVAILABLE:
            logger.warning("YFinance not available")
            return []

        try:
            # Convert timeframe to YFinance format
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "1d": "1d",
                "1w": "1wk",
                "1M": "1mo",
            }

            interval = interval_map.get(timeframe, "1d")

            # Get data from YFinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            market_data = []
            for timestamp, row in df.iterrows():
                market_data.append(
                    MarketData(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        open=row["Open"],
                        high=row["High"],
                        low=row["Low"],
                        close=row["Close"],
                        volume=row["Volume"],
                        timeframe=timeframe,
                        source="yfinance",
                    )
                )

            return market_data

        except Exception as e:
            logger.error(f"Error getting historical data from YFinance: {e}")
            return []


class StreamingPipeline:
    """Main streaming data pipeline."""

    def __init__(self, config: StreamingConfig):
        """Initialize streaming pipeline.

        Args:
            config: Streaming configuration
        """
        self.config = config
        self.cache = InMemoryCache(config.cache_size)

        # Data providers
        self.providers = {}
        self.active_providers = []

        # Data triggers
        self.triggers = []
        self.trigger_lock = threading.Lock()

        # Streaming state
        self.is_running = False
        self.streaming_task = None

        # Callbacks
        self.data_callbacks = []
        self.error_callbacks = []

        # Statistics
        self.stats = {
            "messages_received": 0,
            "triggers_fired": 0,
            "errors": 0,
            "start_time": None,
        }

        logger.info(
            f"Initialized Streaming Pipeline with {len(config.symbols)} symbols"
        )

        # Removed return statement - __init__ should not return values

    def add_provider(self, name: str, provider: DataProvider):
        """Add data provider.

        Args:
            name: Provider name
            provider: Data provider instance
        """
        self.providers[name] = provider
        logger.info(f"Added data provider: {name}")

    def add_data_callback(self, callback: Callable[[MarketData], None]):
        """Add data callback.

        Args:
            callback: Function to call when new data arrives
        """
        self.data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add error callback.

        Args:
            callback: Function to call when error occurs
        """
        self.error_callbacks.append(callback)

    def add_trigger(self, trigger: DataTrigger):
        """Add data trigger.

        Args:
            trigger: Data trigger
        """
        with self.trigger_lock:
            self.triggers.append(trigger)
        logger.info(
            f"Added trigger: {trigger.symbol} {trigger.condition} {trigger.threshold}"
        )

    def remove_trigger(self, symbol: str, condition: str):
        """Remove data trigger.

        Args:
            symbol: Symbol
            condition: Trigger condition
        """
        with self.trigger_lock:
            self.triggers = [
                t
                for t in self.triggers
                if not (t.symbol == symbol and t.condition == condition)
            ]

    async def start_streaming(self):
        """Start streaming data."""
        if self.is_running:
            logger.warning("Streaming already running")
            return

        self.is_running = True
        self.stats["start_time"] = datetime.now()

        try:
            # Connect to providers
            for name, provider in self.providers.items():
                try:
                    await provider.connect()
                    await provider.subscribe(
                        self.config.symbols, self.config.timeframes
                    )
                    self.active_providers.append(name)
                    logger.info(f"Connected to provider: {name}")
                except Exception as e:
                    logger.error(f"Error connecting to provider {name}: {e}")
                    self.stats["errors"] += 1

            # Start streaming task
            self.streaming_task = asyncio.create_task(self._stream_data())

            logger.info("Started streaming pipeline")

        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.stats["errors"] += 1
            self.is_running = False
            raise

    async def stop_streaming(self):
        """Stop streaming data."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel streaming task
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass

        # Disconnect providers
        for name, provider in self.providers.items():
            try:
                await provider.disconnect()
                logger.info(f"Disconnected from provider: {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from provider {name}: {e}")

        self.active_providers.clear()
        logger.info("Stopped streaming pipeline")

    async def _stream_data(self):
        """Main streaming loop with real WebSocket connections."""
        # Start WebSocket listeners for each active provider
        websocket_tasks = []
        
        for name, provider in self.providers.items():
            if name in self.active_providers:
                # Create WebSocket listener task for each provider
                task = asyncio.create_task(
                    self._websocket_listener(name, provider)
                )
                websocket_tasks.append(task)
        
        # If no WebSocket providers, fall back to polling
        if not websocket_tasks:
            logger.warning("No WebSocket providers available, using polling mode")
            while self.is_running:
                try:
                    await self._poll_data_providers()
                    await self._process_triggers()
                    await asyncio.sleep(self.config.update_frequency)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in polling loop: {e}")
                    self.stats["errors"] += 1
                    await asyncio.sleep(self.config.update_frequency)
        else:
            # Wait for all WebSocket tasks
            try:
                await asyncio.gather(*websocket_tasks)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in WebSocket tasks: {e}")
                self.stats["errors"] += 1

    async def _websocket_listener(self, provider_name: str, provider: DataProvider):
        """Listen to WebSocket messages from a provider."""
        reconnect_delay = 1.0
        max_reconnect_delay = 60.0
        
        while self.is_running:
            try:
                # Check if provider supports WebSocket streaming
                if not hasattr(provider, 'websocket') or provider.websocket is None:
                    # Provider doesn't support WebSocket, skip
                    await asyncio.sleep(5)
                    continue
                
                # Listen for messages
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(provider.websocket.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f'WebSocket error: {e}')
                        break

                    if not self.is_running:
                        break
                    
                    try:
                        # Parse message based on provider type
                        market_data = await self._parse_websocket_message(
                            provider_name, message
                        )
                        
                        if market_data:
                            # Add to cache
                            self.cache.add_data(
                                market_data.symbol,
                                market_data.timeframe,
                                market_data
                            )
                            
                            # Call data callbacks
                            for callback in self.data_callbacks:
                                try:
                                    callback(market_data)
                                except Exception as e:
                                    logger.error(f"Error in data callback: {e}")
                            
                            # Update statistics
                            self.stats["messages_received"] += 1
                            
                            # Process triggers
                            await self._process_triggers()
                    
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        try:
                            await provider.websocket.ping()
                        except Exception as e:
                            logger.warning(f"Error in streaming pipeline: {e}")
                            break  # Connection lost, will reconnect
                        continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse WebSocket message: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        self.stats["errors"] += 1
                        continue
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"WebSocket connection closed for {provider_name}, reconnecting...")
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                
                # Attempt to reconnect
                try:
                    await provider.connect()
                    await provider.subscribe(self.config.symbols, self.config.timeframes)
                    reconnect_delay = 1.0  # Reset delay on successful reconnect
                    logger.info(f"Reconnected to {provider_name}")
                except Exception as e:
                    logger.error(f"Failed to reconnect to {provider_name}: {e}")
                    self.stats["errors"] += 1
            
            except Exception as e:
                logger.error(f"Error in WebSocket listener for {provider_name}: {e}")
                self.stats["errors"] += 1
                
                # Call error callbacks
                for callback in self.error_callbacks:
                    try:
                        callback(e)
                    except Exception as callback_error:
                        logger.error(f"Error in error callback: {callback_error}")
                
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _parse_websocket_message(self, provider_name: str, message: Any) -> Optional[MarketData]:
        """Parse WebSocket message into MarketData based on provider."""
        try:
            if provider_name == "polygon":
                return self._parse_polygon_message(message)
            elif provider_name == "finnhub":
                return self._parse_finnhub_message(message)
            elif provider_name == "alpaca":
                return self._parse_alpaca_message(message)
            else:
                # Generic JSON parsing
                if isinstance(message, str):
                    data = json.loads(message)
                else:
                    data = message
                
                # Try to extract market data from generic format
                if "symbol" in data and "price" in data:
                    return MarketData(
                        symbol=data["symbol"],
                        timestamp=datetime.fromtimestamp(
                            data.get("timestamp", datetime.now().timestamp())
                        ),
                        open=data.get("open", data["price"]),
                        high=data.get("high", data["price"]),
                        low=data.get("low", data["price"]),
                        close=data["price"],
                        volume=data.get("volume", 0),
                        timeframe=data.get("timeframe", "1m"),
                        source=provider_name,
                    )
                
                return None
        
        except Exception as e:
            logger.error(f"Error parsing {provider_name} message: {e}")
            return None

    def _parse_polygon_message(self, message: Any) -> Optional[MarketData]:
        """Parse Polygon.io WebSocket message."""
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message
            
            # Polygon message format: [{"ev": "T", "sym": "AAPL", "p": 150.0, "s": 100, "t": 1234567890}]
            if isinstance(data, list):
                data = data[0]
            
            event_type = data.get("ev")  # "T" for trade, "Q" for quote, "A" for aggregate
            
            if event_type == "T":  # Trade
                symbol = data.get("sym", "")
                price = float(data.get("p", 0))
                size = float(data.get("s", 0))
                timestamp_ms = data.get("t", 0)
                
                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=size,
                    timeframe="1m",
                    source="polygon",
                )
            
            elif event_type == "A":  # Aggregate (bar)
                symbol = data.get("sym", "")
                open_price = float(data.get("o", 0))
                high_price = float(data.get("h", 0))
                low_price = float(data.get("l", 0))
                close_price = float(data.get("c", 0))
                volume = float(data.get("v", 0))
                timestamp_ms = data.get("s", 0)  # Start time
                
                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timeframe="1m",
                    source="polygon",
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error parsing Polygon message: {e}")
            return None

    def _parse_finnhub_message(self, message: Any) -> Optional[MarketData]:
        """Parse Finnhub WebSocket message."""
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message
            
            # Finnhub message format: {"type": "trade", "data": [{"s": "AAPL", "p": 150.0, "v": 100, "t": 1234567890}]}
            if data.get("type") == "trade":
                trade_data = data.get("data", [])
                if trade_data:
                    trade = trade_data[0]
                    symbol = trade.get("s", "")
                    price = float(trade.get("p", 0))
                    volume = float(trade.get("v", 0))
                    timestamp_ms = trade.get("t", 0)
                    
                    return MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=volume,
                        timeframe="1m",
                        source="finnhub",
                    )
            
            return None
        
        except Exception as e:
            logger.error(f"Error parsing Finnhub message: {e}")
            return None

    def _parse_alpaca_message(self, message: Any) -> Optional[MarketData]:
        """Parse Alpaca WebSocket message."""
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message
            
            # Alpaca message format: [{"T": "t", "S": "AAPL", "p": 150.0, "s": 100, "t": "2023-01-01T00:00:00Z"}]
            if isinstance(data, list):
                data = data[0]
            
            msg_type = data.get("T")  # "t" for trade, "q" for quote, "b" for bar
            
            if msg_type == "t":  # Trade
                symbol = data.get("S", "")
                price = float(data.get("p", 0))
                size = float(data.get("s", 0))
                timestamp_str = data.get("t", "")
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except Exception as e:
                    logger.warning(f"Error in streaming pipeline: {e}")
                    timestamp = datetime.now()
                
                return MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=size,
                    timeframe="1m",
                    source="alpaca",
                )
            
            elif msg_type == "b":  # Bar
                symbol = data.get("S", "")
                open_price = float(data.get("o", 0))
                high_price = float(data.get("h", 0))
                low_price = float(data.get("l", 0))
                close_price = float(data.get("c", 0))
                volume = float(data.get("v", 0))
                timestamp_str = data.get("t", "")
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except Exception as e:
                    logger.warning(f"Error in streaming pipeline: {e}")
                    timestamp = datetime.now()
                
                return MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timeframe="1m",
                    source="alpaca",
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error parsing Alpaca message: {e}")
            return None

    async def _poll_data_providers(self):
        """Poll data providers when WebSocket is not available (fallback)."""
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                # Try to get latest data from providers
                for name, provider in self.providers.items():
                    if name in self.active_providers:
                        try:
                            # Get historical data and use latest as current
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=1)
                            
                            historical_data = await provider.get_historical_data(
                                symbol, timeframe, start_date, end_date
                            )
                            
                            if historical_data and len(historical_data) > 0:
                                # Use the latest data point
                                latest = historical_data[-1]
                                
                                # Add to cache
                                self.cache.add_data(symbol, timeframe, latest)
                                
                                # Call data callbacks
                                for callback in self.data_callbacks:
                                    try:
                                        callback(latest)
                                    except Exception as e:
                                        logger.error(f"Error in data callback: {e}")
                                
                                break  # Found data, move to next symbol/timeframe
                        
                        except Exception as e:
                            logger.debug(f"Error polling {name} for {symbol}: {e}")
                            continue

    async def _process_triggers(self):
        """Process data triggers."""
        with self.trigger_lock:
            active_triggers = [t for t in self.triggers if t.is_active]

        for trigger in active_triggers:
            try:
                # Get latest data
                latest_data = self.cache.get_latest_data(
                    trigger.symbol, trigger.timeframe
                )

                if not latest_data:
                    continue

                # Check trigger conditions
                if await self._check_trigger_condition(trigger, latest_data):
                    # Fire trigger
                    trigger.trigger_count += 1
                    trigger.last_triggered = datetime.now()

                    # Call trigger callback
                    try:
                        await trigger.callback(trigger, latest_data)
                        self.stats["triggers_fired"] += 1
                        logger.info(
                            f"Trigger fired: {trigger.symbol} {trigger.condition}"
                        )
                    except Exception as e:
                        logger.error(f"Error in trigger callback: {e}")

            except Exception as e:
                logger.error(f"Error processing trigger: {e}")

    async def _check_trigger_condition(
        self, trigger: DataTrigger, data: MarketData
    ) -> bool:
        """Check if trigger condition is met."""
        if trigger.condition == "price_above":
            return data.close > trigger.threshold
        elif trigger.condition == "price_below":
            return data.close < trigger.threshold
        elif trigger.condition == "volume_spike":
            # Compare with recent volume average
            recent_data = self.cache.get_data(
                trigger.symbol, trigger.timeframe, limit=20
            )
            if len(recent_data) >= 10:
                avg_volume = np.mean([d.volume for d in recent_data[-10:]])
                return data.volume > avg_volume * trigger.threshold
        elif trigger.condition == "volatility_spike":
            # Calculate recent volatility
            recent_data = self.cache.get_data(
                trigger.symbol, trigger.timeframe, limit=20
            )
            if len(recent_data) >= 10:
                returns = [
                    d.close / recent_data[i - 1].close - 1
                    for i, d in enumerate(recent_data[1:], 1)
                ]
                volatility = np.std(returns)
                return volatility > trigger.threshold

        return False

    def get_data(
        self, symbol: str, timeframe: str, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get data from cache as DataFrame."""
        return {
            "success": True,
            "result": self.cache.get_dataframe(symbol, timeframe, limit=limit),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def get_latest_data(self, symbol: str, timeframe: str) -> Optional[MarketData]:
        """Get latest data point."""
        return {
            "success": True,
            "result": self.cache.get_latest_data(symbol, timeframe),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        cache_stats = self.cache.get_cache_stats()

        return {
            **self.stats,
            **cache_stats,
            "active_providers": len(self.providers),
            "active_triggers": len([t for t in self.triggers if t.is_active]),
            "uptime": (
                (datetime.now() - self.stats["start_time"]).total_seconds()
                if self.stats["start_time"]
                else 0
            ),
        }


def create_streaming_pipeline(
    symbols: List[str],
    timeframes: List[str] = ["1m", "5m", "1h", "1d"],
    providers: List[str] = ["yfinance"],
    api_keys: Optional[Dict[str, str]] = None,
) -> StreamingPipeline:
    """Create streaming pipeline with default configuration.

    Args:
        symbols: List of symbols to stream
        timeframes: List of timeframes
        providers: List of data providers
        api_keys: API keys for providers

    Returns:
        Streaming pipeline instance
    """
    config = StreamingConfig(
        symbols=symbols, timeframes=timeframes, providers=providers
    )

    pipeline = StreamingPipeline(config)

    # Add providers
    api_keys = api_keys or {}

    if "polygon" in providers and "polygon" in api_keys:
        polygon_provider = PolygonDataProvider(api_keys["polygon"])
        pipeline.add_provider("polygon", polygon_provider)

    if "yfinance" in providers:
        yfinance_provider = YFinanceDataProvider()
        pipeline.add_provider("yfinance", yfinance_provider)

    return pipeline
