"""
DataListener: Real-time data streaming for agent loop.
- Streams price data from Binance/Polygon WebSocket
- Streams news from Yahoo, FinBERT, or OpenAI+web search fallback
- Can pause trading on volatility spikes or news events
"""

import asyncio
import json
import logging
import threading
from typing import Callable, Optional, Dict, Any, List
import time
from datetime import datetime, timedelta
from collections import deque
import statistics

import requests

try:
    import websockets
except ImportError:
    websockets = None

logger = logging.getLogger(__name__)

class DataFeedWatchdog:
    """Monitors data feeds for stalls and timestamp gaps."""
    
    def __init__(self, 
                 stall_threshold_seconds: float = 30.0,
                 gap_threshold_seconds: float = 5.0,
                 max_gap_count: int = 10,
                 alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        """Initialize the watchdog.
        
        Args:
            stall_threshold_seconds: Maximum time without data before considering stalled
            gap_threshold_seconds: Maximum gap between timestamps before alerting
            max_gap_count: Maximum number of gaps before considering feed unhealthy
            alert_callback: Callback function for alerts
        """
        self.stall_threshold = stall_threshold_seconds
        self.gap_threshold = gap_threshold_seconds
        self.max_gap_count = max_gap_count
        self.alert_callback = alert_callback
        
        # Monitoring state
        self.last_data_times: Dict[str, float] = {}
        self.gap_counts: Dict[str, int] = {}
        self.feed_health: Dict[str, str] = {}  # 'healthy', 'warning', 'stalled'
        self.timestamp_history: Dict[str, deque] = {}
        self.max_history_size = 100
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'stall_alerts': 0,
            'gap_alerts': 0,
            'recovery_events': 0
        }
    
    def register_feed(self, feed_name: str):
        """Register a data feed for monitoring.
        
        Args:
            feed_name: Name of the feed to monitor
        """
        self.last_data_times[feed_name] = time.time()
        self.gap_counts[feed_name] = 0
        self.feed_health[feed_name] = 'healthy'
        self.timestamp_history[feed_name] = deque(maxlen=self.max_history_size)
        logger.info(f"Registered feed for monitoring: {feed_name}")
    
    def update_feed(self, feed_name: str, timestamp: Optional[float] = None):
        """Update feed with new data.
        
        Args:
            feed_name: Name of the feed
            timestamp: Optional timestamp (uses current time if None)
        """
        current_time = time.time()
        self.last_data_times[feed_name] = current_time
        
        if timestamp is not None:
            self.timestamp_history[feed_name].append(timestamp)
            
            # Check for timestamp gaps
            if len(self.timestamp_history[feed_name]) > 1:
                gaps = []
                timestamps = list(self.timestamp_history[feed_name])
                for i in range(1, len(timestamps)):
                    gap = timestamps[i] - timestamps[i-1]
                    if gap > self.gap_threshold:
                        gaps.append(gap)
                
                if gaps:
                    self.gap_counts[feed_name] += len(gaps)
                    if self.gap_counts[feed_name] > self.max_gap_count:
                        self._alert_gap(feed_name, gaps)
        
        # Check if feed recovered from stall
        if self.feed_health.get(feed_name) == 'stalled':
            self.feed_health[feed_name] = 'healthy'
            self.stats['recovery_events'] += 1
            logger.info(f"Feed {feed_name} recovered from stall")
    
    def check_feeds(self) -> Dict[str, Any]:
        """Check all feeds for stalls and issues.
        
        Returns:
            Dictionary with feed status information
        """
        current_time = time.time()
        status = {
            'timestamp': current_time,
            'feeds': {},
            'alerts': [],
            'overall_health': 'healthy'
        }
        
        for feed_name in self.last_data_times:
            time_since_last = current_time - self.last_data_times[feed_name]
            
            feed_status = {
                'name': feed_name,
                'last_update': self.last_data_times[feed_name],
                'time_since_last': time_since_last,
                'gap_count': self.gap_counts[feed_name],
                'health': self.feed_health[feed_name],
                'data_points': len(self.timestamp_history[feed_name])
            }
            
            # Check for stalls
            if time_since_last > self.stall_threshold:
                if self.feed_health[feed_name] != 'stalled':
                    self._alert_stall(feed_name, time_since_last)
                feed_status['health'] = 'stalled'
                status['overall_health'] = 'warning'
            
            # Check for excessive gaps
            elif self.gap_counts[feed_name] > self.max_gap_count:
                feed_status['health'] = 'warning'
                status['overall_health'] = 'warning'
            
            status['feeds'][feed_name] = feed_status
        
        return status
    
    def _alert_stall(self, feed_name: str, stall_duration: float):
        """Alert about a stalled feed.
        
        Args:
            feed_name: Name of the stalled feed
            stall_duration: Duration of the stall in seconds
        """
        alert = {
            'type': 'stall',
            'feed_name': feed_name,
            'duration': stall_duration,
            'threshold': self.stall_threshold,
            'timestamp': time.time(),
            'message': f"Data feed '{feed_name}' stalled for {stall_duration:.1f} seconds"
        }
        
        self.feed_health[feed_name] = 'stalled'
        self.stats['stall_alerts'] += 1
        self.stats['total_alerts'] += 1
        
        logger.warning(alert['message'])
        
        if self.alert_callback:
            self.alert_callback('stall', alert)
    
    def _alert_gap(self, feed_name: str, gaps: List[float]):
        """Alert about timestamp gaps.
        
        Args:
            feed_name: Name of the feed with gaps
            gaps: List of gap durations
        """
        avg_gap = statistics.mean(gaps)
        max_gap = max(gaps)
        
        alert = {
            'type': 'gap',
            'feed_name': feed_name,
            'gap_count': len(gaps),
            'avg_gap': avg_gap,
            'max_gap': max_gap,
            'threshold': self.gap_threshold,
            'timestamp': time.time(),
            'message': f"Data feed '{feed_name}' has {len(gaps)} timestamp gaps (avg: {avg_gap:.1f}s, max: {max_gap:.1f}s)"
        }
        
        self.stats['gap_alerts'] += 1
        self.stats['total_alerts'] += 1
        
        logger.warning(alert['message'])
        
        if self.alert_callback:
            self.alert_callback('gap', alert)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get watchdog statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'stats': self.stats.copy(),
            'feed_count': len(self.last_data_times),
            'healthy_feeds': sum(1 for health in self.feed_health.values() if health == 'healthy'),
            'warning_feeds': sum(1 for health in self.feed_health.values() if health == 'warning'),
            'stalled_feeds': sum(1 for health in self.feed_health.values() if health == 'stalled')
        }
    
    def reset_statistics(self):
        """Reset watchdog statistics."""
        self.stats = {
            'total_alerts': 0,
            'stall_alerts': 0,
            'gap_alerts': 0,
            'recovery_events': 0
        }
        logger.info("Watchdog statistics reset")

class DataListener:
    def __init__(self, 
                 on_price: Optional[Callable[[Dict[str, Any]], None]] = None,
                 on_news: Optional[Callable[[Dict[str, Any]], None]] = None,
                 volatility_threshold: float = 0.05,
                 enable_watchdog: bool = True,
                 watchdog_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            on_price: Callback for price data
            on_news: Callback for news data
            volatility_threshold: Pause trading if volatility exceeds this (e.g., 0.05 = 5%)
            enable_watchdog: Whether to enable data feed monitoring
            watchdog_config: Configuration for the watchdog
        """
        self.on_price = on_price
        self.on_news = on_news
        self.volatility_threshold = volatility_threshold
        self.paused = False
        self.last_prices: List[float] = []
        self.price_window = 20  # Number of prices to track for volatility
        self._stop_event = threading.Event()
        
        # Watchdog setup
        self.enable_watchdog = enable_watchdog
        if enable_watchdog:
            watchdog_config = watchdog_config or {}
            self.watchdog = DataFeedWatchdog(
                stall_threshold_seconds=watchdog_config.get('stall_threshold_seconds', 30.0),
                gap_threshold_seconds=watchdog_config.get('gap_threshold_seconds', 5.0),
                max_gap_count=watchdog_config.get('max_gap_count', 10),
                alert_callback=watchdog_config.get('alert_callback')
            )
            self._start_watchdog_monitoring()
        else:
            self.watchdog = None
    
    def _start_watchdog_monitoring(self):
        """Start the watchdog monitoring thread."""
        if self.watchdog:
            def watchdog_worker():
                while not self._stop_event.is_set():
                    try:
                        status = self.watchdog.check_feeds()
                        if status['overall_health'] != 'healthy':
                            logger.warning(f"Data feed health issues detected: {status['overall_health']}")
                        time.sleep(5)  # Check every 5 seconds
                    except Exception as e:
                        logger.error(f"Error in watchdog monitoring: {e}")
                        time.sleep(10)  # Wait longer on error
            
            threading.Thread(target=watchdog_worker, daemon=True).start()
            logger.info("Started data feed watchdog monitoring")

    def start(self, symbols: List[str], news_keywords: List[str],
              binance: bool = True, polygon: bool = False, news: bool = True) -> Dict[str, Any]:
        """Start all listeners in threads.
        
        Returns:
            Dictionary with start status and details
        """
        try:
            started_listeners = []
            
            # Register feeds with watchdog
            if self.watchdog:
                if binance:
                    self.watchdog.register_feed('binance_price')
                if polygon:
                    self.watchdog.register_feed('polygon_price')
                if news:
                    self.watchdog.register_feed('news_feed')
            
            if binance:
                threading.Thread(target=self._run_binance_ws, args=(symbols,), daemon=True).start()
                started_listeners.append('binance')
                
            if polygon:
                threading.Thread(target=self._run_polygon_ws, args=(symbols,), daemon=True).start()
                started_listeners.append('polygon')
                
            if news:
                threading.Thread(target=self._run_news_listener, args=(news_keywords,), daemon=True).start()
                started_listeners.append('news')
            
            return {
                'success': True,
                'message': f'Data listeners started: {", ".join(started_listeners)}',
                'started_listeners': started_listeners,
                'symbols': symbols,
                'news_keywords': news_keywords,
                'watchdog_enabled': self.enable_watchdog,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error starting data listeners: {e}")
            return {
                'success': False,
                'message': f'Error starting data listeners: {str(e)}',
                'symbols': symbols,
                'news_keywords': news_keywords,
                'timestamp': time.time()
            }

    def stop(self) -> Dict[str, Any]:
        """Stop all data listeners.
        
        Returns:
            Dictionary with stop status and details
        """
        try:
            self._stop_event.set()
            
            # Get watchdog statistics if available
            watchdog_stats = None
            if self.watchdog:
                watchdog_stats = self.watchdog.get_statistics()
            
            return {
                'success': True,
                'message': 'Data listeners stopped successfully',
                'paused': self.paused,
                'price_count': len(self.last_prices),
                'watchdog_stats': watchdog_stats,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error stopping data listeners: {e}")
            return {
                'success': False,
                'message': f'Error stopping data listeners: {str(e)}',
                'timestamp': time.time()
            }

    def _run_binance_ws(self, symbols: List[str]):
        """Run Binance WebSocket listener."""
        try:
            if not websockets:
                logger.warning("websockets package not available for Binance streaming.")
                return {'success': False, 'error': 'websockets package not available', 'timestamp': time.time()}
            
            url = f"wss://stream.binance.com:9443/stream?streams={'/'.join([s.lower()+'usdt@trade' for s in symbols])}"
            asyncio.run(self._binance_ws_loop(url))
            
            return {'success': True, 'message': 'Binance WebSocket listener started', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error in Binance WebSocket: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    async def _binance_ws_loop(self, url: str):
        """Binance WebSocket event loop."""
        try:
            async with websockets.connect(url) as ws:
                while not self._stop_event.is_set():
                    try:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        price = float(data['data']['p'])
                        timestamp = data['data'].get('T', time.time() * 1000) / 1000  # Convert to seconds
                        
                        # Update watchdog
                        if self.watchdog:
                            self.watchdog.update_feed('binance_price', timestamp)
                        
                        self._handle_price(price)
                    except ConnectionError:
                        logger.warning("Data stream disconnected")
                        # Attempt to reconnect
                        await asyncio.sleep(5)
                        break
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue
            
            return {'success': True, 'message': 'Binance WebSocket loop completed', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error in Binance WebSocket loop: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    def _run_polygon_ws(self, symbols: List[str]):
        """Run Polygon WebSocket listener."""
        try:
            # Placeholder: Polygon.io WebSocket implementation
            logger.info("Polygon WebSocket streaming not implemented in this template.")
            
            # Simulate data updates for watchdog testing
            if self.watchdog:
                while not self._stop_event.is_set():
                    self.watchdog.update_feed('polygon_price', time.time())
                    time.sleep(1)
            
            return {'success': True, 'message': 'Polygon WebSocket listener placeholder', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error in Polygon WebSocket: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    def _handle_price(self, price: float):
        """Handle incoming price data."""
        try:
            self.last_prices.append(price)
            if len(self.last_prices) > self.price_window:
                self.last_prices.pop(0)
            
            if self.on_price:
                self.on_price({'price': price, 'paused': self.paused})
            
            # Volatility check
            if len(self.last_prices) == self.price_window:
                returns = [self.last_prices[i+1]/self.last_prices[i]-1 for i in range(self.price_window-1)]
                volatility = (sum((r)**2 for r in returns)/len(returns))**0.5
                
                if volatility > self.volatility_threshold:
                    self.paused = True
                    logger.warning(f"Volatility spike detected: {volatility:.4f}. Pausing trading.")
                else:
                    self.paused = False
            
            return {'success': True, 'message': 'Price handled successfully', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error handling price: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    def _run_news_listener(self, keywords: List[str]):
        """Run news listener."""
        try:
            while not self._stop_event.is_set():
                news = self._fetch_news(keywords)
                
                if news.get('success'):
                    # Update watchdog
                    if self.watchdog:
                        self.watchdog.update_feed('news_feed', time.time())
                    
                    for item in news.get('news_items', []):
                        if self.on_news:
                            self.on_news(item)
                        
                        # Pause trading on significant news
                        if self._is_significant_news(item):
                            self.paused = True
                            logger.warning(f"Significant news detected: {item.get('title', 'Unknown')}. Pausing trading.")
                
                time.sleep(60)  # Check news every minute
            
            return {'success': True, 'message': 'News listener completed', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error in news listener: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}
    
    def get_watchdog_status(self) -> Optional[Dict[str, Any]]:
        """Get watchdog status and statistics.
        
        Returns:
            Watchdog status dictionary or None if watchdog is disabled
        """
        if not self.watchdog:
            return None
        
        return {
            'enabled': True,
            'status': self.watchdog.check_feeds(),
            'statistics': self.watchdog.get_statistics()
        }
    
    def reset_watchdog_statistics(self):
        """Reset watchdog statistics."""
        if self.watchdog:
            self.watchdog.reset_statistics()
            logger.info("Watchdog statistics reset")
        else:
            logger.warning("Watchdog is not enabled")
    
    def _fetch_news(self, keywords: List[str]) -> Dict[str, Any]:
        """Fetch news data from various sources.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Dictionary with fetch status and news data
        """
        try:
            news_items = []
            
            # Try Yahoo Finance
            try:
                url = f"https://query1.finance.yahoo.com/v1/finance/search?q={'%20'.join(keywords)}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    items = resp.json().get('news', [])
                    news_items = [{'title': n.get('title'), 'summary': n.get('summary', ''), 'source': 'yahoo'} for n in items]
                    
                    return {
                        'success': True,
                        'message': f'Successfully fetched {len(news_items)} news items from Yahoo',
                        'news_items': news_items,
                        'source': 'yahoo',
                        'keywords': keywords,
                        'timestamp': time.time()
                    }
            except Exception as e:
                logger.warning(f"Yahoo news fetch failed: {e}")
            
            # Try FinBERT (placeholder)
            # Try OpenAI + web search fallback (placeholder)
            
            return {
                'success': False,
                'message': 'All news sources failed',
                'news_items': [],
                'keywords': keywords,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return {
                'success': False,
                'error': str(e),
                'news_items': [],
                'keywords': keywords,
                'timestamp': time.time()
            }

    def _is_significant_news(self, news_item: Dict[str, Any]) -> bool:
        """Check if news is significant enough to pause trading.
        
        Args:
            news_item: News item to check
            
        Returns:
            True if news is significant
        """
        try:
            # Placeholder: Use FinBERT or OpenAI to classify news as significant
            # For now, flag if certain keywords are present
            significant_keywords = [
                'earnings', 'revenue', 'profit', 'loss', 'bankruptcy', 'merger', 'acquisition',
                'ceo', 'cfo', 'resignation', 'federal reserve', 'interest rate', 'inflation',
                'recession', 'crisis', 'scandal', 'lawsuit', 'regulation', 'ban'
            ]
            
            title = news_item.get('title', '').lower()
            summary = news_item.get('summary', '').lower()
            
            for keyword in significant_keywords:
                if keyword in title or keyword in summary:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking news significance: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current status of data listener.
        
        Returns:
            Dictionary with status information
        """
        try:
            status = {
                'paused': self.paused,
                'price_count': len(self.last_prices),
                'volatility_threshold': self.volatility_threshold,
                'stopped': self._stop_event.is_set(),
                'watchdog_enabled': self.enable_watchdog,
                'timestamp': time.time()
            }
            
            # Add watchdog status if enabled
            if self.watchdog:
                status['watchdog_status'] = self.get_watchdog_status()
            
            # Calculate current volatility if we have enough prices
            if len(self.last_prices) >= 2:
                returns = [self.last_prices[i+1]/self.last_prices[i]-1 for i in range(len(self.last_prices)-1)]
                current_volatility = (sum((r)**2 for r in returns)/len(returns))**0.5
                status['current_volatility'] = current_volatility
                status['volatility_exceeded'] = current_volatility > self.volatility_threshold
            
            return {'success': True, 'result': status, 'message': 'Status retrieved successfully', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    def set_volatility_threshold(self, threshold: float) -> Dict[str, Any]:
        """Set volatility threshold.
        
        Args:
            threshold: New volatility threshold
            
        Returns:
            Dictionary with update status
        """
        try:
            if threshold <= 0:
                return {'success': False, 'error': 'Volatility threshold must be positive', 'timestamp': time.time()}
            
            self.volatility_threshold = threshold
            logger.info(f"Volatility threshold updated to {threshold}")
            
            return {'success': True, 'message': f'Volatility threshold updated to {threshold}', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error setting volatility threshold: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    def clear_price_history(self) -> Dict[str, Any]:
        """Clear price history.
        
        Returns:
            Dictionary with clear status
        """
        try:
            self.last_prices.clear()
            logger.info("Price history cleared")
            
            return {'success': True, 'message': 'Price history cleared', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error clearing price history: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    def resume_trading(self) -> Dict[str, Any]:
        """Resume trading after pause.
        
        Returns:
            Dictionary with resume status
        """
        try:
            self.paused = False
            logger.info("Trading resumed")
            
            return {'success': True, 'message': 'Trading resumed', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error resuming trading: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}


class RealTimeDataFeed:
    """Real-time data feed for streaming market data.
    
    This is a compatibility wrapper around DataListener for backward compatibility.
    """
    
    def __init__(self, symbols: List[str] = None, **kwargs):
        """Initialize real-time data feed.
        
        Args:
            symbols: List of symbols to monitor
            **kwargs: Additional arguments passed to DataListener
        """
        self.symbols = symbols or ['BTC', 'ETH', 'AAPL', 'MSFT']
        self.data_listener = DataListener(**kwargs)
        self.is_running = False
        
    def start(self, **kwargs) -> Dict[str, Any]:
        """Start the real-time data feed.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with start status
        """
        try:
            result = self.data_listener.start(
                symbols=self.symbols,
                news_keywords=kwargs.get('news_keywords', ['crypto', 'stock', 'market']),
                binance=kwargs.get('binance', True),
                polygon=kwargs.get('polygon', False),
                news=kwargs.get('news', True)
            )
            
            if result.get('success'):
                self.is_running = True
                
            return result
            
        except Exception as e:
            logger.error(f"Error starting real-time data feed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def stop(self) -> Dict[str, Any]:
        """Stop the real-time data feed.
        
        Returns:
            Dictionary with stop status
        """
        try:
            result = self.data_listener.stop()
            if result.get('success'):
                self.is_running = False
            return result
            
        except Exception as e:
            logger.error(f"Error stopping real-time data feed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the real-time data feed.
        
        Returns:
            Dictionary with status information
        """
        try:
            status = self.data_listener.get_status()
            if status.get('success'):
                status['result']['is_running'] = self.is_running
                status['result']['symbols'] = self.symbols
            return status
            
        except Exception as e:
            logger.error(f"Error getting real-time data feed status: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }


class MarketDataStream:
    """Market data stream for high-frequency data processing.
    
    This is a compatibility wrapper around DataListener for backward compatibility.
    """
    
    def __init__(self, symbols: List[str] = None, **kwargs):
        """Initialize market data stream.
        
        Args:
            symbols: List of symbols to stream
            **kwargs: Additional arguments passed to DataListener
        """
        self.symbols = symbols or ['BTC', 'ETH', 'AAPL', 'MSFT']
        self.data_listener = DataListener(**kwargs)
        self.is_active = False
        
    def connect(self, **kwargs) -> Dict[str, Any]:
        """Connect to market data stream.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with connection status
        """
        try:
            result = self.data_listener.start(
                symbols=self.symbols,
                news_keywords=kwargs.get('news_keywords', ['market', 'trading']),
                binance=kwargs.get('binance', True),
                polygon=kwargs.get('polygon', False),
                news=kwargs.get('news', False)
            )
            
            if result.get('success'):
                self.is_active = True
                
            return result
            
        except Exception as e:
            logger.error(f"Error connecting to market data stream: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def disconnect(self) -> Dict[str, Any]:
        """Disconnect from market data stream.
        
        Returns:
            Dictionary with disconnection status
        """
        try:
            result = self.data_listener.stop()
            if result.get('success'):
                self.is_active = False
            return result
            
        except Exception as e:
            logger.error(f"Error disconnecting from market data stream: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def is_connected(self) -> bool:
        """Check if connected to market data stream.
        
        Returns:
            True if connected
        """
        return self.is_active
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get information about the market data stream.
        
        Returns:
            Dictionary with stream information
        """
        try:
            status = self.data_listener.get_status()
            if status.get('success'):
                info = status['result']
                info['symbols'] = self.symbols
                info['is_connected'] = self.is_active
                return {
                    'success': True,
                    'result': info,
                    'timestamp': time.time()
                }
            return status
            
        except Exception as e:
            logger.error(f"Error getting market data stream info: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            } 