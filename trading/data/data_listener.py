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

import requests

try:
    import websockets
except ImportError:
    websockets = None

logger = logging.getLogger(__name__)

class DataListener:
    def __init__(self, 
                 on_price: Optional[Callable[[Dict[str, Any]], None]] = None,
                 on_news: Optional[Callable[[Dict[str, Any]], None]] = None,
                 volatility_threshold: float = 0.05):
        """
        Args:
            on_price: Callback for price data
            on_news: Callback for news data
            volatility_threshold: Pause trading if volatility exceeds this (e.g., 0.05 = 5%)
        """
        self.on_price = on_price
        self.on_news = on_news
        self.volatility_threshold = volatility_threshold
        self.paused = False
        self.last_prices: List[float] = []
        self.price_window = 20  # Number of prices to track for volatility
        self._stop_event = threading.Event()

    def start(self, symbols: List[str], news_keywords: List[str],
              binance: bool = True, polygon: bool = False, news: bool = True) -> Dict[str, Any]:
        """Start all listeners in threads.
        
        Returns:
            Dictionary with start status and details
        """
        try:
            started_listeners = []
            
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
            return {
                'success': True,
                'message': 'Data listeners stopped successfully',
                'paused': self.paused,
                'price_count': len(self.last_prices),
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
                    msg = await ws.recv()
                    data = json.loads(msg)
                    price = float(data['data']['p'])
                    self._handle_price(price)
            
            return {'success': True, 'message': 'Binance WebSocket loop completed', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error in Binance WebSocket loop: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    def _run_polygon_ws(self, symbols: List[str]):
        """Run Polygon WebSocket listener."""
        try:
            # Placeholder: Polygon.io WebSocket implementation
            logger.info("Polygon WebSocket streaming not implemented in this template.")
            
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
                    for item in news.get('news_items', []):
                        if self.on_news:
                            self.on_news(item)
                        
                        # Pause trading on significant news
                        if self._is_significant_news(item):
                            self.paused = True
                            logger.warning(f"Significant news detected: {item.get('title')}. Pausing trading.")
                
                time.sleep(60)  # Poll every minute
            
            return {'success': True, 'message': 'News listener completed', 'timestamp': time.time()}
            
        except Exception as e:
            logger.error(f"Error in news listener: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

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
                'timestamp': time.time()
            }
            
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