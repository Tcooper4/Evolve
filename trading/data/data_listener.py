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
              binance: bool = True, polygon: bool = False, news: bool = True):
        """Start all listeners in threads."""
        if binance:
            threading.Thread(target=self._run_binance_ws, args=(symbols,), daemon=True).start()
        if polygon:
            threading.Thread(target=self._run_polygon_ws, args=(symbols,), daemon=True).start()
        if news:
            threading.Thread(target=self._run_news_listener, args=(news_keywords,), daemon=True).start()

    def stop(self):
        self._stop_event.set()

    def _run_binance_ws(self, symbols: List[str]):
        if not websockets:
            logger.warning("websockets package not available for Binance streaming.")
            return
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join([s.lower()+'usdt@trade' for s in symbols])}"
        asyncio.run(self._binance_ws_loop(url))

    async def _binance_ws_loop(self, url: str):
        async with websockets.connect(url) as ws:
            while not self._stop_event.is_set():
                msg = await ws.recv()
                data = json.loads(msg)
                price = float(data['data']['p'])
                self._handle_price(price)

    def _run_polygon_ws(self, symbols: List[str]):
        # Placeholder: Polygon.io WebSocket implementation
        logger.info("Polygon WebSocket streaming not implemented in this template.")

    def _handle_price(self, price: float):
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

    def _run_news_listener(self, keywords: List[str]):
        while not self._stop_event.is_set():
            news = self._fetch_news(keywords)
            for item in news:
                if self.on_news:
                    self.on_news(item)
                # Pause trading on significant news
                if self._is_significant_news(item):
                    self.paused = True
                    logger.warning(f"Significant news detected: {item.get('title')}. Pausing trading.")
            time.sleep(60)  # Poll every minute

    def _fetch_news(self, keywords: List[str]) -> List[Dict[str, Any]]:
        # Try Yahoo Finance
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={'%20'.join(keywords)}"
            resp = requests.get(url)
            if resp.status_code == 200:
                items = resp.json().get('news', [])
                return [{'title': n.get('title'), 'summary': n.get('summary', ''), 'source': 'yahoo'} for n in items]
        except Exception as e:
            logger.warning(f"Yahoo news fetch failed: {e}")
        # Try FinBERT (placeholder)
        # Try OpenAI + web search fallback (placeholder)
        return []

    def _is_significant_news(self, news_item: Dict[str, Any]) -> bool:
        # Placeholder: Use FinBERT or OpenAI to classify news as significant
        # For now, flag if certain keywords are present
        text = (news_item.get('title', '') + ' ' + news_item.get('summary', '')).lower()
        for word in ['crash', 'bankruptcy', 'fraud', 'investigation', 'spike', 'collapse']:
            if word in text:
                return True
        return False 