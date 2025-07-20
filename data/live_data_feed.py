Enhanced Live Data Feed with Multiple API Support and UI Integration.

This module provides a robust data feed with automatic failover between
multiple data providers: Alpaca â†’ Polygon â†’ Finnhub â†’ yfinance (fallback).
Includes UI integration for refresh controls and real-time data updates.
"

import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
import asyncio
import threading
import time

import numpy as np
import pandas as pd
import requests

# Try to import optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available. Install with: pip install yfinance")

try:
    from alpaca.data import StockHistoricalDataClient, StockBarsRequest
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("alpaca-py not available. Install with: pip install alpaca-py")

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DataProvider:
    """Base class for data providers."""
    def __init__(self, name: str, api_key: str = None):
        """Initialize data provider."""
        self.name = name
        self.api_key = api_key
        self.is_available = True
        self.last_check = None
        self.error_count = 0
        self.max_errors = 5
        self.last_successful_request = None
        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def check_availability(self) -> bool:
        """Check if provider is available."""
        try:
            result = self._health_check()
            if result:
                self.last_check = datetime.now()
                self.last_successful_request = datetime.now()
                self.error_count = 0
            return result
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.is_available = False
            return False

    def _health_check(self) -> bool:
        """Placeholder for health check in subclasses."""
        return False

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from provider."""
        return None

    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from provider."""
        return None

    def get_ohlcv(self, symbol: str, period: str = "1, interval: str =1d") -> Optional[pd.DataFrame]:
        """Get OHLCV data with flexible period/interval."""
        return None

    def get_provider_status(self) -> Dict[str, Any]:
        """Get provider status information."""
        return {
            "name": self.name,
            "available": self.is_available,
            "error_count": self.error_count,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_successful_request": self.last_successful_request.isoformat()
            if self.last_successful_request
            else None,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset": self.rate_limit_reset,
        }


class AlpacaProvider(DataProvider):
    """Alpaca data provider with real-time capabilities."""
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        """Initialize Alpaca provider."""
        super().__init__("Alpaca", api_key or os.getenv("ALPACA_API_KEY"))
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper = paper
        
        if not ALPACA_AVAILABLE:
            self.is_available = False
            logger.error("Alpaca provider not available - alpaca-py not installed")
            return
            
        try:
            self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self.is_available = False

    def _health_check(self) -> bool:
        """Alpaca API health."""
        try:
            if not ALPACA_AVAILABLE:
                return False
            # Try to get account info
            account = self.trading_client.get_account()
            return account is not None
        except Exception:
            return False

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from Alpaca."""
        try:
            if not ALPACA_AVAILABLE:
                return None
                
            # Convert interval to Alpaca format
            interval_map = {
                "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
                "1h": "1Hour", "1d": "1Day"
            }
            alpaca_interval = interval_map.get(interval, "1Day")
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_interval,
                start=pd.Timestamp(start_date),
                end=pd.Timestamp(end_date)
            )
            
            bars = self.data_client.get_stock_bars(request)
            if not bars.data:
                return None
                
            # Convert to DataFrame
            df = bars.df.reset_index()
            df = df.rename(columns={
                'timestamp': 'timestamp',
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'volume': 'Volume', 'vwap': 'VWAP'
            })
            
            self.last_successful_request = datetime.now()
            return df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]
            
        except Exception as e:
            logger.error(f"Alpaca historical data error: {e}")
            return None

    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from Alpaca."""
        try:
            if not ALPACA_AVAILABLE:
                return None
                
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(request)
            
            if not quote.data:
                return None
                
            latest_quote = quote.data[symbol]
            live_data = {
                "symbol": symbol,
                "price": latest_quote.ask_price or latest_quote.bid_price,
                "bid": latest_quote.bid_price,
                "ask": latest_quote.ask_price,
                "volume": latest_quote.bid_size + latest_quote.ask_size,
                "timestamp": datetime.now().isoformat(),
                "provider": "Alpaca"
            }
            
            self.last_successful_request = datetime.now()
            return live_data
            
        except Exception as e:
            logger.error(f"Alpaca live data error: {e}")
            return None

    def get_ohlcv(self, symbol: str, period: str = "1d", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get OHLCV data with flexible period/interval."""
        try:
            end_date = datetime.now()
            
            # Calculate start date based on period
            period_map = {
                "1d": timedelta(days=1), "5d": timedelta(days=5), "1mo": timedelta(days=30),
                "3mo": timedelta(days=90), "6mo": timedelta(days=180), "1y": timedelta(days=365),
                "2y": timedelta(days=730), "5y": timedelta(days=1825), "10y": timedelta(days=3650),
                "ytd": timedelta(days=datetime.now().timetuple().tm_yday), "max": timedelta(days=3650)
            }
            
            start_date = end_date - period_map.get(period, timedelta(days=1))
            
            return self.get_historical_data(
                symbol, 
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d"), 
                interval
            )
            
        except Exception as e:
            logger.error(f"Alpaca OHLCV error: {e}")
            return None


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider as fallback."""
    def __init__(self):
        """Initialize YFinance provider."""
        super().__init__("YFinance")
        if not YFINANCE_AVAILABLE:
            self.is_available = False
            logger.error("YFinance provider not available - yfinance not installed")

    def _health_check(self) -> bool:
        """YFinance availability."""
        try:
            if not YFINANCE_AVAILABLE:
                return False
            # Try to get a simple quote
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            return info is not None and len(info) > 0
        except Exception:
            return False

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from YFinance."""
        try:
            if not YFINANCE_AVAILABLE:
                return None
                
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                return None
                
            # Standardize column names
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'timestamp', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                'Close': 'Close', 'Volume': 'Volume'
            })
            
            # Add VWAP if not present
            if 'VWAP' not in df.columns:
                df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3
                
            self.last_successful_request = datetime.now()
            return df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]
            
        except Exception as e:
            logger.error(f"YFinance historical data error: {e}")
            return None

    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from YFinance."""
        try:
            if not YFINANCE_AVAILABLE:
                return None
                
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            live_data = {
                "symbol": symbol,
                "price": info.get('regularMarketPrice', 0),
                "bid": info.get('bid', 0),
                "ask": info.get('ask', 0),
                "volume": info.get('volume', 0),
                "timestamp": datetime.now().isoformat(),
                "provider": "YFinance"
            }
            
            self.last_successful_request = datetime.now()
            return live_data
            
        except Exception as e:
            logger.error(f"YFinance live data error: {e}")
            return None

    def get_ohlcv(self, symbol: str, period: str = "1d", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get OHLCV data with flexible period/interval."""
        try:
            if not YFINANCE_AVAILABLE:
                return None
                
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
                
            # Standardize column names
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'timestamp', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                'Close': 'Close', 'Volume': 'Volume'
            })
            
            # Add VWAP if not present
            if 'VWAP' not in df.columns:
                df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3
                
            self.last_successful_request = datetime.now()
            return df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]
            
        except Exception as e:
            logger.error(f"YFinance OHLCV error: {e}")
            return None
