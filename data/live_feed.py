"""Live Data Feed with Fallback Pipeline for Evolve Trading Platform.

This module provides a robust data feed with automatic failover between
multiple data providers: Polygon → Finnhub → Alpha Vantage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import requests
import time
import logging
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

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
    
    def check_availability(self) -> bool:
        """Check if provider is available."""
        try:
            # Simple health check
            return self._health_check()
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.is_available = False
            return False
    
    def _health_check(self) -> bool:
        """Implement health check in subclasses."""
        raise NotImplementedError
    
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical data from provider."""
        raise NotImplementedError
    
    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from provider."""
        raise NotImplementedError

class PolygonProvider(DataProvider):
    """Polygon.io data provider."""
    
    def __init__(self, api_key: str = None):
        """Initialize Polygon provider."""
        super().__init__("Polygon", api_key or os.getenv('POLYGON_API_KEY'))
        self.base_url = "https://api.polygon.io"
    
    def _health_check(self) -> bool:
        """Check Polygon API health."""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-01"
            params = {'apiKey': self.api_key}
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical data from Polygon."""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{interval}/{start_date}/{end_date}"
            params = {'apiKey': self.api_key}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] != 'OK' or not data['results']:
                return None
            
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'Open', 'h': 'High', 'l': 'Low', 
                'c': 'Close', 'v': 'Volume', 'vw': 'VWAP'
            })
            
            return df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]
            
        except Exception as e:
            logger.error(f"Polygon historical data error: {e}")
            return None
    
    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from Polygon."""
        try:
            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            params = {'apiKey': self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'results' not in data:
                return None
            
            result = data['results']
            return {
                'symbol': symbol,
                'price': result.get('lastTrade', {}).get('p', 0),
                'volume': result.get('lastTrade', {}).get('s', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Polygon live data error: {e}")
            return None

class FinnhubProvider(DataProvider):
    """Finnhub data provider."""
    
    def __init__(self, api_key: str = None):
        """Initialize Finnhub provider."""
        super().__init__("Finnhub", api_key or os.getenv('FINNHUB_API_KEY'))
        self.base_url = "https://finnhub.io/api/v1"
    
    def _health_check(self) -> bool:
        """Check Finnhub API health."""
        try:
            url = f"{self.base_url}/quote"
            params = {'symbol': 'AAPL', 'token': self.api_key}
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical data from Finnhub."""
        try:
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())
            
            url = f"{self.base_url}/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': 'D',
                'from': start_ts,
                'to': end_ts,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data['s'] != 'ok' or not data['t']:
                return None
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Finnhub historical data error: {e}")
            return None
    
    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from Finnhub."""
        try:
            url = f"{self.base_url}/quote"
            params = {'symbol': symbol, 'token': self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return {
                'symbol': symbol,
                'price': data.get('c', 0),
                'volume': data.get('v', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Finnhub live data error: {e}")
            return None

class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self, api_key: str = None):
        """Initialize Alpha Vantage provider."""
        super().__init__("Alpha Vantage", api_key or os.getenv('ALPHA_VANTAGE_API_KEY'))
        self.base_url = "https://www.alphavantage.co/query"
    
    def _health_check(self) -> bool:
        """Check Alpha Vantage API health."""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'AAPL',
                'apikey': self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage."""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'Time Series (Daily)' not in data:
                return None
            
            time_series = data['Time Series (Daily)']
            records = []
            
            for date, values in time_series.items():
                if start_date <= date <= end_date:
                    records.append({
                        'timestamp': pd.to_datetime(date),
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': float(values['5. volume'])
                    })
            
            df = pd.DataFrame(records)
            return df.sort_values('timestamp')
            
        except Exception as e:
            logger.error(f"Alpha Vantage historical data error: {e}")
            return None
    
    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from Alpha Vantage."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'Global Quote' not in data:
                return None
            
            quote = data['Global Quote']
            return {
                'symbol': symbol,
                'price': float(quote.get('05. price', 0)),
                'volume': float(quote.get('06. volume', 0)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Alpha Vantage live data error: {e}")
            return None

class LiveDataFeed:
    """Main live data feed with fallback capabilities."""
    
    def __init__(self):
        """Initialize the live data feed."""
        self.providers = [
            PolygonProvider(),
            FinnhubProvider(),
            AlphaVantageProvider()
        ]
        self.current_provider_index = 0
        self.fallback_history = []
        
    def _get_current_provider(self) -> DataProvider:
        """Get current active provider."""
        return self.providers[self.current_provider_index]
    
    def _switch_provider(self) -> bool:
        """Switch to next available provider."""
        original_index = self.current_provider_index
        
        for i in range(len(self.providers)):
            self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)
            provider = self._get_current_provider()
            
            if provider.check_availability():
                if original_index != self.current_provider_index:
                    logger.warning(f"Switched from {self.providers[original_index].name} to {provider.name}")
                    self.fallback_history.append({
                        'from': self.providers[original_index].name,
                        'to': provider.name,
                        'timestamp': datetime.now().isoformat()
                    })
                return True
        
        logger.error("No available data providers")
        return False
    
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical data with fallback."""
        attempts = 0
        max_attempts = len(self.providers)
        
        while attempts < max_attempts:
            provider = self._get_current_provider()
            
            if not provider.check_availability():
                if not self._switch_provider():
                    return None
                continue
            
            data = provider.get_historical_data(symbol, start_date, end_date, interval)
            if data is not None and not data.empty:
                logger.info(f"Retrieved historical data for {symbol} from {provider.name}")
                return data
            
            # Try next provider
            if not self._switch_provider():
                break
            
            attempts += 1
        
        logger.error(f"Failed to get historical data for {symbol} from all providers")
        return None
    
    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data with fallback."""
        attempts = 0
        max_attempts = len(self.providers)
        
        while attempts < max_attempts:
            provider = self._get_current_provider()
            
            if not provider.check_availability():
                if not self._switch_provider():
                    return None
                continue
            
            data = provider.get_live_data(symbol)
            if data is not None:
                logger.info(f"Retrieved live data for {symbol} from {provider.name}")
                return data
            
            # Try next provider
            if not self._switch_provider():
                break
            
            attempts += 1
        
        logger.error(f"Failed to get live data for {symbol} from all providers")
        return None
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        status = {}
        for provider in self.providers:
            status[provider.name] = {
                'available': provider.is_available,
                'error_count': provider.error_count,
                'last_check': provider.last_check
            }
        
        status['current_provider'] = self._get_current_provider().name
        status['fallback_history'] = self.fallback_history[-10:]  # Last 10 switches
        
        return status
    
    def reset_providers(self) -> None:
        """Reset all providers to available state."""
        for provider in self.providers:
            provider.is_available = True
            provider.error_count = 0
            provider.last_check = None
        
        self.current_provider_index = 0
        logger.info("Reset all data providers")

# Global data feed instance
data_feed = LiveDataFeed()

def get_data_feed() -> LiveDataFeed:
    """Get the global data feed instance."""
    return data_feed 