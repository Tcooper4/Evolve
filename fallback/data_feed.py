"""
Fallback Data Feed Implementation

Provides fallback functionality for data retrieval when primary data
sources are unavailable.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

class FallbackDataFeed:
    """
    Fallback implementation of the Data Feed.
    
    Provides mock data generation and basic data operations when primary
    data sources are unavailable. This ensures the system can continue
    to function for testing and demonstration purposes.
    """
    
    def __init__(self) -> None:
        """
        Initialize the fallback data feed.
        
        Sets up basic logging and initializes internal state for
        fallback data operations.
        """
        self._status = "fallback"
        self._providers = ["mock", "yfinance_fallback"]
        self._last_data_request = None
        logger.info("FallbackDataFeed initialized")
    
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol (fallback implementation).
        
        Args:
            symbol: The stock symbol to get data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
            self._last_data_request = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'interval': interval,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try yfinance first as fallback
            try:
                data = self._get_yfinance_data(symbol, start_date, end_date, interval)
                if data is not None and not data.empty:
                    logger.info(f"Successfully retrieved data from yfinance for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"yfinance fallback failed for {symbol}: {e}")
            
            # Generate mock data if yfinance fails
            logger.info(f"Generating mock data for {symbol}")
            return self._generate_mock_data(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return self._generate_mock_data(symbol, start_date, end_date, interval)
    
    def _get_yfinance_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Get data from yfinance as a fallback source.
        
        Args:
            symbol: The stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Optional[pd.DataFrame]: Data from yfinance or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return None
            
            # Ensure column names are consistent
            data.columns = [col.title() for col in data.columns]
            
            # Add timestamp column if not present
            if 'Timestamp' not in data.columns:
                data['Timestamp'] = data.index
            
            return data
            
        except Exception as e:
            logger.warning(f"yfinance data retrieval failed for {symbol}: {e}")
            return None
    
    def _generate_mock_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Generate mock historical data for testing purposes.
        
        Args:
            symbol: The stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            pd.DataFrame: Mock historical price data
        """
        try:
            # Parse dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Generate date range
            if interval == "1d":
                freq = "D"
            elif interval == "1h":
                freq = "H"
            else:
                freq = "D"
            
            dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)
            
            # Generate realistic mock data
            np.random.seed(hash(symbol) % 2**32)  # Deterministic for same symbol
            
            # Base price around 100 with some variation
            base_price = 100 + (hash(symbol) % 50)
            
            # Generate price series with random walk
            returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1))  # Ensure price doesn't go negative
            
            # Generate OHLCV data
            data = pd.DataFrame({
                'Timestamp': dates,
                'Open': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': prices,
                'Volume': np.random.normal(1000000, 200000, len(dates))
            })
            
            # Ensure High >= Low and High >= Open, Close
            data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
            
            # Ensure volume is positive
            data['Volume'] = data['Volume'].abs()
            
            logger.info(f"Generated mock data for {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error generating mock data for {symbol}: {e}")
            # Return minimal fallback data
            return pd.DataFrame({
                'Timestamp': [datetime.now()],
                'Open': [100.0],
                'High': [101.0],
                'Low': [99.0],
                'Close': [100.0],
                'Volume': [1000000]
            })
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback data feed.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            return {
                'status': self._status,
                'available_providers': len(self._providers),
                'providers': self._providers,
                'last_request': self._last_data_request,
                'fallback_mode': True,
                'message': 'Using fallback data feed'
            }
        except Exception as e:
            logger.error(f"Error getting fallback data feed health: {e}")
            return {
                'status': 'error',
                'available_providers': 0,
                'fallback_mode': True,
                'error': str(e)
            }
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols (fallback implementation).
        
        Returns:
            List[str]: List of available symbols
        """
        try:
            # Return common symbols for fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol (fallback implementation).
        
        Args:
            symbol: The stock symbol
            
        Returns:
            Optional[float]: Latest price or None if unavailable
        """
        try:
            # Try to get real data first
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            
            data = self.get_historical_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                return float(data['Close'].iloc[-1])
            
            # Return mock price
            return 100.0 + (hash(symbol) % 50)
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None 