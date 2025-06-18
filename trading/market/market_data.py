"""
Market Data Provider with advanced caching and fallback mechanisms.

This module provides robust market data fetching with configurable fallback sources,
caching, and performance monitoring.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from trading.logs.logger import log_metrics

class MarketDataError(Exception):
    """Base exception for market data errors."""
    pass

class NetworkError(MarketDataError):
    """Network-related errors."""
    pass

class ValidationError(MarketDataError):
    """Data validation errors."""
    pass

class FormatError(MarketDataError):
    """Data format errors."""
    pass

class MarketData:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize market data provider with configuration."""
        self.config = config or {}
        self._validate_config()
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize cache
        self.cache = OrderedDict()
        self.last_update = {}
        self.alpha_vantage = None
        if self.config.get('use_alpha_vantage', False):
            self.alpha_vantage = TimeSeries(key=self.config.get('alpha_vantage_key'))
        
        # Initialize error tracking
        self.error_counts = {
            'network': 0,
            'validation': 0,
            'format': 0,
            'other': 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ['cache_size', 'update_threshold', 'max_retries']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
        
        if not isinstance(self.config['cache_size'], int) or self.config['cache_size'] <= 0:
            raise ValueError("cache_size must be a positive integer")
        
        if not isinstance(self.config['update_threshold'], int) or self.config['update_threshold'] <= 0:
            raise ValueError("update_threshold must be a positive integer")
        
        if not isinstance(self.config['max_retries'], int) or self.config['max_retries'] <= 0:
            raise ValueError("max_retries must be a positive integer")

    def _classify_error(self, error: Exception) -> str:
        """Classify error into categories."""
        error_str = str(error).lower()
        if any(network_term in error_str for network_term in ['connection', 'timeout', 'network', 'http']):
            return 'network'
        elif any(validation_term in error_str for validation_term in ['validation', 'invalid', 'missing']):
            return 'validation'
        elif any(format_term in error_str for format_term in ['format', 'parse', 'json']):
            return 'format'
        return 'other'

    def _log_error(self, error: Exception, symbol: str) -> None:
        """Log error with classification."""
        error_type = self._classify_error(error)
        self.error_counts[error_type] += 1
        self.logger.error(f"Error fetching {symbol}: {error_type} error - {str(error)}")
        
        if self.config.get('metrics_enabled', False):
            log_metrics("market_data_error", {
                'symbol': symbol,
                'error_type': error_type,
                'error_message': str(error),
                'timestamp': datetime.utcnow().isoformat()
            })

    def _manage_cache_size(self) -> None:
        """Manage cache size using FIFO strategy."""
        while len(self.cache) > self.config['cache_size']:
            self.cache.popitem(last=False)  # Remove oldest item

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return min(300, (2 ** attempt) + np.random.uniform(0, 1))

    def fetch_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch market data with fallback mechanisms."""
        start_time = time.time()
        attempt = 0
        
        while attempt < self.config['max_retries']:
            try:
                # Try primary source (yfinance)
                data = yf.download(symbol, start=start_date, end=end_date)
                
                if data.empty:
                    raise ValidationError(f"No data returned for {symbol}")
                
                # Validate data
                if 'Close' not in data.columns:
                    raise ValidationError(f"Missing 'Close' column for {symbol}")
                
                # Calculate metrics
                fetch_time = time.time() - start_time
                row_count = len(data)
                missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                
                # Log metrics if enabled
                if self.config.get('metrics_enabled', False):
                    log_metrics("market_data_fetch", {
                        'symbol': symbol,
                        'fetch_time': fetch_time,
                        'row_count': row_count,
                        'missing_ratio': missing_ratio,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                
                # Update cache
                self.cache[symbol] = data
                self.last_update[symbol] = datetime.now()
                self._manage_cache_size()
                
                return data
                
            except Exception as e:
                attempt += 1
                error_type = self._classify_error(e)
                self._log_error(e, symbol)
                
                if attempt < self.config['max_retries']:
                    # Try fallback to Alpha Vantage
                    if self.alpha_vantage and error_type == 'network':
                        try:
                            data, _ = self.alpha_vantage.get_daily(symbol=symbol, outputsize='full')
                            data = pd.DataFrame(data).astype(float)
                            data.index = pd.to_datetime(data.index)
                            return data
                        except Exception as av_error:
                            self._log_error(av_error, symbol)
                    
                    # Wait with exponential backoff
                    time.sleep(self._exponential_backoff(attempt))
                else:
                    # If all retries failed, try to return cached data
                    if symbol in self.cache:
                        self.logger.warning(f"Using cached data for {symbol} after {attempt} failed attempts")
                        return self.cache[symbol]
                    raise MarketDataError(f"Failed to fetch data for {symbol} after {attempt} attempts")

    def auto_update_all(self) -> Dict[str, Any]:
        """Auto-update all cached symbols that exceed update threshold."""
        results = {
            'updated': [],
            'skipped': [],
            'failed': []
        }
        
        current_time = datetime.now()
        update_threshold = timedelta(minutes=self.config['update_threshold'])
        
        for symbol in list(self.cache.keys()):
            last_update = self.last_update.get(symbol)
            if not last_update or (current_time - last_update) > update_threshold:
                try:
                    self.fetch_data(symbol)
                    results['updated'].append(symbol)
                except Exception as e:
                    self._log_error(e, symbol)
                    results['failed'].append(symbol)
            else:
                results['skipped'].append(symbol)
        
        if self.config.get('metrics_enabled', False):
            log_metrics("market_data_auto_update", {
                'updated_count': len(results['updated']),
                'skipped_count': len(results['skipped']),
                'failed_count': len(results['failed']),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return results

    def get_cached_symbols(self) -> List[str]:
        """Get list of currently cached symbols."""
        return list(self.cache.keys())

    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from cache."""
        if symbol in self.cache:
            del self.cache[symbol]
            del self.last_update[symbol]
            return True
        return False

    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate market data.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Data is empty")
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")
            
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be sorted in ascending order")
            
    def add_data(self, symbol: str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add market data for a symbol.
        
        Args:
            symbol: Market symbol
            data: DataFrame with market data
            metadata: Optional metadata dictionary
        """
        try:
            self._validate_data(data)
            self.cache[symbol] = data
            self.last_update[symbol] = datetime.now()
            self.logger.info(f"Added data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error adding data for {symbol}: {str(e)}")
            raise
            
    def get_data(self, symbol: str, start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> pd.DataFrame:
        """Get market data for a symbol.
        
        Args:
            symbol: Market symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with market data
            
        Raises:
            KeyError: If symbol not found
        """
        try:
            if symbol not in self.cache:
                raise KeyError(f"Symbol {symbol} not found")
                
            data = self.cache[symbol].copy()
            
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {str(e)}")
            raise
            
    def update_data(self, symbol: str, new_data: pd.DataFrame) -> None:
        """Update market data for a symbol.
        
        Args:
            symbol: Market symbol
            new_data: DataFrame with new market data
        """
        try:
            self._validate_data(new_data)
            
            if symbol in self.cache:
                # Merge new data with existing data
                self.cache[symbol] = pd.concat([self.cache[symbol], new_data])
                self.cache[symbol] = self.cache[symbol][~self.cache[symbol].index.duplicated(keep='last')]
                self.cache[symbol] = self.cache[symbol].sort_index()
            else:
                self.cache[symbol] = new_data
                
            self.last_update[symbol] = datetime.now()
            self.logger.info(f"Updated data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {str(e)}")
            raise
            
    def remove_data(self, symbol: str) -> None:
        """Remove market data for a symbol.
        
        Args:
            symbol: Market symbol
        """
        try:
            if symbol in self.cache:
                del self.cache[symbol]
                del self.last_update[symbol]
                self.logger.info(f"Removed data for {symbol}")
            else:
                self.logger.warning(f"Symbol {symbol} not found")
                
        except Exception as e:
            self.logger.error(f"Error removing data for {symbol}: {str(e)}")
            raise
            
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a symbol.
        
        Args:
            symbol: Market symbol
            
        Returns:
            Dictionary with metadata
            
        Raises:
            KeyError: If symbol not found
        """
        try:
            if symbol not in self.cache:
                raise KeyError(f"Symbol {symbol} not found")
                
            return self.cache[symbol].copy()
            
        except Exception as e:
            self.logger.error(f"Error getting metadata for {symbol}: {str(e)}")
            raise
            
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols.
        
        Returns:
            List of symbols
        """
        return list(self.cache.keys())
        
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about stored data.
        
        Returns:
            Dictionary with data information
        """
        info = {
            'num_symbols': len(self.cache),
            'symbols': self.get_all_symbols(),
            'data_ranges': {},
            'last_updated': {}
        }
        
        for symbol in self.cache:
            info['data_ranges'][symbol] = {
                'start': self.cache[symbol].index.min().isoformat(),
                'end': self.cache[symbol].index.max().isoformat()
            }
            info['last_updated'][symbol] = self.last_update.get(symbol)
            
        return info

    def process_data(self, symbol: str) -> pd.DataFrame:
        """Process the fetched market data.

        Args:
            symbol (str): The stock symbol to process data for.

        Returns:
            pd.DataFrame: The processed market data.
        """
        data = self.get_data(symbol)
        data['Daily_Return'] = data['Close'].pct_change()
        return data 