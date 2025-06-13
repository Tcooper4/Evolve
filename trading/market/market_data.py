import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import yfinance as yf

class MarketData:
    """Class for managing market data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the market data manager.
        
        Args:
            config: Configuration dictionary
        """
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
            
        # Initialize data storage
        self.data = {}
        self.metadata = {}
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'cache_size' in self.config:
            if not isinstance(self.config['cache_size'], int) or self.config['cache_size'] <= 0:
                raise ValueError("cache_size must be a positive integer")
                
        if 'update_frequency' in self.config:
            if not isinstance(self.config['update_frequency'], (int, float)) or self.config['update_frequency'] <= 0:
                raise ValueError("update_frequency must be a positive number")
                
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
            self.data[symbol] = data
            self.metadata[symbol] = metadata or {}
            self.metadata[symbol]['last_updated'] = datetime.now().isoformat()
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
            if symbol not in self.data:
                raise KeyError(f"Symbol {symbol} not found")
                
            data = self.data[symbol].copy()
            
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
            
            if symbol in self.data:
                # Merge new data with existing data
                self.data[symbol] = pd.concat([self.data[symbol], new_data])
                self.data[symbol] = self.data[symbol][~self.data[symbol].index.duplicated(keep='last')]
                self.data[symbol] = self.data[symbol].sort_index()
            else:
                self.data[symbol] = new_data
                
            self.metadata[symbol]['last_updated'] = datetime.now().isoformat()
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
            if symbol in self.data:
                del self.data[symbol]
                del self.metadata[symbol]
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
            if symbol not in self.metadata:
                raise KeyError(f"Symbol {symbol} not found")
                
            return self.metadata[symbol].copy()
            
        except Exception as e:
            self.logger.error(f"Error getting metadata for {symbol}: {str(e)}")
            raise
            
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols.
        
        Returns:
            List of symbols
        """
        return list(self.data.keys())
        
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about stored data.
        
        Returns:
            Dictionary with data information
        """
        info = {
            'num_symbols': len(self.data),
            'symbols': self.get_all_symbols(),
            'data_ranges': {},
            'last_updated': {}
        }
        
        for symbol in self.data:
            info['data_ranges'][symbol] = {
                'start': self.data[symbol].index.min().isoformat(),
                'end': self.data[symbol].index.max().isoformat()
            }
            info['last_updated'][symbol] = self.metadata[symbol].get('last_updated')
            
        return info

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch market data for a given symbol and date range.

        Args:
            symbol (str): The stock symbol to fetch data for.
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: The fetched market data.
        """
        data = yf.download(symbol, start=start_date, end=end_date)
        self.add_data(symbol, data)
        return data

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