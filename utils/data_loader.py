"""
Data Loader Module

This module handles data loading operations for the Evolve Trading Platform:
- Historical data retrieval
- Data validation and preprocessing
- Sample data generation
- Data feed management
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading operations for the trading platform."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data loader.

        Args:
            config: Configuration dictionary for data loading settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def load_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical data for a given symbol and date range.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame: Historical price data with OHLCV columns
        """
        try:
            # Try to get data from live feed
            from data.live_feed import get_data_feed

            data_feed = get_data_feed()
            data = data_feed.get_historical_data(symbol, start_date, end_date)

            if not data.empty:
                self.logger.info(f"Loaded {len(data)} records for {symbol}")
                return data
            else:
                self.logger.warning(f"No data returned for {symbol}, using sample data")
                return self._create_sample_data(symbol, start_date, end_date)

        except Exception as e:
            self.logger.error(f"Error loading historical data for {symbol}: {e}")
            return self._create_sample_data(symbol, start_date, end_date)

    def _create_sample_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create sample data for testing and fallback scenarios.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame: Sample OHLCV data
        """
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            dates = pd.date_range(start=start, end=end, freq="D")

            # Generate realistic sample data
            base_price = 100.0
            returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
            prices = base_price * np.exp(np.cumsum(returns))

            data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
                    "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                    "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                    "Close": prices,
                    "Volume": np.random.normal(1000000, 200000, len(dates)),
                }
            )

            # Ensure High >= Low and High >= Open, Close
            data["High"] = data[["Open", "High", "Close"]].max(axis=1)
            data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

            self.logger.info(f"Created sample data for {symbol}: {len(data)} records")
            return data

        except Exception as e:
            self.logger.error(f"Error creating sample data: {e}")
            return pd.DataFrame()

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness.

        Args:
            data: DataFrame to validate

        Returns:
            bool: True if data is valid, False otherwise
        """
        if data.empty:
            self.logger.warning("Data is empty")
            return False

        required_columns = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            return False

        # Check for negative prices
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            if (data[col] <= 0).any():
                self.logger.warning(f"Negative or zero prices found in {col}")
                return False

        # Check for negative volume
        if (data["Volume"] < 0).any():
            self.logger.warning("Negative volume found")
            return False

        # Check for missing values
        if data.isnull().any().any():
            self.logger.warning("Missing values found in data")
            return False

        self.logger.info("Data validation passed")
        return True

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for analysis and modeling.

        Args:
            data: Raw data DataFrame

        Returns:
            DataFrame: Preprocessed data
        """
        try:
            # Make a copy to avoid modifying original
            processed_data = data.copy()

            # Sort by timestamp
            processed_data = processed_data.sort_values("timestamp").reset_index(drop=True)

            # Add technical indicators
            processed_data = self._add_technical_indicators(processed_data)

            # Remove any remaining NaN values
            processed_data = processed_data.dropna()

            self.logger.info(f"Preprocessed data: {len(processed_data)} records")
            return processed_data

        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return data

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.

        Args:
            data: Price data DataFrame

        Returns:
            DataFrame: Data with technical indicators added
        """
        try:
            # Simple Moving Averages
            data["SMA_20"] = data["Close"].rolling(window=20).mean()
            data["SMA_50"] = data["Close"].rolling(window=50).mean()

            # Exponential Moving Averages
            data["EMA_12"] = data["Close"].ewm(span=12).mean()
            data["EMA_26"] = data["Close"].ewm(span=26).mean()

            # RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            data["MACD"] = data["EMA_12"] - data["EMA_26"]
            data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()
            data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]

            # Bollinger Bands
            data["BB_Middle"] = data["Close"].rolling(window=20).mean()
            bb_std = data["Close"].rolling(window=20).std()
            data["BB_Upper"] = data["BB_Middle"] + (bb_std * 2)
            data["BB_Lower"] = data["BB_Middle"] - (bb_std * 2)

            return data

        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the data.

        Args:
            data: DataFrame to summarize

        Returns:
            Dict: Summary statistics
        """
        try:
            summary = {
                "total_records": len(data),
                "date_range": {
                    "start": data["timestamp"].min().strftime("%Y-%m-%d"),
                    "end": data["timestamp"].max().strftime("%Y-%m-%d"),
                },
                "price_stats": {
                    "min_close": data["Close"].min(),
                    "max_close": data["Close"].max(),
                    "mean_close": data["Close"].mean(),
                    "std_close": data["Close"].std(),
                },
                "volume_stats": {
                    "total_volume": data["Volume"].sum(),
                    "avg_volume": data["Volume"].mean(),
                    "max_volume": data["Volume"].max(),
                },
                "missing_values": data.isnull().sum().to_dict(),
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating data summary: {e}")
            return {}


def get_data_loader(config: Optional[Dict[str, Any]] = None) -> DataLoader:
    """
    Get a data loader instance.

    Args:
        config: Configuration dictionary

    Returns:
        DataLoader: Configured data loader instance
    """
    return DataLoader(config)
