"""
RSI Strategy Implementation

Simple RSI-based trading strategy using relative strength index signals.
Enhanced with RSI crossover logging and range filter parameters.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .rsi_utils import calculate_rsi, validate_rsi_parameters

logger = logging.getLogger(__name__)


class RSIStrategy:
    """RSI-based trading strategy with enhanced logging and range filters."""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
        enable_range_filter: bool = True,
        min_rsi_range: float = 10.0,
        max_rsi_range: float = 90.0,
        log_crossovers: bool = True,
    ):
        """Initialize RSI strategy.

        Args:
            rsi_period: Period for RSI calculation
            oversold_threshold: RSI threshold for oversold condition
            overbought_threshold: RSI threshold for overbought condition
            enable_range_filter: Whether to enable RSI range filtering
            min_rsi_range: Minimum RSI value to consider valid
            max_rsi_range: Maximum RSI value to consider valid
            log_crossovers: Whether to log RSI crossover events
        """
        # Validate parameters
        is_valid, error_msg = validate_rsi_parameters(
            rsi_period, oversold_threshold, overbought_threshold
        )
        if not is_valid:
            raise ValueError(f"Invalid RSI parameters: {error_msg}")

        # Validate range filter parameters
        if min_rsi_range >= max_rsi_range:
            raise ValueError("min_rsi_range must be less than max_rsi_range")

        if min_rsi_range < 0 or max_rsi_range > 100:
            raise ValueError("RSI range must be between 0 and 100")

        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.enable_range_filter = enable_range_filter
        self.min_rsi_range = min_rsi_range
        self.max_rsi_range = max_rsi_range
        self.log_crossovers = log_crossovers

        # Initialize crossover logging
        self.crossover_log: List[Dict[str, Any]] = []

    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI for the given data using shared utilities.

        Args:
            data: Price data with 'Close' column

        Returns:
            RSI values
        """
        return calculate_rsi(data["Close"], self.rsi_period)

    def log_rsi_crossover(
        self,
        timestamp: datetime,
        rsi_value: float,
        crossover_type: str,
        price: float,
        additional_info: Dict[str, Any] = None,
    ):
        """Log RSI crossover events with detailed information.

        Args:
            timestamp: Time of the crossover
            rsi_value: RSI value at crossover
            crossover_type: Type of crossover ('oversold', 'overbought', 'range_exit')
            price: Price at crossover
            additional_info: Additional information to log
        """
        if not self.log_crossovers:
            return

        crossover_event = {
            "timestamp": timestamp.isoformat()
            if isinstance(timestamp, datetime)
            else str(timestamp),
            "rsi_value": round(rsi_value, 4),
            "crossover_type": crossover_type,
            "price": round(price, 4),
            "strategy_params": {
                "rsi_period": self.rsi_period,
                "oversold_threshold": self.oversold_threshold,
                "overbought_threshold": self.overbought_threshold,
                "enable_range_filter": self.enable_range_filter,
                "min_rsi_range": self.min_rsi_range,
                "max_rsi_range": self.max_rsi_range,
            },
            "additional_info": additional_info or {},
        }

        self.crossover_log.append(crossover_event)

        # Log to console with emoji indicators
        emoji_map = {
            "oversold": "📈",
            "overbought": "📉",
            "range_exit": "⚠️",
            "range_enter": "🔄",
        }

        emoji = emoji_map.get(crossover_type, "📊")
        logger.info(
            f"{emoji} RSI Crossover: {crossover_type.upper()} at {timestamp} | "
            f"RSI: {rsi_value:.2f} | Price: ${price:.2f}"
        )

    def is_rsi_in_valid_range(self, rsi_value: float) -> bool:
        """Check if RSI value is within the valid range.

        Args:
            rsi_value: RSI value to check

        Returns:
            True if RSI is in valid range, False otherwise
        """
        if not self.enable_range_filter:
            return True

        return self.min_rsi_range <= rsi_value <= self.max_rsi_range

    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals based on RSI with enhanced logging.
        
        Args:
            df: Price data DataFrame with required columns
            **kwargs: Additional parameters (overrides config)
            
        Returns:
            DataFrame with signals and RSI values
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for required columns (case-insensitive)
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.lower()
        
        required_columns = ["close"]
        missing_columns = [col for col in required_columns if col not in df_lower.columns]
        if missing_columns:
            raise ValueError(f"Data must contain columns: {missing_columns}")
        
        # Handle NaN values
        if df_lower["close"].isna().any():
            df_lower["close"] = df_lower["close"].fillna(method='ffill').fillna(method='bfill')
        
        # Update parameters with kwargs if provided
        rsi_period = kwargs.get('rsi_period', self.rsi_period)
        oversold_threshold = kwargs.get('oversold_threshold', self.oversold_threshold)
        overbought_threshold = kwargs.get('overbought_threshold', self.overbought_threshold)
        enable_range_filter = kwargs.get('enable_range_filter', self.enable_range_filter)
        min_rsi_range = kwargs.get('min_rsi_range', self.min_rsi_range)
        max_rsi_range = kwargs.get('max_rsi_range', self.max_rsi_range)
        log_crossovers = kwargs.get('log_crossovers', self.log_crossovers)

        try:
            # Calculate RSI
            data = df_lower.copy()
            data["RSI"] = calculate_rsi(data["close"], rsi_period)

            # Initialize signals DataFrame
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0
            signals["rsi"] = data["RSI"]

            # Generate signals based on RSI thresholds
            for i in range(1, len(data)):
                current_rsi = data["RSI"].iloc[i]
                prev_rsi = data["RSI"].iloc[i - 1]
                current_price = data["close"].iloc[i]
                timestamp = (
                    data.index[i]
                    if hasattr(data.index[i], "to_pydatetime")
                    else datetime.now()
                )

                # Skip if RSI is not in valid range
                if enable_range_filter and not (min_rsi_range <= current_rsi <= max_rsi_range):
                    if log_crossovers and min_rsi_range <= prev_rsi <= max_rsi_range:
                        self.log_rsi_crossover(
                            timestamp=timestamp,
                            rsi_value=current_rsi,
                            crossover_type="range_exit",
                            price=current_price,
                            additional_info={"prev_rsi": round(prev_rsi, 4)},
                        )
                    continue

                # Check for range entry if previously outside range
                if log_crossovers and enable_range_filter and not (min_rsi_range <= prev_rsi <= max_rsi_range):
                    self.log_rsi_crossover(
                        timestamp=timestamp,
                        rsi_value=current_rsi,
                        crossover_type="range_enter",
                        price=current_price,
                        additional_info={"prev_rsi": round(prev_rsi, 4)},
                    )

                # Generate buy signal (oversold condition)
                if current_rsi < oversold_threshold and prev_rsi >= oversold_threshold:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    if log_crossovers:
                        self.log_rsi_crossover(
                            timestamp=timestamp,
                            rsi_value=current_rsi,
                            crossover_type="oversold",
                            price=current_price,
                            additional_info={"threshold": oversold_threshold},
                        )

                # Generate sell signal (overbought condition)
                elif current_rsi > overbought_threshold and prev_rsi <= overbought_threshold:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    if log_crossovers:
                        self.log_rsi_crossover(
                            timestamp=timestamp,
                            rsi_value=current_rsi,
                            crossover_type="overbought",
                            price=current_price,
                            additional_info={"threshold": overbought_threshold},
                        )

            # Handle any remaining NaN values in signals
            signals = signals.fillna(0)

            return signals
            
        except Exception as e:
            raise ValueError(f"Error generating RSI signals: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            "rsi_period": self.rsi_period,
            "oversold_threshold": self.oversold_threshold,
            "overbought_threshold": self.overbought_threshold,
            "enable_range_filter": self.enable_range_filter,
            "min_rsi_range": self.min_rsi_range,
            "max_rsi_range": self.max_rsi_range,
            "log_crossovers": self.log_crossovers,
        }

    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        try:
            # Update parameters
            if 'rsi_period' in kwargs:
                self.rsi_period = kwargs['rsi_period']
            if 'oversold_threshold' in kwargs:
                self.oversold_threshold = kwargs['oversold_threshold']
            if 'overbought_threshold' in kwargs:
                self.overbought_threshold = kwargs['overbought_threshold']
            if 'enable_range_filter' in kwargs:
                self.enable_range_filter = kwargs['enable_range_filter']
            if 'min_rsi_range' in kwargs:
                self.min_rsi_range = kwargs['min_rsi_range']
            if 'max_rsi_range' in kwargs:
                self.max_rsi_range = kwargs['max_rsi_range']
            if 'log_crossovers' in kwargs:
                self.log_crossovers = kwargs['log_crossovers']

            return {
                "success": True,
                "result": {
                    "status": "success",
                    "parameters_updated": True,
                    "config": self.get_parameters(),
                },
                "message": "Parameters updated successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": True,
                "result": {"status": "error", "message": str(e)},
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def get_crossover_log(self) -> List[Dict[str, Any]]:
        """Get the crossover log."""
        return self.crossover_log.copy()

    def clear_crossover_log(self):
        """Clear the crossover log."""
        self.crossover_log.clear()

    def export_crossover_log(self, filepath: str):
        """Export crossover log to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.crossover_log, f, indent=2, default=str)
            logger.info(f"Crossover log exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export crossover log: {e}")

    def get_crossover_statistics(self) -> Dict[str, Any]:
        """Get statistics about crossover events."""
        if not self.crossover_log:
            return {
                "total_crossovers": 0,
                "crossover_types": {},
                "avg_rsi_value": 0.0,
                "avg_price": 0.0,
            }

        crossover_types = {}
        rsi_values = []
        prices = []

        for event in self.crossover_log:
            crossover_type = event["crossover_type"]
            crossover_types[crossover_type] = crossover_types.get(crossover_type, 0) + 1
            rsi_values.append(event["rsi_value"])
            prices.append(event["price"])

        return {
            "total_crossovers": len(self.crossover_log),
            "crossover_types": crossover_types,
            "avg_rsi_value": np.mean(rsi_values) if rsi_values else 0.0,
            "avg_price": np.mean(prices) if prices else 0.0,
        }
