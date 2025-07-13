"""
Data validation utilities for the trading system.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation utility class."""

    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []

    def validate_market_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate market data structure and content."""
        try:
            # Check required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.validation_errors.append(f"Missing required columns: {missing_columns}")
                return False

            # Check for NaN values
            nan_counts = data[required_columns].isna().sum()
            if nan_counts.sum() > 0:
                self.validation_warnings.append(f"Found NaN values: {nan_counts.to_dict()}")

            # Check for negative prices
            price_columns = ["open", "high", "low", "close"]
            negative_prices = data[price_columns] < 0
            if negative_prices.any().any():
                self.validation_errors.append("Found negative prices")
                return False

            # Check for zero volumes
            zero_volumes = data["volume"] == 0
            if zero_volumes.sum() > len(data) * 0.1:  # More than 10% zero volumes
                self.validation_warnings.append("High number of zero volume entries")

            # Check data consistency
            invalid_high_low = (data["high"] < data["low"]).any()
            if invalid_high_low:
                self.validation_errors.append("High price less than low price found")
                return False

            return len(self.validation_errors) == 0

        except Exception as e:
            self.validation_errors.append(f"Validation error: {str(e)}")
            return False

    def validate_forecast_data(self, data: pd.DataFrame) -> bool:
        """Validate forecast data structure."""
        try:
            # Check required columns for forecasts
            required_columns = ["timestamp", "forecast", "confidence"]
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.validation_errors.append(f"Missing required forecast columns: {missing_columns}")
                return False

            # Check confidence values are between 0 and 1
            if "confidence" in data.columns:
                invalid_confidence = (data["confidence"] < 0) | (data["confidence"] > 1)
                if invalid_confidence.any():
                    self.validation_errors.append("Confidence values must be between 0 and 1")
                    return False

            return len(self.validation_errors) == 0

        except Exception as e:
            self.validation_errors.append(f"Forecast validation error: {str(e)}")
            return False

    def validate_model_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate model parameters."""
        try:
            # Check for required parameters
            if "model_type" not in params:
                self.validation_errors.append("Missing model_type parameter")
                return False

            # Validate numeric parameters
            numeric_params = ["learning_rate", "epochs", "batch_size", "validation_split"]
            for param in numeric_params:
                if param in params:
                    if not isinstance(params[param], (int, float)) or params[param] <= 0:
                        self.validation_errors.append(f"Invalid {param}: must be positive number")
                        return False

            return len(self.validation_errors) == 0

        except Exception as e:
            self.validation_errors.append(f"Parameter validation error: {str(e)}")
            return False

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()

    def get_validation_warnings(self) -> List[str]:
        """Get list of validation warnings."""
        return self.validation_warnings.copy()

    def clear_validation_state(self):
        """Clear validation state."""
        self.validation_errors.clear()
        self.validation_warnings.clear()


def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format."""
    if not symbol or not isinstance(symbol, str):
        return False

    # Basic symbol validation (alphanumeric, 1-10 characters)
    if not symbol.replace(".", "").replace("-", "").isalnum():
        return False

    if len(symbol) < 1 or len(symbol) > 10:
        return False

    return True


def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe format."""
    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
    return timeframe in valid_timeframes


def validate_date_range(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> bool:
    """Validate date range."""
    try:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        return start_date < end_date
    except:
        return False
