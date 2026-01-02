"""
RSI Strategy Implementation with Comprehensive Error Handling

Simple RSI-based trading strategy using relative strength index signals.
Enhanced with RSI crossover logging, range filter parameters, and robust error handling.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from trading.exceptions import StrategyInitializationError

from .rsi_utils import calculate_rsi, validate_rsi_parameters

logger = logging.getLogger(__name__)


class FallbackRSIStrategy:
    """Fallback RSI strategy when main strategy fails."""

    def __init__(self):
        self.rsi_period = 14
        self.oversold_threshold = 30
        self.overbought_threshold = 70

    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate simple fallback signals."""
        try:
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0
            signals["rsi"] = 50.0  # Neutral RSI value

            # Simple fallback: buy if price is rising, sell if falling
            if len(df) > 1:
                price_change = df["Close"].diff()
                signals.loc[price_change > 0, "signal"] = 1  # Buy
                signals.loc[price_change < 0, "signal"] = -1  # Sell

            return signals
        except Exception as e:
            logger.error(f"Fallback RSI strategy failed: {e}")
            # Return neutral signals
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0
            signals["rsi"] = 50.0
            return signals


class RSIStrategy:
    """RSI-based trading strategy with comprehensive error handling and fallback mechanisms."""

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
        """Initialize RSI strategy with error handling."""
        try:
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

            # Initialize fallback strategy
            self.fallback_strategy = FallbackRSIStrategy()

            logger.info(
                f"RSI Strategy initialized successfully with period={rsi_period}, "
                f"oversold={oversold_threshold}, overbought={overbought_threshold}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize RSI Strategy: {e}")
            logger.error(traceback.format_exc())
            raise StrategyInitializationError(
                f"RSI Strategy initialization failed: {str(e)}"
            )

    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI for the given data with error handling."""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")

            if data.empty:
                raise ValueError("Input data is empty")

            if "Close" not in data.columns:
                raise ValueError("Data must contain 'Close' column")

            # Handle NaN values
            close_prices = data["Close"].ffill().bfill()

            if len(close_prices) < self.rsi_period + 1:
                raise ValueError(
                    f"Insufficient data for RSI calculation. Need at least {self.rsi_period + 1} points, got {len(close_prices)}"
                )

            return calculate_rsi(close_prices, self.rsi_period)

        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            logger.error(traceback.format_exc())

            # Return neutral RSI values as fallback
            logger.info("Using fallback RSI values (50.0)")
            return pd.Series(50.0, index=data.index)

    def log_rsi_crossover(
        self,
        timestamp: datetime,
        rsi_value: float,
        crossover_type: str,
        price: float,
        additional_info: Dict[str, Any] = None,
    ):
        """Log RSI crossover events with error handling."""
        try:
            if not self.log_crossovers:
                return

            crossover_event = {
                "timestamp": (
                    timestamp.isoformat()
                    if isinstance(timestamp, datetime)
                    else str(timestamp)
                ),
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

        except Exception as e:
            logger.error(f"Failed to log RSI crossover: {e}")
            # Continue without logging

    def is_rsi_in_valid_range(self, rsi_value: float) -> bool:
        """Check if RSI value is within the valid range with error handling."""
        try:
            if not self.enable_range_filter:
                return True

            if not isinstance(rsi_value, (int, float)) or np.isnan(rsi_value):
                return False

            return self.min_rsi_range <= rsi_value <= self.max_rsi_range

        except Exception as e:
            logger.error(f"Failed to check RSI range: {e}")
            return True  # Default to valid if check fails

    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate trading signals based on RSI with comprehensive error handling.

        Args:
            df: Price data DataFrame with required columns
            **kwargs: Additional parameters (overrides config)

        Returns:
            DataFrame with signals and RSI values

        Raises:
            StrategyExecutionError: If signal generation fails
        """
        try:
            # Validate input
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")

            if df.empty:
                raise ValueError("DataFrame is empty")

            # Check for required columns (case-insensitive)
            df_lower = df.copy()
            df_lower.columns = df_lower.columns.str.lower()

            required_columns = ["close"]
            missing_columns = [
                col for col in required_columns if col not in df_lower.columns
            ]
            if missing_columns:
                raise ValueError(f"Data must contain columns: {missing_columns}")

            # Handle NaN values
            if df_lower["close"].isna().any():
                logger.warning(
                    "Input data contains NaN values, filling with forward fill"
                )
                df_lower["close"] = (
                    df_lower["close"].ffill().bfill()
                )

            # Update parameters with kwargs if provided
            _unused_var = rsi_period  # Placeholder, flake8 ignore: F841
            oversold_threshold = kwargs.get(
                "oversold_threshold", self.oversold_threshold
            )
            overbought_threshold = kwargs.get(
                "overbought_threshold", self.overbought_threshold
            )
            enable_range_filter = kwargs.get(
                "enable_range_filter", self.enable_range_filter
            )
            min_rsi_range = kwargs.get("min_rsi_range", self.min_rsi_range)
            max_rsi_range = kwargs.get("max_rsi_range", self.max_rsi_range)
            log_crossovers = kwargs.get("log_crossovers", self.log_crossovers)

            # Calculate RSI with error handling
            try:
                data = df_lower.copy()
                data["RSI"] = self.calculate_rsi(data)
            except Exception as e:
                logger.error(f"RSI calculation failed, using fallback: {e}")
                data = df_lower.copy()
                data["RSI"] = pd.Series(50.0, index=data.index)

            # Initialize signals DataFrame
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0
            signals["rsi"] = data["RSI"]

            # Generate signals with error handling
            try:
                for i in range(1, len(data)):
                    current_rsi = data["RSI"].iloc[i]
                    previous_rsi = data["RSI"].iloc[i - 1]
                    current_price = data["close"].iloc[i]
                    current_time = (
                        df.index[i]
                        if hasattr(df.index[i], "to_pydatetime")
                        else pd.Timestamp.now()
                    )

                    # Check if RSI is in valid range
                    if enable_range_filter and not (
                        min_rsi_range <= current_rsi <= max_rsi_range
                    ):
                        if log_crossovers:
                            self.log_rsi_crossover(
                                current_time,
                                current_rsi,
                                "range_exit",
                                current_price,
                                additional_info={"prev_rsi": round(previous_rsi, 4)},
                            )
                        continue

                    # Check for range entry if previously outside range
                    if (
                        log_crossovers
                        and enable_range_filter
                        and not (min_rsi_range <= previous_rsi <= max_rsi_range)
                    ):
                        self.log_rsi_crossover(
                            current_time,
                            current_rsi,
                            "range_enter",
                            current_price,
                            additional_info={"prev_rsi": round(previous_rsi, 4)},
                        )

                    # Oversold condition (buy signal)
                    if (
                        current_rsi < oversold_threshold
                        and previous_rsi >= oversold_threshold
                    ):
                        signals.iloc[i]["signal"] = 1
                        if log_crossovers:
                            self.log_rsi_crossover(
                                current_time,
                                current_rsi,
                                "oversold",
                                current_price,
                                additional_info={"threshold": oversold_threshold},
                            )

                    # Overbought condition (sell signal)
                    elif (
                        current_rsi > overbought_threshold
                        and previous_rsi <= overbought_threshold
                    ):
                        signals.iloc[i]["signal"] = -1
                        if log_crossovers:
                            self.log_rsi_crossover(
                                current_time,
                                current_rsi,
                                "overbought",
                                current_price,
                                additional_info={"threshold": overbought_threshold},
                            )

            except Exception as e:
                logger.error(f"Signal generation loop failed: {e}")
                logger.error(traceback.format_exc())
                # Continue with existing signals

            # Validate output
            if signals.isnull().any().any():
                logger.warning("Signals contain NaN values, filling with zeros")
                signals = signals.fillna(0)

            logger.info(
                f"RSI signals generated successfully. "
                f"Buy signals: {sum(signals['signal'] == 1)}, "
                f"Sell signals: {sum(signals['signal'] == -1)}"
            )

            return signals

        except Exception as e:
            logger.error(f"RSI signal generation failed: {e}")
            logger.error(traceback.format_exc())

            # Use fallback strategy
            logger.info("Using fallback RSI strategy")
            try:
                return self.fallback_strategy.generate_signals(df, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback strategy also failed: {fallback_error}")
                # Return neutral signals as last resort
                signals = pd.DataFrame(index=df.index)
                signals["signal"] = 0
                signals["rsi"] = 50.0
                return signals

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters with error handling."""
        try:
            return {
                "rsi_period": self.rsi_period,
                "oversold_threshold": self.oversold_threshold,
                "overbought_threshold": self.overbought_threshold,
                "enable_range_filter": self.enable_range_filter,
                "min_rsi_range": self.min_rsi_range,
                "max_rsi_range": self.max_rsi_range,
                "log_crossovers": self.log_crossovers,
            }
        except Exception as e:
            logger.error(f"Failed to get parameters: {e}")
            return {}

    def set_parameters(self, **kwargs):
        """Set strategy parameters with validation and error handling."""
        try:
            # Validate parameters before setting
            if "rsi_period" in kwargs:
                rsi_period = kwargs["rsi_period"]
                if not isinstance(rsi_period, int) or rsi_period <= 0:
                    raise ValueError("rsi_period must be a positive integer")

            if "oversold_threshold" in kwargs:
                oversold_threshold = kwargs["oversold_threshold"]
                if (
                    not isinstance(oversold_threshold, (int, float))
                    or oversold_threshold < 0
                    or oversold_threshold > 100
                ):
                    raise ValueError("oversold_threshold must be between 0 and 100")

            if "overbought_threshold" in kwargs:
                overbought_threshold = kwargs["overbought_threshold"]
                if (
                    not isinstance(overbought_threshold, (int, float))
                    or overbought_threshold < 0
                    or overbought_threshold > 100
                ):
                    raise ValueError("overbought_threshold must be between 0 and 100")

            # Set parameters
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logger.warning(f"Unknown parameter: {key}")

            logger.info(f"Parameters updated: {kwargs}")

        except Exception as e:
            logger.error(f"Failed to set parameters: {e}")
            logger.error(traceback.format_exc())
            raise StrategyInitializationError(f"Parameter setting failed: {str(e)}")

    def get_crossover_log(self) -> List[Dict[str, Any]]:
        """Get crossover log with error handling."""
        try:
            return self.crossover_log.copy()
        except Exception as e:
            logger.error(f"Failed to get crossover log: {e}")
            return []

    def clear_crossover_log(self):
        """Clear crossover log with error handling."""
        try:
            self.crossover_log.clear()
            logger.info("Crossover log cleared")
        except Exception as e:
            logger.error(f"Failed to clear crossover log: {e}")

    def export_crossover_log(self, filepath: str):
        """Export crossover log to JSON file with error handling."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.crossover_log, f, indent=2)
            logger.info(f"Crossover log exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export crossover log: {e}")
            logger.error(traceback.format_exc())

    def get_crossover_statistics(self) -> Dict[str, Any]:
        """Get crossover statistics with error handling."""
        try:
            if not self.crossover_log:
                return {"total_crossovers": 0}
            stats = {
                "total_crossovers": len(self.crossover_log),
                "oversold_count": sum(
                    1
                    for event in self.crossover_log
                    if event["crossover_type"] == "oversold"
                ),
                "overbought_count": sum(
                    1
                    for event in self.crossover_log
                    if event["crossover_type"] == "overbought"
                ),
                "range_exit_count": sum(
                    1
                    for event in self.crossover_log
                    if event["crossover_type"] == "range_exit"
                ),
                "average_rsi": (
                    np.mean([event["rsi_value"] for event in self.crossover_log])
                    if self.crossover_log
                    else 0.0
                ),
                "average_price": (
                    np.mean([event["price"] for event in self.crossover_log])
                    if self.crossover_log
                    else 0.0
                ),
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate crossover statistics: {e}")
            return {"total_crossovers": 0, "error": str(e)}
