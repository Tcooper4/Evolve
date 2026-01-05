"""
SMA Strategy Implementation.

This module provides a Simple Moving Average (SMA) crossover
trading strategy with signal generation and position calculation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import centralized technical indicators
from utils.technical_indicators import calculate_ema, calculate_sma

logger = logging.getLogger(__name__)


@dataclass
class SMAConfig:
    """Configuration for SMA strategy with enhanced options."""

    short_window: int = 20
    long_window: int = 50
    min_volume: float = 1000.0
    min_price: float = 1.0
    confirmation_periods: int = 3  # Number of periods to confirm trend change
    enable_smoothing: bool = True  # Enable EMA smoothing of SMAs
    smoothing_period: int = 5  # EMA period for smoothing
    asymmetric_windows: bool = False  # Allow different short/long window ratios
    short_window_ratio: float = 0.4  # Ratio for asymmetric short window
    long_window_ratio: float = 1.0  # Ratio for asymmetric long window
    base_window: int = 30  # Base window for asymmetric calculations


class SMAStrategy:
    """SMA crossover trading strategy implementation."""

    def __init__(self, config: Optional[SMAConfig] = None):
        """Initialize the strategy with configuration."""
        self.config = config or SMAConfig()
        self.signals = None
        self.positions = None
        self.smoothed_signals = None

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate strategy configuration."""
        if self.config.short_window >= self.config.long_window:
            raise ValueError("Short window must be less than long window")

        if self.config.short_window <= 0 or self.config.long_window <= 0:
            raise ValueError("Window periods must be positive")

        if self.config.min_volume < 0:
            raise ValueError("Minimum volume must be non-negative")

        if self.config.min_price < 0:
            raise ValueError("Minimum price must be non-negative")

        if self.config.confirmation_periods < 0:
            raise ValueError("Confirmation periods must be non-negative")

        if self.config.smoothing_period <= 0:
            raise ValueError("Smoothing period must be positive")

    def _calculate_asymmetric_windows(self) -> Tuple[int, int]:
        """Calculate asymmetric window sizes based on ratios."""
        short_window = int(self.config.base_window * self.config.short_window_ratio)
        long_window = int(self.config.base_window * self.config.long_window_ratio)

        # Ensure minimum window sizes
        short_window = max(5, short_window)
        long_window = max(short_window + 5, long_window)

        return short_window, long_window

    def calculate_sma(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate short and long SMAs for the given data."""
        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Use asymmetric windows if enabled
        if self.config.asymmetric_windows:
            short_window, long_window = self._calculate_asymmetric_windows()
        else:
            short_window = self.config.short_window
            long_window = self.config.long_window

        # Calculate SMAs using centralized function
        short_sma = calculate_sma(data["close"], short_window)
        long_sma = calculate_sma(data["close"], long_window)

        # Apply smoothing if enabled
        if self.config.enable_smoothing:
            short_sma = self._apply_smoothing(short_sma)
            long_sma = self._apply_smoothing(long_sma)

        return short_sma, long_sma

    def _apply_smoothing(self, series: pd.Series) -> pd.Series:
        """Apply EMA smoothing to a series."""
        try:
            return calculate_ema(series, self.config.smoothing_period)
        except Exception as e:
            logger.warning(f"Failed to apply smoothing: {e}")
            return series

    def _calculate_crossover_strength(
        self, short_sma: pd.Series, long_sma: pd.Series
    ) -> pd.Series:
        """Calculate the strength of crossover signals."""
        # Safely calculate crossover strength with division-by-zero protection
        crossover_strength = np.where(
            np.abs(long_sma) > 1e-10,  # Threshold to avoid near-zero division
            ((short_sma - long_sma) / long_sma) * 100,
            0  # Default to 0 when long_sma is too small
        )

        # Normalize to [-1, 1] range
        crossover_strength = crossover_strength.clip(-10, 10) / 10

        return crossover_strength

    def _detect_crossover_points(
        self, short_sma: pd.Series, long_sma: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Detect bullish and bearish crossover points."""
        # Detect crossovers
        bullish_cross = (short_sma > long_sma) & (
            short_sma.shift(1) <= long_sma.shift(1)
        )
        bearish_cross = (short_sma < long_sma) & (
            short_sma.shift(1) >= long_sma.shift(1)
        )

        # Apply confirmation periods if specified
        if self.config.confirmation_periods > 0:
            bullish_cross = self._confirm_trend_change(
                bullish_cross, short_sma, long_sma, "bullish"
            )
            bearish_cross = self._confirm_trend_change(
                bearish_cross, short_sma, long_sma, "bearish"
            )

        return bullish_cross, bearish_cross

    def _confirm_trend_change(
        self,
        crossover_points: pd.Series,
        short_sma: pd.Series,
        long_sma: pd.Series,
        trend_type: str,
    ) -> pd.Series:
        """Confirm trend change over multiple periods."""
        confirmed_points = crossover_points.copy()

        for i in range(1, self.config.confirmation_periods + 1):
            if trend_type == "bullish":
                # Check if short SMA remains above long SMA
                confirmed_points &= short_sma.shift(-i) > long_sma.shift(-i)
            else:  # bearish
                # Check if short SMA remains below long SMA
                confirmed_points &= short_sma.shift(-i) < long_sma.shift(-i)

        return confirmed_points

    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals based on SMA crossovers with enhanced features.

        Args:
            df: Price data DataFrame with required columns
            **kwargs: Additional parameters (overrides config)

        Returns:
            DataFrame with signals and SMA components

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("DataFrame is empty")

        # Additional validation for 'Close' column presence and all-NaN
        if "Close" not in df.columns or df["Close"].isna().all():
            # Return empty signals DataFrame with correct index and columns
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0
            signals["short_sma"] = np.nan
            signals["long_sma"] = np.nan
            signals["crossover_strength"] = np.nan
            return signals

        # Check for required columns (case-insensitive)
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.lower()

        required_columns = ["close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in df_lower.columns
        ]
        if missing_columns:
            raise ValueError(f"Data must contain columns: {missing_columns}")

        # Handle NaN values
        if df_lower["close"].isna().any():
            # Only forward fill - never use backward fill in backtesting!
            df_lower["close"] = df_lower["close"].ffill()
            
            # For any remaining leading NaNs, use first valid value
            # (This is acceptable as it doesn't use future data)
            first_valid_idx = df_lower["close"].first_valid_index()
            if first_valid_idx is not None:
                first_valid_value = df_lower["close"].loc[first_valid_idx]
                df_lower["close"] = df_lower["close"].fillna(first_valid_value)

        if df_lower["volume"].isna().any():
            df_lower["volume"] = df_lower["volume"].fillna(0)

        # Update config with kwargs if provided
        config = self.config
        if kwargs:
            config = SMAConfig(
                short_window=kwargs.get("short_window", self.config.short_window),
                long_window=kwargs.get("long_window", self.config.long_window),
                min_volume=kwargs.get("min_volume", self.config.min_volume),
                min_price=kwargs.get("min_price", self.config.min_price),
                confirmation_periods=kwargs.get(
                    "confirmation_periods", self.config.confirmation_periods
                ),
                enable_smoothing=kwargs.get(
                    "enable_smoothing", self.config.enable_smoothing
                ),
                smoothing_period=kwargs.get(
                    "smoothing_period", self.config.smoothing_period
                ),
                asymmetric_windows=kwargs.get(
                    "asymmetric_windows", self.config.asymmetric_windows
                ),
                short_window_ratio=kwargs.get(
                    "short_window_ratio", self.config.short_window_ratio
                ),
                long_window_ratio=kwargs.get(
                    "long_window_ratio", self.config.long_window_ratio
                ),
                base_window=kwargs.get("base_window", self.config.base_window),
            )

        try:
            # Calculate SMAs
            short_sma, long_sma = self.calculate_sma(df_lower)

            # Detect crossover points
            bullish_cross, bearish_cross = self._detect_crossover_points(
                short_sma, long_sma
            )

            # Calculate crossover strength
            crossover_strength = self._calculate_crossover_strength(short_sma, long_sma)

            # Initialize signals DataFrame
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0

            # Generate signals
            signals.loc[bullish_cross, "signal"] = 1  # Buy signal
            signals.loc[bearish_cross, "signal"] = -1  # Sell signal

            # Add SMA components to signals
            signals["short_sma"] = short_sma
            signals["long_sma"] = long_sma
            signals["crossover_strength"] = crossover_strength

            # Filter signals based on volume and price
            volume_mask = df_lower["volume"] >= config.min_volume
            price_mask = df_lower["close"] >= config.min_price
            signals.loc[~(volume_mask & price_mask), "signal"] = 0

            # Create smoothed signals if enabled
            if config.enable_smoothing:
                signals = self._create_smoothed_signals(signals)

            # Handle any remaining NaN values in signals
            signals = signals.fillna(0)

            self.signals = signals
            return signals

        except Exception as e:
            raise ValueError(f"Error generating SMA signals: {str(e)}")

    def _create_smoothed_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Create smoothed version of signals using EMA.

        Args:
            signals: Original signals DataFrame

        Returns:
            Signals DataFrame with smoothed signal column
        """
        try:
            # Create smoothed signal using EMA
            smoothed_signal = calculate_ema(
                signals["signal"], self.config.smoothing_period
            )

            # Round smoothed signals to nearest integer
            signals["smoothed_signal"] = np.round(smoothed_signal).astype(int)

            # Clip to valid range
            signals["smoothed_signal"] = signals["smoothed_signal"].clip(-1, 1)

            self.smoothed_signals = signals["smoothed_signal"]
            return signals

        except Exception as e:
            logger.warning(f"Failed to create smoothed signals: {e}")
            signals["smoothed_signal"] = signals["signal"]
            return signals

    def _log_signal_statistics(self, signals: pd.DataFrame):
        """Log signal generation statistics.

        Args:
            signals: Generated signals DataFrame
        """
        buy_signals = (signals["signal"] == 1).sum()
        sell_signals = (signals["signal"] == -1).sum()
        total_signals = buy_signals + sell_signals

        logger.info(
            f"SMA Strategy generated {total_signals} signals: "
            f"{buy_signals} buy, {sell_signals} sell"
        )

        if "smoothed_signal" in signals.columns:
            smoothed_buy = (signals["smoothed_signal"] == 1).sum()
            smoothed_sell = (signals["smoothed_signal"] == -1).sum()
            logger.info(f"Smoothed signals: {smoothed_buy} buy, {smoothed_sell} sell")

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading positions based on signals."""
        if self.signals is None:
            self.generate_signals(data)

        positions = pd.DataFrame(index=data.index)

        # Use smoothed signals if available, otherwise use regular signals
        signal_column = (
            "smoothed_signal" if "smoothed_signal" in self.signals.columns else "signal"
        )
        positions["position"] = self.signals[signal_column].cumsum()

        # Ensure positions are within bounds
        positions["position"] = positions["position"].clip(-1, 1)

        self.positions = positions
        return positions

    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        return {
            "short_window": self.config.short_window,
            "long_window": self.config.long_window,
            "min_volume": self.config.min_volume,
            "min_price": self.config.min_price,
            "confirmation_periods": self.config.confirmation_periods,
            "enable_smoothing": self.config.enable_smoothing,
            "smoothing_period": self.config.smoothing_period,
            "asymmetric_windows": self.config.asymmetric_windows,
            "short_window_ratio": self.config.short_window_ratio,
            "long_window_ratio": self.config.long_window_ratio,
            "base_window": self.config.base_window,
        }

    def set_parameters(self, params: Dict) -> Dict:
        """Set strategy parameters."""
        try:
            self.config = SMAConfig(**params)
            self.signals = None
            self.positions = None
            self.smoothed_signals = None

            # Re-validate configuration
            self._validate_config()

            return {
                "status": "success",
                "parameters_updated": True,
                "config": self.get_parameters(),
            }
        except Exception as e:
            return {
                "success": False,
                "result": {"status": "error", "message": str(e)},
                "message": "Failed to update strategy parameters",
                "timestamp": datetime.now().isoformat(),
            }

    def get_signal_quality_metrics(self) -> Dict[str, float]:
        """Get signal quality metrics."""
        if self.signals is None:
            return {"error": "No signals available"}

        total_signals = len(self.signals)
        buy_signals = (self.signals["signal"] == 1).sum()
        sell_signals = (self.signals["signal"] == -1).sum()

        return {
            "total_signals": total_signals,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "signal_ratio": buy_signals / max(sell_signals, 1),
            "avg_crossover_strength": self.signals["crossover_strength"].mean(),
            "signal_frequency": total_signals / max(len(self.signals), 1),
        }
