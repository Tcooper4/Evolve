"""Simple Moving Average (SMA) trading strategy implementation.
Enhanced with crossover smoothing logic and asymmetric window sizes.
"""

import logging
from dataclasses import dataclass
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
    """Simple Moving Average (SMA) trading strategy implementation with enhanced features."""

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
            raise ValueError("Window sizes must be positive")

        if self.config.smoothing_period <= 0:
            raise ValueError("Smoothing period must be positive")

        if self.config.short_window_ratio <= 0 or self.config.long_window_ratio <= 0:
            raise ValueError("Window ratios must be positive")

        if self.config.short_window_ratio >= self.config.long_window_ratio:
            raise ValueError("Short window ratio must be less than long window ratio")

    def _calculate_asymmetric_windows(self) -> Tuple[int, int]:
        """Calculate asymmetric window sizes based on ratios."""
        if not self.config.asymmetric_windows:
            return self.config.short_window, self.config.long_window

        short_window = max(
            5, int(self.config.base_window * self.config.short_window_ratio)
        )
        long_window = max(
            short_window + 5,
            int(self.config.base_window * self.config.long_window_ratio),
        )

        logger.info(
            f"Using asymmetric windows: short={short_window}, long={long_window}"
        )
        return short_window, long_window

    def calculate_sma(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate short and long SMAs for the given data with optional smoothing."""
        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Calculate actual window sizes
        short_window, long_window = self._calculate_asymmetric_windows()

        # Use centralized SMA calculation
        short_sma = calculate_sma(data["close"], short_window)
        long_sma = calculate_sma(data["close"], long_window)

        # Apply smoothing if enabled
        if self.config.enable_smoothing:
            short_sma = self._apply_smoothing(short_sma)
            long_sma = self._apply_smoothing(long_sma)
            logger.debug(
                f"Applied EMA smoothing with period {self.config.smoothing_period}"
            )

        return short_sma, long_sma

    def _apply_smoothing(self, series: pd.Series) -> pd.Series:
        """Apply EMA smoothing to a series.

        Args:
            series: Series to smooth

        Returns:
            Smoothed series
        """
        try:
            return calculate_ema(series, self.config.smoothing_period)
        except Exception as e:
            logger.warning(f"Failed to apply smoothing: {e}, returning original series")
            return series

    def _calculate_crossover_strength(
        self, short_sma: pd.Series, long_sma: pd.Series
    ) -> pd.Series:
        """Calculate the strength of SMA crossover.

        Args:
            short_sma: Short SMA series
            long_sma: Long SMA series

        Returns:
            Crossover strength series
        """
        # Calculate percentage difference
        crossover_strength = ((short_sma - long_sma) / long_sma) * 100

        # Apply smoothing to crossover strength
        if self.config.enable_smoothing:
            crossover_strength = self._apply_smoothing(crossover_strength)

        return crossover_strength

    def _detect_crossover_points(
        self, short_sma: pd.Series, long_sma: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Detect crossover points with enhanced logic.

        Args:
            short_sma: Short SMA series
            long_sma: Long SMA series

        Returns:
            Tuple of (bullish_crossovers, bearish_crossovers)
        """
        # Calculate crossover points
        bullish_cross = (short_sma > long_sma) & (
            short_sma.shift(1) <= long_sma.shift(1)
        )
        bearish_cross = (short_sma < long_sma) & (
            short_sma.shift(1) >= long_sma.shift(1)
        )

        # Apply confirmation filter
        if self.config.confirmation_periods > 1:
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
        """Confirm trend changes over multiple periods.

        Args:
            crossover_points: Initial crossover points
            short_sma: Short SMA series
            long_sma: Long SMA series
            trend_type: Type of trend ('bullish' or 'bearish')

        Returns:
            Confirmed crossover points
        """
        confirmed_crossovers = pd.Series(False, index=crossover_points.index)

        for i in range(self.config.confirmation_periods, len(crossover_points)):
            if crossover_points.iloc[i]:
                # Check if trend is consistent over confirmation periods
                start_idx = max(0, i - self.config.confirmation_periods + 1)
                recent_short = short_sma.iloc[start_idx : i + 1]
                recent_long = long_sma.iloc[start_idx : i + 1]

                if trend_type == "bullish":
                    # Check if short SMA consistently above long SMA
                    if all(recent_short > recent_long):
                        confirmed_crossovers.iloc[i] = True
                else:  # bearish
                    # Check if short SMA consistently below long SMA
                    if all(recent_short < recent_long):
                        confirmed_crossovers.iloc[i] = True

        return confirmed_crossovers

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on SMA crossover with enhanced features."""
        # Edge case fallback logic
        if data is None or data.empty:
            logger.warning(
                "Invalid data provided to SMA strategy: data is None or empty"
            )
            return pd.DataFrame()  # Return empty DataFrame

        if not isinstance(data, pd.DataFrame):
            logger.warning("Input must be a pandas DataFrame")
            return pd.DataFrame()

        required_columns = ["close", "volume"]
        if not all(col in data.columns for col in required_columns):
            logger.warning(f"Data must contain columns: {required_columns}")
            return pd.DataFrame()

        # Calculate SMAs with smoothing
        short_sma, long_sma = self.calculate_sma(data)

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals["short_sma"] = short_sma
        signals["long_sma"] = long_sma

        # Calculate crossover strength
        signals["crossover_strength"] = self._calculate_crossover_strength(
            short_sma, long_sma
        )

        # Detect crossover points
        bullish_cross, bearish_cross = self._detect_crossover_points(
            short_sma, long_sma
        )

        # Generate signals
        signals.loc[bullish_cross, "signal"] = 1
        signals.loc[bearish_cross, "signal"] = -1

        # Add crossover metadata
        signals["bullish_crossover"] = bullish_cross
        signals["bearish_crossover"] = bearish_cross

        # Filter signals based on volume and price
        volume_mask = data["volume"] >= self.config.min_volume
        price_mask = data["close"] >= self.config.min_price
        signals.loc[~(volume_mask & price_mask), "signal"] = 0

        # Create smoothed signals if enabled
        if self.config.enable_smoothing:
            self.smoothed_signals = self._create_smoothed_signals(signals)
            signals["smoothed_signal"] = self.smoothed_signals["smoothed_signal"]

        self.signals = signals

        # Log signal statistics
        self._log_signal_statistics(signals)

        return signals

    def _create_smoothed_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Create smoothed version of signals using EMA.

        Args:
            signals: Original signals DataFrame

        Returns:
            DataFrame with smoothed signals
        """
        smoothed = signals.copy()

        # Apply EMA smoothing to the signal column
        raw_signals = signals["signal"].replace(0, np.nan)
        smoothed_signals = calculate_ema(
            raw_signals.fillna(method="ffill"), self.config.smoothing_period
        )

        # Convert back to discrete signals
        smoothed["smoothed_signal"] = np.where(
            smoothed_signals > 0.5, 1, np.where(smoothed_signals < -0.5, -1, 0)
        )

        return smoothed

    def _log_signal_statistics(self, signals: pd.DataFrame):
        """Log statistics about generated signals."""
        total_signals = len(signals[signals["signal"] != 0])
        bullish_signals = len(signals[signals["signal"] == 1])
        bearish_signals = len(signals[signals["signal"] == -1])

        logger.info(f"SMA Strategy Statistics:")
        logger.info(f"  Total signals: {total_signals}")
        logger.info(f"  Bullish signals: {bullish_signals}")
        logger.info(f"  Bearish signals: {bearish_signals}")
        logger.info(f"  Smoothing enabled: {self.config.enable_smoothing}")
        logger.info(f"  Asymmetric windows: {self.config.asymmetric_windows}")

        if self.config.enable_smoothing:
            smoothed_total = len(signals[signals["smoothed_signal"] != 0])
            logger.info(f"  Smoothed signals: {smoothed_total}")

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading positions based on signals."""
        if self.signals is None:
            self.generate_signals(data)

        positions = pd.DataFrame(index=data.index)

        # Use smoothed signals if available, otherwise use regular signals
        signal_column = (
            "smoothed_signal"
            if self.config.enable_smoothing
            and "smoothed_signal" in self.signals.columns
            else "signal"
        )
        positions["position"] = self.signals[signal_column].cumsum()

        # Ensure positions are within bounds
        positions["position"] = positions["position"].clip(-1, 1)

        # Add position metadata
        positions["signal_type"] = self.signals[signal_column]
        positions["crossover_strength"] = self.signals["crossover_strength"]

        self.positions = positions
        return positions

    def get_parameters(self) -> Dict:
        """Get strategy parameters including enhanced options."""
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
        """Set strategy parameters with validation."""
        try:
            # Update config with new parameters
            for key, value in params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Validate updated configuration
            self._validate_config()

            # Reset cached data
            self.signals = None
            self.positions = None
            self.smoothed_signals = None

            return {
                "status": "success",
                "parameters_updated": True,
                "config": self.get_parameters(),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_signal_quality_metrics(self) -> Dict[str, float]:
        """Get quality metrics for the generated signals."""
        if self.signals is None:
            return {}

        metrics = {}

        # Signal frequency
        total_periods = len(self.signals)
        signal_periods = len(self.signals[self.signals["signal"] != 0])
        metrics["signal_frequency"] = (
            signal_periods / total_periods if total_periods > 0 else 0
        )

        # Crossover strength statistics
        if "crossover_strength" in self.signals.columns:
            strength = self.signals["crossover_strength"].dropna()
            if len(strength) > 0:
                metrics["avg_crossover_strength"] = float(strength.mean())
                metrics["max_crossover_strength"] = float(strength.max())
                metrics["min_crossover_strength"] = float(strength.min())

        # Smoothing effectiveness (if enabled)
        if self.config.enable_smoothing and "smoothed_signal" in self.signals.columns:
            original_signals = self.signals["signal"]
            smoothed_signals = self.signals["smoothed_signal"]
            signal_changes = (original_signals != smoothed_signals).sum()
            metrics["smoothing_effectiveness"] = 1 - (signal_changes / total_periods)

        return metrics
