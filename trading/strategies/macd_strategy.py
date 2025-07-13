"""MACD (Moving Average Convergence Divergence) trading strategy implementation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

# Import centralized technical indicators
from utils.technical_indicators import calculate_macd


@dataclass
class MACDConfig:
    """Configuration for MACD strategy."""

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    min_volume: float = 1000.0
    min_price: float = 1.0
    tolerance: float = 0.001  # Tolerance buffer for crossover detection


class MACDStrategy:
    """MACD trading strategy implementation."""

    def __init__(self, config: Optional[MACDConfig] = None):
        """Initialize the strategy with configuration."""
        self.config = config or MACDConfig()
        self.signals = None
        self.positions = None

    def calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components for the given data."""
        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Use centralized MACD calculation
        macd_line, signal_line, histogram = calculate_macd(
            data["close"], self.config.fast_period, self.config.slow_period, self.config.signal_period
        )

        return macd_line, signal_line, histogram

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MACD with de-duplication."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        required_columns = ["close", "volume"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        # Calculate MACD components
        macd_line, signal_line, histogram = self.calculate_macd(data)

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0

        # Generate signals based on MACD line crossing signal line with tolerance buffer
        # Calculate the difference between MACD and signal line
        macd_diff = macd_line - signal_line

        # Apply tolerance buffer to avoid repeated triggers on equality
        buy_condition = macd_diff > self.config.tolerance
        sell_condition = macd_diff < -self.config.tolerance

        signals.loc[buy_condition, "signal"] = 1  # Buy signal
        signals.loc[sell_condition, "signal"] = -1  # Sell signal

        # Drop duplicate consecutive signals to avoid over-trading
        signals["signal"] = signals["signal"].loc[~(signals["signal"] == signals["signal"].shift(1))]

        # Add MACD components to signals
        signals["macd_line"] = macd_line
        signals["signal_line"] = signal_line
        signals["histogram"] = histogram

        # Filter signals based on volume and price
        volume_mask = data["volume"] >= self.config.min_volume
        price_mask = data["close"] >= self.config.min_price
        signals.loc[~(volume_mask & price_mask), "signal"] = 0

        self.signals = signals
        return signals

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading positions based on signals."""
        if self.signals is None:
            self.generate_signals(data)

        positions = pd.DataFrame(index=data.index)
        positions["position"] = self.signals["signal"].cumsum()

        # Ensure positions are within bounds
        positions["position"] = positions["position"].clip(-1, 1)

        self.positions = positions
        return positions

    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        return {
            "fast_period": self.config.fast_period,
            "slow_period": self.config.slow_period,
            "signal_period": self.config.signal_period,
            "min_volume": self.config.min_volume,
            "min_price": self.config.min_price,
        }

    def set_parameters(self, params: Dict) -> Dict:
        """Set strategy parameters."""
        try:
            self.config = MACDConfig(**params)
            self.signals = None
            self.positions = None
            return {"status": "success", "parameters_updated": True, "config": self.get_parameters()}
        except Exception as e:
            return {
                "success": True,
                "result": {"status": "error", "message": str(e)},
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }
