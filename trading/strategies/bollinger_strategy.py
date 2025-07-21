"""
Bollinger Bands Strategy

This module implements a Bollinger Bands trading strategy with squeeze detection
and signal generation capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BollingerConfig:
    """Configuration for Bollinger Bands strategy."""

    window: int = 20
    num_std: float = 2.0
    min_volume: float = 1000.0
    min_price: float = 1.0
    squeeze_threshold: float = 0.1  # Threshold for squeeze detection (bandwidth ratio)


class BollingerStrategy(BaseStrategy):
    """Bollinger Bands trading strategy with squeeze detection."""

    def __init__(self, config: Optional[BollingerConfig] = None):
        """Initialize Bollinger Bands strategy."""
        super().__init__("Bollinger_Bands", "Bollinger Bands mean reversion strategy")
        self.config = config or BollingerConfig()
        self.signals = None

    def calculate_bands(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price data with 'close' column

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        close_prices = data["close"]
        middle_band = close_prices.rolling(window=self.config.window).mean()
        std = close_prices.rolling(window=self.config.window).std()
        upper_band = middle_band + (self.config.num_std * std)
        lower_band = middle_band - (self.config.num_std * std)

        return upper_band, middle_band, lower_band

    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.

        Args:
            df: Price data DataFrame with required columns
            **kwargs: Additional parameters (overrides config)

        Returns:
            DataFrame with signals and Bollinger Bands components

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate input
        if not isinstance(df, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            logger.warning("DataFrame is empty; skipping signal generation.")
            raise ValueError("DataFrame is empty")

        # Check for required columns (case-insensitive)
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.lower()

        required_columns = ["close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in df_lower.columns
        ]
        if missing_columns:
            logger.error(f"Data must contain columns: {missing_columns}")
            raise ValueError(f"Data must contain columns: {missing_columns}")

        # Clean NaNs
        df_lower = df_lower.drop_duplicates(subset=df_lower.index)
        df_lower = df_lower.dropna()
        if len(df_lower) < 2:
            logger.warning(
                "Not enough data after NaN cleaning; skipping signal generation."
            )
            raise ValueError("Not enough data after NaN cleaning")

        # Fallback for constant price series
        if df_lower["close"].std() == 0:
            logger.warning(
                "Constant price series detected (stddev=0); skipping signal generation."
            )
            signals = pd.DataFrame(index=df_lower.index)
            signals["signal"] = 0
            signals["upper_band"] = df_lower["close"]
            signals["middle_band"] = df_lower["close"]
            signals["lower_band"] = df_lower["close"]
            signals["bandwidth"] = 0
            signals["squeeze"] = False
            self.signals = signals
            return signals

        # Handle NaN values
        if df_lower["close"].isna().any():
            df_lower["close"] = (
                df_lower["close"].fillna(method="ffill").fillna(method="bfill")
            )

        if df_lower["volume"].isna().any():
            df_lower["volume"] = df_lower["volume"].fillna(0)

        # Update config with kwargs if provided
        config = self.config
        if kwargs:
            config = BollingerConfig(
                window=kwargs.get("window", self.config.window),
                num_std=kwargs.get("num_std", self.config.num_std),
                min_volume=kwargs.get("min_volume", self.config.min_volume),
                min_price=kwargs.get("min_price", self.config.min_price),
                squeeze_threshold=kwargs.get(
                    "squeeze_threshold", self.config.squeeze_threshold
                ),
            )

        try:
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.calculate_bands(df_lower)

            # Calculate squeeze condition (bandwidth)
            bandwidth = (upper_band - lower_band) / middle_band
            squeeze_condition = bandwidth < config.squeeze_threshold

            # Initialize signals DataFrame
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0

            # Generate signals (avoid trades during squeeze)
            buy_condition = (df_lower["close"] < lower_band) & ~squeeze_condition
            sell_condition = (df_lower["close"] > upper_band) & ~squeeze_condition

            signals.loc[buy_condition, "signal"] = 1  # Buy signal
            signals.loc[sell_condition, "signal"] = -1  # Sell signal

            # Add squeeze information
            signals["bandwidth"] = bandwidth
            signals["squeeze"] = squeeze_condition

            # Add bands to signals
            signals["upper_band"] = upper_band
            signals["middle_band"] = middle_band
            signals["lower_band"] = lower_band

            # Add volume filter
            volume_filter = df_lower["volume"] >= config.min_volume
            signals.loc[~volume_filter, "signal"] = 0

            # Add price filter
            price_filter = df_lower["close"] >= config.min_price
            signals.loc[~price_filter, "signal"] = 0

            # Store signals
            self.signals = signals

            logger.info(
                f"Generated {len(signals[signals['signal'] != 0])} Bollinger Bands signals"
            )
            return signals

        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signals: {e}")
            raise

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position sizes based on signals.

        Args:
            data: Market data

        Returns:
            DataFrame with position sizes
        """
        if self.signals is None:
            self.generate_signals(data)

        positions = self.signals.copy()
        positions["position"] = positions["signal"].cumsum()
        positions["position"] = positions["position"].clip(-1, 1)  # Limit positions

        return positions

    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        return {
            "window": self.config.window,
            "num_std": self.config.num_std,
            "min_volume": self.config.min_volume,
            "min_price": self.config.min_price,
            "squeeze_threshold": self.config.squeeze_threshold,
        }

    def set_parameters(self, params: Dict) -> Dict:
        """Set strategy parameters."""
        if "window" in params:
            self.config.window = params["window"]
        if "num_std" in params:
            self.config.num_std = params["num_std"]
        if "min_volume" in params:
            self.config.min_volume = params["min_volume"]
        if "min_price" in params:
            self.config.min_price = params["min_price"]
        if "squeeze_threshold" in params:
            self.config.squeeze_threshold = params["squeeze_threshold"]

        return self.get_parameters()

    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for optimization."""
        return {
            "window": {"type": "int", "min": 5, "max": 50, "default": 20},
            "num_std": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0},
            "min_volume": {
                "type": "float",
                "min": 100.0,
                "max": 10000.0,
                "default": 1000.0,
            },
            "min_price": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0},
            "squeeze_threshold": {
                "type": "float",
                "min": 0.05,
                "max": 0.3,
                "default": 0.1,
            },
        }
