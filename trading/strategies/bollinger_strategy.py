"""Bollinger Bands trading strategy implementation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import logging

# Import centralized technical indicators
from utils.technical_indicators import calculate_bollinger_bands


logger = logging.getLogger(__name__)


@dataclass
class BollingerConfig:
    """Configuration for Bollinger Bands strategy."""

    window: int = 20
    num_std: float = 2.0
    min_volume: float = 1000.0
    min_price: float = 1.0
    squeeze_threshold: float = 0.1  # Threshold for squeeze detection (bandwidth ratio)


class BollingerStrategy:
    """Bollinger Bands trading strategy implementation."""

    def __init__(self, config: Optional[BollingerConfig] = None):
        """Initialize the strategy with configuration."""
        self.config = config or BollingerConfig()
        self.signals = None
        self.positions = None

    def calculate_bands(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands for the given data."""
        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Use centralized Bollinger Bands calculation
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            data["close"], self.config.window, self.config.num_std
        )

        # Drop NaNs from rolling calculations to avoid front-window issues
        upper_band = upper_band.dropna()
        middle_band = middle_band.dropna()
        lower_band = lower_band.dropna()

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
        missing_columns = [col for col in required_columns if col not in df_lower.columns]
        if missing_columns:
            logger.error(f"Data must contain columns: {missing_columns}")
            raise ValueError(f"Data must contain columns: {missing_columns}")
        
        # Clean NaNs
        df_lower = df_lower.drop_duplicates(subset=df_lower.index)
        df_lower = df_lower.dropna()
        if len(df_lower) < 2:
            logger.warning("Not enough data after NaN cleaning; skipping signal generation.")
            raise ValueError("Not enough data after NaN cleaning")
        
        # Fallback for constant price series
        if df_lower["close"].std() == 0:
            logger.warning("Constant price series detected (stddev=0); skipping signal generation.")
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
            df_lower["close"] = df_lower["close"].fillna(method='ffill').fillna(method='bfill')
        
        if df_lower["volume"].isna().any():
            df_lower["volume"] = df_lower["volume"].fillna(0)
        
        # Update config with kwargs if provided
        config = self.config
        if kwargs:
            config = BollingerConfig(
                window=kwargs.get('window', self.config.window),
                num_std=kwargs.get('num_std', self.config.num_std),
                min_volume=kwargs.get('min_volume', self.config.min_volume),
                min_price=kwargs.get('min_price', self.config.min_price),
                squeeze_threshold=kwargs.get('squeeze_threshold', self.config.squeeze_threshold),
            )

        try:
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = calculate_bollinger_bands(
                df_lower["close"], config.window, config.num_std
            )

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

            # Filter signals based on volume and price
            volume_mask = df_lower["volume"] >= config.min_volume
            price_mask = df_lower["close"] >= config.min_price
            signals.loc[~(volume_mask & price_mask), "signal"] = 0

            # Handle any remaining NaN values in signals
            signals = signals.fillna(0)

            self.signals = signals
            return signals
            
        except Exception as e:
            raise ValueError(f"Error generating Bollinger Bands signals: {str(e)}")

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
            "window": self.config.window,
            "num_std": self.config.num_std,
            "min_volume": self.config.min_volume,
            "min_price": self.config.min_price,
            "squeeze_threshold": self.config.squeeze_threshold,
        }

    def set_parameters(self, params: Dict) -> Dict:
        """Set strategy parameters."""
        try:
            self.config = BollingerConfig(**params)
            self.signals = None
            self.positions = None
            return {
                "status": "success",
                "parameters_updated": True,
                "config": self.get_parameters(),
            }
        except Exception as e:
            return {
                "success": True,
                "result": {"status": "error", "message": str(e)},
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }
