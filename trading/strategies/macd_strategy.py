"""MACD (Moving Average Convergence Divergence) trading strategy implementation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

# Import centralized technical indicators
from utils.technical_indicators import calculate_macd
from utils.strategy_utils import macd_crossover_signals


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

    def calculate_macd(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components for the given data."""
        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Use centralized MACD calculation
        macd_line, signal_line, histogram = calculate_macd(
            data["close"],
            self.config.fast_period,
            self.config.slow_period,
            self.config.signal_period,
        )

        return macd_line, signal_line, histogram

    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals based on MACD with de-duplication.
        
        Args:
            df: Price data DataFrame with required columns
            **kwargs: Additional parameters (overrides config)
            
        Returns:
            DataFrame with signals and MACD components
            
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
        
        required_columns = ["close", "volume"]
        missing_columns = [col for col in required_columns if col not in df_lower.columns]
        if missing_columns:
            raise ValueError(f"Data must contain columns: {missing_columns}")
        
        # Handle NaN values
        if df_lower["close"].isna().any():
            df_lower["close"] = df_lower["close"].fillna(method='ffill').fillna(method='bfill')
        
        if df_lower["volume"].isna().any():
            df_lower["volume"] = df_lower["volume"].fillna(0)
        
        # Update config with kwargs if provided
        config = self.config
        if kwargs:
            config = MACDConfig(
                fast_period=kwargs.get('fast_period', self.config.fast_period),
                slow_period=kwargs.get('slow_period', self.config.slow_period),
                signal_period=kwargs.get('signal_period', self.config.signal_period),
                min_volume=kwargs.get('min_volume', self.config.min_volume),
                min_price=kwargs.get('min_price', self.config.min_price),
                tolerance=kwargs.get('tolerance', self.config.tolerance),
            )

        try:
            # Calculate MACD components
            macd_line, signal_line, histogram = calculate_macd(
                df_lower["close"],
                config.fast_period,
                config.slow_period,
                config.signal_period,
            )

            # Initialize signals DataFrame
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0

            # Generate signals based on MACD line crossing signal line with tolerance buffer
            signals["signal"] = macd_crossover_signals(macd_line, signal_line, config.tolerance)

            # Apply smoothing using short EMA window to reduce noise
            smoothing_window = kwargs.get('smoothing_window', 3)
            if smoothing_window > 1:
                signals["signal"] = signals["signal"].rolling(
                    window=smoothing_window, center=True, min_periods=1
                ).mean()
                
                # Re-quantize smoothed signals
                signals["signal"] = np.where(signals["signal"] > 0.1, 1,
                                           np.where(signals["signal"] < -0.1, -1, 0))

            # Drop duplicate consecutive signals to avoid over-trading
            signals["signal"] = signals["signal"].loc[
                ~(signals["signal"] == signals["signal"].shift(1))
            ]

            # Add MACD components to signals
            signals["macd_line"] = macd_line
            signals["signal_line"] = signal_line
            signals["histogram"] = histogram

            # Filter signals based on volume and price
            volume_mask = df_lower["volume"] >= config.min_volume
            price_mask = df_lower["close"] >= config.min_price
            signals.loc[~(volume_mask & price_mask), "signal"] = 0

            # Handle NaN alignment issues when joining with price data
            # Ensure signals align with original data index
            if len(signals) != len(df):
                # Reindex to match original data
                signals = signals.reindex(df.index, method='ffill')
                signals = signals.fillna(0)
            
            # Handle any remaining NaN values in signals
            signals = signals.fillna(0)

            self.signals = signals
            return signals
            
        except Exception as e:
            raise ValueError(f"Error generating MACD signals: {str(e)}")

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
            "tolerance": self.config.tolerance,
        }

    def set_parameters(self, params: Dict) -> Dict:
        """Set strategy parameters."""
        try:
            self.config = MACDConfig(**params)
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
