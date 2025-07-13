"""
Average True Range (ATR) Strategy Implementation

This module implements a trading strategy based on the Average True Range (ATR),
which is a volatility indicator used to measure market volatility and set stop losses.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ATRConfig:
    """Configuration for ATR strategy."""

    period: int = 14
    multiplier: float = 2.0
    min_volume: float = 1000.0
    min_price: float = 1.0
    use_volatility_filter: bool = True
    volatility_threshold: float = 0.02


class ATRStrategy:
    """Average True Range (ATR) trading strategy implementation."""

    def __init__(self, config: Optional[ATRConfig] = None):
        """
        Initialize the ATR strategy.

        Args:
            config: ATR strategy configuration
        """
        self.config = config or ATRConfig()
        self.logger = logging.getLogger(__name__)
        self.signals = None
        self.positions = None

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the Average True Range (ATR).

        Args:
            data: DataFrame with OHLC data

        Returns:
            pd.Series: ATR values
        """
        if not all(col in data.columns for col in ["high", "low", "close"]):
            raise ValueError("Data must contain 'high', 'low', 'close' columns")

        # Calculate True Range
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift(1))
        low_close = np.abs(data["low"] - data["close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR using exponential moving average
        atr = true_range.ewm(span=self.config.period).mean()

        return atr

    def calculate_bollinger_bands_atr(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands using ATR instead of standard deviation.

        Args:
            data: DataFrame with OHLC data

        Returns:
            Tuple: (upper_band, middle_band, lower_band)
        """
        atr = self.calculate_atr(data)
        middle_band = data["close"].rolling(window=self.config.period).mean()

        upper_band = middle_band + (self.config.multiplier * atr)
        lower_band = middle_band - (self.config.multiplier * atr)

        return upper_band, middle_band, lower_band

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on ATR.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            pd.DataFrame: Signals DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        required_columns = ["high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        # Calculate ATR
        atr = self.calculate_atr(data)

        # Calculate ATR-based Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands_atr(data)

        # Calculate volatility filter
        volatility = data["close"].pct_change().rolling(window=self.config.period).std()

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals["atr"] = atr
        signals["upper_band"] = upper_band
        signals["middle_band"] = middle_band
        signals["lower_band"] = lower_band
        signals["volatility"] = volatility
        signals["signal"] = 0

        # Generate signals based on price position relative to ATR bands
        # Buy signal when price touches lower band
        buy_condition = data["close"] <= lower_band

        # Sell signal when price touches upper band
        sell_condition = data["close"] >= upper_band

        # Apply volatility filter if enabled
        if self.config.use_volatility_filter:
            volatility_condition = volatility >= self.config.volatility_threshold
            buy_condition = buy_condition & volatility_condition
            sell_condition = sell_condition & volatility_condition

        signals.loc[buy_condition, "signal"] = 1
        signals.loc[sell_condition, "signal"] = -1

        # Filter signals based on volume and price
        volume_mask = data["volume"] >= self.config.min_volume
        price_mask = data["close"] >= self.config.min_price
        signals.loc[~(volume_mask & price_mask), "signal"] = 0

        # Add signal strength based on ATR
        signals["signal_strength"] = atr / atr.rolling(window=self.config.period).mean()
        signals["signal_strength"] = signals["signal_strength"].clip(0.5, 2.0)

        # Add stop loss levels
        signals["stop_loss_long"] = data["close"] - (self.config.multiplier * atr)
        signals["stop_loss_short"] = data["close"] + (self.config.multiplier * atr)

        self.signals = signals
        return signals

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading positions based on signals.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            pd.DataFrame: Positions DataFrame
        """
        if self.signals is None:
            self.generate_signals(data)

        positions = pd.DataFrame(index=data.index)
        positions["position"] = self.signals["signal"].cumsum()

        # Ensure positions are within bounds
        positions["position"] = positions["position"].clip(-1, 1)

        # Add position size based on signal strength
        positions["position_size"] = positions["position"] * self.signals["signal_strength"]

        # Add stop loss levels
        positions["stop_loss"] = np.where(
            positions["position"] > 0, self.signals["stop_loss_long"], self.signals["stop_loss_short"]
        )

        self.positions = positions
        return positions

    def calculate_dynamic_stop_loss(self, entry_price: float, atr_value: float, position_type: str = "long") -> float:
        """
        Calculate dynamic stop loss based on ATR.

        Args:
            entry_price: Entry price
            atr_value: Current ATR value
            position_type: 'long' or 'short'

        Returns:
            float: Stop loss price
        """
        if position_type == "long":
            return entry_price - (self.config.multiplier * atr_value)
        else:
            return entry_price + (self.config.multiplier * atr_value)

    def calculate_position_size(
        self, capital: float, risk_per_trade: float, entry_price: float, stop_loss: float
    ) -> float:
        """
        Calculate position size based on risk management.

        Args:
            capital: Available capital
            risk_per_trade: Risk per trade as percentage of capital
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            float: Position size
        """
        risk_amount = capital * (risk_per_trade / 100)
        price_risk = abs(entry_price - stop_loss)

        if price_risk > 0:
            return risk_amount / price_risk
        else:
            return 0

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information.

        Returns:
            Dict: Strategy information
        """
        return {
            "name": "ATR Strategy",
            "description": "Average True Range volatility-based strategy",
            "parameters": {
                "period": self.config.period,
                "multiplier": self.config.multiplier,
                "use_volatility_filter": self.config.use_volatility_filter,
                "volatility_threshold": self.config.volatility_threshold,
            },
            "signal_types": ["buy", "sell", "hold"],
            "indicators": ["atr", "bollinger_bands_atr"],
            "features": ["dynamic_stop_loss", "position_sizing"],
        }

    def optimize_parameters(self, data: pd.DataFrame, optimization_metric: str = "sharpe_ratio") -> ATRConfig:
        """
        Optimize strategy parameters using historical data.

        Args:
            data: Historical data for optimization
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')

        Returns:
            ATRConfig: Optimized configuration
        """
        self.logger.info("Optimizing ATR strategy parameters...")

        # Parameter ranges to test
        periods = [10, 14, 20, 30]
        multipliers = [1.5, 2.0, 2.5, 3.0]
        volatility_thresholds = [0.01, 0.02, 0.03, 0.05]

        best_config = None
        best_metric = float("-inf")

        for period in periods:
            for multiplier in multipliers:
                for vol_threshold in volatility_thresholds:
                    # Test configuration
                    test_config = ATRConfig(period=period, multiplier=multiplier, volatility_threshold=vol_threshold)

                    # Create temporary strategy with test config
                    temp_strategy = ATRStrategy(test_config)
                    signals = temp_strategy.generate_signals(data)

                    # Calculate performance metric
                    metric_value = self._calculate_performance_metric(signals, data, optimization_metric)

                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_config = test_config

        if best_config:
            self.config = best_config
            self.logger.info(f"Optimized parameters: {best_config}")

        return best_config or self.config

    def _calculate_performance_metric(self, signals: pd.DataFrame, data: pd.DataFrame, metric: str) -> float:
        """
        Calculate performance metric for optimization.

        Args:
            signals: Trading signals
            data: Price data
            metric: Metric to calculate

        Returns:
            float: Performance metric value
        """
        try:
            # Calculate returns
            returns = data["close"].pct_change()
            strategy_returns = signals["signal"].shift(1) * returns

            if metric == "sharpe_ratio":
                return strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
            elif metric == "total_return":
                return strategy_returns.sum()
            elif metric == "max_drawdown":
                cumulative_returns = (1 + strategy_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                return drawdown.min()
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating performance metric: {e}")
            return 0.0


def create_atr_strategy(config: Optional[ATRConfig] = None) -> ATRStrategy:
    """
    Create an ATR strategy instance.

    Args:
        config: Optional ATR configuration

    Returns:
        ATRStrategy: Configured ATR strategy
    """
    return ATRStrategy(config)


def generate_atr_signals(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Generate ATR trading signals.

    Args:
        data: DataFrame with OHLCV data
        **kwargs: Strategy parameters

    Returns:
        Dict: Signals and metadata
    """
    try:
        config = ATRConfig(**kwargs)
        strategy = ATRStrategy(config)
        signals = strategy.generate_signals(data)

        return {
            "success": True,
            "signals": signals,
            "strategy_info": strategy.get_strategy_info(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error generating ATR signals: {e}")
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
