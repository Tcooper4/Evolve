"""
Enhanced Strategy Pipeline for Signal Combination

This module provides comprehensive signal combination functionality for multiple trading strategies.
It integrates with the existing trading strategies system and provides multiple combination modes.

Features:
- Multiple combination modes: intersection, union, weighted, voting, confidence-based
- Integration with existing trading strategies
- Backward compatibility with existing code
- Advanced signal validation and conflict resolution
- Performance tracking and optimization
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for strategy combination."""

    name: str
    weight: float = 1.0
    confidence_threshold: float = 0.6
    enabled: bool = True
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class CombinationConfig:
    """Configuration for signal combination."""

    mode: str = (
        "intersection"  # 'intersection', 'union', 'weighted', 'voting', 'confidence'
    )
    min_agreement: float = 0.5  # Minimum agreement for consensus
    confidence_threshold: float = 0.6  # Minimum confidence for signal execution
    conflict_resolution: str = (
        "majority"  # 'majority', 'highest_confidence', 'weighted'
    )
    smoothing_window: Optional[int] = None  # Optional smoothing window
    enable_validation: bool = True  # Enable signal validation


class StrategyPipeline:
    """
    Enhanced strategy pipeline for combining multiple trading strategies.

    This class provides advanced signal combination functionality with multiple modes
    and integrates with the existing trading strategies system.
    """

    def __init__(
        self,
        strategies: Optional[List[StrategyConfig]] = None,
        combination_config: Optional[CombinationConfig] = None,
    ):
        """
        Initialize the strategy pipeline.

        Args:
            strategies: List of strategy configurations
            combination_config: Configuration for signal combination
        """
        self.strategies = strategies or []
        self.combination_config = combination_config or CombinationConfig()

        # Performance tracking
        self.performance_history = []
        self.signal_history = []

        # Integration with existing strategies
        self._load_existing_strategies()

        logger.info(
            f"StrategyPipeline initialized with {len(self.strategies)} strategies"
        )

    def _load_existing_strategies(self):
        """Load existing strategies from the trading strategies system."""
        try:
            # Import existing strategy functions
            from strategies.strategy_pipeline import (
                bollinger_strategy,
                macd_strategy,
                rsi_strategy,
                sma_strategy,
            )

            # Add default strategies if none provided
            if not self.strategies:
                self.strategies = [
                    StrategyConfig(name="RSI", weight=1.0, parameters={"window": 14}),
                    StrategyConfig(
                        name="MACD", weight=1.0, parameters={"fast": 12, "slow": 26}
                    ),
                    StrategyConfig(
                        name="Bollinger", weight=1.0, parameters={"window": 20}
                    ),
                    StrategyConfig(name="SMA", weight=1.0, parameters={"window": 20}),
                ]

            # Strategy function mapping
            self.strategy_functions = {
                "RSI": rsi_strategy,
                "MACD": macd_strategy,
                "Bollinger": bollinger_strategy,
                "SMA": sma_strategy,
            }

        except ImportError as e:
            logger.warning(f"Could not load existing strategies: {e}")
            self.strategy_functions = {}

    def add_strategy(
        self,
        name: str,
        function: Callable,
        weight: float = 1.0,
        confidence_threshold: float = 0.6,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a strategy to the pipeline.

        Args:
            name: Strategy name
            function: Strategy function
            weight: Strategy weight
            confidence_threshold: Confidence threshold
            parameters: Strategy parameters
        """
        strategy_config = StrategyConfig(
            name=name,
            weight=weight,
            confidence_threshold=confidence_threshold,
            parameters=parameters or {},
        )

        self.strategies.append(strategy_config)
        self.strategy_functions[name] = function

        logger.info(f"Added strategy: {name} with weight {weight}")

    def remove_strategy(self, name: str):
        """Remove a strategy from the pipeline."""
        self.strategies = [s for s in self.strategies if s.name != name]
        if name in self.strategy_functions:
            del self.strategy_functions[name]

        logger.info(f"Removed strategy: {name}")

    def combine_signals(
        self,
        signals_list: List[pd.Series],
        mode: Optional[str] = None,
        weights: Optional[List[float]] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Enhanced signal combination with multiple modes.

        Args:
            signals_list: List of signal series
            mode: Combination mode ('intersection', 'union', 'weighted', 'voting', 'confidence')
            weights: Optional weights for weighted combination
            **kwargs: Additional parameters

        Returns:
            Combined signal series
        """
        if not signals_list:
            raise ValueError("signals_list must not be empty")

        mode = mode or self.combination_config.mode

        # Validate signals
        if self.combination_config.enable_validation:
            signals_list = self._validate_signals(signals_list)

        # Combine signals based on mode
        if mode == "intersection":
            combined = self._combine_intersection(signals_list)
        elif mode == "union":
            combined = self._combine_union(signals_list)
        elif mode == "weighted":
            combined = self._combine_weighted(signals_list, weights)
        elif mode == "voting":
            combined = self._combine_voting(signals_list, weights)
        elif mode == "confidence":
            combined = self._combine_confidence_based(signals_list)
        else:
            raise ValueError(f"Unknown combination mode: {mode}")

        # Apply smoothing if configured
        if self.combination_config.smoothing_window:
            combined = self._apply_smoothing(
                combined, self.combination_config.smoothing_window
            )

        # Store signal history
        self.signal_history.append(
            {
                "timestamp": datetime.now(),
                "mode": mode,
                "signals_count": len(signals_list),
                "combined_signal": combined.copy(),
            }
        )

        logger.info(f"Combined {len(signals_list)} signals using {mode} mode")
        return combined

    def _combine_intersection(self, signals_list: List[pd.Series]) -> pd.Series:
        """Combine signals using intersection (all must agree)."""
        signals_matrix = np.vstack([s.fillna(0).values for s in signals_list])

        # All strategies must agree for signal
        buy = (signals_matrix == 1).all(axis=0)
        sell = (signals_matrix == -1).all(axis=0)

        result = np.zeros(signals_matrix.shape[1])
        result[buy] = 1
        result[sell] = -1

        return pd.Series(result, index=signals_list[0].index)

    def _combine_union(self, signals_list: List[pd.Series]) -> pd.Series:
        """Combine signals using union (any strategy triggers signal)."""
        signals_matrix = np.vstack([s.fillna(0).values for s in signals_list])

        # Any strategy can trigger signal
        buy = (signals_matrix == 1).any(axis=0)
        sell = (signals_matrix == -1).any(axis=0)

        result = np.zeros(signals_matrix.shape[1])
        result[buy] = 1
        result[sell] = -1

        return pd.Series(result, index=signals_list[0].index)

    def _combine_weighted(
        self, signals_list: List[pd.Series], weights: Optional[List[float]] = None
    ) -> pd.Series:
        """Combine signals using weighted average."""
        if weights is None:
            weights = [1.0] * len(signals_list)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Calculate weighted sum
        signals_matrix = np.vstack([s.fillna(0).values for s in signals_list])
        weighted_sum = np.dot(weights, signals_matrix)

        # Convert to directional signals
        result = np.sign(weighted_sum)

        return pd.Series(result, index=signals_list[0].index)

    def _combine_voting(
        self, signals_list: List[pd.Series], weights: Optional[List[float]] = None
    ) -> pd.Series:
        """Combine signals using voting mechanism."""
        if weights is None:
            weights = [1.0] * len(signals_list)

        signals_matrix = np.vstack([s.fillna(0).values for s in signals_list])
        weights = np.array(weights)

        # Calculate weighted votes
        weighted_votes = np.dot(weights, signals_matrix)

        # Determine signal based on voting threshold
        threshold = self.combination_config.min_agreement * np.sum(weights)
        result = np.where(
            weighted_votes > threshold, 1, np.where(weighted_votes < -threshold, -1, 0)
        )

        return pd.Series(result, index=signals_list[0].index)

    def _combine_confidence_based(self, signals_list: List[pd.Series]) -> pd.Series:
        """Combine signals using confidence-based weighting."""
        # Calculate confidence for each signal
        confidences = []
        for signal in signals_list:
            # Simple confidence based on signal strength and consistency
            signal_strength = np.abs(signal).mean()
            signal_consistency = 1 - np.std(signal) / (np.abs(signal).mean() + 1e-8)
            confidence = signal_strength * signal_consistency
            confidences.append(confidence)

        # Normalize confidences
        confidences = np.array(confidences)
        confidences = confidences / np.sum(confidences)

        # Apply confidence-based weighting
        return self._combine_weighted(signals_list, confidences.tolist())

    def _validate_signals(self, signals_list: List[pd.Series]) -> List[pd.Series]:
        """Validate and clean signal series."""
        validated_signals = []

        for signal in signals_list:
            # Fill NaN values
            signal = signal.fillna(0)

            # Ensure signal values are in [-1, 0, 1] range
            signal = np.sign(signal)

            # Remove outliers (optional)
            if len(signal) > 10:
                rolling_std = signal.rolling(window=10).std()
                outlier_mask = rolling_std > 2 * rolling_std.mean()
                signal[outlier_mask] = 0

            validated_signals.append(signal)

        return validated_signals

    def _apply_smoothing(self, signal: pd.Series, window: int) -> pd.Series:
        """Apply smoothing to the combined signal."""
        if window <= 1:
            return signal

        # Simple moving average smoothing
        smoothed = signal.rolling(window=window, center=True).mean()

        # Convert back to directional signals
        smoothed = np.sign(smoothed)

        return smoothed

    def generate_combined_signals(
        self, data: pd.DataFrame, strategy_names: Optional[List[str]] = None
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate combined signals from multiple strategies.

        Args:
            data: Market data
            strategy_names: List of strategy names to use (optional)

        Returns:
            Tuple of (combined_signals, metadata)
        """
        if strategy_names is None:
            strategy_names = [s.name for s in self.strategies if s.enabled]

        # Generate individual signals
        individual_signals = {}
        signals_list = []

        for strategy_name in strategy_names:
            if strategy_name in self.strategy_functions:
                try:
                    strategy_config = next(
                        s for s in self.strategies if s.name == strategy_name
                    )
                    function = self.strategy_functions[strategy_name]

                    # Apply strategy parameters
                    params = strategy_config.parameters or {}
                    signal = function(data, **params)

                    individual_signals[strategy_name] = signal
                    signals_list.append(signal)

                except Exception as e:
                    logger.error(f"Error generating signal for {strategy_name}: {e}")
                    continue

        if not signals_list:
            raise ValueError("No valid signals generated")

        # Combine signals
        combined_signal = self.combine_signals(signals_list)

        # Calculate metadata
        metadata = {
            "strategies_used": strategy_names,
            "combination_mode": self.combination_config.mode,
            "individual_signals": individual_signals,
            "signal_agreement": self._calculate_agreement(signals_list),
            "timestamp": datetime.now(),
        }

        return combined_signal, metadata

    def _calculate_agreement(self, signals_list: List[pd.Series]) -> float:
        """Calculate agreement level among strategies."""
        if len(signals_list) < 2:
            return 1.0

        # Convert to directional signals
        directional_signals = []
        for signal in signals_list:
            directional = np.sign(signal.fillna(0))
            directional_signals.append(directional)

        # Calculate pairwise agreement
        agreements = []
        for i in range(len(directional_signals)):
            for j in range(i + 1, len(directional_signals)):
                agreement = (directional_signals[i] == directional_signals[j]).mean()
                agreements.append(agreement)

        return np.mean(agreements) if agreements else 0.0

    def update_weights(self, performance_metrics: Dict[str, float]):
        """Update strategy weights based on performance."""
        for strategy in self.strategies:
            if strategy.name in performance_metrics:
                # Simple weight update based on Sharpe ratio
                sharpe = performance_metrics[strategy.name].get("sharpe_ratio", 0)
                strategy.weight = max(0.1, min(2.0, 1.0 + sharpe * 0.5))

        logger.info("Updated strategy weights based on performance")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the pipeline."""
        return {
            "total_signals_generated": len(self.signal_history),
            "strategies_count": len(self.strategies),
            "enabled_strategies": len([s for s in self.strategies if s.enabled]),
            "combination_mode": self.combination_config.mode,
            "last_signal_time": (
                self.signal_history[-1]["timestamp"] if self.signal_history else None
            ),
        }


# Backward compatibility functions
def combine_signals(
    signals_list: List[pd.Series],
    mode: str = "intersection",
    weights: Optional[List[float]] = None,
) -> pd.Series:
    """
    Backward compatibility function for signal combination.

    Args:
        signals_list: List of signal series
        mode: Combination mode ('intersection', 'union', 'weighted')
        weights: Optional weights for weighted combination

    Returns:
        Combined signal series
    """
    pipeline = StrategyPipeline()
    return pipeline.combine_signals(signals_list, mode, weights)


# Individual Strategy Functions (maintained for backward compatibility)
def rsi_strategy(
    data: pd.DataFrame, window: int = 14, overbought: float = 70, oversold: float = 30
) -> pd.Series:
    """RSI strategy: 1 for buy, -1 for sell, 0 for hold."""
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    signals = pd.Series(0, index=data.index)
    signals[rsi > overbought] = -1
    signals[rsi < oversold] = 1
    return signals


def macd_strategy(
    data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD strategy: 1 for buy, -1 for sell, 0 for hold."""
    ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    signals = pd.Series(0, index=data.index)
    signals[macd > macd_signal] = 1
    signals[macd < macd_signal] = -1
    return signals


def bollinger_strategy(
    data: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> pd.Series:
    """Bollinger Bands strategy: 1 for buy, -1 for sell, 0 for hold."""
    sma = data["close"].rolling(window=window).mean()
    std = data["close"].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    signals = pd.Series(0, index=data.index)
    signals[data["close"] > upper] = -1
    signals[data["close"] < lower] = 1
    return signals


def sma_strategy(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """SMA crossover strategy: 1 for buy, -1 for sell, 0 for hold."""
    sma = data["close"].rolling(window=window).mean()
    signals = pd.Series(0, index=data.index)
    signals[data["close"] > sma] = 1
    signals[data["close"] < sma] = -1
    return signals


# Strategy function mapping for UI integration
STRATEGY_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "RSI": rsi_strategy,
    "MACD": macd_strategy,
    "Bollinger": bollinger_strategy,
    "SMA": sma_strategy,
}

COMBINE_MODES = ["intersection", "union", "weighted", "voting", "confidence"]


def get_strategy_names() -> List[str]:
    """Get available strategy names."""
    return list(STRATEGY_FUNCTIONS.keys())


def get_combine_modes() -> List[str]:
    """Get available combination modes."""
    return COMBINE_MODES


def create_strategy_combo(
    strategy_names: List[str],
    mode: str = "intersection",
    weights: Optional[List[float]] = None,
) -> StrategyPipeline:
    """
    Create a strategy combination pipeline.

    Args:
        strategy_names: List of strategy names to combine
        mode: Combination mode
        weights: Optional weights for strategies

    Returns:
        Configured StrategyPipeline instance
    """
    # Create strategy configurations
    strategies = []
    for i, name in enumerate(strategy_names):
        weight = weights[i] if weights and i < len(weights) else 1.0
        strategy_config = StrategyConfig(
            name=name, weight=weight, parameters={}  # Default parameters
        )
        strategies.append(strategy_config)

    # Create combination configuration
    combination_config = CombinationConfig(mode=mode)

    # Create and configure pipeline
    pipeline = StrategyPipeline(strategies, combination_config)

    # Add strategy functions
    for name in strategy_names:
        if name in STRATEGY_FUNCTIONS:
            pipeline.strategy_functions[name] = STRATEGY_FUNCTIONS[name]

    return pipeline
