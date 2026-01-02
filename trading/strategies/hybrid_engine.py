"""Multi-Strategy Hybrid Engine.

This engine combines multiple trading strategies (RSI, MACD, Bollinger, Breakout, etc.)
with conditional filters and confidence scoring for optimal signal generation.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class StrategySignal:
    """Individual strategy signal."""

    strategy_name: str
    signal_type: SignalType
    confidence: float
    strength: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class HybridSignal:
    """Combined hybrid signal."""

    signal_type: SignalType
    confidence: float
    strength: float
    timestamp: datetime
    contributing_strategies: List[str]
    strategy_weights: Dict[str, float]
    consensus_score: float
    risk_level: str
    metadata: Dict[str, Any]


class HybridEngine:
    """Multi-strategy hybrid engine with conditional filters and confidence scoring."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hybrid engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Strategy weights and configurations
        self.strategy_weights = self.config.get(
            "strategy_weights",
            {
                "RSI Mean Reversion": 0.25,
                "MACD Strategy": 0.20,
                "Bollinger Bands": 0.20,
                "Moving Average Crossover": 0.15,
                "Breakout Strategy": 0.10,
                "Momentum Strategy": 0.10,
            },
        )

        # Conditional filters
        self.filters = self.config.get(
            "filters",
            {
                "trend_confirmation": True,
                "volume_confirmation": True,
                "volatility_filter": True,
                "correlation_filter": True,
            },
        )

        # Confidence thresholds
        self.confidence_thresholds = self.config.get(
            "confidence_thresholds",
            {
                "minimum_confidence": 0.6,
                "strong_signal_threshold": 0.8,
                "consensus_threshold": 0.7,
            },
        )

        # Risk management
        self.risk_config = self.config.get(
            "risk_config",
            {
                "max_position_size": 0.2,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.15,
                "max_drawdown": 0.1,
            },
        )

        # Strategy instances
        self.strategies = {}
        self.signal_history = []

        logger.info("Hybrid Engine initialized")

    def add_strategy(self, name: str, strategy_func: Callable, weight: float = 1.0):
        """Add a strategy to the hybrid engine.

        Args:
            name: Strategy name
            strategy_func: Strategy function that returns signals
            weight: Strategy weight
        """
        self.strategies[name] = {"function": strategy_func, "weight": weight}
        logger.info(f"Added strategy: {name} with weight {weight}")

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> HybridSignal:
        """Generate hybrid signals from all strategies.

        Args:
            data: Market data
            symbol: Trading symbol

        Returns:
            Combined hybrid signal
        """
        try:
            # Generate signals from all strategies
            strategy_signals = []

            for name, strategy_info in self.strategies.items():
                try:
                    signal = strategy_info["function"](data, symbol)
                    if signal:
                        strategy_signals.append(signal)
                except Exception as e:
                    logger.error(f"Error generating signal for {name}: {e}")

            if not strategy_signals:
                return self._create_default_signal()

            # Apply conditional filters
            filtered_signals = self._apply_filters(strategy_signals, data)

            # Combine signals
            hybrid_signal = self._combine_signals(filtered_signals, data)

            # Store in history
            self.signal_history.append(hybrid_signal)

            logger.info(
                f"Generated hybrid signal for {symbol}: {hybrid_signal.signal_type.value} "
                f"(confidence: {hybrid_signal.confidence:.2%})"
            )

            return hybrid_signal

        except Exception as e:
            logger.error(f"Error generating hybrid signals: {e}")
            return self._create_default_signal()

    def _apply_filters(
        self, signals: List[StrategySignal], data: pd.DataFrame
    ) -> List[StrategySignal]:
        """Apply conditional filters to strategy signals.

        Args:
            signals: List of strategy signals
            data: Market data

        Returns:
            Filtered signals
        """
        filtered_signals = []

        for signal in signals:
            should_include = True

            # Trend confirmation filter
            if self.filters.get("trend_confirmation", False):
                if not self._confirm_trend(signal, data):
                    should_include = False

            # Volume confirmation filter
            if self.filters.get("volume_confirmation", False):
                if not self._confirm_volume(signal, data):
                    should_include = False

            # Volatility filter
            if self.filters.get("volatility_filter", False):
                if not self._check_volatility(signal, data):
                    should_include = False

            # Correlation filter
            if self.filters.get("correlation_filter", False):
                if not self._check_correlation(signal, data):
                    should_include = False

            if should_include:
                filtered_signals.append(signal)

        return filtered_signals

    def _confirm_trend(self, signal: StrategySignal, data: pd.DataFrame) -> bool:
        """Confirm signal with trend analysis.

        Args:
            signal: Strategy signal
            data: Market data

        Returns:
            True if trend confirms signal
        """
        try:
            close = data["close"].values
            if len(close) < 20:
                return True

            # Calculate moving averages
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20

            current_price = close[-1]

            # Bullish signal confirmation
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                return current_price > sma_20 and sma_20 > sma_50

            # Bearish signal confirmation
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                return current_price < sma_20 and sma_20 < sma_50

            return True

        except Exception as e:
            logger.error(f"Error in trend confirmation: {e}")
            return True

    def _confirm_volume(self, signal: StrategySignal, data: pd.DataFrame) -> bool:
        """Confirm signal with volume analysis.

        Args:
            signal: Strategy signal
            data: Market data

        Returns:
            True if volume confirms signal
        """
        try:
            volume = data["volume"].values
            if len(volume) < 10:
                return True

            # Calculate volume moving average
            avg_volume = np.mean(volume[-10:])
            current_volume = volume[-1]

            # Volume should be above average for strong signals
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                return current_volume > avg_volume * 1.2

            return True

        except Exception as e:
            logger.error(f"Error in volume confirmation: {e}")
            return True

    def _check_volatility(self, signal: StrategySignal, data: pd.DataFrame) -> bool:
        """Check if volatility is acceptable for signal.

        Args:
            signal: Strategy signal
            data: Market data

        Returns:
            True if volatility is acceptable
        """
        try:
            close = data["close"].values
            if len(close) < 20:
                return True

            # Calculate volatility - Safely calculate returns with division-by-zero protection
            returns = np.where(
                close[:-1] > 1e-10,
                np.diff(close) / close[:-1],
                0.0
            )
            volatility = np.std(returns[-20:])

            # Reject signals in extremely high volatility
            if volatility > 0.05:  # 5% daily volatility
                return False

            return True

        except Exception as e:
            logger.error(f"Error in volatility check: {e}")
            return True

    def _check_correlation(self, signal: StrategySignal, data: pd.DataFrame) -> bool:
        """Check signal correlation with market conditions.

        Args:
            signal: Strategy signal
            data: Market data

        Returns:
            True if correlation is acceptable
        """
        # This is a placeholder for more sophisticated correlation analysis
        # Could include market regime, sector correlation, etc.
        return True

    def _combine_signals(
        self, signals: List[StrategySignal], data: pd.DataFrame
    ) -> HybridSignal:
        """Combine multiple strategy signals into a hybrid signal.

        Args:
            signals: List of strategy signals
            data: Market data

        Returns:
            Combined hybrid signal
        """
        if not signals:
            return self._create_default_signal()

        # Calculate weighted consensus
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0
        contributing_strategies = []
        strategy_weights = {}

        for signal in signals:
            weight = self.strategy_weights.get(signal.strategy_name, 1.0)
            confidence = signal.confidence

            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                buy_weight += weight * confidence
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                sell_weight += weight * confidence

            total_weight += weight
            contributing_strategies.append(signal.strategy_name)
            strategy_weights[signal.strategy_name] = weight

        # Determine signal type
        if total_weight == 0:
            signal_type = SignalType.HOLD
            confidence = 0.0
        elif buy_weight > sell_weight:
            if (
                buy_weight / total_weight
                > self.confidence_thresholds["strong_signal_threshold"]
            ):
                signal_type = SignalType.STRONG_BUY
            else:
                signal_type = SignalType.BUY
            confidence = buy_weight / total_weight
        elif sell_weight > buy_weight:
            if (
                sell_weight / total_weight
                > self.confidence_thresholds["strong_signal_threshold"]
            ):
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL
            confidence = sell_weight / total_weight
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5

        # Calculate consensus score
        consensus_score = self._calculate_consensus(signals)

        # Determine risk level
        risk_level = self._assess_risk_level(signals, data)

        # Calculate signal strength
        strength = self._calculate_strength(signals, confidence)

        return HybridSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=strength,
            timestamp=datetime.now(),
            contributing_strategies=contributing_strategies,
            strategy_weights=strategy_weights,
            consensus_score=consensus_score,
            risk_level=risk_level,
            metadata={
                "signal_count": len(signals),
                "filtered_count": len(signals),
                "market_conditions": self._analyze_market_conditions(data),
            },
        )

    def _calculate_consensus(self, signals: List[StrategySignal]) -> float:
        """Calculate consensus score among strategies.

        Args:
            signals: List of strategy signals

        Returns:
            Consensus score (0-1)
        """
        if not signals:
            return 0.0

        # Count signal types
        signal_counts = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

        # Find most common signal
        if signal_counts:
            max_count = max(signal_counts.values())
            consensus = max_count / len(signals)
            return consensus

        return 0.0

    def _assess_risk_level(
        self, signals: List[StrategySignal], data: pd.DataFrame
    ) -> str:
        """Assess risk level based on signals and market data.

        Args:
            signals: List of strategy signals
            data: Market data

        Returns:
            Risk level string
        """
        risk_score = 0

        # Signal-based risk
        for signal in signals:
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                risk_score += 2
            elif signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                risk_score += 1

        # Market-based risk
        try:
            close = data["close"].values
            if len(close) >= 20:
                # Safely calculate returns with division-by-zero protection
                returns = np.where(
                    close[:-1] > 1e-10,
                    np.diff(close) / close[:-1],
                    0.0
                )
                volatility = np.std(returns[-20:])

                if volatility > 0.04:
                    risk_score += 2
                elif volatility > 0.02:
                    risk_score += 1
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.debug(f"Risk calculation failed: {e}")

        # Determine risk level
        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_strength(
        self, signals: List[StrategySignal], confidence: float
    ) -> float:
        """Calculate signal strength.

        Args:
            signals: List of strategy signals
            confidence: Overall confidence

        Returns:
            Signal strength (0-1)
        """
        if not signals:
            return 0.0

        # Average strength of contributing signals
        avg_strength = np.mean([s.strength for s in signals])

        # Combine with confidence
        strength = (avg_strength + confidence) / 2

        return min(strength, 1.0)

    def _analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions.

        Args:
            data: Market data

        Returns:
            Market conditions dictionary
        """
        try:
            close = data["close"].values
            volume = data["volume"].values

            conditions = {}

            if len(close) >= 20:
                # Trend analysis
                sma_20 = np.mean(close[-20:])
                current_price = close[-1]
                conditions["trend"] = "bullish" if current_price > sma_20 else "bearish"

                # Volatility analysis
                # Safely calculate returns with division-by-zero protection
                returns = np.where(
                    close[:-1] > 1e-10,
                    np.diff(close) / close[:-1],
                    0.0
                )
                volatility = np.std(returns[-20:])
                conditions["volatility"] = "high" if volatility > 0.03 else "low"

                # Volume analysis
                if len(volume) >= 10:
                    avg_volume = np.mean(volume[-10:])
                    current_volume = volume[-1]
                    conditions["volume"] = (
                        "high" if current_volume > avg_volume else "low"
                    )

            return conditions

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {}

    def _create_default_signal(self) -> HybridSignal:
        """Create default signal when no strategies generate signals.

        Returns:
            Default hybrid signal
        """
        return HybridSignal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strength=0.0,
            timestamp=datetime.now(),
            contributing_strategies=[],
            strategy_weights={},
            consensus_score=0.0,
            risk_level="LOW",
            metadata={"note": "No strategies generated signals"},
        )

    def get_signal_history(self, days: int = 30) -> List[HybridSignal]:
        """Get recent signal history.

        Args:
            days: Number of days to look back

        Returns:
            List of recent signals
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        return [s for s in self.signal_history if s.timestamp > cutoff_date]

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the hybrid engine.

        Returns:
            Performance metrics
        """
        if not self.signal_history:
            return {}

        # Calculate metrics based on signal accuracy
        # This would require actual trade results to be meaningful
        metrics = {
            "total_signals": len(self.signal_history),
            "buy_signals": len(
                [
                    s
                    for s in self.signal_history
                    if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]
                ]
            ),
            "sell_signals": len(
                [
                    s
                    for s in self.signal_history
                    if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]
                ]
            ),
            "hold_signals": len(
                [s for s in self.signal_history if s.signal_type == SignalType.HOLD]
            ),
            "avg_confidence": np.mean([s.confidence for s in self.signal_history]),
            "avg_consensus": np.mean([s.consensus_score for s in self.signal_history]),
        }

        return metrics

    def update_strategy_weights(self, new_weights: Dict[str, float]):
        """Update strategy weights based on performance.

        Args:
            new_weights: New strategy weights
        """
        self.strategy_weights.update(new_weights)
        logger.info(f"Updated strategy weights: {new_weights}")

    def export_signals(self, filepath: str) -> bool:
        """Export signal history to file.

        Args:
            filepath: Output file path

        Returns:
            True if export successful
        """
        try:
            signal_data = []

            for signal in self.signal_history:
                row = {
                    "timestamp": signal.timestamp,
                    "signal_type": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength,
                    "consensus_score": signal.consensus_score,
                    "risk_level": signal.risk_level,
                    "contributing_strategies": ",".join(signal.contributing_strategies),
                    "strategy_count": len(signal.contributing_strategies),
                }
                signal_data.append(row)

            df = pd.DataFrame(signal_data)
            df.to_csv(filepath, index=False)

            logger.info(f"Signal history exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting signals: {e}")
            return False


# Global hybrid engine instance
hybrid_engine = HybridEngine()


def get_hybrid_engine() -> HybridEngine:
    """Get the global hybrid engine instance."""
    return hybrid_engine
