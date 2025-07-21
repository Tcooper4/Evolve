"""
Strategy Gatekeeper for Evolve Trading Platform

A comprehensive strategy management system that provides:
- Market regime classification and detection
- Dynamic strategy selection based on performance and market conditions
- Risk management and position sizing
- Performance monitoring and strategy retirement
- Multi-timeframe analysis and regime transitions
- Advanced decision-making with confidence scoring
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    SIDEWAYS = "sideways"
    TRENDING = "trending"


class StrategyStatus(Enum):
    """Strategy status indicators."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    RETIRED = "retired"
    TESTING = "testing"
    OPTIMIZING = "optimizing"


class GatekeeperDecision(Enum):
    """Gatekeeper decision types."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SUSPEND = "suspend"
    RETIRE = "retire"
    TEST = "test"


@dataclass
class RegimeMetrics:
    """Market regime metrics."""

    regime: MarketRegime
    confidence: float
    volatility: float
    momentum: float
    trend_strength: float
    volume_profile: str
    support_resistance: Dict[str, float]
    regime_duration: int
    transition_probability: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""

    strategy_name: str
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    regime_performance: Dict[MarketRegime, float]
    recent_performance: float
    risk_adjusted_return: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GatekeeperDecision:
    """Gatekeeper decision with reasoning."""

    decision: GatekeeperDecision
    strategy_name: str
    confidence: float
    reasoning: List[str]
    modifications: Dict[str, Any]
    risk_score: float
    expected_return: float
    position_size: float
    stop_loss: float
    take_profit: float
    timestamp: datetime = field(default_factory=datetime.now)


class RegimeClassifier:
    """Advanced market regime classifier using multiple indicators."""

    def __init__(self, lookback_period: int = 252):
        """Initialize regime classifier.

        Args:
            lookback_period: Number of days for regime analysis
        """
        self.lookback_period = lookback_period
        self.regime_history = []
        self.transition_matrix = None

    def classify_regime(self, data: pd.DataFrame) -> RegimeMetrics:
        """Classify current market regime using multiple indicators.

        Args:
            data: OHLCV data with datetime index

        Returns:
            RegimeMetrics object with regime classification and confidence
        """
        if len(data) < self.lookback_period:
            return self._fallback_regime()

        # Calculate technical indicators
        returns = data["Close"].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]
        momentum = self._calculate_momentum(data)
        trend_strength = self._calculate_trend_strength(data)
        volume_profile = self._analyze_volume_profile(data)
        support_resistance = self._calculate_support_resistance(data)

        # Determine regime based on multiple factors
        regime, confidence = self._determine_regime(
            volatility, momentum, trend_strength, volume_profile
        )

        # Calculate regime duration and transition probability
        regime_duration = self._calculate_regime_duration(regime)
        transition_probability = self._calculate_transition_probability(regime)

        metrics = RegimeMetrics(
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            momentum=momentum,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            support_resistance=support_resistance,
            regime_duration=regime_duration,
            transition_probability=transition_probability,
        )

        self.regime_history.append(metrics)
        return metrics

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum indicator."""
        # RSI-based momentum
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD-based momentum
        exp1 = data["Close"].ewm(span=12).mean()
        exp2 = data["Close"].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()

        # Combine momentum indicators
        momentum = (rsi.iloc[-1] - 50) / 50 + (macd.iloc[-1] - signal.iloc[-1]) / data[
            "Close"
        ].iloc[-1]
        return momentum

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX."""
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # Calculate Directional Movement
        dm_plus = (high - high.shift()).where(
            (high - high.shift()) > (low.shift() - low), 0
        )
        dm_minus = (low.shift() - low).where(
            (low.shift() - low) > (high - high.shift()), 0
        )

        # Calculate ADX
        di_plus = 100 * (dm_plus.rolling(window=14).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=14).mean() / atr)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=14).mean()

        return adx.iloc[-1] / 100  # Normalize to 0-1

    def _analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """Analyze volume profile."""
        volume_ma = data["Volume"].rolling(window=20).mean()
        current_volume = data["Volume"].iloc[-1]
        avg_volume = volume_ma.iloc[-1]

        if current_volume > avg_volume * 1.5:
            return "high"
        elif current_volume < avg_volume * 0.5:
            return "low"
        else:
            return "normal"

    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        high = data["High"].rolling(window=20).max()
        low = data["Low"].rolling(window=20).min()

        return {
            "resistance": high.iloc[-1],
            "support": low.iloc[-1],
            "current": data["Close"].iloc[-1],
        }

    def _determine_regime(
        self,
        volatility: float,
        momentum: float,
        trend_strength: float,
        volume_profile: str,
    ) -> Tuple[MarketRegime, float]:
        """Determine market regime based on indicators."""
        # High volatility scenarios
        if volatility > 0.03:
            if momentum < -0.2:
                return MarketRegime.CRISIS, 0.85
            elif momentum > 0.2:
                return MarketRegime.VOLATILE, 0.75
            else:
                return MarketRegime.VOLATILE, 0.70

        # Low volatility scenarios
        if volatility < 0.01:
            if trend_strength > 0.7:
                return MarketRegime.TRENDING, 0.80
            else:
                return MarketRegime.SIDEWAYS, 0.75

        # Normal volatility scenarios
        if momentum > 0.3 and trend_strength > 0.6:
            return MarketRegime.BULL, 0.85
        elif momentum < -0.3 and trend_strength > 0.6:
            return MarketRegime.BEAR, 0.85
        elif abs(momentum) < 0.1:
            return MarketRegime.NEUTRAL, 0.70
        else:
            return MarketRegime.SIDEWAYS, 0.65

    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long current regime has been active."""
        if not self.regime_history:
            return 1

        duration = 1
        for metrics in reversed(self.regime_history[-10:]):
            if metrics.regime == current_regime:
                duration += 1
            else:
                break

        return duration

    def _calculate_transition_probability(self, current_regime: MarketRegime) -> float:
        """Calculate probability of regime transition."""
        if len(self.regime_history) < 10:
            return 0.1

        # Simple transition probability based on regime duration
        duration = self._calculate_regime_duration(current_regime)
        if duration > 50:
            return 0.8  # High probability of transition
        elif duration > 20:
            return 0.5  # Medium probability
        else:
            return 0.2  # Low probability

    def _fallback_regime(self) -> RegimeMetrics:
        """Fallback regime when insufficient data."""
        return RegimeMetrics(
            regime=MarketRegime.NEUTRAL,
            confidence=0.5,
            volatility=0.02,
            momentum=0.0,
            trend_strength=0.5,
            volume_profile="normal",
            support_resistance={"support": 0, "resistance": 0, "current": 0},
            regime_duration=1,
            transition_probability=0.1,
        )


class StrategyGatekeeper:
    """Comprehensive strategy gatekeeper with institutional-grade capabilities."""

    def __init__(self, strategy_configs: Optional[Dict[str, Any]] = None):
        """Initialize strategy gatekeeper.

        Args:
            strategy_configs: Dictionary of strategy configurations
        """
        self.strategy_configs = strategy_configs or {}
        self.regime_classifier = RegimeClassifier()
        self.strategy_performance = {}
        self.decision_history = []
        self.active_strategies = {}
        self.retired_strategies = set()

        # Performance thresholds
        self.min_sharpe_ratio = 0.5
        self.max_drawdown_threshold = 0.15
        self.min_win_rate = 0.45
        self.performance_window = 252  # 1 year

        # Risk management
        self.max_position_size = 0.1  # 10% of portfolio
        self.min_position_size = 0.01  # 1% of portfolio
        self.max_correlation = 0.7

        # Initialize logging
        self._setup_logging()

        logger.info("Strategy Gatekeeper initialized successfully")

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path("logs/strategy_gatekeeper")
        log_dir.mkdir(parents=True, exist_ok=True)

        # File handler for detailed logs
        file_handler = logging.FileHandler(log_dir / "gatekeeper.log")
        file_handler.setLevel(logging.DEBUG)

        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def evaluate_strategy(
        self, strategy_name: str, data: pd.DataFrame, strategy_signals: pd.DataFrame
    ) -> GatekeeperDecision:
        """Evaluate a strategy and make a decision.

        Args:
            strategy_name: Name of the strategy
            data: Market data (OHLCV)
            strategy_signals: Strategy signals dataframe

        Returns:
            GatekeeperDecision with approval/rejection and modifications
        """
        try:
            # Classify current market regime
            regime_metrics = self.regime_classifier.classify_regime(data)

            # Calculate strategy performance
            performance = self._calculate_strategy_performance(
                strategy_name, data, strategy_signals, regime_metrics
            )

            # Update performance history
            self.strategy_performance[strategy_name] = performance

            # Make decision based on performance and regime
            decision = self._make_decision(strategy_name, performance, regime_metrics)

            # Log decision
            self._log_decision(decision)

            return decision

        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy_name}: {e}")
            return self._create_rejection_decision(
                strategy_name, f"Evaluation error: {e}"
            )

    def _calculate_strategy_performance(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        regime_metrics: RegimeMetrics,
    ) -> StrategyPerformance:
        """Calculate comprehensive strategy performance metrics."""
        try:
            # Calculate returns
            returns = data["Close"].pct_change().dropna()
            strategy_returns = self._calculate_strategy_returns(signals, returns)

            # Calculate performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(strategy_returns)
            max_drawdown = self._calculate_max_drawdown(strategy_returns)
            total_return = (1 + strategy_returns).prod() - 1
            win_rate = self._calculate_win_rate(strategy_returns)
            profit_factor = self._calculate_profit_factor(strategy_returns)
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(strategy_returns)

            # Calculate regime-specific performance
            regime_performance = self._calculate_regime_performance(
                strategy_returns, regime_metrics.regime
            )

            # Calculate recent performance (last 30 days)
            recent_performance = strategy_returns.tail(30).mean() * 252

            # Calculate risk-adjusted return
            risk_adjusted_return = sharpe_ratio * total_return

            return StrategyPerformance(
                strategy_name=strategy_name,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_return=total_return,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                regime_performance=regime_performance,
                recent_performance=recent_performance,
                risk_adjusted_return=risk_adjusted_return,
            )

        except Exception as e:
            logger.error(f"Error calculating performance for {strategy_name}: {e}")
            return self._create_fallback_performance(strategy_name)

    def _calculate_strategy_returns(
        self, signals: pd.DataFrame, returns: pd.Series
    ) -> pd.Series:
        """Calculate strategy returns from signals."""
        # Simple long-only strategy for now
        # In a real implementation, this would handle long/short positions
        strategy_returns = signals["signal"].shift(1) * returns
        return strategy_returns.fillna(0)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - 0.02 / 252  # Assuming 2% risk-free rate
        if excess_returns.std() == 0:
            return 0.0

        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / losses if losses > 0 else float("inf")

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - 0.02 / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_regime_performance(
        self, returns: pd.Series, current_regime: MarketRegime
    ) -> Dict[MarketRegime, float]:
        """Calculate performance by market regime."""
        # This is a simplified version - in practice, you'd track regime changes
        # and calculate performance for each regime separately
        return {current_regime: returns.mean() * 252}

    def _make_decision(
        self,
        strategy_name: str,
        performance: StrategyPerformance,
        regime_metrics: RegimeMetrics,
    ) -> GatekeeperDecision:
        """Make comprehensive decision about strategy."""
        reasoning = []
        modifications = {}
        risk_score = 0.0
        expected_return = performance.total_return

        # Check performance thresholds
        if performance.sharpe_ratio < self.min_sharpe_ratio:
            reasoning.append(
                f"Sharpe ratio {performance.sharpe_ratio:.3f} below threshold {self.min_sharpe_ratio}"
            )
            risk_score += 0.3

        if performance.max_drawdown > self.max_drawdown_threshold:
            reasoning.append(
                f"Max drawdown {performance.max_drawdown:.3f} above threshold {self.max_drawdown_threshold}"
            )
            risk_score += 0.4

        if performance.win_rate < self.min_win_rate:
            reasoning.append(
                f"Win rate {performance.win_rate:.3f} below threshold {self.min_win_rate}"
            )
            risk_score += 0.2

        # Check regime compatibility
        strategy_config = self.strategy_configs.get(strategy_name, {})
        preferred_regimes = strategy_config.get("preferred_regimes", [])

        if preferred_regimes and regime_metrics.regime.value not in preferred_regimes:
            reasoning.append(
                f"Strategy not optimized for current regime: {regime_metrics.regime.value}"
            )
            risk_score += 0.2

        # Calculate position size based on performance and risk
        base_position_size = self.max_position_size
        if performance.sharpe_ratio > 1.5:
            base_position_size *= 1.2
        elif performance.sharpe_ratio < 0.5:
            base_position_size *= 0.5

        if risk_score > 0.5:
            base_position_size *= 0.5

        position_size = max(
            self.min_position_size, min(self.max_position_size, base_position_size)
        )

        # Calculate stop loss and take profit
        stop_loss = performance.max_drawdown * 1.5
        take_profit = performance.total_return * 0.5

        # Determine decision
        if risk_score > 0.8:
            decision_type = GatekeeperDecision.REJECT
            confidence = 0.9
        elif risk_score > 0.5:
            decision_type = GatekeeperDecision.MODIFY
            confidence = 0.7
        elif performance.sharpe_ratio > 1.0 and performance.win_rate > 0.55:
            decision_type = GatekeeperDecision.APPROVE
            confidence = 0.8
        else:
            decision_type = GatekeeperDecision.TEST
            confidence = 0.6

        return GatekeeperDecision(
            decision=decision_type,
            strategy_name=strategy_name,
            confidence=confidence,
            reasoning=reasoning,
            modifications=modifications,
            risk_score=risk_score,
            expected_return=expected_return,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _create_rejection_decision(
        self, strategy_name: str, reason: str
    ) -> GatekeeperDecision:
        """Create a rejection decision."""
        return GatekeeperDecision(
            decision=GatekeeperDecision.REJECT,
            strategy_name=strategy_name,
            confidence=0.9,
            reasoning=[reason],
            modifications={},
            risk_score=1.0,
            expected_return=0.0,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
        )

    def _create_fallback_performance(self, strategy_name: str) -> StrategyPerformance:
        """Create fallback performance metrics."""
        return StrategyPerformance(
            strategy_name=strategy_name,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            regime_performance={},
            recent_performance=0.0,
            risk_adjusted_return=0.0,
        )

    def _log_decision(self, decision: GatekeeperDecision):
        """Log decision for audit trail."""
        self.decision_history.append(decision)

        log_entry = {
            "timestamp": decision.timestamp.isoformat(),
            "strategy": decision.strategy_name,
            "decision": decision.decision.value,
            "confidence": decision.confidence,
            "risk_score": decision.risk_score,
            "expected_return": decision.expected_return,
            "position_size": decision.position_size,
            "reasoning": decision.reasoning,
        }

        logger.info(f"Gatekeeper Decision: {json.dumps(log_entry, indent=2)}")

    def get_active_strategies(self) -> Dict[str, StrategyPerformance]:
        """Get currently active strategies."""
        return {
            name: perf
            for name, perf in self.strategy_performance.items()
            if perf.sharpe_ratio > self.min_sharpe_ratio
            and perf.max_drawdown < self.max_drawdown_threshold
        }

    def get_strategy_recommendations(
        self, regime: MarketRegime
    ) -> List[Tuple[str, float]]:
        """Get strategy recommendations for current regime."""
        active_strategies = self.get_active_strategies()
        recommendations = []

        for name, performance in active_strategies.items():
            # Calculate regime-specific score
            regime_performance = performance.regime_performance.get(regime, 0.0)
            score = (
                performance.sharpe_ratio * 0.4
                + (1 - performance.max_drawdown) * 0.3
                + regime_performance * 0.3
            )

            recommendations.append((name, score))

        # Sort by score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def save_gatekeeper_state(self, filepath: str):
        """Save gatekeeper state for persistence."""
        state = {
            "strategy_performance": {
                name: {
                    "sharpe_ratio": perf.sharpe_ratio,
                    "max_drawdown": perf.max_drawdown,
                    "total_return": perf.total_return,
                    "win_rate": perf.win_rate,
                    "timestamp": perf.timestamp.isoformat(),
                }
                for name, perf in self.strategy_performance.items()
            },
            "decision_history": [
                {
                    "strategy_name": d.strategy_name,
                    "decision": d.decision.value,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat(),
                }
                for d in self.decision_history[-100:]  # Keep last 100 decisions
            ],
            "active_strategies": list(self.active_strategies.keys()),
            "retired_strategies": list(self.retired_strategies),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Gatekeeper state saved to {filepath}")

    def load_gatekeeper_state(self, filepath: str):
        """Load gatekeeper state from file."""
        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # Restore strategy performance
            for name, perf_data in state.get("strategy_performance", {}).items():
                self.strategy_performance[name] = StrategyPerformance(
                    strategy_name=name,
                    sharpe_ratio=perf_data["sharpe_ratio"],
                    max_drawdown=perf_data["max_drawdown"],
                    total_return=perf_data["total_return"],
                    win_rate=perf_data["win_rate"],
                    profit_factor=0.0,  # Not stored in legacy format
                    calmar_ratio=0.0,
                    sortino_ratio=0.0,
                    regime_performance={},
                    recent_performance=0.0,
                    risk_adjusted_return=0.0,
                    timestamp=datetime.fromisoformat(perf_data["timestamp"]),
                )

            # Restore other state
            self.active_strategies = {
                name: True for name in state.get("active_strategies", [])
            }
            self.retired_strategies = set(state.get("retired_strategies", []))

            logger.info(f"Gatekeeper state loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading gatekeeper state: {e}")


def create_strategy_gatekeeper(
    config: Optional[Dict[str, Any]] = None,
) -> StrategyGatekeeper:
    """Factory function to create a strategy gatekeeper.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured StrategyGatekeeper instance
    """
    return StrategyGatekeeper(config)
