"""
Strategy Fallback - Batch 18
Enhanced strategy fallback with ranked fallback pool based on historical performance
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Available fallback strategies."""

    RSI = "RSI"
    SMA = "SMA"
    MACD = "MACD"
    BOLLINGER = "Bollinger"
    MOMENTUM = "Momentum"
    MEAN_REVERSION = "MeanReversion"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""

    strategy_name: str
    win_rate: float
    total_trades: int
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime
    confidence_score: float = 0.0


@dataclass
class FallbackResult:
    """Result of fallback strategy execution."""

    strategy_name: str
    signal: str
    confidence: float
    performance_metrics: Dict[str, float]
    execution_time: float
    fallback_rank: int
    timestamp: datetime = field(default_factory=datetime.now)


class StrategyFallback:
    """
    Enhanced strategy fallback with ranked fallback pool.

    Features:
    - Ranked fallback strategies based on historical win rates
    - Dynamic performance tracking
    - Multiple fallback strategies (RSI, SMA, MACD)
    - Performance-based strategy selection
    """

    def __init__(
        self,
        fallback_pool: Optional[List[str]] = None,
        performance_window: int = 30,
        min_trades_for_ranking: int = 5,
    ):
        """
        Initialize strategy fallback.

        Args:
            fallback_pool: List of fallback strategy names
            performance_window: Days to look back for performance
            min_trades_for_ranking: Minimum trades required for ranking
        """
        self.fallback_pool = fallback_pool or ["RSI", "SMA", "MACD"]
        self.performance_window = performance_window
        self.min_trades_for_ranking = min_trades_for_ranking

        # Performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.trade_history: List[Dict[str, Any]] = []

        # Initialize default performance for fallback strategies
        self._initialize_default_performance()

        logger.info(f"StrategyFallback initialized with pool: {self.fallback_pool}")

    def _initialize_default_performance(self):
        """Initialize default performance metrics for fallback strategies."""
        default_performance = {
            "RSI": {"win_rate": 0.55, "avg_return": 0.02, "sharpe": 0.8},
            "SMA": {"win_rate": 0.52, "avg_return": 0.015, "sharpe": 0.6},
            "MACD": {"win_rate": 0.58, "avg_return": 0.025, "sharpe": 0.9},
            "Bollinger": {"win_rate": 0.53, "avg_return": 0.018, "sharpe": 0.7},
            "Momentum": {"win_rate": 0.56, "avg_return": 0.022, "sharpe": 0.85},
            "MeanReversion": {"win_rate": 0.54, "avg_return": 0.019, "sharpe": 0.75},
        }

        for strategy_name in self.fallback_pool:
            if strategy_name in default_performance:
                perf = default_performance[strategy_name]
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    win_rate=perf["win_rate"],
                    total_trades=10,  # Default trade count
                    avg_return=perf["avg_return"],
                    sharpe_ratio=perf["sharpe"],
                    max_drawdown=0.05,
                    last_updated=datetime.now(),
                    confidence_score=0.7,
                )
            else:
                # Initialize with conservative defaults
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    win_rate=0.5,
                    total_trades=0,
                    avg_return=0.01,
                    sharpe_ratio=0.5,
                    max_drawdown=0.1,
                    last_updated=datetime.now(),
                    confidence_score=0.5,
                )

    def get_ranked_fallbacks(self) -> List[Tuple[str, float]]:
        """
        Get ranked fallback strategies based on historical performance.

        Returns:
            List of (strategy_name, score) tuples ranked by performance
        """
        ranked_strategies = []

        for strategy_name in self.fallback_pool:
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]

                # Calculate composite score
                score = self._calculate_strategy_score(perf)
                ranked_strategies.append((strategy_name, score))

        # Sort by score (descending)
        ranked_strategies.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Ranked fallbacks: {ranked_strategies}")
        return ranked_strategies

    def _calculate_strategy_score(self, performance: StrategyPerformance) -> float:
        """
        Calculate composite score for strategy ranking.

        Args:
            performance: Strategy performance metrics

        Returns:
            Composite score (higher is better)
        """
        # Weighted combination of metrics
        win_rate_weight = 0.4
        sharpe_weight = 0.3
        return_weight = 0.2
        confidence_weight = 0.1

        # Normalize metrics
        normalized_win_rate = min(performance.win_rate, 0.8) / 0.8  # Cap at 80%
        normalized_sharpe = max(0, min(performance.sharpe_ratio, 2.0)) / 2.0
        normalized_return = max(0, min(performance.avg_return, 0.05)) / 0.05

        # Penalty for insufficient data
        data_penalty = 1.0
        if performance.total_trades < self.min_trades_for_ranking:
            data_penalty = 0.7

        score = (
            win_rate_weight * normalized_win_rate
            + sharpe_weight * normalized_sharpe
            + return_weight * normalized_return
            + confidence_weight * performance.confidence_score
        ) * data_penalty

        return score

    def execute_fallback(
        self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> FallbackResult:
        """
        Execute the best fallback strategy.

        Args:
            market_data: Market data for strategy execution
            context: Additional context

        Returns:
            FallbackResult with execution details
        """
        start_time = datetime.now()

        # Get ranked fallbacks
        ranked_fallbacks = self.get_ranked_fallbacks()

        if not ranked_fallbacks:
            logger.error("No fallback strategies available")
            return self._create_error_result("No fallbacks available")

        # Try strategies in order of ranking
        for rank, (strategy_name, score) in enumerate(ranked_fallbacks):
            try:
                result = self._execute_strategy(strategy_name, market_data, context)
                if result:
                    result.fallback_rank = rank + 1
                    execution_time = (datetime.now() - start_time).total_seconds()
                    result.execution_time = execution_time

                    logger.info(
                        f"Executed fallback strategy: {strategy_name} (rank {rank + 1}, score: {score:.3f})"
                    )
                    return result

            except Exception as e:
                logger.warning(
                    f"Failed to execute fallback strategy {strategy_name}: {e}"
                )
                continue

        # If all strategies fail, return error result
        logger.error("All fallback strategies failed")
        return self._create_error_result("All fallbacks failed")

    def _execute_strategy(
        self,
        strategy_name: str,
        market_data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[FallbackResult]:
        """
        Execute a specific fallback strategy.

        Args:
            strategy_name: Name of the strategy
            market_data: Market data
            context: Execution context

        Returns:
            FallbackResult or None if execution fails
        """
        try:
            if strategy_name == "RSI":
                return self._execute_rsi_strategy(market_data, context)
            elif strategy_name == "SMA":
                return self._execute_sma_strategy(market_data, context)
            elif strategy_name == "MACD":
                return self._execute_macd_strategy(market_data, context)
            elif strategy_name == "Bollinger":
                return self._execute_bollinger_strategy(market_data, context)
            elif strategy_name == "Momentum":
                return self._execute_momentum_strategy(market_data, context)
            elif strategy_name == "MeanReversion":
                return self._execute_mean_reversion_strategy(market_data, context)
            else:
                logger.warning(f"Unknown fallback strategy: {strategy_name}")
                return None

        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {e}")
            return None

    def _execute_rsi_strategy(
        self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> FallbackResult:
        """Execute RSI strategy."""
        try:
            # Simple RSI logic
            if "close" in market_data.columns:
                close_prices = market_data["close"]
                rsi = self._calculate_rsi(close_prices, period=14)
                current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50

                if current_rsi < 30:
                    signal = "BUY"
                    confidence = 0.8
                elif current_rsi > 70:
                    signal = "SELL"
                    confidence = 0.8
                else:
                    signal = "HOLD"
                    confidence = 0.5
            else:
                signal = "HOLD"
                confidence = 0.3

            return FallbackResult(
                strategy_name="RSI",
                signal=signal,
                confidence=confidence,
                performance_metrics=self._get_performance_metrics("RSI"),
                execution_time=0.0,
                fallback_rank=0,
            )

        except Exception as e:
            logger.error(f"Error in RSI strategy: {e}")
            return self._create_error_result(f"RSI error: {e}")

    def _execute_sma_strategy(
        self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> FallbackResult:
        """Execute SMA strategy."""
        try:
            if "close" in market_data.columns:
                close_prices = market_data["close"]
                sma_short = close_prices.rolling(window=10).mean()
                sma_long = close_prices.rolling(window=20).mean()

                if len(sma_short) > 0 and len(sma_long) > 0:
                    current_short = sma_short.iloc[-1]
                    current_long = sma_long.iloc[-1]

                    if current_short > current_long:
                        signal = "BUY"
                        confidence = 0.7
                    else:
                        signal = "SELL"
                        confidence = 0.7
                else:
                    signal = "HOLD"
                    confidence = 0.3
            else:
                signal = "HOLD"
                confidence = 0.3

            return FallbackResult(
                strategy_name="SMA",
                signal=signal,
                confidence=confidence,
                performance_metrics=self._get_performance_metrics("SMA"),
                execution_time=0.0,
                fallback_rank=0,
            )

        except Exception as e:
            logger.error(f"Error in SMA strategy: {e}")
            return self._create_error_result(f"SMA error: {e}")

    def _execute_macd_strategy(
        self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> FallbackResult:
        """Execute MACD strategy."""
        try:
            if "close" in market_data.columns:
                close_prices = market_data["close"]
                macd, signal_line = self._calculate_macd(close_prices)

                if len(macd) > 0 and len(signal_line) > 0:
                    current_macd = macd.iloc[-1]
                    current_signal = signal_line.iloc[-1]

                    if current_macd > current_signal:
                        signal = "BUY"
                        confidence = 0.75
                    else:
                        signal = "SELL"
                        confidence = 0.75
                else:
                    signal = "HOLD"
                    confidence = 0.3
            else:
                signal = "HOLD"
                confidence = 0.3

            return FallbackResult(
                strategy_name="MACD",
                signal=signal,
                confidence=confidence,
                performance_metrics=self._get_performance_metrics("MACD"),
                execution_time=0.0,
                fallback_rank=0,
            )

        except Exception as e:
            logger.error(f"Error in MACD strategy: {e}")
            return self._create_error_result(f"MACD error: {e}")

    def _execute_bollinger_strategy(
        self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> FallbackResult:
        """Execute Bollinger Bands strategy."""
        try:
            if "close" in market_data.columns:
                close_prices = market_data["close"]
                upper, middle, lower = self._calculate_bollinger_bands(close_prices)

                if len(upper) > 0 and len(lower) > 0:
                    current_price = close_prices.iloc[-1]
                    current_upper = upper.iloc[-1]
                    current_lower = lower.iloc[-1]

                    if current_price < current_lower:
                        signal = "BUY"
                        confidence = 0.8
                    elif current_price > current_upper:
                        signal = "SELL"
                        confidence = 0.8
                    else:
                        signal = "HOLD"
                        confidence = 0.5
                else:
                    signal = "HOLD"
                    confidence = 0.3
            else:
                signal = "HOLD"
                confidence = 0.3

            return FallbackResult(
                strategy_name="Bollinger",
                signal=signal,
                confidence=confidence,
                performance_metrics=self._get_performance_metrics("Bollinger"),
                execution_time=0.0,
                fallback_rank=0,
            )

        except Exception as e:
            logger.error(f"Error in Bollinger strategy: {e}")
            return self._create_error_result(f"Bollinger error: {e}")

    def _execute_momentum_strategy(
        self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> FallbackResult:
        """Execute Momentum strategy."""
        try:
            if "close" in market_data.columns:
                close_prices = market_data["close"]
                momentum = self._calculate_momentum(close_prices, period=10)

                if len(momentum) > 0:
                    current_momentum = momentum.iloc[-1]

                    if current_momentum > 0:
                        signal = "BUY"
                        confidence = 0.7
                    else:
                        signal = "SELL"
                        confidence = 0.7
                else:
                    signal = "HOLD"
                    confidence = 0.3
            else:
                signal = "HOLD"
                confidence = 0.3

            return FallbackResult(
                strategy_name="Momentum",
                signal=signal,
                confidence=confidence,
                performance_metrics=self._get_performance_metrics("Momentum"),
                execution_time=0.0,
                fallback_rank=0,
            )

        except Exception as e:
            logger.error(f"Error in Momentum strategy: {e}")
            return self._create_error_result(f"Momentum error: {e}")

    def _execute_mean_reversion_strategy(
        self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> FallbackResult:
        """Execute Mean Reversion strategy."""
        try:
            if "close" in market_data.columns:
                close_prices = market_data["close"]
                mean = close_prices.rolling(window=20).mean()
                std = close_prices.rolling(window=20).std()

                if len(mean) > 0 and len(std) > 0:
                    current_price = close_prices.iloc[-1]
                    current_mean = mean.iloc[-1]
                    current_std = std.iloc[-1]

                    if current_std > 0:
                        z_score = (current_price - current_mean) / current_std

                        if z_score > 1.5:
                            signal = "SELL"
                            confidence = 0.8
                        elif z_score < -1.5:
                            signal = "BUY"
                            confidence = 0.8
                        else:
                            signal = "HOLD"
                            confidence = 0.5
                    else:
                        signal = "HOLD"
                        confidence = 0.3
                else:
                    signal = "HOLD"
                    confidence = 0.3
            else:
                signal = "HOLD"
                confidence = 0.3

            return FallbackResult(
                strategy_name="MeanReversion",
                signal=signal,
                confidence=confidence,
                performance_metrics=self._get_performance_metrics("MeanReversion"),
                execution_time=0.0,
                fallback_rank=0,
            )

        except Exception as e:
            logger.error(f"Error in Mean Reversion strategy: {e}")
            return self._create_error_result(f"Mean Reversion error: {e}")

    def _create_error_result(self, error_message: str) -> FallbackResult:
        """Create error result."""
        return FallbackResult(
            strategy_name="ERROR",
            signal="HOLD",
            confidence=0.0,
            performance_metrics={},
            execution_time=0.0,
            fallback_rank=999,
        )

    def update_strategy_performance(
        self, strategy_name: str, trade_result: Dict[str, Any]
    ):
        """
        Update performance metrics for a strategy.

        Args:
            strategy_name: Name of the strategy
            trade_result: Trade result data
        """
        try:
            if strategy_name not in self.strategy_performance:
                return

            # Add trade to history
            trade_result["strategy"] = strategy_name
            trade_result["timestamp"] = datetime.now()
            self.trade_history.append(trade_result)

            # Recalculate performance
            self._recalculate_performance(strategy_name)

        except Exception as e:
            logger.error(f"Error updating performance for {strategy_name}: {e}")

    def _recalculate_performance(self, strategy_name: str):
        """Recalculate performance metrics for a strategy."""
        try:
            # Filter trades for this strategy
            strategy_trades = [
                trade
                for trade in self.trade_history
                if trade.get("strategy") == strategy_name
            ]

            if not strategy_trades:
                return

            # Calculate metrics
            wins = sum(1 for trade in strategy_trades if trade.get("pnl", 0) > 0)
            total_trades = len(strategy_trades)
            win_rate = wins / total_trades if total_trades > 0 else 0.0

            pnls = [trade.get("pnl", 0) for trade in strategy_trades]
            avg_return = sum(pnls) / len(pnls) if pnls else 0.0

            # Update performance
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
                perf.win_rate = win_rate
                perf.total_trades = total_trades
                perf.avg_return = avg_return
                perf.last_updated = datetime.now()

        except Exception as e:
            logger.error(f"Error recalculating performance for {strategy_name}: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all strategies.

        Returns:
            Performance summary dictionary
        """
        summary = {"total_strategies": len(self.strategy_performance), "strategies": {}}

        for name, perf in self.strategy_performance.items():
            summary["strategies"][name] = {
                "win_rate": perf.win_rate,
                "total_trades": perf.total_trades,
                "avg_return": perf.avg_return,
                "sharpe_ratio": perf.sharpe_ratio,
                "confidence_score": perf.confidence_score,
                "last_updated": perf.last_updated.isoformat(),
            }

        return summary

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _calculate_momentum(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate momentum indicator."""
        return prices.diff(period)

    def _get_performance_metrics(self, strategy_name: str) -> Dict[str, float]:
        """Get performance metrics for a strategy."""
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            return {
                "win_rate": perf.win_rate,
                "avg_return": perf.avg_return,
                "sharpe_ratio": perf.sharpe_ratio,
                "confidence_score": perf.confidence_score,
            }
        return {}


def create_strategy_fallback(
    fallback_pool: Optional[List[str]] = None,
) -> StrategyFallback:
    """
    Create a strategy fallback instance.

    Args:
        fallback_pool: List of fallback strategy names

    Returns:
        StrategyFallback instance
    """
    return StrategyFallback(fallback_pool=fallback_pool)
