"""
Strategy Comparison and Stacking Module

This module provides comprehensive strategy comparison and stacking capabilities
for the Evolve trading system.
Enhanced with normalized metrics and confidence intervals for fair comparison.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from trading.strategies.registry import (
    BollingerBandsStrategy,
    MACDStrategy,
    RSIStrategy,
)
from trading.utils.performance_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy with confidence intervals."""

    strategy_name: str
    sharpe_ratio: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    win_rate: float
    win_rate_ci_lower: float
    win_rate_ci_upper: float
    max_drawdown: float
    total_return: float
    return_ci_lower: float
    return_ci_upper: float
    volatility: float
    profit_factor: float
    avg_trade: float
    num_trades: int
    avg_holding_period: float
    normalized_score: float
    confidence_level: float
    timestamp: datetime


@dataclass
class StrategyComparison:
    """Comparison results between strategies with statistical significance."""

    strategy_a: str
    strategy_b: str
    sharpe_diff: float
    sharpe_diff_significant: bool
    win_rate_diff: float
    win_rate_diff_significant: bool
    drawdown_diff: float
    return_diff: float
    return_diff_significant: bool
    correlation: float
    correlation_significant: bool
    combined_sharpe: float
    statistical_power: float
    timestamp: datetime


class MetricNormalizer:
    """Normalizes strategy metrics for fair comparison."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize the normalizer.

        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)

    def normalize_metric(
        self, values: List[float], metric_type: str = "ratio"
    ) -> Dict[str, float]:
        """Normalize a metric to a 0-1 scale.

        Args:
            values: List of metric values
            metric_type: Type of metric ('ratio', 'percentage', 'absolute')

        Returns:
            Dictionary with normalized values and confidence intervals
        """
        if not values or len(values) == 0:
            return {"normalized": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        values = np.array(values)

        if metric_type == "ratio":
            # For ratios like Sharpe, normalize using log transformation
            log_values = np.log(np.abs(values) + 1)  # Add 1 to handle zero/negative
            normalized = (log_values - log_values.min()) / (
                log_values.max() - log_values.min()
            )
        elif metric_type == "percentage":
            # For percentages like win rate, use min-max normalization
            normalized = (values - values.min()) / (values.max() - values.min())
        else:  # absolute
            # For absolute values like returns, use robust normalization
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            normalized = (values - median) / (mad + 1e-8)  # Add small epsilon
            normalized = 1 / (1 + np.exp(-normalized))  # Sigmoid to 0-1

        # Calculate confidence intervals
        mean_val = np.mean(normalized)
        std_val = np.std(normalized)
        ci_lower = max(0, mean_val - self.z_score * std_val / np.sqrt(len(values)))
        ci_upper = min(1, mean_val + self.z_score * std_val / np.sqrt(len(values)))

        return {
            "normalized": float(mean_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    def calculate_confidence_intervals(
        self, returns: pd.Series, metric: str
    ) -> Tuple[float, float, float]:
        """Calculate confidence intervals for a performance metric.

        Args:
            returns: Strategy returns
            metric: Metric name ('sharpe', 'return', 'volatility')

        Returns:
            Tuple of (metric_value, ci_lower, ci_upper)
        """
        if len(returns) < 30:  # Need sufficient data for reliable intervals
            logger.warning(
                f"Insufficient data for confidence intervals: {len(returns)} observations"
            )
            return 0.0, 0.0, 0.0

        try:
            if metric == "sharpe":
                # Bootstrap Sharpe ratio confidence interval
                sharpe_ratios = []
                n_bootstrap = 1000

                for _ in range(n_bootstrap):
                    sample = returns.sample(n=len(returns), replace=True)
                    if sample.std() > 0:
                        sharpe = sample.mean() / sample.std() * np.sqrt(252)
                        sharpe_ratios.append(sharpe)

                sharpe_ratios = np.array(sharpe_ratios)
                ci_lower = np.percentile(
                    sharpe_ratios, (1 - self.confidence_level) * 50
                )
                ci_upper = np.percentile(
                    sharpe_ratios, (1 + self.confidence_level) * 50
                )
                return float(np.mean(sharpe_ratios)), float(ci_lower), float(ci_upper)

            elif metric == "return":
                # Bootstrap return confidence interval
                total_returns = []
                n_bootstrap = 1000

                for _ in range(n_bootstrap):
                    sample = returns.sample(n=len(returns), replace=True)
                    total_return = (1 + sample).prod() - 1
                    total_returns.append(total_return)

                total_returns = np.array(total_returns)
                ci_lower = np.percentile(
                    total_returns, (1 - self.confidence_level) * 50
                )
                ci_upper = np.percentile(
                    total_returns, (1 + self.confidence_level) * 50
                )
                return float(np.mean(total_returns)), float(ci_lower), float(ci_upper)

            elif metric == "win_rate":
                # Binomial confidence interval for win rate
                wins = (returns > 0).sum()
                total_trades = len(returns)

                if total_trades > 0:
                    win_rate = wins / total_trades
                    # Wilson confidence interval
                    z = self.z_score
                    denominator = 1 + z**2 / total_trades
                    centre_adjusted = win_rate + z * z / (2 * total_trades)
                    adjusted_standard_error = z * np.sqrt(
                        (win_rate * (1 - win_rate) + z * z / (4 * total_trades))
                        / total_trades
                    )
                    lower_bound = (
                        centre_adjusted - adjusted_standard_error
                    ) / denominator
                    upper_bound = (
                        centre_adjusted + adjusted_standard_error
                    ) / denominator

                    return win_rate, max(0, lower_bound), min(1, upper_bound)

                return 0.0, 0.0, 0.0

            else:
                # Default: use standard error
                mean_val = returns.mean()
                std_error = returns.std() / np.sqrt(len(returns))
                ci_lower = mean_val - self.z_score * std_error
                ci_upper = mean_val + self.z_score * std_error
                return float(mean_val), float(ci_lower), float(ci_upper)

        except Exception as e:
            logger.error(f"Error calculating confidence intervals for {metric}: {e}")
            return 0.0, 0.0, 0.0


class StrategyComparisonMatrix:
    """Multi-strategy comparison matrix with normalized metrics."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize the strategy comparison matrix.

        Args:
            confidence_level: Confidence level for intervals
        """
        self.strategies = {
            "rsi": RSIStrategy(),
            "macd": MACDStrategy(),
            "bollinger": BollingerBandsStrategy(),
        }
        self.comparison_history = []
        self.performance_cache = {}
        self.normalizer = MetricNormalizer(confidence_level)
        self.confidence_level = confidence_level

    def generate_comparison_matrix(
        self, data: pd.DataFrame, strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate a comprehensive strategy comparison matrix with normalized metrics.

        Args:
            data: Market data
            strategies: List of strategy names to compare (default: all)

        Returns:
            DataFrame with comparison matrix and confidence intervals
        """
        try:
            if strategies is None:
                strategies = list(self.strategies.keys())

            # Calculate performance for each strategy
            performances = {}
            for strategy_name in strategies:
                if strategy_name in self.strategies:
                    performance = self._calculate_strategy_performance(
                        strategy_name, data
                    )
                    performances[strategy_name] = performance

            # Create comparison matrix with normalized metrics
            matrix_data = []
            metrics = [
                "sharpe_ratio",
                "win_rate",
                "max_drawdown",
                "total_return",
                "volatility",
                "profit_factor",
                "normalized_score",
            ]

            for strategy_name, performance in performances.items():
                row = {"Strategy": strategy_name}
                for metric in metrics:
                    row[metric] = getattr(performance, metric, 0.0)
                    # Add confidence intervals for key metrics
                    if metric in ["sharpe_ratio", "win_rate", "total_return"]:
                        ci_lower = getattr(performance, f"{metric}_ci_lower", 0.0)
                        ci_upper = getattr(performance, f"{metric}_ci_upper", 0.0)
                        row[f"{metric}_ci"] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                matrix_data.append(row)

            matrix_df = pd.DataFrame(matrix_data)

            # Add ranking columns with confidence intervals
            for metric in [
                "sharpe_ratio",
                "win_rate",
                "total_return",
                "normalized_score",
            ]:
                matrix_df[f"{metric}_rank"] = matrix_df[metric].rank(ascending=False)

                # Add significance indicators
                if metric in ["sharpe_ratio", "win_rate", "total_return"]:
                    matrix_df[f"{metric}_significant"] = self._check_significance(
                        matrix_df, metric
                    )

            # Cache results
            self.performance_cache = performances

            return matrix_df

        except Exception as e:
            logger.error(f"Error generating comparison matrix: {e}")
            return pd.DataFrame()

    def _check_significance(self, matrix_df: pd.DataFrame, metric: str) -> pd.Series:
        """Check if differences between strategies are statistically significant.

        Args:
            matrix_df: Comparison matrix
            metric: Metric to check

        Returns:
            Series indicating significance
        """
        significance = pd.Series(False, index=matrix_df.index)

        if len(matrix_df) < 2:
            return significance

        # Get confidence intervals
        ci_lower_col = f"{metric}_ci_lower"
        ci_upper_col = f"{metric}_ci_upper"

        if ci_lower_col in matrix_df.columns and ci_upper_col in matrix_df.columns:
            for i, row in matrix_df.iterrows():
                # Check if confidence intervals overlap with other strategies
                overlaps = 0
                for j, other_row in matrix_df.iterrows():
                    if i != j:
                        # Check for overlap
                        if not (
                            row[ci_upper_col] < other_row[ci_lower_col]
                            or row[ci_lower_col] > other_row[ci_upper_col]
                        ):
                            overlaps += 1

                # Mark as significant if no overlaps (or minimal overlaps)
                significance.iloc[i] = overlaps <= len(matrix_df) * 0.3

        return significance

    def _calculate_strategy_performance(
        self, strategy_name: str, data: pd.DataFrame
    ) -> StrategyPerformance:
        """Calculate performance metrics for a strategy with confidence intervals.

        Args:
            strategy_name: Name of the strategy
            data: Market data

        Returns:
            StrategyPerformance object with confidence intervals
        """
        try:
            strategy = self.strategies[strategy_name]
            signals = strategy.generate_signals(data)

            # Calculate returns
            returns = self._calculate_strategy_returns(data, signals)

            if len(returns) == 0:
                return self._create_empty_performance(strategy_name)

            # Calculate metrics with confidence intervals
            (
                sharpe,
                sharpe_ci_lower,
                sharpe_ci_upper,
            ) = self.normalizer.calculate_confidence_intervals(returns, "sharpe")

            (
                win_rate,
                win_rate_ci_lower,
                win_rate_ci_upper,
            ) = self.normalizer.calculate_confidence_intervals(returns, "win_rate")

            (
                total_return,
                return_ci_lower,
                return_ci_upper,
            ) = self.normalizer.calculate_confidence_intervals(returns, "return")

            # Calculate other metrics
            max_dd = calculate_max_drawdown(returns)
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Calculate trade-based metrics
            trade_metrics = self._calculate_trade_metrics(returns, signals)

            # Calculate normalized score
            normalized_score = self._calculate_normalized_score(
                sharpe, win_rate, max_dd, total_return, volatility
            )

            return StrategyPerformance(
                strategy_name=strategy_name,
                sharpe_ratio=sharpe,
                sharpe_ci_lower=sharpe_ci_lower,
                sharpe_ci_upper=sharpe_ci_upper,
                win_rate=win_rate,
                win_rate_ci_lower=win_rate_ci_lower,
                win_rate_ci_upper=win_rate_ci_upper,
                max_drawdown=max_dd,
                total_return=total_return,
                return_ci_lower=return_ci_lower,
                return_ci_upper=return_ci_upper,
                volatility=volatility,
                profit_factor=trade_metrics["profit_factor"],
                avg_trade=trade_metrics["avg_trade"],
                num_trades=trade_metrics["num_trades"],
                avg_holding_period=trade_metrics["avg_holding_period"],
                normalized_score=normalized_score,
                confidence_level=self.confidence_level,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating performance for {strategy_name}: {e}")
            return self._create_empty_performance(strategy_name)

    def _create_empty_performance(self, strategy_name: str) -> StrategyPerformance:
        """Create empty performance object for failed calculations."""
        return StrategyPerformance(
            strategy_name=strategy_name,
            sharpe_ratio=0.0,
            sharpe_ci_lower=0.0,
            sharpe_ci_upper=0.0,
            win_rate=0.0,
            win_rate_ci_lower=0.0,
            win_rate_ci_upper=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            return_ci_lower=0.0,
            return_ci_upper=0.0,
            volatility=0.0,
            profit_factor=0.0,
            avg_trade=0.0,
            num_trades=0,
            avg_holding_period=0.0,
            normalized_score=0.0,
            confidence_level=self.confidence_level,
            timestamp=datetime.now(),
        )

    def _calculate_normalized_score(
        self,
        sharpe: float,
        win_rate: float,
        max_dd: float,
        total_return: float,
        volatility: float,
    ) -> float:
        """Calculate a normalized composite score for strategy ranking.

        Args:
            sharpe: Sharpe ratio
            win_rate: Win rate
            max_dd: Maximum drawdown
            total_return: Total return
            volatility: Volatility

        Returns:
            Normalized score between 0 and 1
        """
        try:
            # Normalize each metric to 0-1 scale
            sharpe_norm = max(0, min(1, (sharpe + 2) / 4))  # Assume range -2 to 2
            win_rate_norm = win_rate  # Already 0-1
            drawdown_norm = max(0, min(1, 1 - abs(max_dd)))  # Invert drawdown
            return_norm = max(
                0, min(1, (total_return + 0.5) / 1.0)
            )  # Assume range -50% to 50%
            vol_norm = max(
                0, min(1, 1 - volatility / 0.5)
            )  # Invert volatility, assume max 50%

            # Weighted average (can be adjusted based on preferences)
            weights = {
                "sharpe": 0.3,
                "win_rate": 0.25,
                "drawdown": 0.2,
                "return": 0.15,
                "volatility": 0.1,
            }

            normalized_score = (
                weights["sharpe"] * sharpe_norm
                + weights["win_rate"] * win_rate_norm
                + weights["drawdown"] * drawdown_norm
                + weights["return"] * return_norm
                + weights["volatility"] * vol_norm
            )

            return float(normalized_score)

        except Exception as e:
            logger.error(f"Error calculating normalized score: {e}")
            return 0.0

    def _calculate_strategy_returns(
        self, data: pd.DataFrame, signals: pd.DataFrame
    ) -> pd.Series:
        """Calculate strategy returns from signals.

        Args:
            data: Market data
            signals: Strategy signals

        Returns:
            Series of returns
        """
        try:
            # Calculate price returns
            price_returns = data["Close"].pct_change()

            # Apply signals (1 = buy, -1 = sell, 0 = hold)
            strategy_returns = signals["signal"].shift(1) * price_returns

            # Remove NaN values
            strategy_returns = strategy_returns.dropna()

            return strategy_returns

        except Exception as e:
            logger.error(f"Error calculating strategy returns: {e}")
            return pd.Series([0.0] * len(data))

    def _calculate_trade_metrics(
        self, returns: pd.Series, signals: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate trade-based performance metrics.

        Args:
            returns: Strategy returns
            signals: Strategy signals

        Returns:
            Dictionary of trade metrics
        """
        try:
            # Identify trades
            signal_changes = signals["signal"].diff().fillna(0)
            trade_starts = signal_changes != 0

            if not trade_starts.any():
                return {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "avg_trade": 0.0,
                    "num_trades": 0,
                    "avg_holding_period": 0.0,
                }

            # Calculate trade returns
            trade_returns = []
            holding_periods = []

            start_idx = None
            for i, is_start in enumerate(trade_starts):
                if is_start:
                    if start_idx is not None:
                        # End previous trade
                        trade_return = (1 + returns.iloc[start_idx:i]).prod() - 1
                        trade_returns.append(trade_return)
                        holding_periods.append(i - start_idx)
                    start_idx = i

            # End last trade if still open
            if start_idx is not None and start_idx < len(returns):
                trade_return = (1 + returns.iloc[start_idx:]).prod() - 1
                trade_returns.append(trade_return)
                holding_periods.append(len(returns) - start_idx)

            if not trade_returns:
                return {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "avg_trade": 0.0,
                    "num_trades": 0,
                    "avg_holding_period": 0.0,
                }

            # Calculate metrics
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            win_rate = (
                len(winning_trades) / len(trade_returns) if trade_returns else 0.0
            )

            if losing_trades:
                profit_factor = abs(sum(winning_trades)) / abs(sum(losing_trades))
            else:
                profit_factor = float("inf") if winning_trades else 0.0

            avg_trade = np.mean(trade_returns)
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0

            return {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_trade": avg_trade,
                "num_trades": len(trade_returns),
                "avg_holding_period": avg_holding_period,
            }

        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade": 0.0,
                "num_trades": 0,
                "avg_holding_period": 0.0,
            }

    def get_best_strategy(
        self, data: pd.DataFrame, metric: str = "sharpe_ratio"
    ) -> Tuple[str, float]:
        """Get the best performing strategy based on a metric.

        Args:
            data: Market data
            metric: Performance metric to optimize

        Returns:
            Tuple of (strategy_name, metric_value)
        """
        try:
            matrix = self.generate_comparison_matrix(data)

            if matrix.empty:
                return None, 0.0

            best_idx = matrix[metric].idxmax()
            best_strategy = matrix.loc[best_idx, "Strategy"]
            best_value = matrix.loc[best_idx, metric]

            return best_strategy, best_value

        except Exception as e:
            logger.error(f"Error getting best strategy: {e}")
            return None, 0.0

    def compare_strategies(
        self, strategy_a: str, strategy_b: str, data: pd.DataFrame
    ) -> StrategyComparison:
        """Compare two specific strategies.

        Args:
            strategy_a: First strategy name
            strategy_b: Second strategy name
            data: Market data

        Returns:
            StrategyComparison object
        """
        try:
            # Calculate performance for both strategies
            perf_a = self._calculate_strategy_performance(strategy_a, data)
            perf_b = self._calculate_strategy_performance(strategy_b, data)

            # Calculate returns for correlation
            returns_a = self._calculate_strategy_returns(
                data, self.strategies[strategy_a].generate_signals(data)
            )
            returns_b = self._calculate_strategy_returns(
                data, self.strategies[strategy_b].generate_signals(data)
            )

            # Align returns
            aligned_returns = pd.concat([returns_a, returns_b], axis=1).dropna()
            correlation = (
                aligned_returns.corr().iloc[0, 1] if len(aligned_returns) > 1 else 0.0
            )

            # Calculate combined performance (equal weight)
            combined_returns = (
                aligned_returns.iloc[:, 0] + aligned_returns.iloc[:, 1]
            ) / 2
            combined_sharpe = calculate_sharpe_ratio(combined_returns)

            # Calculate differences with confidence intervals
            (
                sharpe_diff,
                sharpe_diff_ci_lower,
                sharpe_diff_ci_upper,
            ) = self.normalizer.calculate_confidence_intervals(
                combined_returns, "sharpe"
            )

            (
                win_rate_diff,
                win_rate_diff_ci_lower,
                win_rate_diff_ci_upper,
            ) = self.normalizer.calculate_confidence_intervals(
                combined_returns, "win_rate"
            )

            (
                return_diff,
                return_diff_ci_lower,
                return_diff_ci_upper,
            ) = self.normalizer.calculate_confidence_intervals(
                combined_returns, "return"
            )

            # Check for statistical significance
            sharpe_diff_significant = not (
                sharpe_diff_ci_lower > 0 or sharpe_diff_ci_upper < 0
            )
            win_rate_diff_significant = not (
                win_rate_diff_ci_lower > 0 or win_rate_diff_ci_upper < 0
            )
            return_diff_significant = not (
                return_diff_ci_lower > 0 or return_diff_ci_upper < 0
            )

            # Calculate statistical power (simplified)
            # This is a placeholder and would require a more sophisticated statistical test
            # For now, we'll just indicate if the difference is positive/negative
            statistical_power = 0.0  # Placeholder

            comparison = StrategyComparison(
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                sharpe_diff=sharpe_diff,
                sharpe_diff_significant=sharpe_diff_significant,
                win_rate_diff=win_rate_diff,
                win_rate_diff_significant=win_rate_diff_significant,
                drawdown_diff=perf_a.max_drawdown
                - perf_b.max_drawdown,  # Original drawdown diff
                return_diff=return_diff,
                return_diff_significant=return_diff_significant,
                correlation=correlation,
                correlation_significant=False,  # Placeholder
                combined_sharpe=combined_sharpe,
                statistical_power=statistical_power,
                timestamp=datetime.now(),
            )

            # Store in history
            self.comparison_history.append(comparison)

            return comparison

        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return None


class StrategyStacker:
    """Strategy stacking implementation."""

    def __init__(self, max_strategies: int = 3):
        """Initialize strategy stacker.

        Args:
            max_strategies: Maximum number of strategies to stack
        """
        self.max_strategies = max_strategies
        self.comparison_matrix = StrategyComparisonMatrix()
        self.stacking_history = []

    def create_strategy_stack(
        self,
        data: pd.DataFrame,
        strategy_names: Optional[List[str]] = None,
        method: str = "performance_weighted",
    ) -> Dict[str, Any]:
        """Create a stacked strategy combining multiple strategies.

        Args:
            data: Market data
            strategy_names: List of strategies to stack (default: best performers)
            method: Stacking method ('equal', 'performance_weighted', 'correlation_weighted')

        Returns:
            Dictionary with stacked strategy results
        """
        try:
            if strategy_names is None:
                # Select best strategies based on Sharpe ratio
                matrix = self.comparison_matrix.generate_comparison_matrix(data)
                if matrix.empty:
                    return self._create_fallback_stack(data)

                # Get top strategies
                top_strategies = matrix.nlargest(self.max_strategies, "sharpe_ratio")[
                    "Strategy"
                ].tolist()
                strategy_names = top_strategies

            # Limit to max strategies
            strategy_names = strategy_names[: self.max_strategies]

            # Generate signals for each strategy
            strategy_signals = {}
            strategy_returns = {}

            for strategy_name in strategy_names:
                if strategy_name in self.comparison_matrix.strategies:
                    strategy = self.comparison_matrix.strategies[strategy_name]
                    signals = strategy.generate_signals(data)
                    returns = self.comparison_matrix._calculate_strategy_returns(
                        data, signals
                    )

                    strategy_signals[strategy_name] = signals
                    strategy_returns[strategy_name] = returns

            # Calculate weights based on method
            weights = self._calculate_stack_weights(strategy_returns, method)

            # Create stacked signals
            stacked_signals = self._combine_signals(strategy_signals, weights)

            # Calculate stacked performance
            stacked_returns = self.comparison_matrix._calculate_strategy_returns(
                data, stacked_signals
            )
            stacked_performance = (
                self.comparison_matrix._calculate_strategy_performance("stacked", data)
            )

            # Store stacking result
            stacking_result = {
                "strategies": strategy_names,
                "weights": weights,
                "method": method,
                "performance": stacked_performance,
                "signals": stacked_signals,
                "returns": stacked_returns,
                "timestamp": datetime.now(),
            }

            self.stacking_history.append(stacking_result)

            return stacking_result

        except Exception as e:
            logger.error(f"Error creating strategy stack: {e}")
            return self._create_fallback_stack(data)

    def _calculate_stack_weights(
        self, strategy_returns: Dict[str, pd.Series], method: str
    ) -> Dict[str, float]:
        """Calculate weights for strategy stacking.

        Args:
            strategy_returns: Dictionary of strategy returns
            method: Weighting method

        Returns:
            Dictionary of strategy weights
        """
        try:
            if method == "equal":
                # Equal weights
                n_strategies = len(strategy_returns)
                return {name: 1.0 / n_strategies for name in strategy_returns.keys()}

            elif method == "performance_weighted":
                # Weight by Sharpe ratio
                sharpe_ratios = {}
                for name, returns in strategy_returns.items():
                    sharpe_ratios[name] = calculate_sharpe_ratio(returns)

                total_sharpe = sum(max(0, sr) for sr in sharpe_ratios.values())
                if total_sharpe > 0:
                    return {
                        name: max(0, sr) / total_sharpe
                        for name, sr in sharpe_ratios.items()
                    }
                else:
                    return {
                        name: 1.0 / len(strategy_returns)
                        for name in strategy_returns.keys()
                    }

            elif method == "correlation_weighted":
                # Weight by inverse correlation
                returns_df = pd.DataFrame(strategy_returns)
                corr_matrix = returns_df.corr()

                # Calculate average correlation for each strategy
                avg_correlations = {}
                for strategy in strategy_returns.keys():
                    correlations = corr_matrix[strategy].drop(strategy)
                    avg_correlations[strategy] = correlations.mean()

                # Weight inversely to correlation
                inverse_correlations = {
                    name: 1 - corr for name, corr in avg_correlations.items()
                }
                total_inverse = sum(inverse_correlations.values())

                if total_inverse > 0:
                    return {
                        name: inv_corr / total_inverse
                        for name, inv_corr in inverse_correlations.items()
                    }
                else:
                    return {
                        name: 1.0 / len(strategy_returns)
                        for name in strategy_returns.keys()
                    }

            else:
                # Default to equal weights
                n_strategies = len(strategy_returns)
                return {name: 1.0 / n_strategies for name in strategy_returns.keys()}

        except Exception as e:
            logger.error(f"Error calculating stack weights: {e}")
            n_strategies = len(strategy_returns)
            return {name: 1.0 / n_strategies for name in strategy_returns.keys()}

    def _combine_signals(
        self, strategy_signals: Dict[str, pd.DataFrame], weights: Dict[str, float]
    ) -> pd.DataFrame:
        """Combine signals from multiple strategies.

        Args:
            strategy_signals: Dictionary of strategy signals
            weights: Strategy weights

        Returns:
            Combined signals DataFrame
        """
        try:
            # Initialize combined signals
            combined_signals = pd.DataFrame(
                index=next(iter(strategy_signals.values())).index
            )
            combined_signals["signal"] = 0.0

            # Weight and combine signals
            for strategy_name, signals in strategy_signals.items():
                weight = weights.get(strategy_name, 0.0)
                combined_signals["signal"] += weight * signals["signal"]

            # Convert to discrete signals
            combined_signals["signal"] = np.where(
                combined_signals["signal"] > 0.5,
                1,
                np.where(combined_signals["signal"] < -0.5, -1, 0),
            )

            return combined_signals

        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            # Return neutral signals
            return pd.DataFrame(
                {"signal": 0}, index=next(iter(strategy_signals.values())).index
            )

    def _create_fallback_stack(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create a fallback stacked strategy when stacking fails.

        Args:
            data: Market data

        Returns:
            Fallback stacking result
        """
        try:
            # Use simple RSI strategy as fallback
            rsi_strategy = RSIStrategy()
            signals = rsi_strategy.generate_signals(data)
            returns = self.comparison_matrix._calculate_strategy_returns(data, signals)

            return {
                "strategies": ["rsi_fallback"],
                "weights": {"rsi_fallback": 1.0},
                "method": "fallback",
                "performance": self.comparison_matrix._calculate_strategy_performance(
                    "rsi_fallback", data
                ),
                "signals": signals,
                "returns": returns,
                "timestamp": datetime.now(),
                "warning": "Fallback strategy used due to stacking errors",
            }

        except Exception as e:
            logger.error(f"Error creating fallback stack: {e}")
            return {
                "strategies": [],
                "weights": {},
                "method": "error",
                "performance": None,
                "signals": pd.DataFrame(),
                "returns": pd.Series(),
                "timestamp": datetime.now(),
                "error": str(e),
            }

    def get_stacking_summary(self) -> Dict[str, Any]:
        """Get summary of stacking history.

        Returns:
            Dictionary with stacking summary
        """
        try:
            if not self.stacking_history:
                return {"message": "No stacking history available"}

            recent_stacks = self.stacking_history[-10:]  # Last 10 stacks

            summary = {
                "total_stacks": len(self.stacking_history),
                "recent_stacks": len(recent_stacks),
                "methods_used": list(set(stack["method"] for stack in recent_stacks)),
                "avg_strategies_per_stack": np.mean(
                    [len(stack["strategies"]) for stack in recent_stacks]
                ),
                "best_performing_method": self._get_best_method(recent_stacks),
                "last_stack_timestamp": recent_stacks[-1]["timestamp"]
                if recent_stacks
                else None,
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting stacking summary: {e}")
            return {"error": str(e)}

    def _get_best_method(self, stacks: List[Dict[str, Any]]) -> str:
        """Get the best performing stacking method.

        Args:
            stacks: List of stacking results

        Returns:
            Best method name
        """
        try:
            method_performance = {}

            for stack in stacks:
                method = stack["method"]
                if stack["performance"] and hasattr(
                    stack["performance"], "sharpe_ratio"
                ):
                    if method not in method_performance:
                        method_performance[method] = []
                    method_performance[method].append(stack["performance"].sharpe_ratio)

            if not method_performance:
                return "unknown"

            # Calculate average Sharpe ratio for each method
            avg_performance = {
                method: np.mean(perfs) for method, perfs in method_performance.items()
            }

            return max(avg_performance, key=avg_performance.get)

        except Exception as e:
            logger.error(f"Error getting best method: {e}")
            return "unknown"


# Global instances
strategy_comparison = StrategyComparisonMatrix()
strategy_stacker = StrategyStacker()


def get_strategy_comparison() -> StrategyComparisonMatrix:
    """Get the global strategy comparison instance."""
    return strategy_comparison


def get_strategy_stacker() -> StrategyStacker:
    """Get the global strategy stacker instance."""
    return strategy_stacker
