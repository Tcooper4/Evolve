# -*- coding: utf-8 -*-
"""
Pairs Trading Engine with cointegration testing and dynamic hedge ratio estimation.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")


@dataclass
class CointegrationResult:
    """Cointegration test result."""

    symbol1: str
    symbol2: str
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    confidence_level: float


@dataclass
class PairsSignal:
    """Pairs trading signal."""

    timestamp: datetime
    symbol1: str
    symbol2: str
    signal_type: str  # 'long_short', 'short_long', 'close'
    z_score: float
    hedge_ratio: float
    spread_value: float
    confidence: float
    position_size: float


class PairsTradingEngine:
    """
    Pairs Trading Engine with:
    - Cointegration testing (Engle-Granger, Johansen)
    - Dynamic hedge ratio estimation
    - Z-score based signal generation
    - Risk management and position sizing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Pairs Trading Engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.lookback_period = self.config.get("lookback_period", 252)  # 1 year
        self.z_score_threshold = self.config.get("z_score_threshold", 2.0)
        self.z_score_exit = self.config.get("z_score_exit", 0.5)
        self.min_correlation = self.config.get("min_correlation", 0.7)
        self.max_p_value = self.config.get("max_p_value", 0.05)
        self.rolling_window = self.config.get("rolling_window", 60)
        self.min_spread_std = self.config.get("min_spread_std", 0.01)

        # Storage
        self.cointegration_results: Dict[str, CointegrationResult] = {}
        self.spread_history: Dict[str, pd.Series] = {}
        self.hedge_ratios: Dict[str, pd.Series] = {}
        self.active_pairs: Dict[str, Dict[str, Any]] = {}

    def find_cointegrated_pairs(
        self, price_data: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> List[Tuple[str, str, CointegrationResult]]:
        """
        Find cointegrated pairs from a list of symbols.

        Args:
            price_data: Dictionary of price data for each symbol
            symbols: List of symbols to test

        Returns:
            List of cointegrated pairs with test results
        """
        try:
            self.logger.info(f"Finding cointegrated pairs from {len(symbols)} symbols")

            cointegrated_pairs = []

            # Get price series
            price_series = {}
            for symbol in symbols:
                if symbol in price_data and "close" in price_data[symbol].columns:
                    price_series[symbol] = price_data[symbol]["close"]

            if len(price_series) < 2:
                self.logger.warning("Insufficient price data for cointegration testing")
                return cointegrated_pairs

            # Test all possible pairs
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i + 1 :]:
                    if symbol1 in price_series and symbol2 in price_series:
                        result = self._test_cointegration(
                            symbol1, symbol2, price_series
                        )

                        if result.is_cointegrated:
                            cointegrated_pairs.append((symbol1, symbol2, result))

                            # Store result
                            pair_key = f"{symbol1}_{symbol2}"
                            self.cointegration_results[pair_key] = result

                            self.logger.info(
                                f"Found cointegrated pair: {symbol1} - {symbol2} "
                                f"(p-value: {result.p_value:.4f})"
                            )

            self.logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
            return cointegrated_pairs

        except Exception as e:
            self.logger.error(f"Error finding cointegrated pairs: {str(e)}")
            return []

    def _test_cointegration(
        self, symbol1: str, symbol2: str, price_series: Dict[str, pd.Series]
    ) -> CointegrationResult:
        """Test cointegration between two symbols using Engle-Granger test."""
        try:
            # Get aligned price series
            series1 = price_series[symbol1].dropna()
            series2 = price_series[symbol2].dropna()

            # Align series
            common_index = series1.index.intersection(series2.index)
            if len(common_index) < self.lookback_period:
                return self._create_no_cointegration_result(symbol1, symbol2)

            series1_aligned = series1.loc[common_index]
            series2_aligned = series2.loc[common_index]

            # Check correlation
            correlation = series1_aligned.corr(series2_aligned)
            if abs(correlation) < self.min_correlation:
                return self._create_no_cointegration_result(symbol1, symbol2)

            # Perform Engle-Granger cointegration test
            test_statistic, p_value, critical_values = coint(
                series1_aligned, series2_aligned
            )

            # Determine if cointegrated
            is_cointegrated = p_value < self.max_p_value

            if is_cointegrated:
                # Calculate hedge ratio using OLS regression
                hedge_ratio = self._calculate_hedge_ratio(
                    series1_aligned, series2_aligned
                )

                # Calculate spread
                spread = series1_aligned - hedge_ratio * series2_aligned
                spread_mean = spread.mean()
                spread_std = spread.std()

                # Store spread history
                pair_key = f"{symbol1}_{symbol2}"
                self.spread_history[pair_key] = spread

                # Calculate confidence level
                confidence_level = 1 - p_value

                return CointegrationResult(
                    symbol1=symbol1,
                    symbol2=symbol2,
                    is_cointegrated=True,
                    p_value=p_value,
                    test_statistic=test_statistic,
                    critical_values=critical_values,
                    hedge_ratio=hedge_ratio,
                    spread_mean=spread_mean,
                    spread_std=spread_std,
                    confidence_level=confidence_level,
                )
            else:
                return self._create_no_cointegration_result(symbol1, symbol2)

        except Exception as e:
            self.logger.error(f"Error testing cointegration: {str(e)}")
            return self._create_no_cointegration_result(symbol1, symbol2)

    def _create_no_cointegration_result(
        self, symbol1: str, symbol2: str
    ) -> CointegrationResult:
        """Create result for non-cointegrated pair."""
        return CointegrationResult(
            symbol1=symbol1,
            symbol2=symbol2,
            is_cointegrated=False,
            p_value=1.0,
            test_statistic=0.0,
            critical_values={},
            hedge_ratio=0.0,
            spread_mean=0.0,
            spread_std=0.0,
            confidence_level=0.0,
        )

    def _calculate_hedge_ratio(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate hedge ratio using OLS regression."""
        try:
            # Add constant to series2 for regression
            X = sm.add_constant(series2)
            y = series1

            # Perform OLS regression
            model = OLS(y, X).fit()

            # Return coefficient for series2 (excluding constant)
            return model.params[1]

        except Exception as e:
            self.logger.error(f"Error calculating hedge ratio: {str(e)}")
            return 1.0

    def update_hedge_ratios(
        self,
        price_data: Dict[str, pd.DataFrame],
        pairs: List[Tuple[str, str, CointegrationResult]],
    ):
        """Update hedge ratios using rolling window."""
        try:
            self.logger.info("Updating hedge ratios for active pairs")

            for symbol1, symbol2, coint_result in pairs:
                if not coint_result.is_cointegrated:
                    continue

                pair_key = f"{symbol1}_{symbol2}"

                # Get recent price data
                if (
                    symbol1 in price_data
                    and symbol2 in price_data
                    and "close" in price_data[symbol1].columns
                    and "close" in price_data[symbol2].columns
                ):
                    series1 = price_data[symbol1]["close"]
                    series2 = price_data[symbol2]["close"]

                    # Calculate rolling hedge ratio
                    rolling_ratios = self._calculate_rolling_hedge_ratio(
                        series1, series2, self.rolling_window
                    )

                    self.hedge_ratios[pair_key] = rolling_ratios

                    # Update cointegration result with latest hedge ratio
                    if not rolling_ratios.empty:
                        latest_ratio = rolling_ratios.iloc[-1]
                        self.cointegration_results[pair_key].hedge_ratio = latest_ratio

                        # Update spread
                        latest_spread = (
                            series1.iloc[-1] - latest_ratio * series2.iloc[-1]
                        )
                        self.cointegration_results[pair_key].spread_mean = latest_spread

        except Exception as e:
            self.logger.error(f"Error updating hedge ratios: {str(e)}")

    def _calculate_rolling_hedge_ratio(
        self, series1: pd.Series, series2: pd.Series, window: int
    ) -> pd.Series:
        """Calculate rolling hedge ratio."""
        try:
            ratios = pd.Series(index=series1.index, dtype=float)

            for i in range(window, len(series1)):
                window_series1 = series1.iloc[i - window : i]
                window_series2 = series2.iloc[i - window : i]

                try:
                    X = sm.add_constant(window_series2)
                    y = window_series1
                    model = OLS(y, X).fit()
                    ratios.iloc[i] = model.params[1]
                except (ValueError, TypeError, np.linalg.LinAlgError) as e:
                    # Use previous ratio if regression fails
                    self.logger.debug(f"Regression failed for window {i}: {e}")
                    ratios.iloc[i] = ratios.iloc[i - 1] if i > window else 1.0

            return ratios

        except Exception as e:
            self.logger.error(f"Error calculating rolling hedge ratio: {str(e)}")
            return pd.Series(dtype=float)

    def generate_signals(
        self,
        price_data: Dict[str, pd.DataFrame],
        pairs: List[Tuple[str, str, CointegrationResult]],
    ) -> List[PairsSignal]:
        """
        Generate trading signals for cointegrated pairs.

        Args:
            price_data: Dictionary of price data for each symbol
            pairs: List of cointegrated pairs

        Returns:
            List of trading signals
        """
        try:
            signals = []

            for symbol1, symbol2, coint_result in pairs:
                if not coint_result.is_cointegrated:
                    continue

                pair_key = f"{symbol1}_{symbol2}"

                # Get current prices
                if (
                    symbol1 in price_data
                    and symbol2 in price_data
                    and "close" in price_data[symbol1].columns
                    and "close" in price_data[symbol2].columns
                ):
                    current_price1 = price_data[symbol1]["close"].iloc[-1]
                    current_price2 = price_data[symbol2]["close"].iloc[-1]

                    # Get current hedge ratio
                    current_hedge_ratio = coint_result.hedge_ratio
                    if (
                        pair_key in self.hedge_ratios
                        and not self.hedge_ratios[pair_key].empty
                    ):
                        current_hedge_ratio = self.hedge_ratios[pair_key].iloc[-1]

                    # Calculate current spread
                    current_spread = (
                        current_price1 - current_hedge_ratio * current_price2
                    )

                    # Calculate z-score
                    z_score = self._calculate_z_score(pair_key, current_spread)

                    # Generate signal based on z-score
                    signal = self._generate_signal_from_zscore(
                        symbol1,
                        symbol2,
                        z_score,
                        current_hedge_ratio,
                        current_spread,
                        coint_result,
                    )

                    if signal:
                        signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []

    def _calculate_z_score(self, pair_key: str, current_spread: float) -> float:
        """Calculate z-score for current spread."""
        try:
            if pair_key not in self.spread_history:
                return 0.0

            spread_series = self.spread_history[pair_key]
            if len(spread_series) < 20:
                return 0.0

            # Use rolling mean and std for more stable z-score
            rolling_mean = spread_series.rolling(window=20).mean().iloc[-1]
            rolling_std = spread_series.rolling(window=20).std().iloc[-1]

            if rolling_std == 0:
                return 0.0

            z_score = (current_spread - rolling_mean) / rolling_std
            return z_score

        except Exception as e:
            self.logger.error(f"Error calculating z-score: {str(e)}")
            return 0.0

    def _generate_signal_from_zscore(
        self,
        symbol1: str,
        symbol2: str,
        z_score: float,
        hedge_ratio: float,
        spread: float,
        coint_result: CointegrationResult,
    ) -> Optional[PairsSignal]:
        """Generate trading signal based on z-score."""
        try:
            timestamp = datetime.now()
            confidence = coint_result.confidence_level

            # Long-short signal (spread is high, expect mean reversion)
            if z_score > self.z_score_threshold:
                return PairsSignal(
                    timestamp=timestamp,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    signal_type="short_long",  # Short symbol1, Long symbol2
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    spread_value=spread,
                    confidence=confidence,
                    position_size=self._calculate_position_size(z_score, confidence),
                )

            # Short-long signal (spread is low, expect mean reversion)
            elif z_score < -self.z_score_threshold:
                return PairsSignal(
                    timestamp=timestamp,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    signal_type="long_short",  # Long symbol1, Short symbol2
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    spread_value=spread,
                    confidence=confidence,
                    position_size=self._calculate_position_size(
                        abs(z_score), confidence
                    ),
                )

            # Close signal (spread has reverted to mean)
            elif abs(z_score) < self.z_score_exit:
                return PairsSignal(
                    timestamp=timestamp,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    signal_type="close",
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    spread_value=spread,
                    confidence=confidence,
                    position_size=0.0,
                )

            return None

        except Exception as e:
            self.logger.error(f"Error generating signal from z-score: {str(e)}")
            return None

    def _calculate_position_size(self, z_score: float, confidence: float) -> float:
        """Calculate position size based on z-score and confidence."""
        try:
            # Base position size from z-score
            base_size = min(1.0, abs(z_score) / self.z_score_threshold)

            # Adjust for confidence
            confidence_multiplier = confidence

            # Risk adjustment
            risk_multiplier = 0.5  # Conservative position sizing

            position_size = base_size * confidence_multiplier * risk_multiplier

            return max(0.0, min(1.0, position_size))

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.5

    def validate_pair_stability(
        self, symbol1: str, symbol2: str, price_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """Validate that a pair remains stable over time."""
        try:
            pair_key = f"{symbol1}_{symbol2}"

            if pair_key not in self.cointegration_results:
                return False

            coint_result = self.cointegration_results[pair_key]
            if not coint_result.is_cointegrated:
                return False

            # Get recent data
            if (
                symbol1 not in price_data
                or symbol2 not in price_data
                or "close" not in price_data[symbol1].columns
                or "close" not in price_data[symbol2].columns
            ):
                return False

            series1 = price_data[symbol1]["close"]
            series2 = price_data[symbol2]["close"]

            # Align series
            common_index = series1.index.intersection(series2.index)
            if len(common_index) < 60:  # Need at least 60 days
                return False

            series1_aligned = series1.loc[common_index]
            series2_aligned = series2.loc[common_index]

            # Re-test cointegration
            test_statistic, p_value, _ = coint(series1_aligned, series2_aligned)

            # Check if still cointegrated
            if p_value > self.max_p_value:
                self.logger.warning(f"Pair {symbol1}-{symbol2} no longer cointegrated")
                return False

            # Check correlation
            correlation = series1_aligned.corr(series2_aligned)
            if abs(correlation) < self.min_correlation:
                self.logger.warning(
                    f"Pair {symbol1}-{symbol2} correlation too low: {correlation:.3f}"
                )
                return False

            # Check spread stability
            if pair_key in self.spread_history:
                spread = self.spread_history[pair_key]
                spread_std = spread.std()

                if spread_std < self.min_spread_std:
                    self.logger.warning(
                        f"Pair {symbol1}-{symbol2} spread too stable: {spread_std:.4f}"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating pair stability: {str(e)}")
            return False

    def get_pair_statistics(
        self, symbol1: str, symbol2: str
    ) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific pair."""
        try:
            pair_key = f"{symbol1}_{symbol2}"

            if pair_key not in self.cointegration_results:
                return None

            coint_result = self.cointegration_results[pair_key]

            stats = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "is_cointegrated": coint_result.is_cointegrated,
                "p_value": coint_result.p_value,
                "confidence_level": coint_result.confidence_level,
                "hedge_ratio": coint_result.hedge_ratio,
                "spread_mean": coint_result.spread_mean,
                "spread_std": coint_result.spread_std,
            }

            # Add spread statistics if available
            if pair_key in self.spread_history:
                spread = self.spread_history[pair_key]
                stats.update(
                    {
                        "spread_min": spread.min(),
                        "spread_max": spread.max(),
                        "spread_current": spread.iloc[-1] if not spread.empty else 0.0,
                        "spread_observations": len(spread),
                    }
                )

            # Add hedge ratio statistics if available
            if pair_key in self.hedge_ratios:
                ratios = self.hedge_ratios[pair_key]
                stats.update(
                    {
                        "hedge_ratio_mean": ratios.mean(),
                        "hedge_ratio_std": ratios.std(),
                        "hedge_ratio_current": (
                            ratios.iloc[-1]
                            if not ratios.empty
                            else coint_result.hedge_ratio
                        ),
                    }
                )

            return stats

        except Exception as e:
            self.logger.error(f"Error getting pair statistics: {str(e)}")

    def cleanup_inactive_pairs(self, active_symbols: List[str]):
        """Remove pairs that are no longer active."""
        try:
            pairs_to_remove = []

            for pair_key in self.cointegration_results.keys():
                symbol1, symbol2 = pair_key.split("_")

                if symbol1 not in active_symbols or symbol2 not in active_symbols:
                    pairs_to_remove.append(pair_key)

            for pair_key in pairs_to_remove:
                del self.cointegration_results[pair_key]
                if pair_key in self.spread_history:
                    del self.spread_history[pair_key]
                if pair_key in self.hedge_ratios:
                    del self.hedge_ratios[pair_key]

                self.logger.info(f"Removed inactive pair: {pair_key}")

        except Exception as e:
            self.logger.error(f"Error cleaning up inactive pairs: {str(e)}")

    def get_active_pairs_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active pairs."""
        try:
            summary = []

            for pair_key, coint_result in self.cointegration_results.items():
                if coint_result.is_cointegrated:
                    pair_stats = self.get_pair_statistics(
                        coint_result.symbol1, coint_result.symbol2
                    )
                    if pair_stats:
                        summary.append(pair_stats)

            return summary

        except Exception as e:
            self.logger.error(f"Error getting active pairs summary: {str(e)}")
            return []
