"""
Backtest Optimizer with Walk-Forward Analysis and Regime Detection

Advanced backtesting framework with walk-forward optimization, regime detection,
and robust performance evaluation.
Enhanced with strategy return consistency validation and top N configuration logging.
"""

import json
import logging
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""

    period_start: str
    period_end: str
    training_period: str
    validation_period: str
    best_params: Dict[str, Any]
    validation_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    regime: str
    regime_confidence: float
    timestamp: str
    return_consistency_score: float = 0.0
    validation_passed: bool = True


@dataclass
class RegimeInfo:
    """Information about detected market regime."""

    regime_id: int
    regime_name: str
    start_date: str
    end_date: str
    characteristics: Dict[str, float]
    volatility: float
    trend_strength: float
    correlation_structure: Dict[str, float]
    duration_days: int
    confidence: float


@dataclass
class OptimizationResult:
    """Result from strategy optimization."""

    params: Dict[str, Any]
    performance: Dict[str, float]
    sharpe_ratio: float
    return_consistency: float
    validation_passed: bool
    rank: int
    timestamp: str


class StrategyValidator:
    """Validates strategy return consistency and performance metrics."""

    def __init__(self, min_sharpe: float = 0.5, max_drawdown: float = 0.3, min_consistency: float = 0.6):
        """Initialize strategy validator.

        Args:
            min_sharpe: Minimum acceptable Sharpe ratio
            max_drawdown: Maximum acceptable drawdown
            min_consistency: Minimum return consistency score
        """
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_consistency = min_consistency
        self.logger = logging.getLogger(__name__)

    def validate_returns(self, returns: pd.Series) -> Dict[str, Any]:
        """Validate strategy returns for consistency and quality.

        Args:
            returns: Series of strategy returns

        Returns:
            Validation results
        """
        if len(returns) < 30:
            return {
                "valid": False,
                "reason": "Insufficient data (need at least 30 observations)",
                "consistency_score": 0.0,
            }

        # Calculate basic metrics
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0

        # Calculate drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Calculate return consistency
        consistency_score = self._calculate_return_consistency(returns)

        # Check for excessive volatility
        volatility_score = self._check_volatility_stability(returns)

        # Check for return clustering
        clustering_score = self._check_return_clustering(returns)

        # Overall validation
        valid = (
            sharpe_ratio >= self.min_sharpe
            and max_drawdown <= self.max_drawdown
            and consistency_score >= self.min_consistency
            and volatility_score >= 0.5
            and clustering_score >= 0.5
        )

        return {
            "valid": valid,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "consistency_score": consistency_score,
            "volatility_score": volatility_score,
            "clustering_score": clustering_score,
            "mean_return": mean_return,
            "std_return": std_return,
            "total_return": cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0,
        }

    def _calculate_return_consistency(self, returns: pd.Series) -> float:
        """Calculate return consistency score."""
        try:
            # Calculate rolling Sharpe ratios
            rolling_sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std()
            rolling_sharpe = rolling_sharpe.dropna()

            if len(rolling_sharpe) < 10:
                return 0.0

            # Calculate consistency as stability of rolling Sharpe
            sharpe_std = rolling_sharpe.std()
            sharpe_mean = abs(rolling_sharpe.mean())

            if sharpe_mean == 0:
                return 0.0

            # Consistency is inverse of coefficient of variation
            consistency = sharpe_mean / (sharpe_mean + sharpe_std)

            return min(consistency, 1.0)

        except Exception as e:
            self.logger.warning(f"Error calculating return consistency: {e}")
            return 0.0

    def _check_volatility_stability(self, returns: pd.Series) -> float:
        """Check if volatility is stable over time."""
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20).std()
            rolling_vol = rolling_vol.dropna()

            if len(rolling_vol) < 10:
                return 0.0

            # Calculate volatility of volatility
            vol_of_vol = rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() > 0 else 1.0

            # Stability score (lower vol of vol = more stable)
            stability_score = max(0, 1 - vol_of_vol)

            return stability_score

        except Exception as e:
            self.logger.warning(f"Error checking volatility stability: {e}")
            return 0.0

    def _check_return_clustering(self, returns: pd.Series) -> float:
        """Check for excessive return clustering (indicates potential overfitting)."""
        try:
            # Calculate autocorrelation
            autocorr = returns.autocorr()

            # High positive autocorrelation might indicate overfitting
            # We want some autocorrelation but not too much
            if autocorr > 0.3:
                clustering_score = max(0, 1 - (autocorr - 0.3) / 0.7)
            else:
                clustering_score = 1.0

            return clustering_score

        except Exception as e:
            self.logger.warning(f"Error checking return clustering: {e}")
            return 0.5


class RegimeDetector:
    """Market regime detection using clustering and statistical methods."""

    def __init__(self, n_regimes: int = 3, lookback_window: int = 252):
        """Initialize regime detector.

        Args:
            n_regimes: Number of regimes to detect
            lookback_window: Rolling window for feature calculation
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.regime_history = []
        self.logger = logging.getLogger(__name__)

    def calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection.

        Args:
            data: Market data with OHLCV columns

        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=data.index)

        # Price-based features
        if "close" in data.columns:
            returns = data["close"].pct_change()
            features["volatility"] = returns.rolling(self.lookback_window).std() * np.sqrt(252)
            features["returns"] = returns.rolling(self.lookback_window).mean() * 252
            features["skewness"] = returns.rolling(self.lookback_window).skew()
            features["kurtosis"] = returns.rolling(self.lookback_window).kurt()

            # Trend features
            sma_20 = data["close"].rolling(20).mean()
            sma_50 = data["close"].rolling(50).mean()
            features["trend_strength"] = (sma_20 - sma_50) / sma_50

            # Momentum features
            features["momentum"] = data["close"].pct_change(20)
            features["rsi"] = self._calculate_rsi(data["close"])

        # Volume-based features
        if "volume" in data.columns:
            features["volume_ratio"] = (
                data["volume"].rolling(self.lookback_window).mean()
                / data["volume"].rolling(self.lookback_window * 2).mean()
            )
            features["volume_volatility"] = data["volume"].pct_change().rolling(self.lookback_window).std()

        # Volatility regime features
        if "high" in data.columns and "low" in data.columns:
            features["volatility_regime"] = (data["high"] - data["low"]) / data["close"]
            features["volatility_regime"] = features["volatility_regime"].rolling(self.lookback_window).mean()

        # Remove NaN values
        features = features.dropna()

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def detect_regimes(self, data: pd.DataFrame) -> List[RegimeInfo]:
        """Detect market regimes in the data.

        Args:
            data: Market data

        Returns:
            List of detected regimes
        """
        # Calculate features
        features = self.calculate_regime_features(data)

        if len(features) < self.lookback_window:
            self.logger.warning("Insufficient data for regime detection")
            return []

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Cluster to detect regimes
        cluster_labels = self.kmeans.fit_predict(features_scaled)

        # Create regime information
        regimes = []
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            regime_mask = cluster_labels == label
            regime_dates = features.index[regime_mask]

            if len(regime_dates) == 0:
                continue

            # Calculate regime characteristics
            regime_features = features.loc[regime_dates]
            regime_characteristics = {
                "mean_volatility": regime_features["volatility"].mean(),
                "mean_returns": regime_features["returns"].mean(),
                "mean_trend_strength": regime_features["trend_strength"].mean(),
                "mean_momentum": regime_features["momentum"].mean(),
                "mean_rsi": regime_features["rsi"].mean(),
            }

            # Determine regime name based on characteristics
            regime_name = self._classify_regime(regime_characteristics)

            # Calculate confidence based on cluster stability
            confidence = self._calculate_regime_confidence(features_scaled, cluster_labels, label)

            regime = RegimeInfo(
                regime_id=int(label),
                regime_name=regime_name,
                start_date=regime_dates[0].isoformat(),
                end_date=regime_dates[-1].isoformat(),
                characteristics=regime_characteristics,
                volatility=regime_characteristics["mean_volatility"],
                trend_strength=regime_characteristics["mean_trend_strength"],
                correlation_structure={},  # Could be enhanced
                duration_days=len(regime_dates),
                confidence=confidence,
            )
            regimes.append(regime)

        self.regime_history = regimes
        return regimes

    def _classify_regime(self, characteristics: Dict[str, float]) -> str:
        """Classify regime based on characteristics."""
        volatility = characteristics["mean_volatility"]
        returns = characteristics["mean_returns"]
        trend = characteristics["mean_trend_strength"]

        if volatility > 0.25:
            if returns < -0.05:
                return "High Volatility Bear Market"
            elif returns > 0.05:
                return "High Volatility Bull Market"
            else:
                return "High Volatility Sideways"
        elif volatility < 0.15:
            if trend > 0.02:
                return "Low Volatility Bull Market"
            elif trend < -0.02:
                return "Low Volatility Bear Market"
            else:
                return "Low Volatility Sideways"
        else:
            if returns > 0.05:
                return "Moderate Bull Market"
            elif returns < -0.05:
                return "Moderate Bear Market"
            else:
                return "Moderate Sideways"

    def _calculate_regime_confidence(
        self, features_scaled: np.ndarray, cluster_labels: np.ndarray, regime_label: int
    ) -> float:
        """Calculate confidence in regime classification."""
        regime_mask = cluster_labels == regime_label
        regime_features = features_scaled[regime_mask]

        if len(regime_features) == 0:
            return 0.0

        # Calculate silhouette score-like metric
        regime_center = self.kmeans.cluster_centers_[regime_label]
        intra_cluster_dist = np.mean(np.linalg.norm(regime_features - regime_center, axis=1))

        # Calculate distance to nearest other cluster
        other_centers = [self.kmeans.cluster_centers_[i] for i in range(self.n_regimes) if i != regime_label]
        if other_centers:
            min_inter_cluster_dist = min(np.linalg.norm(regime_center - other_center) for other_center in other_centers)
            confidence = min_inter_cluster_dist / (min_inter_cluster_dist + intra_cluster_dist)
        else:
            confidence = 1.0

        return min(confidence, 1.0)

    def get_current_regime(self, data: pd.DataFrame, lookback_days: int = 30) -> Optional[RegimeInfo]:
        """Get the current market regime.

        Args:
            data: Recent market data
            lookback_days: Number of days to look back

        Returns:
            Current regime information
        """
        if len(data) < lookback_days:
            return None

        recent_data = data.tail(lookback_days)
        regimes = self.detect_regimes(recent_data)

        if not regimes:
            return None

        # Return the most recent regime
        return max(regimes, key=lambda r: r.end_date)


class WalkForwardOptimizer:
    """Walk-forward optimization with regime detection and validation."""

    def __init__(
        self,
        training_window: int = 252,
        validation_window: int = 63,
        step_size: int = 21,
        regime_detector: Optional[RegimeDetector] = None,
        validator: Optional[StrategyValidator] = None,
        top_n_configs: int = 10,
    ):
        """Initialize walk-forward optimizer.

        Args:
            training_window: Training period length in days
            validation_window: Validation period length in days
            step_size: Step size for walk-forward in days
            regime_detector: Regime detector instance
            validator: Strategy validator instance
            top_n_configs: Number of top configurations to log
        """
        self.training_window = training_window
        self.validation_window = validation_window
        self.step_size = step_size
        self.regime_detector = regime_detector or RegimeDetector()
        self.validator = validator or StrategyValidator()
        self.top_n_configs = top_n_configs
        self.results = []
        self.top_configurations = []
        self.logger = logging.getLogger(__name__)

    def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        strategy_class: Any,
        param_space: Dict[str, Any],
        optimization_method: str = "bayesian",
        **kwargs,
    ) -> List[WalkForwardResult]:
        """Run walk-forward analysis with validation.

        Args:
            data: Market data
            strategy_class: Strategy class to optimize
            param_space: Parameter space for optimization
            optimization_method: Optimization method to use
            **kwargs: Additional arguments

        Returns:
            List of walk-forward results
        """
        self.logger.info("Starting walk-forward analysis...")

        # Detect regimes
        regimes = self.regime_detector.detect_regimes(data)
        self.logger.info(f"Detected {len(regimes)} market regimes")

        # Generate time windows
        windows = self._generate_windows(data.index)
        self.logger.info(f"Generated {len(windows)} walk-forward windows")

        results = []
        all_configurations = []

        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}: {train_start.date()} to {val_end.date()}")

            # Get training and validation data
            train_data = data.loc[train_start:train_end]
            val_data = data.loc[val_start:val_end]

            if len(train_data) < self.training_window * 0.8:
                self.logger.warning(f"Window {i+1}: Insufficient training data")
                continue

            # Optimize strategy
            optimization_results = self._optimize_strategy(
                train_data, strategy_class, param_space, optimization_method, **kwargs
            )

            # Validate and rank configurations
            validated_results = self._validate_configurations(optimization_results, val_data, strategy_class)

            # Log top configurations
            self._log_top_configurations(validated_results, i + 1, len(windows))

            # Store all configurations for final ranking
            all_configurations.extend(validated_results)

            # Get best configuration
            if validated_results:
                best_result = max(validated_results, key=lambda x: x.sharpe_ratio)

                # Evaluate on validation set
                val_performance = self._evaluate_strategy(val_data, strategy_class, best_result.params)

                # Get regime for this period
                regime = self._get_regime_for_period(regimes, val_start, val_end)

                # Create walk-forward result
                result = WalkForwardResult(
                    period_start=train_start.isoformat(),
                    period_end=val_end.isoformat(),
                    training_period=f"{train_start.date()} to {train_end.date()}",
                    validation_period=f"{val_start.date()} to {val_end.date()}",
                    best_params=best_result.params,
                    validation_performance=val_performance,
                    out_of_sample_performance={},  # Will be filled later
                    regime=regime.regime_name if regime else "Unknown",
                    regime_confidence=regime.confidence if regime else 0.0,
                    timestamp=datetime.now().isoformat(),
                    return_consistency_score=best_result.return_consistency,
                    validation_passed=best_result.validation_passed,
                )

                results.append(result)

        self.results = results

        # Final ranking of all configurations
        self._rank_all_configurations(all_configurations)

        self.logger.info(f"Walk-forward analysis completed. {len(results)} valid results.")
        return results

    def _validate_configurations(
        self, optimization_results: List[Dict[str, Any]], val_data: pd.DataFrame, strategy_class: Any
    ) -> List[OptimizationResult]:
        """Validate optimization results and return ranked configurations."""
        validated_results = []

        for result in optimization_results:
            try:
                # Evaluate on validation set
                val_performance = self._evaluate_strategy(val_data, strategy_class, result["params"])

                # Calculate returns for validation
                if "returns" in val_performance:
                    returns = pd.Series(val_performance["returns"])
                    validation = self.validator.validate_returns(returns)

                    # Create optimization result
                    opt_result = OptimizationResult(
                        params=result["params"],
                        performance=val_performance,
                        sharpe_ratio=validation.get("sharpe_ratio", 0.0),
                        return_consistency=validation.get("consistency_score", 0.0),
                        validation_passed=validation.get("valid", False),
                        rank=0,  # Will be set later
                        timestamp=datetime.now().isoformat(),
                    )

                    validated_results.append(opt_result)

            except Exception as e:
                self.logger.warning(f"Failed to validate configuration: {e}")
                continue

        # Sort by Sharpe ratio
        validated_results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        # Assign ranks
        for i, result in enumerate(validated_results):
            result.rank = i + 1

        return validated_results

    def _log_top_configurations(self, configurations: List[OptimizationResult], window_num: int, total_windows: int):
        """Log top N configurations for current window."""
        if not configurations:
            return

        self.logger.info(
            f"\n=== Top {min(self.top_n_configs, len(configurations))} Configurations (Window {window_num}/{total_windows}) ==="
        )

        for i, config in enumerate(configurations[: self.top_n_configs]):
            self.logger.info(
                f"Rank {i+1}: Sharpe={config.sharpe_ratio:.3f}, "
                f"Consistency={config.return_consistency:.3f}, "
                f"Valid={config.validation_passed}, "
                f"Params={config.params}"
            )

        # Store top configurations
        self.top_configurations.extend(configurations[: self.top_n_configs])

    def _rank_all_configurations(self, all_configurations: List[OptimizationResult]):
        """Rank all configurations across all windows."""
        if not all_configurations:
            return

        # Group by parameter combinations
        param_groups = defaultdict(list)
        for config in all_configurations:
            param_key = json.dumps(config.params, sort_keys=True)
            param_groups[param_key].append(config)

        # Calculate aggregate scores
        aggregate_scores = []
        for param_key, configs in param_groups.items():
            avg_sharpe = np.mean([c.sharpe_ratio for c in configs])
            avg_consistency = np.mean([c.return_consistency for c in configs])
            success_rate = np.mean([c.validation_passed for c in configs])

            aggregate_scores.append(
                {
                    "params": configs[0].params,
                    "avg_sharpe": avg_sharpe,
                    "avg_consistency": avg_consistency,
                    "success_rate": success_rate,
                    "occurrences": len(configs),
                    "configs": configs,
                }
            )

        # Sort by average Sharpe ratio
        aggregate_scores.sort(key=lambda x: x["avg_sharpe"], reverse=True)

        # Log top configurations across all windows
        self.logger.info(f"\n=== Top {self.top_n_configs} Configurations (All Windows) ===")
        for i, score in enumerate(aggregate_scores[: self.top_n_configs]):
            self.logger.info(
                f"Rank {i+1}: Avg Sharpe={score['avg_sharpe']:.3f}, "
                f"Avg Consistency={score['avg_consistency']:.3f}, "
                f"Success Rate={score['success_rate']:.2f}, "
                f"Occurrences={score['occurrences']}, "
                f"Params={score['params']}"
            )

    def _generate_windows(
        self, dates: pd.DatetimeIndex
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate walk-forward windows."""
        windows = []
        start_idx = 0

        while start_idx + self.training_window + self.validation_window <= len(dates):
            train_start = dates[start_idx]
            train_end = dates[start_idx + self.training_window - 1]
            val_start = dates[start_idx + self.training_window]
            val_end = dates[start_idx + self.training_window + self.validation_window - 1]

            windows.append((train_start, train_end, val_start, val_end))
            start_idx += self.step_size

        return windows

    def _get_regime_for_period(
        self, regimes: List[RegimeInfo], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> Optional[RegimeInfo]:
        """Get regime for a specific time period."""
        for regime in regimes:
            regime_start = pd.Timestamp(regime.start_date)
            regime_end = pd.Timestamp(regime.end_date)

            if regime_start <= start_date <= regime_end or regime_start <= end_date <= regime_end:
                return regime

        return None

    def _optimize_strategy(
        self, data: pd.DataFrame, strategy_class: Any, param_space: Dict[str, Any], optimization_method: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Optimize strategy parameters."""
        # This would integrate with the actual optimization framework
        # For now, return mock results
        return [
            {"params": {"param1": 0.5, "param2": 10}, "performance": {"sharpe": 1.2, "returns": [0.01, -0.005, 0.02]}},
            {
                "params": {"param1": 0.7, "param2": 15},
                "performance": {"sharpe": 1.5, "returns": [0.015, -0.003, 0.025]},
            },
        ]

    def _evaluate_strategy(self, data: pd.DataFrame, strategy_class: Any, params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate strategy performance."""
        # This would run the actual strategy backtest
        # For now, return mock performance
        return {
            "sharpe_ratio": 1.2,
            "total_return": 0.15,
            "max_drawdown": 0.08,
            "volatility": 0.12,
            "returns": [0.01, -0.005, 0.02, 0.015, -0.003],
        }

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze walk-forward results."""
        if not self.results:
            return {"error": "No results to analyze"}

        # Calculate aggregate statistics
        sharpe_ratios = [r.validation_performance.get("sharpe_ratio", 0) for r in self.results]
        consistency_scores = [r.return_consistency_score for r in self.results]
        validation_passed = [r.validation_passed for r in self.results]

        analysis = {
            "total_windows": len(self.results),
            "successful_windows": sum(validation_passed),
            "success_rate": sum(validation_passed) / len(self.results) if self.results else 0,
            "avg_sharpe_ratio": np.mean(sharpe_ratios),
            "std_sharpe_ratio": np.std(sharpe_ratios),
            "avg_consistency_score": np.mean(consistency_scores),
            "regime_breakdown": self._analyze_regime_performance(),
            "parameter_stability": self._analyze_parameter_stability(),
        }

        return analysis

    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze performance by regime."""
        regime_performance = defaultdict(list)

        for result in self.results:
            regime = result.regime
            sharpe = result.validation_performance.get("sharpe_ratio", 0)
            regime_performance[regime].append(sharpe)

        regime_analysis = {}
        for regime, sharpe_ratios in regime_performance.items():
            regime_analysis[regime] = {
                "count": len(sharpe_ratios),
                "avg_sharpe": np.mean(sharpe_ratios),
                "std_sharpe": np.std(sharpe_ratios),
            }

        return dict(regime_analysis)

    def _analyze_parameter_stability(self) -> Dict[str, Any]:
        """Analyze parameter stability across windows."""
        param_values = defaultdict(list)

        for result in self.results:
            for param, value in result.best_params.items():
                param_values[param].append(value)

        stability_analysis = {}
        for param, values in param_values.items():
            stability_analysis[param] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "stability_score": 1 - (np.std(values) / (np.mean(values) + 1e-8)),
            }

        return stability_analysis

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot walk-forward results."""
        if not self.results:
            self.logger.warning("No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Sharpe ratios over time
        sharpe_ratios = [r.validation_performance.get("sharpe_ratio", 0) for r in self.results]
        axes[0, 0].plot(sharpe_ratios)
        axes[0, 0].set_title("Sharpe Ratios Over Time")
        axes[0, 0].set_ylabel("Sharpe Ratio")

        # Consistency scores
        consistency_scores = [r.return_consistency_score for r in self.results]
        axes[0, 1].plot(consistency_scores)
        axes[0, 1].set_title("Return Consistency Scores")
        axes[0, 1].set_ylabel("Consistency Score")

        # Regime distribution
        regimes = [r.regime for r in self.results]
        regime_counts = pd.Series(regimes).value_counts()
        axes[1, 0].pie(regime_counts.values, labels=regime_counts.index, autopct="%1.1f%%")
        axes[1, 0].set_title("Regime Distribution")

        # Validation success rate
        validation_passed = [r.validation_passed for r in self.results]
        success_rate = sum(validation_passed) / len(validation_passed)
        axes[1, 1].bar(["Passed", "Failed"], [success_rate, 1 - success_rate])
        axes[1, 1].set_title("Validation Success Rate")
        axes[1, 1].set_ylabel("Proportion")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Results plot saved to {save_path}")

        plt.show()

    def export_results(self, filepath: str) -> Dict[str, Any]:
        """Export results to file."""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for export
            export_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_windows": len(self.results),
                    "training_window": self.training_window,
                    "validation_window": self.validation_window,
                    "step_size": self.step_size,
                },
                "results": [asdict(r) for r in self.results],
                "top_configurations": [asdict(c) for c in self.top_configurations[: self.top_n_configs]],
                "analysis": self.analyze_results(),
            }

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Results exported to {output_path}")
            return {"success": True, "filepath": str(output_path)}

        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return {"success": False, "error": str(e)}


class BacktestOptimizer:
    """Comprehensive backtest optimizer with validation and logging."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize backtest optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.regime_detector = RegimeDetector(
            n_regimes=self.config.get("n_regimes", 3), lookback_window=self.config.get("lookback_window", 252)
        )

        self.validator = StrategyValidator(
            min_sharpe=self.config.get("min_sharpe", 0.5),
            max_drawdown=self.config.get("max_drawdown", 0.3),
            min_consistency=self.config.get("min_consistency", 0.6),
        )

        self.walk_forward_optimizer = WalkForwardOptimizer(
            training_window=self.config.get("training_window", 252),
            validation_window=self.config.get("validation_window", 63),
            step_size=self.config.get("step_size", 21),
            regime_detector=self.regime_detector,
            validator=self.validator,
            top_n_configs=self.config.get("top_n_configs", 10),
        )

        self.logger = logging.getLogger(__name__)

    def run_comprehensive_backtest(
        self, data: pd.DataFrame, strategy_class: Any, param_space: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Run comprehensive backtest with validation and logging.

        Args:
            data: Market data
            strategy_class: Strategy class to test
            param_space: Parameter space for optimization
            **kwargs: Additional arguments

        Returns:
            Comprehensive backtest results
        """
        self.logger.info("Starting comprehensive backtest...")

        # Run walk-forward analysis
        walk_forward_results = self.walk_forward_optimizer.run_walk_forward_analysis(
            data, strategy_class, param_space, **kwargs
        )

        # Analyze results
        analysis = self.walk_forward_optimizer.analyze_results()

        # Get regime recommendations
        regime_recommendations = self.get_regime_recommendations(data)

        # Compile comprehensive results
        results = {
            "walk_forward_results": walk_forward_results,
            "analysis": analysis,
            "regime_recommendations": regime_recommendations,
            "top_configurations": self.walk_forward_optimizer.top_configurations[:10],
            "validation_summary": {
                "total_tests": len(walk_forward_results),
                "passed_tests": sum(1 for r in walk_forward_results if r.validation_passed),
                "success_rate": sum(1 for r in walk_forward_results if r.validation_passed) / len(walk_forward_results)
                if walk_forward_results
                else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info("Comprehensive backtest completed")
        return results

    def get_regime_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Get regime-based recommendations."""
        try:
            current_regime = self.regime_detector.get_current_regime(data)

            if not current_regime:
                return ["Unable to determine current regime"]

            recommendations = []

            # Add regime-specific recommendations
            if "High Volatility" in current_regime.regime_name:
                recommendations.append("Consider reducing position sizes due to high volatility")
                recommendations.append("Implement tighter stop-losses")
            elif "Low Volatility" in current_regime.regime_name:
                recommendations.append("Consider increasing position sizes due to low volatility")
                recommendations.append("Wider stop-losses may be appropriate")

            if "Bull Market" in current_regime.regime_name:
                recommendations.append("Favor long positions and trend-following strategies")
            elif "Bear Market" in current_regime.regime_name:
                recommendations.append("Consider defensive positioning and short strategies")

            recommendations.append(
                f"Current regime: {current_regime.regime_name} (confidence: {current_regime.confidence:.2f})"
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating regime recommendations: {e}")
            return ["Error generating recommendations"]


def get_backtest_optimizer(config: Optional[Dict[str, Any]] = None) -> BacktestOptimizer:
    """Get backtest optimizer instance.

    Args:
        config: Configuration dictionary

    Returns:
        BacktestOptimizer instance
    """
    return BacktestOptimizer(config)
