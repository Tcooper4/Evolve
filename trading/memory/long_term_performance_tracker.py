"""
Long-Term Performance Tracker

Tracks and analyzes system performance over extended periods.
Provides insights into performance trends and degradation.
Enhanced with rolling Sharpe, drawdown tracking, and improvement monitoring.
"""

import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.safe_json_saver import safe_save_historical_data

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""

    timestamp: datetime
    metric_name: str
    value: float
    context: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceTrend:
    """Represents a performance trend."""

    metric_name: str
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0.0 to 1.0
    period_days: int
    average_value: float
    volatility: float
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    days_since_improvement: int = 0
    improvement_threshold: float = 0.01


@dataclass
class DrawdownInfo:
    """Information about drawdown periods."""

    start_date: datetime
    end_date: Optional[datetime] = None
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int = 0
    recovery_days: int = 0


@dataclass
class SharpeMetrics:
    """Sharpe ratio and related metrics."""

    sharpe_ratio: float
    returns: List[float]
    volatility: float
    risk_free_rate: float
    excess_returns: List[float]
    rolling_sharpe: List[float]


class LongTermPerformanceTracker:
    """Tracks long-term performance metrics and trends with advanced financial metrics."""

    def __init__(self, retention_days: int = 365, risk_free_rate: float = 0.02):
        """
        Initialize the performance tracker.

        Args:
            retention_days: Number of days to retain performance data
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.retention_days = retention_days
        self.risk_free_rate = risk_free_rate
        self.metrics: List[PerformanceMetric] = []
        self.trends: Dict[str, PerformanceTrend] = {}
        self.alerts: List[Dict[str, Any]] = []

        # Advanced tracking
        self.drawdowns: Dict[str, List[DrawdownInfo]] = defaultdict(list)
        self.sharpe_metrics: Dict[str, SharpeMetrics] = {}
        self.improvement_tracking: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Rolling windows for calculations
        self.rolling_window_days = 30
        self.rolling_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.rolling_window_days)
        )

        # Thread safety
        self._lock = threading.RLock()

        # Load historical data if available
        self._load_historical_data()

        logger.info("Long-term performance tracker initialized with advanced metrics")

    def _load_historical_data(self):
        """Load historical performance data from disk."""
        try:
            data_file = Path("data/performance_history.json")
            if data_file.exists():
                with open(data_file, "r") as f:
                    data = json.load(f)

                # Load metrics
                for metric_data in data.get("metrics", []):
                    metric = PerformanceMetric(
                        timestamp=datetime.fromisoformat(metric_data["timestamp"]),
                        metric_name=metric_data["metric_name"],
                        value=metric_data["value"],
                        context=metric_data.get("context", {}),
                        metadata=metric_data.get("metadata", {}),
                    )
                    self.metrics.append(metric)

                # Load trends
                for name, trend_data in data.get("trends", {}).items():
                    trend = PerformanceTrend(**trend_data)
                    self.trends[name] = trend

                logger.info(f"Loaded {len(self.metrics)} historical metrics")

        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")

    def _save_historical_data(self):
        """Save performance data to disk."""
        try:
            data_file = Path("data/performance_history.json")

            data = {
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "metric_name": m.metric_name,
                        "value": m.value,
                        "context": m.context,
                        "metadata": m.metadata,
                    }
                    for m in self.metrics
                ],
                "trends": {name: asdict(trend) for name, trend in self.trends.items()},
            }

            # Use safe JSON saving to prevent data loss
            result = safe_save_historical_data(data, data_file)
            if not result["success"]:
                logger.error(f"Failed to save historical data: {result['error']}")
            else:
                logger.debug(
                    f"Successfully saved historical data: {result['filepath']}"
                )

        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")

    def record_metric(
        self,
        metric_name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a performance metric with advanced tracking.

        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Context information
            metadata: Additional metadata
        """
        with self._lock:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                context=context or {},
                metadata=metadata or {},
            )

            self.metrics.append(metric)

            # Update rolling metrics
            self.rolling_metrics[metric_name].append(metric)

            # Update advanced metrics
            self._update_sharpe_metrics(metric_name)
            self._update_drawdown_tracking(metric_name)
            self._update_improvement_tracking(metric_name)

            # Clean old metrics
            self._clean_old_metrics()

            # Check for alerts
            self._check_alerts(metric)

            # Save periodically
            if len(self.metrics) % 100 == 0:
                self._save_historical_data()

            logger.info(f"Recorded metric: {metric_name} = {value}")

    def _clean_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_date]

    def _update_sharpe_metrics(self, metric_name: str):
        """Update Sharpe ratio metrics for a given metric."""
        try:
            metrics = [m for m in self.metrics if m.metric_name == metric_name]
            if len(metrics) < 2:
                return

            # Calculate returns with safe division
            values = np.array([m.value for m in metrics])

            if len(values) < 2:
                return

            from trading.utils.safe_math import safe_returns
            returns = safe_returns(values, method='simple')

            if len(returns) == 0:
                return

            # Calculate excess returns
            daily_rf_rate = (1 + self.risk_free_rate) ** (1 / 365) - 1
            excess_returns = returns - daily_rf_rate

            # Calculate Sharpe ratio
            if np.std(excess_returns) > 0:
                sharpe_ratio = (
                    np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)
                )
            else:
                sharpe_ratio = 0.0

            # Calculate rolling Sharpe
            rolling_sharpe = []
            window_size = min(30, len(returns))

            for i in range(window_size, len(returns)):
                window_returns = returns[i - window_size : i]
                window_excess = window_returns - daily_rf_rate
                if np.std(window_excess) > 0:
                    rolling_sharpe.append(
                        np.mean(window_excess) / np.std(window_excess) * np.sqrt(365)
                    )
                else:
                    rolling_sharpe.append(0.0)

            self.sharpe_metrics[metric_name] = SharpeMetrics(
                sharpe_ratio=sharpe_ratio,
                returns=returns.tolist(),
                volatility=np.std(returns) * np.sqrt(365),
                risk_free_rate=self.risk_free_rate,
                excess_returns=excess_returns.tolist(),
                rolling_sharpe=rolling_sharpe,
            )

        except Exception as e:
            logger.error(f"Failed to update Sharpe metrics for {metric_name}: {e}")

    def _update_drawdown_tracking(self, metric_name: str):
        """Update drawdown tracking for a given metric."""
        try:
            metrics = [m for m in self.metrics if m.metric_name == metric_name]
            if len(metrics) < 2:
                return

            values = [m.value for m in metrics]
            timestamps = [m.timestamp for m in metrics]

            # Find current drawdown
            peak_value = max(values)
            current_value = values[-1]
            current_drawdown = (peak_value - current_value) / peak_value

            # Check if we're in a new drawdown
            if current_drawdown > 0.01:  # 1% threshold
                # Check if we need to start a new drawdown period
                if (
                    not self.drawdowns[metric_name]
                    or self.drawdowns[metric_name][-1].end_date is not None
                ):
                    # Start new drawdown
                    drawdown = DrawdownInfo(
                        start_date=timestamps[-1],
                        peak_value=peak_value,
                        trough_value=current_value,
                        drawdown_pct=current_drawdown,
                    )
                    self.drawdowns[metric_name].append(drawdown)
                else:
                    # Update existing drawdown
                    current_dd = self.drawdowns[metric_name][-1]
                    if current_value < current_dd.trough_value:
                        current_dd.trough_value = current_value
                        current_dd.drawdown_pct = (
                            current_dd.peak_value - current_value
                        ) / current_dd.peak_value
            else:
                # Check if we've recovered from drawdown
                if (
                    self.drawdowns[metric_name]
                    and self.drawdowns[metric_name][-1].end_date is None
                ):
                    current_dd = self.drawdowns[metric_name][-1]
                    current_dd.end_date = timestamps[-1]
                    current_dd.duration_days = (
                        current_dd.end_date - current_dd.start_date
                    ).days
                    current_dd.recovery_days = (
                        0  # Will be updated when we reach new peak
                    )

        except Exception as e:
            logger.error(f"Failed to update drawdown tracking for {metric_name}: {e}")

    def _update_improvement_tracking(self, metric_name: str):
        """Update improvement tracking for a given metric."""
        try:
            metrics = [m for m in self.metrics if m.metric_name == metric_name]
            if len(metrics) < 2:
                return

            current_value = metrics[-1].value
            previous_value = metrics[-2].value

            # Check for improvement
            improvement = (current_value - previous_value) / previous_value

            if improvement > self.improvement_tracking[metric_name].get(
                "improvement_threshold", 0.01
            ):
                # Record improvement
                self.improvement_tracking[metric_name] = {
                    "last_improvement_date": metrics[-1].timestamp,
                    "last_improvement_value": current_value,
                    "improvement_pct": improvement,
                    "days_since_improvement": 0,
                    "improvement_threshold": 0.01,
                }
            else:
                # Update days since last improvement
                if "last_improvement_date" in self.improvement_tracking[metric_name]:
                    days_since = (
                        metrics[-1].timestamp
                        - self.improvement_tracking[metric_name][
                            "last_improvement_date"
                        ]
                    ).days
                    self.improvement_tracking[metric_name][
                        "days_since_improvement"
                    ] = days_since

        except Exception as e:
            logger.error(
                f"Failed to update improvement tracking for {metric_name}: {e}"
            )

    def _check_alerts(self, metric: PerformanceMetric) -> None:
        """Check for performance alerts with advanced criteria."""
        try:
            # Get recent metrics for this metric type
            recent_metrics = [
                m for m in self.metrics[-100:] if m.metric_name == metric.metric_name
            ]  # Last 100 metrics

            if len(recent_metrics) < 10:
                return

            # Calculate statistics
            values = [m.value for m in recent_metrics]
            mean_value = np.mean(values)
            std_value = np.std(values)

            # Check for significant deviations
            if abs(metric.value - mean_value) > 2 * std_value:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "expected_range": f"{mean_value - 2 * std_value:.2f} to {mean_value + 2 * std_value:.2f}",
                    "severity": (
                        "high"
                        if abs(metric.value - mean_value) > 3 * std_value
                        else "medium"
                    ),
                    "context": metric.context,
                }

                self.alerts.append(alert)
                logger.warning(f"Performance alert: {alert}")

            # Check for drawdown alerts
            if (
                metric.metric_name in self.drawdowns
                and self.drawdowns[metric.metric_name]
            ):
                current_dd = self.drawdowns[metric.metric_name][-1]
                if current_dd.drawdown_pct > 0.1:  # 10% drawdown
                    alert = {
                        "timestamp": datetime.now().isoformat(),
                        "metric_name": metric.metric_name,
                        "alert_type": "drawdown",
                        "drawdown_pct": current_dd.drawdown_pct,
                        "duration_days": current_dd.duration_days,
                        "severity": (
                            "high" if current_dd.drawdown_pct > 0.2 else "medium"
                        ),
                    }
                    self.alerts.append(alert)
                    logger.warning(f"Drawdown alert: {alert}")

            # Check for improvement stagnation
            if metric.metric_name in self.improvement_tracking:
                days_since = self.improvement_tracking[metric.metric_name].get(
                    "days_since_improvement", 0
                )
                if days_since > 30:  # 30 days without improvement
                    alert = {
                        "timestamp": datetime.now().isoformat(),
                        "metric_name": metric.metric_name,
                        "alert_type": "stagnation",
                        "days_since_improvement": days_since,
                        "severity": "medium",
                    }
                    self.alerts.append(alert)
                    logger.warning(f"Stagnation alert: {alert}")

        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")

    def analyze_trends(
        self, metric_name: Optional[str] = None, period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze performance trends with advanced metrics.

        Args:
            metric_name: Specific metric to analyze (None for all)
            period_days: Analysis period in days

        Returns:
            Dictionary with trend analysis including Sharpe and drawdown
        """
        with self._lock:
            cutoff_date = datetime.now() - timedelta(days=period_days)

            # Filter metrics by period and name
            recent_metrics = [
                m
                for m in self.metrics
                if m.timestamp > cutoff_date
                and (metric_name is None or m.metric_name == metric_name)
            ]

            if not recent_metrics:
                return {"message": "No metrics available for analysis"}

            # Group by metric name
            metrics_by_name = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_name[metric.metric_name].append(metric)

            trends = {}
            for name, metrics in metrics_by_name.items():
                if len(metrics) < 5:  # Need at least 5 data points
                    continue

                # Calculate basic trend
                values = [m.value for m in metrics]
                timestamps = [m.timestamp for m in metrics]

                # Simple linear trend
                x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
                y = np.array(values)

                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    trend_strength = abs(slope) / (np.std(y) + 1e-8)

                    # Determine trend direction
                    if slope > 0.01:
                        direction = "improving"
                    elif slope < -0.01:
                        direction = "declining"
                    else:
                        direction = "stable"

                    # Get advanced metrics
                    sharpe_ratio = self.sharpe_metrics.get(
                        name, SharpeMetrics(0.0, [], 0.0, 0.0, [], [])
                    ).sharpe_ratio
                    max_drawdown = self._get_current_drawdown(name)
                    days_since_improvement = self.improvement_tracking.get(
                        name, {}
                    ).get("days_since_improvement", 0)

                    trend = PerformanceTrend(
                        metric_name=name,
                        trend_direction=direction,
                        trend_strength=min(1.0, trend_strength),
                        period_days=period_days,
                        average_value=np.mean(values),
                        volatility=np.std(values),
                        sharpe_ratio=sharpe_ratio,
                        max_drawdown=max_drawdown,
                        days_since_improvement=days_since_improvement,
                    )

                    trends[name] = trend
                    self.trends[name] = trend

            return {
                "period_days": period_days,
                "total_metrics": len(recent_metrics),
                "trends": {name: asdict(trend) for name, trend in trends.items()},
            }

    def _get_current_drawdown(self, metric_name: str) -> float:
        """Get current drawdown percentage for a metric."""
        if metric_name not in self.drawdowns or not self.drawdowns[metric_name]:
            return 0.0

        current_dd = self.drawdowns[metric_name][-1]
        if current_dd.end_date is None:
            return current_dd.drawdown_pct
        else:
            return 0.0

    def get_rolling_sharpe(
        self, metric_name: str, window_days: int = 30
    ) -> List[float]:
        """Get rolling Sharpe ratio for a metric.

        Args:
            metric_name: Name of the metric
            window_days: Rolling window size in days

        Returns:
            List of rolling Sharpe ratios
        """
        if metric_name not in self.sharpe_metrics:
            return []

        return self.sharpe_metrics[metric_name].rolling_sharpe

    def get_drawdown_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get drawdown history for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of drawdown periods
        """
        if metric_name not in self.drawdowns:
            return []

        return [
            {
                "start_date": dd.start_date.isoformat(),
                "end_date": dd.end_date.isoformat() if dd.end_date else None,
                "peak_value": dd.peak_value,
                "trough_value": dd.trough_value,
                "drawdown_pct": dd.drawdown_pct,
                "duration_days": dd.duration_days,
                "recovery_days": dd.recovery_days,
            }
            for dd in self.drawdowns[metric_name]
        ]

    def get_improvement_tracking(self, metric_name: str) -> Dict[str, Any]:
        """Get improvement tracking for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Improvement tracking information
        """
        return self.improvement_tracking.get(metric_name, {})

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary with advanced metrics."""
        with self._lock:
            if not self.metrics:
                return {"message": "No performance data available"}

            # Overall statistics
            all_values = [m.value for m in self.metrics]

            # Recent vs historical comparison
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_date]
            historical_metrics = [m for m in self.metrics if m.timestamp <= cutoff_date]

            summary = {
                "total_metrics": len(self.metrics),
                "recent_metrics": len(recent_metrics),
                "historical_metrics": len(historical_metrics),
                "overall_stats": {
                    "mean": np.mean(all_values),
                    "std": np.std(all_values),
                    "min": np.min(all_values),
                    "max": np.max(all_values),
                },
                "recent_vs_historical": self._compare_periods(
                    recent_metrics, historical_metrics
                ),
                "sharpe_metrics": {
                    name: {
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "volatility": metrics.volatility,
                        "rolling_sharpe_avg": (
                            np.mean(metrics.rolling_sharpe)
                            if metrics.rolling_sharpe
                            else 0.0
                        ),
                    }
                    for name, metrics in self.sharpe_metrics.items()
                },
                "drawdown_summary": {
                    name: {
                        "current_drawdown": self._get_current_drawdown(name),
                        "max_drawdown": (
                            max([dd.drawdown_pct for dd in drawdowns])
                            if drawdowns
                            else 0.0
                        ),
                        "total_drawdowns": len(drawdowns),
                    }
                    for name, drawdowns in self.drawdowns.items()
                },
                "improvement_summary": {
                    name: {
                        "days_since_improvement": tracking.get(
                            "days_since_improvement", 0
                        ),
                        "last_improvement_pct": tracking.get("improvement_pct", 0.0),
                    }
                    for name, tracking in self.improvement_tracking.items()
                },
                "active_alerts": len(
                    [a for a in self.alerts if a.get("severity") == "high"]
                ),
                "timestamp": datetime.now().isoformat(),
            }

            return summary

    def _compare_periods(
        self,
        recent_metrics: List[PerformanceMetric],
        historical_metrics: List[PerformanceMetric],
    ) -> Dict[str, Any]:
        """Compare recent vs historical performance."""
        if not recent_metrics or not historical_metrics:
            return {"message": "Insufficient data for comparison"}

        recent_values = [m.value for m in recent_metrics]
        historical_values = [m.value for m in historical_metrics]

        recent_mean = np.mean(recent_values)
        historical_mean = np.mean(historical_values)

        change_pct = (
            ((recent_mean - historical_mean) / historical_mean * 100)
            if historical_mean != 0
            else 0
        )

        return {
            "recent_mean": recent_mean,
            "historical_mean": historical_mean,
            "change_percent": change_pct,
            "improvement": change_pct > 0,
        }

    def get_performance_forecast(
        self, metric_name: str, forecast_days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate performance forecast with confidence intervals.

        Args:
            metric_name: Metric to forecast
            forecast_days: Number of days to forecast

        Returns:
            Forecast with confidence intervals
        """
        try:
            metrics = [m for m in self.metrics if m.metric_name == metric_name]
            if len(metrics) < 10:
                return {"error": "Insufficient data for forecasting"}

            values = [m.value for m in metrics]

            # Simple linear regression forecast
            x = np.arange(len(values))
            y = np.array(values)

            # Fit polynomial (degree 2 for non-linear trends)
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)

            # Generate forecast points
            future_x = np.arange(len(values), len(values) + forecast_days)
            forecast_values = poly(future_x)

            # Calculate confidence intervals
            residuals = y - poly(x)
            std_residual = np.std(residuals)

            confidence_interval = 1.96 * std_residual  # 95% confidence

            # Calculate forecast confidence
            confidence = self._calculate_forecast_confidence(values)

            return {
                "metric_name": metric_name,
                "forecast_days": forecast_days,
                "forecast_values": forecast_values.tolist(),
                "confidence_interval": confidence_interval,
                "forecast_confidence": confidence,
                "upper_bound": (forecast_values + confidence_interval).tolist(),
                "lower_bound": (forecast_values - confidence_interval).tolist(),
                "trend_direction": (
                    "increasing"
                    if forecast_values[-1] > forecast_values[0]
                    else "decreasing"
                ),
            }

        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            return {"error": str(e)}

    def _calculate_forecast_confidence(self, values: List[float]) -> float:
        """Calculate forecast confidence based on data stability."""
        if len(values) < 2:
            return 0.0

        # Calculate coefficient of variation
        cv = np.std(values) / np.mean(values)

        # Calculate trend consistency
        diffs = np.diff(values)
        trend_consistency = np.sum(np.sign(diffs[:-1]) == np.sign(diffs[1:])) / max(
            1, len(diffs) - 1
        )

        # Combine factors for confidence score
        confidence = (1 - cv) * 0.6 + trend_consistency * 0.4
        return max(0.0, min(1.0, confidence))

    def run(self) -> Dict[str, Any]:
        """Run the performance tracker analysis."""
        try:
            # Analyze trends for all metrics
            trend_analysis = self.analyze_trends()

            # Generate summary
            summary = self.get_performance_summary()

            # Check for critical alerts
            critical_alerts = [a for a in self.alerts if a.get("severity") == "high"]

            return {
                "success": True,
                "trend_analysis": trend_analysis,
                "summary": summary,
                "critical_alerts": len(critical_alerts),
                "total_alerts": len(self.alerts),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Performance tracker run failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def apply_performance_decay(self, decay_rate: float = 0.95) -> Dict[str, Any]:
        """
        Apply performance decay to older metrics.

        Args:
            decay_rate: Decay rate per day (0.95 = 5% decay per day)

        Returns:
            Decay application results
        """
        try:
            with self._lock:
                decayed_metrics = []
                total_decay_applied = 0

                for metric in self.metrics:
                    # Calculate days since metric was recorded
                    days_old = (datetime.now() - metric.timestamp).days

                    if days_old > 0:
                        # Apply decay
                        decay_factor = decay_rate**days_old
                        original_value = metric.value
                        metric.value *= decay_factor
                        total_decay_applied += original_value - metric.value

                        # Add decay info to metadata
                        if "decay_info" not in metric.metadata:
                            metric.metadata["decay_info"] = {}

                        metric.metadata["decay_info"].update(
                            {
                                "decay_rate": decay_rate,
                                "days_old": days_old,
                                "decay_factor": decay_factor,
                                "original_value": original_value,
                                "decay_applied": original_value - metric.value,
                            }
                        )

                        decayed_metrics.append(metric.metric_name)

                # Recalculate advanced metrics after decay
                for metric_name in set(decayed_metrics):
                    self._update_sharpe_metrics(metric_name)
                    self._update_drawdown_tracking(metric_name)

                return {
                    "success": True,
                    "metrics_decayed": len(decayed_metrics),
                    "total_decay_applied": total_decay_applied,
                    "decay_rate": decay_rate,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to apply performance decay: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_decay_adjusted_metrics(
        self, metric_name: str, decay_rate: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Get decay-adjusted metrics for analysis.

        Args:
            metric_name: Name of the metric
            decay_rate: Decay rate to apply

        Returns:
            List of decay-adjusted metrics
        """
        try:
            metrics = [m for m in self.metrics if m.metric_name == metric_name]
            adjusted_metrics = []

            for metric in metrics:
                days_old = (datetime.now() - metric.timestamp).days
                decay_factor = decay_rate**days_old
                adjusted_value = metric.value / decay_factor

                adjusted_metrics.append(
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "original_value": metric.value,
                        "adjusted_value": adjusted_value,
                        "decay_factor": decay_factor,
                        "days_old": days_old,
                    }
                )

            return adjusted_metrics

        except Exception as e:
            logger.error(f"Failed to get decay-adjusted metrics: {e}")
            return []

    def calculate_decay_weighted_average(
        self, metric_name: str, decay_rate: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate decay-weighted average for a metric.

        Args:
            metric_name: Name of the metric
            decay_rate: Decay rate to apply

        Returns:
            Decay-weighted average statistics
        """
        try:
            metrics = [m for m in self.metrics if m.metric_name == metric_name]
            if not metrics:
                return {"error": "No metrics found"}

            total_weight = 0
            weighted_sum = 0

            for metric in metrics:
                days_old = (datetime.now() - metric.timestamp).days
                weight = decay_rate**days_old
                weighted_sum += metric.value * weight
                total_weight += weight

            if total_weight == 0:
                return {"error": "No valid weights calculated"}

            weighted_average = weighted_sum / total_weight

            return {
                "metric_name": metric_name,
                "weighted_average": weighted_average,
                "total_metrics": len(metrics),
                "decay_rate": decay_rate,
                "effective_weight": total_weight,
            }

        except Exception as e:
            logger.error(f"Failed to calculate decay-weighted average: {e}")
            return {"error": str(e)}


# Global instance
_performance_tracker = None
_tracker_lock = threading.RLock()


def get_performance_tracker() -> LongTermPerformanceTracker:
    """Get the global performance tracker instance."""
    global _performance_tracker
    with _tracker_lock:
        if _performance_tracker is None:
            _performance_tracker = LongTermPerformanceTracker()
        return _performance_tracker
