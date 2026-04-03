"""
Performance Analyzer Module

This module contains performance analysis functionality for the optimizer agent.
Extracted from the original optimizer_agent.py for modularity.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .strategy_optimizer import OptimizationMetric


@dataclass
class OptimizationResult:
    """Optimization result data class."""

    parameter_combination: Dict[str, Any]
    performance_metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    optimization_score: float
    timestamp: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result_dict = asdict(self)
        result_dict["timestamp"] = self.timestamp.isoformat()
        return result_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        """Create from dictionary."""
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class PerformanceAnalyzer:
    """Analyzes optimization performance and manages results."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Storage
        self.results_storage = Path("data/optimization")
        self.results_storage.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.optimization_history = []
        self.best_results = {}

        # Load existing results
        self._load_optimization_history()

    def _load_optimization_history(self) -> None:
        """Load optimization history from storage."""
        try:
            history_file = self.results_storage / "optimization_history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    history_data = json.load(f)
                    self.optimization_history = [
                        OptimizationResult.from_dict(result) for result in history_data
                    ]
        except Exception as e:
            self.logger.error(f"Failed to load optimization history: {e}")
            self.optimization_history = []

    def _save_optimization_history(self) -> None:
        """Save optimization history to storage."""
        try:
            history_file = self.results_storage / "optimization_history.json"
            history_data = [result.to_dict() for result in self.optimization_history]
            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save optimization history: {e}")

    def add_optimization_result(self, result: OptimizationResult) -> None:
        """Add an optimization result to history."""
        self.optimization_history.append(result)
        self._save_optimization_history()

        # Update best results
        self._update_best_results(result)

    def _update_best_results(self, result: OptimizationResult) -> None:
        """Update best results tracking."""
        # Track best result by optimization score
        if not self.best_results or result.optimization_score > self.best_results.get(
            "best_score", 0
        ):
            self.best_results["best_score"] = result.optimization_score
            self.best_results["best_result"] = result.to_dict()

    def get_optimization_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get optimization history."""
        recent_history = (
            self.optimization_history[-limit:]
            if limit > 0
            else self.optimization_history
        )
        return [result.to_dict() for result in recent_history]

    def get_best_results(self) -> Dict[str, OptimizationResult]:
        """Get best optimization results."""
        return self.best_results

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {
                "total_optimizations": 0,
                "average_score": 0.0,
                "best_score": 0.0,
                "success_rate": 0.0,
            }

        scores = [result.optimization_score for result in self.optimization_history]
        successful_results = [
            result
            for result in self.optimization_history
            if result.optimization_score > 0
        ]

        # Safely calculate metrics with division-by-zero protection
        total_optimizations = len(self.optimization_history)
        num_scores = len(scores)
        num_successful = len(successful_results)

        return {
            "total_optimizations": total_optimizations,
            "average_score": sum(scores) / num_scores if num_scores > 0 else 0.0,
            "best_score": max(scores) if scores else 0.0,
            "success_rate": (
                num_successful / total_optimizations 
                if total_optimizations > 0 
                else 0.0
            ),
            "recent_trend": self._calculate_recent_trend(),
        }

    def _calculate_recent_trend(self) -> str:
        """Calculate recent optimization trend."""
        if len(self.optimization_history) < 5:
            return "insufficient_data"

        recent_scores = [
            result.optimization_score for result in self.optimization_history[-5:]
        ]

        if len(recent_scores) < 2:
            return "insufficient_data"

        # Calculate trend with safe division
        mid_point = len(recent_scores) // 2
        first_half_len = mid_point
        second_half_len = len(recent_scores) - mid_point

        if first_half_len > 0 and second_half_len > 0:
            first_half = sum(recent_scores[:mid_point]) / first_half_len
            second_half = sum(recent_scores[mid_point:]) / second_half_len
        else:
            return "insufficient_data"

        if second_half > first_half * 1.1:
            return "improving"
        elif second_half < first_half * 0.9:
            return "declining"
        else:
            return "stable"

    def analyze_performance_by_metric(
        self, metric: OptimizationMetric
    ) -> Dict[str, Any]:
        """Analyze performance by specific metric."""
        if not self.optimization_history:
            return {}

        metric_values = []
        for result in self.optimization_history:
            metric_value = result.performance_metrics.get(metric.value, 0.0)
            metric_values.append(metric_value)

        if not metric_values:
            return {}

        return {
            "metric": metric.value,
            "average": sum(metric_values) / len(metric_values),
            "min": min(metric_values),
            "max": max(metric_values),
            "std_dev": self._calculate_std_dev(metric_values),
            "top_10_percentile": sorted(metric_values)[-len(metric_values) // 10],
        }

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def find_optimal_parameters(
        self, metric: OptimizationMetric, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Find optimal parameter combinations for a metric."""
        if not self.optimization_history:
            return []

        # Sort results by the specified metric
        sorted_results = sorted(
            self.optimization_history,
            key=lambda x: x.performance_metrics.get(metric.value, 0.0),
            reverse=True,
        )

        # Return top N results
        top_results = []
        for result in sorted_results[:top_n]:
            top_results.append(
                {
                    "parameter_combination": result.parameter_combination,
                    "performance_metrics": result.performance_metrics,
                    "optimization_score": result.optimization_score,
                    "timestamp": result.timestamp.isoformat(),
                }
            )

        return top_results

    def compare_optimization_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple optimization runs."""
        # This would require run IDs to be stored with results
        # For now, return a placeholder
        return {"message": "Run comparison not implemented yet", "run_ids": run_ids}

    def export_optimization_report(self, format: str = "json") -> str:
        """Export optimization report."""
        try:
            report_data = {
                "optimization_stats": self.get_optimization_stats(),
                "best_results": self.best_results,
                "recent_history": self.get_optimization_history(10),
                "export_timestamp": datetime.utcnow().isoformat(),
            }

            if format.lower() == "json":
                report_file = (
                    self.results_storage
                    / f"optimization_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(report_file, "w") as f:
                    json.dump(report_data, f, indent=2, default=str)
                return str(report_file)

            else:
                self.logger.warning(f"Unsupported export format: {format}")
                return ""

        except Exception as e:
            self.logger.error(f"Failed to export optimization report: {e}")
            return ""

    def clear_optimization_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history = []
        self.best_results = {}
        self._save_optimization_history()
