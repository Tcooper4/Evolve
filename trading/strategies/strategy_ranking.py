"""
Strategy Ranking

Ranks strategies based on past success, prompt frequency, and confidence metrics.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyRanking:
    """Strategy ranking with metrics."""

    strategy_name: str
    rank: int
    score: float
    success_rate: float
    usage_frequency: float
    confidence_score: float
    recent_performance: float
    metadata: Dict[str, Any]


class StrategyRanker:
    """Ranks strategies based on multiple metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy ranker.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.history_file = self.config.get("history_file", "strategy_history.json")
        self.ranking_weights = self.config.get(
            "ranking_weights",
            {
                "success_rate": 0.4,
                "usage_frequency": 0.2,
                "confidence_score": 0.2,
                "recent_performance": 0.2,
            },
        )
        self.history_window = self.config.get("history_window_days", 30)
        self.min_usage_count = self.config.get("min_usage_count", 3)

        # Load strategy history
        self.strategy_history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        """Load strategy usage history from file."""
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"Creating new strategy history file: {self.history_file}")
            return {
                "strategies": {},
                "prompts": [],
                "last_updated": datetime.now().isoformat(),
            }

    def _save_history(self):
        """Save strategy history to file."""
        self.strategy_history["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.strategy_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save strategy history: {e}")

    def record_strategy_usage(
        self,
        strategy_name: str,
        prompt: str,
        success: bool,
        confidence: float,
        performance_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Record strategy usage for ranking.

        Args:
            strategy_name: Name of the strategy used
            prompt: Original prompt that triggered the strategy
            success: Whether the strategy execution was successful
            confidence: Confidence score of the strategy selection
            performance_metrics: Optional performance metrics
        """
        timestamp = datetime.now().isoformat()

        # Record prompt
        self.strategy_history["prompts"].append(
            {
                "timestamp": timestamp,
                "prompt": prompt,
                "strategy": strategy_name,
                "success": success,
                "confidence": confidence,
            }
        )

        # Update strategy statistics
        if strategy_name not in self.strategy_history["strategies"]:
            self.strategy_history["strategies"][strategy_name] = {
                "usage_count": 0,
                "success_count": 0,
                "total_confidence": 0.0,
                "performance_history": [],
                "last_used": timestamp,
                "first_used": timestamp,
            }

        strategy_stats = self.strategy_history["strategies"][strategy_name]
        strategy_stats["usage_count"] += 1
        strategy_stats["total_confidence"] += confidence
        strategy_stats["last_used"] = timestamp

        if success:
            strategy_stats["success_count"] += 1

        if performance_metrics:
            strategy_stats["performance_history"].append(
                {"timestamp": timestamp, "metrics": performance_metrics}
            )

        # Keep only recent history
        self._cleanup_old_history()

        # Save updated history
        self._save_history()

    def rank_strategies(self, prompt: str = None) -> List[StrategyRanking]:
        """
        Rank strategies based on historical performance and metrics.

        Args:
            prompt: Optional prompt for context-aware ranking

        Returns:
            List of strategy rankings
        """
        rankings = []

        for strategy_name, stats in self.strategy_history["strategies"].items():
            # Skip strategies with insufficient usage
            if stats["usage_count"] < self.min_usage_count:
                continue

            # Calculate metrics with division-by-zero protection
            usage_count = stats["usage_count"]
            if usage_count > 0:
                success_rate = stats["success_count"] / usage_count
                avg_confidence = stats["total_confidence"] / usage_count
            else:
                logger.warning(f"Strategy {strategy_name} has zero usage_count")
                success_rate = 0.0
                avg_confidence = 0.0
            usage_frequency = self._calculate_usage_frequency(strategy_name)
            recent_performance = self._calculate_recent_performance(strategy_name)

            # Calculate overall score
            score = (
                self.ranking_weights["success_rate"] * success_rate
                + self.ranking_weights["usage_frequency"] * usage_frequency
                + self.ranking_weights["confidence_score"] * avg_confidence
                + self.ranking_weights["recent_performance"] * recent_performance
            )

            ranking = StrategyRanking(
                strategy_name=strategy_name,
                rank=0,  # Will be set after sorting
                score=score,
                success_rate=success_rate,
                usage_frequency=usage_frequency,
                confidence_score=avg_confidence,
                recent_performance=recent_performance,
                metadata={
                    "usage_count": stats["usage_count"],
                    "last_used": stats["last_used"],
                    "first_used": stats["first_used"],
                },
            )

            rankings.append(ranking)

        # Sort by score (highest first) and assign ranks
        rankings.sort(key=lambda x: x.score, reverse=True)
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings

    def get_top_strategies(
        self, n: int = 5, prompt: str = None
    ) -> List[StrategyRanking]:
        """
        Get top N strategies.

        Args:
            n: Number of top strategies to return
            prompt: Optional prompt for context

        Returns:
            List of top N strategy rankings
        """
        rankings = self.rank_strategies(prompt)
        return rankings[:n]

    def get_strategy_recommendations(self, prompt: str) -> List[str]:
        """
        Get strategy recommendations for a given prompt.

        Args:
            prompt: User prompt

        Returns:
            List of recommended strategy names
        """
        # Get top strategies
        top_strategies = self.get_top_strategies(n=3, prompt=prompt)

        # Filter by prompt relevance if possible
        relevant_strategies = self._filter_by_prompt_relevance(top_strategies, prompt)

        return [s.strategy_name for s in relevant_strategies]

    def _calculate_usage_frequency(self, strategy_name: str) -> float:
        """Calculate usage frequency for a strategy."""
        if strategy_name not in self.strategy_history["strategies"]:
            return 0.0

        self.strategy_history["strategies"][strategy_name]

        # Calculate frequency based on recent usage
        cutoff_date = datetime.now() - timedelta(days=self.history_window)
        recent_usage = 0

        for prompt_record in self.strategy_history["prompts"]:
            if prompt_record["strategy"] == strategy_name:
                prompt_date = datetime.fromisoformat(prompt_record["timestamp"])
                if prompt_date >= cutoff_date:
                    recent_usage += 1

        # Normalize by time window
        frequency = recent_usage / self.history_window

        return min(1.0, frequency)  # Cap at 1.0

    def _calculate_recent_performance(self, strategy_name: str) -> float:
        """Calculate recent performance for a strategy."""
        if strategy_name not in self.strategy_history["strategies"]:
            return 0.0

        self.strategy_history["strategies"][strategy_name]

        # Calculate performance based on recent history
        cutoff_date = datetime.now() - timedelta(days=self.history_window)
        recent_success = 0
        recent_total = 0

        for prompt_record in self.strategy_history["prompts"]:
            if prompt_record["strategy"] == strategy_name:
                prompt_date = datetime.fromisoformat(prompt_record["timestamp"])
                if prompt_date >= cutoff_date:
                    recent_total += 1
                    if prompt_record["success"]:
                        recent_success += 1

        if recent_total == 0:
            return 0.0

        return recent_success / recent_total

    def _filter_by_prompt_relevance(
        self, strategies: List[StrategyRanking], prompt: str
    ) -> List[StrategyRanking]:
        """
        Filter strategies by prompt relevance.

        Args:
            strategies: List of strategies to filter
            prompt: User prompt

        Returns:
            Filtered list of strategies
        """
        if not prompt:
            return strategies

        # Simple keyword matching
        prompt_words = set(prompt.lower().split())
        relevant_strategies = []

        for strategy in strategies:
            strategy_keywords = self._get_strategy_keywords(strategy.strategy_name)
            strategy_words = set(strategy_keywords)

            # Calculate relevance score
            if prompt_words and strategy_words:
                overlap = len(prompt_words.intersection(strategy_words))
                relevance_score = overlap / len(prompt_words)
                if relevance_score > 0.1:  # At least 10% overlap
                    relevant_strategies.append(strategy)

        return relevant_strategies if relevant_strategies else strategies

    def _get_strategy_keywords(self, strategy_name: str) -> List[str]:
        """Get keywords associated with a strategy."""
        # Simple keyword mapping
        keyword_map = {
            "RSI": [
                "rsi",
                "relative",
                "strength",
                "index",
                "momentum",
                "overbought",
                "oversold",
            ],
            "SMA": ["sma", "simple", "moving", "average", "trend", "crossover"],
            "MACD": ["macd", "moving", "average", "convergence", "divergence", "trend"],
            "Bollinger": ["bollinger", "bands", "volatility", "mean", "reversion"],
            "Momentum": ["momentum", "trend", "acceleration", "velocity"],
            "MeanReversion": ["mean", "reversion", "bounce", "correction", "reversal"],
        }

        return keyword_map.get(strategy_name, [strategy_name.lower()])

    def _cleanup_old_history(self):
        """Remove old history entries."""
        cutoff_date = datetime.now() - timedelta(days=self.history_window)

        # Clean up prompts
        self.strategy_history["prompts"] = [
            prompt
            for prompt in self.strategy_history["prompts"]
            if datetime.fromisoformat(prompt["timestamp"]) >= cutoff_date
        ]

        # Clean up performance history
        for strategy_name in self.strategy_history["strategies"]:
            strategy_stats = self.strategy_history["strategies"][strategy_name]
            strategy_stats["performance_history"] = [
                perf
                for perf in strategy_stats["performance_history"]
                if datetime.fromisoformat(perf["timestamp"]) >= cutoff_date
            ]

    def get_ranking_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about strategy rankings.

        Returns:
            Dictionary with ranking statistics
        """
        total_strategies = len(self.strategy_history["strategies"])
        total_prompts = len(self.strategy_history["prompts"])

        # Calculate overall success rate
        total_success = sum(
            prompt["success"] for prompt in self.strategy_history["prompts"]
        )
        overall_success_rate = (
            total_success / total_prompts if total_prompts > 0 else 0.0
        )

        # Get most used strategies
        strategy_usage = {}
        for strategy_name, stats in self.strategy_history["strategies"].items():
            strategy_usage[strategy_name] = stats["usage_count"]

        most_used = sorted(strategy_usage.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_strategies": total_strategies,
            "total_prompts": total_prompts,
            "overall_success_rate": overall_success_rate,
            "most_used_strategies": most_used,
            "history_window_days": self.history_window,
            "last_updated": self.strategy_history.get("last_updated", ""),
        }

    def reset_history(self):
        """Reset all strategy history."""
        self.strategy_history = {
            "strategies": {},
            "prompts": [],
            "last_updated": datetime.now().isoformat(),
        }
        self._save_history()
        logger.info("Strategy history reset")
