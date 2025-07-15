"""
Strategy Ranking

Ranks strategies based on past success, prompt frequency, and confidence metrics.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter

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
        self.ranking_weights = self.config.get("ranking_weights", {
            "success_rate": 0.4,
            "usage_frequency": 0.2,
            "confidence_score": 0.2,
            "recent_performance": 0.2
        })
        self.history_window = self.config.get("history_window_days", 30)
        self.min_usage_count = self.config.get("min_usage_count", 3)
        
        # Load strategy history
        self.strategy_history = self._load_history()
        
    def _load_history(self) -> Dict[str, Any]:
        """Load strategy usage history from file."""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"Creating new strategy history file: {self.history_file}")
            return {
                "strategies": {},
                "prompts": [],
                "last_updated": datetime.now().isoformat()
            }
            
    def _save_history(self):
        """Save strategy history to file."""
        self.strategy_history["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.strategy_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save strategy history: {e}")
            
    def record_strategy_usage(
        self, 
        strategy_name: str, 
        prompt: str, 
        success: bool, 
        confidence: float,
        performance_metrics: Optional[Dict[str, Any]] = None
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
        self.strategy_history["prompts"].append({
            "timestamp": timestamp,
            "prompt": prompt,
            "strategy": strategy_name,
            "success": success,
            "confidence": confidence
        })
        
        # Update strategy statistics
        if strategy_name not in self.strategy_history["strategies"]:
            self.strategy_history["strategies"][strategy_name] = {
                "usage_count": 0,
                "success_count": 0,
                "total_confidence": 0.0,
                "performance_history": [],
                "last_used": timestamp,
                "first_used": timestamp
            }
            
        strategy_stats = self.strategy_history["strategies"][strategy_name]
        strategy_stats["usage_count"] += 1
        strategy_stats["total_confidence"] += confidence
        strategy_stats["last_used"] = timestamp
        
        if success:
            strategy_stats["success_count"] += 1
            
        if performance_metrics:
            strategy_stats["performance_history"].append({
                "timestamp": timestamp,
                "metrics": performance_metrics
            })
            
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
                
            # Calculate metrics
            success_rate = stats["success_count"] / stats["usage_count"]
            avg_confidence = stats["total_confidence"] / stats["usage_count"]
            usage_frequency = self._calculate_usage_frequency(strategy_name)
            recent_performance = self._calculate_recent_performance(strategy_name)
            
            # Calculate overall score
            score = (
                self.ranking_weights["success_rate"] * success_rate +
                self.ranking_weights["usage_frequency"] * usage_frequency +
                self.ranking_weights["confidence_score"] * avg_confidence +
                self.ranking_weights["recent_performance"] * recent_performance
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
                    "first_used": stats["first_used"]
                }
            )
            
            rankings.append(ranking)
            
        # Sort by score (highest first) and assign ranks
        rankings.sort(key=lambda x: x.score, reverse=True)
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1
            
        return rankings
        
    def get_top_strategies(self, n: int = 5, prompt: str = None) -> List[StrategyRanking]:
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
            
        stats = self.strategy_history["strategies"][strategy_name]
        
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
            
        stats = self.strategy_history["strategies"][strategy_name]
        
        if not stats["performance_history"]:
            return 0.0
            
        # Calculate average of recent performance metrics
        cutoff_date = datetime.now() - timedelta(days=self.history_window)
        recent_metrics = []
        
        for perf_record in stats["performance_history"]:
            perf_date = datetime.fromisoformat(perf_record["timestamp"])
            if perf_date >= cutoff_date:
                metrics = perf_record["metrics"]
                # Extract relevant performance metric (e.g., returns, accuracy)
                if "returns" in metrics:
                    recent_metrics.append(metrics["returns"])
                elif "accuracy" in metrics:
                    recent_metrics.append(metrics["accuracy"])
                elif "sharpe_ratio" in metrics:
                    recent_metrics.append(metrics["sharpe_ratio"])
                    
        if not recent_metrics:
            return 0.0
            
        # Normalize performance to 0-1 range
        avg_performance = sum(recent_metrics) / len(recent_metrics)
        return max(0.0, min(1.0, avg_performance))
        
    def _filter_by_prompt_relevance(self, strategies: List[StrategyRanking], prompt: str) -> List[StrategyRanking]:
        """Filter strategies by prompt relevance."""
        if not prompt:
            return strategies
            
        # Simple keyword-based relevance scoring
        prompt_lower = prompt.lower()
        relevant_strategies = []
        
        for strategy in strategies:
            relevance_score = 0.0
            
            # Check for strategy-specific keywords
            strategy_keywords = self._get_strategy_keywords(strategy.strategy_name)
            for keyword in strategy_keywords:
                if keyword in prompt_lower:
                    relevance_score += 0.3
                    
            # Boost score for relevant strategies
            if relevance_score > 0:
                strategy.score *= (1 + relevance_score)
                relevant_strategies.append(strategy)
            else:
                relevant_strategies.append(strategy)
                
        # Re-sort by updated scores
        relevant_strategies.sort(key=lambda x: x.score, reverse=True)
        
        return relevant_strategies
        
    def _get_strategy_keywords(self, strategy_name: str) -> List[str]:
        """Get keywords associated with a strategy."""
        keyword_mapping = {
            "rsi": ["rsi", "relative strength index", "oversold", "overbought"],
            "macd": ["macd", "moving average convergence divergence", "crossover"],
            "bollinger": ["bollinger", "bands", "volatility", "squeeze"],
            "sma": ["sma", "simple moving average", "moving average"],
            "ema": ["ema", "exponential moving average"],
            "momentum": ["momentum", "velocity", "acceleration"],
            "mean_reversion": ["mean reversion", "revert", "bounce"],
            "trend_following": ["trend", "follow", "direction"]
        }
        
        strategy_lower = strategy_name.lower()
        for key, keywords in keyword_mapping.items():
            if key in strategy_lower:
                return keywords
                
        return [strategy_name.lower()]
        
    def _cleanup_old_history(self):
        """Remove old history entries."""
        cutoff_date = datetime.now() - timedelta(days=self.history_window)
        
        # Clean up prompts
        self.strategy_history["prompts"] = [
            p for p in self.strategy_history["prompts"]
            if datetime.fromisoformat(p["timestamp"]) >= cutoff_date
        ]
        
        # Clean up performance history
        for strategy_name, stats in self.strategy_history["strategies"].items():
            stats["performance_history"] = [
                p for p in stats["performance_history"]
                if datetime.fromisoformat(p["timestamp"]) >= cutoff_date
            ]
            
    def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get statistics about strategy rankings."""
        total_strategies = len(self.strategy_history["strategies"])
        total_prompts = len(self.strategy_history["prompts"])
        
        if total_strategies == 0:
            return {
                "total_strategies": 0,
                "total_prompts": 0,
                "ranking_weights": self.ranking_weights,
                "history_window_days": self.history_window
            }
            
        # Calculate average metrics
        avg_success_rate = sum(
            stats["success_count"] / stats["usage_count"]
            for stats in self.strategy_history["strategies"].values()
        ) / total_strategies
        
        return {
            "total_strategies": total_strategies,
            "total_prompts": total_prompts,
            "avg_success_rate": avg_success_rate,
            "ranking_weights": self.ranking_weights,
            "history_window_days": self.history_window,
            "last_updated": self.strategy_history.get("last_updated")
        }
        
    def reset_history(self):
        """Reset strategy history."""
        self.strategy_history = {
            "strategies": {},
            "prompts": [],
            "last_updated": datetime.now().isoformat()
        }
        self._save_history()
        logger.info("Strategy history reset") 