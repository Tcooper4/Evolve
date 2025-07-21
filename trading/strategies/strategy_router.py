"""
Strategy Router

This module provides intelligent strategy routing and selection based on
user prompts and market conditions with signal management capabilities.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    OPTIONS = "options"
    CRYPTO = "crypto"
    FOREX = "forex"
    GENERAL = "general"


class SignalPriority(Enum):
    """Signal priority levels."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


@dataclass
class StrategyMatch:
    """Strategy match with relevance score."""

    strategy_name: str
    strategy_type: StrategyType
    relevance_score: float
    priority: int
    keywords_matched: List[str]
    confidence: float


@dataclass
class TradeSignal:
    """Trade signal with priority and deduplication info."""

    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    timestamp: datetime
    strategy_name: str
    priority: SignalPriority
    signal_id: str
    metadata: Dict[str, Any]
    strength: float  # Signal strength (0-1)
    expiration: Optional[datetime] = None


class StrategyRouter:
    """Router for strategy selection and matching with signal management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy router.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.strategies = self._initialize_strategies()
        self.priority_rules = self._initialize_priority_rules()
        self.signal_cache: Dict[str, TradeSignal] = {}
        self.signal_history: List[TradeSignal] = []
        self.deduplication_window = self.config.get(
            "deduplication_window", 300
        )  # 5 minutes
        self.max_cache_size = self.config.get("max_cache_size", 1000)

    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available strategies with keywords and metadata."""
        return {
            "RSI_Strategy": {
                "type": StrategyType.MEAN_REVERSION,
                "keywords": [
                    "rsi",
                    "relative strength index",
                    "oversold",
                    "overbought",
                    "mean reversion",
                ],
                "priority": 1,
                "confidence": 0.85,
                "description": "RSI-based mean reversion strategy",
            },
            "MACD_Strategy": {
                "type": StrategyType.MOMENTUM,
                "keywords": [
                    "macd",
                    "moving average convergence divergence",
                    "momentum",
                    "crossover",
                ],
                "priority": 1,
                "confidence": 0.80,
                "description": "MACD-based momentum strategy",
            },
            "Bollinger_Bands": {
                "type": StrategyType.MEAN_REVERSION,
                "keywords": ["bollinger", "bands", "volatility", "squeeze", "breakout"],
                "priority": 2,
                "confidence": 0.75,
                "description": "Bollinger Bands mean reversion strategy",
            },
            "Moving_Average_Crossover": {
                "type": StrategyType.TREND_FOLLOWING,
                "keywords": [
                    "moving average",
                    "ma",
                    "sma",
                    "ema",
                    "crossover",
                    "trend",
                ],
                "priority": 2,
                "confidence": 0.70,
                "description": "Moving average crossover strategy",
            },
            "Breakout_Strategy": {
                "type": StrategyType.BREAKOUT,
                "keywords": ["breakout", "resistance", "support", "break", "level"],
                "priority": 3,
                "confidence": 0.65,
                "description": "Breakout detection strategy",
            },
            "Momentum_Strategy": {
                "type": StrategyType.MOMENTUM,
                "keywords": ["momentum", "velocity", "acceleration", "speed"],
                "priority": 3,
                "confidence": 0.60,
                "description": "General momentum strategy",
            },
        }

    def _initialize_priority_rules(self) -> Dict[str, int]:
        """Initialize priority rules for strategy selection."""
        return {
            "exact_match": 10,
            "keyword_match": 5,
            "type_match": 3,
            "partial_match": 2,
            "fallback": 1,
        }

    def find_strategy_matches(self, prompt: str) -> List[StrategyMatch]:
        """
        Find strategy matches for a given prompt.

        Args:
            prompt: User prompt or strategy request

        Returns:
            List of strategy matches with relevance scores
        """
        prompt_lower = prompt.lower()
        matches = []

        for strategy_name, strategy_info in self.strategies.items():
            relevance_score = 0
            keywords_matched = []

            # Check keyword matches
            for keyword in strategy_info["keywords"]:
                if keyword.lower() in prompt_lower:
                    relevance_score += self.priority_rules["keyword_match"]
                    keywords_matched.append(keyword)

            # Check for exact strategy name match
            if strategy_name.lower().replace("_", " ") in prompt_lower:
                relevance_score += self.priority_rules["exact_match"]

            # Check for strategy type mentions
            strategy_type = strategy_info["type"].value
            if strategy_type in prompt_lower:
                relevance_score += self.priority_rules["type_match"]

            # Only include matches with some relevance
            if relevance_score > 0:
                match = StrategyMatch(
                    strategy_name=strategy_name,
                    strategy_type=strategy_info["type"],
                    relevance_score=relevance_score,
                    priority=strategy_info["priority"],
                    keywords_matched=keywords_matched,
                    confidence=strategy_info["confidence"],
                )
                matches.append(match)

        # Sort by relevance score (highest first)
        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        return matches

    def select_best_strategy(self, prompt: str) -> Optional[StrategyMatch]:
        """
        Select the best strategy for a given prompt.

        Args:
            prompt: User prompt or strategy request

        Returns:
            Best strategy match or None if no suitable match
        """
        matches = self.find_strategy_matches(prompt)

        if not matches:
            logger.warning(f"No strategy matches found for prompt: {prompt}")
            return None

        # Apply priority rules to select best match
        best_match = self._apply_priority_rules(matches, prompt)

        logger.info(
            f"Selected strategy '{best_match.strategy_name}' for prompt: {prompt}"
        )
        return best_match

    def _apply_priority_rules(
        self, matches: List[StrategyMatch], prompt: str
    ) -> StrategyMatch:
        """Apply priority rules to select the best strategy match."""
        if len(matches) == 1:
            return matches[0]

        # Sort by priority first, then by relevance score
        sorted_matches = sorted(matches, key=lambda x: (x.priority, -x.relevance_score))

        # Return the highest priority match with highest relevance
        return sorted_matches[0]

    def get_strategy_suggestions(
        self, prompt: str, max_suggestions: int = 3
    ) -> List[str]:
        """
        Get strategy suggestions for a given prompt.

        Args:
            prompt: User prompt
            max_suggestions: Maximum number of suggestions

        Returns:
            List of strategy suggestions
        """
        matches = self.find_strategy_matches(prompt)
        suggestions = []

        for match in matches[:max_suggestions]:
            suggestion = f"{match.strategy_name} (confidence: {match.confidence:.2f})"
            suggestions.append(suggestion)

        return suggestions

    def validate_strategy_request(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a strategy request.

        Args:
            prompt: User prompt

        Returns:
            Validation result dictionary
        """
        matches = self.find_strategy_matches(prompt)

        return {
            "valid": len(matches) > 0,
            "matches_count": len(matches),
            "suggestions": self.get_strategy_suggestions(prompt),
            "best_match": matches[0] if matches else None,
        }

    def align_data_indices(
        self, dataframes: List[pd.DataFrame], method: str = "inner"
    ) -> List[pd.DataFrame]:
        """
        Align multiple dataframes to have the same index.

        Args:
            dataframes: List of dataframes to align
            method: Alignment method ('inner', 'outer', 'left', 'right')

        Returns:
            List of aligned dataframes
        """
        if len(dataframes) <= 1:
            return dataframes

        # Get common index
        indices = [df.index for df in dataframes]
        common_index = indices[0]

        for idx in indices[1:]:
            if method == "inner":
                common_index = common_index.intersection(idx)
            elif method == "outer":
                common_index = common_index.union(idx)
            elif method == "left":
                common_index = common_index.intersection(idx)
            elif method == "right":
                common_index = idx.intersection(common_index)

        # Align dataframes
        aligned_dataframes = []
        for df in dataframes:
            aligned_df = df.loc[common_index]
            aligned_dataframes.append(aligned_df)

        return aligned_dataframes

    def create_trade_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        strategy_name: str,
        priority: SignalPriority = SignalPriority.MEDIUM,
        strength: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        expiration_minutes: Optional[int] = None,
    ) -> TradeSignal:
        """
        Create a new trade signal.

        Args:
            symbol: Trading symbol
            signal_type: Type of signal ('buy', 'sell', 'hold')
            confidence: Signal confidence (0-1)
            strategy_name: Name of the strategy
            priority: Signal priority
            strength: Signal strength (0-1)
            metadata: Additional metadata
            expiration_minutes: Signal expiration in minutes

        Returns:
            TradeSignal object
        """
        # Check for duplicate signals
        if self._is_duplicate_signal(symbol, signal_type, strategy_name):
            logger.warning(
                f"Duplicate signal detected: {symbol} {signal_type} {strategy_name}"
            )
            return None

        # Generate unique signal ID
        signal_id = str(uuid.uuid4())

        # Calculate expiration time
        expiration = None
        if expiration_minutes:
            expiration = datetime.now() + timedelta(minutes=expiration_minutes)

        # Create signal
        signal = TradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            timestamp=datetime.now(),
            strategy_name=strategy_name,
            priority=priority,
            signal_id=signal_id,
            metadata=metadata or {},
            strength=strength,
            expiration=expiration,
        )

        # Cache signal
        self._cache_signal(signal)

        logger.info(
            f"Created signal: {signal_id} - {symbol} {signal_type} (confidence: {confidence:.2f})"
        )
        return signal

    def _is_duplicate_signal(
        self, symbol: str, signal_type: str, strategy_name: str
    ) -> bool:
        """Check if a signal is a duplicate within the deduplication window."""
        cutoff_time = datetime.now() - timedelta(seconds=self.deduplication_window)

        for signal in self.signal_history:
            if (
                signal.symbol == symbol
                and signal.signal_type == signal_type
                and signal.strategy_name == strategy_name
                and signal.timestamp > cutoff_time
            ):
                return True

        return False

    def _cache_signal(self, signal: TradeSignal):
        """Cache a signal and maintain cache size limits."""
        self.signal_cache[signal.signal_id] = signal
        self.signal_history.append(signal)

        # Cleanup old signals
        self._cleanup_old_signals()

        # Maintain cache size
        if len(self.signal_cache) > self.max_cache_size:
            # Remove oldest signals
            oldest_signals = sorted(
                self.signal_cache.values(), key=lambda x: x.timestamp
            )
            for old_signal in oldest_signals[: len(oldest_signals) // 2]:
                del self.signal_cache[old_signal.signal_id]

    def _cleanup_old_signals(self):
        """Remove expired signals from cache."""
        current_time = datetime.now()
        expired_signals = []

        for signal_id, signal in self.signal_cache.items():
            if signal.expiration and signal.expiration < current_time:
                expired_signals.append(signal_id)

        for signal_id in expired_signals:
            del self.signal_cache[signal_id]

        if expired_signals:
            logger.info(f"Cleaned up {len(expired_signals)} expired signals")

    def get_active_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        strategy_name: Optional[str] = None,
        min_confidence: float = 0.0,
        min_priority: SignalPriority = SignalPriority.INFO,
    ) -> List[TradeSignal]:
        """
        Get active signals with optional filtering.

        Args:
            symbol: Filter by symbol
            signal_type: Filter by signal type
            strategy_name: Filter by strategy name
            min_confidence: Minimum confidence threshold
            min_priority: Minimum priority threshold

        Returns:
            List of filtered signals
        """
        active_signals = []

        for signal in self.signal_cache.values():
            # Check if signal is expired
            if signal.expiration and signal.expiration < datetime.now():
                continue

            # Apply filters
            if symbol and signal.symbol != symbol:
                continue
            if signal_type and signal.signal_type != signal_type:
                continue
            if strategy_name and signal.strategy_name != strategy_name:
                continue
            if signal.confidence < min_confidence:
                continue
            if signal.priority.value > min_priority.value:
                continue

            active_signals.append(signal)

        # Sort by timestamp (newest first)
        active_signals.sort(key=lambda x: x.timestamp, reverse=True)
        return active_signals

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of all signals."""
        current_time = datetime.now()
        active_signals = [
            s
            for s in self.signal_cache.values()
            if not s.expiration or s.expiration > current_time
        ]

        summary = {
            "total_signals": len(self.signal_history),
            "active_signals": len(active_signals),
            "expired_signals": len(self.signal_history) - len(active_signals),
            "signals_by_type": {},
            "signals_by_strategy": {},
            "signals_by_priority": {},
            "average_confidence": 0.0,
            "cache_size": len(self.signal_cache),
        }

        if active_signals:
            # Calculate statistics
            summary["average_confidence"] = sum(
                s.confidence for s in active_signals
            ) / len(active_signals)

            for signal in active_signals:
                # Count by signal type
                signal_type = signal.signal_type
                summary["signals_by_type"][signal_type] = (
                    summary["signals_by_type"].get(signal_type, 0) + 1
                )

                # Count by strategy
                strategy = signal.strategy_name
                summary["signals_by_strategy"][strategy] = (
                    summary["signals_by_strategy"].get(strategy, 0) + 1
                )

                # Count by priority
                priority = signal.priority.name
                summary["signals_by_priority"][priority] = (
                    summary["signals_by_priority"].get(priority, 0) + 1
                )

        return summary
