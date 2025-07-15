"""
Strategy Router

Routes strategy requests to appropriate strategy implementations based on
priority rules and relevance scoring. Includes index alignment and signal
priority/deduplication for trade signals.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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
        self.deduplication_window = self.config.get("deduplication_window", 300)  # 5 minutes
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available strategies with keywords and metadata."""
        return {
            "RSI_Strategy": {
                "type": StrategyType.MEAN_REVERSION,
                "keywords": ["rsi", "relative strength index", "oversold", "overbought", "mean reversion"],
                "priority": 1,
                "confidence": 0.85,
                "description": "RSI-based mean reversion strategy"
            },
            "MACD_Strategy": {
                "type": StrategyType.MOMENTUM,
                "keywords": ["macd", "moving average convergence divergence", "momentum", "crossover"],
                "priority": 1,
                "confidence": 0.80,
                "description": "MACD-based momentum strategy"
            },
            "Bollinger_Bands": {
                "type": StrategyType.MEAN_REVERSION,
                "keywords": ["bollinger", "bands", "volatility", "squeeze", "breakout"],
                "priority": 2,
                "confidence": 0.75,
                "description": "Bollinger Bands mean reversion strategy"
            },
            "Moving_Average_Crossover": {
                "type": StrategyType.TREND_FOLLOWING,
                "keywords": ["moving average", "ma", "sma", "ema", "crossover", "trend"],
                "priority": 2,
                "confidence": 0.70,
                "description": "Moving average crossover strategy"
            },
            "Breakout_Strategy": {
                "type": StrategyType.BREAKOUT,
                "keywords": ["breakout", "resistance", "support", "break", "level"],
                "priority": 3,
                "confidence": 0.65,
                "description": "Breakout detection strategy"
            },
            "Momentum_Strategy": {
                "type": StrategyType.MOMENTUM,
                "keywords": ["momentum", "velocity", "acceleration", "speed"],
                "priority": 3,
                "confidence": 0.60,
                "description": "General momentum strategy"
            }
        }
        
    def _initialize_priority_rules(self) -> Dict[str, int]:
        """Initialize priority rules for strategy selection."""
        return {
            "exact_match": 10,
            "keyword_match": 5,
            "type_match": 3,
            "partial_match": 2,
            "fallback": 1
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
                keywords_matched.append(strategy_name)
                
            # Check for partial matches
            for keyword in strategy_info["keywords"]:
                if any(word in prompt_lower for word in keyword.split()):
                    relevance_score += self.priority_rules["partial_match"]
                    
            # Add base priority
            relevance_score += strategy_info["priority"]
            
            if relevance_score > 0:
                matches.append(StrategyMatch(
                    strategy_name=strategy_name,
                    strategy_type=strategy_info["type"],
                    relevance_score=relevance_score,
                    priority=strategy_info["priority"],
                    keywords_matched=keywords_matched,
                    confidence=strategy_info["confidence"]
                ))
                
        # Sort by relevance score (highest first)
        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return matches
        
    def select_best_strategy(self, prompt: str) -> Optional[StrategyMatch]:
        """
        Select the best strategy for a given prompt.
        
        Args:
            prompt: User prompt or strategy request
            
        Returns:
            Best strategy match or None if no matches found
        """
        matches = self.find_strategy_matches(prompt)
        
        if not matches:
            logger.warning(f"No strategy matches found for prompt: {prompt}. Falling back to base strategy.")
            # Fallback to base strategy (RSI, SMA, etc)
            base_strategies = [s for s in self.strategies.keys() if s in ("RSI_Strategy", "Moving_Average_Crossover", "SMA_Strategy")]
            if base_strategies:
                base_name = base_strategies[0]
                base_info = self.strategies[base_name]
                logger.info(f"Fallback to base strategy: {base_name}")
                return StrategyMatch(
                    strategy_name=base_name,
                    strategy_type=base_info["type"],
                    relevance_score=1,
                    priority=base_info["priority"],
                    keywords_matched=[],
                    confidence=base_info["confidence"]
                )
            else:
                logger.error("No base strategies available for fallback.")
                return None
            
        # If multiple matches, use priority rules
        if len(matches) > 1:
            best_match = self._apply_priority_rules(matches, prompt)
        else:
            best_match = matches[0]
            
        logger.info(f"Selected strategy: {best_match.strategy_name} (score: {best_match.relevance_score})")
        return best_match
        
    def _apply_priority_rules(self, matches: List[StrategyMatch], prompt: str) -> StrategyMatch:
        """
        Apply priority rules when multiple strategy matches are found.
        
        Args:
            matches: List of strategy matches
            prompt: Original prompt for context
            
        Returns:
            Best strategy match based on priority rules
        """
        # If top matches have same score, use additional criteria
        top_score = matches[0].relevance_score
        top_matches = [m for m in matches if m.relevance_score == top_score]
        
        if len(top_matches) == 1:
            return top_matches[0]
            
        # Use confidence as tiebreaker
        best_match = max(top_matches, key=lambda x: x.confidence)
        
        # If still tied, use priority
        tied_matches = [m for m in top_matches if m.confidence == best_match.confidence]
        if len(tied_matches) > 1:
            best_match = min(tied_matches, key=lambda x: x.priority)
            
        return best_match
        
    def get_strategy_suggestions(self, prompt: str, max_suggestions: int = 3) -> List[str]:
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
            prompt: Strategy request prompt
            
        Returns:
            Validation result dictionary
        """
        matches = self.find_strategy_matches(prompt)
        
        validation_result = {
            "valid": len(matches) > 0,
            "matches_found": len(matches),
            "suggestions": self.get_strategy_suggestions(prompt),
            "best_match": matches[0].strategy_name if matches else None,
            "confidence": matches[0].confidence if matches else 0.0
        }
        
        return validation_result
        
    def align_data_indices(self, dataframes: List[pd.DataFrame], method: str = "inner") -> List[pd.DataFrame]:
        """
        Align multiple dataframes to have the same index.
        
        Args:
            dataframes: List of dataframes to align
            method: Alignment method ('inner', 'outer', 'left', 'right')
            
        Returns:
            List of aligned dataframes
        """
        if not dataframes or len(dataframes) < 2:
            return dataframes
            
        try:
            # Get all indices
            all_indices = [df.index for df in dataframes]
            
            # Find common index
            if method == "inner":
                common_index = all_indices[0]
                for idx in all_indices[1:]:
                    common_index = common_index.intersection(idx)
            elif method == "outer":
                common_index = all_indices[0]
                for idx in all_indices[1:]:
                    common_index = common_index.union(idx)
            else:
                # Use first dataframe's index as reference
                common_index = all_indices[0]
                
            # Align all dataframes
            aligned_dfs = []
            for df in dataframes:
                aligned_df = df.reindex(common_index, method=method)
                aligned_dfs.append(aligned_df)
                
            logger.info(f"Aligned {len(dataframes)} dataframes with {len(common_index)} common indices")
            return aligned_dfs
            
        except Exception as e:
            logger.error(f"Error aligning data indices: {e}")
            return dataframes
            
    def create_trade_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        strategy_name: str,
        priority: SignalPriority = SignalPriority.MEDIUM,
        strength: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        expiration_minutes: Optional[int] = None
    ) -> TradeSignal:
        """
        Create a new trade signal with deduplication.
        
        Args:
            symbol: Trading symbol
            signal_type: Signal type ('buy', 'sell', 'hold')
            confidence: Signal confidence (0-1)
            strategy_name: Name of the strategy
            priority: Signal priority
            strength: Signal strength (0-1)
            metadata: Additional metadata
            expiration_minutes: Signal expiration in minutes
            
        Returns:
            Trade signal object
        """
        timestamp = datetime.now()
        signal_id = f"{symbol}_{signal_type}_{strategy_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Check for duplicates
        if self._is_duplicate_signal(symbol, signal_type, strategy_name):
            logger.info(f"Duplicate signal detected for {symbol} {signal_type} from {strategy_name}")
            return None
            
        # Create signal
        signal = TradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            timestamp=timestamp,
            strategy_name=strategy_name,
            priority=priority,
            signal_id=signal_id,
            metadata=metadata or {},
            strength=strength,
            expiration=timestamp + timedelta(minutes=expiration_minutes) if expiration_minutes else None
        )
        
        # Cache signal
        self._cache_signal(signal)
        
        logger.info(f"Created trade signal: {signal_id} (priority: {priority.name})")
        return signal
        
    def _is_duplicate_signal(self, symbol: str, signal_type: str, strategy_name: str) -> bool:
        """
        Check if a signal is a duplicate within the deduplication window.
        
        Args:
            symbol: Trading symbol
            signal_type: Signal type
            strategy_name: Strategy name
            
        Returns:
            True if duplicate found
        """
        cutoff_time = datetime.now() - timedelta(seconds=self.deduplication_window)
        
        # Check cache
        cache_key = f"{symbol}_{signal_type}_{strategy_name}"
        if cache_key in self.signal_cache:
            cached_signal = self.signal_cache[cache_key]
            if cached_signal.timestamp > cutoff_time:
                return True
                
        # Check history
        for signal in self.signal_history:
            if (signal.symbol == symbol and 
                signal.signal_type == signal_type and 
                signal.strategy_name == strategy_name and
                signal.timestamp > cutoff_time):
                return True
                
        return False
        
    def _cache_signal(self, signal: TradeSignal):
        """Cache a signal for deduplication."""
        cache_key = f"{signal.symbol}_{signal.signal_type}_{signal.strategy_name}"
        self.signal_cache[cache_key] = signal
        self.signal_history.append(signal)
        
        # Clean up old signals
        self._cleanup_old_signals()
        
    def _cleanup_old_signals(self):
        """Clean up old signals from cache and history."""
        cutoff_time = datetime.now() - timedelta(seconds=self.deduplication_window)
        
        # Clean cache
        expired_keys = []
        for key, signal in self.signal_cache.items():
            if signal.timestamp < cutoff_time:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.signal_cache[key]
            
        # Clean history
        self.signal_history = [s for s in self.signal_history if s.timestamp > cutoff_time]
        
        # Limit cache size
        if len(self.signal_cache) > self.max_cache_size:
            # Remove oldest signals
            sorted_signals = sorted(self.signal_cache.items(), key=lambda x: x[1].timestamp)
            for key, _ in sorted_signals[:-self.max_cache_size]:
                del self.signal_cache[key]
                
    def get_active_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        strategy_name: Optional[str] = None,
        min_confidence: float = 0.0,
        min_priority: SignalPriority = SignalPriority.INFO
    ) -> List[TradeSignal]:
        """
        Get active signals with filtering options.
        
        Args:
            symbol: Filter by symbol
            signal_type: Filter by signal type
            strategy_name: Filter by strategy name
            min_confidence: Minimum confidence threshold
            min_priority: Minimum priority level
            
        Returns:
            List of active signals
        """
        current_time = datetime.now()
        active_signals = []
        
        for signal in self.signal_history:
            # Check expiration
            if signal.expiration and signal.expiration < current_time:
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
            
        # Sort by priority and timestamp
        active_signals.sort(key=lambda x: (x.priority.value, x.timestamp))
        
        return active_signals
        
    def get_signal_summary(self) -> Dict[str, Any]:
        """
        Get summary of all signals.
        
        Returns:
            Signal summary dictionary
        """
        current_time = datetime.now()
        
        # Count by type
        signal_counts = {}
        strategy_counts = {}
        priority_counts = {}
        
        for signal in self.signal_history:
            # Signal type counts
            signal_counts[signal.signal_type] = signal_counts.get(signal.signal_type, 0) + 1
            
            # Strategy counts
            strategy_counts[signal.strategy_name] = strategy_counts.get(signal.strategy_name, 0) + 1
            
            # Priority counts
            priority_counts[signal.priority.name] = priority_counts.get(signal.priority.name, 0) + 1
            
        # Active signals
        active_signals = [s for s in self.signal_history if not s.expiration or s.expiration > current_time]
        
        summary = {
            "total_signals": len(self.signal_history),
            "active_signals": len(active_signals),
            "cached_signals": len(self.signal_cache),
            "signal_counts": signal_counts,
            "strategy_counts": strategy_counts,
            "priority_counts": priority_counts,
            "deduplication_window_seconds": self.deduplication_window,
            "max_cache_size": self.max_cache_size,
        }
        
        return summary 