"""
Composite Strategy with Conflict Resolution

This module implements a composite strategy that combines multiple individual strategies
with intelligent conflict resolution for simultaneous signals. It uses majority voting
across available strategies and allows override via a strategy_priority list.

Features:
- Majority voting across available strategies
- Priority-based override system
- Conflict detection and resolution
- Signal strength aggregation
- Performance tracking per strategy
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for conflict resolution."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class StrategySignal:
    """Individual strategy signal with metadata."""
    
    strategy_name: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    price: float
    volume: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """Conflict resolution result."""
    
    final_signal: SignalType
    confidence: float
    participating_strategies: List[str]
    conflict_detected: bool
    resolution_method: str
    override_applied: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompositeStrategy:
    """Composite strategy with conflict resolution capabilities."""
    
    def __init__(
        self,
        strategy_priority: Optional[List[str]] = None,
        min_confidence_threshold: float = 0.6,
        majority_threshold: float = 0.5,
        enable_override: bool = True,
        log_conflicts: bool = True
    ):
        """Initialize the composite strategy.
        
        Args:
            strategy_priority: Ordered list of strategy names for priority override
            min_confidence_threshold: Minimum confidence for signal acceptance
            majority_threshold: Threshold for majority voting (0.5 = 50%)
            enable_override: Whether to allow priority-based overrides
            log_conflicts: Whether to log conflict resolution details
        """
        self.strategy_priority = strategy_priority or []
        self.min_confidence_threshold = min_confidence_threshold
        self.majority_threshold = majority_threshold
        self.enable_override = enable_override
        self.log_conflicts = log_conflicts
        
        self.strategy_signals: Dict[str, StrategySignal] = {}
        self.conflict_history: List[Dict[str, Any]] = []
        self.performance_tracker: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"CompositeStrategy initialized with {len(self.strategy_priority)} priority strategies")
    
    def add_strategy_signal(
        self,
        strategy_name: str,
        signal_type: Union[str, SignalType],
        confidence: float,
        price: float,
        volume: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a signal from an individual strategy.
        
        Args:
            strategy_name: Name of the strategy
            signal_type: Type of signal (buy, sell, hold, etc.)
            confidence: Confidence level (0.0 to 1.0)
            price: Current price
            volume: Trading volume (optional)
            metadata: Additional signal metadata
            
        Returns:
            True if signal added successfully
            
        Raises:
            ValueError: If signal_type is invalid
        """
        try:
            # Convert string to enum if needed
            if isinstance(signal_type, str):
                signal_type = SignalType(signal_type.lower())
            
            # Validate confidence
            if not 0.0 <= confidence <= 1.0:
                logger.warning(f"Invalid confidence {confidence} for {strategy_name}, clamping to [0, 1]")
                confidence = max(0.0, min(1.0, confidence))
            
            signal = StrategySignal(
                strategy_name=strategy_name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                metadata=metadata or {}
            )
            
            self.strategy_signals[strategy_name] = signal
            logger.debug(f"Added signal from {strategy_name}: {signal_type.value} (confidence: {confidence:.2f})")
            return True
            
        except ValueError as e:
            # Re-raise ValueError for invalid signal types
            logger.error(f"Invalid signal type '{signal_type}' for {strategy_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error adding signal from {strategy_name}: {e}")
            return False
    
    def resolve_conflicts(self) -> ConflictResolution:
        """Resolve conflicts between simultaneous signals.
        
        Returns:
            ConflictResolution object with final decision
        """
        if not self.strategy_signals:
            logger.warning("No strategy signals to resolve")
            return ConflictResolution(
                final_signal=SignalType.HOLD,
                confidence=0.0,
                participating_strategies=[],
                conflict_detected=False,
                resolution_method="no_signals",
                override_applied=False
            )
        
        # Analyze signal distribution
        signal_counts = self._count_signals()
        signal_confidences = self._aggregate_confidences()
        
        # Detect conflicts
        conflict_detected = self._detect_conflicts(signal_counts)
        
        # Apply resolution method
        if conflict_detected and self.enable_override and self.strategy_priority:
            resolution = self._apply_priority_override(signal_counts, signal_confidences)
        else:
            resolution = self._apply_majority_voting(signal_counts, signal_confidences)
        
        # Log conflict if enabled
        if self.log_conflicts and conflict_detected:
            self._log_conflict_resolution(resolution, signal_counts, signal_confidences)
        
        return resolution
    
    def _count_signals(self) -> Dict[SignalType, int]:
        """Count signals by type."""
        counts = {signal_type: 0 for signal_type in SignalType}
        
        for signal in self.strategy_signals.values():
            counts[signal.signal_type] += 1
        
        return counts
    
    def _aggregate_confidences(self) -> Dict[SignalType, List[float]]:
        """Aggregate confidences by signal type."""
        confidences = {signal_type: [] for signal_type in SignalType}
        
        for signal in self.strategy_signals.values():
            confidences[signal.signal_type].append(signal.confidence)
        
        return confidences
    
    def _detect_conflicts(self, signal_counts: Dict[SignalType, int]) -> bool:
        """Detect if there are conflicting signals."""
        # Check for opposing signals
        buy_signals = signal_counts[SignalType.BUY] + signal_counts[SignalType.STRONG_BUY]
        sell_signals = signal_counts[SignalType.SELL] + signal_counts[SignalType.STRONG_SELL]
        
        # Conflict if both buy and sell signals exist
        if buy_signals > 0 and sell_signals > 0:
            return True
        
        # Check for mixed signal strengths
        strong_buy = signal_counts[SignalType.STRONG_BUY]
        strong_sell = signal_counts[SignalType.STRONG_SELL]
        weak_buy = signal_counts[SignalType.BUY]
        weak_sell = signal_counts[SignalType.SELL]
        
        # Conflict if strong and weak signals of same type exist
        if (strong_buy > 0 and weak_buy > 0) or (strong_sell > 0 and weak_sell > 0):
            return True
        
        return False
    
    def _apply_majority_voting(
        self,
        signal_counts: Dict[SignalType, int],
        signal_confidences: Dict[SignalType, List[float]]
    ) -> ConflictResolution:
        """Apply majority voting to resolve conflicts."""
        total_signals = sum(signal_counts.values())
        if total_signals == 0:
            return ConflictResolution(
                final_signal=SignalType.HOLD,
                confidence=0.0,
                participating_strategies=[],
                conflict_detected=False,
                resolution_method="majority_voting",
                override_applied=False
            )
        
        # Find signal type with highest count
        max_count = max(signal_counts.values())
        winning_signals = [
            signal_type for signal_type, count in signal_counts.items()
            if count == max_count and count > 0
        ]
        
        if len(winning_signals) == 1:
            final_signal = winning_signals[0]
        else:
            # Tie - use confidence to break
            avg_confidences = {}
            for signal_type in winning_signals:
                confs = signal_confidences[signal_type]
                avg_confidences[signal_type] = np.mean(confs) if confs else 0.0
            
            final_signal = max(avg_confidences.items(), key=lambda x: x[1])[0]
        
        # Calculate aggregate confidence
        confs = signal_confidences[final_signal]
        confidence = np.mean(confs) if confs else 0.0
        
        # Get participating strategies
        participating = [
            signal.strategy_name for signal in self.strategy_signals.values()
            if signal.signal_type == final_signal
        ]
        
        return ConflictResolution(
            final_signal=final_signal,
            confidence=confidence,
            participating_strategies=participating,
            conflict_detected=self._detect_conflicts(signal_counts),
            resolution_method="majority_voting",
            override_applied=False
        )
    
    def _apply_priority_override(
        self,
        signal_counts: Dict[SignalType, int],
        signal_confidences: Dict[SignalType, List[float]]
    ) -> ConflictResolution:
        """Apply priority-based override to resolve conflicts."""
        # Find highest priority strategy with a signal
        for strategy_name in self.strategy_priority:
            if strategy_name in self.strategy_signals:
                signal = self.strategy_signals[strategy_name]
                
                # Check if confidence meets threshold
                if signal.confidence >= self.min_confidence_threshold:
                    return ConflictResolution(
                        final_signal=signal.signal_type,
                        confidence=signal.confidence,
                        participating_strategies=[strategy_name],
                        conflict_detected=True,
                        resolution_method="priority_override",
                        override_applied=True,
                        metadata={"override_strategy": strategy_name}
                    )
        
        # If no priority strategy qualifies, fall back to majority voting
        logger.info("No priority strategy qualified, falling back to majority voting")
        return self._apply_majority_voting(signal_counts, signal_confidences)
    
    def _log_conflict_resolution(
        self,
        resolution: ConflictResolution,
        signal_counts: Dict[SignalType, int],
        signal_confidences: Dict[SignalType, List[float]]
    ):
        """Log conflict resolution details."""
        conflict_record = {
            "timestamp": datetime.now().isoformat(),
            "signal_counts": {k.value: v for k, v in signal_counts.items()},
            "resolution": {
                "final_signal": resolution.final_signal.value,
                "confidence": resolution.confidence,
                "method": resolution.resolution_method,
                "override_applied": resolution.override_applied,
                "participating_strategies": resolution.participating_strategies
            },
            "all_signals": {
                name: {
                    "signal_type": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "price": signal.price
                }
                for name, signal in self.strategy_signals.items()
            }
        }
        
        self.conflict_history.append(conflict_record)
        
        logger.info(f"Conflict resolved: {resolution.final_signal.value} "
                   f"(confidence: {resolution.confidence:.2f}, "
                   f"method: {resolution.resolution_method})")
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current signals."""
        if not self.strategy_signals:
            return {"status": "no_signals", "count": 0}
        
        signal_counts = self._count_signals()
        total_signals = sum(signal_counts.values())
        
        # Calculate average confidence
        all_confidences = [signal.confidence for signal in self.strategy_signals.values()]
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return {
            "status": "active",
            "total_signals": total_signals,
            "signal_distribution": {k.value: v for k, v in signal_counts.items()},
            "average_confidence": avg_confidence,
            "strategies": list(self.strategy_signals.keys()),
            "conflict_detected": self._detect_conflicts(signal_counts)
        }
    
    def clear_signals(self):
        """Clear all current signals."""
        self.strategy_signals.clear()
        logger.debug("Cleared all strategy signals")
    
    def update_strategy_priority(self, new_priority: List[str]):
        """Update the strategy priority list.
        
        Args:
            new_priority: New ordered list of strategy names
        """
        self.strategy_priority = new_priority
        logger.info(f"Updated strategy priority: {new_priority}")
    
    def get_conflict_history(self) -> List[Dict[str, Any]]:
        """Get conflict resolution history."""
        return self.conflict_history.copy()
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            "strategy_priority": self.strategy_priority,
            "min_confidence_threshold": self.min_confidence_threshold,
            "majority_threshold": self.majority_threshold,
            "enable_override": self.enable_override,
            "log_conflicts": self.log_conflicts
        }
    
    def import_config(self, config: Dict[str, Any]) -> bool:
        """Import configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if import successful
        """
        try:
            self.strategy_priority = config.get("strategy_priority", [])
            self.min_confidence_threshold = config.get("min_confidence_threshold", 0.6)
            self.majority_threshold = config.get("majority_threshold", 0.5)
            self.enable_override = config.get("enable_override", True)
            self.log_conflicts = config.get("log_conflicts", True)
            
            logger.info("Configuration imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
