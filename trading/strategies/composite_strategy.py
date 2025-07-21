"""
Composite Strategy

This module implements a composite strategy that combines multiple
individual strategies with conflict resolution capabilities.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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
        log_conflicts: bool = True,
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

        logger.info(
            f"CompositeStrategy initialized with {len(self.strategy_priority)} priority strategies"
        )

    def add_strategy_signal(
        self,
        strategy_name: str,
        signal_type: Union[str, SignalType],
        confidence: float,
        price: float,
        volume: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
                logger.warning(
                    f"Invalid confidence {confidence} for {strategy_name}, clamping to [0, 1]"
                )
                confidence = max(0.0, min(1.0, confidence))

            signal = StrategySignal(
                strategy_name=strategy_name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                metadata=metadata or {},
            )

            self.strategy_signals[strategy_name] = signal
            logger.debug(
                f"Added signal from {strategy_name}: {signal_type.value} (confidence: {confidence:.2f})"
            )
            return True

        except ValueError as e:
            logger.error(
                f"Invalid signal type '{signal_type}' for {strategy_name}: {e}"
            )
            return False
        except Exception as e:
            logger.error(f"Error adding signal from {strategy_name}: {e}")
            return False

    def resolve_conflicts(self) -> ConflictResolution:
        """
        Resolve conflicts between multiple strategy signals.

        Returns:
            ConflictResolution object with final decision
        """
        if not self.strategy_signals:
            return ConflictResolution(
                final_signal=SignalType.HOLD,
                confidence=0.0,
                participating_strategies=[],
                conflict_detected=False,
                resolution_method="no_signals",
                override_applied=False,
            )

        # Count signals by type
        signal_counts = self._count_signals()
        signal_confidences = self._aggregate_confidences()

        # Check for conflicts
        conflict_detected = self._detect_conflicts(signal_counts)

        if conflict_detected:
            # Apply conflict resolution
            if self.enable_override and self.strategy_priority:
                resolution = self._apply_priority_override(
                    signal_counts, signal_confidences
                )
            else:
                resolution = self._apply_majority_voting(
                    signal_counts, signal_confidences
                )

            # Log conflict resolution if enabled
            if self.log_conflicts:
                self._log_conflict_resolution(
                    resolution, signal_counts, signal_confidences
                )

            # Store in conflict history
            self.conflict_history.append(
                {
                    "timestamp": datetime.now(),
                    "resolution": resolution,
                    "signal_counts": signal_counts,
                    "signal_confidences": signal_confidences,
                }
            )

            return resolution
        else:
            # No conflicts - use highest confidence signal
            best_signal = max(
                signal_counts.keys(),
                key=lambda x: max(signal_confidences.get(x, [0.0])),
            )
            best_confidence = max(signal_confidences.get(best_signal, [0.0]))

            return ConflictResolution(
                final_signal=best_signal,
                confidence=best_confidence,
                participating_strategies=list(self.strategy_signals.keys()),
                conflict_detected=False,
                resolution_method="highest_confidence",
                override_applied=False,
            )

    def _count_signals(self) -> Dict[SignalType, int]:
        """Count signals by type."""
        counts = {}
        for signal in self.strategy_signals.values():
            counts[signal.signal_type] = counts.get(signal.signal_type, 0) + 1
        return counts

    def _aggregate_confidences(self) -> Dict[SignalType, List[float]]:
        """Aggregate confidence scores by signal type."""
        confidences = {}
        for signal in self.strategy_signals.values():
            if signal.signal_type not in confidences:
                confidences[signal.signal_type] = []
            confidences[signal.signal_type].append(signal.confidence)
        return confidences

    def _detect_conflicts(self, signal_counts: Dict[SignalType, int]) -> bool:
        """Detect if there are conflicting signals."""
        if len(signal_counts) <= 1:
            return False

        # Check for opposing signals
        buy_signals = signal_counts.get(SignalType.BUY, 0) + signal_counts.get(
            SignalType.STRONG_BUY, 0
        )
        sell_signals = signal_counts.get(SignalType.SELL, 0) + signal_counts.get(
            SignalType.STRONG_SELL, 0
        )

        # Conflict if both buy and sell signals exist
        if buy_signals > 0 and sell_signals > 0:
            return True

        # Conflict if no clear majority
        total_signals = sum(signal_counts.values())
        max_signals = max(signal_counts.values())
        if max_signals / total_signals < self.majority_threshold:
            return True

        return False

    def _apply_majority_voting(
        self,
        signal_counts: Dict[SignalType, int],
        signal_confidences: Dict[SignalType, List[float]],
    ) -> ConflictResolution:
        """Apply majority voting to resolve conflicts."""
        total_signals = sum(signal_counts.values())
        best_signal = max(signal_counts.keys(), key=lambda x: signal_counts[x])
        best_count = signal_counts[best_signal]
        max(signal_confidences.get(best_signal, [0.0]))

        # Calculate weighted confidence
        weighted_confidence = sum(signal_confidences.get(best_signal, [0.0])) / len(
            signal_confidences.get(best_signal, [1.0])
        )

        return ConflictResolution(
            final_signal=best_signal,
            confidence=weighted_confidence,
            participating_strategies=list(self.strategy_signals.keys()),
            conflict_detected=True,
            resolution_method="majority_voting",
            override_applied=False,
            metadata={
                "total_signals": total_signals,
                "winning_count": best_count,
                "majority_threshold": self.majority_threshold,
            },
        )

    def _apply_priority_override(
        self,
        signal_counts: Dict[SignalType, int],
        signal_confidences: Dict[SignalType, List[float]],
    ) -> ConflictResolution:
        """Apply priority-based override to resolve conflicts."""
        # Find highest priority strategy with a signal
        for strategy_name in self.strategy_priority:
            if strategy_name in self.strategy_signals:
                signal = self.strategy_signals[strategy_name]
                return ConflictResolution(
                    final_signal=signal.signal_type,
                    confidence=signal.confidence,
                    participating_strategies=list(self.strategy_signals.keys()),
                    conflict_detected=True,
                    resolution_method="priority_override",
                    override_applied=True,
                    metadata={
                        "override_strategy": strategy_name,
                        "priority_rank": self.strategy_priority.index(strategy_name),
                    },
                )

        # Fallback to majority voting if no priority strategy has signals
        return self._apply_majority_voting(signal_counts, signal_confidences)

    def _log_conflict_resolution(
        self,
        resolution: ConflictResolution,
        signal_counts: Dict[SignalType, int],
        signal_confidences: Dict[SignalType, List[float]],
    ):
        """Log conflict resolution details."""
        logger.info("Conflict resolution applied:")
        logger.info(f"  - Final signal: {resolution.final_signal.value}")
        logger.info(f"  - Confidence: {resolution.confidence:.2f}")
        logger.info(f"  - Method: {resolution.resolution_method}")
        logger.info(f"  - Override applied: {resolution.override_applied}")
        logger.info(f"  - Signal counts: {signal_counts}")

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current signals."""
        if not self.strategy_signals:
            return {"message": "No signals available"}

        summary = {
            "total_strategies": len(self.strategy_signals),
            "signals_by_type": {},
            "average_confidence": 0.0,
            "conflict_detected": False,
        }

        # Count signals by type
        for signal in self.strategy_signals.values():
            signal_type = signal.signal_type.value
            summary["signals_by_type"][signal_type] = (
                summary["signals_by_type"].get(signal_type, 0) + 1
            )

        # Calculate average confidence
        confidences = [signal.confidence for signal in self.strategy_signals.values()]
        summary["average_confidence"] = sum(confidences) / len(confidences)

        # Check for conflicts
        signal_counts = self._count_signals()
        summary["conflict_detected"] = self._detect_conflicts(signal_counts)

        return summary

    def clear_signals(self):
        """Clear all current signals."""
        self.strategy_signals.clear()
        logger.info("All strategy signals cleared")

    def update_strategy_priority(self, new_priority: List[str]):
        """Update the strategy priority list."""
        self.strategy_priority = new_priority
        logger.info(f"Strategy priority updated: {new_priority}")

    def get_conflict_history(self) -> List[Dict[str, Any]]:
        """Get conflict resolution history."""
        return self.conflict_history

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            "strategy_priority": self.strategy_priority,
            "min_confidence_threshold": self.min_confidence_threshold,
            "majority_threshold": self.majority_threshold,
            "enable_override": self.enable_override,
            "log_conflicts": self.log_conflicts,
        }

    def import_config(self, config: Dict[str, Any]) -> bool:
        """Import configuration."""
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
