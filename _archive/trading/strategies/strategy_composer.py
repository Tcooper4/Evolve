"""
Strategy Composer

Enhanced with Batch 11 features: toggles to enable/disable individual sub-strategies
at runtime via config or flags.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StrategyToggle(Enum):
    """Strategy toggle states."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    CONDITIONAL = "conditional"


@dataclass
class SubStrategyConfig:
    """Configuration for individual sub-strategies."""

    name: str
    enabled: bool = True
    weight: float = 1.0
    priority: int = 1
    conditions: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_threshold: float = 0.0
    last_performance: float = 0.0
    toggle_state: StrategyToggle = StrategyToggle.ENABLED


class StrategyComposer:
    """Strategy composer with runtime sub-strategy toggles."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy composer.

        Args:
            config: Configuration dictionary with sub-strategy settings
        """
        self.config = config or {}
        self.sub_strategies: Dict[str, SubStrategyConfig] = {}
        self.strategy_history = []
        self.performance_tracker = {}

        # Load sub-strategies from config
        self._load_sub_strategies()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"StrategyComposer initialized with {len(self.sub_strategies)} sub-strategies"
        )

    def _load_sub_strategies(self):
        """Load sub-strategies from configuration."""
        sub_strategies_config = self.config.get("sub_strategies", {})

        for strategy_name, strategy_config in sub_strategies_config.items():
            sub_config = SubStrategyConfig(
                name=strategy_name,
                enabled=strategy_config.get("enabled", True),
                weight=strategy_config.get("weight", 1.0),
                priority=strategy_config.get("priority", 1),
                conditions=strategy_config.get("conditions", {}),
                parameters=strategy_config.get("parameters", {}),
                performance_threshold=strategy_config.get("performance_threshold", 0.0),
            )
            self.sub_strategies[strategy_name] = sub_config

    def add_sub_strategy(
        self,
        name: str,
        enabled: bool = True,
        weight: float = 1.0,
        priority: int = 1,
        conditions: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        performance_threshold: float = 0.0,
    ) -> bool:
        """Add a new sub-strategy with toggle capability.

        Args:
            name: Strategy name
            enabled: Whether strategy is enabled by default
            weight: Strategy weight in composition
            priority: Strategy priority (higher = more important)
            conditions: Conditions for conditional enabling
            parameters: Strategy parameters
            performance_threshold: Minimum performance threshold

        Returns:
            True if strategy added successfully
        """
        try:
            sub_config = SubStrategyConfig(
                name=name,
                enabled=enabled,
                weight=weight,
                priority=priority,
                conditions=conditions or {},
                parameters=parameters or {},
                performance_threshold=performance_threshold,
            )

            self.sub_strategies[name] = sub_config
            self.logger.info(f"Added sub-strategy: {name} (enabled={enabled})")
            return True

        except Exception as e:
            self.logger.error(f"Error adding sub-strategy {name}: {e}")
            return False

    def toggle_sub_strategy(
        self, name: str, enabled: bool, reason: Optional[str] = None
    ) -> bool:
        """Toggle a sub-strategy on/off at runtime.

        Args:
            name: Strategy name
            enabled: Whether to enable or disable
            reason: Reason for toggle (for logging)

        Returns:
            True if toggle successful
        """
        try:
            if name not in self.sub_strategies:
                self.logger.error(f"Sub-strategy {name} not found")
                return False

            old_state = self.sub_strategies[name].enabled
            self.sub_strategies[name].enabled = enabled
            self.sub_strategies[name].toggle_state = (
                StrategyToggle.ENABLED if enabled else StrategyToggle.DISABLED
            )

            # Log the toggle
            toggle_record = {
                "timestamp": datetime.now().isoformat(),
                "strategy": name,
                "old_state": old_state,
                "new_state": enabled,
                "reason": reason or "manual toggle",
            }
            self.strategy_history.append(toggle_record)

            self.logger.info(
                f"Toggled {name}: {old_state} -> {enabled} (reason: {reason})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error toggling sub-strategy {name}: {e}")
            return False

    def set_conditional_toggle(
        self, name: str, conditions: Dict[str, Any], reason: Optional[str] = None
    ) -> bool:
        """Set conditional toggle for a sub-strategy.

        Args:
            name: Strategy name
            conditions: Conditions for enabling/disabling
            reason: Reason for conditional toggle

        Returns:
            True if conditional toggle set successfully
        """
        try:
            if name not in self.sub_strategies:
                self.logger.error(f"Sub-strategy {name} not found")
                return False

            self.sub_strategies[name].conditions = conditions
            self.sub_strategies[name].toggle_state = StrategyToggle.CONDITIONAL

            # Log the conditional toggle
            toggle_record = {
                "timestamp": datetime.now().isoformat(),
                "strategy": name,
                "toggle_type": "conditional",
                "conditions": conditions,
                "reason": reason or "conditional toggle",
            }
            self.strategy_history.append(toggle_record)

            self.logger.info(f"Set conditional toggle for {name}: {conditions}")
            return True

        except Exception as e:
            self.logger.error(f"Error setting conditional toggle for {name}: {e}")
            return False

    def evaluate_conditions(self, market_data: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate conditions for all conditional sub-strategies.

        Args:
            market_data: Current market data

        Returns:
            Dictionary of strategy names to enabled states
        """
        results = {}

        for name, config in self.sub_strategies.items():
            if config.toggle_state == StrategyToggle.CONDITIONAL:
                try:
                    enabled = self._evaluate_strategy_conditions(
                        config.conditions, market_data
                    )
                    results[name] = enabled

                    # Update strategy state if conditions changed
                    if enabled != config.enabled:
                        old_state = config.enabled
                        config.enabled = enabled
                        self.logger.info(
                            f"Conditional toggle for {name}: {old_state} -> {enabled}"
                        )

                except Exception as e:
                    self.logger.error(f"Error evaluating conditions for {name}: {e}")
                    results[name] = False

        return results

    def _evaluate_strategy_conditions(
        self, conditions: Dict[str, Any], market_data: Dict[str, Any]
    ) -> bool:
        """Evaluate conditions for a single strategy.

        Args:
            conditions: Strategy conditions
            market_data: Current market data

        Returns:
            True if conditions are met
        """
        try:
            for condition_key, condition_value in conditions.items():
                if condition_key not in market_data:
                    return False

                market_value = market_data[condition_key]

                # Simple equality check - can be extended for more complex conditions
                if market_value != condition_value:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error evaluating conditions: {e}")
            return False

    def get_active_strategies(self) -> List[str]:
        """Get list of currently active strategies.

        Returns:
            List of active strategy names
        """
        return [name for name, config in self.sub_strategies.items() if config.enabled]

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get weights for all active strategies.

        Returns:
            Dictionary of strategy names to weights
        """
        weights = {}
        total_weight = 0.0

        for name, config in self.sub_strategies.items():
            if config.enabled:
                weights[name] = config.weight
                total_weight += config.weight

        # Normalize weights
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight

        return weights

    def compose_signals(
        self,
        individual_signals: Dict[str, pd.DataFrame],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Compose signals from individual strategies.

        Args:
            individual_signals: Dictionary of strategy names to signal DataFrames
            market_data: Current market data for condition evaluation

        Returns:
            Composed signal DataFrame
        """
        try:
            # Evaluate conditions if market data provided
            if market_data:
                self.evaluate_conditions(market_data)

            # Get active strategies and weights
            active_strategies = self.get_active_strategies()
            weights = self.get_strategy_weights()

            if not active_strategies:
                self.logger.warning("No active strategies found")
                return pd.DataFrame()

            # Initialize composed signal
            composed_signal = pd.DataFrame()

            for strategy_name in active_strategies:
                if strategy_name in individual_signals:
                    signal_df = individual_signals[strategy_name]
                    weight = weights.get(strategy_name, 1.0)

                    # Weight the signal
                    weighted_signal = signal_df * weight

                    if composed_signal.empty:
                        composed_signal = weighted_signal
                    else:
                        composed_signal += weighted_signal

            self.logger.info(
                f"Composed signals from {len(active_strategies)} strategies"
            )
            return composed_signal

        except Exception as e:
            self.logger.error(f"Error composing signals: {e}")
            return pd.DataFrame()

    def update_performance(self, strategy_name: str, performance: float):
        """Update performance for a strategy.

        Args:
            strategy_name: Name of the strategy
            performance: Performance metric
        """
        try:
            if strategy_name in self.sub_strategies:
                self.sub_strategies[strategy_name].last_performance = performance
                self.performance_tracker[strategy_name] = performance

                # Check if performance is below threshold
                config = self.sub_strategies[strategy_name]
                if performance < config.performance_threshold:
                    self.logger.warning(
                        f"Strategy {strategy_name} performance {performance} below threshold {config.performance_threshold}"
                    )

        except Exception as e:
            self.logger.error(f"Error updating performance for {strategy_name}: {e}")

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategies.

        Returns:
            Dictionary with strategy status information
        """
        status = {
            "total_strategies": len(self.sub_strategies),
            "active_strategies": len(self.get_active_strategies()),
            "strategies": {},
        }

        for name, config in self.sub_strategies.items():
            status["strategies"][name] = {
                "enabled": config.enabled,
                "weight": config.weight,
                "priority": config.priority,
                "toggle_state": config.toggle_state.value,
                "performance": config.last_performance,
                "performance_threshold": config.performance_threshold,
            }

        return status

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration.

        Returns:
            Configuration dictionary
        """
        config = {"sub_strategies": {}}

        for name, sub_config in self.sub_strategies.items():
            config["sub_strategies"][name] = {
                "enabled": sub_config.enabled,
                "weight": sub_config.weight,
                "priority": sub_config.priority,
                "conditions": sub_config.conditions,
                "parameters": sub_config.parameters,
                "performance_threshold": sub_config.performance_threshold,
                "toggle_state": sub_config.toggle_state.value,
            }

        return config

    def import_config(self, config: Dict[str, Any]) -> bool:
        """Import configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if import successful
        """
        try:
            sub_strategies_config = config.get("sub_strategies", {})

            for strategy_name, strategy_config in sub_strategies_config.items():
                if strategy_name in self.sub_strategies:
                    # Update existing strategy
                    sub_config = self.sub_strategies[strategy_name]
                    sub_config.enabled = strategy_config.get(
                        "enabled", sub_config.enabled
                    )
                    sub_config.weight = strategy_config.get("weight", sub_config.weight)
                    sub_config.priority = strategy_config.get(
                        "priority", sub_config.priority
                    )
                    sub_config.conditions = strategy_config.get(
                        "conditions", sub_config.conditions
                    )
                    sub_config.parameters = strategy_config.get(
                        "parameters", sub_config.parameters
                    )
                    sub_config.performance_threshold = strategy_config.get(
                        "performance_threshold", sub_config.performance_threshold
                    )
                    sub_config.toggle_state = StrategyToggle(
                        strategy_config.get(
                            "toggle_state", sub_config.toggle_state.value
                        )
                    )

            self.logger.info(
                f"Imported configuration for {len(sub_strategies_config)} strategies"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False
