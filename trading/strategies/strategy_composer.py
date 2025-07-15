"""
Strategy Composer

Enhanced with Batch 11 features: toggles to enable/disable individual sub-strategies
at runtime via config or flags.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

import pandas as pd
import numpy as np

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
        self.logger.info(f"StrategyComposer initialized with {len(self.sub_strategies)} sub-strategies")
    
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
                performance_threshold=strategy_config.get("performance_threshold", 0.0)
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
        performance_threshold: float = 0.0
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
                performance_threshold=performance_threshold
            )
            
            self.sub_strategies[name] = sub_config
            self.logger.info(f"Added sub-strategy: {name} (enabled={enabled})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding sub-strategy {name}: {e}")
            return False
    
    def toggle_sub_strategy(
        self, 
        name: str, 
        enabled: bool,
        reason: Optional[str] = None
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
                "reason": reason or "manual toggle"
            }
            self.strategy_history.append(toggle_record)
            
            self.logger.info(f"Toggled {name}: {old_state} -> {enabled} (reason: {reason})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error toggling sub-strategy {name}: {e}")
            return False
    
    def set_conditional_toggle(
        self, 
        name: str, 
        conditions: Dict[str, Any],
        reason: Optional[str] = None
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
                "reason": reason or "conditional toggle"
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
                    enabled = self._evaluate_strategy_conditions(config.conditions, market_data)
                    results[name] = enabled
                    
                    # Update strategy state if conditions changed
                    if enabled != config.enabled:
                        old_state = config.enabled
                        config.enabled = enabled
                        self.logger.info(f"Conditional toggle for {name}: {old_state} -> {enabled}")
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating conditions for {name}: {e}")
                    results[name] = False
        
        return results
    
    def _evaluate_strategy_conditions(
        self, 
        conditions: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> bool:
        """Evaluate conditions for a single strategy.
        
        Args:
            conditions: Strategy conditions
            market_data: Current market data
            
        Returns:
            True if conditions are met
        """
        try:
            for condition_type, condition_value in conditions.items():
                if condition_type == "volatility_threshold":
                    current_vol = market_data.get("volatility", 0.0)
                    if current_vol > condition_value:
                        return False
                        
                elif condition_type == "market_regime":
                    current_regime = market_data.get("market_regime", "neutral")
                    if current_regime not in condition_value:
                        return False
                        
                elif condition_type == "performance_threshold":
                    strategy_perf = self.performance_tracker.get("strategy_name", 0.0)
                    if strategy_perf < condition_value:
                        return False
                        
                elif condition_type == "time_window":
                    current_time = datetime.now()
                    start_time = condition_value.get("start")
                    end_time = condition_value.get("end")
                    
                    if start_time and current_time < start_time:
                        return False
                    if end_time and current_time > end_time:
                        return False
                        
                elif condition_type == "custom":
                    # Custom condition evaluation
                    custom_func = condition_value.get("function")
                    if custom_func and not custom_func(market_data):
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating conditions: {e}")
            return False
    
    def get_active_strategies(self) -> List[str]:
        """Get list of currently active sub-strategies.
        
        Returns:
            List of active strategy names
        """
        return [
            name for name, config in self.sub_strategies.items() 
            if config.enabled
        ]
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get weights for all active sub-strategies.
        
        Returns:
            Dictionary of strategy names to weights
        """
        active_strategies = self.get_active_strategies()
        total_weight = sum(
            self.sub_strategies[name].weight 
            for name in active_strategies
        )
        
        if total_weight == 0:
            return {name: 1.0 / len(active_strategies) for name in active_strategies}
        
        return {
            name: self.sub_strategies[name].weight / total_weight
            for name in active_strategies
        }
    
    def compose_signals(
        self, 
        individual_signals: Dict[str, pd.DataFrame],
        market_data: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Compose signals from individual sub-strategies.
        
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
                self.logger.warning("No active sub-strategies found")
                return pd.DataFrame()
            
            # Compose signals
            composed_signals = []
            
            for strategy_name in active_strategies:
                if strategy_name in individual_signals:
                    signals_df = individual_signals[strategy_name].copy()
                    weight = weights[strategy_name]
                    
                    # Apply weight to signal strength
                    if "signal_strength" in signals_df.columns:
                        signals_df["signal_strength"] *= weight
                    if "confidence" in signals_df.columns:
                        signals_df["confidence"] *= weight
                    
                    signals_df["strategy"] = strategy_name
                    signals_df["weight"] = weight
                    composed_signals.append(signals_df)
            
            if not composed_signals:
                return pd.DataFrame()
            
            # Combine signals
            combined_df = pd.concat(composed_signals, ignore_index=True)
            
            # Aggregate by timestamp and symbol
            if "timestamp" in combined_df.columns and "symbol" in combined_df.columns:
                aggregated = combined_df.groupby(["timestamp", "symbol"]).agg({
                    "signal_strength": "sum",
                    "confidence": "mean",
                    "weight": "sum"
                }).reset_index()
                
                # Normalize signal strength
                if "signal_strength" in aggregated.columns:
                    aggregated["signal_strength"] = aggregated["signal_strength"].clip(-1, 1)
                
                return aggregated
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error composing signals: {e}")
            return pd.DataFrame()
    
    def update_performance(
        self, 
        strategy_name: str, 
        performance: float
    ):
        """Update performance for a sub-strategy.
        
        Args:
            strategy_name: Strategy name
            performance: Performance metric
        """
        try:
            if strategy_name in self.sub_strategies:
                self.sub_strategies[strategy_name].last_performance = performance
                self.performance_tracker[strategy_name] = performance
                
                # Check if performance is below threshold
                threshold = self.sub_strategies[strategy_name].performance_threshold
                if performance < threshold and self.sub_strategies[strategy_name].enabled:
                    self.toggle_sub_strategy(
                        strategy_name, 
                        False, 
                        f"Performance {performance} below threshold {threshold}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error updating performance for {strategy_name}: {e}")
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all sub-strategies.
        
        Returns:
            Dictionary with strategy status information
        """
        status = {
            "total_strategies": len(self.sub_strategies),
            "active_strategies": len(self.get_active_strategies()),
            "strategies": {}
        }
        
        for name, config in self.sub_strategies.items():
            status["strategies"][name] = {
                "enabled": config.enabled,
                "toggle_state": config.toggle_state.value,
                "weight": config.weight,
                "priority": config.priority,
                "performance": config.last_performance,
                "threshold": config.performance_threshold,
                "conditions": config.conditions
            }
        
        return status
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration.
        
        Returns:
            Configuration dictionary
        """
        config = {
            "timestamp": datetime.now().isoformat(),
            "sub_strategies": {}
        }
        
        for name, sub_config in self.sub_strategies.items():
            config["sub_strategies"][name] = {
                "enabled": sub_config.enabled,
                "weight": sub_config.weight,
                "priority": sub_config.priority,
                "conditions": sub_config.conditions,
                "parameters": sub_config.parameters,
                "performance_threshold": sub_config.performance_threshold,
                "toggle_state": sub_config.toggle_state.value
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
            
            for name, strategy_config in sub_strategies_config.items():
                if name in self.sub_strategies:
                    self.sub_strategies[name].enabled = strategy_config.get("enabled", True)
                    self.sub_strategies[name].weight = strategy_config.get("weight", 1.0)
                    self.sub_strategies[name].priority = strategy_config.get("priority", 1)
                    self.sub_strategies[name].conditions = strategy_config.get("conditions", {})
                    self.sub_strategies[name].parameters = strategy_config.get("parameters", {})
                    self.sub_strategies[name].performance_threshold = strategy_config.get("performance_threshold", 0.0)
                    self.sub_strategies[name].toggle_state = StrategyToggle(strategy_config.get("toggle_state", "enabled"))
            
            self.logger.info("Configuration imported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False 