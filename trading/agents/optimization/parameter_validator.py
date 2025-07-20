"""
Parameter Validator Module

This module contains parameter validation functionality for the optimizer agent.
Extracted from the original optimizer_agent.py for modularity.
"""

from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass


@dataclass
class OptimizationParameter:
    """Optimization parameter configuration."""

    name: str
    min_value: Union[float, int]
    max_value: Union[float, int]
    step: Union[float, int]
    parameter_type: str  # 'float', 'int', 'categorical'
    categories: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "parameter_type": self.parameter_type,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationParameter":
        """Create from dictionary."""
        return cls(**data)


class ParameterValidator:
    """Validates optimization parameters and parameter combinations."""

    def __init__(self):
        self.validation_rules = {
            "rsi_period": {"min": 2, "max": 100, "type": "int"},
            "macd_fast": {"min": 2, "max": 50, "type": "int"},
            "macd_slow": {"min": 5, "max": 100, "type": "int"},
            "bollinger_period": {"min": 5, "max": 100, "type": "int"},
            "bollinger_std": {"min": 0.5, "max": 3.0, "type": "float"},
            "stop_loss": {"min": 0.001, "max": 0.1, "type": "float"},
            "take_profit": {"min": 0.001, "max": 0.2, "type": "float"},
            "confidence_threshold": {"min": 0.1, "max": 0.9, "type": "float"}
        }

    def validate_optimization_parameters(
        self, parameters: List[OptimizationParameter]
    ) -> List[OptimizationParameter]:
        """Validate a list of optimization parameters."""
        validated_parameters = []
        
        for param in parameters:
            if self._validate_parameter(param):
                validated_parameters.append(param)
        
        return validated_parameters

    def _validate_parameter(self, param: OptimizationParameter) -> bool:
        """Validate a single parameter."""
        # Check parameter type
        if param.parameter_type == "float":
            return self._validate_numeric_parameter(param)
        elif param.parameter_type == "int":
            return self._validate_numeric_parameter(param)
        elif param.parameter_type == "categorical":
            return self._validate_categorical_parameter(param)
        else:
            return False

    def _validate_numeric_parameter(self, param: OptimizationParameter) -> bool:
        """Validate numeric parameters."""
        # Check bounds
        if param.min_value >= param.max_value:
            return False
        
        # Check step size
        if param.step <= 0:
            return False
        
        # Check if step size is reasonable for the range
        range_size = param.max_value - param.min_value
        if param.step > range_size:
            return False
        
        # Check against known parameter rules
        if param.name in self.validation_rules:
            rule = self.validation_rules[param.name]
            
            # Check type
            if rule["type"] != param.parameter_type:
                return False
            
            # Check bounds
            if param.min_value < rule["min"] or param.max_value > rule["max"]:
                return False
        
        # Check realistic bounds
        return self._check_realistic_bounds(param)

    def _validate_categorical_parameter(self, param: OptimizationParameter) -> bool:
        """Validate categorical parameters."""
        if not param.categories:
            return False
        
        if len(param.categories) < 2:
            return False
        
        # Check for duplicates
        if len(param.categories) != len(set(param.categories)):
            return False
        
        return True

    def _check_realistic_bounds(self, param: OptimizationParameter) -> bool:
        """Check if parameter bounds are realistic."""
        # RSI period bounds
        if param.name == "rsi_period":
            return 2 <= param.min_value <= param.max_value <= 100
        
        # MACD bounds
        elif param.name == "macd_fast":
            return 2 <= param.min_value <= param.max_value <= 50
        elif param.name == "macd_slow":
            return 5 <= param.min_value <= param.max_value <= 100
        
        # Bollinger Bands bounds
        elif param.name == "bollinger_period":
            return 5 <= param.min_value <= param.max_value <= 100
        elif param.name == "bollinger_std":
            return 0.5 <= param.min_value <= param.max_value <= 3.0
        
        # Risk management bounds
        elif param.name == "stop_loss":
            return 0.001 <= param.min_value <= param.max_value <= 0.1
        elif param.name == "take_profit":
            return 0.001 <= param.min_value <= param.max_value <= 0.2
        
        # Confidence bounds
        elif param.name == "confidence_threshold":
            return 0.1 <= param.min_value <= param.max_value <= 0.9
        
        # Default: accept if min < max
        return param.min_value < param.max_value

    def generate_parameter_range(
        self, param: OptimizationParameter
    ) -> List[Union[float, int, str]]:
        """Generate a range of values for a parameter."""
        if param.parameter_type == "categorical":
            return param.categories or []
        
        elif param.parameter_type in ["float", "int"]:
            values = []
            current = param.min_value
            
            while current <= param.max_value:
                values.append(current)
                current += param.step
                
                # Prevent infinite loops
                if len(values) > 1000:
                    break
            
            return values
        
        return []

    def validate_parameter_combination(
        self, combination: Dict[str, Any], parameters: List[OptimizationParameter]
    ) -> bool:
        """Validate a parameter combination."""
        # Check if all required parameters are present
        param_names = {param.name for param in parameters}
        combination_names = set(combination.keys())
        
        if not param_names.issubset(combination_names):
            return False
        
        # Check parameter dependencies
        if not self._check_parameter_dependencies(combination):
            return False
        
        # Check parameter consistency
        if not self._check_parameter_consistency(combination):
            return False
        
        # Check parameter relationships
        if not self._check_parameter_relationships(combination):
            return False
        
        return True

    def _check_parameter_dependencies(self, combination: Dict[str, Any]) -> bool:
        """Check parameter dependencies."""
        # MACD fast must be less than MACD slow
        if "macd_fast" in combination and "macd_slow" in combination:
            if combination["macd_fast"] >= combination["macd_slow"]:
                return False
        
        # Take profit must be greater than stop loss
        if "take_profit" in combination and "stop_loss" in combination:
            if combination["take_profit"] <= combination["stop_loss"]:
                return False
        
        return True

    def _check_parameter_consistency(self, combination: Dict[str, Any]) -> bool:
        """Check parameter consistency."""
        # RSI period should be reasonable
        if "rsi_period" in combination:
            rsi_period = combination["rsi_period"]
            if rsi_period < 2 or rsi_period > 100:
                return False
        
        # Bollinger standard deviation should be reasonable
        if "bollinger_std" in combination:
            bollinger_std = combination["bollinger_std"]
            if bollinger_std < 0.5 or bollinger_std > 3.0:
                return False
        
        return True

    def _check_parameter_relationships(self, combination: Dict[str, Any]) -> bool:
        """Check relationships between parameters."""
        # If using MACD, both fast and slow should be present
        if "macd_fast" in combination and "macd_slow" not in combination:
            return False
        if "macd_slow" in combination and "macd_fast" not in combination:
            return False
        
        # If using Bollinger Bands, both period and std should be present
        if "bollinger_period" in combination and "bollinger_std" not in combination:
            return False
        if "bollinger_std" in combination and "bollinger_period" not in combination:
            return False
        
        return True

    def get_parameter_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter validation rules."""
        return self.validation_rules.copy()

    def add_parameter_rule(self, param_name: str, rule: Dict[str, Any]) -> None:
        """Add a new parameter validation rule."""
        self.validation_rules[param_name] = rule
