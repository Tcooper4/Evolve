"""
Strategy Parameter Validator

This module provides parameter validation and guardrails for trading strategies
to prevent crashes and ensure robust operation.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for parameters."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    is_valid: bool
    level: ValidationLevel
    message: str
    suggested_value: Optional[Any] = None
    parameter_name: Optional[str] = None


@dataclass
class ParameterConstraint:
    """Constraint for a parameter."""

    name: str
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    data_type: str = "float"
    required: bool = True
    description: str = ""


class StrategyParameterValidator:
    """Validator for strategy parameters with guardrails."""

    def __init__(self):
        """Initialize the parameter validator."""
        self.constraints = self._initialize_constraints()
        self.validation_history = []

    def _initialize_constraints(self) -> Dict[str, List[ParameterConstraint]]:
        """Initialize parameter constraints for different strategies."""
        constraints = {
            "rsi": [
                ParameterConstraint(
                    name="period", min_value=1, max_value=100, data_type="int", description="RSI calculation period"
                ),
                ParameterConstraint(
                    name="oversold", min_value=0, max_value=50, data_type="int", description="Oversold threshold"
                ),
                ParameterConstraint(
                    name="overbought", min_value=50, max_value=100, data_type="int", description="Overbought threshold"
                ),
            ],
            "macd": [
                ParameterConstraint(
                    name="fast_period", min_value=1, max_value=50, data_type="int", description="Fast EMA period"
                ),
                ParameterConstraint(
                    name="slow_period", min_value=1, max_value=100, data_type="int", description="Slow EMA period"
                ),
                ParameterConstraint(
                    name="signal_period", min_value=1, max_value=50, data_type="int", description="Signal line period"
                ),
            ],
            "bollinger": [
                ParameterConstraint(
                    name="period", min_value=1, max_value=100, data_type="int", description="Moving average period"
                ),
                ParameterConstraint(
                    name="std_dev",
                    min_value=0.1,
                    max_value=5.0,
                    data_type="float",
                    description="Standard deviation multiplier",
                ),
            ],
            "sma": [
                ParameterConstraint(
                    name="short_period",
                    min_value=1,
                    max_value=50,
                    data_type="int",
                    description="Short moving average period",
                ),
                ParameterConstraint(
                    name="long_period",
                    min_value=1,
                    max_value=200,
                    data_type="int",
                    description="Long moving average period",
                ),
            ],
            "ema": [
                ParameterConstraint(
                    name="period", min_value=1, max_value=100, data_type="int", description="EMA period"
                ),
                ParameterConstraint(
                    name="alpha", min_value=0.01, max_value=1.0, data_type="float", description="Smoothing factor"
                ),
            ],
            "stochastic": [
                ParameterConstraint(
                    name="k_period", min_value=1, max_value=50, data_type="int", description="%K period"
                ),
                ParameterConstraint(
                    name="d_period", min_value=1, max_value=20, data_type="int", description="%D period"
                ),
                ParameterConstraint(
                    name="oversold", min_value=0, max_value=50, data_type="int", description="Oversold threshold"
                ),
                ParameterConstraint(
                    name="overbought", min_value=50, max_value=100, data_type="int", description="Overbought threshold"
                ),
            ],
        }
        return constraints

    def validate_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> List[ValidationResult]:
        """Validate parameters for a strategy.

        Args:
            strategy_name: Name of the strategy
            parameters: Parameters to validate

        Returns:
            List of validation results
        """
        try:
            results = []

            if strategy_name not in self.constraints:
                results.append(
                    ValidationResult(
                        is_valid=False, level=ValidationLevel.ERROR, message=f"Unknown strategy: {strategy_name}"
                    )
                )
                return results

            strategy_constraints = self.constraints[strategy_name]

            # Validate each parameter
            for constraint in strategy_constraints:
                param_name = constraint.name

                if param_name not in parameters:
                    if constraint.required:
                        results.append(
                            ValidationResult(
                                is_valid=False,
                                level=ValidationLevel.ERROR,
                                message=f"Missing required parameter: {param_name}",
                                parameter_name=param_name,
                            )
                        )
                    continue

                param_value = parameters[param_name]
                validation_result = self._validate_parameter(param_value, constraint)
                results.append(validation_result)

            # Validate parameter relationships
            relationship_results = self._validate_parameter_relationships(strategy_name, parameters)
            results.extend(relationship_results)

            # Store validation history
            self.validation_history.append(
                {
                    "strategy": strategy_name,
                    "parameters": parameters,
                    "results": results,
                    "timestamp": np.datetime64("now"),
                }
            )

            return results

        except Exception as e:
            logger.error(f"Error validating parameters: {e}")
            return [
                ValidationResult(is_valid=False, level=ValidationLevel.CRITICAL, message=f"Validation error: {str(e)}")
            ]

    def _validate_parameter(self, value: Any, constraint: ParameterConstraint) -> ValidationResult:
        """Validate a single parameter against its constraint.

        Args:
            value: Parameter value
            constraint: Parameter constraint

        Returns:
            Validation result
        """
        try:
            param_name = constraint.name

            # Check data type
            if constraint.data_type == "int":
                if not isinstance(value, (int, np.integer)):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        return ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.ERROR,
                            message=f"{param_name} must be an integer",
                            parameter_name=param_name,
                            suggested_value=int(value) if isinstance(value, (int, float)) else 1,
                        )

            elif constraint.data_type == "float":
                if not isinstance(value, (int, float, np.number)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        return ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.ERROR,
                            message=f"{param_name} must be a number",
                            parameter_name=param_name,
                            suggested_value=float(value) if isinstance(value, (int, float)) else 0.0,
                        )

            # Check allowed values
            if constraint.allowed_values is not None:
                if value not in constraint.allowed_values:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"{param_name} must be one of {constraint.allowed_values}",
                        parameter_name=param_name,
                        suggested_value=constraint.allowed_values[0],
                    )

            # Check min/max values
            if constraint.min_value is not None and value < constraint.min_value:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"{param_name} ({value}) is below minimum ({constraint.min_value})",
                    parameter_name=param_name,
                    suggested_value=constraint.min_value,
                )

            if constraint.max_value is not None and value > constraint.max_value:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"{param_name} ({value}) is above maximum ({constraint.max_value})",
                    parameter_name=param_name,
                    suggested_value=constraint.max_value,
                )

            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"{param_name} is valid",
                parameter_name=param_name,
            )

        except Exception as e:
            logger.error(f"Error validating parameter {constraint.name}: {e}")
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message=f"Validation error for {constraint.name}: {str(e)}",
                parameter_name=constraint.name,
            )

    def _validate_parameter_relationships(
        self, strategy_name: str, parameters: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate relationships between parameters.

        Args:
            strategy_name: Name of the strategy
            parameters: Parameters to validate

        Returns:
            List of validation results
        """
        results = []

        try:
            if strategy_name == "rsi":
                # RSI: oversold should be less than overbought
                oversold = parameters.get("oversold")
                overbought = parameters.get("overbought")

                if oversold is not None and overbought is not None:
                    if oversold >= overbought:
                        results.append(
                            ValidationResult(
                                is_valid=False,
                                level=ValidationLevel.ERROR,
                                message="Oversold threshold must be less than overbought threshold",
                                parameter_name="oversold",
                                suggested_value=min(oversold, overbought - 10),
                            )
                        )

            elif strategy_name == "macd":
                # MACD: fast period should be less than slow period
                fast_period = parameters.get("fast_period")
                slow_period = parameters.get("slow_period")

                if fast_period is not None and slow_period is not None:
                    if fast_period >= slow_period:
                        results.append(
                            ValidationResult(
                                is_valid=False,
                                level=ValidationLevel.ERROR,
                                message="Fast period must be less than slow period",
                                parameter_name="fast_period",
                                suggested_value=min(fast_period, slow_period - 5),
                            )
                        )

            elif strategy_name == "sma":
                # SMA: short period should be less than long period
                short_period = parameters.get("short_period")
                long_period = parameters.get("long_period")

                if short_period is not None and long_period is not None:
                    if short_period >= long_period:
                        results.append(
                            ValidationResult(
                                is_valid=False,
                                level=ValidationLevel.ERROR,
                                message="Short period must be less than long period",
                                parameter_name="short_period",
                                suggested_value=min(short_period, long_period - 10),
                            )
                        )

            elif strategy_name == "stochastic":
                # Stochastic: oversold should be less than overbought
                oversold = parameters.get("oversold")
                overbought = parameters.get("overbought")

                if oversold is not None and overbought is not None:
                    if oversold >= overbought:
                        results.append(
                            ValidationResult(
                                is_valid=False,
                                level=ValidationLevel.ERROR,
                                message="Oversold threshold must be less than overbought threshold",
                                parameter_name="oversold",
                                suggested_value=min(oversold, overbought - 10),
                            )
                        )

        except Exception as e:
            logger.error(f"Error validating parameter relationships: {e}")
            results.append(
                ValidationResult(
                    is_valid=False, level=ValidationLevel.CRITICAL, message=f"Relationship validation error: {str(e)}"
                )
            )

        return results

    def fix_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fix invalid parameters by applying suggested values.

        Args:
            strategy_name: Name of the strategy
            parameters: Parameters to fix

        Returns:
            Fixed parameters dictionary
        """
        try:
            validation_results = self.validate_parameters(strategy_name, parameters)
            fixed_parameters = parameters.copy()

            for result in validation_results:
                if not result.is_valid and result.suggested_value is not None:
                    fixed_parameters[result.parameter_name] = result.suggested_value
                    logger.info(f"Fixed {result.parameter_name}: {result.message}")

            return fixed_parameters

        except Exception as e:
            logger.error(f"Error fixing parameters: {e}")
            return parameters

    def get_safe_defaults(self, strategy_name: str) -> Dict[str, Any]:
        """Get safe default parameters for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary of safe default parameters
        """
        try:
            if strategy_name not in self.constraints:
                return {}

            defaults = {}
            for constraint in self.constraints[strategy_name]:
                if constraint.data_type == "int":
                    if constraint.min_value is not None:
                        defaults[constraint.name] = constraint.min_value
                    else:
                        defaults[constraint.name] = 1
                elif constraint.data_type == "float":
                    if constraint.min_value is not None:
                        defaults[constraint.name] = constraint.min_value
                    else:
                        defaults[constraint.name] = 0.0
                elif constraint.allowed_values:
                    defaults[constraint.name] = constraint.allowed_values[0]

            return defaults

        except Exception as e:
            logger.error(f"Error getting safe defaults: {e}")
            return {}

    def add_constraint(self, strategy_name: str, constraint: ParameterConstraint):
        """Add a new parameter constraint.

        Args:
            strategy_name: Name of the strategy
            constraint: Parameter constraint to add
        """
        try:
            if strategy_name not in self.constraints:
                self.constraints[strategy_name] = []

            # Check if constraint already exists
            existing_names = [c.name for c in self.constraints[strategy_name]]
            if constraint.name in existing_names:
                # Update existing constraint
                for i, existing in enumerate(self.constraints[strategy_name]):
                    if existing.name == constraint.name:
                        self.constraints[strategy_name][i] = constraint
                        break
            else:
                # Add new constraint
                self.constraints[strategy_name].append(constraint)

            logger.info(f"Added constraint for {strategy_name}.{constraint.name}")

        except Exception as e:
            logger.error(f"Error adding constraint: {e}")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation history.

        Returns:
            Dictionary with validation summary
        """
        try:
            if not self.validation_history:
                return {"message": "No validation history available"}

            recent_validations = self.validation_history[-50:]  # Last 50 validations

            total_validations = len(recent_validations)
            failed_validations = sum(1 for v in recent_validations if any(not r.is_valid for r in v["results"]))

            strategy_counts = {}
            error_counts = {}

            for validation in recent_validations:
                strategy = validation["strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                for result in validation["results"]:
                    if not result.is_valid:
                        error_counts[result.level.value] = error_counts.get(result.level.value, 0) + 1

            return {
                "total_validations": total_validations,
                "failed_validations": failed_validations,
                "success_rate": (total_validations - failed_validations) / total_validations
                if total_validations > 0
                else 0.0,
                "strategy_counts": strategy_counts,
                "error_counts": error_counts,
                "last_validation": recent_validations[-1]["timestamp"] if recent_validations else None,
            }

        except Exception as e:
            logger.error(f"Error getting validation summary: {e}")
            return {"error": str(e)}


# Global instance
parameter_validator = StrategyParameterValidator()


def get_parameter_validator() -> StrategyParameterValidator:
    """Get the global parameter validator instance."""
    return parameter_validator


def validate_strategy_parameters(strategy_name: str, parameters: Dict[str, Any]) -> List[ValidationResult]:
    """Convenience function to validate strategy parameters.

    Args:
        strategy_name: Name of the strategy
        parameters: Parameters to validate

    Returns:
        List of validation results
    """
    return parameter_validator.validate_parameters(strategy_name, parameters)


def fix_strategy_parameters(strategy_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to fix strategy parameters.

    Args:
        strategy_name: Name of the strategy
        parameters: Parameters to fix

    Returns:
        Fixed parameters dictionary
    """
    return parameter_validator.fix_parameters(strategy_name, parameters)
