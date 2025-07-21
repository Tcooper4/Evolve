"""
Strategy Dispatcher with Validation

This module provides a strategy dispatcher that validates strategies
before invoking downstream logic and manages strategy execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VALIDATED = "validated"
    INVALID = "invalid"


class ValidationLevel(Enum):
    """Validation levels for strategies."""

    BASIC = "basic"  # Basic syntax and structure
    INTERMEDIATE = "intermediate"  # Parameter validation
    ADVANCED = "advanced"  # Full validation including dependencies
    STRICT = "strict"  # Strict validation with all checks


@dataclass
class StrategyConfig:
    """Strategy configuration."""

    name: str
    strategy_type: str
    parameters: Dict[str, Any]
    risk_level: str = "medium"
    max_position_size: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    enabled: bool = True
    priority: int = 3
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of strategy validation."""

    is_valid: bool
    validation_level: ValidationLevel
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyExecution:
    """Strategy execution record."""

    execution_id: str
    strategy_name: str
    strategy_config: StrategyConfig
    status: StrategyStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    validation_result: Optional[ValidationResult] = None
    execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyValidator:
    """Validates strategies before execution."""

    def __init__(
        self, validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE
    ):
        """
        Initialize strategy validator.

        Args:
            validation_level: Level of validation to perform
        """
        self.validation_level = validation_level
        self.required_parameters = self._load_required_parameters()
        self.parameter_constraints = self._load_parameter_constraints()
        self.strategy_templates = self._load_strategy_templates()

    def _load_required_parameters(self) -> Dict[str, List[str]]:
        """Load required parameters for different strategy types."""
        return {
            "rsi": ["period", "overbought", "oversold"],
            "macd": ["fast_period", "slow_period", "signal_period"],
            "bollinger": ["period", "std_dev"],
            "moving_average": ["period", "ma_type"],
            "momentum": ["period"],
            "mean_reversion": ["period", "threshold"],
            "breakout": ["period", "threshold"],
            "custom": [],  # Custom strategies have no predefined requirements
        }

    def _load_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Load parameter constraints for validation."""
        return {
            "period": {"min": 1, "max": 500, "type": "int"},
            "fast_period": {"min": 1, "max": 100, "type": "int"},
            "slow_period": {"min": 1, "max": 200, "type": "int"},
            "signal_period": {"min": 1, "max": 50, "type": "int"},
            "overbought": {"min": 50, "max": 100, "type": "float"},
            "oversold": {"min": 0, "max": 50, "type": "float"},
            "std_dev": {"min": 0.1, "max": 5.0, "type": "float"},
            "threshold": {"min": 0.0, "max": 1.0, "type": "float"},
            "risk_level": {"values": ["low", "medium", "high"], "type": "str"},
            "max_position_size": {"min": 0.0, "max": 1.0, "type": "float"},
        }

    def _load_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load strategy templates for validation."""
        return {
            "rsi": {
                "description": "Relative Strength Index strategy",
                "parameters": ["period", "overbought", "oversold"],
                "signals": ["buy", "sell"],
                "timeframe": "any",
            },
            "macd": {
                "description": "Moving Average Convergence Divergence strategy",
                "parameters": ["fast_period", "slow_period", "signal_period"],
                "signals": ["buy", "sell"],
                "timeframe": "any",
            },
            "bollinger": {
                "description": "Bollinger Bands strategy",
                "parameters": ["period", "std_dev"],
                "signals": ["buy", "sell"],
                "timeframe": "any",
            },
            "moving_average": {
                "description": "Moving Average crossover strategy",
                "parameters": ["period", "ma_type"],
                "signals": ["buy", "sell"],
                "timeframe": "any",
            },
        }

    def validate_strategy(self, strategy_config: StrategyConfig) -> ValidationResult:
        """
        Validate a strategy configuration.

        Args:
            strategy_config: Strategy configuration to validate

        Returns:
            ValidationResult with validation details
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        suggestions = []

        try:
            # Basic validation
            if self.validation_level in [
                ValidationLevel.BASIC,
                ValidationLevel.INTERMEDIATE,
                ValidationLevel.ADVANCED,
                ValidationLevel.STRICT,
            ]:
                basic_errors, basic_warnings = self._validate_basic(strategy_config)
                errors.extend(basic_errors)
                warnings.extend(basic_warnings)

            # Parameter validation
            if self.validation_level in [
                ValidationLevel.INTERMEDIATE,
                ValidationLevel.ADVANCED,
                ValidationLevel.STRICT,
            ]:
                param_errors, param_warnings, param_suggestions = (
                    self._validate_parameters(strategy_config)
                )
                errors.extend(param_errors)
                warnings.extend(param_warnings)
                suggestions.extend(param_suggestions)

            # Advanced validation
            if self.validation_level in [
                ValidationLevel.ADVANCED,
                ValidationLevel.STRICT,
            ]:
                adv_errors, adv_warnings = self._validate_advanced(strategy_config)
                errors.extend(adv_errors)
                warnings.extend(adv_warnings)

            # Strict validation
            if self.validation_level == ValidationLevel.STRICT:
                strict_errors, strict_warnings = self._validate_strict(strategy_config)
                errors.extend(strict_errors)
                warnings.extend(strict_warnings)

            # Calculate validation time
            validation_time = (datetime.now() - start_time).total_seconds()

            # Determine if strategy is valid
            is_valid = len(errors) == 0

            return ValidationResult(
                is_valid=is_valid,
                validation_level=self.validation_level,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                validation_time=validation_time,
                metadata={
                    "strategy_name": strategy_config.name,
                    "strategy_type": strategy_config.strategy_type,
                    "validation_level": self.validation_level.value,
                },
            )

        except Exception as e:
            logger.error(f"Error validating strategy {strategy_config.name}: {e}")
            return ValidationResult(
                is_valid=False,
                validation_level=self.validation_level,
                errors=[f"Validation error: {str(e)}"],
                validation_time=(datetime.now() - start_time).total_seconds(),
            )

    def _validate_basic(
        self, strategy_config: StrategyConfig
    ) -> Tuple[List[str], List[str]]:
        """Perform basic validation."""
        errors = []
        warnings = []

        # Check required fields
        if not strategy_config.name:
            errors.append("Strategy name is required")

        if not strategy_config.strategy_type:
            errors.append("Strategy type is required")

        if not strategy_config.parameters:
            errors.append("Strategy parameters are required")

        # Check strategy type validity
        if strategy_config.strategy_type not in self.strategy_templates:
            warnings.append(f"Unknown strategy type: {strategy_config.strategy_type}")

        # Check risk level
        valid_risk_levels = ["low", "medium", "high"]
        if strategy_config.risk_level not in valid_risk_levels:
            errors.append(f"Invalid risk level: {strategy_config.risk_level}")

        return errors, warnings

    def _validate_parameters(
        self, strategy_config: StrategyConfig
    ) -> Tuple[List[str], List[str], List[str]]:
        """Validate strategy parameters."""
        errors = []
        warnings = []
        suggestions = []

        # Check required parameters for strategy type
        required_params = self.required_parameters.get(
            strategy_config.strategy_type, []
        )
        for param in required_params:
            if param not in strategy_config.parameters:
                errors.append(f"Required parameter missing: {param}")

        # Validate parameter values
        for param_name, param_value in strategy_config.parameters.items():
            if param_name in self.parameter_constraints:
                constraint = self.parameter_constraints[param_name]

                # Type validation
                expected_type = constraint.get("type")
                if expected_type == "int" and not isinstance(param_value, int):
                    errors.append(f"Parameter {param_name} must be an integer")
                elif expected_type == "float" and not isinstance(
                    param_value, (int, float)
                ):
                    errors.append(f"Parameter {param_name} must be a number")
                elif expected_type == "str" and not isinstance(param_value, str):
                    errors.append(f"Parameter {param_name} must be a string")

                # Range validation
                if "min" in constraint and param_value < constraint["min"]:
                    errors.append(
                        f"Parameter {param_name} must be >= {constraint['min']}"
                    )

                if "max" in constraint and param_value > constraint["max"]:
                    errors.append(
                        f"Parameter {param_name} must be <= {constraint['max']}"
                    )

                # Value validation
                if "values" in constraint and param_value not in constraint["values"]:
                    errors.append(
                        f"Parameter {param_name} must be one of: {constraint['values']}"
                    )

        # Check for unused parameters
        template = self.strategy_templates.get(strategy_config.strategy_type, {})
        expected_params = template.get("parameters", [])
        for param_name in strategy_config.parameters:
            if (
                param_name not in expected_params
                and strategy_config.strategy_type != "custom"
            ):
                warnings.append(f"Unexpected parameter: {param_name}")

        # Generate suggestions
        if strategy_config.strategy_type == "rsi":
            if (
                "period" in strategy_config.parameters
                and strategy_config.parameters["period"] < 10
            ):
                suggestions.append(
                    "Consider using a longer RSI period (14-21) for more stable signals"
                )

        return errors, warnings, suggestions

    def _validate_advanced(
        self, strategy_config: StrategyConfig
    ) -> Tuple[List[str], List[str]]:
        """Perform advanced validation."""
        errors = []
        warnings = []

        # Check parameter relationships
        if strategy_config.strategy_type == "macd":
            fast_period = strategy_config.parameters.get("fast_period")
            slow_period = strategy_config.parameters.get("slow_period")

            if fast_period and slow_period and fast_period >= slow_period:
                errors.append("MACD fast_period must be less than slow_period")

        # Check risk management parameters
        if strategy_config.max_position_size > 1.0:
            errors.append("max_position_size cannot exceed 1.0 (100%)")

        if strategy_config.stop_loss and strategy_config.stop_loss <= 0:
            errors.append("stop_loss must be positive")

        if strategy_config.take_profit and strategy_config.take_profit <= 0:
            errors.append("take_profit must be positive")

        return errors, warnings

    def _validate_strict(
        self, strategy_config: StrategyConfig
    ) -> Tuple[List[str], List[str]]:
        """Perform strict validation."""
        errors = []
        warnings = []

        # Check for optimal parameter ranges
        if strategy_config.strategy_type == "rsi":
            period = strategy_config.parameters.get("period")
            if period and (period < 5 or period > 50):
                warnings.append("RSI period outside recommended range (5-50)")

        # Check for high-risk configurations
        if (
            strategy_config.risk_level == "high"
            and strategy_config.max_position_size > 0.5
        ):
            warnings.append("High risk level with large position size may be dangerous")

        # Check for missing risk management
        if not strategy_config.stop_loss and strategy_config.risk_level in [
            "medium",
            "high",
        ]:
            warnings.append("Consider adding stop_loss for risk management")

        return errors, warnings


class StrategyDispatcher:
    """
    Strategy dispatcher with validation and execution management.
    """

    def __init__(
        self, validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE
    ):
        """
        Initialize strategy dispatcher.

        Args:
            validation_level: Validation level for strategies
        """
        self.validator = StrategyValidator(validation_level)
        self.executions: Dict[str, StrategyExecution] = {}
        self.strategy_registry: Dict[str, StrategyConfig] = {}

        logger.info(
            f"StrategyDispatcher initialized with {validation_level.value} validation"
        )

    def register_strategy(self, strategy_config: StrategyConfig) -> bool:
        """
        Register a strategy with validation.

        Args:
            strategy_config: Strategy configuration

        Returns:
            True if registration successful
        """
        try:
            # Validate strategy before registration
            validation_result = self.validator.validate_strategy(strategy_config)

            if validation_result.is_valid:
                self.strategy_registry[strategy_config.name] = strategy_config
                logger.info(f"Strategy registered: {strategy_config.name}")
                return True
            else:
                logger.error(f"Strategy validation failed: {strategy_config.name}")
                for error in validation_result.errors:
                    logger.error(f"  - {error}")
                return False

        except Exception as e:
            logger.error(f"Error registering strategy: {e}")
            return False

    def is_valid_strategy(
        self, strategy_name: str
    ) -> Tuple[bool, Optional[ValidationResult]]:
        """
        Check if a strategy is valid.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Tuple of (is_valid, validation_result)
        """
        if strategy_name not in self.strategy_registry:
            return False, None

        strategy_config = self.strategy_registry[strategy_name]
        validation_result = self.validator.validate_strategy(strategy_config)

        return validation_result.is_valid, validation_result

    def execute_strategy(
        self, strategy_name: str, execution_params: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Execute a strategy with validation.

        Args:
            strategy_name: Name of the strategy to execute
            execution_params: Additional execution parameters

        Returns:
            Execution ID if successful, None otherwise
        """
        try:
            # Check if strategy exists
            if strategy_name not in self.strategy_registry:
                logger.error(f"Strategy not found: {strategy_name}")
                return None

            strategy_config = self.strategy_registry[strategy_name]

            # Validate strategy before execution
            is_valid, validation_result = self.is_valid_strategy(strategy_name)

            if not is_valid:
                logger.error(f"Strategy validation failed: {strategy_name}")
                return None

            # Create execution record
            execution_id = (
                f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )

            execution = StrategyExecution(
                execution_id=execution_id,
                strategy_name=strategy_name,
                strategy_config=strategy_config,
                status=StrategyStatus.VALIDATED,
                start_time=datetime.now(),
                validation_result=validation_result,
                metadata=execution_params or {},
            )

            self.executions[execution_id] = execution

            # Start execution
            asyncio.create_task(self._execute_strategy_async(execution_id))

            logger.info(f"Strategy execution started: {execution_id}")
            return execution_id

        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return None

    async def _execute_strategy_async(self, execution_id: str):
        """Execute strategy asynchronously."""
        try:
            execution = self.executions[execution_id]
            execution.status = StrategyStatus.RUNNING

            # Simulate strategy execution
            await asyncio.sleep(1)  # Placeholder for actual execution

            # Update execution result
            execution.status = StrategyStatus.COMPLETED
            execution.end_time = datetime.now()
            execution.execution_result = {
                "status": "success",
                "execution_time": (
                    execution.end_time - execution.start_time
                ).total_seconds(),
                "performance_metrics": {
                    "accuracy": 0.85,
                    "profit_loss": 0.02,
                    "sharpe_ratio": 1.2,
                },
            }

            logger.info(f"Strategy execution completed: {execution_id}")

        except Exception as e:
            execution = self.executions[execution_id]
            execution.status = StrategyStatus.FAILED
            execution.end_time = datetime.now()
            execution.error_message = str(e)

            logger.error(f"Strategy execution failed: {execution_id} - {e}")

    def get_execution_status(self, execution_id: str) -> Optional[StrategyExecution]:
        """
        Get execution status.

        Args:
            execution_id: Execution ID

        Returns:
            StrategyExecution object or None
        """
        return self.executions.get(execution_id)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        List all registered strategies.

        Returns:
            List of strategy information
        """
        strategies = []

        for name, config in self.strategy_registry.items():
            is_valid, validation_result = self.is_valid_strategy(name)

            strategy_info = {
                "name": name,
                "type": config.strategy_type,
                "risk_level": config.risk_level,
                "enabled": config.enabled,
                "priority": config.priority,
                "is_valid": is_valid,
                "validation_errors": (
                    validation_result.errors if validation_result else []
                ),
                "tags": config.tags,
            }

            strategies.append(strategy_info)

        return strategies

    def update_strategy(self, strategy_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update strategy configuration.

        Args:
            strategy_name: Name of the strategy
            updates: Updates to apply

        Returns:
            True if update successful
        """
        try:
            if strategy_name not in self.strategy_registry:
                logger.error(f"Strategy not found: {strategy_name}")
                return False

            strategy_config = self.strategy_registry[strategy_name]

            # Apply updates
            for key, value in updates.items():
                if hasattr(strategy_config, key):
                    setattr(strategy_config, key, value)

            # Re-validate
            is_valid, validation_result = self.is_valid_strategy(strategy_name)

            if not is_valid:
                logger.error(
                    f"Strategy validation failed after update: {strategy_name}"
                )
                return False

            logger.info(f"Strategy updated: {strategy_name}")
            return True

        except Exception as e:
            logger.error(f"Error updating strategy: {e}")
            return False

    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            True if removal successful
        """
        try:
            if strategy_name in self.strategy_registry:
                del self.strategy_registry[strategy_name]
                logger.info(f"Strategy removed: {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy not found for removal: {strategy_name}")
                return False

        except Exception as e:
            logger.error(f"Error removing strategy: {e}")
            return False


def create_strategy_dispatcher(
    validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE,
) -> StrategyDispatcher:
    """
    Create a strategy dispatcher instance.

    Args:
        validation_level: Validation level for strategies

    Returns:
        StrategyDispatcher instance
    """
    return StrategyDispatcher(validation_level)
