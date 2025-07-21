"""
Prompt Response Validator - Batch 18
Schema validation for strategy results and prompt responses
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels."""

    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"


@dataclass
class ValidationResult:
    """Result of validation process."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_data: Optional[Dict[str, Any]] = None
    validation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategySchema:
    """Schema definition for strategy results."""

    required_fields: List[str] = field(default_factory=lambda: ["buy", "sell", "price"])
    field_types: Dict[str, type] = field(default_factory=dict)
    field_validators: Dict[str, Callable] = field(default_factory=dict)
    optional_fields: List[str] = field(default_factory=list)


class PromptResponseValidator:
    """
    Enhanced prompt response validator with schema validation.

    Features:
    - Schema validation for strategy results
    - Multiple validation levels
    - Custom field validators
    - Error reporting and correction suggestions
    """

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.NORMAL,
        enable_auto_correction: bool = False,
    ):
        """
        Initialize prompt response validator.

        Args:
            validation_level: Validation strictness level
            enable_auto_correction: Enable automatic correction attempts
        """
        self.validation_level = validation_level
        self.enable_auto_correction = enable_auto_correction

        # Strategy result schema
        self.strategy_schema = self._create_strategy_schema()

        # Validation history
        self.validation_history: List[ValidationResult] = []

        logger.info(
            f"PromptResponseValidator initialized with level: {validation_level.value}"
        )

    def _create_strategy_schema(self) -> StrategySchema:
        """Create default strategy result schema."""
        schema = StrategySchema()

        # Define field types
        schema.field_types = {
            "buy": pd.Series,
            "sell": pd.Series,
            "price": pd.Series,
            "confidence": float,
            "timestamp": str,
            "strategy_name": str,
            "metadata": dict,
        }

        # Define field validators
        schema.field_validators = {
            "buy": self._validate_buy_sell_series,
            "sell": self._validate_buy_sell_series,
            "price": self._validate_price_series,
            "confidence": self._validate_confidence,
            "timestamp": self._validate_timestamp,
            "strategy_name": self._validate_strategy_name,
        }

        # Define optional fields
        schema.optional_fields = [
            "confidence",
            "timestamp",
            "strategy_name",
            "metadata",
        ]

        return schema

    def validate_strategy_result(
        self, data: Dict[str, Any], schema: Optional[StrategySchema] = None
    ) -> ValidationResult:
        """
        Validate strategy result against schema.

        Args:
            data: Strategy result data
            schema: Custom schema (uses default if None)

        Returns:
            ValidationResult with validation details
        """
        start_time = datetime.now()

        if schema is None:
            schema = self.strategy_schema

        result = ValidationResult(is_valid=True)

        try:
            # Check required fields
            missing_fields = self._check_required_fields(data, schema)
            if missing_fields:
                result.errors.extend(
                    [f"Missing required field: {field}" for field in missing_fields]
                )
                result.is_valid = False

            # Validate field types and values
            for field_name, field_value in data.items():
                field_errors = self._validate_field(field_name, field_value, schema)
                result.errors.extend(field_errors)

            # Check for warnings
            warnings = self._check_warnings(data, schema)
            result.warnings.extend(warnings)

            # Auto-correction if enabled
            if self.enable_auto_correction and not result.is_valid:
                corrected_data = self._attempt_correction(data, result.errors)
                if corrected_data:
                    result.validated_data = corrected_data
                    result.warnings.append("Data was auto-corrected")
                else:
                    result.validated_data = data
            else:
                result.validated_data = data

            # Update validation time
            result.validation_time = (datetime.now() - start_time).total_seconds()

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
            result.validation_time = (datetime.now() - start_time).total_seconds()

        # Store in history
        self.validation_history.append(result)

        # Log validation result
        if result.is_valid:
            logger.debug(
                f"Strategy result validation passed with {len(result.warnings)} warnings"
            )
        else:
            logger.warning(f"Strategy result validation failed: {result.errors}")

        return result

    def _check_required_fields(
        self, data: Dict[str, Any], schema: StrategySchema
    ) -> List[str]:
        """Check for missing required fields."""
        missing_fields = []

        for field in schema.required_fields:
            if field not in data:
                missing_fields.append(field)

        return missing_fields

    def _validate_field(
        self, field_name: str, field_value: Any, schema: StrategySchema
    ) -> List[str]:
        """Validate a single field."""
        errors = []

        # Check field type
        if field_name in schema.field_types:
            expected_type = schema.field_types[field_name]
            if not isinstance(field_value, expected_type):
                errors.append(
                    f"Field '{field_name}' has wrong type. Expected {
                        expected_type.__name__}, got {
                        type(field_value).__name__}")

        # Run field-specific validator
        if field_name in schema.field_validators:
            try:
                validator_errors = schema.field_validators[field_name](field_value)
                if validator_errors:
                    errors.extend(
                        [f"Field '{field_name}': {error}" for error in validator_errors]
                    )
            except Exception as e:
                errors.append(f"Field '{field_name}' validation error: {str(e)}")

        return errors

    def _validate_buy_sell_series(self, series: Any) -> List[str]:
        """Validate buy/sell signal series."""
        errors = []

        if not isinstance(series, pd.Series):
            errors.append("Must be a pandas Series")
            return errors

        # Check for boolean values
        if not series.dtype == bool:
            errors.append("Series must contain boolean values")

        # Check for empty series
        if series.empty:
            errors.append("Series cannot be empty")

        # Check for all False values (no signals)
        if not series.any():
            errors.append("Series contains no True values (no signals)")

        return errors

    def _validate_price_series(self, series: Any) -> List[str]:
        """Validate price series."""
        errors = []

        if not isinstance(series, pd.Series):
            errors.append("Must be a pandas Series")
            return errors

        # Check for numeric values
        if not pd.api.types.is_numeric_dtype(series):
            errors.append("Series must contain numeric values")

        # Check for empty series
        if series.empty:
            errors.append("Series cannot be empty")

        # Check for negative values
        if (series < 0).any():
            errors.append("Price series contains negative values")

        # Check for NaN values
        if series.isna().any():
            errors.append("Price series contains NaN values")

        return errors

    def _validate_confidence(self, confidence: Any) -> List[str]:
        """Validate confidence value."""
        errors = []

        if not isinstance(confidence, (int, float)):
            errors.append("Confidence must be numeric")
            return errors

        if not 0 <= confidence <= 1:
            errors.append("Confidence must be between 0 and 1")

        return errors

    def _validate_timestamp(self, timestamp: Any) -> List[str]:
        """Validate timestamp string."""
        errors = []

        if not isinstance(timestamp, str):
            errors.append("Timestamp must be a string")
            return errors

        # Try to parse timestamp
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            errors.append("Timestamp must be in ISO format")

        return errors

    def _validate_strategy_name(self, name: Any) -> List[str]:
        """Validate strategy name."""
        errors = []

        if not isinstance(name, str):
            errors.append("Strategy name must be a string")
            return errors

        if not name.strip():
            errors.append("Strategy name cannot be empty")

        # Check for valid characters
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            errors.append("Strategy name contains invalid characters")

        return errors

    def _check_warnings(
        self, data: Dict[str, Any], schema: StrategySchema
    ) -> List[str]:
        """Check for warnings (non-critical issues)."""
        warnings = []

        # Check for missing optional fields
        for field in schema.optional_fields:
            if field not in data:
                warnings.append(f"Optional field '{field}' is missing")

        # Check data consistency
        if "buy" in data and "sell" in data:
            buy_series = data["buy"]
            sell_series = data["sell"]

            if isinstance(buy_series, pd.Series) and isinstance(sell_series, pd.Series):
                # Check for overlapping signals
                if buy_series.any() and sell_series.any():
                    overlap = buy_series & sell_series
                    if overlap.any():
                        warnings.append("Buy and sell signals overlap")

        # Check confidence level
        if "confidence" in data:
            confidence = data["confidence"]
            if isinstance(confidence, (int, float)):
                if confidence < 0.3:
                    warnings.append("Low confidence level (< 0.3)")
                elif confidence > 0.95:
                    warnings.append("Very high confidence level (> 0.95)")

        return warnings

    def _attempt_correction(
        self, data: Dict[str, Any], errors: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to correct validation errors."""
        corrected_data = data.copy()

        for error in errors:
            if "Missing required field" in error:
                field_name = error.split(": ")[1]
                corrected_data[field_name] = self._create_default_value(field_name)

            elif "wrong type" in error:
                field_name = error.split("'")[1]
                corrected_data[field_name] = self._convert_field_type(
                    corrected_data[field_name], field_name
                )

        return corrected_data

    def _create_default_value(self, field_name: str) -> Any:
        """Create default value for missing field."""
        if field_name in ["buy", "sell"]:
            return pd.Series([False] * 10)  # Default empty signal series
        elif field_name == "price":
            return pd.Series([100.0] * 10)  # Default price series
        elif field_name == "confidence":
            return 0.5
        elif field_name == "timestamp":
            return datetime.now().isoformat()
        elif field_name == "strategy_name":
            return "unknown_strategy"
        elif field_name == "metadata":
            return {}
        else:
            return None

    def _convert_field_type(self, value: Any, field_name: str) -> Any:
        """Convert field to correct type."""
        try:
            if field_name in ["buy", "sell"]:
                if isinstance(value, (list, np.ndarray)):
                    return pd.Series(value, dtype=bool)
                elif isinstance(value, pd.Series):
                    return value.astype(bool)
                else:
                    return pd.Series([bool(value)])

            elif field_name == "price":
                if isinstance(value, (list, np.ndarray)):
                    return pd.Series(value, dtype=float)
                elif isinstance(value, pd.Series):
                    return value.astype(float)
                else:
                    return pd.Series([float(value)])

            elif field_name == "confidence":
                return float(value)

            elif field_name == "timestamp":
                return str(value)

            elif field_name == "strategy_name":
                return str(value)

            else:
                return value

        except Exception:
            return self._create_default_value(field_name)

    def validate_prompt_response(
        self, response: str, expected_format: str = "json"
    ) -> ValidationResult:
        """
        Validate prompt response format.

        Args:
            response: Raw prompt response
            expected_format: Expected response format

        Returns:
            ValidationResult
        """
        start_time = datetime.now()
        result = ValidationResult(is_valid=True)

        try:
            if expected_format.lower() == "json":
                # Try to parse JSON
                try:
                    parsed_data = json.loads(response)
                    result.validated_data = parsed_data
                except json.JSONDecodeError as e:
                    result.is_valid = False
                    result.errors.append(f"Invalid JSON format: {str(e)}")

            elif expected_format.lower() == "strategy":
                # Validate strategy-specific format
                strategy_errors = self._validate_strategy_format(response)
                if strategy_errors:
                    result.is_valid = False
                    result.errors.extend(strategy_errors)
                else:
                    result.validated_data = response

            else:
                # Basic string validation
                if not isinstance(response, str) or not response.strip():
                    result.is_valid = False
                    result.errors.append("Response must be a non-empty string")
                else:
                    result.validated_data = response

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")

        result.validation_time = (datetime.now() - start_time).total_seconds()
        self.validation_history.append(result)

        return result

    def _validate_strategy_format(self, response: str) -> List[str]:
        """Validate strategy-specific response format."""
        errors = []

        # Check for required keywords
        required_keywords = ["buy", "sell", "price"]
        for keyword in required_keywords:
            if keyword not in response.lower():
                errors.append(f"Missing required keyword: {keyword}")

        # Check for signal indicators
        signal_indicators = ["signal", "action", "recommendation"]
        if not any(indicator in response.lower() for indicator in signal_indicators):
            errors.append("Missing signal/action indicators")

        return errors

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        if not self.validation_history:
            return {}

        total_validations = len(self.validation_history)
        successful_validations = sum(
            1 for result in self.validation_history if result.is_valid
        )
        failed_validations = total_validations - successful_validations

        # Error frequency analysis
        error_counts = {}
        for result in self.validation_history:
            for error in result.errors:
                error_type = error.split(":")[0] if ":" in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        summary = {
            "total_validations": total_validations,
            "success_rate": successful_validations / total_validations,
            "failed_validations": failed_validations,
            "avg_validation_time": np.mean(
                [r.validation_time for r in self.validation_history]
            ),
            "common_errors": dict(
                sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "validation_level": self.validation_level.value,
            "auto_correction_enabled": self.enable_auto_correction,
        }

        return summary

    def set_validation_level(self, level: ValidationLevel):
        """Set validation level."""
        self.validation_level = level
        logger.info(f"Validation level set to: {level.value}")

    def enable_auto_correction(self, enable: bool = True):
        """Enable or disable auto-correction."""
        self.enable_auto_correction = enable
        logger.info(f"Auto-correction {'enabled' if enable else 'disabled'}")


def create_prompt_response_validator(
    validation_level: ValidationLevel = ValidationLevel.NORMAL,
) -> PromptResponseValidator:
    """Factory function to create prompt response validator."""
    return PromptResponseValidator(validation_level=validation_level)
