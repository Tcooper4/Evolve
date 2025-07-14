"""Validation utilities for data and parameters.

This module provides utilities for validating data, parameters, and configurations
with comprehensive error checking and reporting.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for validation errors."""


def ensure_array_compatible(data: Any, name: str = "data") -> np.ndarray:
    """Ensure data is numpy or pandas compatible and convert to numpy array.

    Args:
        data: Input data to convert
        name: Name of the data for error messages

    Returns:
        Numpy array

    Raises:
        ValidationError: If data cannot be converted to numpy array
    """
    try:
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.Series):
            return data.to_numpy()
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                return data.iloc[:, 0].to_numpy()
            else:
                raise ValidationError(
                    f"{name} is a DataFrame with multiple columns, cannot convert to 1D array"
                )
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        elif hasattr(data, "__array__"):
            return np.array(data)
        else:
            raise ValidationError(
                f"{name} is not numpy or pandas compatible: {type(data)}"
            )
    except Exception as e:
        raise ValidationError(f"Failed to convert {name} to numpy array: {str(e)}")


def validate_array_shape(
    array: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "array",
) -> bool:
    """Validate array shape integrity.

    Args:
        array: Numpy array to validate
        expected_shape: Expected shape tuple
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        name: Name of the array for error messages

    Returns:
        Whether shape is valid

    Raises:
        ValidationError: If shape validation fails
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(f"{name} is not a numpy array: {type(array)}")

    # Check for NaN or infinite values
    if np.any(np.isnan(array)):
        raise ValidationError(f"{name} contains NaN values")

    if np.any(np.isinf(array)):
        raise ValidationError(f"{name} contains infinite values")

    # Check dimensions
    if min_dims is not None and array.ndim < min_dims:
        raise ValidationError(
            f"{name} has {array.ndim} dimensions, expected at least {min_dims}"
        )

    if max_dims is not None and array.ndim > max_dims:
        raise ValidationError(
            f"{name} has {array.ndim} dimensions, expected at most {max_dims}"
        )

    # Check expected shape
    if expected_shape is not None:
        if array.shape != expected_shape:
            raise ValidationError(
                f"{name} has shape {array.shape}, expected {expected_shape}"
            )

    return True


def validate_dataframe_integrity(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    datetime_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    max_rows: Optional[int] = None,
    allow_missing: bool = False,
    allow_duplicates: bool = False,
    check_shape: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate DataFrame integrity with comprehensive checks.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
        datetime_columns: List of datetime column names
        min_rows: Minimum number of rows required
        max_rows: Maximum number of rows allowed
        allow_missing: Whether to allow missing values
        allow_duplicates: Whether to allow duplicate rows
        check_shape: Whether to check shape integrity

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Ensure DataFrame is pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        issues.append(f"Input is not a pandas DataFrame: {type(df)}")
        return False, issues

    # Check shape integrity
    if check_shape:
        if df.shape[0] == 0:
            issues.append("DataFrame is empty (0 rows)")
        if df.shape[1] == 0:
            issues.append("DataFrame has no columns")

    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

    # Check row count
    if len(df) < min_rows:
        issues.append(f"Insufficient rows: {len(df)} < {min_rows}")
    if max_rows and len(df) > max_rows:
        issues.append(f"Too many rows: {len(df)} > {max_rows}")

    # Check missing values
    if not allow_missing:
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")

    # Check duplicates
    if not allow_duplicates:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")

    # Check column types and convert to numpy arrays for validation
    for col in numeric_columns or []:
        if col in df.columns:
            try:
                array = ensure_array_compatible(df[col], f"column '{col}'")
                validate_array_shape(
                    array, min_dims=1, max_dims=1, name=f"column '{col}'"
                )

                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"Column {col} is not numeric")
            except ValidationError as e:
                issues.append(f"Column {col} validation failed: {str(e)}")

    for col in categorical_columns or []:
        if col in df.columns:
            if not pd.api.types.is_categorical_dtype(df[col]):
                issues.append(f"Column {col} is not categorical")

    for col in datetime_columns or []:
        if col in df.columns:
            if not pd.api.types.is_datetime64_dtype(df[col]):
                issues.append(f"Column {col} is not datetime")

    return len(issues) == 0, issues


class DataValidator:
    """Validator for data quality and structure."""

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        max_rows: Optional[int] = None,
        allow_missing: bool = False,
        allow_duplicates: bool = False,
    ):
        """Initialize data validator.

        Args:
            required_columns: List of required column names
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names
            datetime_columns: List of datetime column names
            min_rows: Minimum number of rows required
            max_rows: Maximum number of rows allowed
            allow_missing: Whether to allow missing values
            allow_duplicates: Whether to allow duplicate rows
        """
        self.required_columns = required_columns or []
        self.numeric_columns = numeric_columns or []
        self.categorical_columns = categorical_columns or []
        self.datetime_columns = datetime_columns or []
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.allow_missing = allow_missing
        self.allow_duplicates = allow_duplicates

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate a DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        return validate_dataframe_integrity(
            df,
            required_columns=self.required_columns,
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
            datetime_columns=self.datetime_columns,
            min_rows=self.min_rows,
            max_rows=self.max_rows,
            allow_missing=self.allow_missing,
            allow_duplicates=self.allow_duplicates,
        )

    def validate_array(self, data: Any, name: str = "data") -> np.ndarray:
        """Validate and convert data to numpy array.

        Args:
            data: Input data to validate
            name: Name of the data for error messages

        Returns:
            Validated numpy array

        Raises:
            ValidationError: If validation fails
        """
        return ensure_array_compatible(data, name)

    def validate_array_shape(
        self,
        array: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None,
        name: str = "array",
    ) -> bool:
        """Validate array shape integrity.

        Args:
            array: Numpy array to validate
            expected_shape: Expected shape tuple
            min_dims: Minimum number of dimensions
            max_dims: Maximum number of dimensions
            name: Name of the array for error messages

        Returns:
            Whether shape is valid

        Raises:
            ValidationError: If shape validation fails
        """
        return validate_array_shape(array, expected_shape, min_dims, max_dims, name)


class ParameterValidator:
    """Validator for parameter values and types."""

    def __init__(self, param_schema: Dict[str, Dict[str, Any]]):
        """Initialize parameter validator.

        Args:
            param_schema: Parameter schema dictionary
        """
        self.param_schema = param_schema

    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate parameter values.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        for name, schema in self.param_schema.items():
            # Check required parameters
            if schema.get("required", False) and name not in params:
                issues.append(f"Missing required parameter: {name}")
                continue

            if name not in params:
                continue

            value = params[name]

            # Check type
            expected_type = schema.get("type")
            if expected_type and not isinstance(value, expected_type):
                issues.append(
                    f"Parameter {name} has wrong type: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )

            # Check range
            if "min" in schema and value < schema["min"]:
                issues.append(
                    f"Parameter {name} below minimum: {value} < {schema['min']}"
                )
            if "max" in schema and value > schema["max"]:
                issues.append(
                    f"Parameter {name} above maximum: {value} > {schema['max']}"
                )

            # Check choices
            if "choices" in schema and value not in schema["choices"]:
                issues.append(
                    f"Parameter {name} not in choices: {value} not in {schema['choices']}"
                )

            # Check pattern
            if "pattern" in schema and not re.match(schema["pattern"], str(value)):
                issues.append(
                    f"Parameter {name} does not match pattern: {schema['pattern']}"
                )

        return len(issues) == 0, issues


class ConfigValidator:
    """Validator for configuration files."""

    def __init__(
        self,
        required_sections: Optional[List[str]] = None,
        required_keys: Optional[Dict[str, List[str]]] = None,
        value_validators: Optional[Dict[str, Callable[[Any], bool]]] = None,
    ):
        """Initialize config validator.

        Args:
            required_sections: List of required section names
            required_keys: Dictionary of section names to required keys
            value_validators: Dictionary of key names to validation functions
        """
        self.required_sections = required_sections or []
        self.required_keys = required_keys or {}
        self.value_validators = value_validators or {}

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration dictionary.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check required sections
        missing_sections = [
            section for section in self.required_sections if section not in config
        ]
        if missing_sections:
            issues.append(f"Missing required sections: {missing_sections}")

        # Check required keys in each section
        for section, keys in self.required_keys.items():
            if section not in config:
                continue

            missing_keys = [key for key in keys if key not in config[section]]
            if missing_keys:
                issues.append(
                    f"Missing required keys in section {section}: {missing_keys}"
                )

        # Validate values
        for key, validator in self.value_validators.items():
            if key in config:
                if not validator(config[key]):
                    issues.append(f"Invalid value for key {key}")

        return len(issues) == 0, issues

    def validate_config_file(
        self, file_path: Union[str, Path]
    ) -> Tuple[bool, List[str]]:
        """Validate configuration file.

        Args:
            file_path: Path to configuration file

        Returns:
            Tuple of (is_valid, list of issues)
        """
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
            return self.validate_config(config)
        except Exception as e:
            return False, [f"Failed to load config file: {str(e)}"]


def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
) -> bool:
    """Validate numeric value is within range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Whether value is within range
    """
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def validate_string_length(
    value: str, min_length: Optional[int] = None, max_length: Optional[int] = None
) -> bool:
    """Validate string length is within range.

    Args:
        value: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Whether string length is within range
    """
    if min_length is not None and len(value) < min_length:
        return False
    if max_length is not None and len(value) > max_length:
        return False
    return True


def validate_datetime_range(
    value: datetime,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
) -> bool:
    """Validate datetime is within range.

    Args:
        value: Datetime to validate
        min_date: Minimum allowed date
        max_date: Maximum allowed date

    Returns:
        Whether datetime is within range
    """
    if min_date is not None and value < min_date:
        return False
    if max_date is not None and value > max_date:
        return False
    return True


def validate_file_exists(file_path: Union[str, Path]) -> bool:
    """Validate file exists.

    Args:
        file_path: Path to file

    Returns:
        Whether file exists
    """
    return Path(file_path).exists()


def validate_directory_exists(dir_path: Union[str, Path]) -> bool:
    """Validate directory exists.

    Args:
        dir_path: Path to directory

    Returns:
        Whether directory exists
    """
    return Path(dir_path).is_dir()


def validate_array_compatibility(data: Any, name: str = "data") -> np.ndarray:
    """Validate and convert data to numpy array with comprehensive checks.

    Args:
        data: Input data to validate
        name: Name of the data for error messages

    Returns:
        Validated numpy array

    Raises:
        ValidationError: If validation fails
    """
    return ensure_array_compatible(data, name)


def validate_shape_integrity(
    array: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "array",
) -> bool:
    """Validate array shape integrity with comprehensive checks.

    Args:
        array: Numpy array to validate
        expected_shape: Expected shape tuple
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        name: Name of the array for error messages

    Returns:
        Whether shape is valid

    Raises:
        ValidationError: If shape validation fails
    """
    return validate_array_shape(array, expected_shape, min_dims, max_dims, name)
