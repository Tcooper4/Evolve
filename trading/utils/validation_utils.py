"""Validation utilities for data and parameters.

This module provides utilities for validating data, parameters, and configurations
with comprehensive error checking and reporting.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

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
        allow_duplicates: bool = False):
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
        issues = []
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check row count
        if len(df) < self.min_rows:
            issues.append(f"Insufficient rows: {len(df)} < {self.min_rows}")
        if self.max_rows and len(df) > self.max_rows:
            issues.append(f"Too many rows: {len(df)} > {self.max_rows}")
        
        # Check missing values
        if not self.allow_missing:
            missing = df.isnull().sum()
            if missing.any():
                issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
        
        # Check duplicates
        if not self.allow_duplicates:
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate rows")
        
        # Check column types
        for col in self.numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"Column {col} is not numeric")
        
        for col in self.categorical_columns:
            if col in df.columns:
                if not pd.api.types.is_categorical_dtype(df[col]):
                    issues.append(f"Column {col} is not categorical")
        
        for col in self.datetime_columns:
            if col in df.columns:
                if not pd.api.types.is_datetime64_dtype(df[col]):
                    issues.append(f"Column {col} is not datetime")
        
        return len(issues) == 0, issues

class ParameterValidator:
    """Validator for parameter values and types."""
    
    def __init__(self, param_schema: Dict[str, Dict[str, Any]]):
        """Initialize parameter validator.
        
        Args:
            param_schema: Parameter schema dictionary
        """
        self.param_schema = param_schema
    
    def validate_parameters(
        self,
        params: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate parameter values.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        for name, schema in self.param_schema.items():
            # Check required parameters
            if schema.get('required', False) and name not in params:
                issues.append(f"Missing required parameter: {name}")
                continue
            
            if name not in params:
                continue
            
            value = params[name]
            
            # Check type
            expected_type = schema.get('type')
            if expected_type and not isinstance(value, expected_type):
                issues.append(
                    f"Parameter {name} has wrong type: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )
            
            # Check range
            if 'min' in schema and value < schema['min']:
                issues.append(
                    f"Parameter {name} below minimum: {value} < {schema['min']}"
                )
            if 'max' in schema and value > schema['max']:
                issues.append(
                    f"Parameter {name} above maximum: {value} > {schema['max']}"
                )
            
            # Check choices
            if 'choices' in schema and value not in schema['choices']:
                issues.append(
                    f"Parameter {name} not in choices: {value} not in {schema['choices']}"
                )
            
            # Check pattern
            if 'pattern' in schema and not re.match(schema['pattern'], str(value)):
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
        value_validators: Optional[Dict[str, Callable[[Any], bool]]] = None):
        """Initialize config validator.
        
        Args:
            required_sections: List of required section names
            required_keys: Dictionary of section names to required keys
            value_validators: Dictionary of key names to validation functions
        """
        self.required_sections = required_sections or []
        self.required_keys = required_keys or {}
        self.value_validators = value_validators or {}
    
    def validate_config(
        self,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check required sections
        missing_sections = [
            section for section in self.required_sections
            if section not in config
        ]
        if missing_sections:
            issues.append(f"Missing required sections: {missing_sections}")
        
        # Check required keys in each section
        for section, keys in self.required_keys.items():
            if section not in config:
                continue
            
            missing_keys = [
                key for key in keys
                if key not in config[section]
            ]
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
        self,
        file_path: Union[str, Path]
    ) -> Tuple[bool, List[str]]:
        """Validate configuration file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        try:
            with open(file_path) as f:
                if str(file_path).endswith('.json'):
                    config = json.load(f)
                elif str(file_path).endswith(('.yaml', '.yml')):
                    import yaml
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
            
            return self.validate_config(config)
        except Exception as e:
            return False, [f"Error loading config file: {e}"]

def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None
) -> bool:
    """Validate numeric value is in range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Whether value is valid
    """
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True

def validate_string_length(
    value: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> bool:
    """Validate string length is in range.
    
    Args:
        value: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        Whether string is valid
    """
    if min_length is not None and len(value) < min_length:
        return False
    if max_length is not None and len(value) > max_length:
        return False
    return True

def validate_datetime_range(
    value: datetime,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None
) -> bool:
    """Validate datetime is in range.
    
    Args:
        value: Datetime to validate
        min_date: Minimum allowed date
        max_date: Maximum allowed date
        
    Returns:
        Whether datetime is valid
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