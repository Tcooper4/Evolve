"""
Common Helper Functions

This module consolidates shared utility functions from across the codebase
to provide a single source of truth for common operations.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import uuid
import re

logger = logging.getLogger(__name__)

# ============================================================================
# DATA VALIDATION HELPERS
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    try:
        if df is None or df.empty:
            return {"valid": False, "error": "DataFrame is empty or None"}
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {"valid": False, "error": f"Missing required columns: {missing_columns}"}
        
        # Check for all NaN rows
        if df.isnull().all().all():
            return {"valid": False, "error": "All rows contain only NaN values"}
        
        return {"valid": True, "message": "DataFrame validation passed"}
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

def validate_numeric_column(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Validate that a column contains numeric data.
    
    Args:
        df: DataFrame to check
        column: Column name to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        if column not in df.columns:
            return {"valid": False, "error": f"Column '{column}' not found"}
        
        # Try to convert to numeric
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        non_numeric_count = numeric_series.isnull().sum()
        
        if non_numeric_count > 0:
            return {
                "valid": False, 
                "error": f"Column '{column}' contains {non_numeric_count} non-numeric values"
            }
        
        return {"valid": True, "message": f"Column '{column}' is numeric"}
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

def validate_date_column(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Validate that a column contains date data.
    
    Args:
        df: DataFrame to check
        column: Column name to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        if column not in df.columns:
            return {"valid": False, "error": f"Column '{column}' not found"}
        
        # Try to convert to datetime
        date_series = pd.to_datetime(df[column], errors='coerce')
        non_date_count = date_series.isnull().sum()
        
        if non_date_count > 0:
            return {
                "valid": False, 
                "error": f"Column '{column}' contains {non_date_count} non-date values"
            }
        
        return {"valid": True, "message": f"Column '{column}' contains valid dates"}
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

# ============================================================================
# DATA PROCESSING HELPERS
# ============================================================================

def clean_dataframe(df: pd.DataFrame, remove_duplicates: bool = True, fill_method: str = 'ffill') -> pd.DataFrame:
    """Clean DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: DataFrame to clean
        remove_duplicates: Whether to remove duplicate rows
        fill_method: Method to fill missing values ('ffill', 'bfill', 'drop')
        
    Returns:
        Cleaned DataFrame
    """
    try:
        cleaned_df = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        if fill_method == 'drop':
            cleaned_df = cleaned_df.dropna()
        elif fill_method in ['ffill', 'bfill']:
            cleaned_df = cleaned_df.fillna(method=fill_method)
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning DataFrame: {e}")
        return df

def normalize_dataframe(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """Normalize DataFrame columns.
    
    Args:
        df: DataFrame to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized DataFrame
    """
    try:
        normalized_df = df.copy()
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if method == 'minmax':
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val > min_val:
                    normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[column].mean()
                std_val = df[column].std()
                if std_val > 0:
                    normalized_df[column] = (df[column] - mean_val) / std_val
            elif method == 'robust':
                median_val = df[column].median()
                mad_val = df[column].mad()
                if mad_val > 0:
                    normalized_df[column] = (df[column] - median_val) / mad_val
        
        return normalized_df
        
    except Exception as e:
        logger.error(f"Error normalizing DataFrame: {e}")
        return df

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculate returns from price series.
    
    Args:
        prices: Price series
        method: Return calculation method ('simple', 'log')
        
    Returns:
        Returns series
    """
    try:
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown return method: {method}")
        
        return returns
        
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        return pd.Series(dtype=float)

# ============================================================================
# FILE AND PATH HELPERS
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> bool:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Whether directory creation was successful
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False

def safe_file_path(base_path: str, filename: str) -> str:
    """Create a safe file path by sanitizing the filename.
    
    Args:
        base_path: Base directory path
        filename: Filename to sanitize
        
    Returns:
        Safe file path
    """
    try:
        # Remove or replace unsafe characters
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_filename = safe_filename.strip()
        
        # Ensure filename is not empty
        if not safe_filename:
            safe_filename = f"file_{uuid.uuid4().hex[:8]}"
        
        return os.path.join(base_path, safe_filename)
        
    except Exception as e:
        logger.error(f"Error creating safe file path: {e}")
        return os.path.join(base_path, f"file_{uuid.uuid4().hex[:8]}")

def get_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        File hash as hex string
    """
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return ""

# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        Whether save was successful
    """
    try:
        ensure_directory(os.path.dirname(config_path))
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        return False

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    try:
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
        
    except Exception as e:
        logger.error(f"Error merging configs: {e}")
        return base_config

# ============================================================================
# TIME AND DATE HELPERS
# ============================================================================

def parse_date_range(date_range: str) -> Tuple[datetime, datetime]:
    """Parse date range string into start and end dates.
    
    Args:
        date_range: Date range string (e.g., "7d", "30d", "1y")
        
    Returns:
        Tuple of (start_date, end_date)
    """
    try:
        end_date = datetime.now()
        
        if date_range.endswith('d'):
            days = int(date_range[:-1])
            start_date = end_date - timedelta(days=days)
        elif date_range.endswith('w'):
            weeks = int(date_range[:-1])
            start_date = end_date - timedelta(weeks=weeks)
        elif date_range.endswith('m'):
            months = int(date_range[:-1])
            start_date = end_date - timedelta(days=months * 30)
        elif date_range.endswith('y'):
            years = int(date_range[:-1])
            start_date = end_date - timedelta(days=years * 365)
        else:
            raise ValueError(f"Invalid date range format: {date_range}")
        
        return start_date, end_date
        
    except Exception as e:
        logger.error(f"Error parsing date range {date_range}: {e}")
        # Return default 30-day range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return start_date, end_date

def format_timestamp(timestamp: Union[datetime, str, float]) -> str:
    """Format timestamp to ISO string.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string
    """
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, float):
            timestamp = datetime.fromtimestamp(timestamp)
        
        return timestamp.isoformat()
        
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return datetime.now().isoformat()

# ============================================================================
# LOGGING HELPERS
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Set up a logger with consistent configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    try:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            ensure_directory(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logger {name}: {e}")
        return logging.getLogger(name)

def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None) -> None:
    """Log function call for debugging.
    
    Args:
        func_name: Name of the function
        args: Function arguments
        kwargs: Function keyword arguments
    """
    try:
        args_str = str(args) if args else "()"
        kwargs_str = str(kwargs) if kwargs else "{}"
        logger.debug(f"Calling {func_name} with args={args_str}, kwargs={kwargs_str}")
    except Exception as e:
        logger.error(f"Error logging function call: {e}")

# ============================================================================
# ERROR HANDLING HELPERS
# ============================================================================

def safe_execute(func: callable, *args, default_return=None, **kwargs) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value on error
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        return default_return

def retry_on_error(func: callable, max_retries: int = 3, delay: float = 1.0, *args, **kwargs) -> Any:
    """Retry function execution on error.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or raises last exception
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                import time
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
    
    raise last_exception

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    try:
        unique_id = uuid.uuid4().hex
        return f"{prefix}{unique_id}" if prefix else unique_id
    except Exception as e:
        logger.error(f"Error generating ID: {e}")
        return f"{prefix}{int(datetime.now().timestamp())}"

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    try:
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    except Exception as e:
        logger.error(f"Error chunking list: {e}")
        return [lst]

def flatten_list(nested_list: List[Any]) -> List[Any]:
    """Flatten a nested list.
    
    Args:
        nested_list: Nested list to flatten
        
    Returns:
        Flattened list
    """
    try:
        flattened = []
        for item in nested_list:
            if isinstance(item, list):
                flattened.extend(flatten_list(item))
            else:
                flattened.append(item)
        return flattened
    except Exception as e:
        logger.error(f"Error flattening list: {e}")
        return nested_list

# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def safe_mkdir(path: Union[str, Path]) -> bool:
    """Safely create directory with error handling.
    
    Args:
        path: Directory path to create
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def log_to_file(message: str, log_file: str, level: str = "INFO") -> None:
    """Log message to a specific file.
    
    Args:
        message: Message to log
        log_file: Path to log file
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    try:
        safe_mkdir(os.path.dirname(log_file))
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {level}: {message}\n")
    except Exception as e:
        logger.error(f"Failed to log to file {log_file}: {e}")

def slugify(text: str) -> str:
    """Convert text to URL-friendly slug.
    
    Args:
        text: Text to convert
        
    Returns:
        URL-friendly slug
    """
    try:
        # Remove special characters and convert to lowercase
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        # Replace spaces and hyphens with single hyphens
        slug = re.sub(r'[-\s]+', '-', slug)
        # Remove leading/trailing hyphens
        return slug.strip('-')
    except Exception as e:
        logger.error(f"Error creating slug: {e}")
        return str(uuid.uuid4())[:8]

def timestamp() -> str:
    """Get current timestamp in ISO format.
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.now().isoformat()

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    try:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    except Exception as e:
        logger.error(f"Error formatting bytes: {e}")
        return "0 B"

def hash_file(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA-256 hash string
    """
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return "" 