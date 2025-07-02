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
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import uuid
import re
import functools
import time

logger = logging.getLogger(__name__)

# ============================================================================
# DATA VALIDATION HELPERS
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content.
    
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
    """
    Validate that a column contains numeric data.
    
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
    """
    Validate that a column contains date data.
    
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

def validate_data(data: Any, data_type: str = "any") -> Dict[str, Any]:
    """
    Generic data validation function.
    
    Args:
        data: Data to validate
        data_type: Type of data to validate ("any", "numeric", "date", "string")
        
    Returns:
        Dictionary with validation results
    """
    try:
        if data is None:
            return {"valid": False, "error": "Data is None"}
        
        if data_type == "numeric":
            if not isinstance(data, (int, float, np.number)):
                return {"valid": False, "error": "Data is not numeric"}
        elif data_type == "date":
            if not isinstance(data, (datetime, pd.Timestamp)):
                return {"valid": False, "error": "Data is not a date"}
        elif data_type == "string":
            if not isinstance(data, str):
                return {"valid": False, "error": "Data is not a string"}
        
        return {"valid": True, "message": f"Data validation passed for type: {data_type}"}
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

# ============================================================================
# DATA PROCESSING HELPERS
# ============================================================================

def clean_dataframe(df: pd.DataFrame, remove_duplicates: bool = True, fill_method: str = 'ffill') -> pd.DataFrame:
    """
    Clean DataFrame by removing duplicates and handling missing values.
    
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
    """
    Normalize DataFrame columns.
    
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
    """
    Calculate returns from price series.
    
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
# FILE SYSTEM HELPERS
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> bool:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False

def safe_file_path(base_path: str, filename: str) -> str:
    """
    Create a safe file path by ensuring directory exists.
    
    Args:
        base_path: Base directory path
        filename: Filename
        
    Returns:
        Safe file path
    """
    try:
        full_path = Path(base_path) / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return str(full_path)
    except Exception as e:
        logger.error(f"Error creating safe file path: {e}")
        return os.path.join(base_path, filename)

def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
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
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if saved successfully
    """
    try:
        ensure_directory(str(Path(config_path).parent))
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        return False

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    try:
        merged = base_config.copy()
        
        def deep_merge(d1: Dict[str, Any], d2: Dict[str, Any]) -> None:
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value
        
        deep_merge(merged, override_config)
        return merged
        
    except Exception as e:
        logger.error(f"Error merging configs: {e}")
        return base_config

# ============================================================================
# DATE/TIME HELPERS
# ============================================================================

def parse_date_range(date_range: str) -> Tuple[datetime, datetime]:
    """
    Parse date range string into start and end dates.
    
    Args:
        date_range: Date range string (e.g., "2023-01-01:2023-12-31")
        
    Returns:
        Tuple of (start_date, end_date)
    """
    try:
        if ':' in date_range:
            start_str, end_str = date_range.split(':', 1)
            start_date = pd.to_datetime(start_str.strip())
            end_date = pd.to_datetime(end_str.strip())
        else:
            # Single date
            start_date = pd.to_datetime(date_range.strip())
            end_date = start_date + timedelta(days=1)
        
        return start_date, end_date
        
    except Exception as e:
        logger.error(f"Error parsing date range '{date_range}': {e}")
        # Return default range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return start_date, end_date

def format_timestamp(timestamp: Union[datetime, str, float]) -> str:
    """
    Format timestamp to ISO string.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string
    """
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif isinstance(timestamp, float):
            timestamp = pd.to_datetime(timestamp, unit='s')
        
        return timestamp.isoformat()
        
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return datetime.now().isoformat()

def get_timestamp() -> str:
    """
    Get current timestamp as ISO string.
    
    Returns:
        Current timestamp string
    """
    return datetime.now().isoformat()

# ============================================================================
# LOGGING HELPERS
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        try:
            ensure_directory(str(Path(log_file).parent))
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Error setting up file handler: {e}")
    
    return logger

def log_function_call(func_name: str, args: Optional[tuple] = None, kwargs: Optional[dict] = None) -> None:
    """
    Log function call details.
    
    Args:
        func_name: Name of the function
        args: Function arguments
        kwargs: Function keyword arguments
    """
    try:
        args_str = str(args) if args else "()"
        kwargs_str = str(kwargs) if kwargs else "{}"
        logger.debug(f"Function call: {func_name}{args_str} {kwargs_str}")
    except Exception as e:
        logger.error(f"Error logging function call: {e}")

# ============================================================================
# EXECUTION HELPERS
# ============================================================================

def safe_execute(func: Callable, *args, default_return: Any = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
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

def retry_on_error(func: Callable, max_retries: int = 3, delay: float = 1.0, *args, **kwargs) -> Any:
    """
    Retry function execution on error.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
    
    raise last_exception

# ============================================================================
# UTILITY HELPERS
# ============================================================================

def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    return f"{prefix}{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:8]

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: Nested list to flatten
        
    Returns:
        Flattened list
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug.
    
    Args:
        text: Text to slugify
        
    Returns:
        Slugified text
    """
    # Convert to lowercase and replace spaces with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')

def normalize_indicator_name(name: str) -> str:
    """
    Normalize indicator name for consistent formatting.
    
    Args:
        name: Indicator name to normalize
        
    Returns:
        Normalized indicator name
    """
    try:
        # Convert to lowercase
        normalized = name.lower()
        
        # Replace common variations
        replacements = {
            'rsi': 'RSI',
            'macd': 'MACD',
            'bollinger': 'BB',
            'bb': 'BB',
            'moving_average': 'MA',
            'ma': 'MA',
            'exponential_moving_average': 'EMA',
            'ema': 'EMA',
            'simple_moving_average': 'SMA',
            'sma': 'SMA',
            'stochastic': 'STOCH',
            'stoch': 'STOCH',
            'williams_r': 'WILLR',
            'willr': 'WILLR',
            'average_true_range': 'ATR',
            'atr': 'ATR',
            'commodity_channel_index': 'CCI',
            'cci': 'CCI',
            'money_flow_index': 'MFI',
            'mfi': 'MFI',
            'on_balance_volume': 'OBV',
            'obv': 'OBV',
            'price_rate_of_change': 'ROC',
            'roc': 'ROC',
            'momentum': 'MOM',
            'mom': 'MOM',
            'parabolic_sar': 'PSAR',
            'psar': 'PSAR',
            'trix': 'TRIX',
            'ultimate_oscillator': 'UO',
            'uo': 'UO',
            'chaikin_money_flow': 'CMF',
            'cmf': 'CMF',
            'volume_price_trend': 'VPT',
            'vpt': 'VPT',
            'accumulation_distribution': 'AD',
            'ad': 'AD',
            'chaikin_oscillator': 'CO',
            'co': 'CO',
            'ease_of_movement': 'EOM',
            'eom': 'EOM',
            'force_index': 'FI',
            'fi': 'FI',
            'mass_index': 'MI',
            'mi': 'MI',
            'negative_volume_index': 'NVI',
            'nvi': 'NVI',
            'positive_volume_index': 'PVI',
            'pvi': 'PVI',
            'volume_oscillator': 'VO',
            'vo': 'VO',
            'volume_rate_of_change': 'VROC',
            'vroc': 'VROC',
            'volume_weighted_average_price': 'VWAP',
            'vwap': 'VWAP',
            'volume_weighted_moving_average': 'VWMA',
            'vwma': 'VWMA',
            'kaufman_adaptive_moving_average': 'KAMA',
            'kama': 'KAMA',
            'mESA': 'MESA',
            'mesa': 'MESA',
            'aroon': 'AROON',
            'aroon_oscillator': 'AROON_OSC',
            'aroon_osc': 'AROON_OSC',
            'balance_of_power': 'BOP',
            'bop': 'BOP',
            'center_of_gravity': 'COG',
            'cog': 'COG',
            'center_of_gravity_oscillator': 'COG_OSC',
            'cog_osc': 'COG_OSC',
            'detrended_price_oscillator': 'DPO',
            'dpo': 'DPO',
            'efficiency_ratio': 'ER',
            'er': 'ER',
            'fisher_transform': 'FISHER',
            'fisher': 'FISHER',
            'fisher_inverse': 'FISHER_INV',
            'fisher_inv': 'FISHER_INV',
            'high_low_index': 'HLI',
            'hli': 'HLI',
            'high_low_ratio': 'HLR',
            'hlr': 'HLR',
            'inertia': 'INERTIA',
            'inertia': 'INERTIA',
            'kst': 'KST',
            'know_sure_thing': 'KST',
            'linear_regression': 'LINREG',
            'linreg': 'LINREG',
            'linear_regression_angle': 'LINREG_ANGLE',
            'linreg_angle': 'LINREG_ANGLE',
            'linear_regression_intercept': 'LINREG_INTERCEPT',
            'linreg_intercept': 'LINREG_INTERCEPT',
            'linear_regression_slope': 'LINREG_SLOPE',
            'linreg_slope': 'LINREG_SLOPE',
            'midpoint': 'MIDPOINT',
            'midpoint': 'MIDPOINT',
            'midprice': 'MIDPRICE',
            'midprice': 'MIDPRICE',
            'sar': 'SAR',
            'sar_ext': 'SAR_EXT',
            'sar_extended': 'SAR_EXT',
            'sine_wave': 'SINE',
            'sine': 'SINE',
            'sine_wave_lead': 'SINE_LEAD',
            'sine_lead': 'SINE_LEAD',
            'trend_intensity': 'TREND_INTENSITY',
            'trend_intensity': 'TREND_INTENSITY',
            'trend_vigor': 'TREND_VIGOR',
            'trend_vigor': 'TREND_VIGOR',
            'tsf': 'TSF',
            'time_series_forecast': 'TSF',
            'ultrafast_ma': 'UF_MA',
            'uf_ma': 'UF_MA',
            'variable_moving_average': 'VMA',
            'vma': 'VMA',
            'volume_adjusted_moving_average': 'VAMA',
            'vama': 'VAMA',
            'volume_price_oscillator': 'VPO',
            'vpo': 'VPO',
            'volume_ratio': 'VR',
            'vr': 'VR',
            'volume_weighted_moving_average': 'VWMA',
            'vwma': 'VWMA',
            'wad': 'WAD',
            'williams_alligator': 'WILLIAMS_ALLIGATOR',
            'williams_alligator': 'WILLIAMS_ALLIGATOR',
            'williams_fractal': 'WILLIAMS_FRACTAL',
            'williams_fractal': 'WILLIAMS_FRACTAL',
            'zlema': 'ZLEMA',
            'zero_lag_exponential_moving_average': 'ZLEMA'
        }
        
        # Apply replacements
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Handle common patterns
        normalized = re.sub(r'_+', '_', normalized)  # Replace multiple underscores with single
        normalized = re.sub(r'^_|_$', '', normalized)  # Remove leading/trailing underscores
        
        # Capitalize first letter of each word
        normalized = ' '.join(word.capitalize() for word in normalized.split('_'))
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing indicator name '{name}': {e}")
        return name

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    try:
        return f"{value:.{decimals}f}"
    except Exception as e:
        logger.error(f"Error formatting number: {e}")
        return str(value)

def format_currency(value: float, currency: str = "USD") -> str:
    """
    Format a number as currency.
    
    Args:
        value: Number to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    try:
        if currency == "USD":
            return f"${value:,.2f}"
        else:
            return f"{value:,.2f} {currency}"
    except Exception as e:
        logger.error(f"Error formatting currency: {e}")
        return str(value)

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Number to format (as decimal)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    try:
        return f"{value * 100:.{decimals}f}%"
    except Exception as e:
        logger.error(f"Error formatting percentage: {e}")
        return str(value)

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        window: Rolling window size
        annualize: Whether to annualize the ratio
        
    Returns:
        Rolling Sharpe ratio series
    """
    try:
        excess_returns = returns - risk_free_rate
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()
        
        sharpe = rolling_mean / rolling_std
        
        if annualize:
            sharpe = sharpe * np.sqrt(252)  # Assuming daily data
        
        return sharpe
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return pd.Series(dtype=float)

def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    
    Args:
        prices: Price series
        
    Returns:
        Drawdown series
    """
    try:
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown
        
    except Exception as e:
        logger.error(f"Error calculating drawdown: {e}")
        return pd.Series(dtype=float)

def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Maximum drawdown as float
    """
    try:
        drawdown = calculate_drawdown(prices)
        return drawdown.min()
        
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return 0.0

def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate from returns.
    
    Args:
        returns: Return series
        
    Returns:
        Win rate as float
    """
    try:
        positive_returns = returns[returns > 0]
        return len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating win rate: {e}")
        return 0.0

# ============================================================================
# DECORATORS
# ============================================================================

def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def handle_exceptions(
    logger: Optional[logging.Logger] = None, 
    default_return: Any = None,
    reraise: bool = True
) -> Callable:
    """
    Decorator to handle exceptions.
    
    Args:
        logger: Logger to use
        default_return: Default return value on error
        reraise: Whether to reraise exceptions
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log = logger or logging.getLogger(func.__module__)
                log.error(f"Error in {func.__name__}: {e}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator 

def safe_json_load(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with error handling.
    
    Args:
        json_str: JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default

def safe_json_dump(obj: Any, default: str = "{}") -> str:
    """
    Safely dump object to JSON string with error handling.
    
    Args:
        obj: Object to serialize
        default: Default string to return if serialization fails
        
    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize to JSON: {e}")
        return default 

def safe_json_save(obj: Any, file_path: Union[str, Path], default: bool = False) -> bool:
    """
    Safely save object to JSON file with error handling.
    
    Args:
        obj: Object to serialize
        file_path: Path to save the JSON file
        default: Default return value if save fails
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        ensure_directory(str(Path(file_path).parent))
        with open(file_path, 'w') as f:
            json.dump(obj, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return default

def validate_config(config: Dict[str, Any], required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
        
    Returns:
        Dictionary with validation results
    """
    try:
        if not isinstance(config, dict):
            return {"valid": False, "error": "Config must be a dictionary"}
        
        if required_keys:
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                return {"valid": False, "error": f"Missing required keys: {missing_keys}"}
        
        return {"valid": True, "message": "Configuration validation passed"}
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    try:
        # Look for common project root indicators
        current_path = Path.cwd()
        while current_path != current_path.parent:
            if any((current_path / indicator).exists() for indicator in ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt']):
                return current_path
            current_path = current_path.parent
        return Path.cwd()
    except Exception as e:
        logger.error(f"Error finding project root: {e}")
        return Path.cwd() 