"""
Sanity Checks Module

Comprehensive data validation and system health monitoring for the Evolve trading system.
Provides functions to check data quality, strategy configurations, and system integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

# ============================================================================
# DATA VALIDATION CHECKS
# ============================================================================

def check_missing_columns(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Check for missing required columns in DataFrame.
    
    Args:
        df: DataFrame to check
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    try:
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                "valid": False,
                "error": f"Missing required columns: {missing_columns}",
                "missing_columns": missing_columns,
                "available_columns": list(df.columns)
            }
        
        return {
            "valid": True,
            "message": "All required columns present",
            "required_columns": required_columns
        }
        
    except Exception as e:
        logger.error(f"Error checking missing columns: {e}")
        return {
            "valid": False,
            "error": f"Error during column validation: {str(e)}"
        }

def check_sorted_index(df: pd.DataFrame) -> Dict[str, Any]:
    """Check if DataFrame index is properly sorted.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with validation results
    """
    try:
        if df.empty:
            return {
                "valid": False,
                "error": "DataFrame is empty"
            }
        
        # Check if index is sorted
        is_sorted = df.index.is_monotonic_increasing
        
        if not is_sorted:
            # Check if it's sorted in descending order
            is_descending = df.index.is_monotonic_decreasing
            
            if is_descending:
                return {
                    "valid": False,
                    "error": "Index is sorted in descending order, should be ascending",
                    "suggestion": "Use df.sort_index() to fix"
                }
            else:
                return {
                    "valid": False,
                    "error": "Index is not sorted",
                    "suggestion": "Use df.sort_index() to fix"
                }
        
        return {
            "valid": True,
            "message": "Index is properly sorted in ascending order"
        }
        
    except Exception as e:
        logger.error(f"Error checking sorted index: {e}")
        return {
            "valid": False,
            "error": f"Error during index validation: {str(e)}"
        }

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality check.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        quality_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": {},
            "duplicate_rows": 0,
            "data_types": {},
            "numeric_columns": [],
            "datetime_columns": [],
            "issues": []
        }
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                quality_report["missing_values"][col] = missing_count
                quality_report["issues"].append(f"Column '{col}' has {missing_count} missing values")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        quality_report["duplicate_rows"] = duplicate_count
        if duplicate_count > 0:
            quality_report["issues"].append(f"Found {duplicate_count} duplicate rows")
        
        # Check data types
        for col in df.columns:
            dtype = str(df[col].dtype)
            quality_report["data_types"][col] = dtype
            
            # Identify numeric columns
            if np.issubdtype(df[col].dtype, np.number):
                quality_report["numeric_columns"].append(col)
            
            # Identify datetime columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                quality_report["datetime_columns"].append(col)
        
        # Check for infinite values in numeric columns
        for col in quality_report["numeric_columns"]:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                quality_report["issues"].append(f"Column '{col}' has {inf_count} infinite values")
        
        # Overall quality score
        total_issues = len(quality_report["issues"])
        if total_issues == 0:
            quality_report["quality_score"] = 1.0
            quality_report["status"] = "excellent"
        elif total_issues <= 3:
            quality_report["quality_score"] = 0.8
            quality_report["status"] = "good"
        elif total_issues <= 5:
            quality_report["quality_score"] = 0.6
            quality_report["status"] = "fair"
        else:
            quality_report["quality_score"] = 0.4
            quality_report["status"] = "poor"
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Error checking data quality: {e}")
        return {
            "error": f"Error during quality check: {str(e)}",
            "quality_score": 0.0,
            "status": "error"
        }

def check_price_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Check price data for common issues.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        Dictionary with price validation results
    """
    try:
        price_columns = ['open', 'high', 'low', 'close']
        volume_columns = ['volume']
        
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check for required price columns
        missing_price_cols = [col for col in price_columns if col not in df.columns]
        if missing_price_cols:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Missing price columns: {missing_price_cols}")
        
        # Check for volume column
        if 'volume' not in df.columns:
            validation_result["warnings"].append("Volume column not found")
        
        # Check price relationships
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # High should be >= Low
            invalid_high_low = (df['high'] < df['low']).sum()
            if invalid_high_low > 0:
                validation_result["issues"].append(f"Found {invalid_high_low} rows where high < low")
            
            # Close should be between high and low
            invalid_close = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
            if invalid_close > 0:
                validation_result["issues"].append(f"Found {invalid_close} rows where close is outside high/low range")
        
        # Check for negative prices
        for col in price_columns:
            if col in df.columns:
                negative_prices = (df[col] < 0).sum()
                if negative_prices > 0:
                    validation_result["issues"].append(f"Found {negative_prices} negative values in {col}")
        
        # Check for zero prices
        for col in price_columns:
            if col in df.columns:
                zero_prices = (df[col] == 0).sum()
                if zero_prices > 0:
                    validation_result["warnings"].append(f"Found {zero_prices} zero values in {col}")
        
        # Check for extreme price movements
        if 'close' in df.columns:
            returns = df['close'].pct_change().abs()
            extreme_moves = (returns > 0.5).sum()  # >50% move
            if extreme_moves > 0:
                validation_result["warnings"].append(f"Found {extreme_moves} extreme price movements (>50%)")
        
        # Check volume
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                validation_result["issues"].append(f"Found {negative_volume} negative volume values")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error checking price data: {e}")
        return {
            "valid": False,
            "error": f"Error during price validation: {str(e)}"
        }

# ============================================================================
# STRATEGY CONFIGURATION CHECKS
# ============================================================================

def check_strategy_thresholds(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check strategy configuration for reasonable thresholds.
    
    Args:
        config: Strategy configuration dictionary
        
    Returns:
        Dictionary with validation results
    """
    try:
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check for required config keys
        required_keys = ['strategy_name', 'parameters']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Missing required config keys: {missing_keys}")
        
        # Check parameters
        if 'parameters' in config:
            params = config['parameters']
            
            # Check window sizes
            window_params = ['window', 'short_window', 'long_window', 'fast_period', 'slow_period']
            for param in window_params:
                if param in params:
                    value = params[param]
                    if not isinstance(value, int) or value <= 0:
                        validation_result["issues"].append(f"Invalid {param}: {value} (must be positive integer)")
                    elif value > 1000:
                        validation_result["warnings"].append(f"Large {param}: {value} (may be inefficient)")
            
            # Check standard deviation
            if 'num_std' in params:
                value = params['num_std']
                if not isinstance(value, (int, float)) or value <= 0:
                    validation_result["issues"].append(f"Invalid num_std: {value} (must be positive)")
                elif value > 5:
                    validation_result["warnings"].append(f"Large num_std: {value} (may be too conservative)")
            
            # Check thresholds
            threshold_params = ['min_volume', 'min_price', 'stop_loss', 'take_profit']
            for param in threshold_params:
                if param in params:
                    value = params[param]
                    if not isinstance(value, (int, float)) or value < 0:
                        validation_result["issues"].append(f"Invalid {param}: {value} (must be non-negative)")
        
        # Check risk parameters
        if 'risk_management' in config:
            risk_config = config['risk_management']
            
            if 'max_position_size' in risk_config:
                value = risk_config['max_position_size']
                if not isinstance(value, (int, float)) or value <= 0:
                    validation_result["issues"].append(f"Invalid max_position_size: {value}")
                elif value > 1.0:
                    validation_result["warnings"].append(f"Large max_position_size: {value} (>100%)")
            
            if 'max_drawdown' in risk_config:
                value = risk_config['max_drawdown']
                if not isinstance(value, (int, float)) or value <= 0:
                    validation_result["issues"].append(f"Invalid max_drawdown: {value}")
                elif value > 0.5:
                    validation_result["warnings"].append(f"Large max_drawdown: {value} (>50%)")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error checking strategy thresholds: {e}")
        return {
            "valid": False,
            "error": f"Error during strategy validation: {str(e)}"
        }

def check_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check model configuration for validity.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dictionary with validation results
    """
    try:
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check for required keys
        required_keys = ['model_type', 'parameters']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Missing required config keys: {missing_keys}")
        
        # Check model type
        if 'model_type' in config:
            valid_types = ['lstm', 'prophet', 'xgboost', 'random_forest', 'linear', 'ensemble']
            model_type = config['model_type'].lower()
            if model_type not in valid_types:
                validation_result["issues"].append(f"Invalid model_type: {model_type}")
        
        # Check parameters
        if 'parameters' in config:
            params = config['parameters']
            
            # Check learning rate
            if 'learning_rate' in params:
                lr = params['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                    validation_result["issues"].append(f"Invalid learning_rate: {lr} (must be 0-1)")
            
            # Check epochs/iterations
            for param in ['epochs', 'n_estimators', 'max_iter']:
                if param in params:
                    value = params[param]
                    if not isinstance(value, int) or value <= 0:
                        validation_result["issues"].append(f"Invalid {param}: {value}")
                    elif value > 10000:
                        validation_result["warnings"].append(f"Large {param}: {value} (may be slow)")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error checking model config: {e}")
        return {
            "valid": False,
            "error": f"Error during model validation: {str(e)}"
        }

# ============================================================================
# SYSTEM HEALTH CHECKS
# ============================================================================

def check_system_resources() -> Dict[str, Any]:
    """Check system resource availability.
    
    Returns:
        Dictionary with resource status
    """
    try:
        import psutil
        import os
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "issues": [],
            "warnings": []
        }
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        health_report["cpu_usage"] = cpu_percent
        if cpu_percent > 90:
            health_report["status"] = "warning"
            health_report["warnings"].append(f"High CPU usage: {cpu_percent}%")
        elif cpu_percent > 80:
            health_report["warnings"].append(f"Elevated CPU usage: {cpu_percent}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        health_report["memory_usage"] = memory.percent
        if memory.percent > 90:
            health_report["status"] = "warning"
            health_report["warnings"].append(f"High memory usage: {memory.percent}%")
        elif memory.percent > 80:
            health_report["warnings"].append(f"Elevated memory usage: {memory.percent}%")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        health_report["disk_usage"] = disk_percent
        if disk_percent > 90:
            health_report["status"] = "warning"
            health_report["warnings"].append(f"Low disk space: {disk_percent:.1f}% used")
        elif disk_percent > 80:
            health_report["warnings"].append(f"Elevated disk usage: {disk_percent:.1f}%")
        
        # Check available memory
        health_report["available_memory_gb"] = memory.available / (1024**3)
        if memory.available < 1 * 1024**3:  # Less than 1GB
            health_report["warnings"].append("Low available memory")
        
        return health_report
        
    except ImportError:
        logger.warning("psutil not available for system resource monitoring")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "message": "psutil not available for resource monitoring"
        }
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": f"Error during resource check: {str(e)}"
        }

def check_data_freshness(df: pd.DataFrame, max_age_hours: int = 24) -> Dict[str, Any]:
    """Check if data is fresh enough for trading.
    
    Args:
        df: DataFrame with datetime index
        max_age_hours: Maximum age in hours
        
    Returns:
        Dictionary with freshness check results
    """
    try:
        if df.empty:
            return {
                "valid": False,
                "error": "DataFrame is empty"
            }
        
        # Get the most recent timestamp
        latest_time = df.index.max()
        current_time = pd.Timestamp.now()
        
        # Calculate age
        age = current_time - latest_time
        age_hours = age.total_seconds() / 3600
        
        freshness_result = {
            "valid": age_hours <= max_age_hours,
            "latest_data": latest_time.isoformat(),
            "current_time": current_time.isoformat(),
            "age_hours": age_hours,
            "max_age_hours": max_age_hours
        }
        
        if age_hours > max_age_hours:
            freshness_result["warning"] = f"Data is {age_hours:.1f} hours old (max: {max_age_hours})"
        
        return freshness_result
        
    except Exception as e:
        logger.error(f"Error checking data freshness: {e}")
        return {
            "valid": False,
            "error": f"Error during freshness check: {str(e)}"
        }

# ============================================================================
# COMPREHENSIVE VALIDATION
# ============================================================================

def run_comprehensive_validation(df: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run comprehensive validation on data and configuration.
    
    Args:
        df: DataFrame to validate
        config: Optional configuration to validate
        
    Returns:
        Comprehensive validation report
    """
    try:
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "pass",
            "checks": {},
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0
            }
        }
        
        # Data quality checks
        validation_report["checks"]["data_quality"] = check_data_quality(df)
        validation_report["checks"]["sorted_index"] = check_sorted_index(df)
        validation_report["checks"]["price_data"] = check_price_data(df)
        
        # Configuration checks
        if config:
            validation_report["checks"]["strategy_config"] = check_strategy_thresholds(config)
            if "model_type" in config:
                validation_report["checks"]["model_config"] = check_model_config(config)
        
        # System checks
        validation_report["checks"]["system_resources"] = check_system_resources()
        
        # Calculate summary
        for check_name, check_result in validation_report["checks"].items():
            validation_report["summary"]["total_checks"] += 1
            
            if check_result.get("valid", True):
                validation_report["summary"]["passed_checks"] += 1
            else:
                validation_report["summary"]["failed_checks"] += 1
                validation_report["overall_status"] = "fail"
            
            if "warnings" in check_result and check_result["warnings"]:
                validation_report["summary"]["warnings"] += len(check_result["warnings"])
        
        return validation_report
        
    except Exception as e:
        logger.error(f"Error running comprehensive validation: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "error",
            "error": f"Error during validation: {str(e)}"
        } 