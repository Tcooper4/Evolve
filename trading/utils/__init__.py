"""Utility modules for the trading package.

This module provides comprehensive utilities for:
- Data processing and validation
- Logging and monitoring
- Mathematical calculations
- File operations
- Time utilities
- Validation functions
- And more...
"""

import json
import logging as std_logging
import os

# Import from root utils directory
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

# Trading utilities
from .cache_manager import CacheManager, cache_result, clear_cache
from .config_manager import ConfigLoader, ConfigManager, ConfigValidator
from .data_transformer import DataTransformer
from .data_validation import DataValidator
from .feature_engineering import FeatureEngineer
from .logging import (
    LoggingManager,
    cleanup_logs,
    get_logging_stats,
    log_event,
    setup_logging,
)
from .model_evaluation import ModelEvaluator, ModelValidator
from .model_monitoring import ModelMonitor
from .performance_metrics import PerformanceMetrics, RiskMetrics, TradingMetrics

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
# Import from core utils
from utils.common_helpers import ensure_directory

utils_logger = std_logging.getLogger(__name__)


def check_model_performance(metrics: Dict[str, Any]) -> bool:
    """Check if model performance meets minimum criteria."""
    try:
        # Basic performance checks
        if "accuracy" in metrics and metrics["accuracy"] < 0.6:
            return False
        if "sharpe_ratio" in metrics and metrics["sharpe_ratio"] < 0.5:
            return False
        if "max_drawdown" in metrics and metrics["max_drawdown"] < -0.2:
            return False
        return True
    except Exception as e:
        utils_logger.error(f"Error checking model performance: {e}")
        return False


def detect_model_drift(model_id: str) -> bool:
    """Detect if model has drifted."""
    try:
        # Simple drift detection - in practice, you'd use statistical tests
        metrics_path = f"metrics/{model_id}_metrics.json"
        if not os.path.exists(metrics_path):
            return False

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Check if recent performance has degraded
        recent_accuracy = metrics.get("recent_accuracy", 1.0)
        if recent_accuracy < 0.5:
            return True

        return False
    except Exception as e:
        utils_logger.error(f"Error detecting model drift: {e}")
        return False


def validate_update_result(model_id: str, result: Dict[str, Any]) -> bool:
    """Validate model update result."""
    try:
        # Basic validation
        required_keys = ["accuracy", "sharpe_ratio", "timestamp"]
        for key in required_keys:
            if key not in result:
                return False

        # Check if performance improved
        if result.get("accuracy", 0) < 0.5:
            return False

        return True
    except Exception as e:
        utils_logger.error(f"Error validating update result: {e}")
        return False


def calculate_reweighting_factors(models: List[str]) -> Dict[str, float]:
    """Calculate ensemble reweighting factors."""
    try:
        factors = {}
        total_performance = 0.0

        for model_id in models:
            metrics = get_model_metrics(model_id)
            performance = metrics.get("sharpe_ratio", 0.0)
            factors[model_id] = max(performance, 0.0)
            total_performance += factors[model_id]

        # Normalize weights
        if total_performance > 0:
            for model_id in factors:
                factors[model_id] /= total_performance
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(models)
            for model_id in models:
                factors[model_id] = equal_weight

        return factors
    except Exception as e:
        utils_logger.error(f"Error calculating reweighting factors: {e}")
        return {model_id: 1.0 / len(models) for model_id in models}


def get_model_metrics(model_id: str) -> Dict[str, Any]:
    """Get metrics for a specific model."""
    try:
        metrics_path = f"metrics/{model_id}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        utils_logger.error(f"Error getting model metrics: {e}")
        return {}


def check_update_frequency(model_id: str) -> bool:
    """Check if model needs updating based on frequency."""
    try:
        metrics = get_model_metrics(model_id)
        last_update = metrics.get("last_update")
        if not last_update:
            return True

        last_update_dt = datetime.fromisoformat(last_update)
        days_since_update = (datetime.now() - last_update_dt).days

        # Update if more than 7 days old
        return days_since_update > 7
    except Exception as e:
        utils_logger.error(f"Error checking update frequency: {e}")
        return True


def get_ensemble_weights() -> Dict[str, float]:
    """Get current ensemble weights."""
    try:
        weights_path = "models/ensemble_weights.json"
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        utils_logger.error(f"Error getting ensemble weights: {e}")
        return {}


def save_ensemble_weights(weights: Dict[str, float]) -> None:
    """Save ensemble weights."""
    try:
        weights_path = "models/ensemble_weights.json"
        try:
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        except Exception as e:
            utils_logger.error(f"Failed to create directory for weights: {e}")
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)
    except Exception as e:
        utils_logger.error(f"Error saving ensemble weights: {e}")


def check_data_quality(data: pd.DataFrame) -> bool:
    """Check data quality for model training."""
    try:
        # Basic quality checks
        if data.empty:
            return False

        # Check for missing values
        if data.isnull().sum().sum() > len(data) * 0.1:
            return False

        # Check for sufficient data
        if len(data) < 100:
            return False

        return True
    except Exception as e:
        utils_logger.error(f"Error checking data quality: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    try:
        import platform
        import sys

        import psutil

        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage("/").percent,
            "current_time": datetime.now().isoformat(),
        }
    except Exception as e:
        utils_logger.error(f"Error getting system info: {e}")
        return {"error": str(e)}


# Missing utility functions that are expected by the system


def get_current_time() -> datetime:
    """Get current time."""
    return datetime.now()


def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp."""
    return dt.strftime(format_str)


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string."""
    return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))


def get_time_difference(dt1: datetime, dt2: datetime) -> timedelta:
    """Get time difference between two datetimes."""
    return abs(dt1 - dt2)


def is_market_open() -> bool:
    """Check if market is currently open."""
    market_hours = MarketHours()
    return market_hours.is_market_open(datetime.now())


def get_next_market_open() -> datetime:
    """Get next market open time."""
    market_hours = MarketHours()
    return market_hours.get_next_market_open(datetime.now())


def get_previous_market_close() -> datetime:
    """Get previous market close time."""
    MarketHours()
    # This is a simplified implementation
    return datetime.now() - timedelta(hours=16)


def convert_timezone(dt: datetime, timezone: str) -> datetime:
    """Convert datetime to specified timezone."""
    time_utils = TimeUtils()
    return time_utils.to_timezone(dt, timezone)


def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Get list of trading days between dates."""
    # Simplified implementation - returns weekdays
    trading_days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday to Friday
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days


def get_holidays() -> List[datetime]:
    """Get list of market holidays."""
    # Simplified implementation - returns empty list
    return []


# File utility functions


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data as JSON."""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def save_pickle(data: Any, filepath: str) -> None:
    """Save data as pickle."""
    import pickle

    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """Load data from pickle."""
    import pickle

    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def save_csv(data: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame as CSV."""
    ensure_directory(os.path.dirname(filepath))
    data.to_csv(filepath, index=False)


def load_csv(filepath: str) -> pd.DataFrame:
    """Load DataFrame from CSV."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0


def get_file_modified_time(filepath: str) -> datetime:
    """Get file modification time."""
    if os.path.exists(filepath):
        return datetime.fromtimestamp(os.path.getmtime(filepath))
    return datetime.now()


def backup_file(filepath: str, backup_dir: str = "backups") -> str:
    """Create backup of file."""
    if not os.path.exists(filepath):
        return ""

    ensure_directory(backup_dir)
    filename = os.path.basename(filepath)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}")

    import shutil

    shutil.copy2(filepath, backup_path)
    return backup_path


def cleanup_old_files(directory: str, days: int = 30) -> int:
    """Clean up files older than specified days."""
    if not os.path.exists(directory):
        return 0

    cutoff_time = datetime.now() - timedelta(days=days)
    deleted_count = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff_time:
                os.remove(filepath)
                deleted_count += 1

    return deleted_count


# Data utility functions


def load_data(
    symbol: str, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """Load data for a symbol."""
    # This is a stub - in practice would use data providers
    return pd.DataFrame()


# Validation utility functions


def validate_symbol(symbol: str) -> bool:
    """Validate symbol format."""
    return bool(symbol and len(symbol) > 0)


def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe format."""
    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
    return timeframe in valid_timeframes


def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range."""
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        return start < end
    except (ValueError, TypeError) as e:
        utils_logger.warning(
            f"Invalid date format: {start_date} or {end_date}. Error: {e}"
        )
        return False


def validate_model_type(model_type: str) -> bool:
    """Validate model type."""
    valid_types = ["lstm", "transformer", "xgboost", "prophet", "ensemble"]
    return model_type.lower() in valid_types


def validate_strategy_type(strategy_type: str) -> bool:
    """Validate strategy type."""
    valid_types = ["rsi", "macd", "bollinger", "custom"]
    return strategy_type.lower() in valid_types


def validate_portfolio_data(data: pd.DataFrame) -> bool:
    """Validate portfolio data."""
    return not data.empty and "symbol" in data.columns


def validate_trade_data(data: pd.DataFrame) -> bool:
    """Validate trade data."""
    return not data.empty and "symbol" in data.columns


def validate_forecast_data(data: pd.DataFrame) -> bool:
    """Validate forecast data."""
    return not data.empty and "forecast" in data.columns


__all__ = [
    "CacheManager",
    "cache_result",
    "clear_cache",
    "LoggingManager",
    "log_event",
    "setup_logging",
    "get_logging_stats",
    "cleanup_logs",
    "DataValidator",
    "DataTransformer",
    "FeatureEngineer",
    "ConfigManager",
    "ConfigValidator",
    "ConfigLoader",
    "PerformanceMetrics",
    "RiskMetrics",
    "TradingMetrics",
    "ModelEvaluator",
    "ModelValidator",
    "ModelMonitor",
]
