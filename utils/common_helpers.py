"""
Common helper utilities.
Replacement for the removed core.utils.common_helpers module.
"""

import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_directory(path: str) -> bool:
    """Ensure a directory exists, create if it doesn't."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


def handle_exceptions(func: Callable) -> Callable:
    """Decorator to handle exceptions gracefully."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None

    return wrapper


def safe_execute(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Safely execute a function and return result with status."""
    try:
        result = func(*args, **kwargs)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        return {"success": False, "result": None, "error": str(e)}


def validate_data(data: Any, expected_type: type = None) -> bool:
    """Validate data structure and type."""
    try:
        if data is None:
            return False

        if expected_type and not isinstance(data, expected_type):
            return False

        if isinstance(data, pd.DataFrame) and data.empty:
            return False

        if isinstance(data, list) and len(data) == 0:
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False


def normalize_indicator_name(name: str) -> str:
    """Normalize indicator name for consistency."""
    if not name:
        return ""

    # Remove special characters and convert to lowercase
    normalized = name.lower().replace(" ", "_").replace("-", "_")
    normalized = "".join(c for c in normalized if c.isalnum() or c == "_")

    return normalized


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    try:
        if returns.empty:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return 0.0


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown."""
    try:
        if equity_curve.empty:
            return 0.0

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown.min()
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return 0.0


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """Calculate win rate from trades."""
    try:
        if not trades:
            return 0.0

        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        return len(winning_trades) / len(trades)
    except Exception as e:
        logger.error(f"Error calculating win rate: {e}")
        return 0.0


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    try:
        return f"{value:.2%}"
    except Exception:
        return "0.00%"


def format_currency(value: float) -> str:
    """Format value as currency."""
    try:
        return f"${value:,.2f}"
    except Exception:
        return "$0.00"


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except Exception:
        return 0.0


def clean_filename(filename: str) -> str:
    """Clean filename for safe saving."""
    import re

    # Remove or replace invalid characters
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing spaces and dots
    cleaned = cleaned.strip(". ")
    return cleaned


def create_backup_filename(original_path: str) -> str:
    """Create backup filename with timestamp."""
    try:
        path = Path(original_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
        return str(path.parent / backup_name)
    except Exception as e:
        logger.error(f"Error creating backup filename: {e}")
        return original_path


def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with dict2 taking precedence."""
    result = dict1.copy()
    result.update(dict2)
    return result


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}"
                    )
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary with success status
    """
    import json
    from datetime import datetime
    
    try:
        config_path = Path(config_file)
        
        if not config_path.exists():
            return {
                "success": False,
                "error": f"Config file not found: {config_file}",
                "timestamp": datetime.now().isoformat(),
            }
        
        suffix = config_path.suffix.lower()
        
        with open(config_path, "r", encoding="utf-8") as f:
            if suffix == ".json":
                config = json.load(f)
            elif suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                except ImportError:
                    logger.error("PyYAML not installed. Cannot load YAML config.")
                    return {
                        "success": False,
                        "error": "PyYAML not installed",
                        "timestamp": datetime.now().isoformat(),
                    }
                config = yaml.safe_load(f)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported config format: {suffix}",
                    "timestamp": datetime.now().isoformat(),
                }
        
        logger.info(f"Config loaded from {config_file}")
        
        return {
            "success": True,
            "result": config,
            "message": "Config loaded successfully",
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error loading config {config_file}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def safe_json_save(data: Dict[str, Any], filepath: str) -> bool:
    """Safely save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    import json
    
    try:
        ensure_directory(str(Path(filepath).parent))
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False


def safe_json_saver(data: Dict[str, Any], filepath: str) -> bool:
    """Alias for safe_json_save for backward compatibility."""
    return safe_json_save(data, filepath)