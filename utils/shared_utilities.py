"""
Shared Utilities Module

This module consolidates common functions that are duplicated across the codebase:
- setup_logging: Unified logging setup function
- create_sample_data: Sample data generation for testing
- main_runner: Common main function patterns
- service_launcher: Common service launching patterns
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils.logging import setup_logging as setup_enhanced_logging


def setup_logging(
    service_name: Optional[str] = None,
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    enable_file_output: bool = True,
    enable_rotating_handlers: bool = True
) -> logging.Logger:
    """
    Unified logging setup function.
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Use enhanced logging setup
    logger = setup_enhanced_logging(
        service_name=service_name,
        log_dir=log_dir,
        log_level=log_level,
        enable_file_output=enable_file_output,
        enable_rotating_handlers=enable_rotating_handlers
    )

    return logger


def create_sample_data(
    rows: int = 100,
    start_date: str = '2023-01-01',
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0
) -> pd.DataFrame:
    """
    Create sample market data for testing.
    """
    dates = pd.date_range(start=start_date, periods=rows, freq='D')
    np.random.seed(42)  # For reproducible results

    # Generate price series with trend and volatility
    returns = np.random.normal(trend, volatility, rows)
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, rows)
    }, index=dates)

    return data


def create_sample_forecast_data(
    rows: int = 100,
    features: int = 10,
    target_column: str = 'target'
) -> pd.DataFrame:
    """
    Create sample forecasting data with features and target.
    """
    np.random.seed(42)  # For reproducible results

    # Generate feature data
    feature_data = np.random.randn(rows, features)
    feature_names = [f'feature_{i}' for i in range(features)]

    # Generate target with some correlation to features
    target = np.sum(feature_data[:, :3], axis=1) + np.random.normal(0, 0.1, rows)

    data = pd.DataFrame(feature_data, columns=feature_names)
    data[target_column] = target

    return data


def main_runner(
    main_func: callable,
    service_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Common main function pattern for services.
    """
    try:
        # Setup logging
        logger = setup_logging(service_name=service_name)
        logger.info(f"Starting {service_name or 'service'}")

        # Run main function
        if config:
            main_func(config)
        else:
            main_func()

        logger.info(f"{service_name or 'Service'} completed successfully")

    except KeyboardInterrupt:
        logger.info(f"{service_name or 'Service'} interrupted by user")
    except Exception as e:
        logger.error(f"Error in {service_name or 'service'}: {e}")
        sys.exit(1)


async def service_launcher(
    service_class: type,
    service_name: str,
    config: Dict[str, Any],
    shutdown_handler: Optional[callable] = None
) -> None:
    """
    Common async service launching pattern.
    """
    logger = setup_logging(service_name=service_name)
    service = None

    try:
        logger.info(f"Starting {service_name}")
        service = service_class(**config)
        await service.start()

        # Keep service running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info(f"Shutting down {service_name}")
    except Exception as e:
        logger.error(f"Error in {service_name}: {e}")
    finally:
        if service and hasattr(service, 'stop'):
            await service.stop()
        if shutdown_handler:
            await shutdown_handler()


def load_config_from_env(
    config_mapping: Dict[str, str],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    """
    config = defaults or {}

    for config_key, env_var in config_mapping.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                if env_value.isdigit():
                    config[config_key] = int(env_value)
                elif env_value.lower() in ('true', 'false'):
                    config[config_key] = env_value.lower() == 'true'
                else:
                    config[config_key] = env_value
            except ValueError:
                config[config_key] = env_value

    return config


def validate_config(
    config: Dict[str, Any],
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None
) -> bool:
    """
    Validate configuration dictionary.
    """
    # Check required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logging.error(f"Missing required configuration keys: {missing_keys}")
        return False

    # Check optional keys (warn if missing)
    if optional_keys:
        missing_optional = [key for key in optional_keys if key not in config]
        if missing_optional:
            logging.warning(f"Missing optional configuration keys: {missing_optional}")

    return True


def create_directory_structure(base_dir: str, structure: Dict[str, Any]) -> None:
    """
    Create directory structure recursively.
    """
    for name, content in structure.items():
        path = Path(base_dir) / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_directory_structure(str(path), content)
        else:
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_DIRECTORY_STRUCTURE = {
    "logs": {},
    "data": {
        "raw": {},
        "processed": {},
        "cache": {}
    },
    "models": {
        "trained": {},
        "checkpoints": {}
    },
    "reports": {
        "html": {},
        "pdf": {},
        "csv": {}
    },
    "config": {},
    "backups": {}
}


def initialize_application_directories(base_dir: str = ".") -> None:
    """
    Initialize standard application directory structure.
    """
    create_directory_structure(base_dir, DEFAULT_DIRECTORY_STRUCTURE)


def get_application_root() -> Path:
    """
    Get the application root directory.
    """
    return Path(__file__).parent.parent


def format_timestamp(timestamp: Optional[Union[datetime, str, float]] = None) -> str:
    """
    Format timestamp for consistent use across the application.
    """
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, (int, float)):
        timestamp = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    return timestamp.strftime("%Y%m%d_%H%M%S")


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.
    """
    import re
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')
    return safe_name
