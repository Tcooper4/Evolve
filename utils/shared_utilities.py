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
    Unified logging setup function for all services and scripts.
    
    Args:
        service_name: Name of the service for logging identification
        log_dir: Directory for log files
        log_level: Logging level
        enable_file_output: Whether to enable file output
        enable_rotating_handlers: Whether to enable rotating file handlers
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Use enhanced logging setup
    log_manager = setup_enhanced_logging(log_dir, log_level)
    
    # Get logger for the service
    logger_name = service_name or __name__
    logger = log_manager.get_enhanced_logger(
        logger_name,
        log_file=f"{logger_name}.log" if enable_file_output else None,
        enable_console=True,
        enable_file=enable_file_output
    )
    
    logger.info(f"Logging initialized for {logger_name}")
    return logger


def create_sample_data(
    rows: int = 100,
    start_date: str = '2023-01-01',
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0
) -> pd.DataFrame:
    """
    Create sample market data for testing and examples.
    
    Args:
        rows: Number of data points to generate
        start_date: Start date for the data
        base_price: Base price for the asset
        volatility: Price volatility (standard deviation of returns)
        trend: Daily trend component
        
    Returns:
        pd.DataFrame: Sample market data with OHLCV columns
    """
    dates = pd.date_range(start=start_date, periods=rows, freq='D')
    
    # Generate price series with trend and volatility
    returns = np.random.normal(trend, volatility, rows)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, rows)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, rows))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, rows))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, rows)
    }, index=dates)
    
    # Ensure OHLC relationships are maintained
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


def create_sample_forecast_data(
    rows: int = 100,
    features: int = 10,
    target_column: str = 'target'
) -> pd.DataFrame:
    """
    Create sample data for forecasting models.
    
    Args:
        rows: Number of data points
        features: Number of feature columns
        target_column: Name of the target column
        
    Returns:
        pd.DataFrame: Sample data with features and target
    """
    dates = pd.date_range(start='2023-01-01', periods=rows, freq='D')
    
    # Generate feature data
    feature_data = {}
    for i in range(features):
        feature_data[f'feature_{i}'] = np.random.randn(rows)
    
    # Generate target with some correlation to features
    target = np.random.randn(rows)
    for i in range(min(3, features)):  # Correlate with first 3 features
        target += 0.3 * feature_data[f'feature_{i}']
    
    # Add some trend and seasonality
    trend = np.linspace(0, 2, rows)
    seasonality = 0.5 * np.sin(2 * np.pi * np.arange(rows) / 30)  # Monthly seasonality
    target += trend + seasonality
    
    data = pd.DataFrame(feature_data, index=dates)
    data[target_column] = target
    
    return data


def main_runner(
    main_func: callable,
    service_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Common main function runner for services and scripts.
    
    Args:
        main_func: Main function to run
        service_name: Name of the service for logging
        config: Configuration dictionary
    """
    try:
        # Setup logging
        logger = setup_logging(service_name)
        
        # Log startup
        logger.info(f"Starting {service_name or 'application'}")
        if config:
            logger.info(f"Configuration: {config}")
        
        # Run main function
        if asyncio.iscoroutinefunction(main_func):
            asyncio.run(main_func())
        else:
            main_func()
            
    except KeyboardInterrupt:
        logger.info(f"Shutting down {service_name or 'application'}...")
    except Exception as e:
        logger.error(f"Error in {service_name or 'application'}: {e}")
        sys.exit(1)


async def service_launcher(
    service_class: type,
    service_name: str,
    config: Dict[str, Any],
    shutdown_handler: Optional[callable] = None
) -> None:
    """
    Common service launcher for async services.
    
    Args:
        service_class: Service class to instantiate
        service_name: Name of the service
        config: Service configuration
        shutdown_handler: Optional shutdown handler function
    """
    logger = setup_logging(service_name)
    logger.info(f"Starting {service_name}...")
    logger.info(f"Configuration: {config}")
    
    try:
        # Initialize service
        service = service_class(**config)
        
        # Start service
        if hasattr(service, 'start'):
            service.start()
        elif hasattr(service, 'run'):
            await service.run()
        else:
            logger.warning(f"Service {service_name} has no start/run method")
            return
        
        # Keep service running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info(f"Shutting down {service_name}...")
        if hasattr(service, 'stop'):
            service.stop()
        if shutdown_handler:
            await shutdown_handler(service)
    except Exception as e:
        logger.error(f"Error in {service_name}: {e}")
        raise


def load_config_from_env(
    config_mapping: Dict[str, str],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Args:
        config_mapping: Mapping of config keys to environment variable names
        defaults: Default values for configuration
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = defaults or {}
    
    for config_key, env_var in config_mapping.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Try to convert to appropriate type
            try:
                # Try integer first
                config[config_key] = int(env_value)
            except ValueError:
                try:
                    # Try float
                    config[config_key] = float(env_value)
                except ValueError:
                    # Keep as string
                    config[config_key] = env_value
    
    return config


def validate_config(
    config: Dict[str, Any],
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None
) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required configuration keys
        optional_keys: List of optional configuration keys
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    if optional_keys:
        unknown_keys = [key for key in config.keys() 
                       if key not in required_keys and key not in optional_keys]
        if unknown_keys:
            logging.warning(f"Unknown configuration keys: {unknown_keys}")
    
    return True


def create_directory_structure(base_dir: str, structure: Dict[str, Any]) -> None:
    """
    Create directory structure for the application.
    
    Args:
        base_dir: Base directory path
        structure: Dictionary defining directory structure
    """
    base_path = Path(base_dir)
    
    def create_dirs(current_path: Path, struct: Dict[str, Any]):
        for name, content in struct.items():
            dir_path = current_path / name
            if isinstance(content, dict):
                dir_path.mkdir(parents=True, exist_ok=True)
                create_dirs(dir_path, content)
            else:
                # Create file
                dir_path.parent.mkdir(parents=True, exist_ok=True)
                if not dir_path.exists():
                    dir_path.touch()
    
    create_dirs(base_path, structure)


# Common directory structure for the application
DEFAULT_DIRECTORY_STRUCTURE = {
    "logs": {
        "models": {},
        "data": {},
        "performance": {},
        "agents": {},
        "services": {}
    },
    "data": {
        "raw": {},
        "processed": {},
        "cache": {}
    },
    "models": {
        "checkpoints": {},
        "saved": {},
        "configs": {}
    },
    "reports": {
        "html": {},
        "pdf": {},
        "json": {}
    }
}


def initialize_application_directories(base_dir: str = ".") -> None:
    """
    Initialize standard application directory structure.
    
    Args:
        base_dir: Base directory for the application
    """
    create_directory_structure(base_dir, DEFAULT_DIRECTORY_STRUCTURE)
    logging.info(f"Application directories initialized in {base_dir}")


def get_application_root() -> Path:
    """
    Get the application root directory.
    
    Returns:
        Path: Application root directory
    """
    # Look for common root indicators
    current = Path.cwd()
    
    while current != current.parent:
        # Check for common root indicators
        if any((current / indicator).exists() for indicator in [
            'requirements.txt', 'setup.py', 'pyproject.toml', 
            'README.md', '.git', 'config'
        ]):
            return current
        current = current.parent
    
    return Path.cwd()


def format_timestamp(timestamp: Optional[Union[datetime, str, float]] = None) -> str:
    """
    Format timestamp for logging and file naming.
    
    Args:
        timestamp: Timestamp to format (defaults to current time)
        
    Returns:
        str: Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, (int, float)):
        timestamp = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    return timestamp.strftime('%Y%m%d_%H%M%S')


def safe_filename(filename: str) -> str:
    """
    Convert a string to a safe filename.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Safe filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure it's not empty
    if not filename:
        filename = 'unnamed'
    
    return filename 