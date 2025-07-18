"""
Service Utilities Module

Common utilities for service setup and configuration.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from utils.logging import setup_logging as setup_enhanced_logging


def setup_service_logging(service_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Setup logging for a service with standard configuration.
    
    Args:
        service_name: Name of the service
        log_dir: Directory for log files
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Use enhanced logging setup
    log_manager = setup_enhanced_logging(log_dir, logging.INFO)
    
    # Get logger for the service
    logger = log_manager.get_enhanced_logger(
        service_name,
        log_file=f"{service_name}.log",
        enable_console=True,
        enable_file=True
    )
    
    logger.info(f"Logging initialized for {service_name}")
    return logger


def load_service_config(
    env_mapping: Dict[str, str],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load service configuration from environment variables.
    
    Args:
        env_mapping: Mapping of config keys to environment variable names
        defaults: Default values for configuration
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = defaults or {}
    
    for config_key, env_var in env_mapping.items():
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


def create_sample_market_data(rows: int = 100) -> Dict[str, Any]:
    """
    Create sample market data for testing.
    
    Args:
        rows: Number of data points
        
    Returns:
        Dict[str, Any]: Sample market data
    """
    import numpy as np
    import pandas as pd
    
    dates = pd.date_range(start='2023-01-01', periods=rows, freq='D')
    prices = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    
    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, rows)
    }, index=dates)
    
    return {
        'data': data,
        'symbol': 'SAMPLE',
        'start_date': dates[0],
        'end_date': dates[-1],
        'rows': rows
    } 