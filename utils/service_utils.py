"""
Service Utilities Module

Common utilities for service setup and configuration.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logging import setup_logging as setup_enhanced_logging


def setup_service_logging(service_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Setup logging for a service with standard configuration.
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Setup enhanced logging
    logger = setup_enhanced_logging(
        service_name=service_name,
        log_dir=log_dir,
        enable_file_output=True,
        enable_rotating_handlers=True,
    )

    return logger


def load_service_config(
    env_mapping: Dict[str, str], defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load service configuration from environment variables.
    """
    config = defaults or {}

    for config_key, env_var in env_mapping.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Try to convert to int if it looks like a number
            try:
                if env_value.isdigit():
                    config[config_key] = int(env_value)
                else:
                    config[config_key] = env_value
            except ValueError:
                config[config_key] = env_value

    return config


def create_sample_market_data(rows: int = 100) -> Dict[str, Any]:
    """
    Create sample market data for testing.
    """
    import numpy as np
    import pandas as pd

    dates = pd.date_range(start="2023-01-01", periods=rows, freq="D")
    prices = 100 + np.cumsum(np.random.randn(rows) * 0.5)

    data = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, rows),
        },
        index=dates,
    )

    return {
        "data": data,
        "symbol": "SAMPLE",
        "start_date": dates[0],
        "end_date": dates[-1],
        "rows": rows,
    }
