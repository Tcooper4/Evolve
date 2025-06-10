#!/usr/bin/env python3
"""
Application initialization script.
Creates necessary directories, sets up logging, and validates configuration.
"""

import os
import sys
import yaml
import logging
import logging.config
from pathlib import Path

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "logs",
        "data",
        "data/market",
        "data/models",
        "data/backtests",
        "data/portfolio",
        "config",
        "trading/nlp/config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_logging():
    """Initialize logging configuration."""
    log_config_path = Path("config/logging_config.yaml")
    if not log_config_path.exists():
        print("Error: logging_config.yaml not found")
        sys.exit(1)
    
    with open(log_config_path) as f:
        log_config = yaml.safe_load(f)
    
    logging.config.dictConfig(log_config)
    logger = logging.getLogger("trading")
    logger.info("Logging initialized")

def validate_config():
    """Validate application configuration."""
    config_path = Path("config/app_config.yaml")
    if not config_path.exists():
        print("Error: app_config.yaml not found")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    required_sections = [
        "server",
        "logging",
        "database",
        "market_data",
        "models",
        "strategies",
        "risk",
        "nlp",
        "api",
        "monitoring",
        "security"
    ]
    
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required section '{section}' in app_config.yaml")
            sys.exit(1)
    
    logger = logging.getLogger("trading")
    logger.info("Configuration validated")

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import numpy
        import pandas
        import torch
        import streamlit
        import plotly
        import yfinance
        import ta
        import redis
        import fastapi
        import uvicorn
        logger = logging.getLogger("trading")
        logger.info("All dependencies are installed")
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        sys.exit(1)

def main():
    """Main initialization function."""
    print("Initializing application...")
    
    # Create directories
    setup_directories()
    
    # Setup logging
    setup_logging()
    
    # Validate configuration
    validate_config()
    
    # Check dependencies
    check_dependencies()
    
    print("Application initialization completed successfully")

if __name__ == "__main__":
    main() 