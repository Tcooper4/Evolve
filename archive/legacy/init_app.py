#!/usr/bin/env python3
"""
Application initialization script.
Provides commands for initializing the application, including environment setup and initial configuration.

This script supports:
- Initializing the application environment
- Setting up initial configuration
- Running initial setup tasks

Usage:
    python init_app.py <command> [options]

Commands:
    env         Initialize environment
    config      Set up initial configuration
    setup       Run initial setup tasks

Examples:
    # Initialize environment
    python init_app.py env

    # Set up initial configuration
    python init_app.py config

    # Run initial setup tasks
    python init_app.py setup
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

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == "__main__":
    main() 