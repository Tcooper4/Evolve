#!/usr/bin/env python3
"""
Application runner script.
Handles command-line arguments and starts the application in the appropriate mode.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Trading Application Runner")
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="Application mode (default: dev)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the application on (default: 8501)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to run the application on (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--config",
        default="config/app_config.yaml",
        help="Path to configuration file (default: config/app_config.yaml)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    return parser.parse_args()

def setup_logging(log_level):
    """Initialize logging configuration."""
    log_config_path = Path("config/logging_config.yaml")
    if not log_config_path.exists():
        print("Error: logging_config.yaml not found")
        sys.exit(1)
    
    with open(log_config_path) as f:
        log_config = yaml.safe_load(f)
    
    # Update log level
    log_config["handlers"]["console"]["level"] = log_level
    log_config["handlers"]["file"]["level"] = log_level
    log_config["loggers"]["trading"]["level"] = log_level
    
    logging.config.dictConfig(log_config)
    logger = logging.getLogger("trading")
    logger.info(f"Logging initialized with level: {log_level}")

def load_config(config_path):
    """Load application configuration."""
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger = logging.getLogger("trading")
    logger.info("Configuration loaded")
    return config

def run_application(args, config):
    """Run the application in the specified mode."""
    logger = logging.getLogger("trading")
    
    # Update configuration with command line arguments
    config["server"]["port"] = args.port
    config["server"]["host"] = args.host
    config["server"]["debug"] = args.mode == "dev"
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = args.host
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    if args.mode == "dev":
        os.environ["STREAMLIT_DEBUG"] = "true"
        logger.info("Starting application in development mode")
    else:
        os.environ["STREAMLIT_DEBUG"] = "false"
        logger.info("Starting application in production mode")
    
    try:
        import streamlit.web.cli as stcli
        sys.argv = [
            "streamlit",
            "run",
            "app.py",
            "--server.port", str(args.port),
            "--server.address", args.host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        stcli.main()
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Run application
    run_application(args, config)

if __name__ == "__main__":
    main() 