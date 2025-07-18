#!/usr/bin/env python3
"""
Reasoning Service Launcher

Launches the real-time reasoning monitoring service.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from utils.reasoning_service import ReasoningService
from utils.service_utils import setup_service_logging, load_service_config


def main():
    """Main launcher function."""
    # Setup logging
    logger = setup_service_logging("reasoning_service")

    # Get configuration from environment variables
    config = load_service_config({
        "redis_host": "REDIS_HOST",
        "redis_port": "REDIS_PORT", 
        "redis_db": "REDIS_DB",
        "service_name": "REASONING_SERVICE_NAME"
    }, defaults={
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "service_name": "reasoning_service"
    })

    logger.info("Starting Reasoning Service...")
    logger.info(f"Configuration: {config}")

    try:
        # Initialize and start service
        service = ReasoningService(
            redis_host=config["redis_host"],
            redis_port=config["redis_port"],
            redis_db=config["redis_db"],
            service_name=config["service_name"],
        )

        # Start the service
        service.start()

    except KeyboardInterrupt:
        logger.info("Shutting down Reasoning Service...")
        service.stop()
    except Exception as e:
        logger.error(f"Error starting Reasoning Service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
