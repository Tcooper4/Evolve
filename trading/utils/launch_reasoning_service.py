#!/usr/bin/env python3
"""
Reasoning Service Launcher

Launches the real-time reasoning monitoring service.
"""

import logging
import os
import sys
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from utils.reasoning_service import ReasoningService


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/reasoning_service.log"), logging.StreamHandler()],
    )

    return {"success": True, "message": "Initialization completed", "timestamp": datetime.now().isoformat()}


def main():
    """Main launcher function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Get configuration from environment variables
    config = {
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "redis_db": int(os.getenv("REDIS_DB", "0")),
        "service_name": os.getenv("REASONING_SERVICE_NAME", "reasoning_service"),
    }

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
