#!/usr/bin/env python3
"""
Multimodal Service Launcher

Launches the MultimodalService as a standalone process.
"""

import logging
import os
import signal
import sys
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.multimodal_service import MultimodalService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/multimodal_service.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, "service"):
        signal_handler.service.stop()
    sys.exit(0)


def main():
    """Main function to launch the MultimodalService."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        logger.info("Starting MultimodalService...")

        # Initialize the service
        service = MultimodalService(redis_host="localhost", redis_port=6379, redis_db=0)

        # Store service reference for signal handler
        signal_handler.service = service

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the service
        service.start()

        logger.info("MultimodalService started successfully")
        logger.info(
            f"Listening on channels: {service.input_channel}, {service.control_channel}"
        )

        # Keep the service running
        try:
            while service.is_running:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            service.stop()

    except Exception as e:
        logger.error(f"Error starting MultimodalService: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
