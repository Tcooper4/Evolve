#!/usr/bin/env python3
"""
Model Builder Service Launcher

Launches the ModelBuilderService as a standalone process.
Enhanced with system health check and LLM identity logging.
"""

import logging
import os
import platform
import signal
import socket
import sys
from pathlib import Path

import psutil
import redis

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.model_builder_service import ModelBuilderService

LLM_IDENTITY = os.environ.get("LLM_IDENTITY", "LLM-v1.0")

# Configure logging


class LLMIdentityFilter(logging.Filter):
    def filter(self, record):
        record.llm_identity = LLM_IDENTITY
        return True


log_format = (
    "%(asctime)s - %(name)s - %(levelname)s - [LLM:%(llm_identity)s] - %(message)s"
)
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("logs/model_builder_service.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
logger.addFilter(LLMIdentityFilter())


def system_health_check(
    redis_host="localhost", redis_port=6379, min_mem_gb=2, min_disk_gb=2
):
    """Perform a basic system health check before launching the service."""
    logger.info("Performing system health check...")
    # Check memory
    mem = psutil.virtual_memory()
    if mem.available < min_mem_gb * 1024**3:
        logger.error(
            f"Insufficient memory: {mem.available / (1024 ** 3):.2f} GB available, {min_mem_gb} GB required."
        )
        return False
    # Check disk
    disk = psutil.disk_usage(".")
    if disk.free < min_disk_gb * 1024**3:
        logger.error(
            f"Insufficient disk space: {disk.free / (1024 ** 3):.2f} GB available, {min_disk_gb} GB required."
        )
        return False
    # Check Redis connectivity
    try:
        r = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=2)
        r.ping()
    except Exception as e:
        logger.error(f"Redis connectivity check failed: {e}")
        return False
    # Log system info
    logger.info(
        f"System: {platform.system()} {platform.release()} | Host: {socket.gethostname()} | Python: {platform.python_version()}"
    )
    logger.info(
        f"Memory: {mem.available / (1024 ** 3):.2f} GB available | Disk: {disk.free / (1024 ** 3):.2f} GB free"
    )
    logger.info(f"Redis: {redis_host}:{redis_port} reachable")
    logger.info(f"LLM Identity: {LLM_IDENTITY}")
    return True


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down... [LLM:{LLM_IDENTITY}]")
    if hasattr(signal_handler, "service"):
        signal_handler.service.stop()
    sys.exit(0)


def main():
    """Main function to launch the ModelBuilderService."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        logger.info(f"Starting ModelBuilderService... [LLM:{LLM_IDENTITY}]")

        # System health check
        if not system_health_check():
            logger.error(
                f"System health check failed. Aborting launch. [LLM:{LLM_IDENTITY}]"
            )
            sys.exit(2)

        # Initialize the service
        service = ModelBuilderService(
            redis_host="localhost", redis_port=6379, redis_db=0
        )

        # Store service reference for signal handler
        signal_handler.service = service

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the service
        service.start()

        logger.info(f"ModelBuilderService started successfully [LLM:{LLM_IDENTITY}]")
        logger.info(
            f"Listening on channels: {service.input_channel}, {service.control_channel} [LLM:{LLM_IDENTITY}]"
        )

        # Keep the service running
        try:
            while service.is_running:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(
                f"Received keyboard interrupt, shutting down... [LLM:{LLM_IDENTITY}]"
            )
            service.stop()

    except Exception as e:
        logger.error(f"Error starting ModelBuilderService: {e} [LLM:{LLM_IDENTITY}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
