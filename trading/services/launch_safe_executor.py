#!/usr/bin/env python3
"""
Safe Executor Service Launcher

Launches the SafeExecutor service for safe execution of user-defined models and strategies.
"""

import sys
import os
import signal
import logging
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.safe_executor_service import SafeExecutorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/safe_executor_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'service'):
        signal_handler.service.stop()
    sys.exit(0)


    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def main():
    """Main function to launch the SafeExecutor service."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logger.info("Starting SafeExecutor service...")
        
        # Get configuration from environment
        timeout_seconds = int(os.getenv('SAFE_EXECUTOR_TIMEOUT', '300'))
        memory_limit_mb = int(os.getenv('SAFE_EXECUTOR_MEMORY_LIMIT', '1024'))
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_db = int(os.getenv('REDIS_DB', '0'))
        
        # Initialize service
        service = SafeExecutorService(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb
        )
        
        # Store reference for signal handler
        signal_handler.service = service
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("SafeExecutor service started successfully")
        logger.info(f"Configuration: timeout={timeout_seconds}s, memory_limit={memory_limit_mb}MB")
        logger.info("Ready to execute user-defined models and strategies safely")
        
        # Start the service
        service.start()
        
    except Exception as e:
        logger.error(f"Error starting SafeExecutor service: {e}")
        sys.exit(1)


    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == "__main__":
    main() 