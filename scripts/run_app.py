#!/usr/bin/env python3
"""
Main application runner for the trading system.
Handles startup, configuration, and graceful shutdown.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.config.config_loader import ConfigLoader
from trading.core.trading_agent import TradingAgent
from trading.utils.logger_utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting trading application...")

        # Load configuration
        config = ConfigLoader()
        logger.info("Configuration loaded successfully")

        # Initialize trading agent
        agent = TradingAgent(config)
        logger.info("Trading agent initialized")

        # Start the agent
        agent.start()
        logger.info("Trading agent started successfully")

        # Keep running until interrupted
        try:
            while True:
                agent.run_iteration()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            agent.shutdown()
            logger.info("Trading application shutdown complete")

    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
