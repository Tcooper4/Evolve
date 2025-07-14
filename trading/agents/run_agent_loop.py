#!/usr/bin/env python3
"""
Agent Loop Runner

This script runs the main agent loop for continuous trading operations.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading.agents.base_agent_interface import AgentConfig
from trading.agents.execution_agent import ExecutionAgent
from trading.portfolio.portfolio_manager import PortfolioManager

# Global variables
running = True
execution_agent = None
portfolio_manager = None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global running
    logger.info("\nReceived interrupt signal, shutting down gracefully...")
    running = False


async def run_agent_loop():
    """Run the main agent loop."""
    global execution_agent, portfolio_manager

    try:
        # Initialize components
        config = {
            "name": "agent_loop",
            "enabled": True,
            "custom_config": {
                "execution_mode": "simulation",
                "max_positions": 3,
                "min_confidence": 0.6,
            },
        }

        agent_config = AgentConfig(**config)
        execution_agent = ExecutionAgent(agent_config)
        portfolio_manager = PortfolioManager()

        # Initialize portfolio
        await portfolio_manager.initialize()

        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Track number of loops completed per agent
        loop_counts = {"execution_agent": 0}

        # Main loop
        while running:
            # Agent logic here
            loop_counts["execution_agent"] += 1
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if execution_agent:
            await execution_agent.shutdown()


if __name__ == "__main__":
    asyncio.run(run_agent_loop())
