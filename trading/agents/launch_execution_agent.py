#!/usr/bin/env python3
"""
Execution Agent Launcher

Launches the Execution Agent as a standalone service.
"""

import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict

from trading.agents.base_agent_interface import AgentConfig
from trading.agents.execution_agent import ExecutionAgent
from trading.portfolio.portfolio_manager import PortfolioManager


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("trading/agents/logs/execution_agent.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return {
        "success": True,
        "message": "Initialization completed",
        "timestamp": datetime.now().isoformat(),
    }


def load_config(
    config_path: str = "trading/agents/execution_config.json",
) -> Dict[str, Any]:
    """Load configuration from file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        # Default configuration
        default_config = {
            "agent": {
                "name": "execution_agent",
                "enabled": True,
                "priority": 1,
                "max_concurrent_runs": 1,
                "timeout_seconds": 300,
                "retry_attempts": 3,
                "custom_config": {
                    "execution_mode": "simulation",
                    "max_positions": 10,
                    "min_confidence": 0.7,
                    "max_slippage": 0.001,
                    "execution_delay": 1.0,
                    "risk_per_trade": 0.02,
                    "max_position_size": 0.2,
                    "base_fee": 0.001,
                    "min_fee": 1.0,
                },
            },
            "service": {
                "port": 8080,
                "host": "localhost",
                "debug": False,
                "auto_start": True,
            },
        }

        # Save default config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=2)

        return default_config


async def main():
    """Main function."""
    global execution_agent, portfolio_manager

    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)

        # Load configuration
        config = load_config()
        agent_config = config["agent"]

        # Create execution agent
        agent_config_obj = AgentConfig(**agent_config)
        execution_agent = ExecutionAgent(agent_config_obj)
        portfolio_manager = PortfolioManager()

        logger.info("🚀 Launching Execution Agent")
        logger.info("=" * 50)

        # Initialize portfolio
        await portfolio_manager.initialize()

        logger.info(f"✅ Execution Agent initialized")
        logger.info(f"📊 Execution mode: {execution_agent.execution_mode.value}")
        logger.info(f"💰 Max positions: {execution_agent.max_positions}")
        logger.info(f"🎯 Min confidence: {execution_agent.min_confidence}")
        logger.info(f"📈 Max slippage: {execution_agent.max_slippage}")

        # Show initial portfolio status
        portfolio_status = await portfolio_manager.get_portfolio_status()
        logger.info(f"\n💰 Initial Portfolio Status:")
        logger.info(f"  Cash: ${portfolio_status['cash']:.2f}")
        logger.info(f"  Equity: ${portfolio_status['equity']:.2f}")
        logger.info(
            f"  Available Capital: ${portfolio_status['available_capital']:.2f}"
        )
        logger.info(f"  Open Positions: {len(portfolio_status['open_positions'])}")

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"\n🛑 Received signal {signum}, shutting down...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(f"\n🎯 Execution Agent is running...")
        logger.info(f"   Press Ctrl+C to stop")
        logger.info(f"   Logs: trading/agents/logs/execution_agent.log")
        logger.info(f"   Trade log: trading/agents/logs/trade_log.json")

        # Keep the agent running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Shutting down Execution Agent...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise
    finally:
        if execution_agent:
            await execution_agent.shutdown()
        logger.info(f"✅ Execution Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())
