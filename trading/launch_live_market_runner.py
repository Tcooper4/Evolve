#!/usr/bin/env python3
"""
Live Market Runner Launcher

Launches the LiveMarketRunner as a standalone service.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from trading.live_market_runner import create_live_market_runner


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("trading/live/logs/launcher.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return {
        "success": True,
        "message": "Initialization completed",
        "timestamp": datetime.now().isoformat(),
    }


def load_config(config_path: str = "trading/live/config.json") -> Dict[str, Any]:
    """Load configuration from file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        # Default configuration
        default_config = {
            "symbols": ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
            "market_data_config": {
                "cache_size": 1000,
                "update_threshold": 5,
                "max_retries": 3,
            },
            "triggers": {
                "model_builder": {
                    "trigger_type": "time_based",
                    "interval_seconds": 3600,
                    "enabled": True,
                },
                "performance_critic": {
                    "trigger_type": "time_based",
                    "interval_seconds": 1800,
                    "enabled": True,
                },
                "execution_agent": {
                    "trigger_type": "price_move",
                    "price_move_threshold": 0.005,
                    "enabled": True,
                },
            },
        }

        # Save default config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=2)

        return default_config


async def main():
    """Main function."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Launching Live Market Runner")
    logger.info("=" * 50)

    try:
        # Setup logging
        setup_logging()

        # Load configuration
        try:
            config = load_config()
            logger.info("[OK] Configuration loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"[FAIL] Configuration file not found: {e}")
            logger.info("[NOTE] Creating default configuration...")
            config = load_config()  # This will create default config
        except json.JSONDecodeError as e:
            logger.error(f"[FAIL] Invalid JSON in configuration file: {e}")
            logger.info("[NOTE] Using default configuration...")
            config = load_config()  # This will create default config
        except Exception as e:
            logger.error(f"[FAIL] Error loading configuration: {e}")
            logger.info("[NOTE] Using default configuration...")
            config = load_config()  # This will create default config

        # Create runner
        try:
            runner = create_live_market_runner(config)
            logger.info("[OK] Live Market Runner created successfully")
        except Exception as e:
            logger.error(f"[FAIL] Failed to create Live Market Runner: {e}")
            logger.error("[STOP] Exiting due to initialization failure")
            return

        logger.info(f"[DATA] Symbols: {config['symbols']}")
        logger.info(f"[AGENTS] Agents: {list(config['triggers'].keys())}")

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"\n[STOP] Received signal {signum}, shutting down...")
            asyncio.create_task(runner.stop())
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the runner
        logger.info(f"\n[START] Starting Live Market Runner...")
        try:
            await runner.start()
            logger.info("[OK] Live Market Runner started successfully")
        except Exception as e:
            logger.error(f"[FAIL] Failed to start Live Market Runner: {e}")
            logger.error("[STOP] Exiting due to startup failure")
            return

        # Add success log on launch
        logger.info(f"[OK] Live Market Runner initialized.")

        logger.info(f"[OK] Live Market Runner is running!")
        logger.info(f"   Press Ctrl+C to stop")
        logger.info(f"   Logs: trading/live/logs/")
        logger.info(f"   Forecasts: trading/live/forecast_results.json")

        # Keep running
        try:
            while runner.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"\n[STOP] Shutting down...")
        except Exception as e:
            logger.error(f"[FAIL] Unexpected error during execution: {e}")
        finally:
            try:
                await runner.stop()
                logger.info(f"[OK] Live Market Runner stopped gracefully")
            except Exception as e:
                logger.error(f"[FAIL] Error during shutdown: {e}")

    except Exception as e:
        logger.error(f"[FAIL] Critical error in main function: {e}")
        logger.error("[STOP] Live Market Runner failed to start")
        return


if __name__ == "__main__":
    asyncio.run(main())
