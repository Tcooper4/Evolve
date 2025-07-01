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
from pathlib import Path
from typing import Dict, Any

from trading.live_market_runner import create_live_market_runner

def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading/live/logs/launcher.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
def load_config(config_path: str = "trading/live/config.json") -> Dict[str, Any]:
    """Load configuration from file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        default_config = {
            "symbols": ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
            "market_data_config": {
                "cache_size": 1000,
                "update_threshold": 5,
                "max_retries": 3
            },
            "triggers": {
                "model_builder": {
                    "trigger_type": "time_based",
                    "interval_seconds": 3600,
                    "enabled": True
                },
                "performance_critic": {
                    "trigger_type": "time_based",
                    "interval_seconds": 1800,
                    "enabled": True
                },
                "execution_agent": {
                    "trigger_type": "price_move",
                    "price_move_threshold": 0.005,
                    "enabled": True
                }
            }
        }
        
        # Save default config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config

async def main():
    """Main function."""
    print("🚀 Launching Live Market Runner")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config()
    
    # Create runner
    runner = create_live_market_runner(config)
    
    print(f"✅ Live Market Runner created")
    print(f"📊 Symbols: {config['symbols']}")
    print(f"🤖 Agents: {list(config['triggers'].keys())}")
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down...")
        asyncio.create_task(runner.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the runner
    print(f"\n🔄 Starting Live Market Runner...")
    await runner.start()
    
    print(f"✅ Live Market Runner is running!")
    print(f"   Press Ctrl+C to stop")
    print(f"   Logs: trading/live/logs/")
    print(f"   Forecasts: trading/live/forecast_results.json")
    
    # Keep running
    try:
        while runner.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down...")
    finally:
        await runner.stop()
        print(f"✅ Live Market Runner stopped")

if __name__ == "__main__":
    asyncio.run(main()) 