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
from typing import Dict, Any

from trading.agents.execution_agent import ExecutionAgent, create_execution_agent
from trading.agents.base_agent_interface import AgentConfig


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading/agents/logs/execution_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
def load_config(config_path: str = "trading/agents/execution_config.json") -> Dict[str, Any]:
    """Load configuration from file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            return {'success': True, 'result': json.load(f), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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
                    "min_fee": 1.0
                }
            },
            "service": {
                "port": 8080,
                "host": "localhost",
                "debug": False,
                "auto_start": True
            }
        }
        
        # Save default config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config


async def main():
    """Main function."""
    print("🚀 Launching Execution Agent")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config()
    agent_config = config["agent"]
    
    # Create execution agent
    agent_config_obj = AgentConfig(**agent_config)
    execution_agent = ExecutionAgent(agent_config_obj)
    
    print(f"✅ Execution Agent initialized")
    print(f"📊 Execution mode: {execution_agent.execution_mode.value}")
    print(f"💰 Max positions: {execution_agent.max_positions}")
    print(f"🎯 Min confidence: {execution_agent.min_confidence}")
    print(f"📈 Max slippage: {execution_agent.max_slippage}")
    
    # Show initial portfolio status
    portfolio_status = execution_agent.get_portfolio_status()
    print(f"\n💰 Initial Portfolio Status:")
    print(f"  Cash: ${portfolio_status['cash']:.2f}")
    print(f"  Equity: ${portfolio_status['equity']:.2f}")
    print(f"  Available Capital: ${portfolio_status['available_capital']:.2f}")
    print(f"  Open Positions: {len(portfolio_status['open_positions'])}")
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down...")
        sys.exit(0)
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"\n🎯 Execution Agent is running...")
    print(f"   Press Ctrl+C to stop")
    print(f"   Logs: trading/agents/logs/execution_agent.log")
    print(f"   Trade log: trading/agents/logs/trade_log.json")
    
    # Keep the agent running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down Execution Agent...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise
    finally:
        print(f"✅ Execution Agent stopped")


if __name__ == "__main__":
    asyncio.run(main()) 