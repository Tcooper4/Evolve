#!/usr/bin/env python3
"""
Agent Loop Runner

Simple script to run the autonomous 3-agent model management system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from trading.agents.agent_loop_manager import AgentLoopManager


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = Path(log_config.get('file', 'trading/agents/logs/agent_loop.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "agent_config.json"
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return {'success': True, 'result': json.load(f), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Autonomous Model Management Agent Loop")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--cycle-interval", type=int, help="Cycle interval in seconds")
    parser.add_argument("--max-models", type=int, help="Maximum number of active models")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help="Logging level")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.cycle_interval:
            config['agent_loop']['cycle_interval'] = args.cycle_interval
        if args.max_models:
            config['agent_loop']['max_models'] = args.max_models
        if args.log_level:
            config['logging']['level'] = args.log_level
        
        # Setup logging
        setup_logging(config)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Agent Loop Runner")
        logger.info(f"Configuration loaded from: {args.config or 'default'}")
        
        # Create and start agent loop manager
        manager = AgentLoopManager(config)
        
        # Print initial status
        status = manager.get_loop_status()
        logger.info(f"Agent Loop Status: {status}")
        
        # Start the loop
        await manager.start_loop()
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down gracefully...")
    except FileNotFoundError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 