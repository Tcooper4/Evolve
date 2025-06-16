"""
Monitoring Setup

This module implements monitoring setup functionality.

Note: This module was adapted from the legacy automation/scripts/monitoring_setup.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
from ..metrics_collector import MetricsCollector
from ..alert_manager import AlertManager

async def setup_monitoring(config_path: str) -> None:
    """Set up monitoring components."""
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        # Setup logging
        log_path = Path("logs/monitoring")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "monitoring_setup.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector(config)
        await metrics_collector.initialize()
        logger.info("Initialized metrics collector")
        
        # Initialize alert manager
        alert_manager = AlertManager(config)
        await alert_manager.initialize()
        logger.info("Initialized alert manager")
        
        logger.info("Monitoring setup completed successfully")
    except Exception as e:
        logging.error(f"Error setting up monitoring: {str(e)}")
        raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Set up monitoring')
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    try:
        asyncio.run(setup_monitoring(args.config))
    except KeyboardInterrupt:
        logging.info("Monitoring setup interrupted")
    except Exception as e:
        logging.error(f"Error setting up monitoring: {str(e)}")
        raise

if __name__ == '__main__':
    main() 