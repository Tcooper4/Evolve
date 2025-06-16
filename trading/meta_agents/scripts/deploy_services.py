"""
Service Deployment

This module implements service deployment functionality.

Note: This module was adapted from the legacy automation/scripts/deploy_services.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
import subprocess
import shutil
import os

async def deploy_services(config_path: str) -> None:
    """Deploy services."""
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
        log_path = Path("logs/deployment")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "deploy_services.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Create necessary directories
        for service in config['services']:
            service_dir = Path(service['path'])
            service_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {service_dir}")
        
        # Copy service files
        for service in config['services']:
            source_dir = Path(service['source'])
            target_dir = Path(service['path'])
            
            if source_dir.exists():
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                logger.info(f"Copied service files from {source_dir} to {target_dir}")
            else:
                logger.warning(f"Source directory not found: {source_dir}")
        
        # Install dependencies
        for service in config['services']:
            if 'requirements' in service:
                req_file = Path(service['path']) / service['requirements']
                if req_file.exists():
                    subprocess.run(['pip', 'install', '-r', str(req_file)], check=True)
                    logger.info(f"Installed dependencies for {service['name']}")
        
        # Start services
        for service in config['services']:
            if 'start_command' in service:
                service_dir = Path(service['path'])
                os.chdir(service_dir)
                subprocess.Popen(service['start_command'].split())
                logger.info(f"Started service: {service['name']}")
        
        logger.info("Service deployment completed successfully")
    except Exception as e:
        logging.error(f"Error deploying services: {str(e)}")
        raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy services')
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    try:
        asyncio.run(deploy_services(args.config))
    except KeyboardInterrupt:
        logging.info("Service deployment interrupted")
    except Exception as e:
        logging.error(f"Error deploying services: {str(e)}")
        raise

if __name__ == '__main__':
    main() 