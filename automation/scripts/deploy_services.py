#!/usr/bin/env python3
"""
Deploy Services Script

This script orchestrates the deployment of all services defined in the deployment configuration
using the updated deploy.py script.
"""

import os
import sys
import yaml
import logging
import argparse
import subprocess
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_deployment_config(config_path: str) -> Dict:
    """Load the deployment configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load deployment config: {e}")
        sys.exit(1)

def deploy_services(config: Dict) -> bool:
    """Deploy all services defined in the configuration."""
    for service_name, service_config in config['services'].items():
        version = service_config.get('version', 'latest')
        logger.info(f"Deploying service: {service_name} with version: {version}")
        
        # Construct the command to run deploy.py
        cmd = [
            sys.executable,
            'deploy.py',
            '--config', 'deployment.yaml',
            '--service', service_name,
            '--version', version
        ]
        
        # Run the deployment command
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully deployed service: {service_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy service {service_name}: {e}")
            return False
    
    return True

def main():
    """Main function to orchestrate service deployment."""
    parser = argparse.ArgumentParser(description='Deploy all automation services')
    parser.add_argument('--config', required=True, help='Path to deployment config')
    
    args = parser.parse_args()
    
    # Load deployment configuration
    config = load_deployment_config(args.config)
    
    # Deploy services
    if not deploy_services(config):
        sys.exit(1)
    
    logger.info("All services deployed successfully.")

if __name__ == '__main__':
    main() 