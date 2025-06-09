#!/usr/bin/env python3
"""
Deployment Script

This script handles the deployment of the automation system services,
including configuration, verification, and rollback capabilities.
"""

import os
import sys
import yaml
import json
import time
import logging
import argparse
import subprocess
from typing import Dict, List, Optional
from pathlib import Path
import docker
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentConfig:
    """Deployment configuration."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.docker_client = docker.from_env()
        self.services = self.config.get('services', {})
        self.health_checks = self.config.get('health_checks', {})
        self.rollback_config = self.config.get('rollback', {})
    
    def _load_config(self) -> Dict:
        """Load deployment configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load deployment config: {e}")
            sys.exit(1)
    
    def get_service_config(self, service_name: str) -> Dict:
        """Get configuration for a specific service."""
        return self.services.get(service_name, {})
    
    def get_health_check(self, service_name: str) -> Dict:
        """Get health check configuration for a service."""
        return self.health_checks.get(service_name, {})

class DeploymentManager:
    """Manages the deployment process."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_history = []
        self.current_deployment = None
    
    def deploy_service(self, service_name: str, version: str) -> bool:
        """Deploy a single service."""
        try:
            service_config = self.config.get_service_config(service_name)
            if not service_config:
                logger.error(f"Service {service_name} not found in configuration")
                return False
            
            # Build and push Docker image
            if not self._build_and_push_image(service_name, version):
                return False
            
            # Deploy service
            if not self._deploy_service(service_name, version):
                return False
            
            # Verify deployment
            if not self._verify_deployment(service_name):
                return False
            
            # Update deployment history
            self._update_deployment_history(service_name, version)
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy service {service_name}: {e}")
            return False
    
    def _build_and_push_image(self, service_name: str, version: str) -> bool:
        """Build and push Docker image for a service."""
        try:
            service_config = self.config.get_service_config(service_name)
            dockerfile_path = service_config.get('dockerfile_path')
            registry = service_config.get('registry')
            
            # Build image
            image_name = f"{registry}/{service_name}:{version}"
            logger.info(f"Building image {image_name}")
            
            self.config.docker_client.images.build(
                path=dockerfile_path,
                tag=image_name,
                rm=True
            )
            
            # Push image
            logger.info(f"Pushing image {image_name}")
            self.config.docker_client.images.push(image_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build/push image for {service_name}: {e}")
            return False
    
    def _deploy_service(self, service_name: str, version: str) -> bool:
        """Deploy a service using Docker Swarm."""
        try:
            service_config = self.config.get_service_config(service_name)
            registry = service_config.get('registry')
            image_name = f"{registry}/{service_name}:{version}"
            
            # Create or update service
            service_spec = {
                'name': service_name,
                'image': image_name,
                'replicas': service_config.get('replicas', 1),
                'update_config': {
                    'parallelism': 1,
                    'delay': 10,
                    'failure_action': 'rollback'
                },
                'restart_policy': {
                    'condition': 'on-failure',
                    'max_attempts': 3
                }
            }
            
            # Add environment variables
            if 'environment' in service_config:
                service_spec['env'] = service_config['environment']
            
            # Add networks
            if 'networks' in service_config:
                service_spec['networks'] = service_config['networks']
            
            # Add volumes
            if 'volumes' in service_config:
                service_spec['mounts'] = service_config['volumes']
            
            # Deploy service
            logger.info(f"Deploying service {service_name}")
            self.config.docker_client.services.create(**service_spec)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy service {service_name}: {e}")
            return False
    
    def _verify_deployment(self, service_name: str) -> bool:
        """Verify service deployment using health checks."""
        try:
            health_check = self.config.get_health_check(service_name)
            if not health_check:
                logger.warning(f"No health check configured for {service_name}")
                return True
            
            endpoint = health_check.get('endpoint')
            max_retries = health_check.get('max_retries', 3)
            retry_delay = health_check.get('retry_delay', 10)
            
            for i in range(max_retries):
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Health check passed for {service_name}")
                        return True
                except Exception as e:
                    logger.warning(f"Health check attempt {i+1} failed: {e}")
                
                if i < max_retries - 1:
                    time.sleep(retry_delay)
            
            logger.error(f"Health check failed for {service_name}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to verify deployment for {service_name}: {e}")
            return False
    
    def _update_deployment_history(self, service_name: str, version: str):
        """Update deployment history."""
        deployment = {
            'service': service_name,
            'version': version,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'success'
        }
        self.deployment_history.append(deployment)
        self.current_deployment = deployment
    
    def rollback(self, service_name: str) -> bool:
        """Rollback a service to its previous version."""
        try:
            # Find previous deployment
            previous_deployment = None
            for deployment in reversed(self.deployment_history):
                if deployment['service'] == service_name:
                    previous_deployment = deployment
                    break
            
            if not previous_deployment:
                logger.error(f"No previous deployment found for {service_name}")
                return False
            
            # Rollback to previous version
            return self.deploy_service(
                service_name,
                previous_deployment['version']
            )
            
        except Exception as e:
            logger.error(f"Failed to rollback {service_name}: {e}")
            return False

def check_deployment_config(config_path: str) -> bool:
    """Check if the deployment configuration file exists and is valid."""
    if not os.path.exists(config_path):
        logger.error(f"Deployment configuration file not found: {config_path}")
        return False
    try:
        with open(config_path, 'r') as f:
            yaml.safe_load(f)
        return True
    except Exception as e:
        logger.error(f"Invalid deployment configuration file: {e}")
        return False

def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description='Deploy automation services')
    parser.add_argument('--config', required=True, help='Path to deployment config')
    parser.add_argument('--service', required=True, help='Service to deploy')
    parser.add_argument('--version', required=True, help='Version to deploy')
    parser.add_argument('--rollback', action='store_true', help='Rollback service')
    
    args = parser.parse_args()
    
    # Check deployment configuration
    if not check_deployment_config(args.config):
        sys.exit(1)
    
    # Initialize deployment
    config = DeploymentConfig(args.config)
    manager = DeploymentManager(config)
    
    if args.rollback:
        success = manager.rollback(args.service)
    else:
        success = manager.deploy_service(args.service, args.version)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 