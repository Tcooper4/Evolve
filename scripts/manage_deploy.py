#!/usr/bin/env python3
"""
Deployment management script.
Provides commands for managing the application's deployment.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

class DeploymentManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the deployment manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.deploy_dir = Path("deploy")
        self.deploy_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration."""
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)
        
        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)
        
        logging.config.dictConfig(log_config)

    def build_docker(self, tag: Optional[str] = None):
        """Build Docker image."""
        self.logger.info("Building Docker image...")
        
        try:
            # Generate tag if not provided
            if not tag:
                tag = f"trading-app:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Build Docker image
            cmd = ["docker", "build", "-t", tag, "."]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Docker image built: {tag}")
                return True
            else:
                self.logger.error(f"Failed to build Docker image: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to build Docker image: {e}")
            return False

    def deploy_docker(self, image: str, port: int = 8501):
        """Deploy Docker container."""
        self.logger.info(f"Deploying Docker container from {image}...")
        
        try:
            # Stop existing container if running
            self.stop_docker()
            
            # Run new container
            cmd = [
                "docker", "run",
                "-d",
                "--name", "trading-app",
                "-p", f"{port}:8501",
                "--restart", "unless-stopped",
                "-v", f"{Path.cwd()}/data:/app/data",
                "-v", f"{Path.cwd()}/logs:/app/logs",
                image
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Docker container deployed")
                return True
            else:
                self.logger.error(f"Failed to deploy Docker container: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to deploy Docker container: {e}")
            return False

    def stop_docker(self):
        """Stop Docker container."""
        self.logger.info("Stopping Docker container...")
        
        try:
            # Stop container
            cmd = ["docker", "stop", "trading-app"]
            subprocess.run(cmd, capture_output=True)
            
            # Remove container
            cmd = ["docker", "rm", "trading-app"]
            subprocess.run(cmd, capture_output=True)
            
            self.logger.info("Docker container stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop Docker container: {e}")
            return False

    def deploy_kubernetes(self, namespace: str = "trading"):
        """Deploy to Kubernetes."""
        self.logger.info(f"Deploying to Kubernetes namespace: {namespace}...")
        
        try:
            # Create namespace if it doesn't exist
            cmd = ["kubectl", "create", "namespace", namespace, "--dry-run=client", "-o", "yaml"]
            subprocess.run(cmd, capture_output=True)
            
            # Apply Kubernetes manifests
            manifests_dir = self.deploy_dir / "kubernetes"
            if not manifests_dir.exists():
                self.logger.error("Kubernetes manifests not found")
                return False
            
            for manifest in manifests_dir.glob("*.yaml"):
                cmd = ["kubectl", "apply", "-f", str(manifest), "-n", namespace]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to apply {manifest}: {result.stderr}")
                    return False
            
            self.logger.info("Kubernetes deployment completed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to deploy to Kubernetes: {e}")
            return False

    def rollback_kubernetes(self, namespace: str = "trading"):
        """Rollback Kubernetes deployment."""
        self.logger.info(f"Rolling back Kubernetes deployment in namespace: {namespace}...")
        
        try:
            # Get current deployment
            cmd = ["kubectl", "get", "deployment", "-n", namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to get deployment: {result.stderr}")
                return False
            
            deployment = json.loads(result.stdout)
            
            # Rollback to previous version
            cmd = ["kubectl", "rollout", "undo", "deployment", "-n", namespace]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to rollback: {result.stderr}")
                return False
            
            self.logger.info("Kubernetes rollback completed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to rollback Kubernetes deployment: {e}")
            return False

    def check_deployment(self, namespace: str = "trading"):
        """Check deployment status."""
        self.logger.info("Checking deployment status...")
        
        try:
            # Check Kubernetes deployment
            cmd = ["kubectl", "get", "deployment", "-n", namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to get deployment status: {result.stderr}")
                return False
            
            deployment = json.loads(result.stdout)
            
            # Print status
            print("\nDeployment Status:")
            for item in deployment["items"]:
                name = item["metadata"]["name"]
                available = item["status"]["availableReplicas"]
                desired = item["status"]["replicas"]
                print(f"\n{name}:")
                print(f"  Available: {available}/{desired}")
                print(f"  Conditions:")
                for condition in item["status"]["conditions"]:
                    status = "✓" if condition["status"] == "True" else "✗"
                    print(f"    {status} {condition['type']}: {condition['message']}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to check deployment status: {e}")
            return False

    def scale_deployment(self, replicas: int, namespace: str = "trading"):
        """Scale deployment."""
        self.logger.info(f"Scaling deployment to {replicas} replicas...")
        
        try:
            # Scale deployment
            cmd = ["kubectl", "scale", "deployment", "--replicas", str(replicas), "-n", namespace]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to scale deployment: {result.stderr}")
                return False
            
            self.logger.info(f"Deployment scaled to {replicas} replicas")
            return True
        except Exception as e:
            self.logger.error(f"Failed to scale deployment: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deployment Manager")
    parser.add_argument(
        "command",
        choices=["build", "deploy", "stop", "k8s-deploy", "k8s-rollback", "check", "scale"],
        help="Command to execute"
    )
    parser.add_argument(
        "--tag",
        help="Docker image tag"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Docker deployment"
    )
    parser.add_argument(
        "--namespace",
        default="trading",
        help="Kubernetes namespace"
    )
    parser.add_argument(
        "--replicas",
        type=int,
        help="Number of replicas for scaling"
    )
    
    args = parser.parse_args()
    manager = DeploymentManager()
    
    commands = {
        "build": lambda: manager.build_docker(args.tag),
        "deploy": lambda: manager.deploy_docker(args.tag, args.port) if args.tag else False,
        "stop": manager.stop_docker,
        "k8s-deploy": lambda: manager.deploy_kubernetes(args.namespace),
        "k8s-rollback": lambda: manager.rollback_kubernetes(args.namespace),
        "check": lambda: manager.check_deployment(args.namespace),
        "scale": lambda: manager.scale_deployment(args.replicas, args.namespace) if args.replicas else False
    }
    
    if args.command in commands:
        if args.command == "deploy" and not args.tag:
            parser.error("deploy requires --tag")
        elif args.command == "scale" and not args.replicas:
            parser.error("scale requires --replicas")
        
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 