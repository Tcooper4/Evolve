#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
import yaml
import docker
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load deployment configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        sys.exit(1)

def check_requirements() -> bool:
    """Check if all requirements are met."""
    try:
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error("Python 3.8 or higher is required")
            return False
            
        # Check Docker
        try:
            docker_client = docker.from_env()
            docker_client.ping()
        except Exception as e:
            logger.error(f"Docker is not running: {str(e)}")
            return False
            
        # Check Redis
        try:
            import redis
            redis_client = redis.Redis(host='localhost', port=6379)
            redis_client.ping()
        except Exception as e:
            logger.error(f"Redis is not running: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking requirements: {str(e)}")
        return False

def install_dependencies() -> bool:
    """Install Python dependencies."""
    try:
        # Install production dependencies
        subprocess.run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt"
        ], check=True)
        
        # Install development dependencies if needed
        if os.getenv("ENVIRONMENT") == "development":
            subprocess.run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "requirements-dev.txt"
            ], check=True)
            
        return True
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

def build_docker_images(config: Dict) -> bool:
    """Build Docker images."""
    try:
        docker_client = docker.from_env()
        
        # Build base image
        logger.info("Building base image...")
        docker_client.images.build(
            path=".",
            tag=f"{config['docker']['registry']}/automation-base:latest",
            dockerfile="Dockerfile.base"
        )
        
        # Build service images
        for service in config["services"]:
            logger.info(f"Building {service['name']} image...")
            docker_client.images.build(
                path=service["path"],
                tag=f"{config['docker']['registry']}/{service['name']}:{service['version']}",
                dockerfile=service["dockerfile"]
            )
            
        return True
        
    except Exception as e:
        logger.error(f"Error building Docker images: {str(e)}")
        return False

def deploy_services(config: Dict) -> bool:
    """Deploy services."""
    try:
        docker_client = docker.from_env()
        
        # Create network if it doesn't exist
        try:
            docker_client.networks.get(config["docker"]["network"])
        except docker.errors.NotFound:
            docker_client.networks.create(
                config["docker"]["network"],
                driver="bridge"
            )
            
        # Deploy services
        for service in config["services"]:
            logger.info(f"Deploying {service['name']}...")
            
            # Stop and remove existing container
            try:
                container = docker_client.containers.get(service["name"])
                container.stop()
                container.remove()
            except docker.errors.NotFound:
                pass
                
            # Create and start new container
            docker_client.containers.run(
                image=f"{config['docker']['registry']}/{service['name']}:{service['version']}",
                name=service["name"],
                detach=True,
                network=config["docker"]["network"],
                environment=service["environment"],
                volumes=service.get("volumes", []),
                ports=service.get("ports", {}),
                restart_policy={"Name": "always"}
            )
            
        return True
        
    except Exception as e:
        logger.error(f"Error deploying services: {str(e)}")
        return False

def run_migrations() -> bool:
    """Run database migrations."""
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "alembic",
            "upgrade",
            "head"
        ], check=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        return False

def verify_deployment(config: Dict) -> bool:
    """Verify deployment."""
    try:
        # Check if services are running
        docker_client = docker.from_env()
        
        for service in config["services"]:
            container = docker_client.containers.get(service["name"])
            if container.status != "running":
                logger.error(f"Service {service['name']} is not running")
                return False
                
        # Check if services are healthy
        for service in config["services"]:
            if "health_check" in service:
                response = requests.get(service["health_check"])
                if response.status_code != 200:
                    logger.error(f"Service {service['name']} is not healthy")
                    return False
                    
        return True
        
    except Exception as e:
        logger.error(f"Error verifying deployment: {str(e)}")
        return False

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy automation system")
    parser.add_argument(
        "--config",
        default="deployment.yaml",
        help="Path to deployment configuration file"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip requirement checks"
    )
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check requirements
    if not args.skip_checks and not check_requirements():
        sys.exit(1)
        
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
        
    # Build Docker images
    if not build_docker_images(config):
        sys.exit(1)
        
    # Run migrations
    if not run_migrations():
        sys.exit(1)
        
    # Deploy services
    if not deploy_services(config):
        sys.exit(1)
        
    # Verify deployment
    if not verify_deployment(config):
        sys.exit(1)
        
    logger.info("Deployment completed successfully")

if __name__ == "__main__":
    main() 