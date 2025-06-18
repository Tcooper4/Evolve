#!/usr/bin/env python3
"""
Secret Management Script

This script helps manage environment variables and secrets for the automation system.
It provides functionality to:
1. Generate secure secrets
2. Update environment variables
3. Validate configuration
4. Rotate secrets
"""

import os
import sys
import json
import yaml
import secrets
import string
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecretManager:
    def __init__(self, config_dir: str = "automation/config"):
        self.config_dir = Path(config_dir)
        self.env_file = Path(".env")
        self.example_env_file = Path(".env.example")
        
    def generate_secure_secret(self, length: int = 32) -> str:
        """Generate a secure random secret."""
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_jwt_secret(self) -> str:
        """Generate a secure JWT secret."""
        return self.generate_secure_secret(64)
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return self.generate_secure_secret(32)
    
    def load_env_file(self) -> Dict[str, str]:
        """Load environment variables from .env file."""
        if not self.env_file.exists():
            return {}
        
        env_vars = {}
        with open(self.env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        return env_vars
    
    def save_env_file(self, env_vars: Dict[str, str]) -> None:
        """Save environment variables to .env file."""
        with open(self.env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
    
    def update_secret(self, key: str, value: Optional[str] = None) -> None:
        """Update a secret in the .env file."""
        env_vars = self.load_env_file()
        
        if value is None:
            if key == "JWT_SECRET":
                value = self.generate_jwt_secret()
            elif "API_KEY" in key:
                value = self.generate_api_key()
            else:
                value = self.generate_secure_secret()
        
        env_vars[key] = value
        self.save_env_file(env_vars)
        logger.info(f"Updated {key}")
    
    def rotate_secrets(self) -> None:
        """Rotate all secrets in the .env file."""
        env_vars = self.load_env_file()
        rotated = False
        
        for key in env_vars:
            if any(secret_type in key for secret_type in ["SECRET", "KEY", "PASSWORD", "TOKEN"]):
                self.update_secret(key)
                rotated = True
        
        if rotated:
            logger.info("Rotated all secrets")
        else:
            logger.info("No secrets found to rotate")
    
    def validate_config(self) -> bool:
        """Validate configuration files and environment variables."""
        required_vars = [
            "ENVIRONMENT",
            "OPENAI_API_KEY",
            "REDIS_PASSWORD",
            "CONSUL_TOKEN",
            "JWT_SECRET",
            "EMAIL_USERNAME",
            "EMAIL_PASSWORD",
            "EMAIL_FROM",
            "SLACK_WEBHOOK_URL"
        ]
        
        env_vars = self.load_env_file()
        missing_vars = [var for var in required_vars if var not in env_vars]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        # Validate SSL certificate paths
        ssl_paths = [
            "SSL_CERT_PATH",
            "SSL_KEY_PATH",
            "API_CERT_PATH",
            "API_KEY_PATH",
            "MONITORING_CERT_PATH",
            "MONITORING_KEY_PATH"
        ]
        
        for path_var in ssl_paths:
            if path_var in env_vars:
                path = Path(env_vars[path_var])
                if not path.exists():
                    logger.error(f"SSL certificate/key not found: {path}")
                    return False
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Manage environment variables and secrets")
    parser.add_argument("--generate", action="store_true", help="Generate new secrets")
    parser.add_argument("--rotate", action="store_true", help="Rotate all secrets")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--update", metavar="KEY=VALUE", help="Update a specific secret")
    
    args = parser.parse_args()
    manager = SecretManager()
    
    if args.generate:
        manager.update_secret("JWT_SECRET")
        manager.update_secret("OPENAI_API_KEY")
        manager.update_secret("REDIS_PASSWORD")
        manager.update_secret("CONSUL_TOKEN")
        manager.update_secret("EMAIL_PASSWORD")
        manager.update_secret("SLACK_WEBHOOK_URL")
    
    if args.rotate:
        manager.rotate_secrets()
    
    if args.validate:
        if manager.validate_config():
            logger.info("Configuration is valid")
            sys.exit(0)
        else:
            logger.error("Configuration validation failed")
            sys.exit(1)
    
    if args.update:
        try:
            key, value = args.update.split("=", 1)
            manager.update_secret(key, value)
        except ValueError:
            logger.error("Invalid format for --update. Use KEY=VALUE")
            sys.exit(1)

if __name__ == "__main__":
    main() 