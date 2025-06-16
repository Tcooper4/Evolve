"""
Secret Management

This module implements secret management functionality.

Note: This module was adapted from the legacy automation/scripts/manage_secrets.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecretManager:
    """Manages secrets and encryption."""
    
    def __init__(self, config_path: str):
        """Initialize secret manager."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.setup_encryption()
    
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/secrets")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "secret_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    self.config = json.load(f)
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
    
    def setup_encryption(self) -> None:
        """Set up encryption."""
        try:
            # Generate key from password
            password = self.config['encryption']['password'].encode()
            salt = self.config['encryption']['salt'].encode()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.fernet = Fernet(key)
            
            self.logger.info("Encryption setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up encryption: {str(e)}")
            raise
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret."""
        try:
            return self.fernet.encrypt(secret.encode()).decode()
        except Exception as e:
            self.logger.error(f"Error encrypting secret: {str(e)}")
            raise
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret."""
        try:
            return self.fernet.decrypt(encrypted_secret.encode()).decode()
        except Exception as e:
            self.logger.error(f"Error decrypting secret: {str(e)}")
            raise
    
    def save_secret(self, name: str, value: str) -> None:
        """Save a secret."""
        try:
            secrets_file = Path(self.config['secrets']['file'])
            secrets_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing secrets
            if secrets_file.exists():
                with open(secrets_file, 'r') as f:
                    secrets = json.load(f)
            else:
                secrets = {}
            
            # Add new secret
            secrets[name] = self.encrypt_secret(value)
            
            # Save secrets
            with open(secrets_file, 'w') as f:
                json.dump(secrets, f, indent=2)
            
            self.logger.info(f"Saved secret: {name}")
        except Exception as e:
            self.logger.error(f"Error saving secret: {str(e)}")
            raise
    
    def get_secret(self, name: str) -> str:
        """Get a secret."""
        try:
            secrets_file = Path(self.config['secrets']['file'])
            
            if not secrets_file.exists():
                raise FileNotFoundError(f"Secrets file not found: {secrets_file}")
            
            with open(secrets_file, 'r') as f:
                secrets = json.load(f)
            
            if name not in secrets:
                raise KeyError(f"Secret not found: {name}")
            
            return self.decrypt_secret(secrets[name])
        except Exception as e:
            self.logger.error(f"Error getting secret: {str(e)}")
            raise
    
    def delete_secret(self, name: str) -> None:
        """Delete a secret."""
        try:
            secrets_file = Path(self.config['secrets']['file'])
            
            if not secrets_file.exists():
                raise FileNotFoundError(f"Secrets file not found: {secrets_file}")
            
            with open(secrets_file, 'r') as f:
                secrets = json.load(f)
            
            if name not in secrets:
                raise KeyError(f"Secret not found: {name}")
            
            del secrets[name]
            
            with open(secrets_file, 'w') as f:
                json.dump(secrets, f, indent=2)
            
            self.logger.info(f"Deleted secret: {name}")
        except Exception as e:
            self.logger.error(f"Error deleting secret: {str(e)}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage secrets')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--action', required=True, choices=['save', 'get', 'delete'],
                      help='Action to perform')
    parser.add_argument('--name', required=True, help='Secret name')
    parser.add_argument('--value', help='Secret value (required for save action)')
    args = parser.parse_args()
    
    try:
        manager = SecretManager(args.config)
        
        if args.action == 'save':
            if not args.value:
                raise ValueError("Value is required for save action")
            manager.save_secret(args.name, args.value)
        elif args.action == 'get':
            value = manager.get_secret(args.name)
            print(value)
        elif args.action == 'delete':
            manager.delete_secret(args.name)
    except KeyboardInterrupt:
        logging.info("Secret management interrupted")
    except Exception as e:
        logging.error(f"Error managing secrets: {str(e)}")
        raise

if __name__ == '__main__':
    main() 