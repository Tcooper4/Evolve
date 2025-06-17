#!/usr/bin/env python3
"""
Configuration management script.
Provides commands for managing application configuration and secrets.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import base64
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import boto3
import hvac
import kubernetes
from kubernetes import client, config

class ConfigManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the configuration manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.config_dir = Path("config")
        self.secrets_dir = Path("secrets")
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

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

    def generate_config(self, template: str = "default"):
        """Generate configuration files from template."""
        self.logger.info(f"Generating configuration from template: {template}")
        
        try:
            # Load template
            template_file = self.config_dir / f"templates/{template}.yaml"
            if not template_file.exists():
                raise FileNotFoundError(f"Template not found: {template_file}")
            
            with open(template_file) as f:
                template_config = yaml.safe_load(f)
            
            # Generate configurations
            for env in ["development", "staging", "production"]:
                config = self._customize_config(template_config, env)
                
                # Save configuration
                config_file = self.config_dir / f"app_config_{env}.yaml"
                with open(config_file, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                self.logger.info(f"Generated configuration for {env}: {config_file}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate configuration: {e}")
            raise

    def validate_config(self, config_path: str):
        """Validate configuration file."""
        self.logger.info(f"Validating configuration: {config_path}")
        
        try:
            # Load configuration
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = [
                "app", "server", "database", "api_keys",
                "security", "monitoring", "feature_flags"
            ]
            
            missing_sections = [
                section for section in required_sections
                if section not in config
            ]
            
            if missing_sections:
                raise ValueError(f"Missing required sections: {missing_sections}")
            
            # Validate values
            validation_results = {
                "timestamp": datetime.now().isoformat(),
                "config_path": config_path,
                "valid": True,
                "issues": []
            }
            
            # Validate app section
            if not isinstance(config["app"]["name"], str):
                validation_results["issues"].append(
                    "app.name must be a string"
                )
            
            if not isinstance(config["app"]["version"], str):
                validation_results["issues"].append(
                    "app.version must be a string"
                )
            
            # Validate server section
            if not isinstance(config["server"]["host"], str):
                validation_results["issues"].append(
                    "server.host must be a string"
                )
            
            if not isinstance(config["server"]["port"], int):
                validation_results["issues"].append(
                    "server.port must be an integer"
                )
            
            # Validate database section
            if not isinstance(config["database"]["host"], str):
                validation_results["issues"].append(
                    "database.host must be a string"
                )
            
            if not isinstance(config["database"]["port"], int):
                validation_results["issues"].append(
                    "database.port must be an integer"
                )
            
            # Update validation status
            validation_results["valid"] = len(validation_results["issues"]) == 0
            
            # Save validation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.config_dir / f"validation_{Path(config_path).stem}_{timestamp}.json"
            
            with open(results_file, "w") as f:
                json.dump(validation_results, f, indent=2)
            
            # Print validation results
            self._print_validation_results(validation_results)
            
            return validation_results["valid"]
        except Exception as e:
            self.logger.error(f"Failed to validate configuration: {e}")
            raise

    def encrypt_secrets(self, secrets: Dict[str, Any], key: Optional[str] = None):
        """Encrypt secrets."""
        self.logger.info("Encrypting secrets")
        
        try:
            # Generate or load encryption key
            if key:
                # Use provided key
                key_bytes = key.encode()
            else:
                # Generate new key
                key_bytes = Fernet.generate_key()
            
            # Create Fernet instance
            fernet = Fernet(key_bytes)
            
            # Encrypt secrets
            encrypted_secrets = {}
            for name, value in secrets.items():
                if isinstance(value, (str, int, float, bool)):
                    value_str = str(value)
                    encrypted_value = fernet.encrypt(value_str.encode())
                    encrypted_secrets[name] = base64.b64encode(encrypted_value).decode()
                else:
                    self.logger.warning(f"Skipping non-scalar value for {name}")
            
            # Save encrypted secrets
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            secrets_file = self.secrets_dir / f"secrets_{timestamp}.json"
            
            with open(secrets_file, "w") as f:
                json.dump(encrypted_secrets, f, indent=2)
            
            # Save key if generated
            if not key:
                key_file = self.secrets_dir / f"key_{timestamp}.txt"
                with open(key_file, "w") as f:
                    f.write(base64.b64encode(key_bytes).decode())
            
            self.logger.info(f"Encrypted secrets saved to {secrets_file}")
            
            return secrets_file
        except Exception as e:
            self.logger.error(f"Failed to encrypt secrets: {e}")
            raise

    def decrypt_secrets(self, secrets_file: str, key: str):
        """Decrypt secrets."""
        self.logger.info(f"Decrypting secrets from {secrets_file}")
        
        try:
            # Load encrypted secrets
            with open(secrets_file) as f:
                encrypted_secrets = json.load(f)
            
            # Create Fernet instance
            key_bytes = base64.b64decode(key)
            fernet = Fernet(key_bytes)
            
            # Decrypt secrets
            decrypted_secrets = {}
            for name, encrypted_value in encrypted_secrets.items():
                value_bytes = base64.b64decode(encrypted_value)
                decrypted_value = fernet.decrypt(value_bytes)
                decrypted_secrets[name] = decrypted_value.decode()
            
            # Save decrypted secrets
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.secrets_dir / f"decrypted_{timestamp}.json"
            
            with open(output_file, "w") as f:
                json.dump(decrypted_secrets, f, indent=2)
            
            self.logger.info(f"Decrypted secrets saved to {output_file}")
            
            return decrypted_secrets
        except Exception as e:
            self.logger.error(f"Failed to decrypt secrets: {e}")
            raise

    def sync_config(self, target: str, config_path: str):
        """Sync configuration to target system."""
        self.logger.info(f"Syncing configuration to {target}")
        
        try:
            # Load configuration
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            if target == "aws":
                # Sync to AWS Parameter Store
                ssm = boto3.client("ssm")
                
                for key, value in config.items():
                    if isinstance(value, (str, int, float, bool)):
                        ssm.put_parameter(
                            Name=f"/trading/{key}",
                            Value=str(value),
                            Type="String",
                            Overwrite=True
                        )
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            ssm.put_parameter(
                                Name=f"/trading/{key}/{subkey}",
                                Value=str(subvalue),
                                Type="String",
                                Overwrite=True
                            )
            
            elif target == "vault":
                # Sync to HashiCorp Vault
                client = hvac.Client(
                    url=self.config["vault"]["url"],
                    token=self.config["vault"]["token"]
                )
                
                for key, value in config.items():
                    if isinstance(value, (str, int, float, bool)):
                        client.secrets.kv.v2.create_or_update_secret(
                            path=f"trading/{key}",
                            secret=dict(value=str(value))
                        )
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            client.secrets.kv.v2.create_or_update_secret(
                                path=f"trading/{key}/{subkey}",
                                secret=dict(value=str(subvalue))
                            )
            
            elif target == "kubernetes":
                # Sync to Kubernetes ConfigMaps and Secrets
                config.load_kube_config()
                v1 = client.CoreV1Api()
                
                # Create ConfigMap for non-sensitive data
                config_data = {}
                for key, value in config.items():
                    if isinstance(value, (str, int, float, bool)):
                        config_data[key] = str(value)
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            config_data[f"{key}.{subkey}"] = str(subvalue)
                
                config_map = client.V1ConfigMap(
                    metadata=client.V1ObjectMeta(name="trading-config"),
                    data=config_data
                )
                v1.create_namespaced_config_map(
                    namespace="default",
                    body=config_map
                )
                
                # Create Secret for sensitive data
                secret_data = {}
                for key, value in config.get("secrets", {}).items():
                    if isinstance(value, (str, int, float, bool)):
                        secret_data[key] = base64.b64encode(
                            str(value).encode()
                        ).decode()
                
                secret = client.V1Secret(
                    metadata=client.V1ObjectMeta(name="trading-secrets"),
                    data=secret_data
                )
                v1.create_namespaced_secret(
                    namespace="default",
                    body=secret
                )
            
            else:
                raise ValueError(f"Unsupported target: {target}")
            
            self.logger.info(f"Configuration synced to {target}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to sync configuration: {e}")
            raise

    def _customize_config(self, template: Dict[str, Any], env: str) -> Dict[str, Any]:
        """Customize configuration for environment."""
        config = template.copy()
        
        # Set environment-specific values
        config["app"]["environment"] = env
        
        if env == "development":
            config["server"]["port"] = 8000
            config["database"]["port"] = 6379
        elif env == "staging":
            config["server"]["port"] = 8001
            config["database"]["port"] = 6380
        elif env == "production":
            config["server"]["port"] = 8002
            config["database"]["port"] = 6381
        
        return config

    def _print_validation_results(self, results: Dict[str, Any]):
        """Print configuration validation results."""
        print("\nConfiguration Validation Results:")
        print(f"\nTimestamp: {results['timestamp']}")
        print(f"Config Path: {results['config_path']}")
        print(f"Valid: {'���' if results['valid'] else '���'}")
        
        if results["issues"]:
            print("\nIssues:")
            for issue in results["issues"]:
                print(f"  - {issue}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Configuration Manager")
    parser.add_argument(
        "command",
        choices=["generate", "validate", "encrypt", "decrypt", "sync"],
        help="Command to execute"
    )
    parser.add_argument(
        "--template",
        default="default",
        help="Configuration template to use"
    )
    parser.add_argument(
        "--config-path",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--secrets",
        type=json.loads,
        help="Secrets to encrypt/decrypt"
    )
    parser.add_argument(
        "--key",
        help="Encryption/decryption key"
    )
    parser.add_argument(
        "--target",
        choices=["aws", "vault", "kubernetes"],
        help="Target system for configuration sync"
    )
    
    args = parser.parse_args()
    manager = ConfigManager()
    
    commands = {
        "generate": lambda: manager.generate_config(args.template),
        "validate": lambda: manager.validate_config(args.config_path),
        "encrypt": lambda: manager.encrypt_secrets(args.secrets, args.key),
        "decrypt": lambda: manager.decrypt_secrets(args.config_path, args.key),
        "sync": lambda: manager.sync_config(args.target, args.config_path)
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 