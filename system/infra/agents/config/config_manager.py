"""
Configuration Manager.

This module provides centralized configuration management for the agentic
forecasting platform, including:
- Environment variable loading
- Configuration file parsing
- Secure credential management
- Dynamic configuration updates
"""

import os
import yaml
import json
from typing import Any, Dict, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv
import hvac  # HashiCorp Vault client

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(
        self,
        config_dir: str = "config",
        env_file: str = ".env",
        use_vault: bool = False,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None
    ):
        self.config_dir = Path(config_dir)
        self.env_file = env_file
        self.use_vault = use_vault
        self.config_cache: Dict[str, Any] = {}
        
        # Load environment variables
        load_dotenv(env_file)
        
        # Initialize Vault client if enabled
        self.vault_client = None
        if use_vault:
            if not vault_url or not vault_token:
                raise ValueError("Vault URL and token required when use_vault is True")
            self.vault_client = hvac.Client(url=vault_url, token=vault_token)
            
        # Load all config files
        self._load_configs()
        
    def _load_configs(self) -> None:
        """Load all configuration files."""
        try:
            # Load YAML configs
            for config_file in self.config_dir.glob("*.yaml"):
                self._load_yaml_config(config_file)
                
            # Load JSON configs
            for config_file in self.config_dir.glob("*.json"):
                self._load_json_config(config_file)
                
        except Exception as e:
            logger.error(f"Error loading configs: {str(e)}")
            raise
            
    def _load_yaml_config(self, config_file: Path) -> None:
        """Load a YAML configuration file."""
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                self.config_cache[config_file.stem] = config
                
        except Exception as e:
            logger.error(f"Error loading YAML config {config_file}: {str(e)}")
            raise
            
    def _load_json_config(self, config_file: Path) -> None:
        """Load a JSON configuration file."""
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                self.config_cache[config_file.stem] = config
                
        except Exception as e:
            logger.error(f"Error loading JSON config {config_file}: {str(e)}")
            raise
            
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a configuration by name."""
        if config_name not in self.config_cache:
            raise KeyError(f"Configuration {config_name} not found")
        return self.config_cache[config_name]
        
    def get_value(self, config_name: str, key: str, default: Any = None) -> Any:
        """Get a specific value from a configuration."""
        config = self.get_config(config_name)
        return config.get(key, default)
        
    def get_secret(self, secret_path: str) -> Optional[str]:
        """Get a secret from Vault or environment variables."""
        if self.use_vault and self.vault_client:
            try:
                secret = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=secret_path
                )
                return secret["data"]["data"]["value"]
            except Exception as e:
                logger.error(f"Error reading secret from Vault: {str(e)}")
                return None
        else:
            # Fallback to environment variables
            return os.getenv(secret_path)
            
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> None:
        """Update a configuration."""
        if config_name not in self.config_cache:
            raise KeyError(f"Configuration {config_name} not found")
            
        # Update cache
        self.config_cache[config_name].update(updates)
        
        # Update file
        config_file = self.config_dir / f"{config_name}.yaml"
        if config_file.exists():
            with open(config_file, "w") as f:
                yaml.dump(self.config_cache[config_name], f)
        else:
            config_file = self.config_dir / f"{config_name}.json"
            if config_file.exists():
                with open(config_file, "w") as f:
                    json.dump(self.config_cache[config_name], f, indent=2)
            else:
                raise FileNotFoundError(f"Config file for {config_name} not found")
                
    def reload_configs(self) -> None:
        """Reload all configurations."""
        self.config_cache.clear()
        self._load_configs()
        
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all configurations."""
        return self.config_cache.copy()