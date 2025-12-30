"""
Configuration management utilities for the trading system.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration management utility class."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path:
            self.config_path = config_path

        if not self.config_path or not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            return {}

        try:
            with open(self.config_path, "r") as f:
                if self.config_path.endswith(".yaml") or self.config_path.endswith(
                    ".yml"
                ):
                    self.config = yaml.safe_load(f)
                elif self.config_path.endswith(".json"):
                    self.config = json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {self.config_path}")
                    return {}

            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save configuration to file."""
        if config_path:
            self.config_path = config_path

        if not self.config_path:
            logger.error("No config path specified")
            return False

        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, "w") as f:
                if self.config_path.endswith(".yaml") or self.config_path.endswith(
                    ".yml"
                ):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif self.config_path.endswith(".json"):
                    json.dump(self.config, f, indent=2)
                else:
                    logger.error(f"Unsupported config file format: {self.config_path}")
                    return False

            logger.info(f"Configuration saved to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value."""
        try:
            keys = key.split(".")
            config = self.config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value
            return True

        except Exception as e:
            logger.error(f"Error setting configuration: {e}")
            return False

    def update(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with multiple values."""
        try:
            for key, value in updates.items():
                self.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self.config.copy()


class ConfigValidator:
    """Configuration validation utility class."""

    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []

    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate configuration against schema."""
        self.validation_errors.clear()
        self.validation_warnings.clear()

        try:
            self._validate_dict(config, schema, "")
            return len(self.validation_errors) == 0
        except Exception as e:
            self.validation_errors.append(f"Validation error: {e}")
            return False

    def _validate_dict(self, data: Dict[str, Any], schema: Dict[str, Any], path: str):
        """Validate dictionary against schema."""
        for key, schema_info in schema.items():
            if key not in data:
                if schema_info.get("required", False):
                    self.validation_errors.append(
                        f"Missing required field: {path}.{key}"
                    )
                continue

            value = data[key]
            expected_type = schema_info.get("type")

            if expected_type and not isinstance(value, expected_type):
                self.validation_errors.append(
                    f"Invalid type for {path}.{key}: expected {expected_type}, got {type(value)}"
                )
                continue

            if expected_type == dict and "schema" in schema_info:
                self._validate_dict(value, schema_info["schema"], f"{path}.{key}")
            elif expected_type == list and "item_schema" in schema_info:
                for i, item in enumerate(value):
                    self._validate_dict(
                        item, schema_info["item_schema"], f"{path}.{key}[{i}]"
                    )

    def get_validation_errors(self) -> list:
        """Get validation errors."""
        return self.validation_errors.copy()

    def get_validation_warnings(self) -> list:
        """Get validation warnings."""
        return self.validation_warnings.copy()


class ConfigLoader:
    """Configuration loader utility class."""

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            return {}

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON config: {e}")
            return {}

    @staticmethod
    def load_env_vars(prefix: str = "") -> Dict[str, str]:
        """Load environment variables with optional prefix."""
        env_vars = {}
        for key, value in os.environ.items():
            if prefix and key.startswith(prefix):
                env_vars[key[len(prefix) :]] = value
            elif not prefix:
                env_vars[key] = value
        return env_vars
