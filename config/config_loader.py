"""
Configuration Loader

This module provides a centralized configuration loading system with validation,
environment variable support, and modular config management.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import yaml
from jsonschema import validate


@dataclass
class ConfigValidationError:
    """Configuration validation error"""

    field: str
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ConfigSection:
    """Configuration section"""

    name: str
    data: Dict[str, Any]
    source_file: str
    last_modified: datetime
    validation_errors: List[ConfigValidationError] = field(default_factory=list)


class ConfigLoader:
    """Centralized configuration loader with validation and environment support."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)

        # Configuration sections
        self.sections: Dict[str, ConfigSection] = {}

        # Schema definitions for validation
        self.schemas = self._load_schemas()

        # Environment variable mappings
        self.env_mappings = {
            "FORECASTING_ENABLED": ("forecasting", "enabled"),
            "BACKTESTING_ENABLED": ("backtesting", "enabled"),
            "STRATEGIES_ENABLED": ("strategies", "enabled"),
            "DEFAULT_CAPITAL": ("backtesting", "default_initial_capital"),
            "COMMISSION_RATE": ("backtesting", "default_commission_rate"),
            "SLIPPAGE": ("backtesting", "default_slippage"),
            "LEVERAGE": ("backtesting", "default_leverage"),
            "LOG_LEVEL": ("logging", "level"),
            "DATABASE_URL": ("database", "url"),
            "API_KEY": ("api", "key"),
            "API_SECRET": ("api", "secret"),
        }

        # Load all configuration files
        self._load_all_configs()

        # Apply environment overrides
        self._apply_environment_overrides()

        # Validate all configurations
        self._validate_all_configs()

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for configuration validation."""
        schemas = {}

        # Forecasting schema
        schemas["forecasting"] = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "default_horizon_days": {"type": "integer", "minimum": 1},
                "max_horizon_days": {"type": "integer", "minimum": 1},
                "min_data_points": {"type": "integer", "minimum": 10},
                "validation_split": {"type": "number", "minimum": 0, "maximum": 1},
                "models": {"type": "object"},
                "features": {"type": "object"},
                "validation": {"type": "object"},
                "thresholds": {"type": "object"},
                "optimization": {"type": "object"},
                "monitoring": {"type": "object"},
            },
            "required": ["enabled"],
        }

        # Backtesting schema
        schemas["backtesting"] = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "default_initial_capital": {"type": "number", "minimum": 0},
                "default_commission_rate": {"type": "number", "minimum": 0},
                "default_slippage": {"type": "number", "minimum": 0},
                "default_leverage": {"type": "number", "minimum": 0},
                "data": {"type": "object"},
                "strategies": {"type": "object"},
                "risk_management": {"type": "object"},
                "execution": {"type": "object"},
                "metrics": {"type": "object"},
                "reporting": {"type": "object"},
                "optimization": {"type": "object"},
                "validation": {"type": "object"},
            },
            "required": ["enabled"],
        }

        # Strategies schema
        schemas["strategies"] = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "default_position_size": {"type": "number", "minimum": 0, "maximum": 1},
                "max_concurrent_strategies": {"type": "integer", "minimum": 1},
                "strategy_rotation": {"type": "boolean"},
                "rotation_frequency_days": {"type": "integer", "minimum": 1},
                "definitions": {"type": "object"},
                "ensemble": {"type": "object"},
                "risk_management": {"type": "object"},
                "performance": {"type": "object"},
                "optimization": {"type": "object"},
                "monitoring": {"type": "object"},
            },
            "required": ["enabled"],
        }

        return schemas

    def _load_all_configs(self) -> None:
        """Load all configuration files from the config directory."""
        config_files = {
            "forecasting": "forecasting.yaml",
            "backtesting": "backtest.yaml",
            "strategies": "strategies.yaml",
            "app": "app_config.yaml",
            "system": "system_config.yaml",
            "logging": "logging_config.yaml",
        }

        for section_name, filename in config_files.items():
            file_path = self.config_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        data = yaml.safe_load(f)

                    # Extract section data (handle nested structure)
                    section_data = data.get(section_name, data)

                    self.sections[section_name] = ConfigSection(
                        name=section_name,
                        data=section_data,
                        source_file=str(file_path),
                        last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                    )

                    self.logger.info(f"Loaded configuration section: {section_name}")

                except Exception as e:
                    self.logger.error(f"Failed to load config {filename}: {e}")
                    # Create empty section with error
                    self.sections[section_name] = ConfigSection(
                        name=section_name,
                        data={},
                        source_file=str(file_path),
                        last_modified=datetime.now(),
                        validation_errors=[
                            ConfigValidationError(
                                field="file_load",
                                message=f"Failed to load configuration file: {e}",
                                severity="error",
                            )
                        ],
                    )

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        for env_var, (section, key) in self.env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if section in self.sections:
                    # Convert value to appropriate type
                    converted_value = self._convert_env_value(env_value, key)
                    self._set_nested_value(
                        self.sections[section].data, key, converted_value
                    )
                    self.logger.info(
                        f"Applied environment override: {env_var} -> {section}.{key}"
                    )

    def _convert_env_value(self, value: str, key: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Numeric values
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # String values (default)
        return value

    def _set_nested_value(
        self, data: Dict[str, Any], key_path: str, value: Any
    ) -> None:
        """Set a nested value in a dictionary using dot notation."""
        keys = key_path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _validate_all_configs(self) -> None:
        """Validate all loaded configurations."""
        for section_name, section in self.sections.items():
            if section_name in self.schemas:
                try:
                    validate(instance=section.data, schema=self.schemas[section_name])
                    self.logger.info(
                        f"Configuration section '{section_name}' validated successfully"
                    )
                except jsonschema.exceptions.ValidationError as e:
                    error = ConfigValidationError(
                        field=e.path[0] if e.path else "unknown",
                        message=str(e),
                        severity="error",
                    )
                    section.validation_errors.append(error)
                    self.logger.error(
                        f"Configuration validation error in '{section_name}': {e}"
                    )
                except Exception as e:
                    error = ConfigValidationError(
                        field="validation",
                        message=f"Validation failed: {e}",
                        severity="error",
                    )
                    section.validation_errors.append(error)
                    self.logger.error(
                        f"Configuration validation failed for '{section_name}': {e}"
                    )

    def get_config(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value."""
        if section not in self.sections:
            return default

        section_data = self.sections[section].data

        if key is None:
            return section_data

        # Handle nested keys with dot notation
        keys = key.split(".")
        current = section_data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set_config(self, section: str, key: str, value: Any) -> bool:
        """Set configuration value."""
        if section not in self.sections:
            return False

        self._set_nested_value(self.sections[section].data, key, value)
        return True

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration data."""
        return {name: section.data for name, section in self.sections.items()}

    def get_validation_errors(self, section: str = None) -> List[ConfigValidationError]:
        """Get validation errors for a section or all sections."""
        errors = []

        if section:
            if section in self.sections:
                errors.extend(self.sections[section].validation_errors)
        else:
            for section_data in self.sections.values():
                errors.extend(section_data.validation_errors)

        return errors

    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return any(
            error.severity == "error"
            for section in self.sections.values()
            for error in section.validation_errors
        )

    def reload_config(self, section: str = None) -> bool:
        """Reload configuration from files."""
        try:
            if section:
                # Reload specific section
                if section in self.sections:
                    file_path = Path(self.sections[section].source_file)
                    if file_path.exists():
                        with open(file_path, "r") as f:
                            data = yaml.safe_load(f)
                        section_data = data.get(section, data)
                        self.sections[section].data = section_data
                        self.sections[section].last_modified = datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        )
                        self.sections[section].validation_errors.clear()

                        # Re-validate
                        if section in self.schemas:
                            validate(
                                instance=section_data, schema=self.schemas[section]
                            )

                        self.logger.info(f"Reloaded configuration section: {section}")
                        return True
            else:
                # Reload all sections
                self.sections.clear()
                self._load_all_configs()
                self._apply_environment_overrides()
                self._validate_all_configs()
                self.logger.info("Reloaded all configuration sections")
                return True

        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return False

    def export_config(self, format: str = "yaml") -> str:
        """Export configuration to string format."""
        try:
            if format.lower() == "json":
                return json.dumps(self.get_all_configs(), indent=2, default=str)
            else:
                return yaml.dump(
                    self.get_all_configs(), default_flow_style=False, indent=2
                )
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return ""

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary with validation status."""
        summary = {
            "total_sections": len(self.sections),
            "sections": {},
            "validation_status": "valid" if not self.has_errors() else "invalid",
            "total_errors": len(self.get_validation_errors()),
            "environment_overrides": len(
                [k for k, v in self.env_mappings.items() if os.getenv(k)]
            ),
        }

        for section_name, section in self.sections.items():
            summary["sections"][section_name] = {
                "source_file": section.source_file,
                "last_modified": section.last_modified.isoformat(),
                "error_count": len(
                    [e for e in section.validation_errors if e.severity == "error"]
                ),
                "warning_count": len(
                    [e for e in section.validation_errors if e.severity == "warning"]
                ),
                "enabled": section.data.get("enabled", True) if section.data else False,
            }

        return summary


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_config(section: str, key: str = None, default: Any = None) -> Any:
    """Get configuration value using global loader."""
    return get_config_loader().get_config(section, key, default)


def set_config(section: str, key: str, value: Any) -> bool:
    """Set configuration value using global loader."""
    return get_config_loader().set_config(section, key, value)


def validate_config() -> bool:
    """Validate all configurations."""
    loader = get_config_loader()
    return not loader.has_errors()
