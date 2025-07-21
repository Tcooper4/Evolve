import json
import logging
import secrets
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import toml
import yaml
from cachetools import TTLCache
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from ratelimit import limits, sleep_and_retry

from utils.launch_utils import setup_logging

logger = logging.getLogger(__name__)


class ConfigValue(BaseModel):
    """Configuration value model."""

    key: str
    value: Any
    type: str
    description: str
    is_secret: bool = False
    is_required: bool = False
    default: Optional[Any] = None
    validation: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ConfigSection(BaseModel):
    """Configuration section model."""

    name: str
    description: str
    values: Dict[str, ConfigValue] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ConfigFile(BaseModel):
    """Configuration file model."""

    path: str
    format: str
    sections: Dict[str, ConfigSection] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AutomationConfigService:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger("automation")

    def setup_logging(self):
        return setup_logging(service_name="service")

    def setup_encryption(self):
        """Set up encryption for automation config."""
        # Encryption logic here

    def setup_cache(self):
        """Setup configuration caching."""
        self.cache = TTLCache(maxsize=1000, ttl=3600)

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def load_config(self, file_path: str, format: str = "json") -> ConfigFile:
        """Load configuration file."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {file_path}")

            # Try cache first
            cache_key = f"{path}:{format}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Load based on format
            if format == "json":
                with open(path, "r") as f:
                    data = json.load(f)
            elif format == "yaml":
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
            elif format == "toml":
                with open(path, "r") as f:
                    data = toml.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Create config file
            config_file = ConfigFile(path=str(path), format=format)

            # Process sections
            for section_name, section_data in data.items():
                section = ConfigSection(
                    name=section_name, description=section_data.get("description", "")
                )

                # Process values
                for key, value_data in section_data.get("values", {}).items():
                    value = ConfigValue(
                        key=key,
                        value=value_data.get("value"),
                        type=value_data.get("type", "string"),
                        description=value_data.get("description", ""),
                        is_secret=value_data.get("is_secret", False),
                        is_required=value_data.get("is_required", False),
                        default=value_data.get("default"),
                        validation=value_data.get("validation"),
                    )

                    # Decrypt secret values
                    if value.is_secret and value.value:
                        value.value = self._decrypt_value(value.value)

                    section.values[key] = value

                config_file.sections[section_name] = section

            # Cache result
            self.cache[cache_key] = config_file
            self.config_files[str(path)] = config_file

            return config_file

        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def save_config(self, config_file: ConfigFile, format: Optional[str] = None):
        """Save configuration file."""
        try:
            path = Path(config_file.path)
            format = format or config_file.format

            # Prepare data
            data = {}
            for section_name, section in config_file.sections.items():
                section_data = {"description": section.description, "values": {}}

                for key, value in section.values.items():
                    value_data = {
                        "value": value.value,
                        "type": value.type,
                        "description": value.description,
                        "is_secret": value.is_secret,
                        "is_required": value.is_required,
                        "default": value.default,
                        "validation": value.validation,
                    }

                    # Encrypt secret values
                    if value.is_secret and value.value:
                        value_data["value"] = self._encrypt_value(value.value)

                    section_data["values"][key] = value_data

                data[section_name] = section_data

            # Save based on format
            if format == "json":
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
            elif format == "yaml":
                with open(path, "w") as f:
                    yaml.dump(data, f)
            elif format == "toml":
                with open(path, "w") as f:
                    toml.dump(data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Update cache
            cache_key = f"{path}:{format}"
            self.cache[cache_key] = config_file

        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def get_value(
        self, file_path: str, section: str, key: str, format: str = "json"
    ) -> Any:
        """Get configuration value."""
        try:
            config_file = await self.load_config(file_path, format)
            section_data = config_file.sections.get(section)
            if not section_data:
                raise ValueError(f"Section not found: {section}")

            value = section_data.values.get(key)
            if not value:
                raise ValueError(f"Value not found: {key}")

            return value.value

        except Exception as e:
            logger.error(f"Failed to get value: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def set_value(
        self, file_path: str, section: str, key: str, value: Any, format: str = "json"
    ):
        """Set configuration value."""
        try:
            config_file = await self.load_config(file_path, format)
            section_data = config_file.sections.get(section)
            if not section_data:
                raise ValueError(f"Section not found: {section}")

            value_data = section_data.values.get(key)
            if not value_data:
                raise ValueError(f"Value not found: {key}")

            # Update value
            value_data.value = value
            value_data.updated_at = datetime.now()

            # Save changes
            await self.save_config(config_file, format)

        except Exception as e:
            logger.error(f"Failed to set value: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def load_env(self):
        """Load environment variables."""
        try:
            if self.env_file.exists():
                load_dotenv(self.env_file)

        except Exception as e:
            logger.error(f"Failed to load env: {str(e)}")
            raise

    def _encrypt_value(self, value: str) -> str:
        """Encrypt configuration value."""
        try:
            return self.cipher.encrypt(value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt value: {str(e)}")
            raise

    def _decrypt_value(self, value: str) -> str:
        """Decrypt configuration value."""
        try:
            return self.cipher.decrypt(value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def generate_secret(
        self,
        length: int = 32,
        include_uppercase: bool = True,
        include_lowercase: bool = True,
        include_digits: bool = True,
        include_special: bool = True,
    ) -> str:
        """Generate secure secret."""
        try:
            chars = ""
            if include_uppercase:
                chars += string.ascii_uppercase
            if include_lowercase:
                chars += string.ascii_lowercase
            if include_digits:
                chars += string.digits
            if include_special:
                chars += string.punctuation

            if not chars:
                raise ValueError("No character sets selected")

            return "".join(secrets.choice(chars) for _ in range(length))

        except Exception as e:
            logger.error(f"Failed to generate secret: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear caches
            self.cache.clear()
            self.config_files.clear()

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
