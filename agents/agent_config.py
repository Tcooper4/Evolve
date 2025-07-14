"""
Agent Configuration Module

This module provides centralized configuration for all agents including:
- OpenAI fallback settings
- Timeout configurations
- Max tokens limits
- Routing behavior
- Error handling settings
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Centralized configuration for agent behavior."""

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_fallback_model: str = "gpt-3.5-turbo"
    openai_timeout: int = 30
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.7

    # HuggingFace Configuration
    huggingface_model: str = "gpt2"
    huggingface_cache_dir: str = "cache/huggingface"
    huggingface_timeout: int = 60

    # Routing Configuration
    enable_fallback: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0

    # Error Handling
    log_errors: bool = True
    raise_on_critical: bool = False
    error_recovery_enabled: bool = True

    # Performance Settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_concurrent_requests: int = 5

    # Memory Management
    max_memory_usage: int = 1024  # MB
    enable_memory_cleanup: bool = True

    # Logging Configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/agents.log"

    # Agent-specific settings
    agent_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize configuration after creation."""
        self._load_environment_variables()
        self._validate_config()

    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # OpenAI settings
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Override with environment variables if present
        if os.getenv("OPENAI_MODEL"):
            self.openai_model = os.getenv("OPENAI_MODEL")
        if os.getenv("OPENAI_TIMEOUT"):
            self.openai_timeout = int(os.getenv("OPENAI_TIMEOUT"))
        if os.getenv("OPENAI_MAX_TOKENS"):
            self.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS"))

    def _validate_config(self):
        """Validate configuration settings."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is missing in AgentConfig.")
        if self.openai_timeout <= 0:
            raise ValueError("OpenAI timeout must be positive")
        if self.openai_max_tokens <= 0:
            raise ValueError("OpenAI max tokens must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")

    def get_agent_setting(
        self, agent_name: str, setting_name: str, default: Any = None
    ) -> Any:
        """Get agent-specific setting."""
        agent_config = self.agent_settings.get(agent_name, {})
        return agent_config.get(setting_name, default)

    def set_agent_setting(self, agent_name: str, setting_name: str, value: Any):
        """Set agent-specific setting."""
        if agent_name not in self.agent_settings:
            self.agent_settings[agent_name] = {}
        self.agent_settings[agent_name][setting_name] = value

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-specific configuration."""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "fallback_model": self.openai_fallback_model,
            "timeout": self.openai_timeout,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature,
        }

    def get_huggingface_config(self) -> Dict[str, Any]:
        """Get HuggingFace-specific configuration."""
        return {
            "model": self.huggingface_model,
            "cache_dir": self.huggingface_cache_dir,
            "timeout": self.huggingface_timeout,
        }

    def get_routing_config(self) -> Dict[str, Any]:
        """Get routing-specific configuration."""
        return {
            "enable_fallback": self.enable_fallback,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "max_concurrent_requests": self.max_concurrent_requests,
        }

    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return {
            "log_errors": self.log_errors,
            "raise_on_critical": self.raise_on_critical,
            "error_recovery_enabled": self.error_recovery_enabled,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "openai_config": self.get_openai_config(),
            "huggingface_config": self.get_huggingface_config(),
            "routing_config": self.get_routing_config(),
            "error_handling_config": self.get_error_handling_config(),
            "performance_config": {
                "enable_caching": self.enable_caching,
                "cache_ttl": self.cache_ttl,
                "max_memory_usage": self.max_memory_usage,
                "enable_memory_cleanup": self.enable_memory_cleanup,
            },
            "logging_config": {
                "log_level": self.log_level,
                "log_to_file": self.log_to_file,
                "log_file_path": self.log_file_path,
            },
            "agent_settings": self.agent_settings,
            "last_updated": datetime.now().isoformat(),
        }

    def save_to_file(self, file_path: str = "config/agent_config.json"):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Agent configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save agent configuration: {e}")

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main run method for the agent configuration.

        Args:
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dict: Configuration status and settings
        """
        return {
            "success": True,
            "message": "Agent configuration is operational",
            "config": self.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def load_from_file(
        cls, file_path: str = "config/agent_config.json"
    ) -> "AgentConfig":
        """Load configuration from file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    config_data = json.load(f)

                # Create config instance
                config = cls()

                # Update with loaded data
                if "openai_config" in config_data:
                    openai_config = config_data["openai_config"]
                    config.openai_api_key = openai_config.get("api_key")
                    config.openai_model = openai_config.get(
                        "model", config.openai_model
                    )
                    config.openai_fallback_model = openai_config.get(
                        "fallback_model", config.openai_fallback_model
                    )
                    config.openai_timeout = openai_config.get(
                        "timeout", config.openai_timeout
                    )
                    config.openai_max_tokens = openai_config.get(
                        "max_tokens", config.openai_max_tokens
                    )
                    config.openai_temperature = openai_config.get(
                        "temperature", config.openai_temperature
                    )

                if "agent_settings" in config_data:
                    config.agent_settings = config_data["agent_settings"]

                logger.info(f"Agent configuration loaded from {file_path}")
                return config
            else:
                logger.info(f"Configuration file {file_path} not found, using defaults")
                return cls()

        except Exception as e:
            logger.error(f"Failed to load agent configuration: {e}")
            return cls()


# Global configuration instance
_global_config = None


def get_agent_config() -> AgentConfig:
    """Get the global agent configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = AgentConfig.load_from_file()
    return _global_config


def set_agent_config(config: AgentConfig):
    """Set the global agent configuration instance."""
    global _global_config
    _global_config = config


def update_agent_config(**kwargs):
    """Update global agent configuration with new values."""
    config = get_agent_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration key: {key}")


def save_agent_config(file_path: str = "config/agent_config.json"):
    """Save the global agent configuration to file."""
    config = get_agent_config()
    config.save_to_file(file_path)


# Default configurations for common agents
DEFAULT_AGENT_CONFIGS = {
    "prompt_router": {
        "timeout": 45,
        "max_tokens": 6000,
        "enable_fallback": True,
        "max_retries": 5,
    },
    "model_builder": {
        "timeout": 120,
        "max_tokens": 8000,
        "enable_fallback": True,
        "max_retries": 3,
    },
    "voice_prompt": {
        "timeout": 30,
        "max_tokens": 2000,
        "enable_fallback": True,
        "max_retries": 2,
    },
    "performance_critic": {
        "timeout": 60,
        "max_tokens": 4000,
        "enable_fallback": True,
        "max_retries": 3,
    },
}


def initialize_default_configs():
    """Initialize default configurations for common agents."""
    config = get_agent_config()
    for agent_name, agent_config in DEFAULT_AGENT_CONFIGS.items():
        config.agent_settings[agent_name] = agent_config
    save_agent_config()


# Initialize default configs when module is imported
initialize_default_configs()
