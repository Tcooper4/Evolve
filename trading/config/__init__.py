"""Configuration package."""

from trading.config.configuration import (
    ConfigManager,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    WebConfig,
    MonitoringConfig
)

# Import settings for the config object
from trading.config.settings import *

class Config:
    """Configuration object that provides .get() method for accessing settings."""
    
    def get(self, key: str, default=None):
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return globals().get(key, default)

# Create a config instance
config = Config()

__all__ = [
    'ConfigManager',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'WebConfig',
    'MonitoringConfig',
    'config'
] 