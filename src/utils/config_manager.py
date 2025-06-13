"""
Configuration management utilities for market analysis settings.
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Class for managing configuration settings."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> bool:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False
            
            # Load based on file extension
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            logger.info(f"Successfully loaded configuration from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def save_config(self, config_path: Union[str, Path]) -> bool:
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save configuration file
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on file extension
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            logger.info(f"Successfully saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key to get
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        if key is None:
            return self.config
        
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key to set
            value: Value to set
        """
        self.config[key] = value
    
    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary.
        
        Args:
            config_dict: Dictionary of configuration values to update
        """
        self.config.update(config_dict)
    
    def get_market_conditions(self) -> Dict[str, Any]:
        """
        Get market conditions configuration.
        
        Returns:
            Dictionary of market conditions
        """
        return self.config.get('market_conditions', {})
    
    def get_analysis_settings(self) -> Dict[str, Any]:
        """
        Get analysis settings configuration.
        
        Returns:
            Dictionary of analysis settings
        """
        return self.config.get('analysis_settings', {})
    
    def get_visualization_settings(self) -> Dict[str, Any]:
        """
        Get visualization settings configuration.
        
        Returns:
            Dictionary of visualization settings
        """
        return self.config.get('visualization_settings', {})
    
    def get_pipeline_settings(self) -> Dict[str, Any]:
        """
        Get pipeline settings configuration.
        
        Returns:
            Dictionary of pipeline settings
        """
        return self.config.get('pipeline_settings', {})
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate configuration structure.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_sections = ['market_conditions', 'analysis_settings', 'visualization_settings', 'pipeline_settings']
        
        for section in required_sections:
            if section not in self.config:
                return False, f"Missing required section: {section}"
        
        return True, "Configuration validation successful" 