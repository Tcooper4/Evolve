"""Configuration utilities with hot-reload support.

This module provides utilities for loading and managing configuration files,
with support for hot-reloading when files change. It includes file watching,
change detection, and automatic reloading of configurations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging
from dataclasses import dataclass, asdict
import os
import toml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from functools import wraps
import time

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuration for trading system."""
    # Data settings
    data_dir: str = "data"
    cache_dir: str = "cache"
    
    # Model settings
    model_dir: str = "models"
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Strategy settings
    strategy_dir: str = "strategies"
    default_strategy: str = "default"
    ensemble_size: int = 3
    
    # Risk settings
    max_position_size: float = 0.1
    max_drawdown: float = 0.2
    stop_loss: float = 0.05
    take_profit: float = 0.1
    
    # Portfolio settings
    initial_capital: float = 100000.0
    rebalance_frequency: str = "1D"
    max_leverage: float = 1.0
    
    # Logging settings
    log_dir: str = "logs"
    log_level: str = "INFO"
    
    # API settings
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_url: str = "https://api.example.com"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TradingConfig':
        """Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            TradingConfig instance
        """
        return cls(**config_dict)

class ConfigFileHandler(FileSystemEventHandler):
    """Handler for config file changes."""
    
    def __init__(self, callback: Callable[[], None]):
        """Initialize the handler.
        
        Args:
            callback: Function to call when file changes
        """
        self.callback = callback
    
    def on_modified(self, event):
        """Handle file modification events.
        
        Args:
            event: File system event
        """
        if not event.is_directory:
            logger.info(f"Config file changed: {event.src_path}")
            self.callback()

class ConfigManager:
    """Manager for configuration files with hot-reload support."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the config manager.
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        self.observers: Dict[str, Observer] = {}
        self._lock = threading.Lock()
    
    def load_config(
        self,
        filename: str,
        hot_reload: bool = False,
        callback: Optional[Callable[[], None]] = None
    ) -> Dict[str, Any]:
        """Load a configuration file with optional hot-reload.
        
        Args:
            filename: Name of the config file
            hot_reload: Whether to enable hot-reload
            callback: Function to call when config changes
            
        Returns:
            Configuration dictionary
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        # Load initial config
        config = self._load_file(file_path)
        
        with self._lock:
            self.configs[filename] = config
        
        # Set up hot-reload if requested
        if hot_reload:
            self._setup_hot_reload(file_path, callback)
        
        return config
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a configuration file based on its extension.
        
        Args:
            file_path: Path to the config file
            
        Returns:
            Configuration dictionary
        """
        suffix = file_path.suffix.lower()
        
        try:
            with open(file_path) as f:
                if suffix == '.json':
                    return json.load(f)
                elif suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif suffix == '.toml':
                    return toml.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {suffix}")
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            raise
    
    def _setup_hot_reload(
        self,
        file_path: Path,
        callback: Optional[Callable[[], None]]
    ) -> None:
        """Set up hot-reload for a config file.
        
        Args:
            file_path: Path to the config file
            callback: Function to call when config changes
        """
        if callback is None:
            callback = lambda: self._reload_config(file_path)
        
        observer = Observer()
        observer.schedule(
            ConfigFileHandler(callback),
            str(file_path.parent),
            recursive=False
        )
        observer.start()
        
        with self._lock:
            self.observers[str(file_path)] = observer
    
    def _reload_config(self, file_path: Path) -> None:
        """Reload a configuration file.
        
        Args:
            file_path: Path to the config file
        """
        try:
            config = self._load_file(file_path)
            with self._lock:
                self.configs[file_path.name] = config
            logger.info(f"Reloaded config file: {file_path}")
        except Exception as e:
            logger.error(f"Error reloading config file {file_path}: {e}")
    
    def get_config(self, filename: str) -> Dict[str, Any]:
        """Get a configuration by filename.
        
        Args:
            filename: Name of the config file
            
        Returns:
            Configuration dictionary
        """
        with self._lock:
            return self.configs.get(filename, {})
    
    def stop_hot_reload(self, filename: str) -> None:
        """Stop hot-reload for a config file.
        
        Args:
            filename: Name of the config file
        """
        file_path = str(self.config_dir / filename)
        with self._lock:
            if file_path in self.observers:
                self.observers[file_path].stop()
                self.observers[file_path].join()
                del self.observers[file_path]
    
    def stop_all_hot_reload(self) -> None:
        """Stop hot-reload for all config files."""
        with self._lock:
            for observer in self.observers.values():
                observer.stop()
                observer.join()
            self.observers.clear()

def hot_reload_config(func: Callable) -> Callable:
    """Decorator to enable hot-reload for a function using config.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        config_manager = ConfigManager()
        return func(*args, config_manager=config_manager, **kwargs)
    return wrapper

# Create singleton instance
config_manager = ConfigManager()

def load_config(config_file: str) -> TradingConfig:
    """Load configuration from file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        TradingConfig instance
    """
    file_path = Path(config_file)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(file_path, 'r') as f:
        if file_path.suffix == '.json':
            config_dict = json.load(f)
        elif file_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    return TradingConfig.from_dict(config_dict)

def save_config(config: TradingConfig, config_file: str) -> None:
    """Save configuration to file.
    
    Args:
        config: TradingConfig instance
        config_file: Path to save config file
    """
    file_path = Path(config_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(file_path, 'w') as f:
        if file_path.suffix == '.json':
            json.dump(config_dict, f, indent=4)
        elif file_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")

def update_config(config: TradingConfig, updates: Dict[str, Any]) -> TradingConfig:
    """Update configuration with new values.
    
    Args:
        config: Current TradingConfig instance
        updates: Dictionary of updates
        
    Returns:
        Updated TradingConfig instance
    """
    config_dict = config.to_dict()
    config_dict.update(updates)
    return TradingConfig.from_dict(config_dict)

def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables.
    
    Returns:
        Dictionary of environment variables
    """
    config = {}
    
    # Map environment variables to config keys
    env_mapping = {
        'TRADING_API_KEY': 'api_key',
        'TRADING_API_SECRET': 'api_secret',
        'TRADING_API_URL': 'api_url',
        'TRADING_LOG_LEVEL': 'log_level',
        'TRADING_MAX_POSITION_SIZE': 'max_position_size',
        'TRADING_MAX_DRAWDOWN': 'max_drawdown',
        'TRADING_STOP_LOSS': 'stop_loss',
        'TRADING_TAKE_PROFIT': 'take_profit',
        'TRADING_INITIAL_CAPITAL': 'initial_capital',
        'TRADING_MAX_LEVERAGE': 'max_leverage'
    }
    
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Convert numeric values
            if config_key in ['max_position_size', 'max_drawdown', 'stop_loss',
                            'take_profit', 'initial_capital', 'max_leverage']:
                value = float(value)
            
            config[config_key] = value
    
    return config

def create_default_config(config_file: str) -> None:
    """Create default configuration file.
    
    Args:
        config_file: Path to save config file
    """
    config = TradingConfig()
    save_config(config, config_file)

def validate_config(config: TradingConfig) -> None:
    """Validate configuration values.
    
    Args:
        config: TradingConfig instance to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate numeric ranges
    if not 0 < config.max_position_size <= 1:
        raise ValueError("max_position_size must be between 0 and 1")
    
    if not 0 < config.max_drawdown <= 1:
        raise ValueError("max_drawdown must be between 0 and 1")
    
    if not 0 < config.stop_loss <= 1:
        raise ValueError("stop_loss must be between 0 and 1")
    
    if not 0 < config.take_profit <= 1:
        raise ValueError("take_profit must be between 0 and 1")
    
    if config.initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    
    if config.max_leverage <= 0:
        raise ValueError("max_leverage must be positive")
    
    # Validate directories
    for dir_path in [config.data_dir, config.cache_dir, config.model_dir,
                    config.strategy_dir, config.log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Validate API settings
    if config.api_key is None or config.api_secret is None:
        logging.warning("API credentials not configured")
    
    # Validate logging level
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config.log_level not in valid_log_levels:
        raise ValueError(f"log_level must be one of {valid_log_levels}") 