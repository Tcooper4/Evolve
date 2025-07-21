"""Configuration utilities with hot-reload support.

This module provides utilities for loading and managing configuration files,
with support for hot-reloading when files change. It includes file watching,
change detection, and automatic reloading of configurations.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import toml
import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TradingConfig":
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
        callback: Optional[Callable[[], None]] = None,
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
                if suffix == ".json":
                    return json.load(f)
                elif suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                elif suffix == ".toml":
                    return toml.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {suffix}")
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            raise

    def _setup_hot_reload(
        self, file_path: Path, callback: Optional[Callable[[], None]]
    ) -> None:
        """Set up hot-reload for a config file.

        Args:
            file_path: Path to the config file
            callback: Function to call when config changes
        """
        try:
            if callback is None:
                def callback(): return self._reload_config(file_path)

            observer = Observer()
            observer.schedule(
                ConfigFileHandler(callback), str(file_path.parent), recursive=False
            )
            observer.start()

            self.observers[str(file_path)] = observer
            logger.info(f"Hot-reload enabled for {file_path}")

        except Exception as e:
            logger.error(f"Error setting up hot-reload: {e}")

    def _reload_config(self, file_path: Path) -> None:
        """Reload a configuration file.

        Args:
            file_path: Path to the config file
        """
        try:
            # Add small delay to ensure file is fully written
            time.sleep(0.1)

            config = self._load_file(file_path)

            with self._lock:
                self.configs[file_path.name] = config

            logger.info(f"Config reloaded: {file_path}")

        except Exception as e:
            logger.error(f"Error reloading config {file_path}: {e}")

    def get_config(self, filename: str) -> Dict[str, Any]:
        """Get a loaded configuration.

        Args:
            filename: Name of the config file

        Returns:
            Configuration dictionary
        """
        try:
            with self._lock:
                if filename not in self.configs:
                    return {
                        "success": False,
                        "error": f"Config {filename} not loaded",
                        "timestamp": datetime.now().isoformat(),
                    }

                return {
                    "success": True,
                    "result": self.configs[filename],
                    "message": "Config retrieved successfully",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def stop_hot_reload(self, filename: str) -> None:
        """Stop hot-reload for a specific file.

        Args:
            filename: Name of the config file
        """
        try:
            file_path = self.config_dir / filename
            observer_key = str(file_path)

            if observer_key in self.observers:
                self.observers[observer_key].stop()
                self.observers[observer_key].join()
                del self.observers[observer_key]
                logger.info(f"Hot-reload stopped for {filename}")

            return {
                "success": True,
                "message": f"Hot-reload stopped for {filename}",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def stop_all_hot_reload(self) -> None:
        """Stop all hot-reload observers."""
        try:
            for observer in self.observers.values():
                observer.stop()
                observer.join()

            self.observers.clear()
            logger.info("All hot-reload observers stopped")

            return {
                "success": True,
                "message": "All hot-reload observers stopped",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


def hot_reload_config(func: Callable) -> Callable:
    """Decorator to enable hot-reload for config-dependent functions.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return {
                "success": True,
                "result": result,
                "message": "Function executed successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    return wrapper


# Create singleton instance
# config_manager = ConfigManager()  # Removed to prevent import errors


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from file.

    Args:
        config_file: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        config_path = Path(config_file)

        if not config_path.exists():
            return {
                "success": False,
                "error": f"Config file not found: {config_file}",
                "timestamp": datetime.now().isoformat(),
            }

        suffix = config_path.suffix.lower()

        with open(config_path) as f:
            if suffix == ".json":
                config = json.load(f)
            elif suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif suffix == ".toml":
                config = toml.load(f)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported config format: {suffix}",
                    "timestamp": datetime.now().isoformat(),
                }

        logger.info(f"Config loaded from {config_file}")

        return {
            "success": True,
            "result": config,
            "message": "Config loaded successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error loading config {config_file}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def save_config(config: TradingConfig, config_file: str) -> Dict:
    """Save configuration to file.

    Args:
        config: Configuration object
        config_file: Path to save configuration

    Returns:
        Save result
    """
    try:
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = config_path.suffix.lower()
        config_dict = asdict(config)

        with open(config_path, "w") as f:
            if suffix == ".json":
                json.dump(config_dict, f, indent=2)
            elif suffix in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif suffix == ".toml":
                toml.dump(config_dict, f)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported config format: {suffix}",
                    "timestamp": datetime.now().isoformat(),
                }

        logger.info(f"Config saved to {config_file}")

        return {
            "success": True,
            "message": f"Config saved to {config_file}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error saving config {config_file}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def update_config(config: TradingConfig, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values.

    Args:
        config: Configuration object
        updates: Dictionary of updates

    Returns:
        Updated configuration
    """
    try:
        config_dict = asdict(config)
        config_dict.update(updates)

        updated_config = TradingConfig(**config_dict)

        logger.info(f"Config updated with {len(updates)} changes")

        return {
            "success": True,
            "result": updated_config,
            "message": "Config updated successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables.

    Returns:
        Configuration dictionary
    """
    try:
        config = {}

        # Data settings
        config["data_dir"] = os.getenv("TRADING_DATA_DIR", "data")
        config["cache_dir"] = os.getenv("TRADING_CACHE_DIR", "cache")

        # Model settings
        config["model_dir"] = os.getenv("TRADING_MODEL_DIR", "models")
        config["batch_size"] = int(os.getenv("TRADING_BATCH_SIZE", "32"))
        config["learning_rate"] = float(os.getenv("TRADING_LEARNING_RATE", "0.001"))
        config["max_epochs"] = int(os.getenv("TRADING_MAX_EPOCHS", "100"))

        # Strategy settings
        config["strategy_dir"] = os.getenv("TRADING_STRATEGY_DIR", "strategies")
        config["default_strategy"] = os.getenv("TRADING_DEFAULT_STRATEGY", "default")
        config["ensemble_size"] = int(os.getenv("TRADING_ENSEMBLE_SIZE", "3"))

        # Risk settings
        config["max_position_size"] = float(
            os.getenv("TRADING_MAX_POSITION_SIZE", "0.1")
        )
        config["max_drawdown"] = float(os.getenv("TRADING_MAX_DRAWDOWN", "0.2"))
        config["stop_loss"] = float(os.getenv("TRADING_STOP_LOSS", "0.05"))
        config["take_profit"] = float(os.getenv("TRADING_TAKE_PROFIT", "0.1"))

        # Portfolio settings
        config["initial_capital"] = float(
            os.getenv("TRADING_INITIAL_CAPITAL", "100000.0")
        )
        config["rebalance_frequency"] = os.getenv("TRADING_REBALANCE_FREQUENCY", "1D")
        config["max_leverage"] = float(os.getenv("TRADING_MAX_LEVERAGE", "1.0"))

        # Logging settings
        config["log_dir"] = os.getenv("TRADING_LOG_DIR", "logs")
        config["log_level"] = os.getenv("TRADING_LOG_LEVEL", "INFO")

        # API settings
        config["api_key"] = os.getenv("TRADING_API_KEY")
        config["api_secret"] = os.getenv("TRADING_API_SECRET")
        config["api_url"] = os.getenv("TRADING_API_URL", "https://api.example.com")

        logger.info("Environment configuration loaded")

        return {
            "success": True,
            "result": config,
            "message": "Environment config loaded successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error loading environment config: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def create_default_config(config_file: str) -> Dict:
    """Create default configuration file.

    Args:
        config_file: Path to create configuration file

    Returns:
        Creation result
    """
    try:
        default_config = TradingConfig()
        save_config(default_config, config_file)

        logger.info(f"Default config created at {config_file}")

        return {
            "success": True,
            "message": f"Default config created at {config_file}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error creating default config: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def validate_config(config: TradingConfig) -> Dict:
    """Validate configuration values.

    Args:
        config: Configuration object

    Returns:
        Validation result
    """
    try:
        errors = []

        # Validate numeric values
        if config.max_position_size <= 0 or config.max_position_size > 1:
            errors.append("max_position_size must be between 0 and 1")

        if config.max_drawdown <= 0 or config.max_drawdown > 1:
            errors.append("max_drawdown must be between 0 and 1")

        if config.stop_loss <= 0:
            errors.append("stop_loss must be positive")

        if config.take_profit <= 0:
            errors.append("take_profit must be positive")

        if config.initial_capital <= 0:
            errors.append("initial_capital must be positive")

        if config.max_leverage <= 0:
            errors.append("max_leverage must be positive")

        if config.batch_size <= 0:
            errors.append("batch_size must be positive")

        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")

        if config.max_epochs <= 0:
            errors.append("max_epochs must be positive")

        if errors:
            return {
                "success": False,
                "error": f'Config validation failed: {"; ".join(errors)}',
                "timestamp": datetime.now().isoformat(),
            }

        logger.info("Config validation passed")

        return {
            "success": True,
            "message": "Config validation passed",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error validating config: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
