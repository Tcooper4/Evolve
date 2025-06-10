import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

def setup_logger(name: str, log_dir: Optional[str] = None,
                level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log directory is provided
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """Log configuration dictionary.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for subkey, subvalue in value.items():
                logger.info(f"  {subkey}: {subvalue}")
        else:
            logger.info(f"{key}: {value}")

def save_log_config(config: Dict[str, Any], log_dir: str, name: str) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to store config file
        name: Config file name
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = log_path / f"{name}_{timestamp}.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def load_log_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config 