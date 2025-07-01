import os
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional
import logging.handlers

class LogManager:
    """Base class for managing logging operations."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the log manager.
        
        Args:
            log_dir (str): Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logger(self, name: str, level: int = logging.INFO) -> logging.Logger:
        """Set up a logger with file and console handlers.
        
        Args:
            name (str): Name of the logger
            level (int): Logging level
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def cleanup_logs(self, days: int = 30) -> None:
        """Clean up old log files.
        
        Args:
            days (int): Number of days to keep logs
        """
        if days <= 0:
            # Remove all logs
            for log_file in self.log_dir.glob("*.log"):
                log_file.unlink()
        else:
            # Remove logs older than specified days
            current_time = datetime.now()
            for log_file in self.log_dir.glob("*.log"):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if (current_time - file_time).days > days:
                    log_file.unlink()

class ModelLogger(LogManager):
    """Logger for model-related operations."""
    
    def __init__(self, log_dir: str = "logs/models"):
        """Initialize the model logger.
        
        Args:
            log_dir (str): Directory to store model logs
        """
        super().__init__(log_dir)
        self.logger = self.setup_logger("model")
    
    def log_model_info(self, info: Dict[str, Any]) -> None:
        """Log model information.
        
        Args:
            info (Dict[str, Any]): Model information dictionary
        """
        self.logger.info("Model Information:\n%s", json.dumps(info, indent=2))

    def log_training_progress(self, epoch: int, loss: float, val_loss: Optional[float] = None) -> None:
        """Log training progress.
        
        Args:
            epoch (int): Current epoch
            loss (float): Training loss
            val_loss (Optional[float]): Validation loss
        """
        message = f"Epoch {epoch} - Loss: {loss:.4f}"
        if val_loss is not None:
            message += f" - Val Loss: {val_loss:.4f}"
        self.logger.info(message)

    def log_model_metrics(self, metrics: Dict[str, float]) -> None:
        """Log model metrics.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metrics
        """
        self.logger.info("Model Metrics:\n%s", json.dumps(metrics, indent=2))

class DataLogger(LogManager):
    """Logger for data-related operations."""
    
    def __init__(self, log_dir: str = "logs/data"):
        """Initialize the data logger.
        
        Args:
            log_dir (str): Directory to store data logs
        """
        super().__init__(log_dir)
        self.logger = self.setup_logger("data")
    
    def log_data_info(self, info: Dict[str, Any]) -> None:
        """Log data information.
        
        Args:
            info (Dict[str, Any]): Data information dictionary
        """
        self.logger.info("Data Information:\n%s", json.dumps(info, indent=2))

    def log_data_statistics(self, stats: Dict[str, float]) -> None:
        """Log data statistics.
        
        Args:
            stats (Dict[str, float]): Dictionary of statistics
        """
        self.logger.info("Data Statistics:\n%s", json.dumps(stats, indent=2))

    def log_data_validation(self, validation: Dict[str, int]) -> None:
        """Log data validation results.
        
        Args:
            validation (Dict[str, int]): Dictionary of validation results
        """
        self.logger.info("Data Validation:\n%s", json.dumps(validation, indent=2))

class PerformanceLogger(LogManager):
    """Logger for performance-related operations."""
    
    def __init__(self, log_dir: str = "logs/performance"):
        """Initialize the performance logger.
        
        Args:
            log_dir (str): Directory to store performance logs
        """
        super().__init__(log_dir)
        self.logger = self.setup_logger("performance")
    
    def log_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Log performance metrics.
        
        Args:
            metrics (Dict[str, float]): Dictionary of performance metrics
        """
        self.logger.info("Performance Metrics:\n%s", json.dumps(metrics, indent=2))

    def log_trade_info(self, trade: Dict[str, Any]) -> None:
        """Log trade information.
        
        Args:
            trade (Dict[str, Any]): Dictionary of trade information
        """
        self.logger.info("Trade Information:\n%s", json.dumps(trade, indent=2))

    def log_portfolio_value(self, value: float) -> None:
        """Log portfolio value.
        
        Args:
            value (float): Current portfolio value
        """
        self.logger.info("Portfolio Value: %.2f", value)

def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO):
    """Set up logging configuration with rotating file handler and console handler."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'trading.log'
    
    # Create a logger
    logger = logging.getLogger('trading')
    logger.setLevel(log_level)
    
    # Create a rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(log_level)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger