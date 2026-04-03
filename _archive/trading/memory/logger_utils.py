"""
Logging utilities for the trading system.
Provides centralized logging configuration and utilities.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Optional

# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = "logs/trading_system.log"


def setup_logging(
    log_level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Setup logging configuration for the trading system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls with parameters and return values.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised exception: {e}")
            raise

    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    import time

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.4f} seconds: {e}"
            )
            raise

    return wrapper


def setup_trading_logger(
    strategy_name: str, log_dir: str = "logs/strategies"
) -> logging.Logger:
    """
    Setup a logger specifically for a trading strategy.

    Args:
        strategy_name: Name of the trading strategy
        log_dir: Directory for strategy logs

    Returns:
        Configured logger for the strategy
    """
    # Create strategy log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create strategy-specific logger
    logger = logging.getLogger(f"trading.strategy.{strategy_name}")

    # Create file handler for strategy
    strategy_log_file = log_path / f"{strategy_name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        strategy_log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",  # 5MB
    )

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    return logger


def log_trade_execution(
    logger: logging.Logger,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    timestamp: str,
    strategy_name: str = "unknown",
) -> None:
    """
    Log trade execution details.

    Args:
        logger: Logger instance
        symbol: Trading symbol
        side: Trade side (BUY/SELL)
        quantity: Trade quantity
        price: Trade price
        timestamp: Trade timestamp
        strategy_name: Name of the strategy that executed the trade
    """
    logger.info(
        f"TRADE_EXECUTED: {strategy_name} - {side} {quantity} {symbol} @ {price} "
        f"at {timestamp}"
    )


def log_portfolio_update(
    logger: logging.Logger,
    portfolio_value: float,
    cash: float,
    positions: Dict[str, float],
    timestamp: str,
) -> None:
    """
    Log portfolio update details.

    Args:
        logger: Logger instance
        portfolio_value: Total portfolio value
        cash: Available cash
        positions: Current positions
        timestamp: Update timestamp
    """
    logger.info(
        f"PORTFOLIO_UPDATE: Value={portfolio_value:.2f}, Cash={cash:.2f}, "
        f"Positions={positions} at {timestamp}"
    )


def log_signal_generated(
    logger: logging.Logger,
    symbol: str,
    signal_type: str,
    signal_strength: float,
    strategy_name: str,
    timestamp: str,
) -> None:
    """
    Log trading signal generation.

    Args:
        logger: Logger instance
        symbol: Trading symbol
        signal_type: Type of signal (BUY/SELL/HOLD)
        signal_strength: Signal strength/confidence
        strategy_name: Name of the strategy
        timestamp: Signal timestamp
    """
    logger.info(
        f"SIGNAL_GENERATED: {strategy_name} - {signal_type} signal for {symbol} "
        f"with strength {signal_strength:.3f} at {timestamp}"
    )
