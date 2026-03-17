"""Unified logging utilities for strategy and performance tracking."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd


@dataclass
class PerformanceMetrics:
    """Performance metrics for strategies and models."""

    timestamp: str
    strategy: str
    model: str
    sharpe: float
    drawdown: float
    mse: float
    accuracy: float
    notes: Optional[str] = None


@dataclass
class StrategyDecision:
    """Strategy decision record."""

    timestamp: str
    ticker: str
    strategy: str
    action: str
    confidence: float
    price: float
    volume: float
    indicators: Dict[str, float]
    notes: Optional[str] = None


class UnifiedLogger:
    """Unified logger for strategy and performance tracking."""

    def __init__(self, log_dir: Union[str, Path] = "logs"):
        """Initialize the unified logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        _ = self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup loggers
        self.performance_logger = self._setup_logger(
            "performance",
            self.log_dir / "performance.log",
            max_bytes=10 * 1024 * 1024,
            backup_count=5,  # 10MB
        )

        self.strategy_logger = self._setup_logger(
            "strategy",
            self.log_dir / "strategy.log",
            max_bytes=10 * 1024 * 1024,
            backup_count=5,  # 10MB
        )

        # Setup daily rotating logs
        self.daily_performance_logger = self._setup_daily_logger(
            "daily_performance",
            self.log_dir / "daily_performance.log",
            backup_count=30,  # Keep 30 days
        )

        self.daily_strategy_logger = self._setup_daily_logger(
            "daily_strategy",
            self.log_dir / "daily_strategy.log",
            backup_count=30,  # Keep 30 days
        )
        return None

    def _setup_logger(
        self,
        name: str,
        log_file: Path,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> logging.Logger:
        """Setup a rotating file logger.

        Args:
            name: Logger name
            log_file: Log file path
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep

        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Create rotating file handler
        handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )

        # Create JSON formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def _setup_daily_logger(
        self, name: str, log_file: Path, backup_count: int = 30
    ) -> logging.Logger:
        """Setup a daily rotating file logger.

        Args:
            name: Logger name
            log_file: Log file path
            backup_count: Number of backup files to keep

        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Create daily rotating file handler
        handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
        )

        # Create JSON formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def log_performance(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics.

        Args:
            metrics: Performance metrics to log
        """
        # Log to rotating file
        self.performance_logger.info(json.dumps(asdict(metrics)))

        # Log to daily file
        self.daily_performance_logger.info(json.dumps(asdict(metrics)))

    def log_strategy_decision(self, decision: StrategyDecision) -> None:
        """Log strategy decision.

        Args:
            decision: Strategy decision to log
        """
        # Log to rotating file
        self.strategy_logger.info(json.dumps(asdict(decision)))

        # Log to daily file
        self.daily_strategy_logger.info(json.dumps(asdict(decision)))

    def get_performance_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get performance history.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            strategy: Strategy name for filtering

        Returns:
            DataFrame with performance history
        """
        log_file = self.log_dir / "performance.log"
        if not log_file.exists():
            return pd.DataFrame()

        # Read log file
        records = []
        with open(log_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.split(" - ")[-1])
                    records.append(record)
                except (json.JSONDecodeError, IndexError):
                    continue

        # Convert to DataFrame
        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Apply filters
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]
        if strategy:
            df = df[df["strategy"] == strategy]

        return df

    def get_strategy_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get strategy decision history.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            ticker: Ticker symbol for filtering

        Returns:
            DataFrame with strategy history
        """
        log_file = self.log_dir / "strategy.log"
        if not log_file.exists():
            return pd.DataFrame()

        # Read log file
        records = []
        with open(log_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.split(" - ")[-1])
                    records.append(record)
                except (json.JSONDecodeError, IndexError):
                    continue

        # Convert to DataFrame
        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Apply filters
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]
        if ticker:
            df = df[df["ticker"] == ticker]

        return df

    def archive_logs(self, archive_dir: Union[str, Path]) -> None:
        """Archive old log files.

        Args:
            archive_dir: Directory to store archived logs
        """
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Archive performance logs
        for log_file in self.log_dir.glob("performance.log.*"):
            if log_file.suffix.isdigit():
                archive_path = (
                    archive_dir
                    / f"performance_{datetime.now().strftime('%Y%m%d')}_{log_file.suffix[1:]}.log"
                )
                log_file.rename(archive_path)

        # Archive strategy logs
        for log_file in self.log_dir.glob("strategy.log.*"):
            if log_file.suffix.isdigit():
                archive_path = (
                    archive_dir
                    / f"strategy_{datetime.now().strftime('%Y%m%d')}_{log_file.suffix[1:]}.log"
                )
                log_file.rename(archive_path)

        # Archive daily logs
        for log_file in self.log_dir.glob("daily_*.log.*"):
            if log_file.suffix.isdigit():
                archive_path = (
                    archive_dir
                    / f"{log_file.stem}_{datetime.now().strftime('%Y%m%d')}_{log_file.suffix[1:]}.log"
                )
                log_file.rename(archive_path)


# Create singleton instance
logger = UnifiedLogger()

# Convenience functions


def log_performance(metrics: PerformanceMetrics) -> None:
    """Log performance metrics.

    Args:
        metrics: Performance metrics to log
    """
    logger.log_performance(metrics)


def log_strategy_decision(decision: StrategyDecision) -> None:
    """Log strategy decision.

    Args:
        decision: Strategy decision to log
    """
    logger.log_strategy_decision(decision)


def get_performance_history(**kwargs) -> pd.DataFrame:
    """Get performance history.

    Args:
        **kwargs: Filter parameters

    Returns:
        DataFrame with performance history
    """
    return logger.get_performance_history(**kwargs)


def get_strategy_history(**kwargs) -> pd.DataFrame:
    """Get strategy decision history.

    Args:
        **kwargs: Filter parameters

    Returns:
        DataFrame with strategy history
    """
    return logger.get_strategy_history(**kwargs)


def archive_logs(archive_dir: Union[str, Path]) -> None:
    """Archive old log files.

    Args:
        archive_dir: Directory to store archived logs
    """
    logger.archive_logs(archive_dir)
