# -*- coding: utf-8 -*-
"""Performance logging module for tracking model and strategy metrics."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Constants
LOG_DIR = Path("memory/logs")
LOG_FILE = LOG_DIR / "performance_log.csv"
REQUIRED_FIELDS = ["timestamp", "ticker", "model", "strategy", "sharpe", "drawdown", "mse", "accuracy", "notes"]


def ensure_log_directory():
    """Ensure the log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_performance(
    ticker: str,
    model: str,
    strategy: str,
    sharpe: Optional[float] = None,
    drawdown: Optional[float] = None,
    mse: Optional[float] = None,
    accuracy: Optional[float] = None,
    notes: str = "",
) -> None:
    """
    Log performance metrics to CSV file.

    Args:
        ticker: Stock symbol
        model: Model name/type
        strategy: Strategy name/type
        sharpe: Sharpe ratio
        drawdown: Maximum drawdown
        mse: Mean squared error
        accuracy: Model accuracy
        notes: Additional notes
    """
    # Ensure log directory exists
    ensure_log_directory()

    # Prepare row data
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "model": model,
        "strategy": strategy,
        "sharpe": sharpe,
        "drawdown": drawdown,
        "mse": mse,
        "accuracy": accuracy,
        "notes": notes,
    }

    # Check if file exists to determine if we need to write headers
    file_exists = LOG_FILE.exists()

    # Write to CSV
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_performance_log() -> pd.DataFrame:
    """
    Read and return the performance log as a pandas DataFrame.

    Returns:
        DataFrame containing performance metrics
    """
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=REQUIRED_FIELDS)

    return pd.read_csv(LOG_FILE)


def clear_performance_log() -> None:
    """Clear the performance log file."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()
