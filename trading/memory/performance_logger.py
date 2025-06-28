"""Performance logging utilities."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def log_strategy_performance(
    strategy_name: str,
    performance_metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log strategy performance metrics.
    
    Args:
        strategy_name: Name of the strategy
        performance_metrics: Dictionary of performance metrics
        metadata: Additional metadata
    """
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy_name,
        "metrics": performance_metrics,
        "metadata": metadata or {}
    }
    
    logger.info(f"Strategy Performance: {json.dumps(log_data)}")

def log_performance(
    ticker: str,
    model: str,
    agentic: bool,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance metrics for a specific ticker and model.
    
    Args:
        ticker: Stock ticker symbol
        model: Model name used
        agentic: Whether agentic selection was used
        metrics: Dictionary of performance metrics
        metadata: Additional metadata
    """
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "model": model,
        "agentic": agentic,
        "metrics": metrics,
        "metadata": metadata or {}
    }
    
    logger.info(f"Performance Log: {json.dumps(log_data)}") 