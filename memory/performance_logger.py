"""
Performance Logger Module

This module provides functionality for logging and analyzing strategy performance metrics,
including tracking of model performance, agentic vs manual decisions, and historical analysis.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union


PERF_LOG = "memory/performance_log.json"
PERF_ANALYSIS = "memory/performance_analysis.json"


def log_strategy_performance(
    ticker: str, model: str, agentic: bool, metrics: Dict[str, Union[float, int]], metadata: Optional[Dict] = None
) -> None:
    """
    Store performance metrics tied to a strategy selection.

    Args:
        ticker (str): The ticker symbol being analyzed.
        model (str): The model used for the strategy.
        agentic (bool): Whether the selection was made by the agent.
        metrics (Dict[str, Union[float, int]]): Performance metrics dictionary.
            Example: {"sharpe": 1.2, "accuracy": 0.68, "win_rate": 0.75}
        metadata (Dict, optional): Additional metadata to store.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "model": model,
        "agentic": agentic,
        "metrics": metrics,
        "metadata": metadata or {},
    }

    # Load existing log
    if os.path.exists(PERF_LOG):
        with open(PERF_LOG, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Append new entry
    data.append(entry)

    # Save updated log
    with open(PERF_LOG, "w") as f:
        json.dump(data, f, indent=2)

    # Update performance analysis
    _update_performance_analysis(ticker, entry)


def get_performance_history(ticker: Optional[str] = None, model: Optional[str] = None, limit: int = 100) -> List[Dict]:
    """
    Retrieve performance history with optional filtering.

    Args:
        ticker (str, optional): Filter by ticker symbol.
        model (str, optional): Filter by model name.
        limit (int): Maximum number of entries to return.

    Returns:
        List[Dict]: List of performance entries.
    """
    if not os.path.exists(PERF_LOG):
        return []

    with open(PERF_LOG, "r") as f:
        data = json.load(f)

    # Apply filters
    if ticker:
        data = [entry for entry in data if entry["ticker"] == ticker]
    if model:
        data = [entry for entry in data if entry["model"] == model]

    # Sort by timestamp and limit
    data.sort(key=lambda x: x["timestamp"], reverse=True)
    return data[:limit]


def get_performance_analysis(ticker: str) -> Dict:
    """
    Get performance analysis for a ticker.

    Args:
        ticker (str): The ticker symbol to analyze.

    Returns:
        Dict: Performance analysis including aggregated metrics.
    """
    if not os.path.exists(PERF_ANALYSIS):
        return {}

    with open(PERF_ANALYSIS, "r") as f:
        analysis = json.load(f)

    return analysis.get(ticker, {})


def _update_performance_analysis(ticker: str, entry: Dict) -> None:
    """
    Update performance analysis for a ticker.

    Args:
        ticker (str): The ticker symbol being analyzed.
        entry (Dict): The new performance entry.
    """
    # Load existing analysis
    if os.path.exists(PERF_ANALYSIS):
        with open(PERF_ANALYSIS, "r") as f:
            analysis = json.load(f)
    else:
        analysis = {}

    # Get or create ticker analysis
    ticker_analysis = analysis.get(
        ticker,
        {
            "total_entries": 0,
            "agentic_entries": 0,
            "manual_entries": 0,
            "model_performance": {},
            "metric_averages": {},
            "last_updated": None,
        },
    )

    # Update entry counts
    ticker_analysis["total_entries"] += 1
    if entry["agentic"]:
        ticker_analysis["agentic_entries"] += 1
    else:
        ticker_analysis["manual_entries"] += 1

    # Update model performance
    model = entry["model"]
    if model not in ticker_analysis["model_performance"]:
        ticker_analysis["model_performance"][model] = {"count": 0, "metrics_sum": {}, "metrics_avg": {}}

    model_stats = ticker_analysis["model_performance"][model]
    model_stats["count"] += 1

    # Update metric averages
    for metric, value in entry["metrics"].items():
        if metric not in model_stats["metrics_sum"]:
            model_stats["metrics_sum"][metric] = 0
            model_stats["metrics_avg"][metric] = 0

        model_stats["metrics_sum"][metric] += value
        model_stats["metrics_avg"][metric] = model_stats["metrics_sum"][metric] / model_stats["count"]

    # Update timestamp
    ticker_analysis["last_updated"] = entry["timestamp"]

    # Save updated analysis
    analysis[ticker] = ticker_analysis
    with open(PERF_ANALYSIS, "w") as f:
        json.dump(analysis, f, indent=2)
