"""
Strategy Logger Module

This module provides functionality for logging strategy decisions, including
agentic selections and manual overrides, with detailed metadata and analysis.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

LOG_PATH = "memory/strategy_override_log.json"
ANALYSIS_PATH = "memory/strategy_analysis.json"


def log_strategy_decision(
    ticker: str, selected_model: str, is_agentic: bool, confidence: float, metadata: Optional[Dict] = None
) -> Dict:
    """
    Logs whether the selected strategy was agentic or manually overridden.

    Args:
        ticker (str): The ticker symbol being analyzed.
        selected_model (str): The selected forecasting model.
        is_agentic (bool): Whether the selection was made by the agent.
        confidence (float): The confidence score for the selected model.
        metadata (Dict, optional): Additional metadata to log.

    Returns:
        Dict: Operation result.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "model": selected_model,
        "agentic": is_agentic,
        "confidence": round(confidence, 4),
        "metadata": metadata or {},
    }

    # Load existing log
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Append new entry
    data.append(entry)

    # Save updated log
    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Update analysis
    _update_strategy_analysis(ticker, entry)


def get_strategy_history(ticker: Optional[str] = None, model: Optional[str] = None, limit: int = 100) -> List[Dict]:
    """
    Retrieves strategy decision history with optional filtering.

    Args:
        ticker (str, optional): Filter by ticker symbol.
        model (str, optional): Filter by model name.
        limit (int): Maximum number of entries to return.

    Returns:
        List[Dict]: List of strategy decision entries.
    """
    if not os.path.exists(LOG_PATH):
        return []

    with open(LOG_PATH, "r") as f:
        data = json.load(f)

    # Apply filters
    if ticker:
        data = [entry for entry in data if entry["ticker"] == ticker]
    if model:
        data = [entry for entry in data if entry["model"] == model]

    # Sort by timestamp and limit
    data.sort(key=lambda x: x["timestamp"], reverse=True)
    return data[:limit]


def get_strategy_analysis(ticker: str) -> Dict:
    """
    Retrieves analysis of strategy decisions for a ticker.

    Args:
        ticker (str): The ticker symbol to analyze.

    Returns:
        Dict: Analysis of strategy decisions.
    """
    if not os.path.exists(ANALYSIS_PATH):
        return {}

    with open(ANALYSIS_PATH, "r") as f:
        analysis = json.load(f)

    return analysis.get(ticker, {})


def _update_strategy_analysis(ticker: str, entry: Dict) -> None:
    """
    Updates the strategy analysis for a ticker.

    Args:
        ticker (str): The ticker symbol being analyzed.
        entry (Dict): The new strategy decision entry.
    """
    # Load existing analysis
    if os.path.exists(ANALYSIS_PATH):
        with open(ANALYSIS_PATH, "r") as f:
            analysis = json.load(f)
    else:
        analysis = {}

    # Get or create ticker analysis
    ticker_analysis = analysis.get(
        ticker,
        {
            "total_decisions": 0,
            "agentic_decisions": 0,
            "manual_overrides": 0,
            "model_usage": {},
            "average_confidence": 0.0,
            "last_updated": None,
        },
    )

    # Update analysis
    ticker_analysis["total_decisions"] += 1
    if entry["agentic"]:
        ticker_analysis["agentic_decisions"] += 1
    else:
        ticker_analysis["manual_overrides"] += 1

    # Update model usage
    model = entry["model"]
    ticker_analysis["model_usage"][model] = ticker_analysis["model_usage"].get(model, 0) + 1

    # Update average confidence
    current_total = ticker_analysis["average_confidence"] * (ticker_analysis["total_decisions"] - 1)
    ticker_analysis["average_confidence"] = (current_total + entry["confidence"]) / ticker_analysis["total_decisions"]

    # Update timestamp
    ticker_analysis["last_updated"] = entry["timestamp"]

    # Save updated analysis
    analysis[ticker] = ticker_analysis
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(analysis, f, indent=2)
