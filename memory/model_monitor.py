"""
Model Monitoring Module

This module provides functionality for monitoring model performance and weight changes,
including drift detection and strategy priority generation.
"""

import os
import json
import pandas as pd
from datetime import datetime

WEIGHT_HISTORY_PATH = "memory/weight_history.json"
DRIFT_LOG_PATH = "memory/drift_alerts.json"
STRATEGY_PRIORITY_PATH = "memory/strategy_priority.json"

def detect_drift(ticker: str, threshold=0.2) -> dict:
    """
    Detects significant change in model weights over time for a ticker.
    Logs a drift alert if any model's weight changes beyond the threshold.

    Args:
        ticker (str): The ticker symbol to analyze.
        threshold (float): The weight change threshold for drift detection.

    Returns:
        dict: Dictionary containing drift alerts for each affected model.
    """
    if not os.path.exists(WEIGHT_HISTORY_PATH):
        return {}

    with open(WEIGHT_HISTORY_PATH, "r") as f:
        history = json.load(f)

    records = [entry[ticker] for ts, entry in sorted(history.items()) if ticker in entry]
    if len(records) < 2:
        return {'success': True, 'result': {}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    df = pd.DataFrame(records)
    drift_alerts = []

    last = df.iloc[-1]
    prev = df.iloc[-2]

    for model in last.index:
        change = abs(last[model] - prev[model])
        if change > threshold:
            drift_alerts.append({
                "model": model,
                "previous_weight": round(prev[model], 4),
                "current_weight": round(last[model], 4),
                "change": round(change, 4),
                "time": datetime.utcnow().isoformat()
            })

    if drift_alerts:
        _append_json_log(DRIFT_LOG_PATH, {"ticker": ticker, "drift": drift_alerts})
    return drift_alerts


def generate_strategy_priority(ticker: str) -> dict:
    """
    Converts latest smoothed weights into strategy priorities for each model.

    Args:
        ticker (str): The ticker symbol to analyze.

    Returns:
        dict: Dictionary containing prioritized models and their weights.
    """
    if not os.path.exists(WEIGHT_HISTORY_PATH):
        return {'success': True, 'result': {}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    with open(WEIGHT_HISTORY_PATH, "r") as f:
        history = json.load(f)

    latest = sorted(history.items())[-1][1]
    weights = latest.get(ticker, {})
    priority = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    result = {
        "ticker": ticker,
        "prioritized_models": [m for m, _ in priority],
        "weights": dict(priority),
        "time": datetime.utcnow().isoformat()
    }

    _append_json_log(STRATEGY_PRIORITY_PATH, result)
    return result


def _append_json_log(path: str, entry: dict):
    """
    Appends an entry to a JSON log file.

    Args:
        path (str): Path to the log file.
        entry (dict): Log entry to append.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}