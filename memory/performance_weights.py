"""
Performance Weights Module

This module provides functionality for computing, managing, and monitoring model weights
based on historical performance metrics, including drift detection and retraining triggers.
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd

from models.retrain import trigger_retraining_if_needed
from trading.memory.performance_memory import PerformanceMemory

# Configure logging
logger = logging.getLogger(__name__)

WEIGHT_HISTORY_PATH = "memory/weight_history.json"
CURRENT_WEIGHT_PATH = "memory/model_weights.json"
AUDIT_LOG_PATH = "memory/weight_audit_log.json"


def compute_model_weights(ticker: str, strategy="balanced", min_weight=0.05):
    """
    Compute model weights with optional smoothing based on historical performance.

    Args:
        ticker (str): Ticker symbol to evaluate.
        strategy (str): Weight computation strategy.
        min_weight (float): Minimum weight per model.

    Returns:
        dict: Smoothed weights by model name.
    """
    memory = PerformanceMemory()
    metrics = memory.get_metrics(ticker)
    scores = {}

    for model, data in metrics.items():
        mse = data.get("mse", 1)
        sharpe = data.get("sharpe", 0)
        win = data.get("win_rate", 0)

        if strategy == "balanced":
            score = (1 / (mse + 1e-6)) * 0.4 + sharpe * 0.4 + win * 0.2
        else:
            score = sharpe  # Add more strategy types if needed

        scores[model] = max(score, 0.01)

    total = sum(scores.values())
    raw_weights = {k: v / total for k, v in scores.items()}
    smoothed_weights = smooth_weights(raw_weights, ticker)

    return smoothed_weights


def smooth_weights(weights: dict, ticker: str):
    """
    Apply exponential weighted moving average smoothing to weights.

    Args:
        weights (dict): Current raw weights.
        ticker (str): Ticker symbol for historical data lookup.

    Returns:
        dict: Smoothed weights.
    """
    if not os.path.exists(WEIGHT_HISTORY_PATH):
        return weights

    with open(WEIGHT_HISTORY_PATH, "r") as f:
        history = json.load(f)

    df = pd.DataFrame(
        [entry[ticker] for ts, entry in sorted(history.items()) if ticker in entry]
    )

    df = df.tail(5).append(weights, ignore_index=True)  # add current
    smoothed = df.ewm(span=3).mean().iloc[-1].to_dict()
    total = sum(smoothed.values())
    return {k: round(v / total, 4) for k, v in smoothed.items()}


def export_weights_to_file(ticker: str, strategy="balanced"):
    """
    Compute, smooth, and export model weights with audit logging.

    Args:
        ticker (str): Ticker symbol to evaluate.
        strategy (str): Weight computation strategy.

    Returns:
        dict: The computed and smoothed weights.
    """
    weights = compute_model_weights(ticker, strategy)
    now = datetime.utcnow().isoformat()

    # Save current weights
    with open(CURRENT_WEIGHT_PATH, "w") as f:
        json.dump(weights, f, indent=2)

    # Update history
    history = {}
    if os.path.exists(WEIGHT_HISTORY_PATH):
        with open(WEIGHT_HISTORY_PATH, "r") as f:
            history = json.load(f)

    history[now] = {ticker: weights}
    with open(WEIGHT_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Log audit entry
    audit_entry = {
        "time": now,
        "ticker": ticker,
        "weights": weights,
        "justification": "Auto-computed from performance memory and EWMA smoothing",
    }
    append_to_log(audit_entry, AUDIT_LOG_PATH)

    # Trigger retraining if needed
    retrain_log = trigger_retraining_if_needed(weights, threshold=0.1)
    append_to_log({"time": now, "retraining": retrain_log}, "memory/retrain_log.json")

    return weights


def append_to_log(entry, path):
    """
    Append an entry to a JSON log file.

    Args:
        entry (dict): Log entry to append.
        path (str): Path to the log file.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def detect_weight_drift(history: dict, sensitivity=0.1):
    """
    Alerts if model weights change significantly between last two runs.

    Args:
        history (dict): Weight history dictionary.
        sensitivity (float): Threshold for drift detection.
    """
    if len(history) < 2:
        return

    timestamps = sorted(history.keys())[-2:]
    last, current = history[timestamps[0]], history[timestamps[1]]

    for ticker in current:
        for model in current[ticker]:
            old = last.get(ticker, {}).get(model, 0)
            new = current[ticker].get(model, 0)
            drift = abs(new - old)
            if drift > sensitivity:
                logger.warning(
                    f"[DRIFT] {model} weight changed by {drift:.2f} on {ticker}"
                )


def get_latest_weights():
    """
    Retrieve the most recent model weights.

    Returns:
        dict: Latest weights by model name.
    """
    try:
        with open(CURRENT_WEIGHT_PATH, 'r') as f:
            weights = json.load(f)
            return weights
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Weight file not found or corrupted — using default weights.")
        weights = {'LSTM': 0.25, 'XGB': 0.25, 'ARIMA': 0.25, 'Prophet': 0.25}
        return weights
