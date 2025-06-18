"""
Strategy Switcher Module

This module provides functionality for dynamically switching forecasting strategies
based on model trust levels and performance metrics, with automatic drift detection
and retraining triggers.
"""

import os
import json
from datetime import datetime
from memory.model_monitor import detect_drift
from memory.performance_weights import export_weights_to_file
from models.retrain import trigger_retraining_if_needed

STRATEGY_LOG_PATH = "memory/strategy_switches.json"

def switch_strategy_if_needed(ticker: str, strategy_mode="auto"):
    """
    Switches forecasting strategy dynamically based on most trusted model weight.
    Optionally retrains if drift is detected.

    Args:
        ticker (str): The ticker symbol to analyze.
        strategy_mode (str): Either "auto" for automatic selection or a specific model name.

    Returns:
        dict: Dictionary containing strategy switch information including:
            - ticker: The analyzed ticker
            - strategy: The selected primary model
            - weights: Current model weights
            - drift_detected: Boolean indicating if drift was detected
    """
    # Get smoothed weights
    weights = export_weights_to_file(ticker, strategy="balanced")
    sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    # Select primary model
    if strategy_mode == "auto":
        primary_model = sorted_models[0][0]
    else:
        primary_model = strategy_mode  # fallback to manual override

    # Detect drift and retrain if needed
    drift_detected = detect_drift(ticker)
    if drift_detected:
        trigger_retraining_if_needed(weights, threshold=0.1)

    # Log strategy switch
    switch_info = {
        "ticker": ticker,
        "strategy": primary_model,
        "weights": weights,
        "drift_detected": bool(drift_detected),
        "timestamp": datetime.utcnow().isoformat(),
        "mode": strategy_mode
    }
    _log_strategy_switch(switch_info)

    return switch_info


def _log_strategy_switch(switch_info: dict):
    """
    Logs strategy switch information to a JSON file.

    Args:
        switch_info (dict): Dictionary containing strategy switch details.
    """
    if os.path.exists(STRATEGY_LOG_PATH):
        with open(STRATEGY_LOG_PATH, "r") as f:
            log = json.load(f)
    else:
        log = []

    log.append(switch_info)
    with open(STRATEGY_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def get_strategy_history(ticker: str = None):
    """
    Retrieves the history of strategy switches.

    Args:
        ticker (str, optional): Filter history for a specific ticker.

    Returns:
        list: List of strategy switch records.
    """
    if not os.path.exists(STRATEGY_LOG_PATH):
        return []

    with open(STRATEGY_LOG_PATH, "r") as f:
        log = json.load(f)

    if ticker:
        return [entry for entry in log if entry["ticker"] == ticker]
    return log 