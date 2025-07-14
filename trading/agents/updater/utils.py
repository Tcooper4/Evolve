"""
Utility functions for the Updater Agent.

This module contains helper functions for model drift detection,
performance monitoring, and update validation.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("UpdaterUtils")


def check_model_performance(model_id: str, metrics: Dict) -> Tuple[bool, str]:
    """
    Check if a model's performance meets the required thresholds.

    Args:
        model_id: The ID of the model to check
        metrics: Dictionary of model performance metrics

    Returns:
        Tuple[bool, str]: (needs_update, reason)
            - needs_update: True if the model needs updating
            - reason: Explanation for why the model needs updating
    """
    try:
        # Check MSE threshold
        if metrics.get("mse", float("inf")) > 0.1:
            return True, "MSE above threshold"

        # Check Sharpe ratio
        if metrics.get("sharpe_ratio", 0) < 1.0:
            return True, "Sharpe ratio below threshold"

        # Check max drawdown
        if metrics.get("max_drawdown", 0) > 0.2:
            return True, "Max drawdown above threshold"

        return False, ""

    except Exception as e:
        logger.error(f"Error checking model performance: {str(e)}")
        return False, ""


def detect_model_drift(model_id: str, recent_data: pd.DataFrame) -> Tuple[bool, float]:
    """
    Detect if a model has drifted from its expected behavior.

    Args:
        model_id: The ID of the model to check
        recent_data: Recent data for drift detection

    Returns:
        Tuple[bool, float]: (has_drifted, drift_score)
            - has_drifted: True if drift is detected
            - drift_score: Numerical score indicating drift severity
    """
    try:
        # Load historical performance data for comparison
        historical_file = f"logs/model_performance_{model_id}.json"
        if not os.path.exists(historical_file):
            logger.warning(f"No historical data found for model {model_id}")
            return False, 0.0

        with open(historical_file, "r") as f:
            historical_data = json.load(f)

        # Extract historical predictions and actual values
        historical_predictions = historical_data.get("predictions", [])
        historical_actuals = historical_data.get("actuals", [])

        if len(historical_predictions) < 50 or len(recent_data) < 10:
            logger.warning(
                f"Insufficient data for drift detection: historical={len(historical_predictions)}, recent={len(recent_data)}"
            )
            return False, 0.0

        # Calculate drift metrics
        drift_metrics = {}

        # 1. Performance drift (prediction accuracy)
        if len(historical_predictions) > 0:
            historical_mae = np.mean(
                np.abs(np.array(historical_predictions) - np.array(historical_actuals))
            )
            recent_mae = np.mean(
                np.abs(recent_data["predictions"] - recent_data["actuals"])
            )
            performance_drift = abs(recent_mae - historical_mae) / (
                historical_mae + 1e-8
            )
            drift_metrics["performance"] = performance_drift

        # 2. Distribution drift (if we have feature data)
        if (
            "features" in recent_data.columns
            and len(historical_data.get("features", [])) > 0
        ):
            historical_features = np.array(historical_data["features"])
            recent_features = recent_data["features"].values

            # Calculate KL divergence for distribution comparison
            try:
                from scipy.stats import entropy
                from sklearn.preprocessing import StandardScaler

                # Normalize features for comparison
                scaler = StandardScaler()
                historical_norm = scaler.fit_transform(historical_features)
                recent_norm = scaler.transform(recent_features)

                # Calculate distribution drift using histogram comparison
                hist_old, _ = np.histogram(
                    historical_norm.flatten(), bins=20, density=True
                )
                hist_new, _ = np.histogram(recent_norm.flatten(), bins=20, density=True)

                # Add small epsilon to avoid division by zero
                hist_old = hist_old + 1e-10
                hist_new = hist_new + 1e-10

                kl_divergence = entropy(hist_new, hist_old)
                drift_metrics["distribution"] = kl_divergence

            except ImportError:
                logger.warning("scipy not available, skipping distribution drift")

        # 3. Statistical drift (mean, variance changes)
        if len(historical_predictions) > 0:
            hist_mean = np.mean(historical_predictions)
            hist_std = np.std(historical_predictions)
            recent_mean = np.mean(recent_data["predictions"])
            recent_std = np.std(recent_data["predictions"])

            mean_drift = abs(recent_mean - hist_mean) / (hist_std + 1e-8)
            std_drift = abs(recent_std - hist_std) / (hist_std + 1e-8)

            drift_metrics["mean_drift"] = mean_drift
            drift_metrics["std_drift"] = std_drift

        # Calculate overall drift score
        if drift_metrics:
            drift_score = np.mean(list(drift_metrics.values()))
            has_drifted = drift_score > 0.5  # Threshold for drift detection

            logger.info(
                f"Drift detection for model {model_id}: score={drift_score:.3f}, drifted={has_drifted}"
            )
            logger.debug(f"Drift metrics: {drift_metrics}")

            return has_drifted, drift_score
        else:
            return False, 0.0

    except Exception as e:
        logger.error(f"Error detecting model drift: {str(e)}")
        return False, 0.0


def validate_update_result(result: Dict) -> bool:
    """
    Validate the result of an update operation.

    Args:
        result: Dictionary containing update result information

    Returns:
        bool: True if the update was successful
    """
    try:
        required_fields = ["success", "model_id", "timestamp", "metrics"]
        if not all(field in result for field in required_fields):
            return False

        # Check if metrics meet minimum requirements
        metrics = result["metrics"]
        if metrics.get("mse", float("inf")) > 0.1:
            return False
        if metrics.get("sharpe_ratio", 0) < 1.0:
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating update result: {str(e)}")
        return False


def calculate_reweighting_factors(
    performance_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate reweighting factors based on model performance metrics.

    Args:
        performance_metrics: Dictionary of model performance metrics

    Returns:
        Dict[str, float]: Dictionary of reweighting factors
    """
    try:
        total_score = sum(performance_metrics.values())
        if total_score == 0:
            return {
                model_id: 1.0 / len(performance_metrics)
                for model_id in performance_metrics
            }

        return {
            model_id: score / total_score
            for model_id, score in performance_metrics.items()
        }

    except Exception as e:
        logger.error(f"Error calculating reweighting factors: {str(e)}")
        return {}


def get_model_metrics(model_id: str) -> Dict[str, float]:
    """
    Get the current performance metrics for a model.

    Args:
        model_id: The ID of the model

    Returns:
        Dict[str, float]: Dictionary of performance metrics
    """
    try:
        metrics_file = Path(f"models/{model_id}/metrics.json")
        if metrics_file.exists():
            return json.loads(metrics_file.read_text())
        return {}

    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        return {}


def check_update_frequency(model_id: str) -> bool:
    """
    Check if a model is due for an update based on its last update time.

    Args:
        model_id: The ID of the model

    Returns:
        bool: True if the model is due for an update
    """
    try:
        last_update_file = Path(f"models/{model_id}/last_update.txt")
        if not last_update_file.exists():
            return True

        last_update = datetime.fromisoformat(last_update_file.read_text().strip())
        update_interval = timedelta(hours=24)  # Configurable

        return datetime.now() - last_update > update_interval

    except Exception as e:
        logger.error(f"Error checking update frequency: {str(e)}")
        return False


def get_ensemble_weights() -> Dict[str, float]:
    """
    Get the current weights for the model ensemble.

    Returns:
        Dict[str, float]: Dictionary of model weights
    """
    try:
        weights_file = Path("models/ensemble_weights.json")
        if weights_file.exists():
            return json.loads(weights_file.read_text())
        return {}

    except Exception as e:
        logger.error(f"Error getting ensemble weights: {str(e)}")
        return {}


def save_ensemble_weights(weights: Dict[str, float]):
    """
    Save the updated ensemble weights.

    Args:
        weights: Dictionary of model weights to save
    """
    try:
        weights_file = Path("models/ensemble_weights.json")
        weights_file.write_text(json.dumps(weights, indent=2))

    except Exception as e:
        logger.error(f"Error saving ensemble weights: {str(e)}")


def check_data_quality(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Check the quality of input data for model updates.

    Args:
        data: DataFrame containing the input data

    Returns:
        Tuple[bool, str]: (is_valid, reason)
            - is_valid: True if the data is valid
            - reason: Explanation if the data is invalid
    """
    try:
        # Check for missing values
        if data.isnull().any().any():
            return False, "Data contains missing values"

        # Check for infinite values
        if np.isinf(data.select_dtypes(include=np.number)).any().any():
            return False, "Data contains infinite values"

        # Check for sufficient data points
        if len(data) < 100:
            return False, "Insufficient data points"

        return True, ""

    except Exception as e:
        logger.error(f"Error checking data quality: {str(e)}")
        return False, str(e)
