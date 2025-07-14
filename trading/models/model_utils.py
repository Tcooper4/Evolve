"""
Model Utility Functions

This module contains utility functions for model operations including
data validation, device management, and common model operations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""


class ModelError(Exception):
    """Custom exception for model errors."""


def validate_data(data: pd.DataFrame, required_columns: list) -> None:
    """Validate input data for model operations.

    Args:
        data: Input data to validate
        required_columns: List of required column names

    Raises:
        ValidationError: If data is invalid
    """
    # Check for missing values
    if data.isnull().any().any():
        raise ValidationError("Data contains missing values")

    # Check for infinite values
    if np.isinf(data.select_dtypes(include=np.number)).any().any():
        raise ValidationError("Data contains infinite values")

    # Check for required columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")


def to_device(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]], device: torch.device
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Move data to specified device.

    Args:
        data: Data to move to device
        device: Target device

    Returns:
        Data on target device
    """
    try:
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: value.to(device) for key, value in data.items()}
        else:
            return data
    except Exception as e:
        logger.error(f"Error moving data to device: {e}")
        raise ModelError(f"Device transfer failed: {e}")


def from_device(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]]
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Move data from device to CPU.

    Args:
        data: Data to move to CPU

    Returns:
        Data on CPU
    """
    try:
        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {key: value.cpu() for key, value in data.items()}
        else:
            return data
    except Exception as e:
        logger.error(f"Error moving data from device: {e}")
        raise ModelError(f"Device transfer failed: {e}")


def safe_forward(model: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
    """Safely run forward pass with error handling.

    Args:
        model: PyTorch model
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Model output

    Raises:
        ModelError: If forward pass fails
    """
    try:
        if model is None:
            raise ModelError("Model not initialized")

        return model(*args, **kwargs)
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        raise ModelError(f"Forward pass failed: {e}")


def compute_loss(
    criterion: torch.nn.Module, y_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """Compute loss with error handling.

    Args:
        criterion: Loss function
        y_pred: Predicted values
        y_true: True values

    Returns:
        Computed loss
    """
    try:
        return criterion(y_pred, y_true)
    except Exception as e:
        logger.error(f"Loss computation failed: {e}")
        raise ModelError(f"Loss computation failed: {e}")


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> list:
    """Compute common metrics.

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        List of metrics [mse, mae, rmse]
    """
    try:
        y_pred_np = y_pred.detach().cpu().numpy().flatten()
        y_true_np = y_true.detach().cpu().numpy().flatten()

        mse = np.mean((y_pred_np - y_true_np) ** 2)
        mae = np.mean(np.abs(y_pred_np - y_true_np))
        rmse = np.sqrt(mse)

        return [mse, mae, rmse]
    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        return [float("inf"), float("inf"), float("inf")]


def get_model_confidence(val_losses: list) -> Dict[str, float]:
    """Get model confidence metrics.

    Args:
        val_losses: List of validation losses

    Returns:
        Dictionary of confidence metrics
    """
    try:
        if not val_losses:
            return {
                "confidence": 0.0,
                "latest_val_loss": float("inf"),
                "best_val_loss": float("inf"),
                "loss_ratio": float("inf"),
            }

        # Simple confidence based on validation loss
        latest_val_loss = val_losses[-1]
        best_val_loss = min(val_losses)

        # Confidence decreases as validation loss increases
        confidence = max(0.0, 1.0 - (latest_val_loss - best_val_loss) / best_val_loss)

        return {
            "confidence": confidence,
            "latest_val_loss": latest_val_loss,
            "best_val_loss": best_val_loss,
            "loss_ratio": latest_val_loss / best_val_loss
            if best_val_loss > 0
            else float("inf"),
        }
    except Exception as e:
        logger.error(f"Confidence calculation failed: {e}")
        return {
            "confidence": 0.0,
            "latest_val_loss": float("inf"),
            "best_val_loss": float("inf"),
            "loss_ratio": float("inf"),
        }


def get_model_metadata(
    model_type: str,
    config: dict,
    device: torch.device,
    best_val_loss: float,
    train_losses: list,
) -> Dict[str, Any]:
    """Get model metadata.

    Args:
        model_type: Type of model
        config: Model configuration
        device: Model device
        best_val_loss: Best validation loss
        train_losses: Training loss history

    Returns:
        Model metadata dictionary
    """
    try:
        return {
            "model_type": model_type,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "device": str(device),
            "best_val_loss": best_val_loss,
            "training_epochs": len(train_losses),
        }
    except Exception as e:
        logger.warning(f"Could not get model metadata: {e}")
        return {
            "model_type": model_type,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "error": str(e),
        }
