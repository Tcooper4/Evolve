"""
GPU Utility Functions

Centralized GPU/CUDA device management and utilities for the Evolve trading system.
Provides consistent device detection, memory management, and GPU configuration.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Try to import TensorFlow
try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None


def get_device(device_preference: Optional[str] = None) -> str:
    """
    Get the best available device (GPU/CPU) for computation.

    Args:
        device_preference: Preferred device ('cuda', 'cpu', 'mps', or None for auto)

    Returns:
        Device string ('cuda', 'cpu', or 'mps')
    """
    if device_preference:
        device_preference = device_preference.lower()
        if device_preference in ["cuda", "gpu"]:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, using CPU")
                return "cpu"
        elif device_preference == "mps":
            if TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                logger.warning("MPS requested but not available, using CPU")
                return "cpu"
        else:
            return "cpu"

    # Auto-detect best device
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

    return "cpu"


def get_torch_device(device_preference: Optional[str] = None):
    """
    Get PyTorch device object.

    Args:
        device_preference: Preferred device ('cuda', 'cpu', 'mps', or None for auto)

    Returns:
        torch.device object
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")

    device_str = get_device(device_preference)
    return torch.device(device_str)


def is_gpu_available() -> bool:
    """
    Check if GPU is available.

    Returns:
        True if GPU is available, False otherwise
    """
    if TORCH_AVAILABLE:
        return torch.cuda.is_available()
    return False


def get_gpu_info() -> dict:
    """
    Get GPU information if available.

    Returns:
        Dictionary with GPU information or empty dict if no GPU
    """
    info = {}
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            info["available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**3  # GB
            info["memory_reserved"] = torch.cuda.memory_reserved(0) / 1024**3  # GB
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        except Exception as e:
            logger.warning(f"Error getting GPU info: {e}")
            info["available"] = False
    else:
        info["available"] = False

    return info


def clear_gpu_cache():
    """
    Clear GPU memory cache if using CUDA.
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing GPU cache: {e}")


def get_xgboost_device() -> dict:
    """
    Get XGBoost GPU configuration if available.

    Returns:
        Dictionary with XGBoost device parameters
    """
    config = {}
    
    # Check if CUDA is available for XGBoost
    if is_gpu_available():
        try:
            # XGBoost can use GPU via tree_method='gpu_hist'
            config["tree_method"] = "gpu_hist"
            config["predictor"] = "gpu_predictor"
            logger.info("XGBoost configured to use GPU")
        except Exception as e:
            logger.warning(f"XGBoost GPU configuration failed: {e}, using CPU")
            config["tree_method"] = "hist"
            config["predictor"] = "cpu_predictor"
    else:
        config["tree_method"] = "hist"
        config["predictor"] = "cpu_predictor"
        logger.debug("XGBoost using CPU (no GPU available)")

    return config


def setup_tensorflow_gpu():
    """
    Setup TensorFlow GPU configuration if available.

    Returns:
        True if GPU is configured, False otherwise
    """
    if not TENSORFLOW_AVAILABLE:
        return False

    try:
        # Enable memory growth to avoid allocating all GPU memory
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"TensorFlow GPU configured: {len(gpus)} GPU(s) available")
                return True
            except RuntimeError as e:
                logger.warning(f"TensorFlow GPU configuration error: {e}")
        else:
            logger.debug("No TensorFlow GPU devices found")
            return False
    except Exception as e:
        logger.warning(f"TensorFlow GPU setup failed: {e}")
        return False

    return False


def get_device_memory_info(device: Optional[str] = None) -> dict:
    """
    Get memory information for the specified device.

    Args:
        device: Device string ('cuda', 'cpu', etc.) or None for auto

    Returns:
        Dictionary with memory information
    """
    info = {"device": device or get_device(), "memory": {}}

    if info["device"] == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            info["memory"]["allocated_gb"] = torch.cuda.memory_allocated(0) / 1024**3
            info["memory"]["reserved_gb"] = torch.cuda.memory_reserved(0) / 1024**3
            info["memory"]["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["memory"]["free_gb"] = info["memory"]["total_gb"] - info["memory"]["reserved_gb"]
        except Exception as e:
            logger.warning(f"Error getting CUDA memory info: {e}")

    return info


def move_to_device(tensor_or_model, device: Optional[str] = None):
    """
    Move tensor or model to specified device.

    Args:
        tensor_or_model: PyTorch tensor or model
        device: Device string or None for auto

    Returns:
        Tensor or model moved to device
    """
    if not TORCH_AVAILABLE:
        return tensor_or_model

    if device is None:
        device = get_device()

    try:
        return tensor_or_model.to(torch.device(device))
    except Exception as e:
        logger.warning(f"Error moving to device {device}: {e}")
        return tensor_or_model

