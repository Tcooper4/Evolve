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

# TensorFlow support removed - using PyTorch only
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


def setup_pytorch_gpu(device_id: Optional[int] = None, allow_growth: bool = True) -> bool:
    """
    Setup PyTorch GPU configuration with proper memory management.
    
    This function configures PyTorch to use GPU efficiently:
    - Sets CUDA device if available
    - Configures memory management
    - Enables optimizations for GPU computation
    
    Args:
        device_id: Specific GPU device ID to use (None for default)
        allow_growth: If True, allows memory growth (recommended)
    
    Returns:
        True if GPU was successfully configured, False otherwise
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available - cannot setup GPU")
        return False
    
    if not torch.cuda.is_available():
        logger.debug("CUDA not available - GPU setup skipped")
        return False
    
    try:
        # Set device if specified
        if device_id is not None:
            torch.cuda.set_device(device_id)
        
        # Get current device info
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        # Configure memory management
        if allow_growth:
            # PyTorch doesn't have the same memory growth setting as TensorFlow,
            # but we can set memory fraction to allow growth
            # Note: PyTorch handles memory more dynamically by default
            logger.info(f"PyTorch GPU configured: {device_name} (device {current_device})")
        
        # Set memory fraction if needed (optional - PyTorch handles this well by default)
        # torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        logger.info(f"PyTorch GPU setup complete: {device_name}")
        logger.info(f"  Device: {current_device}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"PyTorch GPU setup failed: {e}")
        return False


def setup_tensorflow_gpu():
    """
    TensorFlow GPU setup removed - PyTorch only.
    
    This function is kept for backward compatibility. It redirects to PyTorch GPU setup.
    Use setup_pytorch_gpu() directly for better control.

    Returns:
        True if PyTorch GPU was configured, False otherwise
    """
    logger.info("TensorFlow GPU setup called - redirecting to PyTorch GPU setup")
    return setup_pytorch_gpu()


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

