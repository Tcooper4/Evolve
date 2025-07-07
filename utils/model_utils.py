"""Model utilities for loading and saving model states."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)

def load_model_state(model_path: Path) -> Dict[str, Any]:
    """Load model state from file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model state
    """
    try:
        if model_path.exists():
            state_dict = torch.load(model_path, map_location='cpu')
            logger.info(f"Successfully loaded model state from {model_path}")
            return state_dict
        else:
            logger.warning(f"Model file not found at {model_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading model state from {model_path}: {str(e)}")
        return {}

def save_model_state(model_state: Dict[str, Any], save_path: Path) -> bool:
    """Save model state to file.
    
    Args:
        model_state: Model state dictionary
        save_path: Path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state, save_path)
        logger.info(f"Successfully saved model state to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model state to {save_path}: {str(e)}")
        return False

def get_model_info(model_path: Path) -> Dict[str, Any]:
    """Get information about a model file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model information
    """
    try:
        if model_path.exists():
            state_dict = torch.load(model_path, map_location='cpu')
            return {
                'file_size': model_path.stat().st_size,
                'num_parameters': len(state_dict),
                'parameter_names': list(state_dict.keys()),
                'total_params': sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
            }
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting model info from {model_path}: {str(e)}")
        return {} 