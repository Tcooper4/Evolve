"""
Model Registry

This module provides a registry of available models in the trading system,
allowing for dynamic model discovery and management.
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from trading.config.settings import MODEL_DIR

class ModelRegistry:
    """Registry for managing available models."""
    
    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the model registry.
        
        Args:
            registry_path: Path to the registry file
        """
        self.registry_path = Path(registry_path or MODEL_DIR / "model_registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry from file.
        
        Returns:
            Dictionary containing model registry
        """
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._get_default_registry()
        else:
            return self._get_default_registry()
    
    def _get_default_registry(self) -> Dict[str, Any]:
        """Get the default model registry.
        
        Returns:
            Default registry dictionary
        """
        return {'success': True, 'result': {
            "models": {
                "lstm": {
                    "name": "LSTM Model",
                    "class": "LSTMModel",
                    "module": "trading.models.lstm_model",
                    "description": "Long Short-Term Memory neural network",
                    "parameters": {
                        "input_dim": 10,
                        "hidden_dim": 50,
                        "output_dim": 1,
                        "num_layers": 2,
                        "dropout": 0.2
                    }
                },
                "tcn": {
                    "name": "TCN Model",
                    "class": "TCNModel",
                    "module": "trading.models.tcn_model",
                    "description": "Temporal Convolutional Network",
                    "parameters": {
                        "input_dim": 10,
                        "output_dim": 1,
                        "num_channels": [64, 128, 256],
                        "kernel_size": 3,
                        "dropout": 0.2
                    }
                },
                "transformer": {
                    "name": "Transformer Model",
                    "class": "TransformerForecaster",
                    "module": "trading.models.advanced.transformer.time_series_transformer",
                    "description": "Transformer-based time series model",
                    "parameters": {
                        "input_dim": 10,
                        "output_dim": 1,
                        "d_model": 512,
                        "nhead": 8,
                        "num_layers": 6,
                        "dropout": 0.1
                    }
                },
                "xgboost": {
                    "name": "XGBoost Model",
                    "class": "XGBoostModel",
                    "module": "trading.models.xgboost_model",
                    "description": "XGBoost gradient boosting model",
                    "parameters": {
                        "n_estimators": 100,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "subsample": 0.8
                    }
                }
            },
            "ensembles": {
                "weighted_average": {
                    "name": "Weighted Average Ensemble",
                    "description": "Weighted average of multiple models",
                    "models": ["lstm", "tcn", "transformer"]
                },
                "stacking": {
                    "name": "Stacking Ensemble",
                    "description": "Stacking ensemble with meta-learner",
                    "models": ["lstm", "tcn", "xgboost"]
                }
            }
        }, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _save_registry(self) -> None:
        """Save the model registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names.
        
        Returns:
            List of model names
        """
        return list(self.registry["models"].keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        return self.registry["models"].get(model_name)
    
    def get_model_parameters(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get default parameters for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model parameters dictionary or None if not found
        """
        model_info = self.get_model_info(model_name)
        return model_info["parameters"] if model_info else None
    
    def register_model(self, name: str, info: Dict[str, Any]) -> None:
        """Register a new model.
        
        Args:
            name: Model name
            info: Model information dictionary
        """
        self.registry["models"][name] = info
        self._save_registry()
    
    def unregister_model(self, name: str) -> bool:
        """Unregister a model.
        
        Args:
            name: Model name
            
        Returns:
            True if model was unregistered, False if not found
        """
        if name in self.registry["models"]:
            del self.registry["models"][name]
            self._save_registry()
            return True
        return False
    
    def get_ensembles(self) -> List[str]:
        """Get list of available ensemble methods.
        
        Returns:
            List of ensemble names
        """
        return list(self.registry["ensembles"].keys())
    
    def get_ensemble_info(self, ensemble_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific ensemble.
        
        Args:
            ensemble_name: Name of the ensemble
            
        Returns:
            Ensemble information dictionary or None if not found
        """
        return self.registry["ensembles"].get(ensemble_name)
    
    def update_model_parameters(self, model_name: str, parameters: Dict[str, Any]) -> bool:
        """Update parameters for a model.
        
        Args:
            model_name: Name of the model
            parameters: New parameters dictionary
            
        Returns:
            True if parameters were updated, False if model not found
        """
        if model_name in self.registry["models"]:
            self.registry["models"][model_name]["parameters"] = parameters
            self._save_registry()
            return True
        return False

# Global registry instance
_registry_instance = None

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance.
    
    Returns:
        ModelRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance

def get_available_models() -> List[str]:
    """Get list of available model names.
    
    Returns:
        List of model names
    """
    return get_model_registry().get_available_models()

def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model information dictionary or None if not found
    """
    return get_model_registry().get_model_info(model_name)

def get_model_parameters(model_name: str) -> Optional[Dict[str, Any]]:
    """Get default parameters for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model parameters dictionary or None if not found
    """
    return get_model_registry().get_model_parameters(model_name)