"""
Dynamic Model Registry

This module provides a dynamic registry for available models in the trading system,
automatically discovering and loading models from the models directory.
"""

import glob
import importlib
import logging
from pathlib import Path
from typing import List, Optional, Type

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Dynamic registry for managing available models."""

    def __init__(self, models_dir: Optional[str] = None):
        """Initialize the model registry.

        Args:
            models_dir: Path to the models directory
        """
        self.models_dir = Path(models_dir or "trading/models")
        self.registry = {}
        self.loaded_models = set()
        self._discover_models()

    def _discover_models(self):
        """Dynamically discover models from the models directory."""
        try:
            # Look for model files in the models directory
            model_files = glob.glob(str(self.models_dir / "*_model.py"))

            for model_file in model_files:
                try:
                    # Extract model name from filename
                    model_name = Path(model_file).stem.replace("_model", "")

                    # Check for duplicate model names
                    if model_name in self.loaded_models:
                        logger.warning(f"Duplicate model name detected: {model_name}")
                        continue

                    # Import the module
                    module_path = f"trading.models.{model_name}_model"
                    module = importlib.import_module(module_path)

                    # Look for model class (convention: ModelNameModel)
                    model_class_name = f"{model_name.title()}Model"
                    if hasattr(module, model_class_name):
                        model_class = getattr(module, model_class_name)
                        self.registry[model_name] = model_class
                        self.loaded_models.add(model_name)
                        logger.info(f"Discovered model: {model_name}")
                    else:
                        logger.warning(f"No model class found in {module_path}")

                except ImportError as e:
                    logger.warning(f"Failed to import model from {model_file}: {e}")
                except Exception as e:
                    logger.error(f"Error discovering model {model_file}: {e}")

        except Exception as e:
            logger.error(f"Error during model discovery: {e}")

    def get_model_class(self, model_name: str) -> Optional[Type]:
        """Get model class by name.

        Args:
            model_name: Name of the model

        Returns:
            Model class or None if not found
        """
        return self.registry.get(model_name)

    def get_available_models(self) -> List[str]:
        """Get list of available model names.

        Returns:
            List of model names
        """
        return list(self.registry.keys())

    def register_model(self, name: str, model_class: Type):
        """Register a model manually.

        Args:
            name: Model name
            model_class: Model class
        """
        if name in self.loaded_models:
            raise ValueError(f"Duplicate model name detected: {name}")

        self.registry[name] = model_class
        self.loaded_models.add(name)
        logger.info(f"Registered model: {name}")

    def unregister_model(self, name: str) -> bool:
        """Unregister a model.

        Args:
            name: Model name

        Returns:
            True if model was unregistered, False if not found
        """
        if name in self.registry:
            del self.registry[name]
            self.loaded_models.discard(name)
            logger.info(f"Unregistered model: {name}")
            return True
        return False

    def reload_models(self):
        """Reload all models from directory."""
        self.registry.clear()
        self.loaded_models.clear()
        self._discover_models()
        logger.info("Model registry reloaded")


# Global registry instance
_model_registry = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance.

    Returns:
        ModelRegistry instance
    """
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
