"""Optimizer factory for creating different types of optimizers."""

import importlib
import inspect
import logging
import os
from typing import Dict, List

import pandas as pd

from .base_optimizer import BaseOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading/optimization/logs/optimizer_factory.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Factory for creating optimizers."""

    def __init__(self):
        """Initialize optimizer factory."""
        self.optimizers = {}
        self._discover_optimizers()

    def _discover_optimizers(self):
        """Discover available optimizers."""
        # Get optimization directory
        opt_dir = os.path.dirname(os.path.abspath(__file__))

        # Import all modules
        for filename in os.listdir(opt_dir):
            if filename.endswith("_optimizer.py"):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f"trading.optimization.{module_name}")

                    # Find optimizer classes
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, BaseOptimizer) and obj != BaseOptimizer:
                            self.optimizers[name] = obj
                            logger.info(f"Discovered optimizer: {name}")

                except Exception as e:
                    logger.error(f"Error importing {module_name}: {e}")

    def get_available_optimizers(self) -> List[str]:
        """Get list of available optimizers.

        Returns:
            List of optimizer names
        """
        return list(self.optimizers.keys())

    def create_optimizer(self, optimizer_type: str, data: pd.DataFrame, strategy_type: str, **kwargs) -> BaseOptimizer:
        """Create optimizer instance.

        Args:
            optimizer_type: Type of optimizer to create
            data: DataFrame with OHLCV data
            strategy_type: Type of strategy to optimize
            **kwargs: Additional optimizer arguments

        Returns:
            Optimizer instance
        """
        if optimizer_type not in self.optimizers:
            raise ValueError(
                f"Unknown optimizer type: {optimizer_type}. " f"Available optimizers: {self.get_available_optimizers()}"
            )

        optimizer_class = self.optimizers[optimizer_type]
        return {
            "success": True,
            "result": optimizer_class(data, strategy_type, **kwargs),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def get_optimizer_info(self, optimizer_type: str) -> Dict:
        """Get optimizer information.

        Args:
            optimizer_type: Type of optimizer

        Returns:
            Dictionary with optimizer information
        """
        if optimizer_type not in self.optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        optimizer_class = self.optimizers[optimizer_type]

        return {
            "name": optimizer_type,
            "docstring": optimizer_class.__doc__,
            "parameters": inspect.signature(optimizer_class.__init__).parameters,
            "methods": [name for name, _ in inspect.getmembers(optimizer_class, predicate=inspect.isfunction)],
        }

    def get_optimizer_help(self, optimizer_type: str) -> str:
        """Get optimizer help text.

        Args:
            optimizer_type: Type of optimizer

        Returns:
            Help text
        """
        if optimizer_type not in self.optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        optimizer_class = self.optimizers[optimizer_type]

        help_text = f"Optimizer: {optimizer_type}\n\n"
        help_text += f"Description:\n{optimizer_class.__doc__}\n\n"

        # Add parameter information
        help_text += "Parameters:\n"
        for name, param in inspect.signature(optimizer_class.__init__).parameters.items():
            if name != "self":
                help_text += f"- {name}: {param.annotation}\n"
                if param.default != inspect.Parameter.empty:
                    help_text += f"  Default: {param.default}\n"

        # Add method information
        help_text += "\nMethods:\n"
        for name, method in inspect.getmembers(optimizer_class, predicate=inspect.isfunction):
            if not name.startswith("_"):
                help_text += f"- {name}: {method.__doc__}\n"

        return help_text
