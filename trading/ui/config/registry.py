"""Centralized registry for UI components.

This module provides dynamic access to available strategies, models, and their configurations.
It serves as a single source of truth for UI components and enables agentic interactions.
"""

from typing import Dict, List, Optional, TypedDict, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""
    name: str
    description: str
    category: str  # e.g., "forecasting", "classification"
    parameters: Dict[str, Union[str, float, int, bool]]
    required_data: List[str]
    output_type: str  # e.g., "regression", "classification"
    confidence_available: bool = False
    benchmark_support: bool = False

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    description: str
    category: str  # e.g., "technical", "fundamental", "hybrid"
    parameters: Dict[str, Union[str, float, int, bool]]
    required_models: List[str]
    risk_level: str  # e.g., "low", "medium", "high"
    timeframes: List[str]
    asset_classes: List[str]

class Registry:
    """Central registry for all UI components."""
    
    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._strategies: Dict[str, StrategyConfig] = {}
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Load model and strategy configurations from JSON files."""
        config_dir = Path(__file__).parent / "configs"
        
        # Load model configurations
        try:
            with open(config_dir / "models.json") as f:
                model_configs = json.load(f)
                for config in model_configs:
                    self._models[config["name"]] = ModelConfig(**config)
            logger.info(f"Loaded {len(self._models)} model configurations")
        except Exception as e:
            logger.error(f"Failed to load model configurations: {e}")
            raise
        
        # Load strategy configurations
        try:
            with open(config_dir / "strategies.json") as f:
                strategy_configs = json.load(f)
                for config in strategy_configs:
                    self._strategies[config["name"]] = StrategyConfig(**config)
            logger.info(f"Loaded {len(self._strategies)} strategy configurations")
        except Exception as e:
            logger.error(f"Failed to load strategy configurations: {e}")
            raise
    
    def get_available_models(self, category: Optional[str] = None) -> List[ModelConfig]:
        """Get list of available models, optionally filtered by category."""
        if category:
            return [model for model in self._models.values() 
                   if model.category == category]
        return list(self._models.values())
    
    def get_available_strategies(self, category: Optional[str] = None) -> List[StrategyConfig]:
        """Get list of available strategies, optionally filtered by category."""
        if category:
            return [strategy for strategy in self._strategies.values() 
                   if strategy.category == category]
        return list(self._strategies.values())
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self._models.get(model_name)
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy."""
        return self._strategies.get(strategy_name)
    
    def get_model_parameters(self, model_name: str) -> Dict[str, Union[str, float, int, bool]]:
        """Get parameters for a specific model."""
        model = self.get_model_config(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")
        return model.parameters
    
    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Union[str, float, int, bool]]:
        """Get parameters for a specific strategy."""
        strategy = self.get_strategy_config(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_name}")
        return strategy.parameters

# Create singleton instance
registry = Registry() 