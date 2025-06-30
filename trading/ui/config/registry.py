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
    required_data: List[str]
    output_type: str
    risk_level: str  # e.g., "low", "medium", "high"
    timeframes: List[str]
    asset_classes: List[str]
    confidence_available: bool = False
    benchmark_support: bool = False

class Registry:
    """Central registry for all UI components."""
    
    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._strategies: Dict[str, StrategyConfig] = {}
        self._load_configurations()
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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
                    # Handle missing fields with defaults
                    config_with_defaults = {
                        "required_models": [],
                        "risk_level": "medium",
                        "timeframes": ["1d"],
                        "asset_classes": ["equity"],
                        "confidence_available": False,
                        "benchmark_support": False,
                        **config
                    }
                    self._strategies[config["name"]] = StrategyConfig(**config_with_defaults)
            logger.info(f"Loaded {len(self._strategies)} strategy configurations")
        except Exception as e:
            logger.error(f"Failed to load strategy configurations: {e}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def get_available_models(self, category: Optional[str] = None) -> List[ModelConfig]:
        """Get list of available models, optionally filtered by category."""
        if category:
            return {'success': True, 'result': [model for model in self._models.values(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                   if model.category == category]
        return list(self._models.values())
    
    def get_available_strategies(self, category: Optional[str] = None) -> List[StrategyConfig]:
        """Get list of available strategies, optionally filtered by category."""
        if category:
            return {'success': True, 'result': [strategy for strategy in self._strategies.values(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                   if strategy.category == category]
        return list(self._strategies.values())
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return {'success': True, 'result': self._models.get(model_name), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy."""
        return {'success': True, 'result': self._strategies.get(strategy_name), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_model_parameters(self, model_name: str) -> Dict[str, Union[str, float, int, bool]]:
        """Get parameters for a specific model."""
        model = self.get_model_config(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")
        return {'success': True, 'result': model.parameters, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Union[str, float, int, bool]]:
        """Get parameters for a specific strategy."""
        strategy = self.get_strategy_config(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_name}")
        return {'success': True, 'result': strategy.parameters, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

# Create singleton instance
registry = Registry() 