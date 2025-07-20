"""
Model Discovery Agent

This module provides dynamic model discovery capabilities for the autonomous orchestrator.
It handles model generation, validation, and history tracking.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from trading.agents.base_agent_interface import BaseAgent, AgentResult, AgentConfig


class ModelDiscoveryAgent(BaseAgent):
    """Agent responsible for discovering and validating new models."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or AgentConfig())
        self.logger = logging.getLogger(__name__)
        
        # Model discovery settings
        self.model_types = ['lstm', 'xgboost', 'prophet', 'arima', 'ensemble']
        self.validation_threshold = 0.6
        self.max_models_per_type = 3
        
        # Pre-defined fallback models
        self.predefined_models = [
            {
                'id': 'lstm_fallback_1',
                'type': 'lstm',
                'config': {'layers': [50, 25], 'dropout': 0.2, 'epochs': 100},
                'discovery_time': datetime.now().isoformat(),
                'status': 'predefined'
            },
            {
                'id': 'xgboost_fallback_1',
                'type': 'xgboost',
                'config': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
                'discovery_time': datetime.now().isoformat(),
                'status': 'predefined'
            },
            {
                'id': 'prophet_fallback_1',
                'type': 'prophet',
                'config': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10},
                'discovery_time': datetime.now().isoformat(),
                'status': 'predefined'
            }
        ]
        
        # History tracking
        self.history_file = Path("logs/generation_history.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.generation_history = self._load_generation_history()
        
        # Model registry
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self._load_model_registry()
    
    def _load_generation_history(self) -> List[Dict[str, Any]]:
        """Load model generation history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load generation history: {e}")
        return []
    
    def _save_generation_history(self) -> None:
        """Save model generation history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.generation_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save generation history: {e}")
    
    def _load_model_registry(self) -> None:
        """Load existing model registry."""
        registry_file = Path("data/model_registry.json")
        try:
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    self.model_registry = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load model registry: {e}")
    
    def _save_model_registry(self) -> None:
        """Save model registry to file."""
        registry_file = Path("data/model_registry.json")
        try:
            registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
    
    def _get_predefined_models(self) -> List[Dict[str, Any]]:
        """
        Get pre-defined fallback models.
        
        Returns:
            List of pre-defined model configurations
        """
        self.logger.info("Using pre-defined fallback models")
        return self.predefined_models.copy()
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute model discovery process."""
        try:
            # Extract parameters
            target_model_type = kwargs.get('model_type')
            force_regeneration = kwargs.get('force_regeneration', False)
            validation_data = kwargs.get('validation_data')
            
            if target_model_type and target_model_type not in self.model_types:
                return AgentResult(
                    success=False,
                    error_message=f"Invalid model type: {target_model_type}"
                )
            
            # Discover new models
            discovered_models = await self._discover_models(
                target_type=target_model_type,
                force_regeneration=force_regeneration,
                validation_data=validation_data
            )
            
            # Add fallback to pre-defined model list if none discovered
            if not discovered_models:
                self.logger.warning("No models discovered, falling back to pre-defined model list")
                discovered_models = self._get_predefined_models()
            
            # Validate discovered models
            validated_models = await self._validate_models(
                discovered_models, validation_data
            )
            
            # Register successful models
            registered_count = await self._register_models(validated_models)
            
            # Update history
            self._update_generation_history(discovered_models, validated_models)
            
            return AgentResult(
                success=True,
                data={
                    'discovered_models': len(discovered_models),
                    'validated_models': len(validated_models),
                    'registered_models': registered_count,
                    'model_types': [m['type'] for m in validated_models],
                    'used_fallback': len(discovered_models) > 0 and discovered_models[0].get('status') == 'predefined'
                },
                extra_metrics={
                    'discovery_rate': len(validated_models) / max(len(discovered_models), 1),
                    'total_models': len(self.model_registry)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Model discovery failed: {e}")
            return AgentResult(
                success=False,
                error_message=str(e)
            )
    
    async def _discover_models(
        self, 
        target_type: Optional[str] = None,
        force_regeneration: bool = False,
        validation_data: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Discover new models based on current performance and data."""
        discovered_models = []
        
        model_types = [target_type] if target_type else self.model_types
        
        for model_type in model_types:
            # Check if we need more models of this type
            existing_count = len([m for m in self.model_registry.values() 
                                if m.get('type') == model_type])
            
            if existing_count >= self.max_models_per_type and not force_regeneration:
                continue
            
            # Generate model configurations
            configs = self._generate_model_configs(model_type, validation_data)
            
            for config in configs:
                model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(discovered_models)}"
                
                discovered_models.append({
                    'id': model_id,
                    'type': model_type,
                    'config': config,
                    'discovery_time': datetime.now().isoformat(),
                    'status': 'discovered'
                })
        
        return discovered_models
    
    def _generate_model_configs(
        self, 
        model_type: str, 
        validation_data: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Generate model configurations based on type and data."""
        configs = []
        
        if model_type == 'lstm':
            configs = [
                {'layers': [50, 25], 'dropout': 0.2, 'epochs': 100},
                {'layers': [100, 50, 25], 'dropout': 0.3, 'epochs': 150},
                {'layers': [75, 35], 'dropout': 0.1, 'epochs': 80}
            ]
        elif model_type == 'xgboost':
            configs = [
                {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
                {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
                {'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.08}
            ]
        elif model_type == 'prophet':
            configs = [
                {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10},
                {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 5},
                {'changepoint_prior_scale': 0.02, 'seasonality_prior_scale': 15}
            ]
        elif model_type == 'arima':
            configs = [
                {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
                {'order': (2, 1, 2), 'seasonal_order': (0, 1, 1, 12)},
                {'order': (1, 1, 0), 'seasonal_order': (1, 1, 0, 12)}
            ]
        elif model_type == 'ensemble':
            configs = [
                {'models': ['lstm', 'xgboost'], 'weights': [0.6, 0.4]},
                {'models': ['prophet', 'arima'], 'weights': [0.7, 0.3]},
                {'models': ['lstm', 'prophet', 'xgboost'], 'weights': [0.4, 0.3, 0.3]}
            ]
        
        return configs
    
    async def _validate_models(
        self, 
        models: List[Dict[str, Any]], 
        validation_data: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Validate discovered models."""
        validated_models = []
        
        for model in models:
            if self._validate_model_config(model):
                # Calculate validation score
                score = self._calculate_validation_score(model)
                model['validation_score'] = score
                
                if score >= self.validation_threshold:
                    model['status'] = 'validated'
                    validated_models.append(model)
                else:
                    self.logger.warning(f"Model {model['id']} failed validation: score {score:.3f} < {self.validation_threshold}")
            else:
                self.logger.warning(f"Model {model['id']} has invalid configuration")
        
        return validated_models
    
    def _validate_model_config(self, model: Dict[str, Any]) -> bool:
        """Validate model configuration."""
        try:
            config = model.get('config', {})
            model_type = model.get('type')
            
            if not model_type or model_type not in self.model_types:
                return False
            
            # Type-specific validation
            if model_type == 'lstm':
                return 'layers' in config and 'dropout' in config
            elif model_type == 'xgboost':
                return 'n_estimators' in config and 'max_depth' in config
            elif model_type == 'prophet':
                return 'changepoint_prior_scale' in config
            elif model_type == 'arima':
                return 'order' in config
            elif model_type == 'ensemble':
                return 'models' in config and 'weights' in config
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating model config: {e}")
            return False
    
    def _calculate_validation_score(self, model: Dict[str, Any]) -> float:
        """Calculate validation score for a model."""
        try:
            # Simulate validation score calculation
            # In practice, this would involve actual model training and validation
            base_score = 0.7  # Base score for valid configurations
            
            # Add small random variation
            import random
            variation = random.uniform(-0.1, 0.1)
            
            return min(1.0, max(0.0, base_score + variation))
            
        except Exception as e:
            self.logger.error(f"Error calculating validation score: {e}")
            return 0.0
    
    async def _register_models(self, models: List[Dict[str, Any]]) -> int:
        """Register validated models in the registry."""
        registered_count = 0
        
        for model in models:
            try:
                model_id = model['id']
                self.model_registry[model_id] = {
                    'type': model['type'],
                    'config': model['config'],
                    'validation_score': model.get('validation_score', 0.0),
                    'registration_time': datetime.now().isoformat(),
                    'status': model['status']
                }
                registered_count += 1
                self.logger.info(f"Registered model: {model_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to register model {model.get('id', 'unknown')}: {e}")
        
        # Save registry
        self._save_model_registry()
        
        return registered_count
    
    def _update_generation_history(
        self, 
        discovered_models: List[Dict[str, Any]], 
        validated_models: List[Dict[str, Any]]
    ) -> None:
        """Update generation history."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'discovered_count': len(discovered_models),
            'validated_count': len(validated_models),
            'model_types': list(set(m['type'] for m in discovered_models)),
            'used_fallback': len(discovered_models) > 0 and discovered_models[0].get('status') == 'predefined'
        }
        
        self.generation_history.append(history_entry)
        
        # Keep only recent history
        if len(self.generation_history) > 100:
            self.generation_history = self.generation_history[-100:]
        
        self._save_generation_history()
    
    def get_model_registry(self) -> Dict[str, Any]:
        """Get current model registry."""
        return self.model_registry.copy()
    
    def get_generation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent generation history."""
        return self.generation_history[-limit:]
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        if not self.model_registry:
            return {}
        
        stats = {
            'total_models': len(self.model_registry),
            'by_type': {},
            'avg_validation_score': 0.0,
            'fallback_usage_count': 0
        }
        
        total_score = 0.0
        for model_id, model_data in self.model_registry.items():
            model_type = model_data.get('type', 'unknown')
            if model_type not in stats['by_type']:
                stats['by_type'][model_type] = 0
            stats['by_type'][model_type] += 1
            
            total_score += model_data.get('validation_score', 0.0)
        
        if self.model_registry:
            stats['avg_validation_score'] = total_score / len(self.model_registry)
        
        # Count fallback usage from history
        for entry in self.generation_history:
            if entry.get('used_fallback', False):
                stats['fallback_usage_count'] += 1
        
        return stats


def get_model_discovery_agent() -> ModelDiscoveryAgent:
    """Get singleton instance of ModelDiscoveryAgent."""
    if not hasattr(get_model_discovery_agent, "_instance"):
        get_model_discovery_agent._instance = ModelDiscoveryAgent()
    return get_model_discovery_agent._instance
