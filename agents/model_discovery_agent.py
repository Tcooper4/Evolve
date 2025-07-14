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
                    'model_types': [m['type'] for m in validated_models]
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
                {'weights': [0.4, 0.3, 0.3], 'methods': ['lstm', 'xgboost', 'prophet']},
                {'weights': [0.5, 0.3, 0.2], 'methods': ['lstm', 'arima', 'xgboost']},
                {'weights': [0.3, 0.4, 0.3], 'methods': ['prophet', 'lstm', 'arima']}
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
            try:
                # Basic validation
                if self._validate_model_config(model):
                    model['status'] = 'validated'
                    model['validation_score'] = self._calculate_validation_score(model)
                    validated_models.append(model)
                else:
                    model['status'] = 'invalid'
                    
            except Exception as e:
                self.logger.warning(f"Model validation failed for {model['id']}: {e}")
                model['status'] = 'validation_failed'
                model['error'] = str(e)
        
        return validated_models
    
    def _validate_model_config(self, model: Dict[str, Any]) -> bool:
        """Validate model configuration."""
        model_type = model['type']
        config = model['config']
        
        # Basic validation rules
        if model_type == 'lstm':
            return ('layers' in config and 
                   isinstance(config['layers'], list) and 
                   len(config['layers']) > 0)
        elif model_type == 'xgboost':
            return ('n_estimators' in config and 
                   config['n_estimators'] > 0)
        elif model_type == 'prophet':
            return ('changepoint_prior_scale' in config and 
                   config['changepoint_prior_scale'] > 0)
        elif model_type == 'arima':
            return ('order' in config and 
                   isinstance(config['order'], tuple) and 
                   len(config['order']) == 3)
        elif model_type == 'ensemble':
            return ('weights' in config and 
                   'methods' in config and 
                   len(config['weights']) == len(config['methods']))
        
        return False
    
    def _calculate_validation_score(self, model: Dict[str, Any]) -> float:
        """Calculate validation score for a model."""
        # Simple scoring based on model type and config
        base_score = 0.7
        
        if model['type'] == 'lstm':
            layers = model['config']['layers']
            base_score += min(len(layers) * 0.1, 0.2)
        elif model['type'] == 'xgboost':
            n_estimators = model['config']['n_estimators']
            base_score += min(n_estimators / 1000, 0.2)
        elif model['type'] == 'ensemble':
            base_score += 0.1  # Ensembles get bonus
        
        return min(base_score, 1.0)
    
    async def _register_models(self, models: List[Dict[str, Any]]) -> int:
        """Register validated models in the registry."""
        registered_count = 0
        
        for model in models:
            if model['status'] == 'validated':
                self.model_registry[model['id']] = {
                    'type': model['type'],
                    'config': model['config'],
                    'validation_score': model['validation_score'],
                    'registration_time': datetime.now().isoformat(),
                    'status': 'active'
                }
                registered_count += 1
        
        if registered_count > 0:
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
            'success_rate': len(validated_models) / max(len(discovered_models), 1)
        }
        
        self.generation_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.generation_history) > 100:
            self.generation_history = self.generation_history[-100:]
        
        self._save_generation_history()
    
    def get_model_registry(self) -> Dict[str, Any]:
        """Get current model registry."""
        return self.model_registry
    
    def get_generation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent generation history."""
        return self.generation_history[-limit:] if self.generation_history else []
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for models."""
        if not self.model_registry:
            return {}
        
        type_counts = {}
        avg_scores = {}
        
        for model in self.model_registry.values():
            model_type = model['type']
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
            
            if model_type not in avg_scores:
                avg_scores[model_type] = []
            avg_scores[model_type].append(model.get('validation_score', 0))
        
        # Calculate averages
        for model_type in avg_scores:
            avg_scores[model_type] = np.mean(avg_scores[model_type])
        
        return {
            'total_models': len(self.model_registry),
            'type_distribution': type_counts,
            'average_scores': avg_scores
        } 