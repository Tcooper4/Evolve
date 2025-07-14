"""
Model Registry

Centralized model management system that provides:
- Model registration and discovery
- Performance tracking and history
- Intelligent model selection
- Model metadata management
- Version control and rollback
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"
    DEPRECATED = "deprecated"


class TaskType(Enum):
    """Task type enumeration."""
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TIME_SERIES = "time_series"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    model_class: str
    task_type: TaskType
    version: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    model_name: str
    task_type: TaskType
    timestamp: datetime
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    total_return: Optional[float] = None
    volatility: Optional[float] = None
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class ModelRegistry:
    """
    Centralized model registry for managing models, their metadata,
    and performance history.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the model registry."""
        self.registry_path = registry_path or "data/model_registry"
        self.models: Dict[str, Type] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.model_instances: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.ModelRegistry")
        
        # Create registry directory
        Path(self.registry_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self._load_registry()
        
        self.logger.info("ModelRegistry initialized successfully")
    
    def _load_registry(self):
        """Load existing registry from disk."""
        try:
            # Load metadata
            metadata_path = Path(self.registry_path) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                    for name, meta_dict in metadata_dict.items():
                        meta_dict['created_at'] = datetime.fromisoformat(meta_dict['created_at'])
                        meta_dict['updated_at'] = datetime.fromisoformat(meta_dict['updated_at'])
                        meta_dict['task_type'] = TaskType(meta_dict['task_type'])
                        self.model_metadata[name] = ModelMetadata(**meta_dict)
            
            # Load performance history
            performance_path = Path(self.registry_path) / "performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    performance_dict = json.load(f)
                    for model_name, metrics_list in performance_dict.items():
                        self.performance_history[model_name] = []
                        for metrics_dict in metrics_list:
                            metrics_dict['timestamp'] = datetime.fromisoformat(metrics_dict['timestamp'])
                            metrics_dict['task_type'] = TaskType(metrics_dict['task_type'])
                            self.performance_history[model_name].append(PerformanceMetrics(**metrics_dict))
            
            self.logger.info(f"Loaded {len(self.model_metadata)} models from registry")
            
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk."""
        try:
            # Save metadata
            metadata_path = Path(self.registry_path) / "metadata.json"
            metadata_dict = {name: meta.to_dict() for name, meta in self.model_metadata.items()}
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Save performance history
            performance_path = Path(self.registry_path) / "performance.json"
            performance_dict = {
                name: [metrics.to_dict() for metrics in metrics_list]
                for name, metrics_list in self.performance_history.items()
            }
            with open(performance_path, 'w') as f:
                json.dump(performance_dict, f, indent=2)
            
            self.logger.info("Registry saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def register_model(self, 
                      name: str, 
                      model_class: Type, 
                      task_type: TaskType,
                      version: str = "1.0.0",
                      description: Optional[str] = None,
                      author: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      parameters: Optional[Dict[str, Any]] = None,
                      dependencies: Optional[List[str]] = None) -> bool:
        """Register a new model in the registry."""
        try:
            # Create metadata
            metadata = ModelMetadata(
                name=name,
                model_class=model_class.__name__,
                task_type=task_type,
                version=version,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                description=description,
                author=author,
                tags=tags or [],
                parameters=parameters or {},
                dependencies=dependencies or []
            )
            
            # Register model
            self.models[name] = model_class
            self.model_metadata[name] = metadata
            self.performance_history[name] = []
            
            # Save registry
            self._save_registry()
            
            self.logger.info(f"✅ Model '{name}' registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register model '{name}': {e}")
            return False
    
    def unregister_model(self, name: str) -> bool:
        """Unregister a model from the registry."""
        try:
            if name in self.models:
                del self.models[name]
                del self.model_metadata[name]
                if name in self.performance_history:
                    del self.performance_history[name]
                if name in self.model_instances:
                    del self.model_instances[name]
                
                self._save_registry()
                self.logger.info(f"✅ Model '{name}' unregistered successfully")
                return True
            else:
                self.logger.warning(f"⚠️ Model '{name}' not found in registry")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to unregister model '{name}': {e}")
            return False
    
    def get_model(self, name: str) -> Optional[Type]:
        """Get a model class by name."""
        return self.models.get(name)
    
    def get_model_instance(self, name: str, **kwargs) -> Optional[Any]:
        """Get or create a model instance."""
        try:
            if name not in self.models:
                self.logger.error(f"Model '{name}' not found in registry")
                return None
            
            # Check if instance already exists
            if name in self.model_instances:
                return self.model_instances[name]
            
            # Create new instance
            model_class = self.models[name]
            instance = model_class(**kwargs)
            self.model_instances[name] = instance
            
            self.logger.info(f"✅ Created instance of model '{name}'")
            return instance
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create instance of model '{name}': {e}")
            return None
    
    def get_best_model(self, 
                      task_type: TaskType, 
                      performance_metric: str = "sharpe_ratio",
                      min_performance: Optional[float] = None,
                      max_age_days: Optional[int] = None) -> Optional[str]:
        """Get the best performing model for a given task type."""
        try:
            best_model = None
            best_score = float('-inf')
            
            for name, metadata in self.model_metadata.items():
                if metadata.task_type != task_type:
                    continue
                
                # Check performance history
                if name not in self.performance_history or not self.performance_history[name]:
                    continue
                
                # Get latest performance metrics
                latest_metrics = max(self.performance_history[name], key=lambda x: x.timestamp)
                
                # Check age filter
                if max_age_days:
                    age = datetime.now() - latest_metrics.timestamp
                    if age.days > max_age_days:
                        continue
                
                # Get performance score
                score = getattr(latest_metrics, performance_metric, None)
                if score is None:
                    continue
                
                # Check minimum performance
                if min_performance and score < min_performance:
                    continue
                
                # Update best model
                if score > best_score:
                    best_score = score
                    best_model = name
            
            if best_model:
                self.logger.info(f"✅ Best model for {task_type.value}: {best_model} (score: {best_score:.4f})")
            else:
                self.logger.warning(f"⚠️ No suitable model found for {task_type.value}")
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get best model: {e}")
            return None
    
    def track_performance(self, 
                         model_name: str, 
                         task_type: TaskType,
                         **metrics) -> bool:
        """Track performance metrics for a model."""
        try:
            if model_name not in self.model_metadata:
                self.logger.error(f"Model '{model_name}' not found in registry")
                return False
            
            # Create performance metrics
            performance_metrics = PerformanceMetrics(
                model_name=model_name,
                task_type=task_type,
                timestamp=datetime.now(),
                **metrics
            )
            
            # Add to history
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
            
            self.performance_history[model_name].append(performance_metrics)
            
            # Update metadata
            self.model_metadata[model_name].updated_at = datetime.now()
            
            # Save registry
            self._save_registry()
            
            self.logger.info(f"✅ Performance tracked for model '{model_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to track performance for model '{model_name}': {e}")
            return False
    
    def get_performance_history(self, 
                               model_name: str, 
                               days: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get performance history for a model."""
        try:
            if model_name not in self.performance_history:
                return []
            
            history = self.performance_history[model_name]
            
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                history = [metrics for metrics in history if metrics.timestamp >= cutoff_date]
            
            return sorted(history, key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance history for model '{model_name}': {e}")
            return []
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a model."""
        try:
            if name not in self.model_metadata:
                return None
            
            metadata = self.model_metadata[name]
            performance_history = self.get_performance_history(name)
            
            # Calculate performance statistics
            performance_stats = {}
            if performance_history:
                latest = performance_history[-1]
                performance_stats = {
                    'latest_sharpe_ratio': latest.sharpe_ratio,
                    'latest_max_drawdown': latest.max_drawdown,
                    'latest_win_rate': latest.win_rate,
                    'latest_profit_factor': latest.profit_factor,
                    'latest_total_return': latest.total_return,
                    'performance_count': len(performance_history),
                    'last_updated': latest.timestamp.isoformat()
                }
            
            return {
                'metadata': metadata.to_dict(),
                'performance_stats': performance_stats,
                'is_registered': name in self.models,
                'has_instance': name in self.model_instances
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get model info for '{name}': {e}")
            return None
    
    def list_models(self, 
                   task_type: Optional[TaskType] = None,
                   tags: Optional[List[str]] = None) -> List[str]:
        """List models with optional filtering."""
        try:
            models = list(self.model_metadata.keys())
            
            # Filter by task type
            if task_type:
                models = [name for name in models 
                         if self.model_metadata[name].task_type == task_type]
            
            # Filter by tags
            if tags:
                models = [name for name in models 
                         if any(tag in self.model_metadata[name].tags for tag in tags)]
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list models: {e}")
            return []
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a summary of the registry."""
        try:
            total_models = len(self.model_metadata)
            task_type_counts = {}
            tag_counts = {}
            
            for metadata in self.model_metadata.values():
                # Count by task type
                task_type = metadata.task_type.value
                task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
                
                # Count by tags
                for tag in metadata.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return {
                'total_models': total_models,
                'task_type_distribution': task_type_counts,
                'tag_distribution': tag_counts,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get registry summary: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the registry."""
        try:
            health_status = {
                'status': 'healthy',
                'total_models': len(self.model_metadata),
                'registry_path': self.registry_path,
                'last_check': datetime.now().isoformat(),
                'issues': []
            }
            
            # Check registry path
            if not Path(self.registry_path).exists():
                health_status['issues'].append("Registry path does not exist")
                health_status['status'] = 'degraded'
            
            # Check for models without performance data
            models_without_performance = [
                name for name in self.model_metadata.keys()
                if name not in self.performance_history or not self.performance_history[name]
            ]
            
            if models_without_performance:
                health_status['issues'].append(f"Models without performance data: {models_without_performance}")
                health_status['status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }


# Convenience functions
def get_best_forecasting_model(performance_metric: str = "sharpe_ratio") -> Optional[str]:
    """Get the best forecasting model."""
    registry = ModelRegistry()
    return registry.get_best_model(TaskType.FORECASTING, performance_metric)


def get_best_classification_model(performance_metric: str = "accuracy") -> Optional[str]:
    """Get the best classification model."""
    registry = ModelRegistry()
    return registry.get_best_model(TaskType.CLASSIFICATION, performance_metric)


def track_model_performance(model_name: str, task_type: TaskType, **metrics) -> bool:
    """Track performance for a model."""
    registry = ModelRegistry()
    return registry.track_performance(model_name, task_type, **metrics)


if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("📊 Model Registry Demo")
        print("=" * 50)
        
        registry = ModelRegistry()
        
        # Register some example models
        print("\n🔧 Registering example models...")
        
        # Example model classes (these would be actual model classes in practice)
        class ExampleLSTM:
            def __init__(self, **kwargs):
                self.name = "LSTM"
        
        class ExampleXGBoost:
            def __init__(self, **kwargs):
                self.name = "XGBoost"
        
        # Register models
        registry.register_model(
            name="lstm_forecaster",
            model_class=ExampleLSTM,
            task_type=TaskType.FORECASTING,
            description="LSTM model for time series forecasting",
            tags=["deep_learning", "time_series"]
        )
        
        registry.register_model(
            name="xgboost_classifier",
            model_class=ExampleXGBoost,
            task_type=TaskType.CLASSIFICATION,
            description="XGBoost model for classification",
            tags=["ensemble", "classification"]
        )
        
        # Track performance
        print("\n📈 Tracking performance...")
        registry.track_performance(
            "lstm_forecaster",
            TaskType.FORECASTING,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            win_rate=0.65,
            profit_factor=1.8
        )
        
        registry.track_performance(
            "xgboost_classifier",
            TaskType.CLASSIFICATION,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85
        )
        
        # Get best models
        print("\n🏆 Getting best models...")
        best_forecaster = registry.get_best_model(TaskType.FORECASTING)
        best_classifier = registry.get_best_model(TaskType.CLASSIFICATION)
        
        print(f"Best forecaster: {best_forecaster}")
        print(f"Best classifier: {best_classifier}")
        
        # Get registry summary
        print("\n📋 Registry summary...")
        summary = registry.get_registry_summary()
        print(f"Total models: {summary['total_models']}")
        print(f"Task type distribution: {summary['task_type_distribution']}")
        
        print("\n✅ Demo completed!")
    
    asyncio.run(demo()) 