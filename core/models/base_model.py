"""
Base Model Class

Provides standardized save/load functionality for all forecasting models.
Ensures consistent model persistence across the Evolve system.
"""

import joblib
import os
import logging
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
import json
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all forecasting models with standardized save/load functionality."""
    
    def __init__(self, model_name: str = "base_model", config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model for logging and identification
            config: Optional configuration dictionary
        """
        self.model_name: str = model_name
        self.fitted: bool = False
        self.model_path: Optional[str] = None
        self.config: Dict[str, Any] = config or {}
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'model_type': self.__class__.__name__,
            'version': '1.0',
            'model_name': model_name
        }
        self.training_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.__class__.__name__}: {model_name}")
    
    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs) -> 'BaseModel':
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features to predict on
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
        """
        pass
    
    def save_model(self, path: str, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Save model to disk with safety checks.
        
        Args:
            path: Path where to save the model
            include_metadata: Whether to include model metadata
            
        Returns:
            Dictionary with save status
        """
        try:
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Update metadata
            self.metadata['saved_at'] = datetime.now().isoformat()
            self.metadata['model_path'] = path
            
            # Prepare model data
            model_data = {
                'model': self,
                'config': self.config,
                'fitted': self.fitted,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics
            }
            
            if include_metadata:
                model_data['metadata'] = self.metadata
            
            # Save using joblib
            joblib.dump(model_data, path)
            
            self.model_path = path
            logger.info(f"Model saved successfully to {path}")
            
            return {
                'success': True,
                'message': f'Model saved to {path}',
                'path': path,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @classmethod
    def load_model(cls, path: str) -> 'BaseModel':
        """
        Load model from disk with error handling.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted
        """
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            
            # Load model data
            model_data = joblib.load(path)
            
            if isinstance(model_data, dict):
                # New format with metadata
                model = model_data['model']
                if 'config' in model_data:
                    model.config = model_data['config']
                if 'metadata' in model_data:
                    model.metadata.update(model_data['metadata'])
                if 'training_history' in model_data:
                    model.training_history = model_data['training_history']
                if 'performance_metrics' in model_data:
                    model.performance_metrics = model_data['performance_metrics']
                model.model_path = path
            else:
                # Legacy format - direct model object
                model = model_data
                model.model_path = path
            
            logger.info(f"Model loaded successfully from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Failed to load model from {path}: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'fitted': self.fitted,
            'model_path': self.model_path,
            'config': self.config,
            'metadata': self.metadata,
            'training_history_count': len(self.training_history),
            'performance_metrics': self.performance_metrics
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Base validation - can be overridden by subclasses
            if not isinstance(config, dict):
                return {
                    'valid': False,
                    'error': 'Configuration must be a dictionary'
                }
            
            return {
                'valid': True,
                'message': 'Configuration is valid'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Configuration validation error: {str(e)}'
            }
    
    def set_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set model configuration with validation.
        
        Args:
            config: Configuration to set
            
        Returns:
            Dictionary with set status
        """
        try:
            # Validate configuration
            validation = self.validate_config(config)
            if not validation['valid']:
                return validation
            
            # Update configuration
            self.config.update(config)
            self.metadata['config_updated_at'] = datetime.now().isoformat()
            
            logger.info(f"Configuration updated for {self.model_name}")
            
            return {
                'success': True,
                'message': 'Configuration updated successfully',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error setting configuration: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def export_config(self, path: str) -> Dict[str, Any]:
        """
        Export model configuration to JSON file.
        
        Args:
            path: Path to save configuration
            
        Returns:
            Dictionary with export status
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'config': self.config,
                'metadata': self.metadata,
                'exported_at': datetime.now().isoformat()
            }
            
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration exported to {path}")
            
            return {
                'success': True,
                'message': f'Configuration exported to {path}',
                'path': path,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def import_config(self, path: str) -> Dict[str, Any]:
        """
        Import model configuration from JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Dictionary with import status
        """
        try:
            if not Path(path).exists():
                return {
                    'success': False,
                    'error': f'Configuration file not found: {path}'
                }
            
            with open(path, 'r') as f:
                config_data = json.load(f)
            
            if 'config' in config_data:
                return self.set_config(config_data['config'])
            else:
                return self.set_config(config_data)
                
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def add_training_record(self, record: Dict[str, Any]) -> None:
        """
        Add a training record to the model's history.
        
        Args:
            record: Training record to add
        """
        record['timestamp'] = datetime.now().isoformat()
        self.training_history.append(record)
        logger.debug(f"Added training record for {self.model_name}")
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update the model's performance metrics.
        
        Args:
            metrics: Performance metrics to update
        """
        self.performance_metrics.update(metrics)
        self.performance_metrics['last_updated'] = datetime.now().isoformat()
        logger.info(f"Updated performance metrics for {self.model_name}")
    
    def get_training_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get training history records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of training history records
        """
        if limit is None:
            return self.training_history
        return self.training_history[-limit:]
    
    def clear_training_history(self) -> None:
        """Clear the training history."""
        self.training_history.clear()
        logger.info(f"Cleared training history for {self.model_name}")
    
    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.
        
        Returns:
            True if the model is fitted
        """
        return self.fitted
    
    def get_model_size(self) -> Optional[int]:
        """
        Get the size of the model in bytes.
        
        Returns:
            Model size in bytes or None if not available
        """
        if self.model_path and Path(self.model_path).exists():
            return Path(self.model_path).stat().st_size
        return None
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name='{self.model_name}', fitted={self.fitted})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(name='{self.model_name}', config={self.config}, fitted={self.fitted})" 