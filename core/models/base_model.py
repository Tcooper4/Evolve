"""
Base Model Class

Provides standardized save/load functionality for all forecasting models.
Ensures consistent model persistence across the Evolve system.
"""

import joblib
import os
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all forecasting models with standardized save/load functionality."""
    
    def __init__(self, model_name: str = "base_model"):
        """Initialize base model.
        
        Args:
            model_name: Name of the model for logging and identification
        """
        self.model_name = model_name
        self.fitted = False
        self.model_path = None
        self.config = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': self.__class__.__name__,
            'version': '1.0'
        }
    
    def save_model(self, path: str, include_metadata: bool = True) -> Dict[str, Any]:
        """Save model to disk with safety checks.
        
        Args:
            path: Path where to save the model
            include_metadata: Whether to include model metadata
            
        Returns:
            Dictionary with save status
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Update metadata
            self.metadata['saved_at'] = datetime.now().isoformat()
            self.metadata['model_path'] = path
            
            # Prepare model data
            model_data = {
                'model': self,
                'config': self.config,
                'fitted': self.fitted
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
        """Load model from disk with error handling.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted
        """
        try:
            if not os.path.exists(path):
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
        """Get model information and metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'fitted': self.fitted,
            'model_path': self.model_path,
            'config': self.config,
            'metadata': self.metadata
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration.
        
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
        """Set model configuration with validation.
        
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
        """Export model configuration to JSON file.
        
        Args:
            path: Path to save configuration
            
        Returns:
            Dictionary with export status
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
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
        """Import model configuration from JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Dictionary with import status
        """
        try:
            if not os.path.exists(path):
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