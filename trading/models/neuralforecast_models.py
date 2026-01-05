"""
NeuralForecast Models Wrapper

Provides wrappers for state-of-the-art time series models from neuralforecast:
- Autoformer: Decomposition-based transformer
- Informer: Efficient transformer for long sequences  
- TFT: Temporal Fusion Transformer (multi-horizon)
- N-BEATS: Neural basis expansion (interpretable)
- PatchTST: Latest SOTA patch-based transformer
- N-HiTS: Hierarchical interpolation (fast)
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import neuralforecast
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import (
        Autoformer,
        Informer,
        TFT,
        NBEATS,
        PatchTST,
        NHITS
    )
    NEURALFORECAST_AVAILABLE = True
    logger.info("✅ NeuralForecast available")
except ImportError as e:
    logger.warning(f"⚠️  NeuralForecast not available: {e}")
    NEURALFORECAST_AVAILABLE = False
    NeuralForecast = None
    Autoformer = Informer = TFT = NBEATS = PatchTST = NHITS = None


class NeuralForecastWrapper:
    """Base wrapper for NeuralForecast models."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize NeuralForecast model wrapper.
        
        Args:
            model_name: Name of the model (Autoformer, Informer, etc.)
            **kwargs: Model-specific hyperparameters
        """
        if not NEURALFORECAST_AVAILABLE:
            raise ImportError("NeuralForecast is not installed. Install with: pip install neuralforecast")
        
        self.model_name = model_name
        self.model_params = kwargs
        self.nf_model = None
        self.fitted = False
    
    def _prepare_data(self, data: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        Prepare data for NeuralForecast format.
        
        NeuralForecast expects columns: [unique_id, ds, y]
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Create NeuralForecast format
        df = pd.DataFrame({
            'unique_id': '1',  # Single time series
            'ds': data.index,
            'y': data[target_col]
        })
        
        return df
    
    def fit(self, train_data: pd.DataFrame, target_col: str = 'close', **fit_params):
        """
        Fit the model.
        
        Args:
            train_data: Training data
            target_col: Target column name
            **fit_params: Additional fitting parameters
        """
        # Prepare data
        df = self._prepare_data(train_data, target_col)
        
        # Get model class
        model_class = self._get_model_class()
        
        # Default horizon if not provided
        horizon = fit_params.get('horizon', 30)
        
        # Create model instance with params
        model = model_class(
            h=horizon,
            **self.model_params
        )
        
        # Create NeuralForecast instance
        self.nf_model = NeuralForecast(
            models=[model],
            freq='D'  # Daily frequency
        )
        
        # Fit the model
        self.nf_model.fit(df)
        self.fitted = True
        
        return self
    
    def predict(self, steps: int = 30) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Generate forecast
        forecast_df = self.nf_model.predict()
        
        # Extract predictions (column name is model name)
        predictions = forecast_df[self.model_name].values
        
        return predictions[:steps]
    
    def _get_model_class(self):
        """Get the model class based on model_name."""
        models = {
            'Autoformer': Autoformer,
            'Informer': Informer,
            'TFT': TFT,
            'NBEATS': NBEATS,
            'PatchTST': PatchTST,
            'NHITS': NHITS
        }
        
        if self.model_name not in models:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return models[self.model_name]


class AutoformerModel(NeuralForecastWrapper):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation.
    
    Best for: Long-term forecasting with trend and seasonality
    """
    
    def __init__(self, **kwargs):
        # Default parameters for Autoformer
        default_params = {
            'input_size': 30,
            'hidden_size': 128,
            'n_head': 4,
            'learning_rate': 0.001,
            'scaler_type': 'standard',
            'max_steps': 1000,
            'batch_size': 32,
        }
        default_params.update(kwargs)
        super().__init__(model_name='Autoformer', **default_params)


class InformerModel(NeuralForecastWrapper):
    """
    Informer: Efficient Transformer for long sequences.
    
    Best for: Very long-term forecasting (100+ steps)
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'input_size': 30,
            'hidden_size': 128,
            'n_head': 4,
            'learning_rate': 0.001,
            'scaler_type': 'standard',
            'max_steps': 1000,
            'batch_size': 32,
        }
        default_params.update(kwargs)
        super().__init__(model_name='Informer', **default_params)


class TFTModel(NeuralForecastWrapper):
    """
    Temporal Fusion Transformer.
    
    Best for: Multi-horizon forecasting with interpretability
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'input_size': 30,
            'hidden_size': 128,
            'learning_rate': 0.001,
            'scaler_type': 'standard',
            'max_steps': 1000,
            'batch_size': 32,
        }
        default_params.update(kwargs)
        super().__init__(model_name='TFT', **default_params)


class NBEATSModel(NeuralForecastWrapper):
    """
    N-BEATS: Neural Basis Expansion Analysis.
    
    Best for: Interpretable forecasting with trend/seasonality decomposition
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'input_size': 30,
            'hidden_size': 512,
            'learning_rate': 0.001,
            'scaler_type': 'standard',
            'max_steps': 1000,
            'batch_size': 32,
        }
        default_params.update(kwargs)
        super().__init__(model_name='NBEATS', **default_params)


class PatchTSTModel(NeuralForecastWrapper):
    """
    PatchTST: Patch-based Time Series Transformer.
    
    Best for: Latest SOTA, excellent for long-term forecasting
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'input_size': 30,
            'hidden_size': 128,
            'n_head': 4,
            'learning_rate': 0.001,
            'scaler_type': 'standard',
            'max_steps': 1000,
            'batch_size': 32,
            'patch_len': 16,
            'stride': 8,
        }
        default_params.update(kwargs)
        super().__init__(model_name='PatchTST', **default_params)


class NHITSModel(NeuralForecastWrapper):
    """
    N-HiTS: Neural Hierarchical Interpolation for Time Series.
    
    Best for: Fast training and inference, good accuracy
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'input_size': 30,
            'hidden_size': 512,
            'learning_rate': 0.001,
            'scaler_type': 'standard',
            'max_steps': 1000,
            'batch_size': 32,
        }
        default_params.update(kwargs)
        super().__init__(model_name='NHITS', **default_params)


# Convenience function
def get_neuralforecast_model(model_name: str, **kwargs):
    """
    Get a NeuralForecast model by name.
    
    Args:
        model_name: One of: Autoformer, Informer, TFT, NBEATS, PatchTST, NHITS
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    models = {
        'Autoformer': AutoformerModel,
        'Informer': InformerModel,
        'TFT': TFTModel,
        'NBEATS': NBEATSModel,
        'PatchTST': PatchTSTModel,
        'NHITS': NHITSModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    
    return models[model_name](**kwargs)

