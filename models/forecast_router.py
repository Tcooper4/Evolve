"""Forecast router for managing and selecting forecasting models.

This module provides a router for selecting and managing different forecasting models
based on data characteristics, performance history, and user preferences. It supports
multiple model types including statistical (ARIMA), deep learning (LSTM), and
tree-based (XGBoost) models.

The router implements a dynamic model selection strategy that considers:
1. Data characteristics (time series length, seasonality, trend)
2. Historical model performance
3. Computational requirements
4. User preferences and constraints

Example:
    ```python
    router = ForecastRouter()
    forecast = router.get_forecast(
        data=time_series_data,
        horizon=30,
        model_type='auto'  # Let router select best model
    )
    ```
"""

import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from trading.models.arima_model import ARIMAModel
from trading.models.lstm_model import LSTMModel
from trading.models.xgboost_model import XGBoostModel
# Make Prophet import optional
try:
    from trading.models.prophet_model import ProphetModel
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    ProphetModel = None
from trading.models.autoformer_model import AutoformerModel
from trading.utils.data_utils import prepare_forecast_data
from trading.utils.logging import setup_logging

logger = setup_logging(__name__)

class ForecastRouter:
    """Router for managing and selecting forecasting models.
    
    This class implements a dynamic model selection strategy based on data
    characteristics and historical performance. It supports multiple model types
    and provides a unified interface for forecasting.
    
    Attributes:
        model_registry (Dict[str, Any]): Registry of available models
        performance_history (pd.DataFrame): Historical performance metrics
        model_weights (Dict[str, float]): Weights for model selection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the forecast router.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.model_registry = {
            'arima': ARIMAModel,
            'lstm': LSTMModel,
            'xgboost': XGBoostModel,
            'autoformer': AutoformerModel
        }
        
        # Add Prophet only if available
        if PROPHET_AVAILABLE:
            self.model_registry['prophet'] = ProphetModel
            
        self.performance_history = pd.DataFrame()
        self.model_weights = self._initialize_weights()
        
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize model selection weights.
        
        Returns:
            Dictionary of model weights
        """
        return {model: 1.0 for model in self.model_registry.keys()}
        
    def _analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series data characteristics.
        
        Args:
            data: Time series data to analyze
            
        Returns:
            Dictionary of data characteristics
        """
        characteristics = {
            'length': len(data),
            'has_seasonality': self._check_seasonality(data),
            'has_trend': self._check_trend(data),
            'volatility': data.std().mean(),
            'missing_values': data.isnull().sum().sum()
        }
        return characteristics
        
    def _check_seasonality(self, data: pd.DataFrame) -> bool:
        """Check for seasonality in time series.
        
        Args:
            data: Time series data
            
        Returns:
            True if seasonality is detected
        """
        # Implement seasonality detection
        return False
        
    def _check_trend(self, data: pd.DataFrame) -> bool:
        """Check for trend in time series.
        
        Args:
            data: Time series data
            
        Returns:
            True if trend is detected
        """
        # Implement trend detection
        return False
        
    def _select_model(self, 
                     data: pd.DataFrame,
                     model_type: Optional[str] = None) -> str:
        """Select appropriate model based on data and preferences.
        
        Args:
            data: Time series data
            model_type: Optional preferred model type
            
        Returns:
            Selected model type
        """
        if model_type and model_type in self.model_registry:
            return model_type
            
        # Analyze data characteristics
        characteristics = self._analyze_data(data)
        
        # Apply selection rules
        if characteristics['length'] < 100:
            return 'arima'  # Better for short series
        elif characteristics['has_seasonality'] and PROPHET_AVAILABLE:
            return 'prophet'  # Better for seasonal data
        elif characteristics['has_trend']:
            return 'autoformer'  # Better for trending data
        else:
            return 'lstm'  # Default to LSTM
            
    def get_forecast(self,
                    data: pd.DataFrame,
                    horizon: int = 30,
                    model_type: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
        """Get forecast for time series data.
        
        Args:
            data: Time series data
            horizon: Forecast horizon in periods
            model_type: Optional preferred model type
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing forecast results
            
        Raises:
            ValueError: If data is invalid or model type is not supported
        """
        try:
            # Prepare data
            prepared_data = prepare_forecast_data(data)
            
            # Select model
            selected_model = self._select_model(prepared_data, model_type)
            logger.info(f"Selected model: {selected_model}")
            
            # Initialize and train model
            model_class = self.model_registry[selected_model]
            model = model_class(**kwargs)
            model.fit(prepared_data)
            
            # Generate forecast
            forecast = model.predict(horizon=horizon)
            
            # Log performance
            self._log_performance(selected_model, forecast, data)
            
            return {
                'model': selected_model,
                'forecast': forecast,
                'confidence': model.get_confidence(),
                'metadata': model.get_metadata()
            }
            
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            raise
            
    def _log_performance(self,
                        model_type: str,
                        forecast: pd.DataFrame,
                        actual: pd.DataFrame) -> None:
        """Log model performance metrics.
        
        Args:
            model_type: Type of model used
            forecast: Forecast results
            actual: Actual values
        """
        # Calculate metrics
        metrics = {
            'model': model_type,
            'timestamp': datetime.now().isoformat(),
            'mse': np.mean((forecast - actual) ** 2),
            'mae': np.mean(np.abs(forecast - actual)),
            'mape': np.mean(np.abs((forecast - actual) / actual)) * 100
        }
        
        # Update performance history
        self.performance_history = pd.concat([
            self.performance_history,
            pd.DataFrame([metrics])
        ])
        
        # Update model weights
        self._update_weights()
        
    def _update_weights(self) -> None:
        """Update model selection weights based on performance."""
        if self.performance_history.empty:
            return
            
        # Calculate recent performance
        recent = self.performance_history.tail(100)
        performance = recent.groupby('model')['mse'].mean()
        
        # Update weights (lower MSE = higher weight)
        total = performance.sum()
        if total > 0:
            self.model_weights = {
                model: 1 - (mse / total)
                for model, mse in performance.items()
            }
            
    def get_available_models(self) -> List[str]:
        """Get list of available model types.
        
        Returns:
            List of model type names
        """
        return list(self.model_registry.keys())
        
    def get_model_performance(self,
                            model_type: Optional[str] = None) -> pd.DataFrame:
        """Get performance history for models.
        
        Args:
            model_type: Optional model type to filter
            
        Returns:
            DataFrame with performance history
        """
        if model_type:
            return self.performance_history[
                self.performance_history['model'] == model_type
            ]
        return self.performance_history


# Example usage:
if __name__ == "__main__":
    router = ForecastRouter()
    
    # Example data
    sample_data = {"close": [100, 101, 102, 103, 104]}
    
    # Get forecast
    result = router.get_forecast(
        data=pd.DataFrame(sample_data),
        horizon=30,
        model_type='auto'  # Let router select best model
    )
    print(f"Forecast result: {result}") 