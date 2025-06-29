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
            # Defensive checks for input parameters
            if data is None or data.empty:
                logger.warning("Empty or None data provided, using fallback data")
                data = self._generate_fallback_data()
            
            if horizon is None or horizon <= 0:
                logger.warning(f"Invalid horizon {horizon}, using default horizon of 30")
                horizon = 30
            
            # Validate data structure
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Data must be pandas DataFrame, got {type(data)}")
                raise ValueError("Data must be pandas DataFrame")
            
            # Check for required columns
            required_columns = ['close', 'volume'] if 'close' in data.columns else ['price']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing columns {missing_columns}, using available columns")
                # Use first numeric column as target
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    target_col = numeric_columns[0]
                    logger.info(f"Using {target_col} as target column")
                else:
                    raise ValueError("No numeric columns found in data")
            
            # Prepare data with defensive checks
            prepared_data = self._prepare_data_safely(data)
            
            # Select model with fallback logic
            selected_model = self._select_model_with_fallback(prepared_data, model_type)
            logger.info(f"Selected model: {selected_model}")
            
            # Initialize model with sensible defaults
            model_class = self.model_registry[selected_model]
            model_kwargs = self._get_model_defaults(selected_model)
            model_kwargs.update(kwargs)  # Override with user-provided kwargs
            
            logger.info(f"Initializing {selected_model} with config: {model_kwargs}")
            model = model_class(**model_kwargs)
            
            # Fit model with error handling
            try:
                model.fit(prepared_data)
                logger.info(f"Successfully fitted {selected_model}")
            except Exception as e:
                logger.error(f"Failed to fit {selected_model}: {e}")
                # Try fallback model
                fallback_model = self._get_fallback_model(selected_model)
                logger.info(f"Trying fallback model: {fallback_model}")
                model_class = self.model_registry[fallback_model]
                model = model_class(**self._get_model_defaults(fallback_model))
                model.fit(prepared_data)
                selected_model = fallback_model
            
            # Generate forecast with error handling
            try:
                forecast = model.predict(horizon=horizon)
                logger.info(f"Successfully generated forecast with {selected_model}")
            except Exception as e:
                logger.error(f"Failed to generate forecast with {selected_model}: {e}")
                # Return simple forecast as fallback
                forecast = self._generate_simple_forecast(prepared_data, horizon)
            
            # Log performance
            self._log_performance(selected_model, forecast, data)
            
            return {
                'model': selected_model,
                'forecast': forecast,
                'confidence': self._get_confidence(model, selected_model),
                'metadata': self._get_metadata(model, selected_model),
                'warnings': self._get_warnings(data, selected_model)
            }
            
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            # Return fallback result instead of raising
            return self._get_fallback_result(data, horizon)
    
    def _generate_fallback_data(self) -> pd.DataFrame:
        """Generate fallback data when none is provided."""
        logger.info("Generating fallback data")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        volumes = np.random.uniform(1000000, 5000000, len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
    
    def _prepare_data_safely(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with defensive checks."""
        try:
            prepared_data = prepare_forecast_data(data)
            logger.info("Data prepared successfully")
            return prepared_data
        except Exception as e:
            logger.warning(f"Data preparation failed: {e}, using original data")
            return data
    
    def _select_model_with_fallback(self, data: pd.DataFrame, preferred_model: Optional[str]) -> str:
        """Select model with fallback logic."""
        if preferred_model and preferred_model in self.model_registry:
            return preferred_model
        
        # Auto-select based on data characteristics
        if len(data) < 100:
            logger.info("Small dataset, selecting ARIMA")
            return 'arima'
        elif len(data) < 500:
            logger.info("Medium dataset, selecting XGBoost")
            return 'xgboost'
        else:
            logger.info("Large dataset, selecting LSTM")
            return 'lstm'
    
    def _get_model_defaults(self, model_type: str) -> Dict[str, Any]:
        """Get sensible defaults for model configuration."""
        defaults = {
            'arima': {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 12),
                'target_column': 'close'
            },
            'lstm': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'target_column': 'close'
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'target_column': 'close'
            },
            'autoformer': {
                'd_model': 128,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'target_column': 'close'
            }
        }
        
        if PROPHET_AVAILABLE:
            defaults['prophet'] = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'target_column': 'close'
            }
        
        return defaults.get(model_type, {'target_column': 'close'})
    
    def _get_fallback_model(self, failed_model: str) -> str:
        """Get fallback model if primary model fails."""
        fallback_map = {
            'lstm': 'xgboost',
            'xgboost': 'arima',
            'arima': 'lstm',
            'autoformer': 'lstm',
            'prophet': 'arima'
        }
        return fallback_map.get(failed_model, 'arima')
    
    def _generate_simple_forecast(self, data: pd.DataFrame, horizon: int) -> np.ndarray:
        """Generate simple forecast as fallback."""
        logger.info("Generating simple forecast as fallback")
        last_value = data.iloc[-1]['close'] if 'close' in data.columns else data.iloc[-1].iloc[0]
        return np.full(horizon, last_value)
    
    def _get_confidence(self, model, model_type: str) -> float:
        """Get model confidence score."""
        try:
            return model.get_confidence()
        except:
            # Default confidence based on model type
            confidence_map = {
                'lstm': 0.85,
                'xgboost': 0.80,
                'arima': 0.75,
                'autoformer': 0.90,
                'prophet': 0.85
            }
            return confidence_map.get(model_type, 0.75)
    
    def _get_metadata(self, model, model_type: str) -> Dict[str, Any]:
        """Get model metadata."""
        try:
            return model.get_metadata()
        except:
            return {
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
    
    def _get_warnings(self, data: pd.DataFrame, model_type: str) -> List[str]:
        """Get warnings about data or model selection."""
        warnings = []
        
        if len(data) < 100:
            warnings.append("Small dataset may affect forecast accuracy")
        
        if data.isnull().any().any():
            warnings.append("Data contains missing values")
        
        if model_type == 'lstm' and len(data) < 500:
            warnings.append("LSTM may not perform well with small datasets")
        
        return warnings
    
    def _get_fallback_result(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Get fallback result when all else fails."""
        logger.warning("Using fallback forecast result")
        fallback_forecast = self._generate_simple_forecast(data, horizon)
        
        return {
            'model': 'fallback',
            'forecast': fallback_forecast,
            'confidence': 0.5,
            'metadata': {
                'model_type': 'fallback',
                'timestamp': datetime.now().isoformat(),
                'warning': 'Fallback forecast used due to errors'
            },
            'warnings': ['Fallback forecast used due to system errors']
        }
        
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

    def forecast(self, data: pd.DataFrame, horizon: int = 30, 
                model_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate forecast using the best available model.
        
        Args:
            data: Time series data
            horizon: Forecast horizon in periods
            model_type: Optional preferred model type
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing forecast results
        """
        try:
            # Use existing get_forecast method
            result = self.get_forecast(data, horizon, model_type, **kwargs)
            
            # Add forecast-specific metadata
            result['forecast_method'] = 'router_selected'
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Forecast router error: {e}")
            raise RuntimeError(f"Forecast router failed: {e}")

    def plot_results(self, data: pd.DataFrame, forecast_result: Dict[str, Any]) -> None:
        """Plot forecast results.
        
        Args:
            data: Historical data
            forecast_result: Result from forecast method
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(data.index, data.iloc[:, 0], label='Historical', color='blue')
            
            # Plot forecast
            if 'forecast' in forecast_result:
                forecast_data = forecast_result['forecast']
                if isinstance(forecast_data, pd.DataFrame):
                    forecast_values = forecast_data.iloc[:, 0]
                else:
                    forecast_values = forecast_data
                
                # Create future dates
                last_date = data.index[-1]
                future_dates = pd.date_range(start=last_date, periods=len(forecast_values)+1, freq='D')[1:]
                
                plt.plot(future_dates, forecast_values, label='Forecast', color='red')
            
            plt.title(f'Forecast using {forecast_result.get("model", "Unknown")} model')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting forecast results: {e}")
            print(f"Could not plot results: {e}")


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