"""
Forecast Engine Module

This module handles forecasting operations for the Evolve Trading Platform:
- Model selection and management
- Forecast generation and ensemble methods
- Confidence calculation and validation
- Forecast evaluation and metrics
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available model types."""
    LSTM = "lstm"
    ARIMA = "arima"
    XGBOOST = "xgboost"
    PROPHET = "prophet"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    GARCH = "garch"
    RIDGE = "ridge"

@dataclass
class ForecastResult:
    """Result of a forecasting operation."""
    model_type: ModelType
    forecast_values: List[float]
    confidence: float
    forecast_dates: List[datetime]
    metrics: Dict[str, float]
    model_info: Dict[str, Any]
    processing_time: float
    timestamp: datetime

@dataclass
class EnsembleForecastResult:
    """Result of an ensemble forecasting operation."""
    individual_forecasts: Dict[str, ForecastResult]
    ensemble_forecast: List[float]
    ensemble_confidence: float
    model_weights: Dict[str, float]
    forecast_dates: List[datetime]
    metrics: Dict[str, float]
    processing_time: float
    timestamp: datetime

class ForecastEngine:
    """Handles forecasting operations and model management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the forecast engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_performance = {}
        
        # Initialize available models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available forecasting models."""
        try:
            # Import model classes
            from trading.models.lstm_model import LSTMModel
            from trading.models.arima_model import ARIMAModel
            from trading.models.xgboost_model import XGBoostModel
            from trading.models.ridge_model import RidgeModel
            from trading.models.garch_model import GARCHModel
            
            # Initialize model instances with proper parameters
            self.models[ModelType.LSTM] = LSTMModel(input_dim=1, hidden_dim=50, output_dim=1)
            self.models[ModelType.ARIMA] = ARIMAModel()
            self.models[ModelType.XGBOOST] = XGBoostModel()
            self.models[ModelType.RIDGE] = RidgeModel()
            self.models[ModelType.GARCH] = GARCHModel()
            
            self.logger.info(f"Initialized {len(self.models)} forecasting models")
            
        except ImportError as e:
            self.logger.warning(f"Some models not available: {e}")
    
    def generate_forecast(self, data: pd.DataFrame, model_type: ModelType, 
                         forecast_days: int, **kwargs) -> ForecastResult:
        """
        Generate a forecast using a specific model.
        
        Args:
            data: Historical price data
            model_type: Type of model to use
            forecast_days: Number of days to forecast
            **kwargs: Additional model-specific parameters
            
        Returns:
            ForecastResult: Forecast result
        """
        start_time = datetime.now()
        
        try:
            if model_type not in self.models:
                raise ValueError(f"Model type {model_type} not available")
            
            model = self.models[model_type]
            
            # Prepare data for forecasting
            prepared_data = self._prepare_data(data, model_type)
            
            # Generate forecast
            forecast_values = model.forecast(prepared_data, forecast_days, **kwargs)
            
            # Calculate confidence
            confidence = self._calculate_forecast_confidence(forecast_values, data)
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates(data, forecast_days)
            
            # Calculate metrics
            metrics = self._calculate_forecast_metrics(forecast_values, data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ForecastResult(
                model_type=model_type,
                forecast_values=forecast_values,
                confidence=confidence,
                forecast_dates=forecast_dates,
                metrics=metrics,
                model_info=model.get_model_info(),
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Generated forecast using {model_type.value}: {len(forecast_values)} values")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating forecast with {model_type.value}: {e}")
            raise
    
    def generate_ensemble_forecast(self, data: pd.DataFrame, forecast_days: int,
                                 model_types: Optional[List[ModelType]] = None,
                                 weights: Optional[Dict[str, float]] = None) -> EnsembleForecastResult:
        """
        Generate an ensemble forecast using multiple models.
        
        Args:
            data: Historical price data
            forecast_days: Number of days to forecast
            model_types: List of model types to use (default: all available)
            weights: Optional weights for each model
            
        Returns:
            EnsembleForecastResult: Ensemble forecast result
        """
        start_time = datetime.now()
        
        if model_types is None:
            model_types = list(self.models.keys())
        
        if weights is None:
            weights = self._calculate_model_weights(model_types)
        
        individual_forecasts = {}
        
        # Generate individual forecasts
        for model_type in model_types:
            try:
                forecast_result = self.generate_forecast(data, model_type, forecast_days)
                individual_forecasts[model_type.value] = forecast_result
            except Exception as e:
                self.logger.warning(f"Failed to generate forecast with {model_type.value}: {e}")
        
        if not individual_forecasts:
            raise ValueError("No successful individual forecasts generated")
        
        # Combine forecasts
        ensemble_forecast = self._combine_forecasts(individual_forecasts, weights)
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(individual_forecasts, weights)
        
        # Generate forecast dates
        forecast_dates = self._generate_forecast_dates(data, forecast_days)
        
        # Calculate ensemble metrics
        metrics = self._calculate_ensemble_metrics(ensemble_forecast, data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = EnsembleForecastResult(
            individual_forecasts=individual_forecasts,
            ensemble_forecast=ensemble_forecast,
            ensemble_confidence=ensemble_confidence,
            model_weights=weights,
            forecast_dates=forecast_dates,
            metrics=metrics,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
        self.logger.info(f"Generated ensemble forecast using {len(individual_forecasts)} models")
        return result
    
    def _prepare_data(self, data: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
        """
        Prepare data for a specific model type.
        
        Args:
            data: Raw data
            model_type: Type of model
            
        Returns:
            DataFrame: Prepared data
        """
        # Basic data preparation
        prepared_data = data.copy()
        
        # Ensure we have required columns
        required_columns = ['timestamp', 'Close']
        for col in required_columns:
            if col not in prepared_data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Sort by timestamp
        prepared_data = prepared_data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove any NaN values
        prepared_data = prepared_data.dropna()
        
        # Model-specific preparation
        if model_type == ModelType.LSTM:
            # LSTM requires normalized data
            prepared_data['Close_normalized'] = (prepared_data['Close'] - prepared_data['Close'].mean()) / prepared_data['Close'].std()
        
        elif model_type == ModelType.ARIMA:
            # ARIMA works with time series
            prepared_data = prepared_data.set_index('timestamp')
        
        elif model_type == ModelType.XGBOOST:
            # XGBoost requires feature engineering
            prepared_data = self._add_technical_features(prepared_data)
        
        return prepared_data
    
    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators as features.
        
        Args:
            data: Price data
            
        Returns:
            DataFrame: Data with technical features
        """
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Price changes
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_5'] = data['Close'].pct_change(periods=5)
        
        return data
    
    def _calculate_forecast_confidence(self, forecast_values: List[float], 
                                     historical_data: pd.DataFrame) -> float:
        """
        Calculate confidence in the forecast.
        
        Args:
            forecast_values: Forecasted values
            historical_data: Historical data
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if not forecast_values:
            return 0.0
        
        # Calculate volatility of historical data
        historical_volatility = historical_data['Close'].pct_change().std()
        
        # Calculate forecast volatility
        forecast_volatility = np.std(np.diff(forecast_values) / forecast_values[:-1]) if len(forecast_values) > 1 else 0
        
        # Confidence based on volatility consistency
        volatility_confidence = 1.0 - abs(forecast_volatility - historical_volatility) / max(historical_volatility, 0.001)
        volatility_confidence = max(0.0, min(1.0, volatility_confidence))
        
        # Confidence based on forecast smoothness
        smoothness_confidence = 1.0 - np.std(np.diff(forecast_values)) / np.mean(forecast_values)
        smoothness_confidence = max(0.0, min(1.0, smoothness_confidence))
        
        # Combined confidence
        confidence = (volatility_confidence + smoothness_confidence) / 2
        
        return confidence
    
    def _generate_forecast_dates(self, data: pd.DataFrame, forecast_days: int) -> List[datetime]:
        """
        Generate forecast dates.
        
        Args:
            data: Historical data
            forecast_days: Number of days to forecast
            
        Returns:
            List[datetime]: Forecast dates
        """
        if 'timestamp' not in data.columns:
            return []
        
        last_date = data['timestamp'].max()
        forecast_dates = []
        
        for i in range(1, forecast_days + 1):
            forecast_date = last_date + timedelta(days=i)
            forecast_dates.append(forecast_date)
        
        return forecast_dates
    
    def _calculate_forecast_metrics(self, forecast_values: List[float], 
                                  historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate forecast metrics with standardized evaluation.
        
        Args:
            forecast_values: Forecasted values
            historical_data: Historical data
            
        Returns:
            Dict: Forecast metrics
        """
        if not forecast_values:
            return {}
        
        # Basic forecast statistics
        metrics = {
            'forecast_mean': np.mean(forecast_values),
            'forecast_std': np.std(forecast_values),
            'forecast_min': np.min(forecast_values),
            'forecast_max': np.max(forecast_values),
            'forecast_range': np.max(forecast_values) - np.min(forecast_values),
            'forecast_trend': (forecast_values[-1] - forecast_values[0]) / forecast_values[0] if forecast_values[0] != 0 else 0
        }
        
        # Standardized evaluation metrics
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # If we have historical data for comparison
            if not historical_data.empty and len(historical_data) >= len(forecast_values):
                # Use last n values from historical data for comparison
                historical_values = historical_data['Close'].tail(len(forecast_values)).values
                
                # Calculate standardized metrics
                mse = mean_squared_error(historical_values, forecast_values)
                mae = mean_absolute_error(historical_values, forecast_values)
                rmse = np.sqrt(mse)
                r2 = r2_score(historical_values, forecast_values)
                
                # Add to metrics
                metrics.update({
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2_score': r2,
                    'mape': np.mean(np.abs((historical_values - forecast_values) / historical_values)) * 100
                })
                
                # Log standardized metrics
                self.logger.info(f"Forecast evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
                
        except ImportError:
            self.logger.warning("sklearn not available, skipping standardized metrics")
        except Exception as e:
            self.logger.warning(f"Error calculating standardized metrics: {e}")
        
        # Compare with historical metrics
        if not historical_data.empty:
            historical_mean = historical_data['Close'].mean()
            metrics['forecast_vs_historical_ratio'] = metrics['forecast_mean'] / historical_mean if historical_mean != 0 else 1.0
        
        return metrics
    
    def _calculate_model_weights(self, model_types: List[ModelType]) -> Dict[str, float]:
        """
        Calculate weights for ensemble models.
        
        Args:
            model_types: List of model types
            
        Returns:
            Dict: Model weights
        """
        # Equal weights for now, could be enhanced with performance-based weighting
        weight = 1.0 / len(model_types)
        return {model_type.value: weight for model_type in model_types}
    
    def _combine_forecasts(self, individual_forecasts: Dict[str, ForecastResult], 
                          weights: Dict[str, float]) -> List[float]:
        """
        Combine individual forecasts into ensemble forecast.
        
        Args:
            individual_forecasts: Individual forecast results
            weights: Model weights
            
        Returns:
            List[float]: Combined forecast values
        """
        if not individual_forecasts:
            return []
        
        # Get the length of the shortest forecast
        min_length = min(len(forecast.forecast_values) for forecast in individual_forecasts.values())
        
        # Initialize ensemble forecast
        ensemble_forecast = [0.0] * min_length
        
        # Combine forecasts with weights
        for model_name, forecast_result in individual_forecasts.items():
            weight = weights.get(model_name, 1.0 / len(individual_forecasts))
            forecast_values = forecast_result.forecast_values[:min_length]
            
            for i in range(min_length):
                ensemble_forecast[i] += forecast_values[i] * weight
        
        return ensemble_forecast
    
    def _calculate_ensemble_confidence(self, individual_forecasts: Dict[str, ForecastResult], 
                                     weights: Dict[str, float]) -> float:
        """
        Calculate ensemble confidence.
        
        Args:
            individual_forecasts: Individual forecast results
            weights: Model weights
            
        Returns:
            float: Ensemble confidence
        """
        if not individual_forecasts:
            return 0.0
        
        # Weighted average of individual confidences
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for model_name, forecast_result in individual_forecasts.items():
            weight = weights.get(model_name, 1.0 / len(individual_forecasts))
            weighted_confidence += forecast_result.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_ensemble_metrics(self, ensemble_forecast: List[float], 
                                  historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate ensemble forecast metrics.
        
        Args:
            ensemble_forecast: Ensemble forecast values
            historical_data: Historical data
            
        Returns:
            Dict: Ensemble metrics
        """
        return self._calculate_forecast_metrics(ensemble_forecast, historical_data)
    
    def get_available_models(self) -> List[ModelType]:
        """
        Get list of available models.
        
        Returns:
            List[ModelType]: Available model types
        """
        return list(self.models.keys())
    
    def get_model_performance(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get performance metrics for a specific model.
        
        Args:
            model_type: Model type
            
        Returns:
            Dict: Performance metrics
        """
        return self.model_performance.get(model_type.value, {})

    def detect_drift(self, y_true: List[float], y_pred: List[float], 
                    model_id: str, threshold: float = 0.1) -> bool:
        """
        Detect model performance degradation (drift).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_id: Model identifier
            threshold: Performance degradation threshold
            
        Returns:
            bool: True if drift detected, False otherwise
        """
        try:
            from sklearn.metrics import mean_squared_error
            
            if not y_true or not y_pred or len(y_true) != len(y_pred):
                self.logger.warning(f"Invalid data for drift detection in model {model_id}")
                return False
            
            # Calculate current MSE
            current_mse = mean_squared_error(y_true, y_pred)
            
            # Get historical MSE for comparison
            historical_mse = self.model_performance.get(model_id, {}).get('historical_mse', current_mse)
            
            # Calculate degradation ratio
            degradation_ratio = current_mse / historical_mse if historical_mse > 0 else 1.0
            
            # Check if degradation exceeds threshold
            if degradation_ratio > (1 + threshold):
                self.logger.warning(f"Model drift detected for {model_id}: "
                                  f"current MSE {current_mse:.4f}, historical MSE {historical_mse:.4f}, "
                                  f"degradation ratio {degradation_ratio:.2f}")
                
                # Trigger retrain
                self._trigger_retrain(model_id)
                return True
            
            # Update historical performance
            self.model_performance.setdefault(model_id, {})['historical_mse'] = current_mse
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in drift detection for model {model_id}: {e}")
            return False

    def _trigger_retrain(self, model_id: str):
        """
        Trigger model retraining.
        
        Args:
            model_id: Model identifier
        """
        try:
            self.logger.info(f"Triggering retrain for model {model_id}")
            
            # Log retrain event
            self.model_performance.setdefault(model_id, {})['last_retrain'] = datetime.now()
            self.model_performance[model_id]['retrain_count'] = self.model_performance[model_id].get('retrain_count', 0) + 1
            
            # Here you would implement actual retraining logic
            # For now, just log the event
            self.logger.info(f"Retrain triggered for {model_id} - count: {self.model_performance[model_id]['retrain_count']}")
            
        except Exception as e:
            self.logger.error(f"Error triggering retrain for model {model_id}: {e}")

def get_forecast_engine(config: Optional[Dict[str, Any]] = None) -> ForecastEngine:
    """
    Get a forecast engine instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ForecastEngine: Configured forecast engine
    """
    return ForecastEngine(config) 