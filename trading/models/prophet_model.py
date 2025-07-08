"""ProphetModel: Facebook Prophet wrapper for time series forecasting with holidays and macro support."""
from .base_model import BaseModel, ModelRegistry, ValidationError
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

# Try to import Prophet, but make it optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

# Try to import holidays package
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    holidays = None

logger = logging.getLogger(__name__)

if PROPHET_AVAILABLE:
    @ModelRegistry.register('Prophet')
    class ProphetModel(BaseModel):
        def __init__(self, config):
            super().__init__(config)
            self.model = Prophet(**config.get('prophet_params', {}))
            self.fitted = False
            self.history = None

        def fit(self, train_data: pd.DataFrame, val_data=None, **kwargs):
            """Fit the Prophet model with robust error handling.
            
            Args:
                train_data: Training data DataFrame
                val_data: Validation data (optional)
                **kwargs: Additional arguments
                
            Returns:
                Dictionary with training results
                
            Raises:
                ValueError: If data is missing or malformed
                RuntimeError: If Prophet fitting fails
            """
            try:
                # Validate Prophet configuration before fitting
                self._validate_prophet_config()
                
                # Validate Prophet model has required methods
                if not hasattr(self.model, "add_seasonality"):
                    raise ValueError("Prophet model not properly initialized with seasonal components.")
                
                # Validate input data
                if train_data is None or train_data.empty:
                    raise ValueError("Training data is empty or None")
                
                required_columns = [self.config['date_column'], self.config['target_column']]
                missing_columns = [col for col in required_columns if col not in train_data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Check for NaN values
                if train_data[required_columns].isnull().any().any():
                    logger.warning("NaN values found in training data, attempting to clean")
                    train_data = train_data.dropna(subset=required_columns)
                    if train_data.empty:
                        raise ValueError("No valid data remaining after removing NaN values")
                
                # Prepare data for Prophet
                df = train_data[required_columns].rename(columns={
                    self.config['date_column']: 'ds',
                    self.config['target_column']: 'y'
                })
                
                # Validate Prophet data format
                if df['ds'].dtype != 'datetime64[ns]':
                    df['ds'] = pd.to_datetime(df['ds'])
                
                if df['y'].dtype not in ['float64', 'int64']:
                    df['y'] = pd.to_numeric(df['y'], errors='coerce')
                    if df['y'].isnull().any():
                        raise ValueError("Target column contains non-numeric values")
                
                # Fit the model
                self.model.fit(df)
                self.fitted = True
                self.history = df
                
                logger.info(f"Prophet model fitted successfully with {len(df)} data points")
                
                return {'train_loss': [], 'val_loss': []}
                
            except Exception as e:
                logger.error(f"Error fitting Prophet model: {e}")
                raise RuntimeError(f"Prophet model fitting failed: {e}")

        def _validate_prophet_config(self):
            """Validate Prophet configuration for holidays and seasonality.
            
            Raises:
                ValueError: If configuration is invalid
            """
            try:
                prophet_params = self.config.get('prophet_params', {})
                
                # Validate holidays configuration
                if 'holidays' in prophet_params:
                    holidays = prophet_params['holidays']
                    if holidays is not None:
                        if not isinstance(holidays, pd.DataFrame):
                            raise ValueError("Holidays must be a pandas DataFrame")
                        required_holiday_cols = ['ds', 'holiday']
                        missing_cols = [col for col in required_holiday_cols if col not in holidays.columns]
                        if missing_cols:
                            raise ValueError(f"Holidays DataFrame missing required columns: {missing_cols}")
                        
                        # Validate holiday dates
                        if holidays['ds'].dtype != 'datetime64[ns]':
                            try:
                                holidays['ds'] = pd.to_datetime(holidays['ds'])
                            except (ValueError, TypeError) as e:
                                raise ValueError(f"Holiday dates must be convertible to datetime: {e}")
                
                # Validate seasonality configuration
                if 'seasonality_mode' in prophet_params:
                    seasonality_mode = prophet_params['seasonality_mode']
                    valid_modes = ['additive', 'multiplicative']
                    if seasonality_mode not in valid_modes:
                        raise ValueError(f"Seasonality mode must be one of {valid_modes}, got {seasonality_mode}")
                
                # Validate yearly_seasonality
                if 'yearly_seasonality' in prophet_params:
                    yearly_seasonality = prophet_params['yearly_seasonality']
                    if not isinstance(yearly_seasonality, (bool, int)):
                        raise ValueError("yearly_seasonality must be boolean or integer")
                    if isinstance(yearly_seasonality, int) and yearly_seasonality < 1:
                        raise ValueError("yearly_seasonality integer must be >= 1")
                
                # Validate weekly_seasonality
                if 'weekly_seasonality' in prophet_params:
                    weekly_seasonality = prophet_params['weekly_seasonality']
                    if not isinstance(weekly_seasonality, (bool, int)):
                        raise ValueError("weekly_seasonality must be boolean or integer")
                    if isinstance(weekly_seasonality, int) and weekly_seasonality < 1:
                        raise ValueError("weekly_seasonality integer must be >= 1")
                
                # Validate daily_seasonality
                if 'daily_seasonality' in prophet_params:
                    daily_seasonality = prophet_params['daily_seasonality']
                    if not isinstance(daily_seasonality, (bool, int)):
                        raise ValueError("daily_seasonality must be boolean or integer")
                    if isinstance(daily_seasonality, int) and daily_seasonality < 1:
                        raise ValueError("daily_seasonality integer must be >= 1")
                
                # Validate changepoint_prior_scale
                if 'changepoint_prior_scale' in prophet_params:
                    changepoint_prior_scale = prophet_params['changepoint_prior_scale']
                    if not isinstance(changepoint_prior_scale, (int, float)) or changepoint_prior_scale <= 0:
                        raise ValueError("changepoint_prior_scale must be a positive number")
                
                # Validate seasonality_prior_scale
                if 'seasonality_prior_scale' in prophet_params:
                    seasonality_prior_scale = prophet_params['seasonality_prior_scale']
                    if not isinstance(seasonality_prior_scale, (int, float)) or seasonality_prior_scale <= 0:
                        raise ValueError("seasonality_prior_scale must be a positive number")
                
                logger.info("Prophet configuration validation passed")
                
            except Exception as e:
                raise ValueError(f"Invalid Prophet configuration: {str(e)}")

        def predict(self, data: pd.DataFrame, horizon: int = 1):
            """Make predictions with fallback guards.
            
            Args:
                data: Input data
                horizon: Prediction horizon
                
            Returns:
                Predicted values
            """
            try:
                # Check if input dataframe is empty or has NaNs
                if data is None or data.empty:
                    logger.warning("Prophet predict: Input dataframe is empty, returning empty result")
                    return np.array([])
                
                # Check for required columns
                if self.config['date_column'] not in data.columns:
                    logger.warning(f"Prophet predict: Missing required column '{self.config['date_column']}', returning empty result")
                    return np.array([])
                
                # Check for NaN values
                if data[self.config['date_column']].isnull().any():
                    logger.warning("Prophet predict: NaN values found in date column, attempting to clean")
                    data = data.dropna(subset=[self.config['date_column']])
                    if data.empty:
                        logger.warning("Prophet predict: No valid data after cleaning, returning empty result")
                        return np.array([])
                
                # Validate data size
                if len(data) < 2:
                    logger.warning("Prophet predict: Insufficient data points, returning empty result")
                    return np.array([])
                
                if not self.fitted:
                    logger.warning("Prophet predict: Model not fitted, returning empty result")
                    return np.array([])
                
                # Prepare data for prediction
                future = data[[self.config['date_column']]].rename(columns={self.config['date_column']: 'ds'})
                
                # Validate date format
                if future['ds'].dtype != 'datetime64[ns]':
                    future['ds'] = pd.to_datetime(future['ds'])
                
                # Make prediction
                forecast = self.model.predict(future)
                return forecast['yhat'].values
                
            except Exception as e:
                logger.error(f"Error in Prophet predict: {e}")
                return np.array([])

        def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
            """Generate forecast for future time steps.
            
            Args:
                data: Historical data DataFrame
                horizon: Number of time steps to forecast
                
            Returns:
                Dictionary containing forecast results
            """
            try:
                if not self.fitted:
                    raise RuntimeError('Model must be fit before forecasting.')
                
                # Create future dates for forecasting
                last_date = data[self.config['date_column']].iloc[-1]
                future_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
                future_df = pd.DataFrame({'ds': future_dates})
                
                # Generate forecast
                forecast_result = self.model.predict(future_df)
                
                return {
                    'forecast': forecast_result['yhat'].values,
                    'confidence': 0.85,  # Prophet confidence
                    'model': 'Prophet',
                    'horizon': horizon,
                    'forecast_dates': future_dates,
                    'lower_bound': forecast_result['yhat_lower'].values,
                    'upper_bound': forecast_result['yhat_upper'].values
                }
                
            except Exception as e:
                logger.error(f"Error in Prophet model forecast: {e}")
                raise RuntimeError(f"Prophet model forecasting failed: {e}")

        def summary(self):
            logger.info("ProphetModel: Facebook Prophet wrapper")
            logger.info(str(self.model))

        def infer(self):
            pass  # Prophet is always in inference mode after fitting

        def shap_interpret(self, X_sample):
            logger.warning("SHAP not supported for Prophet. Showing component plots instead.")
            self.model.plot_components(self.model.predict(self.history))

        def save(self, path: str):
            """Save the Prophet model to disk."""
            try:
                os.makedirs(path, exist_ok=True)
                self.model.save(os.path.join(path, 'prophet_model.json'))
            except Exception as e:
                logger.error(f"Failed to save Prophet model to {path}: {e}")
                raise RuntimeError(f"Failed to save Prophet model: {e}")

        def load(self, path: str):
            from prophet.serialize import model_from_json
            with open(os.path.join(path, 'prophet_model.json'), 'r') as fin:
                self.model = model_from_json(fin.read())
            with open(os.path.join(path, 'config.json'), 'r') as f:
                self.config = json.load(f)
            self.fitted = True
else:
    # Create a placeholder class that raises an informative error
    class ProphetModel(BaseModel):
        def __init__(self, config):
            raise ImportError(
                "Prophet is not installed. Please install it with: pip install prophet\n"
                "Note: Prophet requires compilation and may have installation issues on Windows.\n"
                "Consider using an alternative model like ARIMA or LSTM."
            )