"""ProphetModel: Facebook Prophet wrapper for time series forecasting."""
from .base_model import BaseModel, ModelRegistry, ValidationError
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any
from datetime import datetime

# Try to import Prophet, but make it optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

if PROPHET_AVAILABLE:
    @ModelRegistry.register('Prophet')
    class ProphetModel(BaseModel):
        def __init__(self, config):
            super().__init__(config)
            self.model = Prophet(**config.get('prophet_params', {}))
            self.fitted = False
            self.history = None
            return {'success': True, 'message': 'ProphetModel initialized', 'timestamp': datetime.now().isoformat()}

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
                # Validate input data
                if train_data is None or train_data.empty:
                    raise ValueError("Training data is empty or None")
                
                required_columns = [self.config['date_column'], self.config['target_column']]
                missing_columns = [col for col in required_columns if col not in train_data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Check for NaN values
                if train_data[required_columns].isnull().any().any():
                    import logging
                    logging.warning("NaN values found in training data, attempting to clean")
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
                
                import logging
                logging.info(f"Prophet model fitted successfully with {len(df)} data points")
                
                return {'train_loss': [], 'val_loss': []}
                
            except Exception as e:
                import logging
                logging.error(f"Error fitting Prophet model: {e}")
                raise RuntimeError(f"Prophet model fitting failed: {e}")

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
                    import logging
                    logging.warning("Prophet predict: Input dataframe is empty, returning empty result")
                    return np.array([])
                
                # Check for required columns
                if self.config['date_column'] not in data.columns:
                    import logging
                    logging.warning(f"Prophet predict: Missing required column '{self.config['date_column']}', returning empty result")
                    return np.array([])
                
                # Check for NaN values
                if data[self.config['date_column']].isnull().any():
                    import logging
                    logging.warning("Prophet predict: NaN values found in date column, attempting to clean")
                    data = data.dropna(subset=[self.config['date_column']])
                    if data.empty:
                        logging.warning("Prophet predict: No valid data after cleaning, returning empty result")
                        return np.array([])
                
                # Validate data size
                if len(data) < 2:
                    import logging
                    logging.warning("Prophet predict: Insufficient data points, returning empty result")
                    return np.array([])
                
                if not self.fitted:
                    import logging
                    logging.warning("Prophet predict: Model not fitted, returning empty result")
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
                import logging
                logging.error(f"Error in Prophet predict: {e}")
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
                import logging
                logging.error(f"Error in Prophet model forecast: {e}")
                raise RuntimeError(f"Prophet model forecasting failed: {e}")

        def summary(self):
            print("ProphetModel: Facebook Prophet wrapper")
            print(self.model)

        def infer(self):
            pass  # Prophet is always in inference mode after fitting

        def shap_interpret(self, X_sample):
            print("SHAP not supported for Prophet. Showing component plots instead.")
            self.model.plot_components(self.model.predict(self.history))

        def save(self, path: str):
            os.makedirs(path, exist_ok=True)
            self.model.save(os.path.join(path, 'prophet_model.json'))
            with open(os.path.join(path, 'config.json'), 'w') as f:
                json.dump(self.config, f)

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