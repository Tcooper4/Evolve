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
            df = train_data[[self.config['date_column'], self.config['target_column']]].rename(columns={
                self.config['date_column']: 'ds',
                self.config['target_column']: 'y'
            })
            self.model.fit(df)
            self.fitted = True
            self.history = df
            return {'train_loss': [], 'val_loss': []}

        def predict(self, data: pd.DataFrame, horizon: int = 1):
            if not self.fitted:
                raise RuntimeError('Model must be fit before predicting.')
            future = data[[self.config['date_column']]].rename(columns={self.config['date_column']: 'ds'})
            forecast = self.model.predict(future)
            return forecast['yhat'].values

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