"""ARIMA model for time series forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseModel

class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ARIMA model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model = None
        self.fitted_model = None
        self.order = config.get('order', (1, 1, 1)) if config else (1, 1, 1)
        self.seasonal_order = config.get('seasonal_order', None) if config else None
        
    def fit(self, data: pd.Series) -> 'ARIMAModel':
        """Fit the ARIMA model.
        
        Args:
            data: Time series data
            
        Returns:
            Self for chaining
        """
        try:
            # Create ARIMA model
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
            else:
                self.model = ARIMA(data, order=self.order)
            
            # Fit the model
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            return self
            
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Make predictions.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return forecast.values
        except Exception as e:
            print(f"Error making predictions: {e}")
            return np.zeros(steps)
    
    def get_model_summary(self) -> str:
        """Get model summary.
        
        Returns:
            Model summary string
        """
        if not self.is_fitted:
            return "Model not fitted"
        
        try:
            return str(self.fitted_model.summary())
        except Exception as e:
            return f"Error getting summary: {e}"
    
    def get_aic(self) -> float:
        """Get AIC score.
        
        Returns:
            AIC score
        """
        if not self.is_fitted:
            return float('inf')
        
        try:
            return self.fitted_model.aic
        except Exception as e:
            print(f"Error getting AIC: {e}")
            return float('inf')
    
    def get_bic(self) -> float:
        """Get BIC score.
        
        Returns:
            BIC score
        """
        if not self.is_fitted:
            return float('inf')
        
        try:
            return self.fitted_model.bic
        except Exception as e:
            print(f"Error getting BIC: {e}")
            return float('inf')
    
    def check_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Check if the time series is stationary.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        try:
            # Perform Augmented Dickey-Fuller test
            result = adfuller(data.dropna())
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            print(f"Error checking stationarity: {e}")
            return {
                'adf_statistic': None,
                'p_value': None,
                'critical_values': None,
                'is_stationary': False
            }
    
    def find_best_order(self, data: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """Find the best ARIMA order using AIC.
        
        Args:
            data: Time series data
            max_p: Maximum p value
            max_d: Maximum d value
            max_q: Maximum q value
            
        Returns:
            Best (p, d, q) order
        """
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        aic = fitted.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            self.fitted_model.save(filepath)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath: str) -> None:
        """Load a fitted model.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            from statsmodels.tsa.arima.model import ARIMAResults
            self.fitted_model = ARIMAResults.load(filepath)
            self.is_fitted = True
        except Exception as e:
            print(f"Error loading model: {e}") 