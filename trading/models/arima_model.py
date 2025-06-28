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
                    except Exception as e:
                        import logging
                        logging.error(f"Error fitting ARIMA model with order {(p, d, q)}: {e}")
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

    def forecast(self, data: pd.Series, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.
        
        Args:
            data: Historical time series data
            horizon: Number of time steps to forecast
            
        Returns:
            Dictionary containing forecast results
        """
        try:
            if not self.is_fitted:
                # Fit the model if not already fitted
                self.fit(data)
            
            # Generate forecast
            forecast_result = self.fitted_model.forecast(steps=horizon)
            
            return {
                'forecast': forecast_result.values,
                'confidence': 0.8,  # ARIMA confidence
                'model': 'ARIMA',
                'horizon': horizon,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': self.get_aic(),
                'bic': self.get_bic()
            }
            
        except Exception as e:
            import logging
            logging.error(f"Error in ARIMA model forecast: {e}")
            raise RuntimeError(f"ARIMA model forecasting failed: {e}")

    def plot_results(self, data: pd.Series, predictions: np.ndarray = None) -> None:
        """Plot ARIMA model results and predictions.
        
        Args:
            data: Input time series data
            predictions: Optional predictions to plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if predictions is None:
                predictions = self.predict(steps=len(data))
            
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Historical vs Predicted
            plt.subplot(2, 2, 1)
            plt.plot(data.index, data.values, label='Actual', color='blue')
            plt.plot(data.index[-len(predictions):], predictions, label='Predicted', color='red')
            plt.title('ARIMA Model Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Plot 2: Residuals
            plt.subplot(2, 2, 2)
            if len(predictions) == len(data):
                residuals = data.values - predictions
                plt.plot(residuals)
                plt.title('Model Residuals')
                plt.xlabel('Time')
                plt.ylabel('Residual')
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, 'Residuals not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Model Residuals')
            
            # Plot 3: ACF of residuals
            plt.subplot(2, 2, 3)
            if self.is_fitted:
                try:
                    residuals = self.fitted_model.resid
                    plot_acf(residuals, ax=plt.gca(), lags=40)
                    plt.title('ACF of Residuals')
                except:
                    plt.text(0.5, 0.5, 'ACF not available', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('ACF of Residuals')
            else:
                plt.text(0.5, 0.5, 'Model not fitted', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('ACF of Residuals')
            
            # Plot 4: Model information
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, f'Model: ARIMA{self.order}', fontsize=12)
            if self.seasonal_order:
                plt.text(0.1, 0.6, f'Seasonal: {self.seasonal_order}', fontsize=12)
            plt.text(0.1, 0.4, f'AIC: {self.get_aic():.2f}', fontsize=12)
            plt.text(0.1, 0.2, f'BIC: {self.get_bic():.2f}', fontsize=12)
            plt.title('Model Information')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            import logging
            logging.error(f"Error plotting ARIMA results: {e}")
            print(f"Could not plot results: {e}") 