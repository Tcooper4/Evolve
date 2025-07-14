"""ARIMA model for time series forecasting."""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

from .base_model import BaseModel

logger = logging.getLogger(__name__)


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
        self.order = config.get("order", (1, 1, 1)) if config else (1, 1, 1)
        self.seasonal_order = config.get("seasonal_order", None) if config else None
        self.is_fitted = False

    def fit(self, data: pd.Series) -> Dict[str, Any]:
        """Fit the ARIMA model.

        Args:
            data: Time series data

        Returns:
            Dictionary with fit status and model reference
        """
        try:
            # Add input length check
            if len(data) < 20:
                raise ValueError("ARIMA requires at least 20 data points.")

            # Create ARIMA model
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                self.model = SARIMAX(
                    data, order=self.order, seasonal_order=self.seasonal_order
                )
            else:
                self.model = ARIMA(data, order=self.order)

            # Fit the model
            self.fitted_model = self.model.fit()
            self.is_fitted = True

            return {
                "success": True,
                "message": "ARIMA model fitted successfully",
                "timestamp": pd.Timestamp.now().isoformat(),
                "model": self,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
                "model": self,
            }

    def predict(self, steps: int = 1) -> Dict[str, Any]:
        """Make predictions.

        Args:
            steps: Number of steps to predict

        Returns:
            Dictionary with predictions and status
        """
        if not self.is_fitted:
            return {
                "success": False,
                "error": "Model must be fitted before making predictions",
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return {
                "success": True,
                "predictions": forecast.values,
                "timestamp": pd.Timestamp.now().isoformat(),
                "steps": steps,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "predictions": np.zeros(steps),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary.

        Returns:
            Dictionary with summary string and status
        """
        if not self.is_fitted:
            return {
                "success": False,
                "summary": "Model not fitted",
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        try:
            summary = str(self.fitted_model.summary())
            return {
                "success": True,
                "summary": summary,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "summary": f"Error getting summary: {e}",
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    def get_aic(self) -> Dict[str, Any]:
        """Get AIC score.

        Returns:
            Dictionary with AIC score and status
        """
        if not self.is_fitted:
            return {
                "success": False,
                "aic": float("inf"),
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        try:
            return {
                "success": True,
                "aic": self.fitted_model.aic,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "aic": float("inf"),
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    def get_bic(self) -> Dict[str, Any]:
        """Get BIC score.

        Returns:
            Dictionary with BIC score and status
        """
        if not self.is_fitted:
            return {
                "success": False,
                "bic": float("inf"),
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        try:
            return {
                "success": True,
                "bic": self.fitted_model.bic,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "bic": float("inf"),
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

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
                "success": True,
                "adf_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[4],
                "is_stationary": result[1] < 0.05,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "adf_statistic": None,
                "p_value": None,
                "critical_values": None,
                "is_stationary": False,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    def find_best_order(
        self, data: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3
    ) -> Dict[str, Any]:
        """Find the best ARIMA order using AIC.

        Args:
            data: Time series data
            max_p: Maximum p value
            max_d: Maximum d value
            max_q: Maximum q value

        Returns:
            Dictionary with best order and status
        """
        best_aic = float("inf")
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
                        logging.error(
                            f"Error fitting ARIMA model with order {(p, d, q)}: {e}"
                        )
                        continue
        return {
            "success": True,
            "best_order": best_order,
            "best_aic": best_aic,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    def save_model(self, filepath: str) -> Dict[str, Any]:
        """Save the fitted model.

        Args:
            filepath: Path to save the model

        Returns:
            Dictionary with save status
        """
        if not self.is_fitted:
            return {
                "success": False,
                "error": "Model must be fitted before saving",
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        try:
            self.fitted_model.save(filepath)
            return {
                "success": True,
                "message": f"Model saved to {filepath}",
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Load a fitted model.

        Args:
            filepath: Path to the saved model

        Returns:
            Dictionary with load status
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAXResults

            self.fitted_model = SARIMAXResults.load(filepath)
            self.is_fitted = True
            return {
                "success": True,
                "message": f"Model loaded from {filepath}",
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

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
                "forecast": forecast_result.values,
                "confidence": 0.8,  # ARIMA confidence
                "model": "ARIMA",
                "horizon": horizon,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "aic": self.get_aic(),
                "bic": self.get_bic(),
            }

        except Exception as e:
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
            plt.plot(data.index, data.values, label="Actual", color="blue")
            plt.plot(
                data.index[-len(predictions) :],
                predictions,
                label="Predicted",
                color="red",
            )
            plt.title("ARIMA Model Predictions")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            # Plot 2: Residuals
            plt.subplot(2, 2, 2)
            if len(predictions) == len(data):
                residuals = data.values - predictions
                plt.plot(residuals)
                plt.title("Model Residuals")
                plt.xlabel("Time")
                plt.ylabel("Residual")
                plt.grid(True)
            else:
                plt.text(
                    0.5,
                    0.5,
                    "Residuals not available",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
                plt.title("Model Residuals")

            # Plot 3: ACF of residuals
            plt.subplot(2, 2, 3)
            if self.is_fitted:
                try:
                    residuals = self.fitted_model.resid
                    plot_acf(residuals, ax=plt.gca(), lags=40)
                    plt.title("ACF of Residuals")
                except Exception as e:
                    logging.error(f"Error plotting ACF of residuals: {e}")
                    plt.text(
                        0.5,
                        0.5,
                        "ACF not available",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.title("ACF of Residuals")
            else:
                plt.text(
                    0.5,
                    0.5,
                    "Model not fitted",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
                plt.title("ACF of Residuals")

            # Plot 4: Model information
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, f"Model: ARIMA{self.order}", fontsize=12)
            if self.seasonal_order:
                plt.text(0.1, 0.6, f"Seasonal: {self.seasonal_order}", fontsize=12)
            plt.text(0.1, 0.4, f"AIC: {self.get_aic():.2f}", fontsize=12)
            plt.text(0.1, 0.2, f"BIC: {self.get_bic():.2f}", fontsize=12)
            plt.title("Model Information")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.error(f"Error plotting ARIMA results: {e}")
            logger.error(f"Could not plot results: {e}")

    def _prepare_data(self, data: pd.Series) -> pd.Series:
        """Prepare data for ARIMA model (stub for abstract method)."""
        return data

    def build_model(self) -> None:
        """Build ARIMA model (stub for abstract method)."""
