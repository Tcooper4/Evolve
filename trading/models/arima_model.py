"""ARIMA model for time series forecasting with enhanced auto_arima support."""

import logging
import warnings
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from utils.forecast_helpers import (
    log_forecast_performance,
    safe_forecast,
    validate_forecast_input,
)

from .base_model import BaseModel

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


class OptimizationCriterion(Enum):
    """Optimization criteria for auto_arima."""

    AIC = "aic"
    BIC = "bic"
    MSE = "mse"
    RMSE = "rmse"


class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting with enhanced auto_arima support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ARIMA model.

        Args:
            config: Model configuration with enhanced auto_arima options:
                - use_auto_arima: Whether to use auto_arima (default: True)
                - seasonal: Whether to consider seasonal components (default: True)
                - optimization_criterion: AIC, BIC, MSE, or RMSE (default: AIC)
                - auto_arima_config: Additional pmdarima configuration
                - backtest_steps: Number of steps for MSE/RMSE optimization (default: 5)
        """
        super().__init__(config)
        self.model = None
        self.fitted_model = None
        self.order = config.get("order", (1, 1, 1)) if config else (1, 1, 1)
        self.seasonal_order = config.get("seasonal_order", None) if config else None
        self.is_fitted = False

        # Enhanced auto_arima configuration
        self.use_auto_arima = config.get("use_auto_arima", True) if config else True
        self.seasonal = config.get("seasonal", True) if config else True
        self.optimization_criterion = config.get("optimization_criterion", "aic")
        self.backtest_steps = config.get("backtest_steps", 5) if config else 5

        # Auto_arima specific configuration
        self.auto_arima_config = config.get("auto_arima_config", {}) if config else {}

        # Validate optimization criterion
        if self.optimization_criterion not in ["aic", "bic", "mse", "rmse"]:
            logger.warning(
                f"Invalid optimization criterion: {self.optimization_criterion}. Using AIC."
            )
            self.optimization_criterion = "aic"

    def fit(self, data: pd.Series) -> Dict[str, Any]:
        """Fit the ARIMA model using enhanced auto_arima if enabled.

        Args:
            data: Time series data

        Returns:
            Dictionary with fit status and model reference
        """
        try:
            # Add input length check
            if len(data) < 20:
                raise ValueError("ARIMA requires at least 20 data points.")

            # Use enhanced auto_arima if enabled
            if self.use_auto_arima:
                return self._fit_enhanced_auto_arima(data)
            else:
                return self._fit_manual_arima(data)

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
                "model": self,
            }

    def _fit_enhanced_auto_arima(self, data: pd.Series) -> Dict[str, Any]:
        """Fit ARIMA model using enhanced pmdarima.auto_arima with multiple optimization criteria."""
        try:
            import pmdarima as pm

            logger.info(
                f"Using enhanced pmdarima.auto_arima with {self.optimization_criterion.upper()} optimization..."
            )

            # Base auto_arima configuration
            base_config = {
                "start_p": 0,
                "start_q": 0,
                "max_p": 5,
                "max_q": 5,
                "max_d": 2,
                "seasonal": self.seasonal,
                "m": 12 if self.seasonal else 1,
                "D": 1 if self.seasonal else 0,
                "trace": True,
                "error_action": "ignore",
                "suppress_warnings": True,
                "stepwise": True,
                "random_state": 42,
            }

            # Merge with user config
            config = {**base_config, **self.auto_arima_config}

            # Handle different optimization criteria
            if self.optimization_criterion in ["aic", "bic"]:
                # Use built-in AIC/BIC optimization
                config["information_criterion"] = self.optimization_criterion
                auto_model = pm.auto_arima(data, **config)

            elif self.optimization_criterion in ["mse", "rmse"]:
                # Use backtesting for MSE/RMSE optimization
                auto_model = self._optimize_with_backtest(data, config)
            else:
                # Default to AIC
                config["information_criterion"] = "aic"
                auto_model = pm.auto_arima(data, **config)

            # Extract the best model
            self.fitted_model = auto_model
            self.order = auto_model.order
            self.seasonal_order = auto_model.seasonal_order
            self.is_fitted = True

            # Log model selection results
            logger.info(f"Enhanced Auto ARIMA selected order: {self.order}")
            if self.seasonal_order and self.seasonal_order != (0, 0, 0, 0):
                logger.info(
                    f"Enhanced Auto ARIMA selected seasonal order: {self.seasonal_order}"
                )

            # Log optimization results
            aic_result = self.get_aic()
            bic_result = self.get_bic()

            logger.info(f"Model AIC: {aic_result.get('aic', 'N/A')}")
            logger.info(f"Model BIC: {bic_result.get('bic', 'N/A')}")

            return {
                "success": True,
                "message": f"Enhanced Auto ARIMA model fitted successfully with {
                    self.optimization_criterion.upper()} optimization",
                "timestamp": pd.Timestamp.now().isoformat(),
                "model": self,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "optimization_criterion": self.optimization_criterion,
                "aic": aic_result.get("aic"),
                "bic": bic_result.get("bic"),
            }

        except ImportError:
            logger.warning("pmdarima not available, falling back to manual ARIMA")
            return self._fit_manual_arima(data)
        except Exception as e:
            logger.error(f"Enhanced Auto ARIMA fitting failed: {e}")
            return self._fit_manual_arima(data)

    def _optimize_with_backtest(self, data: pd.Series, config: Dict[str, Any]) -> Any:
        """Optimize ARIMA parameters using backtesting for MSE/RMSE."""
        import pmdarima as pm

        logger.info(
            f"Optimizing ARIMA parameters using {self.optimization_criterion.upper()} backtesting..."
        )

        # Define parameter ranges to test
        p_range = range(config.get("start_p", 0), config.get("max_p", 3) + 1)
        d_range = range(0, config.get("max_d", 2) + 1)
        q_range = range(config.get("start_q", 0), config.get("max_q", 3) + 1)

        best_score = float("inf")
        best_model = None
        best_params = None

        # Grid search with backtesting
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        # Create model with current parameters
                        if self.seasonal and config.get("m", 12) > 1:
                            # Try seasonal model
                            for P in [0, 1]:
                                for D in [0, 1]:
                                    for Q in [0, 1]:
                                        try:
                                            model = pm.ARIMA(
                                                order=(p, d, q),
                                                seasonal_order=(
                                                    P,
                                                    D,
                                                    Q,
                                                    config.get("m", 12),
                                                ),
                                            )
                                            model.fit(data)

                                            # Backtest
                                            score = self._backtest_model(model, data)

                                            if score < best_score:
                                                best_score = score
                                                best_model = model
                                                best_params = (p, d, q, P, D, Q)

                                        except BaseException:
                                            continue
                        else:
                            # Non-seasonal model
                            model = pm.ARIMA(order=(p, d, q))
                            model.fit(data)

                            # Backtest
                            score = self._backtest_model(model, data)

                            if score < best_score:
                                best_score = score
                                best_model = model
                                best_params = (p, d, q)

                    except BaseException:
                        continue

        if best_model is None:
            logger.warning("Backtesting optimization failed, using default auto_arima")
            return pm.auto_arima(data, **config)

        logger.info(
            f"Best {self.optimization_criterion.upper()} score: {best_score:.6f}"
        )
        logger.info(f"Best parameters: {best_params}")

        return best_model

    def _backtest_model(self, model: Any, data: pd.Series) -> float:
        """Perform backtesting to calculate MSE or RMSE."""
        try:
            # Use last backtest_steps for validation
            train_data = data[: -self.backtest_steps]
            test_data = data[-self.backtest_steps:]

            # Fit model on training data
            model.fit(train_data)

            # Make predictions
            predictions = model.predict(n_periods=self.backtest_steps)

            # Calculate error
            if self.optimization_criterion == "mse":
                error = np.mean((test_data.values - predictions) ** 2)
            else:  # rmse
                error = np.sqrt(np.mean((test_data.values - predictions) ** 2))

            return error

        except Exception as e:
            logger.warning(f"Backtesting failed: {e}")
            return float("inf")

    def _fit_auto_arima(self, data: pd.Series) -> Dict[str, Any]:
        """Legacy auto_arima method - now calls enhanced version."""
        return self._fit_enhanced_auto_arima(data)

    def _fit_manual_arima(self, data: pd.Series) -> Dict[str, Any]:
        """Fit ARIMA model with manually specified parameters."""
        try:
            logger.info(f"Fitting manual ARIMA with order: {self.order}")
            # Create ARIMA model
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                try:
                    self.model = SARIMAX(
                        data, order=self.order, seasonal_order=self.seasonal_order
                    )
                    self.fitted_model = self.model.fit()
                    self.is_fitted = True
                except Exception as seasonal_exc:
                    logger.warning(
                        f"Seasonal ARIMA fitting failed: {seasonal_exc}. Falling back to non-seasonal ARIMA."
                    )
                    self.model = ARIMA(data, order=self.order)
                    self.fitted_model = self.model.fit()
                    self.is_fitted = True
            else:
                self.model = ARIMA(data, order=self.order)
                self.fitted_model = self.model.fit()
                self.is_fitted = True
            # Log AIC/BIC values
            aic_result = self.get_aic()
            bic_result = self.get_bic()
            logger.info(f"Model AIC: {aic_result.get('aic', 'N/A')}")
            logger.info(f"Model BIC: {bic_result.get('bic', 'N/A')}")
            return {
                "success": True,
                "message": "Manual ARIMA model fitted successfully",
                "timestamp": pd.Timestamp.now().isoformat(),
                "model": self,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "aic": aic_result.get("aic"),
                "bic": bic_result.get("bic"),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
                "model": self,
            }

    def predict(self, steps: int = 1, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Make predictions with confidence intervals.

        Args:
            steps: Number of steps to predict
            confidence_level: Confidence level for intervals (default: 0.95)

        Returns:
            Dictionary with predictions, confidence intervals, and status
        """
        if not self.is_fitted:
            return {
                "success": False,
                "error": "Model must be fitted before making predictions",
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        try:
            # Get forecast with confidence intervals
            if hasattr(self.fitted_model, "forecast"):
                # For pmdarima models
                forecast_result = self.fitted_model.forecast(
                    steps=steps, return_conf_int=True, alpha=1 - confidence_level
                )
                if isinstance(forecast_result, tuple):
                    forecast_values = forecast_result[0]
                    conf_int = forecast_result[1]
                else:
                    forecast_values = forecast_result
                    conf_int = None
            else:
                # For statsmodels ARIMA
                forecast_values = self.fitted_model.forecast(steps=steps)
                conf_int = self.fitted_model.get_forecast(steps=steps).conf_int(
                    alpha=1 - confidence_level
                )

            # Convert to numpy arrays if needed
            if hasattr(forecast_values, "values"):
                forecast_values = forecast_values.values
            if conf_int is not None and hasattr(conf_int, "values"):
                conf_int = conf_int.values

            result = {
                "success": True,
                "predictions": forecast_values,
                "timestamp": pd.Timestamp.now().isoformat(),
                "steps": steps,
                "confidence_level": confidence_level,
            }

            # Add confidence intervals if available
            if conf_int is not None:
                result["confidence_intervals"] = {
                    "lower": conf_int[:, 0] if conf_int.ndim > 1 else conf_int[0],
                    "upper": conf_int[:, 1] if conf_int.ndim > 1 else conf_int[1],
                }

            # Log prediction summary
            logger.info(f"ARIMA prediction completed for {steps} steps")
            if conf_int is not None:
                logger.info(
                    f"Confidence intervals calculated at {confidence_level * 100}% level"
                )

            return result

        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {e}")
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

    @safe_forecast(max_retries=2, retry_delay=0.5, log_errors=True)
    def forecast(self, data: pd.Series, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.

        Args:
            data: Historical time series data
            horizon: Number of time steps to forecast

        Returns:
            Dictionary containing forecast results
        """
        import time

        start_time = time.time()

        # Validate input data
        validate_forecast_input(data, min_length=20, require_numeric=True)

        if not self.is_fitted:
            # Fit the model if not already fitted
            self.fit(data)

        # Generate forecast
        forecast_result = self.fitted_model.forecast(steps=horizon)

        execution_time = time.time() - start_time
        confidence = 0.8  # ARIMA confidence

        # Log performance
        log_forecast_performance(
            model_name="ARIMA",
            execution_time=execution_time,
            data_length=len(data),
            confidence=confidence,
        )

        return {
            "forecast": forecast_result.values,
            "confidence": confidence,
            "model": "ARIMA",
            "horizon": horizon,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "aic": self.get_aic(),
            "bic": self.get_bic(),
        }

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
                data.index[-len(predictions):],
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
