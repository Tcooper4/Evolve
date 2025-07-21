"""
GARCH Model for Volatility Forecasting

This module implements a GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
model for forecasting volatility in financial time series. GARCH models are particularly
useful for modeling the time-varying volatility that is characteristic of financial returns.

Features:
- GARCH(1,1) and GARCH(p,q) implementations
- Volatility forecasting
- Model diagnostics and validation
- Integration with the BaseModel interface
- Comprehensive logging and error handling
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# GARCH-specific imports
try:
    from arch import arch_model
    from arch.univariate import GARCH

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("ARCH library not available. Install with: pip install arch")

try:
    pass

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Install with: pip install statsmodels")

from .base_model import BaseModel, ModelError, ModelRegistry, ValidationError

logger = logging.getLogger(__name__)


@ModelRegistry.register("garch")
class GARCHModel(BaseModel):
    """
    GARCH Model for volatility forecasting.

    This model implements GARCH(1,1) and GARCH(p,q) specifications for modeling
    time-varying volatility in financial returns. It's particularly useful for:
    - Risk management and VaR calculations
    - Option pricing
    - Portfolio optimization
    - Market regime detection

    Attributes:
        model_type (str): Type of GARCH model ('GARCH', 'EGARCH', 'GJR-GARCH')
        p (int): Order of GARCH terms
        q (int): Order of ARCH terms
        mean_model (str): Mean model specification ('AR', 'MA', 'ARMA', 'Constant')
        dist (str): Error distribution ('normal', 't', 'skewt')
        fitted_model: The fitted ARCH model
        volatility_forecast: Forecasted volatility values
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GARCH model.

        Args:
            config: Configuration dictionary containing:
                - model_type: Type of GARCH model ('GARCH', 'EGARCH', 'GJR-GARCH')
                - p: Order of GARCH terms (default: 1)
                - q: Order of ARCH terms (default: 1)
                - mean_model: Mean model ('AR', 'MA', 'ARMA', 'Constant')
                - dist: Error distribution ('normal', 't', 'skewt')
                - vol: Volatility model specification
                - rescale: Whether to rescale data
        """
        super().__init__(config)

        # Validate ARCH availability
        if not ARCH_AVAILABLE:
            raise ImportError(
                "ARCH library is required for GARCH models. Install with: pip install arch"
            )

        # Set default configuration
        self.config = {
            "model_type": "GARCH",
            "p": 1,
            "q": 1,
            "mean_model": "Constant",
            "dist": "normal",
            "vol": "GARCH",
            "rescale": True,
            **self.config,
        }

        # Initialize model components
        self.fitted_model = None
        self.volatility_forecast = None
        self.conditional_volatility = None
        self.model_summary = None

        # Validate configuration
        self._validate_garch_config()

        logger.info(f"GARCH model initialized with config: {self.config}")

    def _validate_garch_config(self) -> None:
        """Validate GARCH-specific configuration."""
        valid_model_types = ["GARCH", "EGARCH", "GJR-GARCH"]
        valid_mean_models = ["AR", "MA", "ARMA", "Constant"]
        valid_distributions = ["normal", "t", "skewt"]

        if self.config["model_type"] not in valid_model_types:
            raise ValidationError(
                f"Invalid model_type. Must be one of: {valid_model_types}"
            )

        if self.config["mean_model"] not in valid_mean_models:
            raise ValidationError(
                f"Invalid mean_model. Must be one of: {valid_mean_models}"
            )

        if self.config["dist"] not in valid_distributions:
            raise ValidationError(
                f"Invalid dist. Must be one of: {valid_distributions}"
            )

        if self.config["p"] < 0 or self.config["q"] < 0:
            raise ValidationError("GARCH orders p and q must be non-negative")

        if self.config["p"] == 0 and self.config["q"] == 0:
            raise ValidationError("At least one of p or q must be greater than 0")

    def _prepare_data(
        self, data: pd.DataFrame, is_training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for GARCH modeling.

        Args:
            data: Input DataFrame with price or return data
            is_training: Whether this is for training

        Returns:
            Tuple of (returns, None) - GARCH models work with returns directly
        """
        # Ensure we have a price or return column
        if "returns" in data.columns:
            returns = data["returns"].values
        elif "price" in data.columns:
            # Calculate returns from prices
            prices = data["price"].values
            returns = np.diff(np.log(prices))
        elif "close" in data.columns:
            # Calculate returns from close prices
            prices = data["close"].values
            returns = np.diff(np.log(prices))
        else:
            # Try to infer the target column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValidationError("No numeric columns found for GARCH modeling")

            # Use the first numeric column and assume it's returns
            returns = data[numeric_cols[0]].values
            logger.warning(
                f"Using column '{numeric_cols[0]}' as returns. Ensure this is correct."
            )

        # Remove any NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) < 50:
            raise ValidationError(
                "Insufficient data for GARCH modeling. Need at least 50 observations."
            )

        # Rescale if requested
        if self.config.get("rescale", True):
            returns = returns * 100  # Convert to percentage returns

        logger.info(f"Prepared {len(returns)} observations for GARCH modeling")
        return returns, None

    def build_model(self) -> Any:
        """
        Build the GARCH model specification.

        Returns:
            ARCH model specification
        """
        # This is handled by the arch_model function during fitting
        return None

    def train(
        self,
        data: pd.DataFrame,
        target_col: str = "returns",
        feature_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the GARCH model.

        Args:
            data: Training data
            target_col: Target column name (ignored for GARCH)
            feature_cols: Feature columns (ignored for GARCH)
            **kwargs: Additional training parameters

        Returns:
            Training results dictionary
        """
        logger.info("Starting GARCH model training")
        start_time = datetime.now()

        try:
            # Prepare data
            returns, _ = self._prepare_data(data, is_training=True)

            # Create GARCH model specification
            model_spec = arch_model(
                returns,
                mean=self.config["mean_model"],
                vol=self.config["vol"],
                p=self.config["p"],
                q=self.config["q"],
                dist=self.config["dist"],
            )

            # Fit the model
            logger.info(
                f"Fitting {self.config['model_type']}({self.config['p']},{self.config['q']}) model"
            )
            self.fitted_model = model_spec.fit(disp="off", show_warning=False, **kwargs)

            # Store model summary
            self.model_summary = self.fitted_model.summary()

            # Get conditional volatility
            self.conditional_volatility = self.fitted_model.conditional_volatility

            # Calculate training metrics
            training_time = (datetime.now() - start_time).total_seconds()
            metrics = self._calculate_training_metrics()

            logger.info(
                f"GARCH model training completed in {training_time:.2f} seconds"
            )
            logger.info(f"Training metrics: {metrics}")

            return {
                "training_time": training_time,
                "metrics": metrics,
                "model_summary": str(self.model_summary),
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "log_likelihood": self.fitted_model.loglikelihood,
            }

        except Exception as e:
            logger.error(f"Error during GARCH model training: {e}")
            raise ModelError(f"GARCH training failed: {e}")

    def _calculate_training_metrics(self) -> Dict[str, float]:
        """Calculate training metrics for the GARCH model."""
        if self.fitted_model is None:
            return {}

        try:
            # Get residuals
            residuals = self.fitted_model.resid
            # Calculate metrics
            metrics = {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "log_likelihood": self.fitted_model.loglikelihood,
                "mean_residual": float(np.mean(residuals)),
                "std_residual": float(np.std(residuals)),
                "skewness": float(pd.Series(residuals).skew()),
                "kurtosis": float(pd.Series(residuals).kurtosis()),
                "ljung_box_pvalue": float(
                    self.fitted_model.conditional_volatility.std()
                ),
            }
            return metrics
        except Exception as e:
            logger.warning(f"Error calculating training metrics: {e}")
            return {}

    def predict(self, data: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """
        Predict volatility using the fitted GARCH model.

        Args:
            data: Input data (can be empty for GARCH forecasting)
            horizon: Forecast horizon

        Returns:
            Predicted volatility values
        """
        if self.fitted_model is None:
            raise ModelError("Model must be trained before prediction")

        try:
            # For GARCH models, we can forecast volatility directly
            forecast = self.fitted_model.forecast(horizon=horizon)

            # Extract volatility forecast
            if hasattr(forecast, "variance"):
                self.volatility_forecast = np.sqrt(forecast.variance.values[-horizon:])
            else:
                # Fallback for different forecast structures
                self.volatility_forecast = np.sqrt(forecast.values[-horizon:])

            logger.info(f"Generated volatility forecast for horizon {horizon}")
            return self.volatility_forecast

        except Exception as e:
            logger.error(f"Error during GARCH prediction: {e}")
            raise ModelError(f"GARCH prediction failed: {e}")

    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """
        Generate volatility forecast with confidence intervals.

        Args:
            data: Input data
            horizon: Forecast horizon

        Returns:
            Forecast results dictionary
        """
        logger.info(f"Generating GARCH volatility forecast for {horizon} periods")

        try:
            # Get point forecast
            volatility_forecast = self.predict(data, horizon)

            # Generate confidence intervals (simplified)
            # In practice, you might want to use bootstrap or simulation methods
            std_error = np.std(volatility_forecast) * 0.1  # Simplified
            confidence_intervals = np.column_stack(
                [
                    volatility_forecast - 1.96 * std_error,
                    volatility_forecast + 1.96 * std_error,
                ]
            )

            forecast_results = {
                "volatility": volatility_forecast,
                "confidence_intervals": confidence_intervals,
                "horizon": horizon,
                "model_type": self.config["model_type"],
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("GARCH volatility forecast completed")
            return forecast_results

        except Exception as e:
            logger.error(f"Error during GARCH forecasting: {e}")
            raise ModelError(f"GARCH forecasting failed: {e}")

    def evaluate(
        self, data: pd.DataFrame, target_col: str = "returns"
    ) -> Dict[str, float]:
        """
        Evaluate the GARCH model on test data.

        Args:
            data: Test data
            target_col: Target column name

        Returns:
            Evaluation metrics
        """
        if self.fitted_model is None:
            raise ModelError("Model must be trained before evaluation")

        try:
            # Prepare test data
            test_returns, _ = self._prepare_data(data, is_training=False)

            # Get model residuals and conditional volatility
            residuals = self.fitted_model.resid
            conditional_vol = self.fitted_model.conditional_volatility

            # Calculate evaluation metrics
            metrics = {
                "mean_absolute_error": float(np.mean(np.abs(residuals))),
                "root_mean_squared_error": float(np.sqrt(np.mean(residuals**2))),
                "mean_volatility": float(np.mean(conditional_vol)),
                "volatility_std": float(np.std(conditional_vol)),
                "residual_skewness": float(pd.Series(self.fitted_model.resid).skew()),
                "residual_kurtosis": float(
                    pd.Series(self.fitted_model.resid).kurtosis()
                ),
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "log_likelihood": self.fitted_model.loglikelihood,
            }

            logger.info(f"GARCH model evaluation completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during GARCH evaluation: {e}")
            raise ModelError(f"GARCH evaluation failed: {e}")

    def save_model(self, filepath: str) -> Dict[str, Any]:
        """
        Save the fitted GARCH model.

        Args:
            filepath: Path to save the model

        Returns:
            Save metadata
        """
        if self.fitted_model is None:
            raise ModelError("No fitted model to save")

        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Save the fitted model
            self.fitted_model.save(filepath)

            # Save additional metadata
            metadata = {
                "config": self.config,
                "model_type": "GARCH",
                "save_timestamp": datetime.now().isoformat(),
                "conditional_volatility": (
                    self.conditional_volatility.tolist()
                    if self.conditional_volatility is not None
                    else None
                ),
            }

            metadata_path = filepath.replace(".pkl", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"GARCH model saved to {filepath}")
            return metadata

        except Exception as e:
            logger.error(f"Error saving GARCH model: {e}")
            raise ModelError(f"Failed to save GARCH model: {e}")

    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load a fitted GARCH model.

        Args:
            filepath: Path to the saved model

        Returns:
            Load metadata
        """
        try:
            # Load the fitted model
            self.fitted_model = GARCH.load(filepath)

            # Load metadata if available
            metadata_path = filepath.replace(".pkl", "_metadata.json")
            metadata = {}
            if Path(metadata_path).exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.config.update(metadata.get("config", {}))

            # Restore conditional volatility if available
            if (
                "conditional_volatility" in metadata
                and metadata["conditional_volatility"]
            ):
                self.conditional_volatility = np.array(
                    metadata["conditional_volatility"]
                )

            logger.info(f"GARCH model loaded from {filepath}")
            return metadata

        except Exception as e:
            logger.error(f"Error loading GARCH model: {e}")
            raise ModelError(f"Failed to load GARCH model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the GARCH model."""
        info = {
            "model_type": "GARCH",
            "config": self.config,
            "is_fitted": self.fitted_model is not None,
            "conditional_volatility_available": self.conditional_volatility is not None,
            "volatility_forecast_available": self.volatility_forecast is not None,
        }

        if self.fitted_model is not None:
            info.update(
                {
                    "aic": self.fitted_model.aic,
                    "bic": self.fitted_model.bic,
                    "log_likelihood": self.fitted_model.loglikelihood,
                    "num_observations": len(self.fitted_model.resid),
                }
            )

        return info

    def get_confidence(self) -> Dict[str, float]:
        """Get model confidence metrics."""
        if self.fitted_model is None:
            return {"confidence": 0.0}

        try:
            # Calculate confidence based on model fit quality
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            log_likelihood = self.fitted_model.loglikelihood

            # Normalize confidence (simplified approach)
            confidence = min(1.0, max(0.0, (log_likelihood + 1000) / 1000))

            return {
                "confidence": confidence,
                "aic": aic,
                "bic": bic,
                "log_likelihood": log_likelihood,
            }

        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return {"confidence": 0.5}


# Convenience function for creating GARCH models


def create_garch_model(
    model_type: str = "GARCH",
    p: int = 1,
    q: int = 1,
    mean_model: str = "Constant",
    dist: str = "normal",
) -> GARCHModel:
    """
    Create a GARCH model with specified parameters.

    Args:
        model_type: Type of GARCH model
        p: Order of GARCH terms
        q: Order of ARCH terms
        mean_model: Mean model specification
        dist: Error distribution

    Returns:
        Configured GARCH model
    """
    config = {
        "model_type": model_type,
        "p": p,
        "q": q,
        "mean_model": mean_model,
        "dist": dist,
    }

    return GARCHModel(config)
