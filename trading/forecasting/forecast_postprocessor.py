"""
Forecast Postprocessor

Handles post-processing of forecast results including NaN sanitization,
formatting, and preparation for plotting/saving.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ForecastPostprocessor:
    """Post-processes forecast results for plotting and saving."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the postprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.nan_strategy = self.config.get("nan_strategy", "forward_fill")
        self.min_valid_ratio = self.config.get("min_valid_ratio", 0.5)

    def sanitize_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize forecast DataFrame by handling NaN values.

        Args:
            df: Forecast DataFrame that may contain NaNs

        Returns:
            Sanitized DataFrame
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to sanitize_forecast")
            return pd.DataFrame()

        original_shape = df.shape
        nan_count = df.isnull().sum().sum()

        if nan_count == 0:
            logger.debug("No NaN values found, returning original DataFrame")
            return df

        logger.info(
            f"Found {nan_count} NaN values in forecast, applying {self.nan_strategy}"
        )

        # Apply NaN handling strategy
        if self.nan_strategy == "dropna":
            df_clean = df.dropna()
        elif self.nan_strategy == "forward_fill":
            df_clean = df.fillna(method="ffill").fillna(method="bfill")
        elif self.nan_strategy == "interpolate":
            df_clean = df.interpolate(method="linear").fillna(method="ffill")
        else:
            logger.warning(
                f"Unknown NaN strategy: {self.nan_strategy}, using forward_fill"
            )
            df_clean = df.fillna(method="ffill").fillna(method="bfill")

        # Check if too much data was lost
        valid_ratio = len(df_clean) / len(df)
        if valid_ratio < self.min_valid_ratio:
            logger.warning(f"Only {valid_ratio:.2%} of data remains after sanitization")

        logger.info(f"Sanitization complete: {original_shape} -> {df_clean.shape}")
        return df_clean

    def dynamic_ewma_smoothing(self, forecast: pd.Series) -> pd.Series:
        """
        Apply EWMA smoothing with dynamic alpha based on forecast volatility.
        alpha = min(0.3, 1 - std(forecast) / max(forecast))
        """
        if forecast.empty or forecast.max() == 0:
            alpha = 0.1
        else:
            alpha = min(0.3, 1 - forecast.std() / max(abs(forecast.max()), 1e-8))
        logger.info(f"Dynamic EWMA smoothing with alpha={alpha:.4f}")
        return forecast.ewm(alpha=alpha).mean()

    def prepare_for_plotting(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare forecast data for plotting.

        Args:
            forecast_data: Raw forecast data dictionary

        Returns:
            Processed data ready for plotting
        """
        processed_data = forecast_data.copy()

        # Sanitize forecast values
        if "forecast" in processed_data:
            if isinstance(processed_data["forecast"], pd.DataFrame):
                processed_data["forecast"] = self.sanitize_forecast(
                    processed_data["forecast"]
                )
                # Apply dynamic EWMA smoothing to first column if present
                col = processed_data["forecast"].columns[0]
                processed_data["forecast"][col] = self.dynamic_ewma_smoothing(
                    processed_data["forecast"][col]
                )
            elif isinstance(processed_data["forecast"], np.ndarray):
                # Convert to DataFrame for consistent handling
                df = pd.DataFrame(processed_data["forecast"])
                df = self.sanitize_forecast(df)
                col = df.columns[0]
                df[col] = self.dynamic_ewma_smoothing(df[col])
                processed_data["forecast"] = df

        # Sanitize confidence intervals if present
        if "confidence_intervals" in processed_data:
            ci_data = processed_data["confidence_intervals"]
            if isinstance(ci_data, pd.DataFrame):
                processed_data["confidence_intervals"] = self.sanitize_forecast(ci_data)

        return processed_data

    def prepare_for_saving(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare forecast data for saving to file.

        Args:
            forecast_data: Raw forecast data dictionary

        Returns:
            Processed data ready for saving
        """
        processed_data = forecast_data.copy()

        # Ensure all DataFrames are sanitized
        for key, value in processed_data.items():
            if isinstance(value, pd.DataFrame):
                processed_data[key] = self.sanitize_forecast(value)

        # Add metadata
        processed_data["postprocessing"] = {
            "nan_strategy": self.nan_strategy,
            "timestamp": pd.Timestamp.now().isoformat(),
            "version": "1.0",
        }

        return processed_data
