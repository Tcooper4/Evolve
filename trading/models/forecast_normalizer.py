"""
Forecast Normalizer

This module provides functionality to normalize forecast indexes and align them
with reference dataframes or datetime ranges for consistent analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NormalizedForecast:
    """Normalized forecast result."""

    original_forecast: pd.DataFrame
    normalized_forecast: pd.DataFrame
    alignment_method: str
    reference_index: pd.DatetimeIndex
    alignment_metadata: Dict[str, Any]


class ForecastNormalizer:
    """
    Forecast normalizer with comprehensive alignment capabilities.

    Features:
    - Align forecast indexes to datetime ranges
    - Normalize to reference DataFrame
    - Handle missing data and outliers
    - Support multiple alignment methods
    - Preserve forecast metadata
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the forecast normalizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Alignment configuration
        self.default_freq = self.config.get("default_freq", "D")
        self.interpolation_method = self.config.get("interpolation_method", "linear")
        self.fill_method = self.config.get("fill_method", "ffill")
        self.max_gap_days = self.config.get("max_gap_days", 7)

        # Validation settings
        self.min_forecast_length = self.config.get("min_forecast_length", 1)
        self.max_forecast_length = self.config.get("max_forecast_length", 365)

        # Storage for alignment history
        self.alignment_history: List[Dict[str, Any]] = []

    def align(
        self,
        forecast: Union[pd.DataFrame, pd.Series, np.ndarray],
        reference: Optional[Union[pd.DataFrame, pd.DatetimeIndex]] = None,
        method: str = "datetime",
        **kwargs,
    ) -> NormalizedForecast:
        """
        Align forecast to reference index or DataFrame.

        Args:
            forecast: Forecast data (DataFrame, Series, or array)
            reference: Reference DataFrame or DatetimeIndex for alignment
            method: Alignment method ('datetime', 'reference', 'auto')
            **kwargs: Additional alignment parameters

        Returns:
            NormalizedForecast object
        """
        try:
            # Convert forecast to DataFrame if needed
            forecast_df = self._ensure_dataframe(forecast)

            # Validate forecast
            self._validate_forecast(forecast_df)

            # Determine alignment method
            if method == "auto":
                method = self._determine_alignment_method(forecast_df, reference)

            # Perform alignment based on method
            if method == "datetime":
                normalized_df, reference_index = self._align_to_datetime(
                    forecast_df, reference, **kwargs
                )
            elif method == "reference":
                normalized_df, reference_index = self._align_to_reference(
                    forecast_df, reference, **kwargs
                )
            else:
                raise ValueError(f"Unknown alignment method: {method}")

            # Create metadata
            alignment_metadata = {
                "method": method,
                "original_shape": forecast_df.shape,
                "normalized_shape": normalized_df.shape,
                "alignment_timestamp": datetime.now().isoformat(),
                "config": self.config,
                **kwargs,
            }

            # Store alignment history
            self.alignment_history.append(alignment_metadata)

            result = NormalizedForecast(
                original_forecast=forecast_df,
                normalized_forecast=normalized_df,
                alignment_method=method,
                reference_index=reference_index,
                alignment_metadata=alignment_metadata,
            )

            self.logger.info(
                f"Forecast aligned using {method} method: "
                f"{forecast_df.shape} -> {normalized_df.shape}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Forecast alignment failed: {e}")
            raise

    def _ensure_dataframe(
        self, data: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """Convert input data to DataFrame format."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, pd.Series):
            return data.to_frame()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame(data, columns=["forecast"])
            else:
                return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _validate_forecast(self, forecast_df: pd.DataFrame) -> None:
        """Validate forecast data."""
        if forecast_df.empty:
            raise ValueError("Forecast data is empty")

        if len(forecast_df) < self.min_forecast_length:
            raise ValueError(
                f"Forecast too short: {len(forecast_df)} < {self.min_forecast_length}"
            )

        if len(forecast_df) > self.max_forecast_length:
            raise ValueError(
                f"Forecast too long: {len(forecast_df)} > {self.max_forecast_length}"
            )

        # Check for excessive NaN values
        nan_ratio = forecast_df.isna().sum().sum() / (
            forecast_df.shape[0] * forecast_df.shape[1]
        )
        if nan_ratio > 0.5:
            self.logger.warning(f"High NaN ratio in forecast: {nan_ratio:.2%}")

    def _determine_alignment_method(
        self,
        forecast_df: pd.DataFrame,
        reference: Optional[Union[pd.DataFrame, pd.DatetimeIndex]],
    ) -> str:
        """Automatically determine the best alignment method."""
        if reference is None:
            return "datetime"

        if isinstance(reference, pd.DatetimeIndex):
            return "datetime"
        elif isinstance(reference, pd.DataFrame):
            return "reference"
        else:
            return "datetime"

    def _align_to_datetime(
        self,
        forecast_df: pd.DataFrame,
        reference: Optional[Union[pd.DatetimeIndex, pd.DataFrame]] = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Align forecast to datetime index."""
        # Generate reference datetime index
        if reference is None:
            # Use default datetime range
            start_date = datetime.now()
            end_date = start_date + timedelta(days=len(forecast_df))
            reference_index = pd.date_range(
                start=start_date, end=end_date, freq=self.default_freq
            )
        elif isinstance(reference, pd.DatetimeIndex):
            reference_index = reference
        elif isinstance(reference, pd.DataFrame):
            reference_index = reference.index
        else:
            raise ValueError(
                f"Invalid reference type for datetime alignment: {type(reference)}"
            )

        # Ensure forecast has datetime index
        if not isinstance(forecast_df.index, pd.DatetimeIndex):
            # Create datetime index for forecast
            forecast_start = datetime.now()
            forecast_index = pd.date_range(
                start=forecast_start, periods=len(forecast_df), freq=self.default_freq
            )
            forecast_df = forecast_df.copy()
            forecast_df.index = forecast_index

        # Align to reference index
        aligned_df = forecast_df.reindex(
            reference_index, method=self.interpolation_method
        )

        # Fill any remaining NaN values
        aligned_df = aligned_df.fillna(method=self.fill_method)

        return aligned_df, reference_index

    def _align_to_reference(
        self, forecast_df: pd.DataFrame, reference: pd.DataFrame, **kwargs
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Align forecast to reference DataFrame."""
        if not isinstance(reference, pd.DataFrame):
            raise ValueError("Reference must be a DataFrame for reference alignment")

        reference_index = reference.index

        # Ensure forecast has datetime index
        if not isinstance(forecast_df.index, pd.DatetimeIndex):
            # Create datetime index for forecast
            forecast_start = (
                reference_index[0] if len(reference_index) > 0 else datetime.now()
            )
            forecast_index = pd.date_range(
                start=forecast_start, periods=len(forecast_df), freq=self.default_freq
            )
            forecast_df = forecast_df.copy()
            forecast_df.index = forecast_index

        # Align to reference index
        aligned_df = forecast_df.reindex(
            reference_index, method=self.interpolation_method
        )

        # Fill any remaining NaN values
        aligned_df = aligned_df.fillna(method=self.fill_method)

        return aligned_df, reference_index

    def normalize_multiple_forecasts(
        self,
        forecasts: Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray]],
        reference: Optional[Union[pd.DataFrame, pd.DatetimeIndex]] = None,
        method: str = "auto",
    ) -> Dict[str, NormalizedForecast]:
        """
        Normalize multiple forecasts to the same reference.

        Args:
            forecasts: Dictionary of forecasts to normalize
            reference: Reference for alignment
            method: Alignment method

        Returns:
            Dictionary of normalized forecasts
        """
        normalized_forecasts = {}

        for name, forecast in forecasts.items():
            try:
                normalized = self.align(forecast, reference, method)
                normalized_forecasts[name] = normalized
                self.logger.info(f"Normalized forecast '{name}' successfully")
            except Exception as e:
                self.logger.error(f"Failed to normalize forecast '{name}': {e}")
                continue

        return normalized_forecasts

    def get_alignment_statistics(self) -> Dict[str, Any]:
        """Get statistics about alignment operations."""
        if not self.alignment_history:
            return {"total_alignments": 0}

        methods_used = [h["method"] for h in self.alignment_history]
        method_counts = pd.Series(methods_used).value_counts().to_dict()

        return {
            "total_alignments": len(self.alignment_history),
            "method_distribution": method_counts,
            "last_alignment": self.alignment_history[-1]["alignment_timestamp"],
            "config": self.config,
        }

    def clear_history(self) -> None:
        """Clear alignment history."""
        self.alignment_history.clear()
        self.logger.info("Alignment history cleared")


def create_forecast_normalizer(
    config: Optional[Dict[str, Any]] = None,
) -> ForecastNormalizer:
    """Create a forecast normalizer instance."""
    return ForecastNormalizer(config)
