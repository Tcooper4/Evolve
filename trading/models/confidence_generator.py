"""
Confidence Generator

This module provides functionality to generate confidence bands and intervals
for forecasts with support for multiple confidence levels and visualization.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBand:
    """Confidence band data structure."""

    level: float  # Confidence level (e.g., 0.95 for 95%)
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    mean_forecast: np.ndarray
    method: str
    metadata: Dict[str, Any]


@dataclass
class ConfidenceBands:
    """Collection of confidence bands for visualization."""

    bands: Dict[float, ConfidenceBand]
    forecast_dates: pd.DatetimeIndex
    original_forecast: np.ndarray
    generation_timestamp: datetime
    metadata: Dict[str, Any]


class ConfidenceGenerator:
    """
    Confidence generator with support for multiple confidence levels and methods.

    Features:
    - Multiple confidence levels (50%, 75%, 95%, etc.)
    - Various confidence calculation methods
    - Time series visualization support
    - Historical volatility-based bands
    - Model uncertainty quantification
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the confidence generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Default confidence levels
        self.default_levels = self.config.get("default_levels", [0.50, 0.75, 0.95])
        self.confidence_level = self.config.get("confidence_level", 0.95)

        # Method configuration
        self.default_method = self.config.get("default_method", "bootstrap")
        self.volatility_window = self.config.get("volatility_window", 30)
        self.bootstrap_samples = self.config.get("bootstrap_samples", 1000)

        # Visualization settings
        self.store_for_visualization = self.config.get("store_for_visualization", True)
        self.visualization_bands = {}

        # Storage for generated bands
        self.generated_bands: List[ConfidenceBands] = []

    def generate_confidence_bands(
        self,
        forecast: Union[np.ndarray, pd.Series, pd.DataFrame],
        historical_data: Optional[pd.DataFrame] = None,
        confidence_levels: Optional[List[float]] = None,
        method: Optional[str] = None,
        **kwargs,
    ) -> ConfidenceBands:
        """
        Generate confidence bands for a forecast.

        Args:
            forecast: Forecast values
            historical_data: Historical data for volatility calculation
            confidence_levels: List of confidence levels to generate
            method: Method for confidence calculation
            **kwargs: Additional parameters

        Returns:
            ConfidenceBands object with all requested bands
        """
        try:
            # Convert forecast to numpy array
            forecast_array = self._ensure_array(forecast)

            # Set defaults
            confidence_levels = confidence_levels or self.default_levels
            method = method or self.default_method

            # Generate forecast dates if not provided
            forecast_dates = self._generate_forecast_dates(forecast_array, **kwargs)

            # Generate bands for each confidence level
            bands = {}
            for level in confidence_levels:
                try:
                    band = self._generate_single_band(
                        forecast_array, historical_data, level, method, **kwargs
                    )
                    bands[level] = band
                    self.logger.info(f"Generated {level * 100:.0f}% confidence band")
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate {level * 100:.0f}% band: {e}"
                    )
                    continue

            # Create confidence bands object
            confidence_bands = ConfidenceBands(
                bands=bands,
                forecast_dates=forecast_dates,
                original_forecast=forecast_array,
                generation_timestamp=datetime.now(),
                metadata={
                    "method": method,
                    "confidence_levels": confidence_levels,
                    "config": self.config,
                    **kwargs,
                },
            )

            # Store for visualization if enabled
            if self.store_for_visualization:
                self.visualization_bands[datetime.now().isoformat()] = confidence_bands

            # Add to history
            self.generated_bands.append(confidence_bands)

            self.logger.info(
                f"Generated confidence bands for {len(confidence_levels)} levels "
                f"using {method} method"
            )

            return confidence_bands

        except Exception as e:
            self.logger.error(f"Confidence band generation failed: {e}")
            raise

    def _ensure_array(
        self, data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        """Convert input data to numpy array."""
        if isinstance(data, np.ndarray):
            return data.flatten()
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, pd.DataFrame):
            return data.iloc[:, 0].values  # Use first column
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _generate_forecast_dates(
        self,
        forecast_array: np.ndarray,
        start_date: Optional[datetime] = None,
        freq: str = "D",
        **kwargs,
    ) -> pd.DatetimeIndex:
        """Generate forecast dates."""
        if start_date is None:
            start_date = datetime.now()

        return pd.date_range(start=start_date, periods=len(forecast_array), freq=freq)

    def _generate_single_band(
        self,
        forecast: np.ndarray,
        historical_data: Optional[pd.DataFrame],
        confidence_level: float,
        method: str,
        **kwargs,
    ) -> ConfidenceBand:
        """Generate a single confidence band."""
        if method == "bootstrap":
            return self._bootstrap_confidence_band(forecast, confidence_level, **kwargs)
        elif method == "volatility":
            return self._volatility_confidence_band(
                forecast, historical_data, confidence_level, **kwargs
            )
        elif method == "parametric":
            return self._parametric_confidence_band(
                forecast, confidence_level, **kwargs
            )
        elif method == "quantile":
            return self._quantile_confidence_band(forecast, confidence_level, **kwargs)
        else:
            raise ValueError(f"Unknown confidence method: {method}")

    def _bootstrap_confidence_band(
        self, forecast: np.ndarray, confidence_level: float, **kwargs
    ) -> ConfidenceBand:
        """Generate confidence band using bootstrap method."""
        n_samples = kwargs.get("bootstrap_samples", self.bootstrap_samples)

        # Generate bootstrap samples
        bootstrap_samples = []
        for _ in range(n_samples):
            # Add random noise to forecast
            noise = np.random.normal(0, np.std(forecast) * 0.1, len(forecast))
            sample = forecast + noise
            bootstrap_samples.append(sample)

        bootstrap_array = np.array(bootstrap_samples)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_array, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_array, upper_percentile, axis=0)
        mean_forecast = np.mean(bootstrap_array, axis=0)

        return ConfidenceBand(
            level=confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mean_forecast=mean_forecast,
            method="bootstrap",
            metadata={"bootstrap_samples": n_samples, "alpha": alpha},
        )

    def _volatility_confidence_band(
        self,
        forecast: np.ndarray,
        historical_data: pd.DataFrame,
        confidence_level: float,
        **kwargs,
    ) -> ConfidenceBand:
        """Generate confidence band using historical volatility."""
        if historical_data is None:
            raise ValueError("Historical data required for volatility method")

        # Calculate historical volatility
        if "close" in historical_data.columns:
            returns = historical_data["close"].pct_change().dropna()
        else:
            returns = historical_data.iloc[:, 0].pct_change().dropna()

        volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]

        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * volatility * np.sqrt(np.arange(1, len(forecast) + 1))

        lower_bound = forecast - margin
        upper_bound = forecast + margin

        return ConfidenceBand(
            level=confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mean_forecast=forecast,
            method="volatility",
            metadata={
                "volatility": volatility,
                "z_score": z_score,
                "volatility_window": self.volatility_window,
            },
        )

    def _parametric_confidence_band(
        self, forecast: np.ndarray, confidence_level: float, **kwargs
    ) -> ConfidenceBand:
        """Generate confidence band using parametric method."""
        # Assume normal distribution
        forecast_std = np.std(forecast)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        margin = z_score * forecast_std

        lower_bound = forecast - margin
        upper_bound = forecast + margin

        return ConfidenceBand(
            level=confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mean_forecast=forecast,
            method="parametric",
            metadata={"forecast_std": forecast_std, "z_score": z_score},
        )

    def _quantile_confidence_band(
        self, forecast: np.ndarray, confidence_level: float, **kwargs
    ) -> ConfidenceBand:
        """Generate confidence band using quantile method."""
        # Calculate quantiles based on forecast distribution
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        # Use forecast as mean and calculate bounds
        forecast_std = np.std(forecast)
        lower_bound = np.quantile(forecast, lower_quantile) - forecast_std
        upper_bound = np.quantile(forecast, upper_quantile) + forecast_std

        # Extend bounds to match forecast length
        lower_bound = np.full_like(forecast, lower_bound)
        upper_bound = np.full_like(forecast, upper_bound)

        return ConfidenceBand(
            level=confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mean_forecast=forecast,
            method="quantile",
            metadata={
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
            },
        )

    def get_visualization_data(self, band_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get confidence bands data for visualization.

        Args:
            band_id: Specific band ID to retrieve

        Returns:
            Dictionary with visualization data
        """
        if band_id is None:
            # Return most recent bands
            if not self.visualization_bands:
                return {}
            band_id = max(self.visualization_bands.keys())

        if band_id not in self.visualization_bands:
            return {}

        bands = self.visualization_bands[band_id]

        # Prepare data for visualization
        viz_data = {
            "dates": bands.forecast_dates,
            "original_forecast": bands.original_forecast,
            "bands": {},
        }

        for level, band in bands.bands.items():
            viz_data["bands"][f"{level * 100:.0f}%"] = {
                "lower": band.lower_bound,
                "upper": band.upper_bound,
                "mean": band.mean_forecast,
                "method": band.method,
            }

        return viz_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated confidence bands."""
        if not self.generated_bands:
            return {"total_generations": 0}

        total_bands = sum(len(bands.bands) for bands in self.generated_bands)
        methods_used = []
        levels_used = []

        for bands in self.generated_bands:
            for band in bands.bands.values():
                methods_used.append(band.method)
                levels_used.append(band.level)

        return {
            "total_generations": len(self.generated_bands),
            "total_bands": total_bands,
            "methods_used": pd.Series(methods_used).value_counts().to_dict(),
            "levels_used": sorted(list(set(levels_used))),
            "last_generation": self.generated_bands[
                -1
            ].generation_timestamp.isoformat(),
        }

    def clear_history(self) -> None:
        """Clear generation history."""
        self.generated_bands.clear()
        self.visualization_bands.clear()
        self.logger.info("Confidence generation history cleared")


def create_confidence_generator(
    config: Optional[Dict[str, Any]] = None,
) -> ConfidenceGenerator:
    """Create a confidence generator instance."""
    return ConfidenceGenerator(config)
