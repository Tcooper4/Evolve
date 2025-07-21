"""
Feature Generator

Generates features for machine learning models with comprehensive logging
and NaN handling.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""

    name: str
    description: str
    category: str
    dependencies: List[str]
    parameters: Dict[str, Any]
    validation_rules: Dict[str, Any]
    is_required: bool = False
    is_custom: bool = False


class FeatureGenerator:
    """Generates features for machine learning models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_cache = {}
        self.feature_configs = self._initialize_feature_configs()
        self.dropped_features = []
        self.logger = logging.getLogger(__name__)

    def _initialize_feature_configs(self) -> Dict[str, FeatureConfig]:
        """Initialize feature configurations."""
        return {
            "rsi": FeatureConfig(
                name="rsi",
                description="Relative Strength Index",
                category="momentum",
                dependencies=["Close"],
                parameters={"period": 14},
                validation_rules={"min_value": 0, "max_value": 100},
            ),
            "macd": FeatureConfig(
                name="macd",
                description="Moving Average Convergence Divergence",
                category="trend",
                dependencies=["Close"],
                parameters={"fast": 12, "slow": 26, "signal": 9},
                validation_rules={"min_value": None, "max_value": None},
            ),
            "bollinger_upper": FeatureConfig(
                name="bollinger_upper",
                description="Bollinger Bands Upper",
                category="volatility",
                dependencies=["Close"],
                parameters={"period": 20, "std": 2},
                validation_rules={"min_value": 0},
            ),
            "bollinger_lower": FeatureConfig(
                name="bollinger_lower",
                description="Bollinger Bands Lower",
                category="volatility",
                dependencies=["Close"],
                parameters={"period": 20, "std": 2},
                validation_rules={"min_value": 0},
            ),
            "sma": FeatureConfig(
                name="sma",
                description="Simple Moving Average",
                category="trend",
                dependencies=["Close"],
                parameters={"period": 20},
                validation_rules={"min_value": 0},
            ),
            "ema": FeatureConfig(
                name="ema",
                description="Exponential Moving Average",
                category="trend",
                dependencies=["Close"],
                parameters={"period": 20},
                validation_rules={"min_value": 0},
            ),
            "volatility": FeatureConfig(
                name="volatility",
                description="Rolling Volatility",
                category="volatility",
                dependencies=["Close"],
                parameters={"period": 20},
                validation_rules={"min_value": 0},
            ),
            "volume_sma": FeatureConfig(
                name="volume_sma",
                description="Volume Simple Moving Average",
                category="volume",
                dependencies=["Volume"],
                parameters={"period": 20},
                validation_rules={"min_value": 0},
            ),
            "price_momentum": FeatureConfig(
                name="price_momentum",
                description="Price Momentum",
                category="momentum",
                dependencies=["Close"],
                parameters={"period": 10},
                validation_rules={"min_value": None, "max_value": None},
            ),
            "volume_momentum": FeatureConfig(
                name="volume_momentum",
                description="Volume Momentum",
                category="volume",
                dependencies=["Volume"],
                parameters={"period": 10},
                validation_rules={"min_value": None, "max_value": None},
            ),
        }

    def generate_features(
        self,
        data: pd.DataFrame,
        feature_list: Optional[List[str]] = None,
        handle_nans: str = "drop",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate features from input data.

        Args:
            data: Input DataFrame with OHLCV data
            feature_list: List of features to generate (None for all)
            handle_nans: How to handle NaNs ('drop', 'fill', 'interpolate')

        Returns:
            Tuple of (DataFrame with features, metadata about dropped features)
        """
        self.logger.info(f"Generating features for {len(data)} rows")
        self.dropped_features = []

        # Use all features if none specified
        if feature_list is None:
            feature_list = list(self.feature_configs.keys())

        # Validate input data
        self._validate_input_data(data, feature_list)

        # Generate features
        feature_df = data.copy()

        for feature_name in feature_list:
            if feature_name not in self.feature_configs:
                self.logger.warning(f"Unknown feature: {feature_name}")
                continue

            try:
                feature_series = self._generate_single_feature(data, feature_name)

                # Check for NaNs in generated feature
                nan_count = feature_series.isna().sum()
                if nan_count > 0:
                    self.logger.warning(
                        f"Feature {feature_name} has {nan_count} NaN values"
                    )

                    if handle_nans == "drop":
                        # Log dropped feature
                        self.dropped_features.append(
                            {
                                "feature_name": feature_name,
                                "nan_count": nan_count,
                                "total_count": len(feature_series),
                                "nan_percentage": nan_count / len(feature_series) * 100,
                                "reason": "NaN values detected",
                            }
                        )
                        self.logger.info(
                            f"Dropped feature {feature_name} due to {nan_count} NaN values"
                        )
                        continue
                    elif handle_nans == "fill":
                        feature_series = feature_series.fillna(method="ffill").fillna(
                            method="bfill"
                        )
                    elif handle_nans == "interpolate":
                        feature_series = feature_series.interpolate()

                feature_df[feature_name] = feature_series
                self.logger.debug(f"Generated feature: {feature_name}")

            except Exception as e:
                self.logger.error(f"Error generating feature {feature_name}: {e}")
                self.dropped_features.append(
                    {
                        "feature_name": feature_name,
                        "nan_count": 0,
                        "total_count": len(data),
                        "nan_percentage": 0,
                        "reason": f"Generation error: {str(e)}",
                    }
                )
                continue

        # Log summary
        self._log_feature_generation_summary(feature_df, feature_list)

        # Create metadata
        metadata = {
            "total_features_requested": len(feature_list),
            "features_generated": len(
                [f for f in feature_list if f in feature_df.columns]
            ),
            "features_dropped": len(self.dropped_features),
            "dropped_features": self.dropped_features,
            "nan_handling_method": handle_nans,
        }

        return feature_df, metadata

    def _generate_single_feature(
        self, data: pd.DataFrame, feature_name: str
    ) -> pd.Series:
        """Generate a single feature."""
        config = self.feature_configs[feature_name]

        if feature_name == "rsi":
            return self._calculate_rsi(data["Close"], config.parameters["period"])
        elif feature_name == "macd":
            return self._calculate_macd(data["Close"], config.parameters)
        elif feature_name == "bollinger_upper":
            return self._calculate_bollinger_upper(data["Close"], config.parameters)
        elif feature_name == "bollinger_lower":
            return self._calculate_bollinger_lower(data["Close"], config.parameters)
        elif feature_name == "sma":
            return self._calculate_sma(data["Close"], config.parameters["period"])
        elif feature_name == "ema":
            return self._calculate_ema(data["Close"], config.parameters["period"])
        elif feature_name == "volatility":
            return self._calculate_volatility(
                data["Close"], config.parameters["period"]
            )
        elif feature_name == "volume_sma":
            return self._calculate_sma(data["Volume"], config.parameters["period"])
        elif feature_name == "price_momentum":
            return self._calculate_momentum(data["Close"], config.parameters["period"])
        elif feature_name == "volume_momentum":
            return self._calculate_momentum(data["Volume"], config.parameters["period"])
        else:
            raise ValueError(f"Unknown feature: {feature_name}")

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, params: Dict[str, int]) -> pd.Series:
        """Calculate MACD."""
        fast_ema = prices.ewm(span=params["fast"]).mean()
        slow_ema = prices.ewm(span=params["slow"]).mean()
        macd = fast_ema - slow_ema
        return macd

    def _calculate_bollinger_upper(
        self, prices: pd.Series, params: Dict[str, int]
    ) -> pd.Series:
        """Calculate Bollinger Bands Upper."""
        sma = prices.rolling(window=params["period"]).mean()
        std = prices.rolling(window=params["period"]).std()
        return sma + (std * params["std"])

    def _calculate_bollinger_lower(
        self, prices: pd.Series, params: Dict[str, int]
    ) -> pd.Series:
        """Calculate Bollinger Bands Lower."""
        sma = prices.rolling(window=params["period"]).mean()
        std = prices.rolling(window=params["period"]).std()
        return sma - (std * params["std"])

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period).mean()

    def _calculate_volatility(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate rolling volatility."""
        returns = prices.pct_change()
        return returns.rolling(window=period).std()

    def _calculate_momentum(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate momentum."""
        return series / series.shift(period) - 1

    def _validate_input_data(self, data: pd.DataFrame, feature_list: List[str]):
        """Validate input data for feature generation."""
        required_columns = set()

        for feature_name in feature_list:
            if feature_name in self.feature_configs:
                required_columns.update(self.feature_configs[feature_name].dependencies)

        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(data) < 50:
            self.logger.warning(f"Small dataset: {len(data)} rows")

    def _log_feature_generation_summary(
        self, feature_df: pd.DataFrame, feature_list: List[str]
    ):
        """Log summary of feature generation."""
        generated_features = [f for f in feature_list if f in feature_df.columns]

        self.logger.info(f"Feature generation summary:")
        self.logger.info(f"  - Requested features: {len(feature_list)}")
        self.logger.info(f"  - Generated features: {len(generated_features)}")
        self.logger.info(f"  - Dropped features: {len(self.dropped_features)}")

        if self.dropped_features:
            self.logger.info("Dropped features:")
            for dropped in self.dropped_features:
                self.logger.info(f"  - {dropped['feature_name']}: {dropped['reason']}")

    def get_dropped_features(self) -> List[Dict[str, Any]]:
        """Get list of dropped features with reasons."""
        return self.dropped_features.copy()

    def add_custom_feature(self, name: str, config: FeatureConfig):
        """Add a custom feature configuration."""
        self.feature_configs[name] = config
        self.logger.info(f"Added custom feature: {name}")

    def get_feature_info(self, feature_name: str) -> Optional[FeatureConfig]:
        """Get information about a feature."""
        return self.feature_configs.get(feature_name)

    def list_available_features(self) -> List[str]:
        """List all available features."""
        return list(self.feature_configs.keys())

    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()
        self.logger.info("Feature cache cleared")
