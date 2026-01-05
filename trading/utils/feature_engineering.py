"""
Feature engineering utilities for the trading system.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

# Try to import scikit-learn
try:
    from sklearn.feature_selection import (
        SelectKBest,
        f_regression,
        mutual_info_regression,
    )

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("⚠️ scikit-learn not available. Disabling feature selection capabilities.")
    print(f"   Missing: {e}")
    SelectKBest = None
    f_regression = None
    mutual_info_regression = None
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering utility class."""

    def __init__(self):
        self.feature_names = []
        self.feature_importance = {}
        self.is_fitted = False

    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        try:
            result = data.copy()

            # Price changes
            result["price_change"] = data["close"].pct_change()
            result["price_change_abs"] = data["close"].pct_change().abs()

            # Price ranges with safe division
            result["high_low_ratio"] = np.where(
                data["low"] > 1e-10,
                data["high"] / data["low"],
                1.0
            )
            result["open_close_ratio"] = np.where(
                data["close"] > 1e-10,
                data["open"] / data["close"],
                1.0
            )

            # Price momentum with safe division
            close_shifted_5 = data["close"].shift(5)
            result["momentum_5"] = np.where(
                close_shifted_5 > 1e-10,
                data["close"] / close_shifted_5 - 1,
                0.0
            )

            close_shifted_10 = data["close"].shift(10)
            result["momentum_10"] = np.where(
                close_shifted_10 > 1e-10,
                data["close"] / close_shifted_10 - 1,
                0.0
            )

            close_shifted_20 = data["close"].shift(20)
            result["momentum_20"] = np.where(
                close_shifted_20 > 1e-10,
                data["close"] / close_shifted_20 - 1,
                0.0
            )

            # Price acceleration
            result["acceleration"] = result["momentum_5"].diff()

            # Volatility features
            result["volatility_5"] = data["close"].rolling(window=5).std()
            result["volatility_10"] = data["close"].rolling(window=10).std()
            result["volatility_20"] = data["close"].rolling(window=20).std()

            # Clean up NaN values after lag/rolling operations
            result.bfill(inplace=True)

            return result
        except Exception as e:
            logger.error(f"Error creating price features: {e}")
            return data

    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        try:
            result = data.copy()

            # Volume changes
            result["volume_change"] = data["volume"].pct_change()
            result["volume_change_abs"] = data["volume"].pct_change().abs()

            # Volume moving averages
            result["volume_sma_5"] = data["volume"].rolling(window=5).mean()
            result["volume_sma_10"] = data["volume"].rolling(window=10).mean()
            result["volume_sma_20"] = data["volume"].rolling(window=20).mean()

            # Volume ratios with safe division
            result["volume_ratio_5"] = np.where(
                result["volume_sma_5"] > 1e-10,
                data["volume"] / result["volume_sma_5"],
                1.0
            )
            result["volume_ratio_10"] = np.where(
                result["volume_sma_10"] > 1e-10,
                data["volume"] / result["volume_sma_10"],
                1.0
            )
            result["volume_ratio_20"] = np.where(
                result["volume_sma_20"] > 1e-10,
                data["volume"] / result["volume_sma_20"],
                1.0
            )

            # Volume momentum with safe division
            volume_shifted_5 = data["volume"].shift(5)
            result["volume_momentum_5"] = np.where(
                volume_shifted_5 > 1e-10,
                data["volume"] / volume_shifted_5 - 1,
                0.0
            )

            volume_shifted_10 = data["volume"].shift(10)
            result["volume_momentum_10"] = np.where(
                volume_shifted_10 > 1e-10,
                data["volume"] / volume_shifted_10 - 1,
                0.0
            )

            # Clean up NaN values after lag/rolling operations
            result.bfill(inplace=True)

            return result
        except Exception as e:
            logger.error(f"Error creating volume features: {e}")
            return data

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        try:
            result = data.copy()

            # RSI features
            if "rsi" in data.columns:
                result["rsi_overbought"] = (data["rsi"] > 70).astype(int)
                result["rsi_oversold"] = (data["rsi"] < 30).astype(int)
                result["rsi_trend"] = data["rsi"].rolling(window=5).mean()

            # MACD features
            if "macd" in data.columns:
                result["macd_signal_cross"] = (
                    (data["macd"] > data["macd_signal"])
                    & (data["macd"].shift(1) <= data["macd_signal"].shift(1))
                ).astype(int)
                result["macd_negative_cross"] = (
                    (data["macd"] < data["macd_signal"])
                    & (data["macd"].shift(1) >= data["macd_signal"].shift(1))
                ).astype(int)

            # Bollinger Bands features
            if all(
                col in data.columns for col in ["bb_upper", "bb_middle", "bb_lower"]
            ):
                # Safe Bollinger Band calculations using safe_divide
                from trading.utils.safe_math import safe_divide
                
                bb_range = data["bb_upper"] - data["bb_lower"]
                result["bb_position"] = safe_divide(
                    data["close"] - data["bb_lower"],
                    bb_range,
                    default=0.5
                )
                # Clamp to [0, 1] range
                result["bb_position"] = result["bb_position"].clip(0.0, 1.0)

                result["bb_squeeze"] = np.where(
                    data["bb_middle"] > 1e-10,
                    bb_range / data["bb_middle"],
                    0.0
                )
                result["bb_breakout_up"] = (data["close"] > data["bb_upper"]).astype(
                    int
                )
                result["bb_breakout_down"] = (data["close"] < data["bb_lower"]).astype(
                    int
                )

            # Moving average features
            if "sma_20" in data.columns and "sma_50" in data.columns:
                result["sma_cross"] = (data["sma_20"] > data["sma_50"]).astype(int)
                # Safe SMA distance calculation
                result["sma_distance"] = np.where(
                    data["sma_50"] > 1e-10,
                    (data["sma_20"] - data["sma_50"]) / data["sma_50"],
                    0.0
                )

            # Clean up NaN values after lag/rolling operations
            result.bfill(inplace=True)

            return result
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return data

    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        try:
            result = data.copy()

            # Extract datetime components
            if isinstance(data.index, pd.DatetimeIndex):
                result["day_of_week"] = data.index.dayofweek
                result["day_of_month"] = data.index.day
                result["month"] = data.index.month
                result["quarter"] = data.index.quarter
                result["year"] = data.index.year

                # Cyclical encoding
                result["day_of_week_sin"] = np.sin(
                    2 * np.pi * result["day_of_week"] / 7
                )
                result["day_of_week_cos"] = np.cos(
                    2 * np.pi * result["day_of_week"] / 7
                )
                result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
                result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

            return result
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            return data

    def create_lag_features(
        self, data: pd.DataFrame, columns: List[str], lags: List[int]
    ) -> pd.DataFrame:
        """Create lagged features."""
        try:
            result = data.copy()

            for column in columns:
                if column in data.columns:
                    for lag in lags:
                        result[f"{column}_lag_{lag}"] = data[column].shift(lag)

            # Clean up NaN values after lag/rolling operations
            result.bfill(inplace=True)

            return result
        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
            return data

    def create_rolling_features(
        self,
        data: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        functions: List[str],
    ) -> pd.DataFrame:
        """Create rolling window features."""
        try:
            result = data.copy()

            for column in columns:
                if column in data.columns:
                    for window in windows:
                        for func in functions:
                            if func == "mean":
                                result[f"{column}_rolling_mean_{window}"] = (
                                    data[column].rolling(window=window).mean()
                                )
                            elif func == "std":
                                result[f"{column}_rolling_std_{window}"] = (
                                    data[column].rolling(window=window).std()
                                )
                            elif func == "min":
                                result[f"{column}_rolling_min_{window}"] = (
                                    data[column].rolling(window=window).min()
                                )
                            elif func == "max":
                                result[f"{column}_rolling_max_{window}"] = (
                                    data[column].rolling(window=window).max()
                                )
                            elif func == "median":
                                result[f"{column}_rolling_median_{window}"] = (
                                    data[column].rolling(window=window).median()
                                )

            # Clean up NaN values after lag/rolling operations
            result.bfill(inplace=True)

            return result
        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
            return data

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info", k: int = 10
    ) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Returning original features.")
            return X
        """Select the best features using statistical methods."""
        try:
            # Remove NaN values
            X_clean = X.dropna()
            y_clean = y[X_clean.index]

            if method == "mutual_info":
                selector = SelectKBest(
                    score_func=mutual_info_regression, k=min(k, X_clean.shape[1])
                )
            elif method == "f_regression":
                selector = SelectKBest(
                    score_func=f_regression, k=min(k, X_clean.shape[1])
                )
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                return X

            # Fit and transform
            X_selected = selector.fit_transform(X_clean, y_clean)

            # Get selected feature names
            selected_features = X_clean.columns[selector.get_support()].tolist()

            # Create result DataFrame
            result = pd.DataFrame(
                X_selected, columns=selected_features, index=X_clean.index
            )

            # Store feature importance
            self.feature_importance = dict(zip(selected_features, selector.scores_))
            self.feature_names = selected_features
            self.is_fitted = True

            return result
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return X

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.copy()

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        return self.feature_names.copy()


def create_all_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create all available features for the dataset."""
    try:
        engineer = FeatureEngineer()

        # Create different types of features
        result = engineer.create_price_features(data)
        result = engineer.create_volume_features(result)
        result = engineer.create_technical_features(result)
        result = engineer.create_time_features(result)

        # Create lag features for price and volume
        price_volume_cols = ["close", "volume", "price_change"]
        result = engineer.create_lag_features(
            result, price_volume_cols, [1, 2, 3, 5, 10]
        )

        # Create rolling features
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        result = engineer.create_rolling_features(
            result, numeric_cols[:5], [5, 10, 20], ["mean", "std"]
        )

        return result
    except Exception as e:
        logger.error(f"Error creating all features: {e}")
        return data
