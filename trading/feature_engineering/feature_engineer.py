from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from trading.data.preprocessing import FeatureEngineering

# Try to import pandas_ta, with fallback
try:
    # Patch numpy for pandas_ta compatibility
    import numpy

    if not hasattr(numpy, "NaN"):
        numpy.NaN = numpy.nan
    if not hasattr(numpy, "float"):
        numpy.float = float
    if not hasattr(numpy, "int"):
        numpy.int = int

    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError as e:
    PANDAS_TA_AVAILABLE = False
    logging.warning(f"pandas_ta not available: {e}")
except Exception as e:
    PANDAS_TA_AVAILABLE = False
    logging.warning(f"pandas_ta import error: {e}")

import logging

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_regression
from sklearn.preprocessing import StandardScaler

from trading.feature_engineering import indicators
from utils.common_helpers import normalize_indicator_name

logger = logging.getLogger(__name__)


class FeatureEngineer(FeatureEngineering):
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature engineer with configuration.

        Args:
            config: Configuration dictionary for feature engineering
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance

        # Feature selection components
        self.variance_threshold = VarianceThreshold(threshold=0.01)  # Remove low variance features
        self.feature_selector = None
        self.selected_features = []

        # Feature selection configuration
        self.feature_selection_config = self.config.get(
            "feature_selection",
            {
                "enable_variance_threshold": True,
                "enable_recursive_elimination": True,
                "enable_k_best": True,
                "k_best_features": 50,
                "variance_threshold": 0.01,
                "rfe_n_features": 30,
            },
        )

        self.feature_columns = []
        # Dictionary of custom indicator functions registered by name
        self.custom_indicators: Dict[str, Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]]] = {}

        # Register built-in custom indicators
        try:
            self.register_custom_indicator(
                "ROLLING_ZSCORE",
                lambda df: indicators.rolling_zscore(df["close"], window=20),
            )
            self.register_custom_indicator("PRICE_RATIOS", indicators.price_ratios)
        except Exception as exc:  # pragma: no cover - log and continue
            logger.warning("Failed to register default indicators: %s", exc)

    def register_custom_indicator(
        self, name: str, func: Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]]
    ) -> None:
        """Register a custom indicator calculation.

        Args:
            name: Name of the indicator column or prefix
            func: Function that takes a DataFrame and returns a Series or DataFrame
        """
        self.custom_indicators[name] = func

    def apply_registered_indicator(self, name: str, df: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Apply a registered indicator by name.

        Args:
            name: Name of the indicator to apply
            df: Input DataFrame
            **kwargs: Additional arguments for the indicator

        Returns:
            Series or DataFrame with indicator values

        Raises:
            ValueError: If indicator name is not found in registry
        """
        # Check if indicator exists in registry
        if name not in indicators.INDICATOR_REGISTRY:
            raise ValueError(f"Indicator '{name}' not found in registry")

        try:
            # Get indicator function and apply it
            indicator_func = indicators.INDICATOR_REGISTRY[name]
            result = indicator_func(df, **kwargs)

            # Log success
            logger.info(f"Successfully applied indicator: {name}")

            return result

        except Exception as e:
            logger.error(f"Error applying indicator {name}: {str(e)}")
            raise

    def feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all available indicators.

        Returns:
            Dictionary mapping indicator names to descriptions
        """
        # Get descriptions from indicators module
        descriptions = indicators.get_indicator_descriptions()

        # Add descriptions for custom indicators
        for name, func in self.custom_indicators.items():
            if name not in descriptions:
                descriptions[name] = func.__doc__ or f"Custom indicator: {name}"

        return descriptions

    def select_features(self, features: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality.

        Args:
            features: Input features DataFrame
            target: Target variable for supervised feature selection

        Returns:
            DataFrame with selected features
        """
        try:
            selected_features = features.copy()

            # 1. Variance Threshold (remove low variance features)
            if self.feature_selection_config.get("enable_variance_threshold", True):
                logger.info("Applying variance threshold feature selection...")
                self.variance_threshold = VarianceThreshold(
                    threshold=self.feature_selection_config.get("variance_threshold", 0.01)
                )
                selected_features = pd.DataFrame(
                    self.variance_threshold.fit_transform(selected_features),
                    index=selected_features.index,
                    columns=selected_features.columns[self.variance_threshold.get_support()],
                )
                logger.info(
                    f"Variance threshold reduced features from {features.shape[1]} to {selected_features.shape[1]}"
                )

            # 2. Recursive Feature Elimination (if target is provided)
            if (
                self.feature_selection_config.get("enable_recursive_elimination", True)
                and target is not None
                and len(target) > 0
            ):
                logger.info("Applying recursive feature elimination...")
                try:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                    rfe = RFE(
                        estimator=estimator,
                        n_features_to_select=self.feature_selection_config.get("rfe_n_features", 30),
                    )
                    selected_features = pd.DataFrame(
                        rfe.fit_transform(selected_features, target),
                        index=selected_features.index,
                        columns=selected_features.columns[rfe.support_],
                    )
                    self.feature_selector = rfe
                    logger.info(f"RFE reduced features to {selected_features.shape[1]}")
                except Exception as e:
                    logger.warning(f"RFE failed: {e}")

            # 3. Select K Best features (if target is provided)
            elif self.feature_selection_config.get("enable_k_best", True) and target is not None and len(target) > 0:
                logger.info("Applying K-best feature selection...")
                try:
                    k_best = SelectKBest(
                        score_func=f_regression,
                        k=min(self.feature_selection_config.get("k_best_features", 50), selected_features.shape[1]),
                    )
                    selected_features = pd.DataFrame(
                        k_best.fit_transform(selected_features, target),
                        index=selected_features.index,
                        columns=selected_features.columns[k_best.get_support()],
                    )
                    self.feature_selector = k_best
                    logger.info(f"K-best reduced features to {selected_features.shape[1]}")
                except Exception as e:
                    logger.warning(f"K-best failed: {e}")

            # Store selected feature names
            self.selected_features = selected_features.columns.tolist()

            logger.info(f"Feature selection completed. Final features: {len(self.selected_features)}")
            return selected_features

        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return features

    def get_feature_importance(self, model=None) -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            model: Trained model with feature_importances_ attribute

        Returns:
            DataFrame with feature importance scores
        """
        try:
            if model and hasattr(model, "feature_importances_"):
                # Use model's feature importance
                importance_df = pd.DataFrame(
                    {
                        "feature": self.selected_features or self.feature_columns,
                        "importance": model.feature_importances_,
                    }
                )
            elif self.feature_selector and hasattr(self.feature_selector, "ranking_"):
                # Use RFE ranking
                importance_df = pd.DataFrame(
                    {"feature": self.selected_features, "ranking": self.feature_selector.ranking_}
                )
            elif self.feature_selector and hasattr(self.feature_selector, "scores_"):
                # Use K-best scores
                importance_df = pd.DataFrame(
                    {"feature": self.selected_features, "score": self.feature_selector.scores_}
                )
            else:
                # Fallback to variance scores
                features = self.selected_features or self.feature_columns
                importance_df = pd.DataFrame({"feature": features, "variance": [1.0] * len(features)})  # Placeholder

            return importance_df.sort_values(
                "importance"
                if "importance" in importance_df.columns
                else "ranking"
                if "ranking" in importance_df.columns
                else "score",
                ascending=False,
            )

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()

    def engineer_features(self, data: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Engineer all features from the input data with feature selection.

        Args:
            data: Input DataFrame with OHLCV data
            target: Optional target variable for supervised feature selection

        Returns:
            DataFrame with engineered features
        """
        required_cols = {"open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(map(str.lower, data.columns))
        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")

        features = pd.DataFrame(index=data.index)

        try:
            tech = self._calculate_technical_indicators(data)
            features = pd.concat([features, tech], axis=1)
        except Exception as exc:  # pragma: no cover - log and continue
            logger.error("Technical indicator calculation failed: %s", exc)

        try:
            stats_features = self._calculate_statistical_features(data)
            features = pd.concat([features, stats_features], axis=1)
        except Exception as exc:  # pragma: no cover - log and continue
            logger.error("Statistical feature calculation failed: %s", exc)

        try:
            micro = self._calculate_microstructure_features(data)
            features = pd.concat([features, micro], axis=1)
        except Exception as exc:  # pragma: no cover - log and continue
            logger.error("Microstructure feature calculation failed: %s", exc)

        try:
            time_feats = self._calculate_time_features(data)
            features = pd.concat([features, time_feats], axis=1)
        except Exception as exc:  # pragma: no cover - log and continue
            logger.error("Time feature calculation failed: %s", exc)

        # Fill NaN values
        features = features.fillna(method="ffill").fillna(0)

        # Apply feature selection to reduce dimensionality
        if self.feature_selection_config.get("enable_variance_threshold", True):
            features = self.select_features(features, target)

        # Scale features
        features = self._scale_features(features)

        # Verify indicators
        self._verify_indicators(features)

        return features

    def _verify_indicators(self, features: pd.DataFrame) -> None:
        """Verify that all indicators are being calculated correctly.

        Args:
            features: DataFrame containing all calculated features
        """
        expected_indicators = {
            # Trend Indicators
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "EMA_20",
            "EMA_50",
            "EMA_200",
            "MACD_12_26_9",
            "MACDh_12_26_9",
            "MACDs_12_26_9",
            "ADX_14",
            "ICHIMOKU_9_26_52",
            "ICHIMOKU_9_26_52_26",
            "ICHIMOKU_9_26_52_52",
            "PSAR_0.02_0.02_0.2",
            "SUPERT_10_3.0",
            # Momentum Indicators
            "RSI_14",
            "STOCH_14_3_3",
            "STOCHk_14_3_3",
            "STOCHd_14_3_3",
            "CCI_14",
            "WILLR_14",
            "MOM_10",
            "ROC_10",
            "MFI_14",
            "TRIX_18_9",
            "MASSI_9",
            "DPO_20",
            "KST_10_15_20_30_10_10_10_15",
            # Volatility Indicators
            "BBL_20_2.0",
            "BBM_20_2.0",
            "BBU_20_2.0",
            "BBB_20_2.0",
            "BBP_20_2.0",
            "ATR_14",
            "NATR_14",
            "TRUERANGE_1",
            # Volume Indicators
            "OBV",
            "VWAP",
            "PVT",
            "EFI_13",
            "CFI_14",
            "EBSW_10",
            # Custom Indicators
            "TSI_13_25",
            "UO_7_14_28",
            "AO_5_34",
            "BOP",
            "CMO_14",
            "PPO_12_26_9",
        }

        # Include user registered custom indicators
        expected_indicators.update(self.custom_indicators.keys())

        # Check for missing indicators
        missing_indicators = expected_indicators - set(features.columns)
        if missing_indicators:
            logger.warning(f"Missing indicators: {missing_indicators}")

        # Check for NaN values
        nan_columns = features.columns[features.isna().any()].tolist()
        if nan_columns:
            logger.warning(f"Columns with NaN values: {nan_columns}")

        # Check for infinite values
        inf_columns = features.columns[np.isinf(features).any()].tolist()
        if inf_columns:
            logger.warning(f"Columns with infinite values: {inf_columns}")

        # Check for extreme values that may indicate calculation errors
        extreme_columns = []
        for col in features.select_dtypes(include=[np.number]).columns:
            series = features[col]
            if series.abs().max() > 1e6:
                extreme_columns.append(col)
        if extreme_columns:
            logger.warning("Columns with extreme values: %s", extreme_columns)

        # Log summary
        logger.info(f"Total indicators calculated: {len(features.columns)}")
        logger.info(f"Expected indicators: {len(expected_indicators)}")
        logger.info(f"Missing indicators: {len(missing_indicators)}")
        logger.info(f"Columns with NaN values: {len(nan_columns)}")
        logger.info(f"Columns with infinite values: {len(inf_columns)}")

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using pandas_ta."""
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - {c.lower() for c in data.columns}
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        # Create a custom strategy
        custom_strategy = ta.Strategy(
            name="custom_strategy",
            description="Custom technical analysis strategy",
            ta=[
                # Trend Indicators
                {"kind": "sma", "length": 20},
                {"kind": "sma", "length": 50},
                {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 20},
                {"kind": "ema", "length": 50},
                {"kind": "ema", "length": 200},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "adx", "length": 14},
                {"kind": "ichimoku", "tenkan": 9, "kijun": 26, "senkou": 52},
                {"kind": "psar", "af0": 0.02, "af": 0.02, "max_af": 0.2},
                {"kind": "supertrend", "length": 10, "multiplier": 3},
                # Momentum Indicators
                {"kind": "rsi", "length": 14},
                {"kind": "stoch", "k": 14, "d": 3},
                {"kind": "cci", "length": 14},
                {"kind": "willr", "length": 14},
                {"kind": "mom", "length": 10},
                {"kind": "roc", "length": 10},
                {"kind": "mfi", "length": 14},
                {"kind": "trix", "length": 18, "signal": 9},
                {"kind": "massi", "length": 9},
                {"kind": "dpo", "length": 20},
                {
                    "kind": "kst",
                    "roc1": 10,
                    "roc2": 15,
                    "roc3": 20,
                    "roc4": 30,
                    "sma1": 10,
                    "sma2": 10,
                    "sma3": 10,
                    "sma4": 15,
                },
                # Volatility Indicators
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "atr", "length": 14},
                {"kind": "natr", "length": 14},
                {"kind": "tr"},
                {"kind": "true_range"},
                # Volume Indicators
                {"kind": "obv"},
                {"kind": "vwap"},
                {"kind": "pvt"},
                {"kind": "efi", "length": 13},
                {"kind": "cfi", "length": 14},
                {"kind": "ebsw", "length": 10},
                # Custom Indicators
                {"kind": "tsi", "fast": 13, "slow": 25},
                {"kind": "uo", "fast": 7, "medium": 14, "slow": 28},
                {"kind": "ao", "fast": 5, "slow": 34},
                {"kind": "bop"},
                {"kind": "cmo", "length": 14},
                {"kind": "ppo", "fast": 12, "slow": 26, "signal": 9},
            ],
        )

        # Add the strategy to the DataFrame
        data.ta.strategy(custom_strategy)

        # Get all the technical indicators
        features = data.ta.indicators()
        # Normalize indicator names for consistency
        features.rename(columns=lambda c: normalize_indicator_name(c), inplace=True)

        # Apply user registered custom indicators
        for name, func in self.custom_indicators.items():
            try:
                result = func(data)
                if isinstance(result, pd.Series):
                    features[name] = result
                elif isinstance(result, pd.DataFrame):
                    prefixed = result.add_prefix(f"{name}_")
                    features = pd.concat([features, prefixed], axis=1)
            except Exception as exc:
                logger.warning(f"Custom indicator '{name}' failed: {exc}")

        # Add basic price-based features
        features["returns"] = data["close"].pct_change()
        features["log_returns"] = np.log(data["close"] / data["close"].shift(1))
        features["volatility"] = features["returns"].rolling(window=20).std()

        return features

    def _calculate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical features."""
        features = pd.DataFrame(index=data.index)

        # Rolling statistics
        for window in [5, 10, 20]:
            features[f"rolling_mean_{window}"] = data["close"].rolling(window=window).mean()
            features[f"rolling_std_{window}"] = data["close"].rolling(window=window).std()
            features[f"rolling_skew_{window}"] = data["close"].rolling(window=window).skew()
            features[f"rolling_kurt_{window}"] = data["close"].rolling(window=window).kurt()

        return features

    def _calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features."""
        features = pd.DataFrame(index=data.index)

        # Bid-ask spread proxy
        features["spread"] = (data["high"] - data["low"]) / data["close"]

        # Volume profile
        features["volume_ma_ratio"] = data["volume"] / data["volume"].rolling(window=20).mean()

        # Price impact
        features["price_impact"] = features["returns"].abs() / data["volume"]

        # Order flow imbalance
        features["flow_imbalance"] = (data["close"] - data["open"]) / (data["high"] - data["low"])

        return features

    def _calculate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features."""
        features = pd.DataFrame(index=data.index)

        # Time of day
        features["hour"] = data.index.hour
        features["day_of_week"] = data.index.dayofweek
        features["month"] = data.index.month

        # Cyclical encoding
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

        return features

    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        # Fit scaler if not already fitted
        if not hasattr(self.scaler, "mean_"):
            self.scaler.fit(features)

        # Transform features
        scaled_features = pd.DataFrame(self.scaler.transform(features), index=features.index, columns=features.columns)

        return scaled_features

    def reduce_dimensions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reduce feature dimensions using PCA."""
        # Fit PCA if not already fitted
        if not hasattr(self.pca, "components_"):
            self.pca.fit(features)

        # Transform features
        reduced_features = pd.DataFrame(
            self.pca.transform(features),
            index=features.index,
            columns=[f"pc_{i+1}" for i in range(self.pca.n_components_)],
        )

        return reduced_features

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features."""
        df = data.copy()

        # Price-based features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Moving averages
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["SMA_200"] = df["close"].rolling(window=200).mean()

        # Volatility
        df["volatility"] = df["returns"].rolling(window=20).std()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df["BB_Middle"] = df["close"].rolling(window=20).mean()
        df["BB_Upper"] = df["BB_Middle"] + 2 * df["close"].rolling(window=20).std()
        df["BB_Lower"] = df["BB_Middle"] - 2 * df["close"].rolling(window=20).std()

        # Volume features
        df["volume_ma"] = df["volume"].rolling(window=20).mean()
        df["volume_std"] = df["volume"].rolling(window=20).std()

        # Price momentum
        df["momentum"] = df["close"] / df["close"].shift(10) - 1

        # Drop NaN values
        df = df.dropna()

        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in ["open", "high", "low", "close", "volume"]]

        return df

    def create_fundamental_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create fundamental analysis features."""
        df = data.copy()

        # Add fundamental features here
        # This is a placeholder for actual fundamental data

        return df

    def create_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment analysis features."""
        df = data.copy()

        # Add sentiment features here
        # This is a placeholder for actual sentiment data

        return df

    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        if not self.feature_columns:
            return data

        df = data.copy()
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        return df

    def create_target_variable(self, data: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Create target variable for prediction."""
        df = data.copy()
        df["target"] = df["close"].shift(-horizon) / df["close"] - 1
        return df

    def prepare_training_data(self, data: pd.DataFrame, target_col: str = "target") -> tuple:
        """Prepare data for training."""
        df = data.copy()

        # Drop rows with NaN values
        df = df.dropna()

        # Split features and target
        X = df[self.feature_columns]
        y = df[target_col]

        return X, y

    def get_feature_metrics(self) -> Dict[str, int]:
        """Get feature engineering metrics."""
        return {
            "num_features": len(self.feature_columns),
            "feature_types": {
                "technical": len([f for f in self.feature_columns if f.startswith(("SMA", "RSI", "MACD", "BB"))]),
                "fundamental": len([f for f in self.feature_columns if f.startswith(("PE", "PB", "ROE"))]),
                "sentiment": len([f for f in self.feature_columns if f.startswith(("sentiment", "emotion"))]),
            },
        }
