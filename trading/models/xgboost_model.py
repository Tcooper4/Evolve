"""
XGBoost Model for Time Series Forecasting

This module provides an XGBoost-based model for time series forecasting
in the Evolve trading system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost-based time series forecasting model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)

        # Default configuration
        self.config = config or {}

        # Load hyperparameters from config file or use defaults
        self.model_params = self._load_hyperparameters()

        self.model = None
        self.is_trained = False
        self.feature_names = None

    def _load_hyperparameters(self) -> Dict[str, Any]:
        """Load XGBoost hyperparameters from config file or environment variables."""
        import os

        import yaml

        # Default hyperparameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }

        # Try to load from config file
        config_paths = [
            "config/xgboost_config.yaml",
            "config/models/xgboost.yaml",
            "trading/config/xgboost.yaml",
        ]

        for config_path in config_paths:
            try:
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_data = yaml.safe_load(f)
                        if config_data and "xgboost" in config_data:
                            logger.info(f"Loaded XGBoost config from {config_path}")
                            return {**default_params, **config_data["xgboost"]}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Try to load from environment variables
        env_params = {}
        env_mappings = {
            "XGB_N_ESTIMATORS": "n_estimators",
            "XGB_MAX_DEPTH": "max_depth",
            "XGB_LEARNING_RATE": "learning_rate",
            "XGB_SUBSAMPLE": "subsample",
            "XGB_COLSAMPLE_BYTREE": "colsample_bytree",
            "XGB_RANDOM_STATE": "random_state",
        }

        for env_var, param_name in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                try:
                    # Convert to appropriate type
                    if param_name in ["n_estimators", "max_depth", "random_state"]:
                        env_params[param_name] = int(env_value)
                    else:
                        env_params[param_name] = float(env_value)
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {env_value}")

        if env_params:
            logger.info("Loaded XGBoost parameters from environment variables")
            return {**default_params, **env_params}

        logger.info("Using default XGBoost parameters")
        return default_params

    def _setup_model(self):
        """Setup XGBoost model with config-driven parameters."""
        try:
            import xgboost as xgb

            # Use the loaded hyperparameters
            self.model = xgb.XGBRegressor(**self.model_params)
            logger.info(
                f"XGBoost model initialized with parameters: {self.model_params}"
            )
        except ImportError as e:
            logger.error(f"XGBoost not available: {e}")
            raise ImportError(
                "XGBoost is required for this model. Install with: pip install xgboost"
            )

    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for XGBoost model.

        Args:
            data: Input time series data

        Returns:
            Tuple of (features, target)
        """
        try:
            # Use auto feature engineering if enabled
            if self.config.get("auto_feature_engineering", False):
                return self._auto_engineer_features(data)

            # Create lag features
            lags = [1, 2, 3, 5, 10]
            features = pd.DataFrame()

            for lag in lags:
                features[f"lag_{lag}"] = data["close"].shift(lag)

            # Add technical indicators
            features["sma_5"] = data["close"].rolling(5).mean()
            features["sma_20"] = data["close"].rolling(20).mean()
            features["rsi"] = self._calculate_rsi(data["close"])
            features["volatility"] = data["close"].rolling(20).std()

            # Add time features
            features["day_of_week"] = data.index.dayofweek
            features["month"] = data.index.month

            # Target variable
            target = data["close"].shift(-1)  # Next day's price

            # Remove NaN values
            features = features.dropna()
            target = target[features.index]

            self.feature_names = features.columns.tolist()

            return features, target

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def _auto_engineer_features(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Automatically engineer features using cross-validation to find optimal lags."""
        try:
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import TimeSeriesSplit

            logger.info("Starting auto feature engineering...")

            # Define candidate lag ranges
            lag_ranges = {
                "short": list(range(1, 6)),
                "medium": list(range(5, 16, 2)),
                "long": list(range(15, 31, 5)),
            }

            # Define candidate rolling windows
            rolling_windows = [5, 10, 20, 30]

            best_features = None
            best_score = float("inf")

            # Cross-validation setup
            tscv = TimeSeriesSplit(n_splits=3)

            # Test different feature combinations
            for lag_type, lags in lag_ranges.items():
                for window in rolling_windows:
                    # Create feature set
                    features = pd.DataFrame()

                    # Add lag features
                    for lag in lags:
                        features[f"lag_{lag}"] = data["close"].shift(lag)

                    # Add rolling features
                    features[f"sma_{window}"] = data["close"].rolling(window).mean()
                    features[f"std_{window}"] = data["close"].rolling(window).std()
                    features[f"min_{window}"] = data["close"].rolling(window).min()
                    features[f"max_{window}"] = data["close"].rolling(window).max()

                    # Add technical indicators
                    features["rsi"] = self._calculate_rsi(data["close"])

                    # Add time features
                    features["day_of_week"] = data.index.dayofweek
                    features["month"] = data.index.month

                    # Target variable
                    target = data["close"].shift(-1)

                    # Remove NaN values
                    features_clean = features.dropna()
                    target_clean = target[features_clean.index]

                    if len(features_clean) < 50:
                        continue

                    # Evaluate with cross-validation
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(features_clean):
                        X_train, X_val = (
                            features_clean.iloc[train_idx],
                            features_clean.iloc[val_idx],
                        )
                        y_train, y_val = (
                            target_clean.iloc[train_idx],
                            target_clean.iloc[val_idx],
                        )

                        # Quick XGBoost fit
                        import xgboost as xgb

                        model = xgb.XGBRegressor(n_estimators=50, random_state=42)
                        model.fit(X_train, y_train)

                        # Predict and score
                        y_pred = model.predict(X_val)
                        score = mean_squared_error(y_val, y_pred)
                        cv_scores.append(score)

                    avg_score = np.mean(cv_scores)

                    if avg_score < best_score:
                        best_score = avg_score
                        best_features = features_clean.copy()
                        logger.info(
                            f"New best feature set: {lag_type} lags, {window} window, CV score: {avg_score:.4f}"
                        )

            if best_features is None:
                logger.warning(
                    "Auto feature engineering failed, using default features"
                )
                return self.prepare_features(data)

            target = data["close"].shift(-1)[best_features.index]
            self.feature_names = best_features.columns.tolist()

            logger.info(
                f"Auto feature engineering completed. Best CV score: {best_score:.4f}"
            )
            return best_features, target

        except Exception as e:
            logger.error(f"Auto feature engineering failed: {e}")
            # Fallback to default features
            return self.prepare_features(data)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, data=50)

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the XGBoost model.

        Args:
            data: Training data

        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting XGBoost model training...")

            # Setup model if not already done
            if self.model is None:
                self._setup_model()

            # Prepare features
            features, target = self.prepare_features(data)

            if len(features) < 50:
                raise ValueError(
                    "Insufficient data for training (need at least 50 samples)"
                )

            # Train model
            self.model.fit(features, target)
            self.is_trained = True

            # Calculate training metrics
            train_predictions = self.model.predict(features)
            mse = np.mean((target - train_predictions) ** 2)
            mae = np.mean(np.abs(target - train_predictions))

            logger.info(f"XGBoost training completed. MSE: {mse:.4f}, MAE: {mae:.4f}")

            return {
                "mse": mse,
                "mae": mae,
                "feature_importance": dict(
                    zip(self.feature_names, self.model.feature_importances_)
                ),
            }

        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model.

        Args:
            data: Input data for prediction

        Returns:
            Array of predictions
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")

            # Prepare features
            features, _ = self.prepare_features(data)

            if features.empty:
                raise ValueError("No valid features for prediction")

            # Validate feature shape
            expected_features = set(self.feature_names) if self.feature_names else set()
            actual_features = set(features.columns)

            if expected_features and actual_features != expected_features:
                missing_features = expected_features - actual_features
                extra_features = actual_features - expected_features
                raise ValueError(
                    f"Feature mismatch. Missing: {missing_features}, Extra: {extra_features}"
                )

            # Validate feature order
            if self.feature_names and list(features.columns) != self.feature_names:
                features = features[self.feature_names]

            # Make predictions
            predictions = self.model.predict(features)

            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained or self.model is None:
            return {}

        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save(self, filepath: str) -> bool:
        """Save the trained model.

        Args:
            filepath: Path to save the model

        Returns:
            True if successful
        """
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained model")

            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Save model and metadata
            model_data = {
                "model": self.model,
                "feature_names": self.feature_names,
                "config": self.config,
                "is_trained": self.is_trained,
                "timestamp": datetime.now().isoformat(),
            }

            joblib.dump(model_data, filepath)
            logger.info(f"XGBoost model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving XGBoost model: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Load a trained model.

        Args:
            filepath: Path to the saved model

        Returns:
            True if successful
        """
        try:
            model_data = joblib.load(filepath)

            self.model = model_data["model"]
            self.feature_names = model_data["feature_names"]
            self.config = model_data.get("config", {})
            self.is_trained = model_data.get("is_trained", False)

            logger.info(f"XGBoost model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.

        Returns:
            Model metadata dictionary
        """
        return {
            "model_type": "xgboost",
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }
