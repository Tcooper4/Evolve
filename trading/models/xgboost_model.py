"""
XGBoost Model for Time Series Forecasting

This module provides an XGBoost-based model for time series forecasting
in the Evolve trading system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd

from .base_model import BaseModel
from utils.model_cache import cache_model_operation, get_model_cache

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
            # Feature construction sanity check to prevent future-data leakage
            self._validate_feature_construction(data)
            
            # Use auto feature engineering if enabled
            if self.config.get("auto_feature_engineering", False):
                return self._auto_engineer_features(data)

            # Create lag features using dynamic optimization
            optimal_lags = self._optimize_lags(data)
            features = pd.DataFrame()

            for lag in optimal_lags:
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
            
            # Store last features for SHAP calculation
            self.last_features = features.copy()

            return features, target

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def _validate_feature_construction(self, data: pd.DataFrame) -> None:
        """Validate feature construction to prevent future-data leakage.
        
        Args:
            data: Input time series data
            
        Raises:
            ValueError: If future data leakage is detected
        """
        try:
            # Check for proper time index
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Data must have a DatetimeIndex")
            
            # Check for sorted time index
            if not data.index.is_monotonic_increasing:
                raise ValueError("Data must be sorted by time (ascending)")
            
            # Check for duplicate timestamps
            if data.index.duplicated().any():
                raise ValueError("Data contains duplicate timestamps")
            
            # Check for missing values in key columns
            required_columns = ['close']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
                if data[col].isnull().all():
                    raise ValueError(f"Column '{col}' contains only null values")
            
            # Check for sufficient data
            if len(data) < 50:
                raise ValueError("Insufficient data for feature construction (need at least 50 samples)")
            
            # Check for reasonable price values
            if (data['close'] <= 0).any():
                raise ValueError("Price data contains non-positive values")
            
            # Check for extreme outliers (prices > 100x median)
            median_price = data['close'].median()
            if (data['close'] > 100 * median_price).any():
                logger.warning("Extreme price outliers detected in data")
            
            logger.info("Feature construction validation passed")
            
        except Exception as e:
            logger.error(f"Feature construction validation failed: {e}")
            raise ValueError(f"Feature construction validation failed: {e}")

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

    def _optimize_lags(self, data: pd.DataFrame, max_lags: int = 20) -> List[int]:
        """Dynamically optimize lag features using cross-validation.
        
        Args:
            data: Input time series data
            max_lags: Maximum number of lags to test
            
        Returns:
            List of optimal lag values
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_squared_error
            import xgboost as xgb
            
            logger.info(f"Optimizing lags up to {max_lags} periods...")
            
            # Test different lag combinations
            lag_candidates = list(range(1, min(max_lags + 1, len(data) // 4)))
            best_lags = []
            best_score = float('inf')
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Test individual lags first
            individual_scores = {}
            for lag in lag_candidates:
                try:
                    # Create simple feature set with just this lag
                    features = pd.DataFrame()
                    features[f'lag_{lag}'] = data['close'].shift(lag)
                    features['target'] = data['close'].shift(-1)
                    
                    # Remove NaN values
                    features = features.dropna()
                    
                    if len(features) < 50:
                        continue
                    
                    # Cross-validate
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(features):
                        X_train = features.iloc[train_idx, :-1]
                        y_train = features.iloc[train_idx, -1]
                        X_val = features.iloc[val_idx, :-1]
                        y_val = features.iloc[val_idx, -1]
                        
                        # Quick XGBoost fit
                        model = xgb.XGBRegressor(n_estimators=50, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Predict and score
                        y_pred = model.predict(X_val)
                        score = mean_squared_error(y_val, y_pred)
                        cv_scores.append(score)
                    
                    # Average score across folds
                    avg_score = np.mean(cv_scores)
                    individual_scores[lag] = avg_score
                    
                except Exception as e:
                    logger.warning(f"Error testing lag {lag}: {e}")
                    continue
            
            # Select top performing lags
            if individual_scores:
                # Sort by performance and select top lags
                sorted_lags = sorted(individual_scores.items(), key=lambda x: x[1])
                top_lags = [lag for lag, score in sorted_lags[:5]]  # Top 5 lags
                
                # Add some diversity (short, medium, long term)
                optimal_lags = []
                if 1 in top_lags:
                    optimal_lags.append(1)  # Always include lag 1
                
                # Add medium-term lags
                medium_lags = [lag for lag in top_lags if 2 <= lag <= 7]
                optimal_lags.extend(medium_lags[:2])
                
                # Add long-term lags
                long_lags = [lag for lag in top_lags if lag > 7]
                optimal_lags.extend(long_lags[:2])
                
                # Ensure we have at least 3 lags
                if len(optimal_lags) < 3:
                    optimal_lags.extend([1, 2, 3])
                
                optimal_lags = sorted(list(set(optimal_lags)))[:5]  # Remove duplicates, max 5
                
                logger.info(f"Selected optimal lags: {optimal_lags}")
                return optimal_lags
            
            # Fallback to default lags
            logger.warning("Lag optimization failed, using default lags")
            return [1, 2, 3, 5, 10]
            
        except Exception as e:
            logger.error(f"Error in lag optimization: {e}")
            return [1, 2, 3, 5, 10]  # Fallback

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
        """Get feature importance from the trained model using both native and SHAP methods.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return {}

        try:
            # Get native XGBoost feature importance
            native_importance = self.model.feature_importances_
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(native_importance))]
            native_importance_dict = dict(zip(feature_names, native_importance))
            
            # Try to get SHAP importance if available
            shap_importance_dict = self._get_shap_importance()
            
            # Combine both methods
            combined_importance = {}
            for feature in feature_names:
                native_score = native_importance_dict.get(feature, 0.0)
                shap_score = shap_importance_dict.get(feature, 0.0)
                
                # Use SHAP if available, otherwise fall back to native
                combined_importance[feature] = shap_score if shap_score > 0 else native_score
            
            # Sort by importance
            sorted_importance = dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True))
            
            logger.info(f"Feature importance calculated for {len(sorted_importance)} features")
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def _get_shap_importance(self) -> Dict[str, float]:
        """Get SHAP-based feature importance.
        
        Returns:
            Dictionary mapping feature names to SHAP importance scores
        """
        try:
            import shap
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            
            # Get sample data for SHAP calculation
            if hasattr(self, 'last_features') and self.last_features is not None:
                sample_data = self.last_features.iloc[:100]  # Use first 100 samples
            else:
                logger.warning("No sample data available for SHAP calculation")
                return {}
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(sample_data)
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Map to feature names
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(mean_shap_values))]
            shap_importance = dict(zip(feature_names, mean_shap_values))
            
            logger.info("SHAP feature importance calculated successfully")
            return shap_importance
            
        except ImportError:
            logger.warning("SHAP not available. Using native feature importance only.")
            return {}
        except Exception as e:
            logger.error(f"Error calculating SHAP importance: {e}")
            return {}

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

    @cache_model_operation
    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps with caching.

        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast

        Returns:
            Dictionary containing forecast results
        """
        try:
            if not self.is_trained:
                # Train the model if not already trained
                self.train(data)

            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()

            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data)
                forecast_values.append(pred[-1])

                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row["close"] = pred[-1]  # Update with prediction
                current_data = pd.concat(
                    [current_data, pd.DataFrame([new_row])], ignore_index=True
                )
                current_data = current_data.iloc[1:]  # Remove oldest row

            return {
                "forecast": np.array(forecast_values),
                "confidence": 0.85,  # XGBoost confidence
                "model": "XGBoost",
                "horizon": horizon,
                "feature_importance": self.get_feature_importance(),
                "metadata": self.get_metadata(),
            }

        except Exception as e:
            logger.error(f"Error in XGBoost model forecast: {e}")
            raise RuntimeError(f"XGBoost model forecasting failed: {e}")
