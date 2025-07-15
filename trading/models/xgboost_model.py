"""
XGBoost Model for Time Series Forecasting

This module provides an XGBoost-based model for time series forecasting
in the Evolve trading system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel
from utils.model_cache import cache_model_operation, get_model_cache
from utils.forecast_helpers import safe_forecast, validate_forecast_input, log_forecast_performance

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
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance_scores = None

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

    def _create_lag_features(self, data: pd.DataFrame, max_lags: int = 20) -> pd.DataFrame:
        """Create lag features with cross-validated selection.
        
        Args:
            data: Input data
            max_lags: Maximum number of lags to create
            
        Returns:
            DataFrame with lag features
        """
        try:
            features = data.copy()
            
            # Create lag features for close price
            if "close" in data.columns:
                for lag in range(1, max_lags + 1):
                    features[f"close_lag_{lag}"] = data["close"].shift(lag)
            
            # Create lag features for volume
            if "volume" in data.columns:
                for lag in range(1, min(max_lags, 10) + 1):
                    features[f"volume_lag_{lag}"] = data["volume"].shift(lag)
            
            # Create rolling statistics
            if "close" in data.columns:
                for window in [5, 10, 20, 50]:
                    features[f"close_ma_{window}"] = data["close"].rolling(window=window).mean()
                    features[f"close_std_{window}"] = data["close"].rolling(window=window).std()
                    features[f"close_min_{window}"] = data["close"].rolling(window=window).min()
                    features[f"close_max_{window}"] = data["close"].rolling(window=window).max()
            
            # Create technical indicators
            features = self._add_technical_indicators(features)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
            raise

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data.
        
        Args:
            data: Input data
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            if "close" in data.columns:
                # RSI
                data["rsi"] = self._calculate_rsi(data["close"])
                
                # MACD
                data["macd"], data["macd_signal"] = self._calculate_macd(data["close"])
                
                # Bollinger Bands
                data["bb_upper"], data["bb_middle"], data["bb_lower"] = self._calculate_bollinger_bands(data["close"])
                
                # Price changes
                data["price_change"] = data["close"].pct_change()
                data["price_change_2"] = data["close"].pct_change(2)
                data["price_change_5"] = data["close"].pct_change(5)
                
                # Volatility
                data["volatility"] = data["close"].rolling(window=20).std()
                
            if "volume" in data.columns:
                # Volume indicators
                data["volume_ma"] = data["volume"].rolling(window=20).mean()
                data["volume_ratio"] = data["volume"] / data["volume_ma"]
                
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data

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

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            return macd, macd_signal
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            return pd.Series(index=prices.index, data=0), pd.Series(index=prices.index, data=0)

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            middle = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            return prices, prices, prices

    def _select_features_cross_validation(self, X: pd.DataFrame, y: pd.Series, max_features: int = 50) -> List[str]:
        """Select features using cross-validated methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            # Remove any remaining NaN values
            X_clean = X.dropna()
            y_clean = y[X_clean.index]
            
            if len(X_clean) < 50:
                logger.warning("Insufficient data for feature selection, using all features")
                return list(X_clean.columns)
            
            # Method 1: SelectKBest with f_regression
            try:
                selector_kbest = SelectKBest(score_func=f_regression, k=min(max_features, len(X_clean.columns)))
                selector_kbest.fit(X_clean, y_clean)
                kbest_features = X_clean.columns[selector_kbest.get_support()].tolist()
                logger.info(f"SelectKBest selected {len(kbest_features)} features")
            except Exception as e:
                logger.warning(f"SelectKBest failed: {e}")
                kbest_features = list(X_clean.columns)
            
            # Method 2: Recursive Feature Elimination with XGBoost
            try:
                import xgboost as xgb
                estimator = xgb.XGBRegressor(n_estimators=50, random_state=42)
                selector_rfe = RFE(estimator=estimator, n_features_to_select=min(max_features//2, len(X_clean.columns)))
                selector_rfe.fit(X_clean, y_clean)
                rfe_features = X_clean.columns[selector_rfe.get_support()].tolist()
                logger.info(f"RFE selected {len(rfe_features)} features")
            except Exception as e:
                logger.warning(f"RFE failed: {e}")
                rfe_features = list(X_clean.columns)
            
            # Combine features from both methods
            combined_features = list(set(kbest_features + rfe_features))
            
            # Limit to max_features
            if len(combined_features) > max_features:
                # Use feature importance from XGBoost to prioritize
                try:
                    temp_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
                    temp_model.fit(X_clean[combined_features], y_clean)
                    feature_importance = pd.Series(temp_model.feature_importances_, index=combined_features)
                    combined_features = feature_importance.nlargest(max_features).index.tolist()
                except Exception as e:
                    logger.warning(f"Feature importance ranking failed: {e}")
                    combined_features = combined_features[:max_features]
            
            logger.info(f"Final feature selection: {len(combined_features)} features")
            return combined_features
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return list(X.columns)

    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features with cross-validated feature selection.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of features and target
        """
        try:
            # Create lag features
            features = self._create_lag_features(data)
            
            if features.empty:
                raise ValueError("No features created from input data")
            
            # Separate features and target
            target_col = "close" if "close" in features.columns else features.columns[0]
            target = features[target_col]
            feature_cols = [col for col in features.columns if col != target_col]
            X = features[feature_cols]
            
            # Select features if not already done
            if self.selected_features is None:
                self.selected_features = self._select_features_cross_validation(X, target)
                self.feature_names = self.selected_features
            
            # Use only selected features
            X = X[self.selected_features]
            
            # Scale features
            if not self.is_trained:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            return X_scaled_df, target
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the XGBoost model with cross-validated feature selection.

        Args:
            data: Training data

        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting XGBoost model training with cross-validated feature selection...")

            # Setup model if not already done
            if self.model is None:
                self._setup_model()

            # Prepare features with cross-validated selection
            features, target = self.prepare_features(data)

            if len(features) < 50:
                raise ValueError(
                    "Insufficient data for training (need at least 50 samples)"
                )

            # Cross-validation to validate model performance
            try:
                from sklearn.model_selection import TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(self.model, features, target, cv=tscv, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores)
                logger.info(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")

            # Train model
            self.model.fit(features, target)
            self.is_trained = True

            # Calculate training metrics
            train_predictions = self.model.predict(features)
            mse = np.mean((target - train_predictions) ** 2)
            mae = np.mean(np.abs(target - train_predictions))

            # Store feature importance
            self.feature_importance_scores = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )

            logger.info(f"XGBoost training completed. MSE: {mse:.4f}, MAE: {mae:.4f}")
            logger.info(f"Selected {len(self.feature_names)} features")

            return {
                "mse": mse,
                "mae": mae,
                "feature_importance": self.feature_importance_scores,
                "selected_features": self.feature_names,
                "cv_rmse": cv_rmse.mean() if 'cv_rmse' in locals() else None,
            }

        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model with error handling.

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

            # Validate predictions
            if np.any(np.isnan(predictions)):
                logger.warning("Model produced NaN predictions, replacing with mean")
                predictions = np.nan_to_num(predictions, nan=np.nanmean(predictions))
            
            if np.any(np.isinf(predictions)):
                logger.warning("Model produced infinite predictions, replacing with mean")
                predictions = np.nan_to_num(predictions, posinf=np.nanmean(predictions), neginf=np.nanmean(predictions))

            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            # Return fallback predictions
            if len(data) > 0 and "close" in data.columns:
                logger.info("Using fallback predictions (mean of close prices)")
                return np.full(len(data), data["close"].mean())
            else:
                return np.array([])

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance scores
        """
        if self.feature_importance_scores is None:
            return {}
        return self.feature_importance_scores.copy()

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
                "selected_features": self.selected_features,
                "feature_importance_scores": self.feature_importance_scores,
                "scaler": self.scaler,
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
            self.selected_features = model_data.get("selected_features")
            self.feature_importance_scores = model_data.get("feature_importance_scores")
            self.scaler = model_data.get("scaler", StandardScaler())
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
            "selected_features": self.selected_features,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

    @cache_model_operation
    @safe_forecast(max_retries=2, retry_delay=0.5, log_errors=True)
    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps with caching and error handling.

        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast

        Returns:
            Dictionary containing forecast results
        """
        import time
        start_time = time.time()
        
        # Validate input data
        validate_forecast_input(data, min_length=20, require_numeric=True)
        
        if not self.is_trained:
            # Train the model if not already trained
            logger.info("Model not trained, training with provided data...")
            self.train(data)

        # Generate multi-step forecast
        forecast_values = []
        current_data = data.copy()

        for i in range(horizon):
            # Get prediction for next step
            pred = self.predict(current_data)
            
            if len(pred) == 0:
                logger.error(f"Empty prediction at step {i}")
                break
                
            forecast_values.append(pred[-1])

            # Update data for next iteration
            new_row = current_data.iloc[-1].copy()
            new_row["close"] = pred[-1]  # Update with prediction
            current_data = pd.concat(
                [current_data, pd.DataFrame([new_row])], ignore_index=True
            )
            current_data = current_data.iloc[1:]  # Remove oldest row

        # Calculate confidence based on prediction stability
        if len(forecast_values) > 1:
            forecast_std = np.std(forecast_values)
            forecast_mean = np.mean(forecast_values)
            confidence = max(0.1, min(0.95, 1.0 - (forecast_std / (forecast_mean + 1e-8))))
        else:
            confidence = 0.5

        execution_time = time.time() - start_time
        
        # Log performance
        log_forecast_performance(
            model_name="XGBoost",
            execution_time=execution_time,
            data_length=len(data),
            confidence=confidence
        )

        return {
            "forecast": np.array(forecast_values),
            "confidence": confidence,
            "model": "XGBoost",
            "horizon": horizon,
            "feature_importance": self.get_feature_importance(),
            "metadata": self.get_metadata(),
        }
