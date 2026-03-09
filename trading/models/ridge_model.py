"""
Ridge Regression Model for Time Series Forecasting

This module implements a Ridge Regression model for time series forecasting with
regularization to prevent overfitting. Ridge regression is particularly useful
for financial forecasting when dealing with multicollinearity and high-dimensional
feature spaces.

Features:
- Ridge regression with L2 regularization
- Feature engineering for time series
- Hyperparameter optimization
- Integration with the BaseModel interface
- Comprehensive logging and error handling
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ridge-specific imports
try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("[WARN] scikit-learn not available. Disabling Ridge models.")
    print(f"   Missing: {e}")
    Ridge = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
    cross_val_score = None
    PolynomialFeatures = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

from .base_model import BaseModel, ModelError, ModelRegistry, ValidationError

logger = logging.getLogger(__name__)


@ModelRegistry.register("ridge")
class RidgeModel(BaseModel):
    """
    Ridge Regression Model for time series forecasting.

    This model implements Ridge regression with L2 regularization for forecasting
    financial time series. It's particularly useful for:
    - High-dimensional feature spaces
    - Multicollinearity problems
    - Stable predictions with regularization
    - Feature importance analysis

    Attributes:
        alpha (float): Regularization strength (L2 penalty)
        max_iter (int): Maximum iterations for optimization
        solver (str): Solver algorithm ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg')
        fit_intercept (bool): Whether to fit intercept
        normalize (bool): Whether to normalize features
        model: The fitted Ridge regression model
        scaler: Feature scaler
        feature_names: Names of features used in training
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Ridge regression model.

        Args:
            config: Configuration dictionary containing:
                - alpha: Regularization strength (default: 1.0)
                - max_iter: Maximum iterations (default: 1000)
                - solver: Solver algorithm (default: 'auto')
                - fit_intercept: Whether to fit intercept (default: True)
                - normalize: Whether to normalize features (default: False)
                - polynomial_degree: Degree of polynomial features (default: 1)
                - use_lags: Whether to use lag features (default: True)
                - lag_periods: Number of lag periods (default: 5)
        """
        super().__init__(config)

        # Validate scikit-learn availability
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not available. Cannot create Ridge model."
            )

        # Set default configuration
        self.config = {
            "alpha": 1.0,
            "max_iter": 1000,
            "solver": "auto",
            "fit_intercept": True,
            "polynomial_degree": 1,
            "use_lags": True,
            "lag_periods": 5,
            "random_state": 42,
            **self.config,
        }

        # Initialize model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.cv_scores = []

        # Validate configuration
        self._validate_ridge_config()

        logger.info(f"Ridge model initialized with config: {self.config}")

    def _validate_ridge_config(self) -> None:
        """Validate Ridge-specific configuration."""
        valid_solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]

        if self.config["alpha"] <= 0:
            raise ValidationError("Alpha must be positive")

        if self.config["max_iter"] <= 0:
            raise ValidationError("max_iter must be positive")

        if self.config["solver"] not in valid_solvers:
            raise ValidationError(f"Invalid solver. Must be one of: {valid_solvers}")

        if self.config["polynomial_degree"] < 1:
            raise ValidationError("Polynomial degree must be at least 1")

        if self.config["lag_periods"] < 0:
            raise ValidationError("Lag periods must be non-negative")

    def _add_lag_features(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add lag features to the feature matrix and align target."""
        lag_periods = self.config.get("lag_periods", 5)
        if lag_periods == 0:
            return features, target
        # Create lag features from target
        lag_features = []
        for lag in range(1, lag_periods + 1):
            lag_feature = np.roll(target, lag)
            lag_feature[:lag] = np.nan  # Set initial values to NaN
            lag_features.append(lag_feature)
        lag_features = np.column_stack(lag_features)
        # Remove rows with NaN values (first lag_periods rows)
        valid_mask = ~np.isnan(lag_features).any(axis=1)
        # Also trim features and target to match
        features = features[lag_periods:][valid_mask[lag_periods:]]
        target = target[lag_periods:][valid_mask[lag_periods:]]
        lag_features = lag_features[lag_periods:][valid_mask[lag_periods:]]
        # Combine original features with lag features
        combined_features = np.column_stack([features, lag_features])
        return combined_features, target

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance lowercase columns to title case."""
        if df is None or df.empty:
            return df
        if "Close" not in df.columns and "close" in df.columns:
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return df

    def _prepare_data(
        self, data: pd.DataFrame, is_training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Ridge regression modeling.

        Args:
            data: Input DataFrame with features and target
            is_training: Whether this is for training

        Returns:
            Tuple of (features, target)
        """
        data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
        # Ensure we have numeric data
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValidationError("No numeric columns found for Ridge modeling")
        # Identify target column (assume last column or specified target)
        if "target" in numeric_data.columns:
            target_col = "target"
        elif "price" in numeric_data.columns:
            target_col = "price"
        elif "Close" in numeric_data.columns:
            target_col = "Close"
        elif "close" in numeric_data.columns:
            target_col = "close"
        else:
            # Use the last column as target
            if len(numeric_data.columns) > 0:
                target_col = numeric_data.columns[-1]
            else:
                raise ValueError("No numeric columns found for target")
        # Separate features and target
        feature_cols = [col for col in numeric_data.columns if col != target_col]
        if not feature_cols:
            raise ValidationError("No feature columns found")
        features = numeric_data[feature_cols].values
        price_series = numeric_data[target_col].values
        # Train on returns (next-period return) for continuity; inference keeps level for lag building
        if is_training and target_col in ("Close", "close") and len(price_series) > 2:
            returns = np.diff(price_series) / (price_series[:-1] + 1e-10)
            target = returns  # predict next return
            features = features[1:]  # align: row i predicts return from i to i+1
            if len(features) != len(target):
                min_len = min(len(features), len(target))
                features = features[:min_len]
                target = target[:min_len]
        else:
            target = price_series
        # Remove any NaN values
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features = features[valid_mask]
        target = target[valid_mask]
        if len(features) < 10:
            raise ValidationError(
                "Insufficient data for Ridge modeling. Need at least 10 observations."
            )
        # Create lag features if requested.
        # We apply the same lag transformation for both training and inference
        # so that the scaler and model see a consistent feature dimensionality.
        if self.config.get("use_lags", True):
            features, target = self._add_lag_features(features, target)
        # Create polynomial features if requested
        if self.config.get("polynomial_degree", 1) > 1:
            features = self._add_polynomial_features(features)
        # Scale features
        if is_training:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        logger.info(
            f"Prepared {len(features)} observations with {features.shape[1]} features for Ridge modeling"
        )
        return features, target

    def _add_polynomial_features(self, features: np.ndarray) -> np.ndarray:
        """Add polynomial features to the feature matrix."""
        degree = self.config.get("polynomial_degree", 1)

        if degree == 1:
            return features

        try:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(features)

            # Keep only the polynomial features (exclude original features to avoid duplication)
            if features.shape[1] > 0:
                original_feature_count = features.shape[1]
            else:
                raise ValueError("No features available for training")
            polynomial_features = poly_features[:, original_feature_count:]

            # Combine original and polynomial features
            combined_features = np.column_stack([features, polynomial_features])

            return combined_features

        except Exception as e:
            logger.warning(
                f"Error creating polynomial features: {e}. Using original features."
            )
            return features

    def build_model(self) -> Ridge:
        """
        Build the Ridge regression model.

        Returns:
            Ridge regression model instance
        """
        # Ensure max_iter is int (config may have wrong type from registry)
        max_iter = self.config.get("max_iter", 1000)
        if not isinstance(max_iter, int):
            try:
                max_iter = int(max_iter)
            except (TypeError, ValueError):
                max_iter = 1000
        max_iter = max(1, max_iter)
        model = Ridge(
            alpha=float(self.config.get("alpha", 1.0)),
            max_iter=max_iter,
            solver=self.config.get("solver", "auto"),
            fit_intercept=self.config.get("fit_intercept", True),
            random_state=int(self.config.get("random_state", 42)),
        )

        return model

    def train(
        self,
        data: pd.DataFrame,
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the Ridge regression model.

        Args:
            data: Training data
            target_col: Target column name
            feature_cols: Feature column names (optional)
            **kwargs: Additional training parameters

        Returns:
            Training results dictionary
        """
        data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
        logger.info("Starting Ridge model training")
        start_time = datetime.now()

        try:
            # Prepare data
            features, target = self._prepare_data(data, is_training=True)

            # Build model
            self.model = self.build_model()

            # Perform cross-validation if requested
            if kwargs.get("cross_validate", False):
                cv_scores = cross_val_score(
                    self.model, features, target, cv=5, scoring="r2"
                )
                self.cv_scores = cv_scores.tolist()
                logger.info(
                    f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
                )

            # Fit the model
            self.model.fit(features, target)

            # Calculate training metrics
            training_time = (datetime.now() - start_time).total_seconds()
            metrics = self._calculate_training_metrics(features, target)

            # Calculate feature importance
            self._calculate_feature_importance(features)

            logger.info(
                f"Ridge model training completed in {training_time:.2f} seconds"
            )
            logger.info(f"Training metrics: {metrics}")

            return {
                "training_time": training_time,
                "metrics": metrics,
                "cv_scores": self.cv_scores,
                "feature_importance": self.feature_importance,
                "model_coefficients": self.model.coef_.tolist(),
                "intercept": float(self.model.intercept_),
            }

        except Exception as e:
            logger.error(f"Error during Ridge model training: {e}")
            raise ModelError(f"Ridge training failed: {e}")

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Convenience wrapper so RidgeModel can be used in generic pipelines/ensembles.

        Args:
            data: Input DataFrame containing price data.
            target: Optional explicit target series. If None, the method will
                infer a target column from ``close``/``Close`` or the last
                numeric column.

        Returns:
            Training result dictionary from :meth:`train`.
        """
        df = data.copy() if hasattr(data, "copy") else data
        if target is not None:
            df = df.assign(_ridge_target=target)
            target_col = "_ridge_target"
        elif "close" in df.columns:
            target_col = "close"
        elif "Close" in df.columns:
            target_col = "Close"
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not len(numeric_cols):
                raise ModelError(
                    "RidgeModel.fit requires at least one numeric column to use as target"
                )
            target_col = numeric_cols[-1]

        return self.train(df, target_col=target_col)

    def _calculate_training_metrics(
        self, features: np.ndarray, target: np.ndarray
    ) -> Dict[str, float]:
        """Calculate training metrics for the Ridge model."""
        if self.model is None:
            return {}

        try:
            # Make predictions
            predictions = self.model.predict(features)

            # Calculate metrics
            metrics = {
                "r2_score": float(r2_score(target, predictions)),
                "mean_squared_error": float(mean_squared_error(target, predictions)),
                "root_mean_squared_error": float(
                    np.sqrt(mean_squared_error(target, predictions))
                ),
                "mean_absolute_error": float(mean_absolute_error(target, predictions)),
                "mean_absolute_percentage_error": float(
                    np.mean(np.abs((target - predictions) / (target + 1e-10))) * 100
                ),
            }

            return metrics

        except Exception as e:
            logger.warning(f"Error calculating training metrics: {e}")
            return {}

    def _calculate_feature_importance(self, features: np.ndarray) -> None:
        """Calculate feature importance based on coefficients."""
        if self.model is None:
            return

        try:
            # Use absolute coefficient values as feature importance
            importance = np.abs(self.model.coef_)

            # Create feature names if not available
            if not self.feature_names:
                self.feature_names = [f"feature_{i}" for i in range(len(importance))]

            # Create importance dictionary
            self.feature_importance = {
                name: float(imp) for name, imp in zip(self.feature_names, importance)
            }

            # Sort by importance
            self.feature_importance = dict(
                sorted(
                    self.feature_importance.items(), key=lambda x: x[1], reverse=True
                )
            )

        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")

    def predict(self, data: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """
        Predict using the fitted Ridge model.

        Args:
            data: Input data
            horizon: Forecast horizon (ignored for Ridge, returns single prediction)

        Returns:
            Predicted values
        """
        data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
        if self.model is None:
            raise ModelError("Model must be trained before prediction")
        try:
            # Prepare data
            features, _ = self._prepare_data(data, is_training=False)
            if hasattr(self.scaler, 'mean_') and len(self.scaler.mean_) > 0:
                expected_features = self.scaler.mean_.shape[0]
            else:
                raise ValueError("Scaler not fitted or has no features")
            if features.shape[1] != expected_features:
                # Try to add lag features if not already present
                if self.config.get("use_lags", True):
                    # Recreate lag features if possible
                    features_orig = (
                        data.select_dtypes(include=[np.number])
                        .drop(columns=["target"], errors="ignore")
                        .values
                    )
                    target_orig = (
                        data["target"].values
                        if "target" in data.columns
                        else np.zeros(len(data))
                    )
                    features_lagged, _ = self._add_lag_features(
                        features_orig, target_orig
                    )
                    if features_lagged.shape[1] == expected_features:
                        features = features_lagged
                    else:
                        raise ModelError(
                            f"Feature mismatch: model expects {expected_features} features, but got {features.shape[1]}. Please provide input data with the same lag structure as used in training."
                        )
                else:
                    raise ModelError(
                        f"Feature mismatch: model expects {expected_features} features, but got {features.shape[1]}. Please provide input data with the same feature structure as used in training."
                    )
            predictions = self.model.predict(features)
            logger.info(
                f"Generated Ridge predictions for {len(predictions)} observations"
            )
            return predictions
        except Exception as e:
            logger.error(f"Error during Ridge prediction: {e}")
            raise ModelError(f"Ridge prediction failed: {e}")

    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """
        Generate forecasts using the Ridge model. Model predicts returns; build ratio path
        so router denormalizes once (forecast in normalized space = ratio to last price).
        """
        data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
        logger.info(f"Generating Ridge forecast for {horizon} periods")

        try:
            price_col = "Close" if "Close" in data.columns else "close"
            current_ratio = float(data[price_col].iloc[-1]) if price_col in data.columns else 1.0
            forecasts = []
            current_data = data.copy()

            for _ in range(horizon):
                pred = self.predict(current_data, horizon=1)
                r = float(pred[-1]) if len(pred) else 0.0
                r = np.clip(r, -0.20, 0.20)
                next_ratio = current_ratio * (1.0 + r)
                forecasts.append(next_ratio)
                # Append a new row for next step (simplified: reuse last row with updated price)
                new_row = current_data.iloc[-1].copy()
                if price_col in new_row.index:
                    new_row[price_col] = next_ratio
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                if len(current_data) > len(data):
                    current_data = current_data.iloc[1:]
                current_ratio = next_ratio

            forecast_array = np.array(forecasts, dtype="float64")
            return {
                "forecast": forecast_array,
                "horizon": horizon,
                "model_type": "Ridge",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error during Ridge forecasting: {e}")
            raise ModelError(f"Ridge forecasting failed: {e}")

    def evaluate(
        self, data: pd.DataFrame, target_col: str = "target"
    ) -> Dict[str, float]:
        """
        Evaluate the Ridge model on test data.

        Args:
            data: Test data
            target_col: Target column name

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ModelError("Model must be trained before evaluation")

        try:
            # Prepare test data
            features, target = self._prepare_data(data, is_training=False)

            # Make predictions
            predictions = self.model.predict(features)

            # Calculate evaluation metrics
            metrics = {
                "r2_score": float(r2_score(target, predictions)),
                "mean_squared_error": float(mean_squared_error(target, predictions)),
                "root_mean_squared_error": float(
                    np.sqrt(mean_squared_error(target, predictions))
                ),
                "mean_absolute_error": float(mean_absolute_error(target, predictions)),
                "mean_absolute_percentage_error": float(
                    np.mean(np.abs((target - predictions) / (target + 1e-10))) * 100
                ),
            }

            logger.info(f"Ridge model evaluation completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during Ridge evaluation: {e}")
            raise ModelError(f"Ridge evaluation failed: {e}")

    def save_model(self, filepath: str) -> Dict[str, Any]:
        """
        Save the fitted Ridge model.

        Args:
            filepath: Path to save the model

        Returns:
            Save metadata
        """
        if self.model is None:
            raise ModelError("No fitted model to save")

        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Save the model and scaler
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
                "config": self.config,
            }

            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

            # Save additional metadata
            metadata = {
                "config": self.config,
                "model_type": "Ridge",
                "save_timestamp": datetime.now().isoformat(),
                "feature_importance": self.feature_importance,
                "cv_scores": self.cv_scores,
            }

            metadata_path = filepath.replace(".pkl", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Ridge model saved to {filepath}")
            return metadata

        except Exception as e:
            logger.error(f"Error saving Ridge model: {e}")
            raise ModelError(f"Failed to save Ridge model: {e}")

    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load a fitted Ridge model.

        Args:
            filepath: Path to the saved model

        Returns:
            Load metadata
        """
        try:
            # Load the model data
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            # Restore model components
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data.get("feature_names", [])
            self.feature_importance = model_data.get("feature_importance", {})
            self.config.update(model_data.get("config", {}))

            # Load metadata if available
            metadata_path = filepath.replace(".pkl", "_metadata.json")
            metadata = {}
            if Path(metadata_path).exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            logger.info(f"Ridge model loaded from {filepath}")
            return metadata

        except Exception as e:
            logger.error(f"Error loading Ridge model: {e}")
            raise ModelError(f"Failed to load Ridge model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ridge model."""
        info = {
            "model_type": "Ridge",
            "config": self.config,
            "is_fitted": self.model is not None,
            "feature_importance_available": bool(self.feature_importance),
            "cv_scores_available": bool(self.cv_scores),
        }

        if self.model is not None:
            info.update(
                {
                    "num_features": len(self.model.coef_),
                    "intercept": float(self.model.intercept_),
                    "alpha": self.model.alpha,
                }
            )

        return info

    def get_confidence(self) -> Dict[str, float]:
        """Get model confidence metrics."""
        if self.model is None:
            return {"confidence": 0.0}

        try:
            # Calculate confidence based on R² score and cross-validation
            confidence = 0.5  # Base confidence

            # Adjust based on cross-validation scores if available
            if self.cv_scores:
                cv_mean = np.mean(self.cv_scores)
                confidence = min(1.0, max(0.0, cv_mean))

            return {
                "confidence": confidence,
                "cv_mean": np.mean(self.cv_scores) if self.cv_scores else 0.0,
                "cv_std": np.std(self.cv_scores) if self.cv_scores else 0.0,
            }

        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return {"confidence": 0.5}


# Convenience function for creating Ridge models


def create_ridge_model(
    alpha: float = 1.0,
    max_iter: int = 1000,
    solver: str = "auto",
    polynomial_degree: int = 1,
    use_lags: bool = True,
    lag_periods: int = 5,
) -> RidgeModel:
    """
    Create a Ridge model with specified parameters.

    Args:
        alpha: Regularization strength
        max_iter: Maximum iterations
        solver: Solver algorithm
        polynomial_degree: Degree of polynomial features
        use_lags: Whether to use lag features
        lag_periods: Number of lag periods

    Returns:
        Configured Ridge model
    """
    config = {
        "alpha": alpha,
        "max_iter": max_iter,
        "solver": solver,
        "polynomial_degree": polynomial_degree,
        "use_lags": use_lags,
        "lag_periods": lag_periods,
    }

    return RidgeModel(config)
