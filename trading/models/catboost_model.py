"""CatBoostModel: CatBoostRegressor wrapper for time series forecasting."""

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .base_model import BaseModel, ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("CatBoost")
class CatBoostModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.fitted = False
        # Use the normalized config stored on the base class
        self.feature_columns = self.config.get("feature_columns", [])
        self.target_column = self.config.get("target_column", "target")

    def build_model(self):
        """Initialize the underlying CatBoostRegressor.

        This satisfies the abstract BaseModel.build_model contract so the
        class can be instantiated directly.
        """
        params = self.config.get("catboost_params", {}) if hasattr(self, "config") else {}
        self.model = CatBoostRegressor(**params)
        return self.model

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance lowercase columns to title case."""
        if df is None or df.empty:
            return df
        if "Close" not in df.columns and "close" in df.columns:
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return df

    def _prepare_data(
        self, data: pd.DataFrame, is_training: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare NumPy feature/target arrays from a DataFrame for BaseModel calls.

        This implementation is resilient to config/data mismatches:
        - Normalizes yfinance lowercase OHLCV to title case
        - Auto-detects feature columns from whatever numeric columns exist
        """
        data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
        if data is None or data.empty:
            raise ValueError("Training data is empty")

        # Resolve target column (prefer configured, then Close/close)
        target_col = self.target_column
        if target_col not in data.columns:
            if target_col == "close" and "Close" in data.columns:
                target_col = "Close"
            elif "Close" in data.columns:
                target_col = "Close"
            elif "close" in data.columns:
                target_col = "close"
            else:
                # Fallback to last numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if not len(numeric_cols):
                    raise ValueError("No numeric columns available for CatBoost target")
                target_col = numeric_cols[-1]

        # Auto-detect usable feature columns
        if self.feature_columns:
            feat_candidates = [
                (c if c in data.columns else ("Close" if c == "close" and "Close" in data.columns else c))
                for c in self.feature_columns
            ]
            feat_cols = [c for c in feat_candidates if c in data.columns and c != target_col]
        else:
            # Use all numeric columns except target
            feat_cols = [
                c for c in data.select_dtypes(include=[np.number]).columns if c != target_col
            ]

        if not feat_cols:
            raise ValueError("No valid feature columns found for CatBoostModel")

        X = data[feat_cols].to_numpy()
        y = data[target_col].to_numpy()
        return X, y

    def fit(self, train_data: pd.DataFrame, val_data=None, **kwargs):
        train_data = self._normalize_columns(train_data.copy() if hasattr(train_data, "copy") else train_data)
        if "Close" in train_data.columns and self.target_column == "close":
            self.target_column = "Close"
        fc = [c if c in train_data.columns else ("Close" if c == "close" else "Volume" if c == "volume" else c) for c in self.feature_columns]
        fc = [c for c in fc if c in train_data.columns]
        if not fc:
            fc = ["Close"] if "Close" in train_data.columns else list(train_data.columns)[:1]
        X = train_data[fc]
        y = train_data[self.target_column]
        eval_set = None
        if val_data is not None:
            val_data = self._normalize_columns(val_data.copy() if hasattr(val_data, "copy") else val_data)
            eval_set = (val_data[fc], val_data[self.target_column])
        self.model.fit(X, y, eval_set=eval_set, use_best_model=True, verbose=False)
        self.fitted = True
        return {"train_loss": [], "val_loss": []}

    def predict(self, data: pd.DataFrame, horizon: int = 1):
        if not self.fitted:
            raise RuntimeError("Model must be fit before predicting.")
        data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
        fc = [c if c in data.columns else ("Close" if c == "close" else "Volume" if c == "volume" else c) for c in self.feature_columns]
        fc = [c for c in fc if c in data.columns] or (["Close"] if "Close" in data.columns else list(data.columns)[:1])
        X = data[fc]
        return self.model.predict(X)

    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.

        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast

        Returns:
            Dictionary containing forecast results
        """
        try:
            data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
            if not self.fitted:
                raise RuntimeError("Model must be fitted before forecasting.")

            # Make initial prediction
            self.predict(data)

            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()

            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data)
                if len(pred) > 0:
                    forecast_values.append(pred[-1])
                else:
                    logger.warning("Empty prediction array, skipping")
                    break

                # Update data for next iteration
                if len(current_data) > 0:
                    new_row = current_data.iloc[-1].copy()
                else:
                    logger.warning("Empty data, cannot continue forecast")
                    break
                new_row[self.target_column] = pred[-1]  # Update with prediction
                current_data = pd.concat(
                    [current_data, pd.DataFrame([new_row])], ignore_index=True
                )
                current_data = current_data.iloc[1:]  # Remove oldest row

            return {
                "forecast": np.array(forecast_values),
                "confidence": 0.85,  # CatBoost confidence
                "model": "CatBoost",
                "horizon": horizon,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
            }

        except Exception as e:
            logging.error(f"Error in CatBoost model forecast: {e}")
            raise RuntimeError(f"CatBoost model forecasting failed: {e}")

    def summary(self):
        logger.info("CatBoostModel: CatBoostRegressor wrapper")
        logger.info(str(self.model))

    def infer(self):
        pass  # CatBoost is always in inference mode after fitting

    def shap_interpret(self, X_sample):
        logger.info("CatBoost SHAP summary plot:")
        shap_values = self.model.get_feature_importance(
            Pool(X_sample, np.zeros(X_sample.shape[0]))
        )
        import matplotlib.pyplot as plt

        plt.bar(self.feature_columns, shap_values)
        plt.title("CatBoost Feature Importances (SHAP)")
        plt.show()

    def save(self, path: str):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create directory {path}: {e}")
        self.model.save_model(os.path.join(path, "catboost_model.cbm"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f)

    def load(self, path: str):
        self.model.load_model(os.path.join(path, "catboost_model.cbm"))
        with open(os.path.join(path, "config.json"), "r") as f:
            self.config = json.load(f)
        self.fitted = True
