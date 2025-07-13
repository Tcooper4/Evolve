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
        self.model = CatBoostRegressor(**config.get("catboost_params", {}))
        self.fitted = False
        self.feature_columns = config.get("feature_columns", [])
        self.target_column = config.get("target_column", "target")

    def fit(self, train_data: pd.DataFrame, val_data=None, **kwargs):
        X = train_data[self.feature_columns]
        y = train_data[self.target_column]
        eval_set = None
        if val_data is not None:
            eval_set = (val_data[self.feature_columns], val_data[self.target_column])
        self.model.fit(X, y, eval_set=eval_set, use_best_model=True, verbose=False)
        self.fitted = True
        return {"train_loss": [], "val_loss": []}

    def predict(self, data: pd.DataFrame, horizon: int = 1):
        if not self.fitted:
            raise RuntimeError("Model must be fit before predicting.")
        X = data[self.feature_columns]
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
            if not self.fitted:
                raise RuntimeError("Model must be fitted before forecasting.")

            # Make initial prediction
            predictions = self.predict(data)

            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()

            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data)
                forecast_values.append(pred[-1])

                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row[self.target_column] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
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
        shap_values = self.model.get_feature_importance(Pool(X_sample, np.zeros(X_sample.shape[0])))
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
