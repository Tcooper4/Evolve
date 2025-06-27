"""CatBoostModel: CatBoostRegressor wrapper for time series forecasting."""
from trading.base_model import BaseModel, ModelRegistry, ValidationError
from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
import os
import json

@ModelRegistry.register('CatBoost')
class CatBoostModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = CatBoostRegressor(**config.get('catboost_params', {}))
        self.fitted = False
        self.feature_columns = config.get('feature_columns', [])
        self.target_column = config.get('target_column', 'target')

    def fit(self, train_data: pd.DataFrame, val_data=None, **kwargs):
        X = train_data[self.feature_columns]
        y = train_data[self.target_column]
        eval_set = None
        if val_data is not None:
            eval_set = (val_data[self.feature_columns], val_data[self.target_column])
        self.model.fit(X, y, eval_set=eval_set, use_best_model=True, verbose=False)
        self.fitted = True
        return {'train_loss': [], 'val_loss': []}

    def predict(self, data: pd.DataFrame, horizon: int = 1):
        if not self.fitted:
            raise RuntimeError('Model must be fit before predicting.')
        X = data[self.feature_columns]
        return self.model.predict(X)

    def summary(self):
        print("CatBoostModel: CatBoostRegressor wrapper")
        print(self.model)

    def infer(self):
        pass  # CatBoost is always in inference mode after fitting

    def shap_interpret(self, X_sample):
        print("CatBoost SHAP summary plot:")
        shap_values = self.model.get_feature_importance(Pool(X_sample, np.zeros(X_sample.shape[0])))
        import matplotlib.pyplot as plt
        plt.bar(self.feature_columns, shap_values)
        plt.title('CatBoost Feature Importances (SHAP)')
        plt.show()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_model(os.path.join(path, 'catboost_model.cbm'))
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f)

    def load(self, path: str):
        self.model.load_model(os.path.join(path, 'catboost_model.cbm'))
        with open(os.path.join(path, 'config.json'), 'r') as f:
            self.config = json.load(f)
        self.fitted = True 