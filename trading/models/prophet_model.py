"""ProphetModel: Facebook Prophet wrapper for time series forecasting."""
from trading.models.base_model import BaseModel, ModelRegistry, ValidationError
from fbprophet import Prophet
import pandas as pd
import numpy as np
import os
import json

@ModelRegistry.register('Prophet')
class ProphetModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Prophet(**config.get('prophet_params', {}))
        self.fitted = False
        self.history = None

    def fit(self, train_data: pd.DataFrame, val_data=None, **kwargs):
        df = train_data[[self.config['date_column'], self.config['target_column']]].rename(columns={
            self.config['date_column']: 'ds',
            self.config['target_column']: 'y'
        })
        self.model.fit(df)
        self.fitted = True
        self.history = df
        return {'train_loss': [], 'val_loss': []}

    def predict(self, data: pd.DataFrame, horizon: int = 1):
        if not self.fitted:
            raise RuntimeError('Model must be fit before predicting.')
        future = data[[self.config['date_column']]].rename(columns={self.config['date_column']: 'ds'})
        forecast = self.model.predict(future)
        return forecast['yhat'].values

    def summary(self):
        print("ProphetModel: Facebook Prophet wrapper")
        print(self.model)

    def infer(self):
        pass  # Prophet is always in inference mode after fitting

    def shap_interpret(self, X_sample):
        print("SHAP not supported for Prophet. Showing component plots instead.")
        self.model.plot_components(self.model.predict(self.history))

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'prophet_model.json'))
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f)

    def load(self, path: str):
        from fbprophet.serialize import model_from_json
        with open(os.path.join(path, 'prophet_model.json'), 'r') as fin:
            self.model = model_from_json(fin.read())
        with open(os.path.join(path, 'config.json'), 'r') as f:
            self.config = json.load(f)
        self.fitted = True 