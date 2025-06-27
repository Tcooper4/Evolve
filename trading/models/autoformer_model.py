"""AutoformerModel: Autoformer wrapper for time series forecasting."""
from trading.base_model import BaseModel, ModelRegistry, ValidationError
import pandas as pd
import numpy as np
import torch
import os
import json

try:
    from autoformer_pytorch import Autoformer
except ImportError:
    Autoformer = None

@ModelRegistry.register('Autoformer')
class AutoformerModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if Autoformer is None:
            raise ImportError('autoformer-pytorch is not installed.')
        self.model = Autoformer(
            num_time_features=len(config.get('feature_columns', [])),
            seq_len=config.get('sequence_length', 24),
            pred_len=config.get('pred_length', 1),
            **config.get('autoformer_params', {})
        )
        self.fitted = False
        self.feature_columns = config.get('feature_columns', [])
        self.target_column = config.get('target_column', 'target')

    def fit(self, train_data: pd.DataFrame, val_data=None, epochs=10, batch_size=32, **kwargs):
        X = train_data[self.feature_columns].values.astype(np.float32)
        y = train_data[self.target_column].values.astype(np.float32)
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(X_tensor.unsqueeze(0))
            loss = torch.nn.functional.mse_loss(y_pred.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        self.fitted = True
        return {'train_loss': [], 'val_loss': []}

    def predict(self, data: pd.DataFrame, horizon: int = 1):
        if not self.fitted:
            raise RuntimeError('Model must be fit before predicting.')
        X = data[self.feature_columns].values.astype(np.float32)
        X_tensor = torch.tensor(X)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor.unsqueeze(0))
        return y_pred.squeeze().cpu().numpy()

    def summary(self):
        print("AutoformerModel: Autoformer wrapper")
        print(self.model)

    def infer(self):
        self.model.eval()

    def shap_interpret(self, X_sample):
        print("SHAP not directly supported for Autoformer. Showing attention weights if available.")
        # If the model exposes attention weights, plot them here
        if hasattr(self.model, 'get_attention_weights'):
            attn = self.model.get_attention_weights(X_sample)
            import matplotlib.pyplot as plt
            plt.imshow(attn, cmap='viridis')
            plt.title('Autoformer Attention Weights')
            plt.colorbar()
            plt.show()
        else:
            print("No attention weights available.")

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'autoformer_model.pt'))
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(os.path.join(path, 'autoformer_model.pt')))
        with open(os.path.join(path, 'config.json'), 'r') as f:
            self.config = json.load(f)
        self.fitted = True 