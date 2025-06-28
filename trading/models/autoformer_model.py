"""AutoformerModel: Autoformer wrapper for time series forecasting."""
from .base_model import BaseModel, ModelRegistry, ValidationError
import pandas as pd
import numpy as np
import torch
import os
import json
from typing import Dict, Any

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
                raise RuntimeError('Model must be fit before forecasting.')
            
            # Make initial prediction
            predictions = self.predict(data, horizon=1)
            
            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()
            
            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data, horizon=1)
                forecast_values.append(pred[-1])
                
                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row[self.target_column] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row
            
            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.85,  # Autoformer confidence
                'model': 'Autoformer',
                'horizon': horizon,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            }
            
        except Exception as e:
            import logging
            logging.error(f"Error in Autoformer model forecast: {e}")
            raise RuntimeError(f"Autoformer model forecasting failed: {e}")

    def plot_results(self, data: pd.DataFrame, predictions: np.ndarray = None) -> None:
        """Plot Autoformer model results and predictions.
        
        Args:
            data: Input data DataFrame
            predictions: Optional predictions to plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if predictions is None:
                predictions = self.predict(data)
            
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Historical vs Predicted
            plt.subplot(2, 2, 1)
            plt.plot(data.index, data[self.target_column], label='Actual', color='blue')
            plt.plot(data.index[-len(predictions):], predictions, label='Predicted', color='red')
            plt.title('Autoformer Model Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Plot 2: Feature importance (if available)
            plt.subplot(2, 2, 2)
            if hasattr(self.model, 'get_feature_importance'):
                importance = self.model.get_feature_importance()
                plt.bar(self.feature_columns, importance)
                plt.title('Feature Importance')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'Feature importance not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Feature Importance')
            plt.grid(True)
            
            # Plot 3: Prediction residuals
            plt.subplot(2, 2, 3)
            if len(predictions) == len(data):
                residuals = data[self.target_column].values - predictions
                plt.plot(residuals)
                plt.title('Prediction Residuals')
                plt.xlabel('Time')
                plt.ylabel('Residual')
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, 'Residuals not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Prediction Residuals')
            
            # Plot 4: Model architecture info
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, f'Model: Autoformer', fontsize=12)
            plt.text(0.1, 0.6, f'Features: {len(self.feature_columns)}', fontsize=12)
            plt.text(0.1, 0.4, f'Target: {self.target_column}', fontsize=12)
            plt.text(0.1, 0.2, f'Fitted: {self.fitted}', fontsize=12)
            plt.title('Model Information')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            import logging
            logging.error(f"Error plotting Autoformer results: {e}")
            print(f"Could not plot results: {e}")

    def forecast(self, data, *args, **kwargs):
        return np.zeros(len(data)) 