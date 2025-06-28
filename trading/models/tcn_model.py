"""Temporal Convolutional Network for time series forecasting."""

# Standard library imports
from typing import Any, Dict, List, Optional, Tuple
import logging

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from .base_model import BaseModel, ValidationError, ModelRegistry

class TemporalBlock(nn.Module):
    """Temporal block for TCN."""
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2):
        """Initialize temporal block.
        
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout rate
        """
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.dropout1,
            self.conv2, self.bn2, nn.ReLU(), self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block.
        
        Args:
            x: Input tensor of shape (batch_size, n_inputs, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, n_outputs, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# @ModelRegistry.register('TCN')
class TCNModel(BaseModel):
    """Temporal Convolutional Network for time series forecasting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TCN model.
        
        Args:
            config: Model configuration dictionary with the following keys:
                - input_size: Number of input features (default: 2)
                - output_size: Number of output features (default: 1)
                - num_channels: List of channel sizes for each layer (default: [64, 128, 256])
                - kernel_size: Size of convolutional kernel (default: 3)
                - dropout: Dropout rate (default: 0.2)
                - sequence_length: Length of input sequences (default: 10)
                - feature_columns: List of column names to use as features (default: ['close', 'volume'])
                - target_column: Column name to predict (default: 'close')
                - learning_rate: Learning rate (default: 0.001)
                - use_lr_scheduler: Whether to use learning rate scheduler (default: True)
        """
        if config is None:
            config = {}
        
        # Set default configuration
        default_config = {
            'input_size': 2,
            'output_size': 1,
            'num_channels': [64, 128, 256],
            'kernel_size': 3,
            'dropout': 0.2,
            'sequence_length': 10,
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'learning_rate': 0.001,
            'use_lr_scheduler': True
        }
        default_config.update(config)
        
        super().__init__(default_config)
        self._validate_config()
        self._setup_model()
    
    def _validate_config(self) -> None:
        """Validate model configuration.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate required parameters
        required_params = ['input_size', 'output_size', 'num_channels', 'kernel_size',
                         'dropout', 'sequence_length', 'feature_columns', 'target_column']
        for param in required_params:
            if param not in self.config:
                raise ValidationError(f"Missing required parameter: {param}")
        
        # Validate parameter values
        if self.config['input_size'] <= 0:
            raise ValidationError("input_size must be positive")
        if self.config['output_size'] <= 0:
            raise ValidationError("output_size must be positive")
        if not self.config['num_channels']:
            raise ValidationError("num_channels cannot be empty")
        if self.config['kernel_size'] <= 0:
            raise ValidationError("kernel_size must be positive")
        if not 0 <= self.config['dropout'] <= 1:
            raise ValidationError("dropout must be between 0 and 1")
        if self.config['sequence_length'] < 2:
            raise ValidationError("sequence_length must be at least 2")
        if not self.config['feature_columns']:
            raise ValidationError("feature_columns cannot be empty")
        if len(self.config['feature_columns']) != self.config['input_size']:
            raise ValidationError(f"Number of feature columns ({len(self.config['feature_columns'])}) "
                                f"must match input_size ({self.config['input_size']})")
    
    def _setup_model(self) -> None:
        """Setup the TCN model architecture."""
        layers = []
        num_levels = len(self.config['num_channels'])
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = self.config['input_size'] if i == 0 else self.config['num_channels'][i-1]
            out_channels = self.config['num_channels'][i]
            layers += [TemporalBlock(
                in_channels, out_channels, self.config['kernel_size'],
                stride=1, dilation=dilation,
                padding=(self.config['kernel_size']-1) * dilation,
                dropout=self.config['dropout']
            )]
        
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(self.config['num_channels'][-1], self.config['output_size'])
        self.model = nn.Sequential(self.tcn, self.linear)
        self.model = self.model.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        x = x.transpose(1, 2)  # Change to (batch_size, input_size, seq_len)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # Change back to (batch_size, seq_len, num_channels)
        x = self.linear(x[:, -1, :])  # Take last time step
        return x
        
    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training or prediction.
        
        Args:
            data: Input data as pandas DataFrame
            is_training: Whether data is for training
            
        Returns:
            Tuple of (X, y) tensors where:
                X: Shape (batch_size, sequence_length, input_size)
                y: Shape (batch_size, output_size)
        """
        # Validate data
        if data.isnull().any().any():
            raise ValidationError("Data contains missing values")
        
        # Check if all required columns exist
        missing_cols = [col for col in self.config['feature_columns'] 
                       if col not in data.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
            
        # Convert to numpy arrays
        X = data[self.config['feature_columns']].values
        y = data[self.config['target_column']].values[self.config['sequence_length']:]
        
        # Create sequences
        X_sequences = []
        for i in range(len(X) - self.config['sequence_length']):
            X_sequences.append(X[i:i + self.config['sequence_length']])
        X = np.array(X_sequences)
        
        # Normalize
        if is_training:
            self.X_mean = X.mean(axis=(0, 1))
            self.X_std = X.std(axis=(0, 1))
            self.y_mean = y.mean()
            self.y_std = y.std()
            
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(-1)
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        return X, y

    def summary(self):
        super().summary()

    def infer(self):
        super().infer()

    def shap_interpret(self, X_sample):
        """Run SHAP interpretability on a sample batch."""
        try:
            import shap
        except ImportError:
            print("SHAP is not installed. Please install it with 'pip install shap'.")
            return None
        explainer = shap.DeepExplainer(self.model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample.cpu().numpy())

    def test_synthetic(self):
        """Test model on synthetic data."""
        import numpy as np, pandas as pd
        n = 100
        df = pd.DataFrame({
            'close': np.sin(np.linspace(0, 10, n)),
            'volume': np.random.rand(n)
        })
        self.fit(df.iloc[:80], df.iloc[80:])
        y_pred = self.predict(df.iloc[80:])
        print('Synthetic test MSE:', ((y_pred.flatten() - df['close'].iloc[80:].values) ** 2).mean())

    def fit(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32, 
            learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Fit the TCN model to the data.
        
        Args:
            data: Training data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary containing training history
        """
        try:
            # Prepare data
            X, y = self._prepare_data(data, is_training=True)
            
            # Setup optimizer and loss function
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Training history
            history = {'train_loss': []}
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Record loss
                history['train_loss'].append(loss.item())
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            return history
            
        except Exception as e:
            logging.error(f"Error in TCN model fit: {e}")
            raise RuntimeError(f"TCN model fitting failed: {e}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted TCN model.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            Numpy array of predictions
        """
        try:
            # Prepare data
            X, _ = self._prepare_data(data, is_training=False)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X)
            
            # Convert to numpy and denormalize
            predictions = predictions.cpu().numpy()
            predictions = predictions * self.y_std + self.y_mean
            
            return predictions.flatten()
            
        except Exception as e:
            logging.error(f"Error in TCN model predict: {e}")
            raise RuntimeError(f"TCN model prediction failed: {e}")

    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.
        
        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast
            
        Returns:
            Dictionary containing forecast results
        """
        try:
            # Make initial prediction
            predictions = self.predict(data)
            
            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()
            
            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data)
                forecast_values.append(pred[-1])
                
                # Update data for next iteration (simple approach)
                # In a production system, you might want more sophisticated handling
                new_row = current_data.iloc[-1].copy()
                new_row['close'] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row
            
            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.8,  # Placeholder confidence
                'model': 'TCN',
                'horizon': horizon
            }
            
        except Exception as e:
            logging.error(f"Error in TCN model forecast: {e}")
            raise RuntimeError(f"TCN model forecasting failed: {e}")

    def plot_results(self, data: pd.DataFrame, predictions: np.ndarray = None) -> None:
        """Plot model results and predictions.
        
        Args:
            data: Input data DataFrame
            predictions: Optional predictions to plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if predictions is None:
                predictions = self.predict(data)
            
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data[self.config['target_column']], label='Actual', color='blue')
            plt.plot(data.index[self.config['sequence_length']:], predictions, label='Predicted', color='red')
            plt.title('TCN Model Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            logging.error(f"Error plotting TCN results: {e}")
            print(f"Could not plot results: {e}") 