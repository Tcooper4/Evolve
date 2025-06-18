"""LSTM model for time series forecasting."""

# Standard library imports
from typing import Dict, Any, Optional, Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap

# Local imports
from trading.models.base_model import BaseModel, ValidationError, ModelRegistry

@ModelRegistry.register('LSTM')
class LSTMForecaster(BaseModel):
    """LSTM model for time series forecasting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LSTM forecaster.
        
        Args:
            config: Configuration dictionary containing:
                - input_size: Size of input features (default: 2)
                - hidden_size: Size of hidden layers (default: 64)
                - num_layers: Number of LSTM layers (default: 2)
                - dropout: Dropout rate (default: 0.2)
                - bidirectional: Whether to use bidirectional LSTM (default: False)
                - sequence_length: Length of input sequences (default: 10)
                - feature_columns: List of feature column names (default: ['close', 'volume'])
                - target_column: Name of target column (default: 'close')
                - learning_rate: Learning rate (default: 0.001)
                - use_lr_scheduler: Whether to use learning rate scheduler (default: True)
        """
        if config is None:
            config = {}
        
        # Set default configuration
        default_config = {
            'input_size': 2,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': False,
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
        required_params = ['input_size', 'hidden_size', 'num_layers',
                         'dropout', 'sequence_length', 'feature_columns',
                         'target_column']
        for param in required_params:
            if param not in self.config:
                raise ValidationError(f"Missing required parameter: {param}")
        
        # Validate parameter values
        if self.config['input_size'] <= 0:
            raise ValidationError("input_size must be positive")
        if self.config['hidden_size'] <= 0:
            raise ValidationError("hidden_size must be positive")
        if self.config['num_layers'] <= 0:
            raise ValidationError("num_layers must be positive")
        if not 0 <= self.config['dropout'] <= 1:
            raise ValidationError("dropout must be between 0 and 1")
        if self.config['sequence_length'] < 2:
            raise ValidationError("sequence_length must be at least 2")
        if not self.config['feature_columns']:
            raise ValidationError("feature_columns cannot be empty")
        if len(self.config['feature_columns']) != self.config['input_size']:
            raise ValidationError(f"Number of feature columns ({len(self.config['feature_columns'])}) "
                                f"must match input_size ({self.config['input_size']})")
    
    def build_model(self) -> nn.Module:
        """Build the LSTM model architecture.
        
        Returns:
            PyTorch model
        """
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'] if self.config['num_layers'] > 1 else 0,
            bidirectional=self.config['bidirectional'],
            batch_first=True
        )
        
        # Calculate output size based on bidirectional flag
        output_size = self.config['hidden_size'] * 2 if self.config['bidirectional'] else self.config['hidden_size']
        
        # Output layer
        self.fc = nn.Linear(output_size, 1)
        
        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        return nn.ModuleList([self.lstm, self.fc])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_hidden = lstm_out[:, -1]
        
        # Output layer
        out = self.fc(last_hidden)
        return out
    
    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training or prediction.
        
        Args:
            data: Input data as pandas DataFrame
            is_training: Whether data is for training
            
        Returns:
            Tuple of (X, y) tensors where:
                X: Shape (batch_size, sequence_length, input_size)
                y: Shape (batch_size, 1)
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