"""Temporal Convolutional Network for time series forecasting."""

# Standard library imports
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local imports
from .base_model import BaseModel

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.dropout1,
                               self.conv2, self.bn2, nn.ReLU(), self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

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
        """
        super().__init__(config)
        self._validate_config()
        self._setup_model()
        
    def _validate_config(self) -> None:
        """Validate model configuration."""
        self.input_size = self.config.get('input_size', 2)
        self.output_size = self.config.get('output_size', 1)
        self.num_channels = self.config.get('num_channels', [64, 128, 256])
        self.kernel_size = self.config.get('kernel_size', 3)
        self.dropout = self.config.get('dropout', 0.2)
        self.sequence_length = self.config.get('sequence_length', 10)
        self.feature_columns = self.config.get('feature_columns', ['close', 'volume'])
        self.target_column = self.config.get('target_column', 'close')
        
        # Validate sequence length
        if self.sequence_length < 2:
            raise ValueError("Sequence length must be at least 2")
            
        # Validate feature columns
        if len(self.feature_columns) != self.input_size:
            raise ValueError(f"Number of feature columns ({len(self.feature_columns)}) must match input_size ({self.input_size})")
            
        # Validate target column
        if self.target_column not in self.feature_columns:
            raise ValueError(f"Target column {self.target_column} must be in feature_columns")
            
    def _setup_model(self) -> None:
        """Setup the TCN model architecture."""
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = self.input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size,
                                   stride=1, dilation=dilation,
                                   padding=(self.kernel_size-1) * dilation,
                                   dropout=self.dropout)]
        
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(self.num_channels[-1], self.output_size)
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
        self._validate_data(data)
        
        # Check if all required columns exist
        missing_cols = [col for col in self.feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert to numpy arrays
        X = data[self.feature_columns].values
        y = data[self.target_column].values[self.sequence_length:]  # Predict next value after sequence
        
        # Create sequences
        X_sequences = []
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:i + self.sequence_length])
        X = np.array(X_sequences)
        
        # Normalize
        if is_training:
            self.X_mean = X.mean(axis=(0, 1))  # Mean across batch and sequence
            self.X_std = X.std(axis=(0, 1))    # Std across batch and sequence
            self.y_mean = y.mean()
            self.y_std = y.std()
            
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(-1)
        
        # Move to device
        X = self._to_device(X)
        y = self._to_device(y)
        
        return X, y 