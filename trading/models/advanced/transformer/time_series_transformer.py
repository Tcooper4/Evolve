import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from trading.models.base_model import BaseModel
from typing import Dict, Any, Optional, Tuple

class TransformerForecaster(BaseModel):
    """Transformer model for time series forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the transformer model.
        
        Args:
            config: Model configuration dictionary containing:
                - input_dim: Input dimension
                - d_model: Model dimension
                - nhead: Number of attention heads
                - num_encoder_layers: Number of encoder layers
                - dim_feedforward: Dimension of feedforward network
                - dropout: Dropout rate
                - activation: Activation function
                - batch_first: Whether batch dimension is first
                - sequence_length: Length of input sequences
                - feature_columns: List of feature column names
                - target_column: Name of target column
        """
        super().__init__(config)
        
        self.input_dim = config['input_dim']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.dim_feedforward = config.get('dim_feedforward', 4 * self.d_model)
        self.dropout = config.get('dropout', 0.1)
        self.activation = config.get('activation', 'gelu')
        self.batch_first = config.get('batch_first', True)
        self.sequence_length = config.get('sequence_length', 10)
        self.feature_columns = config.get('feature_columns', [])
        self.target_column = config.get('target_column', 'Close')
        
        # Initialize model components
        self._setup_model()
    
    def _prepare_data(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Prepare data for the transformer model.
        
        Args:
            data: Input DataFrame with time series data
            
        Returns:
            Dictionary containing:
                - X: Input features tensor of shape (batch_size, seq_len, input_dim)
                - y: Target values tensor of shape (batch_size, seq_len, 1)
        """
        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        if not self.feature_columns:
            raise ValueError("Feature columns must be specified in config")
        
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Check for missing values
        if data[self.feature_columns + [self.target_column]].isna().any().any():
            raise ValueError("Input data contains missing values")
        
        # Extract features and target
        X = data[self.feature_columns].values
        y = data[self.target_column].values.reshape(-1, 1)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(data) - self.sequence_length):
            X_sequences.append(X[i:i + self.sequence_length])
            y_sequences.append(y[i + self.sequence_length])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.array(X_sequences))
        y_tensor = torch.FloatTensor(np.array(y_sequences))
        
        # Normalize data if not already normalized
        if not hasattr(self, 'feature_means'):
            self.feature_means = X_tensor.mean(dim=(0, 1))
            self.feature_stds = X_tensor.std(dim=(0, 1))
            self.target_mean = y_tensor.mean()
            self.target_std = y_tensor.std()
        
        X_tensor = (X_tensor - self.feature_means) / (self.feature_stds + 1e-8)
        y_tensor = (y_tensor - self.target_mean) / (self.target_std + 1e-8)
        
        return {
            'X': X_tensor,
            'y': y_tensor
        }
    
    def _setup_model(self):
        """Set up the transformer model architecture."""
        # Input projection layer
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.d_model,
            dropout=self.dropout,
            max_len=self.sequence_length
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=self.batch_first
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(self.d_model, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, 1)
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask for causal attention
        seq_len = x.size(1)
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer encoder
        x = self.transformer_encoder(x, mask)
        
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence.
        
        Args:
            sz: Size of the sequence
            
        Returns:
            Mask tensor
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) 