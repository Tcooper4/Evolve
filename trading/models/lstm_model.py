import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List, Union
from .base_model import BaseModel
from sklearn.preprocessing import StandardScaler

class LSTMModel(nn.Module):
    """A class to handle LSTM model for time series prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize the LSTM model.

        Args:
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            output_dim (int): The output dimension.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass of the LSTM model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def train_model(self, data: pd.DataFrame, target: pd.Series, epochs: int) -> None:
        """Train the LSTM model.

        Args:
            data (pd.DataFrame): The input data.
            target (pd.Series): The target data.
            epochs (int): The number of epochs to train for.
        """
        # Placeholder for training logic
        pass

    def predict(self, data: pd.DataFrame) -> torch.Tensor:
        """Predict using the LSTM model.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            torch.Tensor: The predicted output.
        """
        # Placeholder for prediction logic
        return torch.tensor([])

class LSTMForecaster(BaseModel):
    """LSTM-based forecasting model with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LSTM model with configuration."""
        super().__init__(config)
        self._validate_config()
        self.model = self.build_model()
        self._init_weights()
    
    def build_model(self) -> nn.Module:
        """Build and return the LSTM model.
        
        Returns:
            nn.Module: The built LSTM model.
        """
        model = nn.Sequential(
            nn.LSTM(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                batch_first=True,
                dropout=self.config['dropout'] if self.config['num_layers'] > 1 else 0
            ),
            nn.Linear(self.config['hidden_size'], 1)
        )
        
        # Add attention if enabled
        if self.config.get('use_attention', False):
            model.add_module('attention', nn.MultiheadAttention(
                embed_dim=self.config['hidden_size'],
                num_heads=self.config.get('num_attention_heads', 4),
                dropout=self.config.get('attention_dropout', 0.1)
            ))
        
        # Add batch normalization if enabled
        if self.config.get('use_batch_norm', False):
            model.add_module('batch_norm', nn.BatchNorm1d(self.config['hidden_size']))
        
        # Add layer normalization if enabled
        if self.config.get('use_layer_norm', False):
            model.add_module('layer_norm', nn.LayerNorm(self.config['hidden_size']))
        
        # Add dropout if enabled
        if self.config.get('additional_dropout', 0) > 0:
            model.add_module('dropout', nn.Dropout(self.config['additional_dropout']))
        
        return model
    
    def _validate_config(self):
        """Validate model configuration."""
        required_keys = [
            'input_size', 'hidden_size', 'num_layers', 'dropout',
            'sequence_length', 'feature_columns', 'target_column'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate sequence length
        if self.config['sequence_length'] <= 0:
            raise ValueError("sequence_length must be positive")
        if self.config['sequence_length'] > self.config.get('max_sequence_length', 100):
            raise ValueError(f"sequence_length exceeds maximum allowed value of {self.config.get('max_sequence_length', 100)}")
        
        # Validate feature columns
        if not self.config['feature_columns']:
            raise ValueError("feature_columns cannot be empty")
        if self.config['target_column'] not in self.config['feature_columns']:
            raise ValueError("target_column must be in feature_columns")
        
        # Validate batch size limits
        if 'max_batch_size' in self.config and self.config['max_batch_size'] <= 0:
            raise ValueError("max_batch_size must be positive")
        
        # Validate epoch limits
        if 'max_epochs' in self.config and self.config['max_epochs'] <= 0:
            raise ValueError("max_epochs must be positive")
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with error handling and memory management."""
        try:
            # Validate input
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be torch.Tensor")
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            
            # Check sequence length
            if x.size(1) > self.config.get('max_sequence_length', 100):
                raise ValueError(f"Input sequence length {x.size(1)} exceeds maximum allowed value of {self.config.get('max_sequence_length', 100)}")
            
            # Check batch size
            if 'max_batch_size' in self.config and x.size(0) > self.config['max_batch_size']:
                raise ValueError(f"Batch size {x.size(0)} exceeds maximum allowed value of {self.config['max_batch_size']}")
            
            return self.model(x)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise MemoryError("GPU out of memory during forward pass")
            raise e
    
    def _prepare_data(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input data for training or prediction."""
        try:
            # Validate input data
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Input data must be pandas DataFrame")
            if not all(col in data.columns for col in self.config['feature_columns']):
                raise ValueError("Missing required feature columns")
            
            # Check data size
            if len(data) < self.config['sequence_length']:
                raise ValueError(f"Data length {len(data)} is less than sequence length {self.config['sequence_length']}")
            
            # Normalize data
            if is_training:
                self.scaler = StandardScaler()
                normalized_data = self.scaler.fit_transform(data[self.config['feature_columns']])
            else:
                if not hasattr(self, 'scaler'):
                    raise ValueError("Model must be trained before prediction")
                normalized_data = self.scaler.transform(data[self.config['feature_columns']])
            
            # Convert to tensor
            X = torch.FloatTensor(normalized_data)
            
            # Create sequences
            X = self._create_sequences(X)
            
            if is_training:
                # Get target values
                target_idx = self.config['feature_columns'].index(self.config['target_column'])
                y = X[:, -1, target_idx].view(-1, 1)
                return X, y
            else:
                return X, None
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, batch_size: int = 32, gradient_accumulation_steps: int = 1) -> Dict[str, float]:
        """Train the model with gradient accumulation and memory management."""
        try:
            # Validate inputs
            if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise TypeError("Inputs must be torch.Tensor")
            if X.shape[0] != y.shape[0]:
                raise ValueError("Input and target batch sizes must match")
            
            # Validate epochs
            if epochs > self.config.get('max_epochs', 100):
                raise ValueError(f"Number of epochs {epochs} exceeds maximum allowed value of {self.config.get('max_epochs', 100)}")
            
            # Validate batch size
            if batch_size > self.config.get('max_batch_size', 512):
                raise ValueError(f"Batch size {batch_size} exceeds maximum allowed value of {self.config.get('max_batch_size', 512)}")
            
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
            criterion = nn.MSELoss()
            
            # Initialize metrics
            metrics = {
                'loss': float('inf'),
                'learning_rate': self.config['learning_rate']
            }
            
            # Training loop
            for epoch in range(epochs):
                epoch_loss = 0
                optimizer.zero_grad()
                
                for i in range(0, len(X), batch_size):
                    try:
                        # Get batch
                        batch_X = X[i:i+batch_size]
                        batch_y = y[i:i+batch_size]
                        
                        # Forward pass
                        pred = self(batch_X)
                        loss = criterion(pred, batch_y)
                        
                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        
                        # Update weights if we've accumulated enough gradients
                        if (i + batch_size) % (batch_size * gradient_accumulation_steps) == 0:
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                        
                        epoch_loss += loss.item() * gradient_accumulation_steps
                        
                        # Check for NaN loss
                        if torch.isnan(loss):
                            raise ValueError("NaN loss detected")
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise MemoryError("GPU out of memory during training")
                        raise e
                
                # Update metrics
                metrics['loss'] = epoch_loss / len(X)
                metrics['learning_rate'] = optimizer.param_groups[0]['lr']
                
                # Early stopping check
                if self._check_early_stopping(metrics['loss']):
                    break
            
            return metrics
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions with error handling."""
        try:
            # Validate input
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Input data must be pandas DataFrame")
            
            # Check data size
            if len(data) < self.config['sequence_length']:
                raise ValueError(f"Data length {len(data)} is less than sequence length {self.config['sequence_length']}")
            
            # Prepare data
            X, _ = self._prepare_data(data, is_training=False)
            
            # Make predictions
            self.eval()
            with torch.no_grad():
                predictions = self(X)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(
                np.column_stack([predictions.numpy(), np.zeros((len(predictions), len(self.config['feature_columns']) - 1))])
            )[:, 0]
            
            return predictions
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    def save(self, path: str) -> None:
        """Save model state.
        
        Args:
            path: Path to save model state
        """
        state = {
            'model_state': self.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_model_state': getattr(self, 'best_model_state', None),
            'best_val_loss': getattr(self, 'best_val_loss', None),
            'X_mean': getattr(self, 'X_mean', None),
            'X_std': getattr(self, 'X_std', None),
            'y_mean': getattr(self, 'y_mean', None),
            'y_std': getattr(self, 'y_std', None)
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load model state.
        
        Args:
            path: Path to load model state from
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state['model_state'])
        self.config = state['config']
        self.history = state['history']
        self.best_model_state = state.get('best_model_state', None)
        self.best_val_loss = state.get('best_val_loss', None)
        self.X_mean = state.get('X_mean', None)
        self.X_std = state.get('X_std', None)
        self.y_mean = state.get('y_mean', None)
        self.y_std = state.get('y_std', None)

    def _create_sequences(self, data: torch.Tensor) -> torch.Tensor:
        """Create sequences for LSTM input.
        
        Args:
            data: Input tensor of shape (n_samples, n_features)
            
        Returns:
            Tensor of shape (n_sequences, sequence_length, n_features)
        """
        try:
            # Validate input
            if not isinstance(data, torch.Tensor):
                raise TypeError("Input must be torch.Tensor")
            if data.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got {data.dim()}D")
            
            # Calculate number of sequences
            n_samples = data.size(0)
            n_sequences = n_samples - self.config['sequence_length'] + 1
            
            if n_sequences <= 0:
                raise ValueError(f"Data length {n_samples} is less than sequence length {self.config['sequence_length']}")
            
            # Create sequences
            sequences = []
            for i in range(n_sequences):
                sequence = data[i:i + self.config['sequence_length']]
                sequences.append(sequence)
            
            # Stack sequences
            return torch.stack(sequences)
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e 