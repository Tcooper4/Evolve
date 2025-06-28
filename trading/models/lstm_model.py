"""LSTM-based forecasting model with advanced features."""

# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from .base_model import BaseModel

class LSTMModel(nn.Module):
    """A class to handle LSTM model for time series prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 1, dropout: float = 0.0):
        """Initialize the LSTM model.

        Args:
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            output_dim (int): The output dimension.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initialize LSTM with specified parameters
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Move input to device
        x = x.to(self.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use only the final timestep's output for prediction
        # This is common in sequence-to-value prediction tasks
        return self.fc(lstm_out[:, -1, :])

    def train_model(self, data: pd.DataFrame, target: pd.Series, 
                   epochs: int, batch_size: int = 32, 
                   learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Train the LSTM model.

        Args:
            data (pd.DataFrame): Input features
            target (pd.Series): Target values
            epochs (int): Number of training epochs
            batch_size (int, optional): Batch size. Defaults to 32.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.

        Returns:
            Dict[str, List[float]]: Training history with loss values
        """
        # Initialize optimizer and loss function
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(data)
        y_scaled = self.scaler.fit_transform(target.values.reshape(-1, 1))
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Create sequences
        X_seq = self._create_sequences(X_tensor)
        y_seq = y_tensor[self.config['sequence_length']:]
        
        # Create DataLoader
        dataset = TensorDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training history
        history = {'train_loss': []}
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # Move batch to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            history['train_loss'].append(avg_epoch_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
        
        return history

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict using the LSTM model.

        Args:
            data (pd.DataFrame): Input features

        Returns:
            np.ndarray: Predicted values
        """
        # Scale the data
        X_scaled = self.scaler.transform(data)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Create sequences
        X_seq = self._create_sequences(X_tensor)
        
        # Move to device
        X_seq = X_seq.to(self.device)
        
        # Set model to evaluation mode
        self.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self(X_seq)
        
        # Convert predictions to numpy array
        predictions = predictions.cpu().numpy()
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions

    def _create_sequences(self, data: torch.Tensor) -> torch.Tensor:
        """Create sequences for LSTM input.

        Args:
            data (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Sequences tensor
        """
        sequences = []
        for i in range(len(data) - self.config['sequence_length']):
            sequences.append(data[i:i + self.config['sequence_length']])
        return torch.stack(sequences)

class LSTMForecaster(BaseModel):
    """LSTM-based forecasting model with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LSTM model with configuration.
        
        Args:
            config (Dict[str, Any]): Model configuration dictionary
        """
        super().__init__(config)
        self._validate_config()
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = self.build_model()
        self.model.to(self.device)
        
        # Initialize weights
        self._init_weights()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def build_model(self) -> nn.Module:
        """Build and return the LSTM model.
        
        Returns:
            nn.Module: The built LSTM model
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
        """Forward pass with error handling and memory management.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
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
            
            # Move input to device
            x = x.to(self.device)
            
            return self.model(x)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise MemoryError("GPU out of memory during forward pass")
            raise e
    
    def _prepare_data(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare input data for training or prediction.
        
        Args:
            data (pd.DataFrame): Input data
            is_training (bool, optional): Whether this is for training. Defaults to True.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Prepared input and target tensors
        """
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
            X_seq = self._create_sequences(X)
            
            if is_training:
                # Get target values
                y = torch.FloatTensor(normalized_data[self.config['sequence_length']:, 
                                                    self.config['feature_columns'].index(self.config['target_column'])])
                return X_seq, y
            else:
                return X_seq, None
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _create_sequences(self, data: torch.Tensor) -> torch.Tensor:
        """Create sequences for LSTM input.
        
        Args:
            data (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Sequences tensor
        """
        sequences = []
        for i in range(len(data) - self.config['sequence_length']):
            sequences.append(data[i:i + self.config['sequence_length']])
        return torch.stack(sequences)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            epochs: int = 100, batch_size: int = 32, 
            learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target values
            epochs (int, optional): Number of epochs. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 32.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Prepare data
        X_seq, y_seq = self._prepare_data(X, is_training=True)
        
        # Create DataLoader
        dataset = TensorDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        history = {'train_loss': []}
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # Move batch to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            history['train_loss'].append(avg_epoch_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
        
        return history
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            data (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        # Prepare data
        X_seq, _ = self._prepare_data(data, is_training=False)
        
        # Move to device
        X_seq = X_seq.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_seq)
        
        # Convert predictions to numpy array
        predictions = predictions.cpu().numpy()
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save the model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler
        }, path)
    
    def load(self, path: str) -> None:
        """Load the model.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.model.to(self.device)

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
                
                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row[self.config['target_column']] = pred[-1]
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row
            
            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.85,  # Placeholder confidence
                'model': 'LSTM',
                'horizon': horizon
            }
            
        except Exception as e:
            self.logger.error(f"Error in LSTM model forecast: {e}")
            raise RuntimeError(f"LSTM model forecasting failed: {e}")

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
            plt.title('LSTM Model Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting LSTM results: {e}")
            print(f"Could not plot results: {e}") 