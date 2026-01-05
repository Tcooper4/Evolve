"""
Graph Neural Network (GNN) Model for Market Forecasting

GNN models relationships between assets using graph structures.
Useful for:
- Multi-asset forecasting with dependencies
- Modeling sector/industry relationships
- Capturing market contagion effects
- Portfolio optimization with relationship awareness
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer for processing graph-structured data."""
    
    def __init__(self, in_features: int, out_features: int):
        """Initialize graph convolution layer.
        
        Args:
            in_features: Number of input features per node
            out_features: Number of output features per node
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GraphConvLayer")
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph convolution.
        
        Args:
            x: Node features [num_nodes, in_features]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Apply linear transformation
        x = self.linear(x)
        
        # Aggregate neighbor information
        # Normalize adjacency matrix
        degree = adjacency.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adjacency_normalized = adjacency / degree
        
        # Message passing: aggregate features from neighbors
        x = torch.matmul(adjacency_normalized, x)
        
        return x


class GNNModel(nn.Module):
    """Graph Neural Network for time series forecasting."""
    
    def __init__(
        self,
        num_nodes: int,
        node_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_lstm: bool = True
    ):
        """Initialize GNN model.
        
        Args:
            num_nodes: Number of nodes in graph (e.g., number of stocks)
            node_features: Number of features per node (e.g., OHLCV = 5)
            hidden_size: Hidden layer size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_lstm: Whether to use LSTM for temporal modeling
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GNNModel")
        super(GNNModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.use_lstm = use_lstm
        
        # Graph convolutional layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer
        self.gnn_layers.append(GraphConvLayer(node_features, hidden_size))
        
        # Middle layers
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GraphConvLayer(hidden_size, hidden_size))
        
        # Optional LSTM for temporal dependencies
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        temporal: bool = True
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [batch, seq_len, num_nodes, node_features] or
               [batch, num_nodes, node_features] if not temporal
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            temporal: Whether to process temporal dimension with LSTM
            
        Returns:
            Predictions [batch, num_nodes] or [batch, seq_len, num_nodes]
        """
        batch_size = x.size(0)
        
        if temporal and len(x.shape) == 4:
            # Temporal data: [batch, seq_len, num_nodes, features]
            seq_len = x.size(1)
            
            # Process each timestep through GNN
            outputs = []
            for t in range(seq_len):
                x_t = x[:, t, :, :]  # [batch, num_nodes, features]
                
                # Apply GNN layers
                for i, gnn_layer in enumerate(self.gnn_layers):
                    x_t = gnn_layer(x_t, adjacency)
                    x_t = F.relu(x_t)
                    x_t = self.dropout(x_t)
                
                outputs.append(x_t)
            
            # Stack outputs: [batch, seq_len, num_nodes, hidden]
            h = torch.stack(outputs, dim=1)
            
            # Apply LSTM if enabled
            if self.use_lstm:
                # Reshape for LSTM: [batch*num_nodes, seq_len, hidden]
                h_reshaped = h.permute(0, 2, 1, 3).reshape(
                    batch_size * self.num_nodes, seq_len, self.hidden_size
                )
                
                lstm_out, _ = self.lstm(h_reshaped)
                
                # Take last timestep and reshape back
                h = lstm_out[:, -1, :].reshape(batch_size, self.num_nodes, self.hidden_size)
            else:
                # Just take last timestep
                h = h[:, -1, :, :]
        else:
            # Non-temporal: [batch, num_nodes, features]
            h = x
            
            # Apply GNN layers
            for i, gnn_layer in enumerate(self.gnn_layers):
                h = gnn_layer(h, adjacency)
                h = F.relu(h)
                h = self.dropout(h)
        
        # Output layer: [batch, num_nodes, hidden] -> [batch, num_nodes, 1]
        output = self.fc(h)
        
        # Squeeze last dimension: [batch, num_nodes]
        output = output.squeeze(-1)
        
        return output


class GNNForecaster:
    """Graph Neural Network Forecaster for multi-asset time series.
    
    Models relationships between assets using graph structure.
    """
    
    def __init__(
        self,
        num_assets: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_length: int = 30,
        learning_rate: float = 0.001,
        correlation_threshold: float = 0.5
    ):
        """Initialize GNN forecaster.
        
        Args:
            num_assets: Number of assets to model
            hidden_size: Hidden layer size
            num_layers: Number of GNN layers
            seq_length: Sequence length for temporal modeling
            learning_rate: Learning rate
            correlation_threshold: Threshold for creating edges (0-1)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GNN models")
        
        self.num_assets = num_assets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.correlation_threshold = correlation_threshold
        
        self.model = None
        self.scaler = StandardScaler()
        self.adjacency_matrix = None
        self.is_fitted = False
        
    def _build_adjacency_matrix(self, data: pd.DataFrame) -> torch.Tensor:
        """Build adjacency matrix from correlation.
        
        Args:
            data: Multi-asset price data
            
        Returns:
            Adjacency matrix as tensor
        """
        # Calculate correlation matrix
        correlation = data.corr().values
        
        # Threshold correlations to create edges
        adjacency = (np.abs(correlation) > self.correlation_threshold).astype(float)
        
        # Add self-loops
        np.fill_diagonal(adjacency, 1.0)
        
        return torch.FloatTensor(adjacency)
    
    def _prepare_graph_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for GNN training.
        
        Args:
            data: Multi-asset price data [time, assets]
            
        Returns:
            Tuple of (features, targets)
        """
        # Assume data has columns for each asset
        num_timesteps = len(data)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data.values)
        
        # Create sequences
        X_list = []
        y_list = []
        
        for i in range(num_timesteps - self.seq_length):
            # Features: [seq_length, num_assets]
            X_seq = scaled_data[i:i + self.seq_length]
            
            # Target: next values for all assets
            y_seq = scaled_data[i + self.seq_length]
            
            X_list.append(X_seq)
            y_list.append(y_seq)
        
        # Stack into tensors
        # X: [num_samples, seq_length, num_assets, 1]
        X = torch.FloatTensor(np.array(X_list)).unsqueeze(-1)
        
        # y: [num_samples, num_assets]
        y = torch.FloatTensor(np.array(y_list))
        
        return X, y
    
    def fit(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32
    ) -> 'GNNForecaster':
        """Train the GNN model.
        
        Args:
            data: Multi-asset price data [time, assets]
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            self
        """
        # Build adjacency matrix from correlations
        self.adjacency_matrix = self._build_adjacency_matrix(data)
        
        # Prepare data
        X, y = self._prepare_graph_data(data)
        
        # Create model
        self.model = GNNModel(
            num_nodes=self.num_assets,
            node_features=1,  # Just price for now
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            use_lstm=True
        )
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        num_batches = len(X) // batch_size
        if num_batches == 0:
            num_batches = 1
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            indices = torch.randperm(len(X))
            
            for i in range(num_batches):
                # Get batch
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                if len(batch_indices) == 0:
                    continue
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(batch_X, self.adjacency_matrix, temporal=True)
                
                # Calculate loss
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Generate predictions for all assets.
        
        Args:
            data: Recent price data [time, assets]
            horizon: Forecast horizon (only 1 supported currently)
            
        Returns:
            Predictions for each asset
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        # Take last sequence
        recent_data = data.values[-self.seq_length:]
        scaled_data = self.scaler.transform(recent_data)
        
        # Prepare input: [1, seq_length, num_assets, 1]
        x = torch.FloatTensor(scaled_data).unsqueeze(0).unsqueeze(-1)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(x, self.adjacency_matrix, temporal=True)
        
        # Inverse transform
        predictions = predictions.numpy()
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        target_asset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate forecast for assets.
        
        Args:
            data: Historical price data
            horizon: Forecast horizon
            target_asset: Specific asset to forecast (or None for all)
            
        Returns:
            Forecast dictionary
        """
        # For simplicity, do multi-step forecasting by iterating
        predictions = []
        current_data = data.copy()
        
        self.model.eval()
        
        for step in range(horizon):
            # Predict next step
            next_pred = self.predict(current_data, horizon=1)
            predictions.append(next_pred)
            
            # Update data with prediction
            new_row = pd.DataFrame([next_pred], columns=data.columns, index=[current_data.index[-1] + pd.Timedelta(days=1)])
            current_data = pd.concat([current_data.iloc[1:], new_row])
        
        predictions = np.array(predictions)
        
        # If target asset specified, extract it
        if target_asset and target_asset in data.columns:
            asset_idx = data.columns.get_loc(target_asset)
            forecast_values = predictions[:, asset_idx]
        else:
            # Return predictions for first asset
            forecast_values = predictions[:, 0]
        
        # Create dates
        last_date = data.index[-1] if hasattr(data.index, 'freq') else pd.Timestamp.now()
        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
        
        return {
            'forecast': forecast_values,
            'all_assets': predictions,  # Predictions for all assets
            'dates': forecast_dates,
            'model_type': 'GNN',
            'horizon': horizon,
            'num_assets': self.num_assets,
            'confidence': np.full(horizon, 0.75)  # Conservative confidence
        }
    
    def get_relationship_matrix(self) -> np.ndarray:
        """Get the learned relationship matrix.
        
        Returns:
            Adjacency matrix showing asset relationships
        """
        if self.adjacency_matrix is None:
            raise ValueError("Model must be fitted first")
        
        return self.adjacency_matrix.numpy()


# For backward compatibility
GraphNeuralNetwork = GNNForecaster

