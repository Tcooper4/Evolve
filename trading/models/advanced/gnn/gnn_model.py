"""Graph Neural Network for time series forecasting."""

# Standard library imports
from typing import Dict, Any, Optional, Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from trading.models.base_model import BaseModel, ValidationError, ModelRegistry

class GNNLayer(nn.Module):
    """Graph Neural Network layer."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        """Initialize GNN layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout: Dropout rate
        """
        try:
            from torch_geometric.nn import GCNConv
        except ImportError:
            raise ImportError("torch_geometric is not installed. Please install it with 'pip install torch-geometric'.")
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN layer.
        
        Args:
            x: Node features tensor of shape (num_nodes, in_channels)
            edge_index: Graph connectivity tensor of shape (2, num_edges)
            
        Returns:
            Updated node features tensor of shape (num_nodes, out_channels)
        """
        try:
            from torch_geometric.nn import GCNConv
        except ImportError:
            raise ImportError("torch_geometric is not installed. Please install it with 'pip install torch-geometric'.")
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

@ModelRegistry.register('GNN')
class GNNForecaster(BaseModel):
    """Graph Neural Network forecaster for time series data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize GNN forecaster.
        
        Args:
            config: Configuration dictionary containing:
                - input_size: Size of input features (default: 2)
                - hidden_size: Size of hidden layers (default: 64)
                - output_size: Size of output features (default: 1)
                - num_layers: Number of GNN layers (default: 2)
                - dropout: Dropout rate (default: 0.2)
                - sequence_length: Length of input sequences (default: 10)
                - feature_columns: List of feature column names (default: ['close', 'volume'])
                - target_column: Name of target column (default: 'close')
                - learning_rate: Learning rate (default: 0.001)
                - use_lr_scheduler: Whether to use learning rate scheduler (default: True)
        """
        try:
            from torch_geometric.nn import global_mean_pool
        except ImportError:
            raise ImportError("torch_geometric is not installed. Please install it with 'pip install torch-geometric'.")
        if config is None:
            config = {}
        
        # Set default configuration
        default_config = {
            'input_size': 2,
            'hidden_size': 64,
            'output_size': 1,
            'num_layers': 2,
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
        required_params = ['input_size', 'hidden_size', 'output_size', 'num_layers',
                         'dropout', 'sequence_length', 'feature_columns', 'target_column']
        for param in required_params:
            if param not in self.config:
                raise ValidationError(f"Missing required parameter: {param}")
        
        # Validate parameter values
        if self.config['input_size'] <= 0:
            raise ValidationError("input_size must be positive")
        if self.config['hidden_size'] <= 0:
            raise ValidationError("hidden_size must be positive")
        if self.config['output_size'] <= 0:
            raise ValidationError("output_size must be positive")
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
        """Build the GNN model architecture.
        
        Returns:
            PyTorch model
        """
        layers = []
        
        # Input layer
        layers.append(GNNLayer(
            self.config['input_size'],
            self.config['hidden_size'],
            self.config['dropout']
        ))
        
        # Hidden layers
        for _ in range(self.config['num_layers'] - 1):
            layers.append(GNNLayer(
                self.config['hidden_size'],
                self.config['hidden_size'],
                self.config['dropout']
            ))
        
        # Output layer
        self.gnn_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(self.config['hidden_size'], self.config['output_size'])
        
        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        return nn.ModuleList([*self.gnn_layers, self.fc])
    
    def _create_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create graph structure from time series data.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Tuple of (node_features, edge_index) where:
                node_features: Shape (batch_size * seq_len, input_size)
                edge_index: Shape (2, num_edges)
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape to node features
        node_features = x.reshape(-1, self.config['input_size'])
        
        # Create temporal edges
        edges = []
        for b in range(batch_size):
            for i in range(seq_len):
                # Connect to previous time step
                if i > 0:
                    edges.append([b * seq_len + i, b * seq_len + i - 1])
                # Connect to next time step
                if i < seq_len - 1:
                    edges.append([b * seq_len + i, b * seq_len + i + 1])
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        
        return node_features, edge_index
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Create graph structure
        node_features, edge_index = self._create_graph(x)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            node_features = layer(node_features, edge_index)
        
        # Global pooling
        batch_size = x.size(0)
        seq_len = x.size(1)
        node_features = node_features.reshape(batch_size, seq_len, -1)
        pooled = global_mean_pool(node_features, torch.zeros(batch_size, dtype=torch.long, device=self.device))
        
        # Output layer
        out = self.fc(pooled)
        return out
    
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

    def save(self, path: str) -> None:
        """Save model state.
        
        Args:
            path: Path to save model state
        """
        if self.model is None:
            raise ValueError("Model not initialized")
            
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history,
            'best_loss': self.best_loss,
            'X_mean': self.X_mean if hasattr(self, 'X_mean') else None,
            'X_std': self.X_std if hasattr(self, 'X_std') else None,
            'y_mean': self.y_mean if hasattr(self, 'y_mean') else None,
            'y_std': self.y_std if hasattr(self, 'y_std') else None
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load model state.
        
        Args:
            path: Path to load model state from
        """
        state = torch.load(path, weights_only=False)
        self.config = state['config']
        self.history = state['history']
        self.best_loss = state['best_loss']
        
        if self.model is not None:
            self.model.load_state_dict(state['model_state'])
            
        if self.optimizer is not None and state['optimizer_state'] is not None:
            self.optimizer.load_state_dict(state['optimizer_state'])
            
        if self.scheduler is not None and state['scheduler_state'] is not None:
            self.scheduler.load_state_dict(state['scheduler_state'])
            
        # Load normalization parameters
        if state['X_mean'] is not None:
            self.X_mean = state['X_mean']
            self.X_std = state['X_std']
            self.y_mean = state['y_mean']
            self.y_std = state['y_std']
        
    def fit(self, train_data: Tuple[torch.Tensor, torch.Tensor],
            val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs) -> Dict[str, Any]:
        """Train the GNN model.
        
        Args:
            train_data: Tuple of (features, adjacency_matrix) for training
            val_data: Optional tuple of (features, adjacency_matrix) for validation
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training history
        """
        train_features, train_adj = train_data
        if val_data is not None:
            val_features, val_adj = val_data
            
        # Training setup
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
        criterion = nn.MSELoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(self.config.get('epochs', 100)):
            # Training step
            self.train()
            optimizer.zero_grad()
            train_pred = self(train_features)
            train_loss = criterion(train_pred, train_features)
            train_loss.backward()
            optimizer.step()
            
            # Validation step
            if val_data is not None:
                self.eval()
                with torch.no_grad():
                    val_pred = self(val_features)
                    val_loss = criterion(val_pred, val_features)
                    
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
                history['val_loss'].append(val_loss.item())
                
            history['train_loss'].append(train_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss = {train_loss.item():.4f}, "
                      f"val_loss = {val_loss.item():.4f if val_data is not None else 'N/A'}")
                
        return history
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using the GNN model.
        
        Args:
            x: Input tensor of shape (batch_size, num_nodes, input_size)
            
        Returns:
            Predicted values
        """
        self.eval()
        with torch.no_grad():
            return self(x)

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
                new_row[self.config.get('target_column', 'close')] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row
            
            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.8,  # GNN confidence
                'model': 'GNN',
                'horizon': horizon,
                'feature_columns': self.config.get('feature_columns', []),
                'target_column': self.config.get('target_column', 'close')
            }
            
        except Exception as e:
            import logging
            logging.error(f"Error in GNN model forecast: {e}")
            raise RuntimeError(f"GNN model forecasting failed: {e}")

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