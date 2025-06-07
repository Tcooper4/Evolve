import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ...base_model import BaseModel
import pandas as pd

class SimpleGNN(nn.Module):
    """Simple Graph Neural Network for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize GNN model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layers
            output_size: Size of output features
        """
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gnn_layer = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GNN.
        
        Args:
            x: Input features tensor of shape (batch_size, num_nodes, input_size)
            adj: Adjacency matrix tensor of shape (num_nodes, num_nodes)
            
        Returns:
            Output tensor of shape (batch_size, num_nodes, output_size)
        """
        # Project input features
        h = self.input_proj(x)
        
        # Apply GNN layer
        h = torch.matmul(adj, h)
        h = self.gnn_layer(h)
        h = F.relu(h)
        
        # Project to output
        out = self.output_proj(h)
        
        return out

class GNNForecaster(BaseModel):
    """Graph Neural Network forecaster for time series data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize GNN forecaster.
        
        Args:
            config: Configuration dictionary containing:
                - input_size: Size of input features (default: 2)
                - hidden_size: Size of hidden layers (default: 64)
                - output_size: Size of output features (default: 1)
                - learning_rate: Learning rate (default: 0.001)
                - epochs: Number of training epochs (default: 100)
                - patience: Early stopping patience (default: 10)
        """
        if config is None:
            config = {}
        default_config = {
            'input_size': 2,
            'hidden_size': 64,
            'output_size': 1,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 10,
            'use_lr_scheduler': True
        }
        default_config.update(config)
        super().__init__(default_config)
        self._setup_model()
        self._setup_optimizer()
        if self.config.get('use_lr_scheduler', False):
            self._setup_scheduler()

    def _setup_model(self) -> None:
        """Setup the GNN model architecture."""
        self.model = SimpleGNN(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            output_size=self.config['output_size']
        ).to(self.device)

    def _setup_optimizer(self) -> None:
        """Setup the optimizer."""
        if self.optimizer is None:
            lr = self.config.get('learning_rate', 0.001)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _setup_scheduler(self) -> None:
        """Setup the learning rate scheduler."""
        if self.config.get('use_lr_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=2,
                threshold=0.01,
                min_lr=1e-6
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_nodes, input_size)
            
        Returns:
            Output tensor
        """
        # Create adjacency matrix based on temporal relationships
        batch_size, seq_len, _ = x.shape
        adj = torch.eye(seq_len, device=self.device)
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
        return self.model(x, adj)

    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> tuple:
        """Prepare data for training or prediction.
        
        Args:
            data: Input data as pandas DataFrame
            is_training: Whether data is for training
            
        Returns:
            Tuple of (X, y) tensors
        """
        # Validate input data
        if data.empty:
            raise ValueError("Input data is empty")
            
        required_columns = ['close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        if data.isnull().any().any():
            raise ValueError("Data contains missing values")
            
        # Convert to numpy arrays
        X = data[['close', 'volume']].values
        y = data['close'].values[1:]  # Predict next day's close
        
        # Normalize
        if is_training:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.y_mean = y.mean()
            self.y_std = y.std()
        
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std
        
        # Convert to tensors and add batch dimension
        X = torch.FloatTensor(X[:-1]).unsqueeze(0)  # (1, seq_len, 2)
        y = torch.FloatTensor(y).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
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