import numpy as np
import torch
from typing import List, Dict, Any, Optional
from trading.models.base_model import BaseModel
import pandas as pd

class EnsembleForecaster(BaseModel):
    """Ensemble model that combines predictions from multiple models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ensemble model.
        
        Args:
            config: Configuration dictionary containing:
                - models: List of model configurations
                - model_weights: Optional list of model weights
                - use_lr_scheduler: Whether to use learning rate scheduler (default: False)
        """
        if config is None:
            config = {}
        default_config = {
            'models': [],
            'model_weights': None,
            'use_lr_scheduler': False
        }
        default_config.update(config)
        super().__init__(default_config)
        self._setup_model()
        self._setup_optimizer()
        if self.config.get('use_lr_scheduler', False):
            self._setup_scheduler()

    def _setup_model(self) -> None:
        """Setup the ensemble model architecture."""
        self.models = []
        for model_config in self.config['models']:
            model_class = model_config.pop('class')
            model = model_class(config=model_config)
            self.models.append(model)
        
        # Initialize model weights
        if self.config['model_weights'] is None:
            self.model_weights = torch.ones(len(self.models)) / len(self.models)
        else:
            self.model_weights = torch.tensor(self.config['model_weights'])
        self.model_weights = self.model_weights.to(self.device)

    def _setup_optimizer(self) -> None:
        """Setup the optimizer."""
        if self.optimizer is None:
            lr = self.config.get('learning_rate', 0.001)
            self.optimizer = torch.optim.Adam([self.model_weights], lr=lr)

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
        """Forward pass through the ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
            
        # Stack predictions and apply weights
        stacked_preds = torch.stack(predictions, dim=0)
        weighted_preds = stacked_preds * self.model_weights.view(-1, 1, 1)
        ensemble_pred = weighted_preds.sum(dim=0)
        
        return ensemble_pred

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
        
        # Convert to tensors
        X = torch.FloatTensor(X[:-1])
        y = torch.FloatTensor(y).unsqueeze(-1)
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        return X, y

    def save(self, path: str) -> None:
        """Save ensemble model state.
        
        Args:
            path: Path to save model state
        """
        if not self.models:
            raise ValueError("No models in ensemble")
            
        state = {
            'model_state': [model.state_dict() for model in self.models],
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history,
            'best_loss': self.best_loss,
            'model_weights': self.model_weights,
            'X_mean': self.X_mean if hasattr(self, 'X_mean') else None,
            'X_std': self.X_std if hasattr(self, 'X_std') else None,
            'y_mean': self.y_mean if hasattr(self, 'y_mean') else None,
            'y_std': self.y_std if hasattr(self, 'y_std') else None
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load ensemble model state.
        
        Args:
            path: Path to load model state from
        """
        state = torch.load(path, weights_only=False)
        self.config = state['config']
        self.history = state['history']
        self.best_loss = state['best_loss']
        self.model_weights = state['model_weights'].to(self.device)
        
        # Load model states
        for i, model_state in enumerate(state['model_state']):
            if i < len(self.models):
                self.models[i].load_state_dict(model_state)
            
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
        
    def fit(self, train_data: torch.Tensor, val_data: Optional[torch.Tensor] = None,
            **kwargs) -> Dict[str, Any]:
        """Train all models in the ensemble.
        
        Args:
            train_data: Training data
            val_data: Validation data
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training history
        """
        history = {}
        
        # Train each model
        for i, model in enumerate(self.models):
            model_history = model.fit(train_data, val_data, **kwargs)
            history[f'model_{i}'] = model_history
            
        return history
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions using the ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing ensemble prediction and individual model predictions
        """
        predictions = {}
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            pred = model(x)
            predictions[f'model_{i}'] = pred
            
        # Get ensemble prediction
        ensemble_pred = self.forward(x)
        predictions['ensemble'] = ensemble_pred
        
        return predictions 