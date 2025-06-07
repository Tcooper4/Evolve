import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging

class BaseModel(nn.Module, ABC):
    """Base class for all forecasting models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.history = []
        self.best_loss = float('inf')
        self.patience = self.config.get('patience', 5)
        self.patience_counter = 0
        self.early_stopping = self.config.get('early_stopping', True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    @abstractmethod
    def _setup_model(self) -> None:
        """Setup the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _setup_model method")
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
        
    def fit(self, data: pd.DataFrame, epochs: int = 10, batch_size: int = 32) -> None:
        """Train the model.
        
        Args:
            data: Input data as pandas DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            raise ValueError("Model not initialized")
            
        self.model.train()
        X, y = self._prepare_data(data, is_training=True)
        
        # Initialize optimizer if not already done
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters())
            
        # Initialize scheduler if enabled
        if self.config.get('use_lr_scheduler', False) and self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2
            )
            
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = nn.MSELoss()(output, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            # Update learning rate if scheduler is enabled
            if self.scheduler is not None:
                self.scheduler.step(epoch_loss / (len(X) / batch_size))
                
            # Update history
            self._update_history(epoch_loss / (len(X) / batch_size))
            
            # Early stopping check
            if self.early_stopping:
                if self._check_early_stopping(epoch_loss / (len(X) / batch_size)):
                    break
                    
    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions.
        
        Args:
            data: Input data as pandas DataFrame
            
        Returns:
            Dictionary containing predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized")
            
        self.model.eval()
        X, _ = self._prepare_data(data, is_training=False)
        
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
            return {'predictions': predictions}
            
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
            'best_loss': self.best_loss
        }
        torch.save(state, path)
        
    def load(self, path: str) -> None:
        """Load model state.
        
        Args:
            path: Path to load model state from
        """
        state = torch.load(path)
        self.config = state['config']
        self.history = state['history']
        self.best_loss = state['best_loss']
        
        if self.model is not None:
            self.model.load_state_dict(state['model_state'])
            
        if self.optimizer is not None and state['optimizer_state'] is not None:
            self.optimizer.load_state_dict(state['optimizer_state'])
            
        if self.scheduler is not None and state['scheduler_state'] is not None:
            self.scheduler.load_state_dict(state['scheduler_state'])
            
    @abstractmethod
    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> tuple:
        """Prepare data for training or prediction.
        
        Args:
            data: Input data as pandas DataFrame
            is_training: Whether data is for training
            
        Returns:
            Tuple of (X, y) tensors
        """
        raise NotImplementedError("Subclasses must implement _prepare_data method")
        
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data.
        
        Args:
            data: Input data as pandas DataFrame
        """
        required_columns = ['close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        if data.isnull().any().any():
            raise ValueError("Data contains missing values")
            
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor on correct device
        """
        return tensor.to(self.device)
        
    def _update_history(self, loss: float) -> None:
        """Update training history.
        
        Args:
            loss: Current loss value
        """
        self.history.append(loss)
        
    def _check_early_stopping(self, loss: float) -> bool:
        """Check if training should stop early.
        
        Args:
            loss: Current loss value
            
        Returns:
            Whether to stop training
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience 