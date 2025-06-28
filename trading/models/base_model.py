"""Base class for all ML models with common functionality."""

# Standard library imports
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pydantic import BaseModel as PydanticBaseModel, Field

# Local imports
import joblib

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ModelError(Exception):
    """Custom exception for model errors."""
    pass

class Recommendation(PydanticBaseModel):
    """Model recommendation for trading signals."""
    model_name: str
    signal: str
    confidence: float

class StrategyPayload(PydanticBaseModel):
    """Payload containing strategy recommendations."""
    recommendations: List[Recommendation] = Field(default_factory=list)

class ModelRegistry:
    """Registry for model types and their configurations."""
    
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a model class.
        
        Args:
            name: Name of the model type
        """
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str, config: Dict[str, Any]) -> 'BaseModel':
        """Get a model instance by name.
        
        Args:
            name: Name of the model type
            config: Model configuration
            
        Returns:
            Model instance
            
        Raises:
            ModelError: If model type not found
        """
        if name not in cls._models:
            raise ModelError(f"Model type '{name}' not found in registry")
        return cls._models[name](config)

class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int, target_col: str,
                 feature_cols: List[str], scaler: Optional[StandardScaler] = None):
        """Initialize dataset.
        
        Args:
            data: Input data
            sequence_length: Length of input sequences
            target_col: Name of target column
            feature_cols: List of feature column names
            scaler: Optional scaler for features
        """
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.feature_cols = feature_cols
        
        # Validate data
        self._validate_data(data)
        
        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(data[feature_cols])
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(data[feature_cols])
        
        # Get targets
        self.targets = data[target_col].values
        
        # Create sequences
        self.sequences = []
        self.sequence_targets = []
        
        for i in range(len(data) - sequence_length):
            self.sequences.append(self.features[i:i + sequence_length])
            self.sequence_targets.append(self.targets[i + sequence_length])
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        # Check for missing values
        if data.isnull().any().any():
            raise ValidationError("Data contains missing values")
        
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=np.number)).any().any():
            raise ValidationError("Data contains infinite values")
        
        # Check for required columns
        missing_cols = [col for col in self.feature_cols + [self.target_col] 
                       if col not in data.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.sequence_targets[idx]])
        )

class BaseModel(ABC):
    """Base class for all ML models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create results directory
        self.results_dir = Path("model_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = StandardScaler()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stopping_counter = 0
        
        # Setup logging
        self._setup_logging()
        
        # Setup device with fallback
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                torch.backends.cudnn.benchmark = True
                self.logger.info("Using CUDA device with cuDNN benchmarking")
            else:
                self.device = torch.device("cpu")
                self.logger.info("CUDA not available, using CPU")
        except Exception as e:
            self.device = torch.device("cpu")
            self.logger.warning(f"Error setting up CUDA, falling back to CPU: {e}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_handler = RotatingFileHandler(
            os.getenv('LOG_FILE', 'trading.log'),
            maxBytes=int(os.getenv('LOG_MAX_SIZE', 10485760)),
            backupCount=int(os.getenv('LOG_BACKUP_COUNT', 5))
        )
        log_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(log_handler)
        self.logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the neural network model.
        
        Returns:
            PyTorch model
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate model configuration.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        required_params = ['input_size', 'output_size', 'sequence_length']
        missing_params = [param for param in required_params if param not in self.config]
        if missing_params:
            raise ValidationError(f"Missing required parameters: {missing_params}")
    
    def prepare_data(self, data: pd.DataFrame, target_col: str,
                    feature_cols: List[str], sequence_length: int,
                    test_size: float = 0.2, val_size: float = 0.1,
                    batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training.
        
        Args:
            data: Input data
            target_col: Name of target column
            feature_cols: List of feature column names
            sequence_length: Length of input sequences
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Validate data
        if data.isnull().any().any():
            raise ValidationError("Data contains missing values")
        
        # Split data
        train_data, test_data = train_test_split(
            data, test_size=test_size, shuffle=False
        )
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, shuffle=False
        )
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data, sequence_length, target_col, feature_cols, self.scaler
        )
        val_dataset = TimeSeriesDataset(
            val_data, sequence_length, target_col, feature_cols, train_dataset.scaler
        )
        test_dataset = TimeSeriesDataset(
            test_data, sequence_length, target_col, feature_cols, train_dataset.scaler
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer with learning rate from config."""
        if self.optimizer is None:
            lr = self.config.get('learning_rate', 0.001)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        if self.config.get('use_lr_scheduler', True):
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, patience: int = 10) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            Dictionary of training and validation losses
        """
        if self.model is None:
            self.model = self.build_model()
            self.model.to(self.device)
        
        self._setup_optimizer()
        self._setup_scheduler()
        
        if self.criterion is None:
            self.criterion = nn.MSELoss()
        
        # Training loop
        best_epoch = 0
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"Restored best model from epoch {best_epoch+1}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ModelError("Model not trained")
        
        self.model.eval()
        test_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                test_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        
        metrics = {
            'test_loss': test_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        self.logger.info("Test metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Make predictions.
        
        Args:
            data: Input data
            horizon: Prediction horizon (number of steps ahead)
            
        Returns:
            Predictions array
        """
        self.model.eval()
        X, _ = self._prepare_data(data, is_training=False)
        
        if horizon == 1:
            with torch.no_grad():
                y_pred = self.model(X)
            return y_pred.cpu().numpy()
        else:
            return self._multi_horizon_predict(X, horizon)
    
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
            predictions = self.predict(data, horizon=1)
            
            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()
            
            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data, horizon=1)
                forecast_values.append(pred[-1])
                
                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row['close'] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row
            
            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.8,  # Base model confidence
                'model': self.__class__.__name__,
                'horizon': horizon
            }
            
        except Exception as e:
            import logging
            logging.error(f"Error in base model forecast: {e}")
            raise RuntimeError(f"Base model forecasting failed: {e}")
    
    def _multi_horizon_predict(self, X: torch.Tensor, horizon: int) -> np.ndarray:
        """Make multi-horizon predictions.
        
        Args:
            X: Input tensor
            horizon: Prediction horizon
            
        Returns:
            Predictions array
        """
        predictions = []
        current_input = X
        
        with torch.no_grad():
            for _ in range(horizon):
                y_pred = self.model(current_input)
                predictions.append(y_pred)
                
                # Update input for next prediction
                if hasattr(self, '_update_input_for_next_step'):
                    current_input = self._update_input_for_next_step(current_input, y_pred)
                else:
                    # Default: shift input and append prediction
                    current_input = torch.cat([current_input[:, 1:], y_pred.unsqueeze(1)], dim=1)
        
        return torch.cat(predictions, dim=1).cpu().numpy()
    
    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss tensor
        """
        return F.mse_loss(y_pred, y_true)
    
    def _compute_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> List[float]:
        """Compute metrics.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            List of metric values
        """
        mse = F.mse_loss(y_pred, y_true).item()
        mae = F.l1_loss(y_pred, y_true).item()
        return [mse, mae]
    
    def _save_model(self, filename: str) -> None:
        """Save model state.
        
        Args:
            filename: Output filename
        """
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
        
        torch.save(state, os.path.join(self.model_dir, filename))
    
    def _load_model(self, filename: str) -> None:
        """Load model state.
        
        Args:
            filename: Input filename
        """
        state = torch.load(os.path.join(self.model_dir, filename))
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        if self.scheduler is not None and 'scheduler_state' in state:
            self.scheduler.load_state_dict(state['scheduler_state'])
        self.training_history = state['training_history']
    
    def _save_training_history(self) -> None:
        """Save training history to CSV and JSON."""
        # Save to CSV
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(os.path.join(self.model_dir, 'training_history.csv'), index=False)
        
        # Save to JSON
        with open(os.path.join(self.model_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=4)
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss'][0] is not None:
            plt.plot(self.training_history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['train_metrics'], label='Train Metrics')
        if self.training_history['val_metrics'][0] is not None:
            plt.plot(self.training_history['val_metrics'], label='Val Metrics')
        plt.title('Training and Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.close()
    
    def infer(self) -> None:
        """Enable inference mode."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
    
    @abstractmethod
    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training or prediction.
        
        Args:
            data: Input data
            is_training: Whether data is for training
            
        Returns:
            Tuple of (X, y) tensors
        """
        pass

    def _to_device(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Move data to the appropriate device.
        
        Args:
            data: Input data (tensor or dict of tensors)
            
        Returns:
            Data moved to the appropriate device
        """
        try:
            if isinstance(data, dict):
                return {k: v.to(self.device) for k, v in data.items()}
            return data.to(self.device)
        except Exception as e:
            self.logger.error(f"Error moving data to device: {e}")
            raise ModelError(f"Failed to move data to device: {e}")
    
    def _from_device(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Move data from device to CPU.
        
        Args:
            data: Input data (tensor or dict of tensors)
            
        Returns:
            Data moved to CPU
        """
        try:
            if isinstance(data, dict):
                return {k: v.cpu() for k, v in data.items()}
            return data.cpu()
        except Exception as e:
            self.logger.error(f"Error moving data from device: {e}")
            raise ModelError(f"Failed to move data from device: {e}")
    
    def _safe_forward(self, *args, **kwargs) -> torch.Tensor:
        """Safely perform forward pass with error handling.
        
        Args:
            *args: Arguments for forward pass
            **kwargs: Keyword arguments for forward pass
            
        Returns:
            Model output tensor
        """
        try:
            return self.model(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            raise ModelError(f"Model forward pass failed: {e}")

    def get_confidence(self) -> Dict[str, float]:
        """Get model confidence metrics.
        
        Returns:
            Dictionary containing confidence metrics
        """
        try:
            # Default confidence metrics
            confidence = {
                'model_confidence': 0.8,  # Default confidence
                'prediction_interval': 0.95,
                'uncertainty': 0.2
            }
            
            # If we have validation losses, use them to estimate confidence
            if hasattr(self, 'val_losses') and self.val_losses:
                # Lower validation loss = higher confidence
                avg_val_loss = np.mean(self.val_losses[-10:])  # Last 10 epochs
                confidence['model_confidence'] = max(0.1, 1.0 - avg_val_loss)
                confidence['uncertainty'] = avg_val_loss
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"Could not compute confidence metrics: {e}")
            return {
                'model_confidence': 0.5,
                'prediction_interval': 0.95,
                'uncertainty': 0.5
            }

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Dictionary containing model metadata
        """
        try:
            metadata = {
                'model_type': self.__class__.__name__,
                'config': self.config,
                'created_at': datetime.now().isoformat(),
                'model_parameters': 0,
                'training_history': {
                    'train_losses': self.train_losses if hasattr(self, 'train_losses') else [],
                    'val_losses': self.val_losses if hasattr(self, 'val_losses') else [],
                    'best_val_loss': self.best_val_loss if hasattr(self, 'best_val_loss') else float('inf')
                }
            }
            
            # Count model parameters if available
            if hasattr(self, 'model') and self.model is not None:
                try:
                    metadata['model_parameters'] = sum(p.numel() for p in self.model.parameters())
                except:
                    metadata['model_parameters'] = 0
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Could not get model metadata: {e}")
            return {
                'model_type': self.__class__.__name__,
                'config': self.config,
                'created_at': datetime.now().isoformat(),
                'error': str(e)
            } 