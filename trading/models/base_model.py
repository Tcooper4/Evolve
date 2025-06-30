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
        """Set up logging for the model."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Create rotating file handler
            log_file = log_dir / f"{self.__class__.__name__}.log"
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            self.logger.error(f"Error setting up logging: {e}")
    
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
        try:
            required_keys = ['input_size', 'hidden_size', 'output_size']
            missing_keys = [key for key in required_keys if key not in self.config]
            
            if missing_keys:
                raise ValidationError(f"Missing required config keys: {missing_keys}")
            
            # Validate numeric values
            for key in ['input_size', 'hidden_size', 'output_size']:
                if not isinstance(self.config[key], int) or self.config[key] <= 0:
                    raise ValidationError(f"{key} must be a positive integer")
            
            return {'success': True, 'message': 'Configuration validation passed', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    def prepare_data(self, data: pd.DataFrame, target_col: str,
                    feature_cols: List[str], sequence_length: int,
                    test_size: float = 0.2, val_size: float = 0.1,
                    batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training.
        
        Args:
            data: Input data
            target_col: Target column name
            feature_cols: Feature column names
            sequence_length: Length of input sequences
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        try:
            # Validate data
            if data.empty:
                raise ValidationError("Data is empty")
            
            if target_col not in data.columns:
                raise ValidationError(f"Target column '{target_col}' not found")
            
            missing_features = [col for col in feature_cols if col not in data.columns]
            if missing_features:
                raise ValidationError(f"Missing feature columns: {missing_features}")
            
            # Split data
            train_size = 1 - test_size - val_size
            train_data = data.iloc[:int(len(data) * train_size)]
            val_data = data.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size))]
            test_data = data.iloc[int(len(data) * (train_size + val_size)):]
            
            # Create datasets
            train_dataset = TimeSeriesDataset(
                train_data, sequence_length, target_col, feature_cols, self.scaler
            )
            val_dataset = TimeSeriesDataset(
                val_data, sequence_length, target_col, feature_cols, self.scaler
            )
            test_dataset = TimeSeriesDataset(
                test_data, sequence_length, target_col, feature_cols, self.scaler
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            self.logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
            
            return {'success': True, 'result': (train_loader, val_loader, test_loader), 'message': 'Data preparation completed', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _setup_optimizer(self) -> None:
        """Set up optimizer."""
        try:
            if self.model is None:
                raise ModelError("Model must be built before setting up optimizer")
            
            lr = self.config.get('learning_rate', 0.001)
            weight_decay = self.config.get('weight_decay', 0.0)
            
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            
            return {'success': True, 'message': 'Optimizer setup completed', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    def _setup_scheduler(self) -> None:
        """Set up learning rate scheduler."""
        try:
            if self.optimizer is None:
                raise ModelError("Optimizer must be set up before scheduler")
            
            if self.config.get('use_scheduler', True):
                patience = self.config.get('scheduler_patience', 10)
                factor = self.config.get('scheduler_factor', 0.5)
                
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=factor,
                    patience=patience,
                    verbose=True
                )
            
            return {'success': True, 'message': 'Scheduler setup completed', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
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
        try:
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
            
            return {'success': True, 'result': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_epoch': best_epoch
            }, 'message': 'Training completed successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if self.model is None:
                raise ModelError("Model must be trained before evaluation")
            
            self.model.eval()
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    predictions.extend(outputs.cpu().numpy())
                    targets.extend(batch_y.cpu().numpy())
            
            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()
            
            # Calculate metrics
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(mse)
            
            # Calculate R-squared
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            
            return {'success': True, 'result': metrics, 'message': 'Evaluation completed', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            data: Input data
            horizon: Prediction horizon
            
        Returns:
            Array of predictions
        """
        try:
            if self.model is None:
                raise ModelError("Model must be trained before making predictions")
            
            self.model.eval()
            
            # Prepare data
            X = self._prepare_data(data, is_training=False)[0]
            X = X.to(self.device)
            
            # Make predictions
            with torch.no_grad():
                predictions = self.model(X)
            
            predictions = predictions.cpu().numpy().flatten()
            
            return {'success': True, 'result': predictions, 'message': 'Predictions generated successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate multi-step forecasts.
        
        Args:
            data: Input data
            horizon: Number of steps to forecast
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        try:
            if self.model is None:
                raise ModelError("Model must be trained before forecasting")
            
            # Prepare data
            X = self._prepare_data(data, is_training=False)[0]
            X = X.to(self.device)
            
            # Generate forecasts
            forecasts = self._multi_horizon_predict(X, horizon)
            
            # Calculate confidence intervals (simple approach)
            confidence_intervals = {
                'lower': forecasts * 0.9,  # 10% lower
                'upper': forecasts * 1.1   # 10% upper
            }
            
            result = {
                'forecasts': forecasts,
                'confidence_intervals': confidence_intervals,
                'horizon': horizon
            }
            
            return {'success': True, 'result': result, 'message': 'Forecast generated successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _multi_horizon_predict(self, X: torch.Tensor, horizon: int) -> np.ndarray:
        """Generate multi-step predictions.
        
        Args:
            X: Input tensor
            horizon: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        try:
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                current_input = X.clone()
                
                for _ in range(horizon):
                    # Make prediction
                    output = self.model(current_input)
                    predictions.append(output.cpu().numpy().flatten()[-1])
                    
                    # Update input for next step (simple approach)
                    # In practice, you might want to use the actual prediction
                    current_input = torch.roll(current_input, -1, dims=1)
                    current_input[:, -1] = output[:, -1]
            
            return np.array(predictions)
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss tensor
        """
        try:
            if self.criterion is None:
                self.criterion = nn.MSELoss()
            
            return self.criterion(y_pred, y_true)
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _compute_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> List[float]:
        """Compute evaluation metrics.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            List of metrics
        """
        try:
            y_pred_np = y_pred.cpu().numpy().flatten()
            y_true_np = y_true.cpu().numpy().flatten()
            
            mse = np.mean((y_pred_np - y_true_np) ** 2)
            mae = np.mean(np.abs(y_pred_np - y_true_np))
            rmse = np.sqrt(mse)
            
            return [mse, mae, rmse]
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _save_model(self, filename: str) -> Dict[str, Any]:
        """Save model to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Save result
        """
        try:
            if self.model is None:
                raise ModelError("No model to save")
            
            save_path = self.results_dir / filename
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'config': self.config,
                'scaler': self.scaler,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            }, save_path)
            
            self.logger.info(f"Model saved to {save_path}")
            
            return {'success': True, 'message': f'Model saved to {save_path}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _load_model(self, filename: str) -> Dict[str, Any]:
        """Load model from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Load result
        """
        try:
            load_path = self.results_dir / filename
            
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")
            
            # Load checkpoint
            checkpoint = torch.load(load_path, map_location=self.device)
            
            # Build model if not exists
            if self.model is None:
                self.model = self.build_model()
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            
            # Load optimizer state
            if checkpoint['optimizer_state_dict'] and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load other components
            self.config = checkpoint.get('config', {})
            self.scaler = checkpoint.get('scaler', StandardScaler())
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            self.logger.info(f"Model loaded from {load_path}")
            
            return {'success': True, 'message': f'Model loaded from {load_path}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _save_training_history(self) -> Dict[str, Any]:
        """Save training history to file.
        
        Returns:
            Save result
        """
        try:
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            history_file = self.results_dir / f"{self.__class__.__name__}_history.json"
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            self.logger.info(f"Training history saved to {history_file}")
            
            return {'success': True, 'message': f'Training history saved to {history_file}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        try:
            if not self.train_losses or not self.val_losses:
                self.logger.warning("No training history to plot")
                return {'success': False, 'error': 'No training history to plot', 'timestamp': datetime.now().isoformat()}
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Training Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_file = self.results_dir / f"{self.__class__.__name__}_training_history.png"
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Training history plot saved to {plot_file}")
            
            return {'success': True, 'message': f'Training history plot saved to {plot_file}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def infer(self) -> None:
        """Run inference mode."""
        try:
            if self.model is not None:
                self.model.eval()
            
            return {'success': True, 'message': 'Model set to inference mode', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    @abstractmethod
    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for model input.
        
        Args:
            data: Input data
            is_training: Whether this is for training
            
        Returns:
            Tuple of (features, targets)
        """
        pass

    def _to_device(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Move data to device.
        
        Args:
            data: Data to move
            
        Returns:
            Data on device
        """
        try:
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, dict):
                return {key: value.to(self.device) for key, value in data.items()}
            else:
                return data
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _from_device(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Move data from device to CPU.
        
        Args:
            data: Data to move
            
        Returns:
            Data on CPU
        """
        try:
            if isinstance(data, torch.Tensor):
                return data.cpu()
            elif isinstance(data, dict):
                return {key: value.cpu() for key, value in data.items()}
            else:
                return data
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _safe_forward(self, *args, **kwargs) -> torch.Tensor:
        """Safely run forward pass with error handling.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Model output
        """
        try:
            if self.model is None:
                raise ModelError("Model not initialized")
            
            return self.model(*args, **kwargs)
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_confidence(self) -> Dict[str, float]:
        """Get model confidence metrics.
        
        Returns:
            Dictionary of confidence metrics
        """
        try:
            if not self.val_losses:
                return {'success': False, 'error': 'No validation history available', 'timestamp': datetime.now().isoformat()}
            
            # Simple confidence based on validation loss
            latest_val_loss = self.val_losses[-1]
            best_val_loss = min(self.val_losses)
            
            # Confidence decreases as validation loss increases
            confidence = max(0.0, 1.0 - (latest_val_loss - best_val_loss) / best_val_loss)
            
            metrics = {
                'confidence': confidence,
                'latest_val_loss': latest_val_loss,
                'best_val_loss': best_val_loss,
                'loss_ratio': latest_val_loss / best_val_loss if best_val_loss > 0 else float('inf')
            }
            
            return {'success': True, 'result': metrics, 'message': 'Confidence metrics calculated', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        try:
            return {
                'model_type': self.__class__.__name__,
                'config': self.config,
                'created_at': datetime.now().isoformat(),
                'device': str(self.device),
                'best_val_loss': self.best_val_loss,
                'training_epochs': len(self.train_losses)
            }
        except Exception as e:
            self.logger.warning(f"Could not get model metadata: {e}")
            return {
                'model_type': self.__class__.__name__,
                'config': self.config,
                'created_at': datetime.now().isoformat(),
                'error': str(e)
            } 