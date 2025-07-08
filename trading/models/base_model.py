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
from .model_utils import ValidationError, ModelError, validate_data, to_device, from_device, safe_forward, compute_loss, compute_metrics, get_model_confidence, get_model_metadata
from .dataset import TimeSeriesDataset

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
                self.device = torch.device('cuda')
                self.logger.info("Using CUDA device")
            else:
                self.device = torch.device('cpu')
                self.logger.info("Using CPU device")
        except Exception as e:
            self.device = torch.device('cpu')
            self.logger.warning(f"CUDA setup failed, using CPU: {e}")
        
        # Validate configuration
        self._validate_config()
        
        # Build model
        self.build_model()
        
        self.logger.info(f"Model initialized on {self.device}")
    
    def _setup_logging(self) -> None:
        """Setup logging for the model."""
        try:
            log_dir = Path("logs/models")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            log_file = log_dir / f"{self.__class__.__name__}.log"
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            self.logger.warning(f"Could not setup file logging: {e}")
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the neural network model.
        
        Returns:
            PyTorch model
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate model configuration."""
        try:
            required_keys = ['input_size', 'hidden_size', 'output_size']
            missing_keys = [key for key in required_keys if key not in self.config]
            
            if missing_keys:
                self.logger.warning(f"Missing config keys: {missing_keys}")
                # Set defaults
                self.config.setdefault('input_size', 10)
                self.config.setdefault('hidden_size', 64)
                self.config.setdefault('output_size', 1)
                self.config.setdefault('num_layers', 2)
                self.config.setdefault('dropout', 0.1)
                self.config.setdefault('learning_rate', 0.001)
                self.config.setdefault('batch_size', 32)
                self.config.setdefault('sequence_length', 20)
                
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            raise ValidationError(f"Invalid configuration: {e}")
    
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
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        try:
            # Validate data
            validate_data(data, feature_cols + [target_col])
            
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
                val_data, sequence_length, target_col, feature_cols, train_dataset.get_scaler()
            )
            test_dataset = TimeSeriesDataset(
                test_data, sequence_length, target_col, feature_cols, train_dataset.get_scaler()
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            self.logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise ModelError(f"Data preparation failed: {e}")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        try:
            if self.model is None:
                raise ModelError("Model not initialized")
            
            lr = self.config.get('learning_rate', 0.001)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
        except Exception as e:
            self.logger.error(f"Optimizer setup failed: {e}")
            raise ModelError(f"Optimizer setup failed: {e}")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        try:
            if self.optimizer is None:
                self._setup_optimizer()
            
            patience = self.config.get('scheduler_patience', 10)
            factor = self.config.get('scheduler_factor', 0.5)
            
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor, verbose=True
            )
            
        except Exception as e:
            self.logger.error(f"Scheduler setup failed: {e}")
            raise ModelError(f"Scheduler setup failed: {e}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, patience: int = 10) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        try:
            if self.model is None:
                raise ModelError("Model not initialized")
            
            # Setup optimizer and scheduler
            self._setup_optimizer()
            self._setup_scheduler()
            
            # Setup loss function
            self.criterion = nn.MSELoss()
            
            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = to_device(data, self.device), to_device(target, self.device)
                    
                    self.optimizer.zero_grad()
                    output = safe_forward(self.model, data)
                    loss = compute_loss(self.criterion, output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = to_device(data, self.device), to_device(target, self.device)
                        output = safe_forward(self.model, data)
                        loss = compute_loss(self.criterion, output, target)
                        val_loss += loss.item()
                
                # Calculate average losses
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                # Store losses
                self.train_losses.append(avg_train_loss)
                self.val_losses.append(avg_val_loss)
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # Log progress
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                # Early stopping check
                if self.early_stopping_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state)
            
            self.logger.info("Training completed")
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise ModelError(f"Training failed: {e}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        try:
            if self.model is None:
                raise ModelError("Model not initialized")
            
            self.model.eval()
            test_loss = 0.0
            predictions = []
            targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = to_device(data, self.device), to_device(target, self.device)
                    output = safe_forward(self.model, data)
                    loss = compute_loss(self.criterion, output, target)
                    test_loss += loss.item()
                    
                    predictions.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())
            
            # Calculate metrics
            avg_test_loss = test_loss / len(test_loader)
            predictions = np.concatenate(predictions).flatten()
            targets = np.concatenate(targets).flatten()
            
            metrics = compute_metrics(torch.tensor(predictions), torch.tensor(targets))
            
            results = {
                'test_loss': avg_test_loss,
                'mse': metrics[0],
                'mae': metrics[1],
                'rmse': metrics[2]
            }
            
            self.logger.info(f"Evaluation completed: {results}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise ModelError(f"Evaluation failed: {e}")
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Make predictions.
        
        Args:
            data: Input data
            horizon: Prediction horizon
            
        Returns:
            Predictions
        """
        try:
            if self.model is None:
                raise ModelError("Model not initialized")
            
            self.model.eval()
            
            # Prepare data
            X, _ = self._prepare_data(data, is_training=False)
            X = to_device(X, self.device)
            
            with torch.no_grad():
                if horizon == 1:
                    predictions = safe_forward(self.model, X)
                else:
                    predictions = self._multi_horizon_predict(X, horizon)
            
            return from_device(predictions).cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ModelError(f"Prediction failed: {e}")
    
    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast.
        
        Args:
            data: Historical data
            horizon: Forecast horizon
            
        Returns:
            Forecast results
        """
        try:
            predictions = self.predict(data, horizon)
            
            # Create forecast DataFrame
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'predicted_value': predictions.flatten()
            })
            
            # Calculate confidence intervals (simple approach)
            confidence = get_model_confidence(self.val_losses)
            
            results = {
                'forecast': forecast_df,
                'predictions': predictions,
                'confidence': confidence,
                'horizon': horizon,
                'model_type': self.__class__.__name__
            }
            
            self.logger.info(f"Forecast generated for {horizon} periods")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Forecast failed: {e}")
            raise ModelError(f"Forecast failed: {e}")
    
    def _multi_horizon_predict(self, X: torch.Tensor, horizon: int) -> np.ndarray:
        """Make multi-horizon predictions.
        
        Args:
            X: Input tensor
            horizon: Prediction horizon
            
        Returns:
            Multi-horizon predictions
        """
        try:
            predictions = []
            current_input = X.clone()
            
            for _ in range(horizon):
                with torch.no_grad():
                    output = safe_forward(self.model, current_input)
                    predictions.append(output)
                    
                    # Update input for next prediction (simple approach)
                    # In practice, you might want to use the predicted value
                    current_input = torch.roll(current_input, -1, dims=1)
                    current_input[:, -1] = output.squeeze()
            
            return torch.cat(predictions, dim=1)
            
        except Exception as e:
            self.logger.error(f"Multi-horizon prediction failed: {e}")
            raise ModelError(f"Multi-horizon prediction failed: {e}")
    
    def _save_model(self, filename: str) -> Dict[str, Any]:
        """Save model to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Save result
        """
        try:
            save_path = self.results_dir / filename
            
            checkpoint = {
                'model_state_dict': self.model.state_dict() if self.model else None,
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'config': self.config,
                'scaler': self.scaler,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'device': str(self.device)
            }
            
            torch.save(checkpoint, save_path)
            
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
            
            checkpoint = torch.load(load_path, map_location=self.device)
            
            # Load model state
            if checkpoint.get('model_state_dict') and self.model:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if checkpoint.get('optimizer_state_dict') and self.optimizer:
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

    def get_confidence(self) -> Dict[str, float]:
        """Get model confidence metrics.
        
        Returns:
            Dictionary of confidence metrics
        """
        return get_model_confidence(self.val_losses)

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return get_model_metadata(self.__class__.__name__, self.config, self.device, self.best_val_loss, self.train_losses) 