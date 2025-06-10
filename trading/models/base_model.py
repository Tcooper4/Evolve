import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime
import json
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

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
        self.criterion = None
        self.scaler = StandardScaler()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the neural network model.
        
        Returns:
            PyTorch model
        """
        pass
    
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
        
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters())
        
        if self.criterion is None:
            self.criterion = nn.MSELoss()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Training loop
        best_epoch = 0
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
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
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
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
            elif epoch - best_epoch >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.numpy())
        
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        self.logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def predict(self, data: pd.DataFrame, feature_cols: List[str],
                sequence_length: int) -> np.ndarray:
        """Generate predictions.
        
        Args:
            data: Input data
            feature_cols: List of feature column names
            sequence_length: Length of input sequences
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        # Scale features
        features = self.scaler.transform(data[feature_cols])
        
        # Create sequences
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(features[i:i + sequence_length])
        
        # Generate predictions
        predictions = []
        with torch.no_grad():
            for sequence in sequences:
                x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                output = self.model(x)
                predictions.append(output.cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def save(self, filepath: str) -> None:
        """Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create save directory
        save_dir = Path(filepath).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, filepath)
        
        # Save scaler
        scaler_path = Path(filepath).with_suffix('.scaler')
        joblib.dump(self.scaler, scaler_path)
        
        self.logger.info(f"Saved model to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model from disk.
        
        Args:
            filepath: Path to saved model
        """
        # Load model state
        checkpoint = torch.load(filepath)
        
        # Build model
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state'])
        
        # Load optimizer state
        if checkpoint['optimizer_state'] is not None:
            self.optimizer = optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load training state
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.config = checkpoint['config']
        
        # Load scaler
        scaler_path = Path(filepath).with_suffix('.scaler')
        self.scaler = joblib.load(scaler_path)
        
        self.logger.info(f"Loaded model from {filepath}")
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save model results to disk.
        
        Args:
            results: Results to save
            filename: Output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"{filename}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Saved results to {filepath}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load model results from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded results
        """
        filepath = self.results_dir / filename
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        self.logger.info(f"Loaded results from {filepath}")
        
        return results 