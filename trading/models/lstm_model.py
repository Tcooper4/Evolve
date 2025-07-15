"""LSTM-based forecasting model with advanced features."""

# Standard library imports
import logging
import os
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import joblib
import time

# Local imports
from .base_model import BaseModel
from utils.model_cache import cache_model_operation, get_model_cache
from utils.forecast_helpers import safe_forecast, validate_forecast_input, log_forecast_performance


class LSTMModel(nn.Module):
    """A class to handle LSTM model for time series prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """Initialize the LSTM model.

        Args:
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            output_dim (int): The output dimension.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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

    def train_model(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        epochs: int,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 10,
        checkpoint_dir: str = "models/checkpoints",
    ) -> Dict[str, List[float]]:
        """Train the LSTM model with data cleaning, early stopping, and checkpointing.

        Args:
            data (pd.DataFrame): Input features
            target (pd.Series): Target values
            epochs (int): Number of training epochs
            batch_size (int, optional): Batch size. Defaults to 32.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            patience (int, optional): Early stopping patience. Defaults to 10.
            checkpoint_dir (str, optional): Directory for model checkpoints. Defaults to "models/checkpoints".

        Returns:
            Dict[str, List[float]]: Training history with loss values
        """
        # Data cleaning: Handle NaN and infinite values
        X = data.copy()
        y = target.copy()

        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        # Remove rows with NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:
            raise ValueError(
                "Insufficient valid data after cleaning (need at least 10 samples)"
            )

        self.logger.info(
            f"Data cleaned: {len(data)} -> {len(X)} samples after removing NaN/infinite values"
        )

        # Initialize optimizer, loss function, and scheduler
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Setup checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1))

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)

        # Create sequences
        X_seq = self._create_sequences(X_tensor)
        y_seq = y_tensor[self.config["sequence_length"] :]

        # Create DataLoader
        dataset = TensorDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training history
        history = {"train_loss": [], "val_loss": []}

        # Early stopping variables
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

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
            history["train_loss"].append(avg_epoch_loss)

            # Update learning rate scheduler
            scheduler.step(avg_epoch_loss)

            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                best_model_state = self.state_dict().copy()

                # Save checkpoint
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"best_model_epoch_{epoch+1}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                        "config": self.config,
                    },
                    checkpoint_path,
                )
                self.logger.info(f"Saved best model checkpoint: {checkpoint_path}")
            else:
                patience_counter += 1

            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Patience: {patience_counter}/{patience}"
                )

            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                # Load best model
                self.load_state_dict(best_model_state)
                break

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
        for i in range(len(data) - self.config["sequence_length"]):
            sequences.append(data[i : i + self.config["sequence_length"]])
        return torch.stack(sequences)

    def set_sequence_length(self, new_length: int):
        """Dynamically adjust sequence length at runtime.

        Args:
            new_length (int): New sequence length

        Raises:
            ValueError: If new length is invalid
        """
        if new_length <= 0:
            raise ValueError("Sequence length must be positive")
        if new_length > self.config.get("max_sequence_length", 100):
            raise ValueError(
                f"Sequence length {new_length} exceeds maximum allowed value of {self.config.get('max_sequence_length', 100)}"
            )

        old_length = self.config["sequence_length"]
        self.config["sequence_length"] = new_length

        self.logger.info(f"Sequence length changed from {old_length} to {new_length}")

        # Rebuild model if necessary (for attention mechanisms)
        if self.config.get("use_attention", False):
            self.model = self.build_model()
            self.model.to(self.device)

    def get_optimal_sequence_length(self, data: pd.DataFrame) -> int:
        """Calculate optimal sequence length based on data characteristics.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            int: Optimal sequence length
        """
        # Calculate based on data length and volatility
        data_length = len(data)

        # Rule of thumb: sequence length should be 10-20% of data length
        # but not less than 10 and not more than max_sequence_length
        optimal_length = max(
            10,
            min(int(data_length * 0.15), self.config.get("max_sequence_length", 100)),
        )

        # Adjust based on volatility
        if (
            "target_column" in self.config
            and self.config["target_column"] in data.columns
        ):
            target_volatility = data[self.config["target_column"]].std()
            if (
                target_volatility > data[self.config["target_column"]].mean() * 0.1
            ):  # High volatility
                optimal_length = min(
                    optimal_length + 5, self.config.get("max_sequence_length", 100)
                )
            elif (
                target_volatility < data[self.config["target_column"]].mean() * 0.01
            ):  # Low volatility
                optimal_length = max(optimal_length - 5, 10)

        return optimal_length

    def auto_adjust_sequence_length(self, data: pd.DataFrame):
        """Automatically adjust sequence length based on data characteristics.

        Args:
            data (pd.DataFrame): Input data
        """
        optimal_length = self.get_optimal_sequence_length(data)
        self.set_sequence_length(optimal_length)


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
        
        # Cache management
        self.cache_dir = "cache/lstm_models"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.last_input_hash = None
        self.compiled_model = None

    def build_model(self) -> nn.Module:
        """Build and return the LSTM model using PyTorch."""
        input_size = len(self.config["feature_columns"])
        hidden_size = self.config["hidden_size"]
        num_layers = self.config.get("num_layers", 1)
        dropout = self.config["dropout"]
        use_batch_norm = self.config.get("use_batch_norm", False)
        use_layer_norm = self.config.get("use_layer_norm", False)
        additional_dropout = self.config.get("additional_dropout", 0)
        
        class LSTMForecasterModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, use_batch_norm, use_layer_norm, additional_dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, 
                    hidden_size, 
                    num_layers=num_layers, 
                    batch_first=True, 
                    dropout=dropout if num_layers > 1 else 0
                )
                
                # Additional layers for enhanced functionality
                self.batch_norm = nn.BatchNorm1d(hidden_size) if use_batch_norm else None
                self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None
                self.dropout_layer = nn.Dropout(additional_dropout)
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                # LSTM forward pass
                lstm_out, _ = self.lstm(x)
                
                # Use only the final timestep's output
                final_output = lstm_out[:, -1, :]
                
                # Apply normalization if enabled
                if self.batch_norm is not None:
                    final_output = self.batch_norm(final_output)
                if self.layer_norm is not None:
                    final_output = self.layer_norm(final_output)
                
                # Apply additional dropout
                final_output = self.dropout_layer(final_output)
                
                # Final prediction
                return self.fc(final_output)
        
        return LSTMForecasterModel(input_size, hidden_size, num_layers, dropout, use_batch_norm, use_layer_norm, additional_dropout)

    def _validate_config(self):
        """Validate model configuration."""
        required_keys = [
            "input_size",
            "hidden_size",
            "num_layers",
            "dropout",
            "sequence_length",
            "feature_columns",
            "target_column",
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate sequence length
        if self.config["sequence_length"] <= 0:
            raise ValueError("sequence_length must be positive")
        if self.config["sequence_length"] > self.config.get("max_sequence_length", 100):
            raise ValueError(
                f"sequence_length exceeds maximum allowed value of {self.config.get('max_sequence_length', 100)}"
            )

        # Validate feature columns
        if not self.config["feature_columns"]:
            raise ValueError("feature_columns cannot be empty")
        if self.config["target_column"] not in self.config["feature_columns"]:
            raise ValueError("target_column must be in feature_columns")

        # Validate batch size limits
        if "max_batch_size" in self.config and self.config["max_batch_size"] <= 0:
            raise ValueError("max_batch_size must be positive")

        # Validate epoch limits
        if "max_epochs" in self.config and self.config["max_epochs"] <= 0:
            raise ValueError("max_epochs must be positive")

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
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
            if x.size(1) > self.config.get("max_sequence_length", 100):
                raise ValueError(
                    f"Input sequence length {x.size(1)} exceeds maximum allowed value of {self.config.get('max_sequence_length', 100)}"
                )

            # Check batch size
            if (
                "max_batch_size" in self.config
                and x.size(0) > self.config["max_batch_size"]
            ):
                raise ValueError(
                    f"Batch size {x.size(0)} exceeds maximum allowed value of {self.config['max_batch_size']}"
                )

            # Move input to device
            x = x.to(self.device)

            return self.model(x)

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise MemoryError("GPU out of memory during forward pass")
            raise e

    def _prepare_data(
        self, data: pd.DataFrame, is_training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
            if not all(col in data.columns for col in self.config["feature_columns"]):
                raise ValueError("Missing required feature columns")

            # Check data size
            if len(data) < self.config["sequence_length"]:
                raise ValueError(
                    f"Data length {len(data)} is less than sequence length {self.config['sequence_length']}"
                )

            # Normalize data
            if is_training:
                self.scaler = StandardScaler()
                normalized_data = self.scaler.fit_transform(
                    data[self.config["feature_columns"]]
                )
            else:
                if not hasattr(self, "scaler"):
                    raise ValueError("Model must be trained before prediction")
                normalized_data = self.scaler.transform(
                    data[self.config["feature_columns"]]
                )

            # Convert to tensor
            X = torch.FloatTensor(normalized_data)

            # Create sequences
            X_seq = self._create_sequences(X)

            if is_training:
                # Get target values
                y = torch.FloatTensor(
                    normalized_data[
                        self.config["sequence_length"] :,
                        self.config["feature_columns"].index(
                            self.config["target_column"]
                        ),
                    ]
                )
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
        for i in range(len(data) - self.config["sequence_length"]):
            sequences.append(data[i : i + self.config["sequence_length"]])
        return torch.stack(sequences)

    def _get_input_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for input data to determine if retraining is needed.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            str: Hash of input data
        """
        # Create a hash of the data shape, column names, and first/last few values
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': str(data.dtypes.to_dict()),
            'first_values': data.head(3).to_dict(),
            'last_values': data.tail(3).to_dict(),
            'config_hash': str(sorted(self.config.items()))
        }
        
        data_str = str(data_info)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _load_cached_model(self, cache_key: str) -> bool:
        """Load cached compiled model if available.
        
        Args:
            cache_key (str): Cache key for the model
            
        Returns:
            bool: True if model was loaded successfully
        """
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.joblib")
            if os.path.exists(cache_path):
                self.compiled_model = joblib.load(cache_path)
                self.logger.info(f"Loaded cached model from {cache_path}")
                return True
        except Exception as e:
            self.logger.warning(f"Failed to load cached model: {e}")
        return False

    def _save_cached_model(self, cache_key: str) -> None:
        """Save compiled model to cache.
        
        Args:
            cache_key (str): Cache key for the model
        """
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.joblib")
            joblib.dump(self.compiled_model, cache_path)
            self.logger.info(f"Saved model to cache: {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save model to cache: {e}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        """Train the model with robust error handling and input validation."""
        # Generate input hash to check if retraining is needed
        input_hash = self._get_input_hash(X)
        
        # Check if we have a cached model for this input
        if input_hash == self.last_input_hash and self.compiled_model is not None:
            self.logger.info("Input data unchanged, using cached model")
            return {"train_loss": [0.0], "val_loss": [0.0], "cached": True}
        
        # Check if we can load from cache
        if self._load_cached_model(input_hash):
            self.last_input_hash = input_hash
            return {"train_loss": [0.0], "val_loss": [0.0], "cached": True}
        
        # Input validation: Drop NaNs
        X = X.copy()
        y = y.copy()
        nan_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[nan_mask]
        y = y[nan_mask]
        if len(X) < 20:
            raise ValueError("At least 20 data points are required after cleaning.")
        
        # Check if data has been normalized
        if not self._check_data_normalization(X):
            self.logger.warning("Data appears to be unnormalized. Consider scaling before training.")
        
        # Log volatility and autocorrelation
        volatility = X.std().mean()
        autocorr = X.apply(lambda col: col.autocorr(lag=1)).mean()
        self.logger.info(f"Input volatility: {volatility:.6f}, autocorrelation: {autocorr:.6f}")
        
        # Prepare data for LSTM (reshape to [samples, timesteps, features])
        seq_len = self.config["sequence_length"]
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_len):
            X_seq.append(X.iloc[i:i+seq_len].values)
            y_seq.append(y.iloc[i+seq_len])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Train/validation split
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training history
        history = {"train_loss": [], "val_loss": []}
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            # Store history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Cache the compiled model
        self.compiled_model = self.model
        self.last_input_hash = input_hash
        self._save_cached_model(input_hash)
        
        return history

    def predict(self, data: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        """Predict using the LSTM model with input validation, logging, and batch-wise evaluation."""
        # Input validation: Drop NaNs
        data = data.copy()
        data = data.dropna()
        if len(data) < 20:
            raise ValueError("At least 20 data points are required for prediction.")
        
        # Check if data has been normalized
        if not self._check_data_normalization(data):
            self.logger.warning("Data appears to be unnormalized. Consider scaling before prediction.")
        
        # Log volatility and autocorrelation
        volatility = data.std().mean()
        autocorr = data.apply(lambda col: col.autocorr(lag=1)).mean()
        self.logger.info(f"Prediction input volatility: {volatility:.6f}, autocorrelation: {autocorr:.6f}")
        
        # Prepare data for LSTM (reshape to [samples, timesteps, features])
        seq_len = self.config["sequence_length"]
        X_seq = []
        for i in range(len(data) - seq_len + 1):
            X_seq.append(data.iloc[i:i+seq_len].values)
        X_seq = np.array(X_seq)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_seq)
        
        # Create dataset and dataloader for batch-wise evaluation
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Predict in batches to reduce memory usage
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)
                batch_preds = self.model(batch_X)
                predictions.append(batch_preds.cpu().numpy())
        
        # Concatenate all batch predictions
        all_predictions = np.concatenate(predictions, axis=0)
        return all_predictions.flatten()

    def _check_data_normalization(self, data: pd.DataFrame) -> bool:
        """Check if data appears to be normalized (mean close to 0, std close to 1)."""
        try:
            mean_abs = data.mean().abs().mean()
            std_val = data.std().mean()
            
            # Check if data is roughly normalized
            is_normalized = mean_abs < 0.1 and 0.5 < std_val < 2.0
            
            self.logger.info(f"Data normalization check - Mean abs: {mean_abs:.4f}, Std: {std_val:.4f}, Normalized: {is_normalized}")
            
            return is_normalized
        except Exception as e:
            self.logger.warning(f"Could not check data normalization: {e}")
            return False

    def _log_input_statistics(self, X: pd.DataFrame, y: pd.Series):
        """Log comprehensive input statistics for monitoring."""
        try:
            stats = {
                "X_shape": X.shape,
                "y_shape": y.shape,
                "X_mean": X.mean().mean(),
                "X_std": X.std().mean(),
                "X_min": X.min().min(),
                "X_max": X.max().max(),
                "y_mean": y.mean(),
                "y_std": y.std(),
                "y_min": y.min(),
                "y_max": y.max(),
                "X_nan_count": X.isnull().sum().sum(),
                "y_nan_count": y.isnull().sum(),
            }
            
            self.logger.info(f"Input statistics: {stats}")
            
        except Exception as e:
            self.logger.warning(f"Could not log input statistics: {e}")

    def save(self, path: str) -> None:
        """Save the model.

        Args:
            path (str): Path to save the model
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "scaler": self.scaler,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load the model.

        Args:
            path (str): Path to load the model from
        """
        try:
            # Try loading with weights_only=True first (PyTorch 2.6+ default)
            checkpoint = torch.load(path, weights_only=True)
            self.config = checkpoint["config"]
            self.model = self.build_model()
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # Note: scaler will need to be recreated since it's not allowed with weights_only=True
            self.scaler = StandardScaler()
            self.model.to(self.device)
        except Exception as e:
            # Fallback to weights_only=False for older PyTorch versions or compatibility
            try:
                checkpoint = torch.load(path, weights_only=False)
                self.config = checkpoint["config"]
                self.model = self.build_model()
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.scaler = checkpoint.get("scaler", StandardScaler())
                self.model.to(self.device)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model from {path}: {e2}")

    @cache_model_operation
    @safe_forecast(max_retries=2, retry_delay=0.5, log_errors=True)
    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.

        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast

        Returns:
            Dictionary containing forecast results
        """
        # Validate input data
        validate_forecast_input(data, min_length=20, require_numeric=True)
        
        start_time = time.time()
        
        # Make initial prediction
        self.predict(data)

        # Generate multi-step forecast
        forecast_values = []
        current_data = data.copy()

        for i in range(horizon):
            # Get prediction for next step
            pred = self.predict(current_data)
            forecast_values.append(pred[-1])

            # Update data for next iteration
            new_row = current_data.iloc[-1].copy()
            new_row[self.config["target_column"]] = pred[-1]
            current_data = pd.concat(
                [current_data, pd.DataFrame([new_row])], ignore_index=True
            )
            current_data = current_data.iloc[1:]  # Remove oldest row

        execution_time = time.time() - start_time
        confidence = 0.85  # Placeholder confidence
        
        # Log performance
        log_forecast_performance(
            model_name="LSTM",
            execution_time=execution_time,
            data_length=len(data),
            confidence=confidence
        )

        return {
            "forecast": np.array(forecast_values),
            "confidence": confidence,
            "model": "LSTM",
            "horizon": horizon,
        }

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
            plt.plot(
                data.index,
                data[self.config["target_column"]],
                label="Actual",
                color="blue",
            )
            plt.plot(
                data.index[self.config["sequence_length"] :],
                predictions,
                label="Predicted",
                color="red",
            )
            plt.title("LSTM Model Predictions")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting LSTM results: {e}")
            logger.error(f"Could not plot results: {e}")
