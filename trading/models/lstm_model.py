"""LSTM-based forecasting model with advanced features and robust error handling."""

# Standard library imports
import logging
import os
import hashlib
import pickle
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ PyTorch not available. Disabling LSTM models.")
    print(f"   Missing: {e}")
    torch = None
    nn = None
    Adam = None
    ReduceLROnPlateau = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ scikit-learn not available. Disabling data preprocessing.")
    print(f"   Missing: {e}")
    StandardScaler = None
    SKLEARN_AVAILABLE = False

import joblib
import time

# Local imports
from .base_model import BaseModel
from utils.model_cache import cache_model_operation, get_model_cache
from utils.forecast_helpers import safe_forecast, validate_forecast_input, log_forecast_performance
from trading.exceptions import ModelTrainingError, ModelPredictionError, ModelInitializationError

# Configure logging
logger = logging.getLogger(__name__)

class FallbackModel:
    """A fallback model using moving average when LSTM fails."""
    def __init__(self, window: int = 20):
        self.window = window
        self.last_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the fallback model."""
        try:
            if 'Close' in X.columns:
                self.last_value = X['Close'].iloc[-1]
            elif len(y) > 0:
                self.last_value = y.iloc[-1]
            else:
                self.last_value = 1000 # Default fallback value
        except Exception as e:
            logger.warning(f"Fallback model fit failed: {e}")
            self.last_value = 100

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict using simple moving average."""
        try:
            if 'Close' in data.columns:
                # Use moving average of close prices
                ma = data['Close'].rolling(window=self.window, min_periods=1).mean()
                return ma.values
            else:
                # Return constant prediction
                return np.full(len(data), self.last_value or 1000)
        except Exception as e:
            logger.warning(f"Fallback model prediction failed: {e}")
            return np.full(len(data), self.last_value or 10)

class LSTMModel(nn.Module):
    """A class to handle LSTM model for time series prediction with robust error handling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """Initialize the LSTM model with error handling."""
        if not TORCH_AVAILABLE:
            raise ModelInitializationError("PyTorch is not available. Cannot initialize LSTM model.")
        
        try:
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
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(f"LSTM model initialized on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize LSTM model: {e}")
            logger.error(traceback.format_exc())
            raise ModelInitializationError(f"LSTM model initialization failed: {str(e)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LSTM model with error handling."""
        try:
            # Move input to device
            x = x.to(self.device)

            # LSTM forward pass
            lstm_out, _ = self.lstm(x)

            # Use only the final timestep's output for prediction
            return self.fc(lstm_out[:, -1, :])
        except Exception as e:
            logger.error(f"LSTM forward pass failed: {e}")
            logger.error(traceback.format_exc())
            raise ModelPredictionError(f"LSTM forward pass failed: {str(e)}")

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
        """Train the LSTM model with comprehensive error handling and fallback."""
        try:
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
                error_msg = f"Insufficient valid data after cleaning (need at least 10 samples, got {len(X)})"
                logger.error(error_msg)
                raise ModelTrainingError(error_msg)

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
                try:
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
                    else:
                        patience_counter += 1

                    # Early stopping
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break

                except Exception as e:
                    logger.error(f"Training failed at epoch {epoch+1}: {e}")
                    logger.error(traceback.format_exc())
                    raise ModelTrainingError(f"Training failed at epoch {epoch+1}: {str(e)}")

            # Load best model
            if best_model_state is not None:
                self.load_state_dict(best_model_state)

            return history

        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            logger.error(traceback.format_exc())
            raise ModelTrainingError(f"LSTM training failed: {str(e)}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict with error handling and fallback."""
        try:
            # Validate input data
            if data.empty:
                raise ModelPredictionError("Input data is empty")

            if data.isnull().any().any():
                logger.warning("Input data contains NaN values, filling with forward fill")
                data = data.fillna(method='ffill').fillna(method='bfill')

            # Perform prediction
            X_scaled = self.scaler.transform(data)
            X_tensor = torch.FloatTensor(X_scaled)
            X_seq = self._create_sequences(X_tensor)

            self.eval()
            with torch.no_grad():
                predictions = self(X_seq)
                predictions = predictions.cpu().numpy()

            return predictions

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            logger.error(traceback.format_exc())

            # Return fallback prediction
            logger.info("Using fallback prediction (moving average)")
            fallback = FallbackModel()
            fallback.fit(data, pd.Series())
            return fallback.predict(data)

    def _create_sequences(self, data: torch.Tensor) -> torch.Tensor:
        """Create sequences for LSTM with error handling."""
        try:
            sequence_length = getattr(self, 'config', {}).get("sequence_length", 10)
            sequences = []
            for i in range(len(data) - sequence_length + 1):
                sequences.append(data[i : i + sequence_length])
            return torch.stack(sequences)
        except Exception as e:
            logger.error(f"Sequence creation failed: {e}")
            raise ModelPredictionError(f"Sequence creation failed: {str(e)}")

    def set_sequence_length(self, new_length: int):
        """Set sequence length with validation."""
        try:
            if new_length <= 0:
                raise ValueError("Sequence length must be positive")
            if not hasattr(self, 'config'):
                self.config = {}
            self.config["sequence_length"] = new_length
            self.logger.info(f"Sequence length set to {new_length}")
        except Exception as e:
            logger.error(f"Failed to set sequence length: {e}")
            raise ModelInitializationError(f"Failed to set sequence length: {str(e)}")

    def get_optimal_sequence_length(self, data: pd.DataFrame) -> int:
        """Get optimal sequence length with error handling."""
        try:
            # Simple heuristic: use 10% of data length, min 5, max 50
            optimal_length = max(5, min(50, len(data) // 10))
            self.logger.info(f"Optimal sequence length calculated: {optimal_length}")
            return optimal_length
        except Exception as e:
            logger.error(f"Failed to calculate optimal sequence length: {e}")
            return 10 # Default fallback

    def auto_adjust_sequence_length(self, data: pd.DataFrame):
        """Automatically adjust sequence length with error handling."""
        try:
            optimal_length = self.get_optimal_sequence_length(data)
            self.set_sequence_length(optimal_length)
        except Exception as e:
            logger.error(f"Failed to auto-adjust sequence length: {e}")
            # Keep current length or use default


class LSTMForecaster(BaseModel):
    """LSTM-based forecasting model with comprehensive error handling and fallback mechanisms."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LSTM forecaster with error handling."""
        # Initialize availability flag
        self.available = True
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is not available. Cannot initialize LSTM forecaster.")
            self.available = False
            print("âš ï¸ LSTMForecaster unavailable due to missing PyTorch")
            return
        
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is not available. Cannot initialize LSTM forecaster.")
            self.available = False
            print("âš ï¸ LSTMForecaster unavailable due to missing scikit-learn")
            return
        
        try:
            super().__init__(config)
            
            # Validate configuration
            self._validate_config()
            
            # Initialize components
            self.model = None
            self.scaler = StandardScaler()
            self.is_trained = False
            self.training_history = {"loss": [], "val_loss": []}
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize fallback model
            self.fallback_model = FallbackModel()
            
            # Build model with error handling
            try:
                self.model = self.build_model()
                self.model.to(self.device)
                self._init_weights()
                logger.info(f"LSTM model initialized successfully on device: {self.device}")
            except Exception as e:
                logger.error(f"LSTM model build failed: {e}")
                self.model = None
                print("âš ï¸ LSTM model unavailable due to model build failure")
                print(f"   Error: {e}")
                # Don't set available to False here as we have fallback
                
        except Exception as e:
            logger.error(f"Failed to initialize LSTM forecaster: {e}")
            logger.error(traceback.format_exc())
            self.available = False
            print("âš ï¸ LSTMForecaster unavailable due to initialization failure")
            print(f"   Error: {e}")
            # Don't raise exception, just mark as unavailable

    def _build_fallback_model(self) -> nn.Module:
        """Build a simple fallback LSTM model when the main model fails."""
        try:
            class SimpleLSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])
            
            input_size = len(self.config["feature_columns"])
            hidden_size = min(32, input_size * 2) # Conservative hidden size
            
            return SimpleLSTMModel(input_size, hidden_size)
            
        except Exception as e:
            logger.error(f"Fallback model building failed: {e}")
            # Return a dummy model that always predicts the last value
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(1, 1)
                
                def forward(self, x):
                    return self.fc(x[:, -1, :])
            
            return DummyModel()

    def build_model(self) -> nn.Module:
        """Build and return the LSTM model with error handling."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using fallback model")
            return self._build_fallback_model()
        
        try:
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
                    try:
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
                    except Exception as e:
                        logger.error(f"LSTM model forward pass failed: {e}")
                        # Return zeros as fallback
                        return torch.zeros(x.size(0), 1, device=x.device)

            return LSTMForecasterModel(input_size, hidden_size, num_layers, dropout, use_batch_norm, use_layer_norm, additional_dropout)
            
        except Exception as e:
            logger.error(f"Model building failed: {e}")
            logger.error(traceback.format_exc())
            raise ModelInitializationError(f"LSTM model building failed: {str(e)}")

    def _validate_config(self):
        """Validate model configuration with detailed error messages."""
        try:
            required_keys = [
                "input_size",
                "hidden_size", 
                "num_layers",
                "dropout",
                "sequence_length",
                "feature_columns",
                "target_column",
            ]
            
            missing_keys = [key for key in required_keys if key not in self.config]
            if missing_keys:
                raise ValueError(f"Missing required config keys: {missing_keys}")

            # Validate sequence length
            if self.config["sequence_length"] <= 0:
                raise ValueError("sequence_length must be positive")
            if self.config["sequence_length"] > self.config.get("max_sequence_length", 100):
                raise ValueError(
                    f"sequence_length {self.config['sequence_length']} exceeds maximum allowed value of {self.config.get('max_sequence_length', 100)}"
                )

            # Validate feature columns
            if not self.config["feature_columns"]:
                raise ValueError("feature_columns cannot be empty")
            if self.config["target_column"] not in self.config["feature_columns"]:
                raise ValueError(f"target_column '{self.config['target_column']}' must be in feature_columns")

            # Validate batch size limits
            if "max_batch_size" in self.config and self.config["max_batch_size"] <= 0:
                raise ValueError("max_batch_size must be positive")

            # Validate epoch limits
            if "max_epochs" in self.config and self.config["max_epochs"] <= 0:
                raise ValueError("max_epochs must be positive")
                
            # Validate dropout
            if not 0 <= self.config["dropout"] <= 1:
                raise ValueError("dropout must be between 0 and 1")
                
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            logger.error(traceback.format_exc())
            raise ModelInitializationError(f"Configuration validation failed: {str(e)}")

    def _init_weights(self):
        """Initialize model weights with error handling."""
        try:
            for name, param in self.model.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            self.logger.info("Model weights initialized successfully")
        except Exception as e:
            logger.warning(f"Weight initialization failed: {e}")
            # Continue without weight initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with comprehensive error handling and memory management."""
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
                logger.error("GPU out of memory during forward pass")
                raise MemoryError("GPU out of memory during forward pass")
            raise e
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            logger.error(traceback.format_exc())
            raise ModelPredictionError(f"Forward pass failed: {str(e)}")

    def _prepare_data(
        self, data: pd.DataFrame, is_training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare input data for training or prediction with error handling."""
        try:
            # Validate input data
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Input data must be pandas DataFrame")
            if data.empty:
                raise ValueError("Input data is empty")
                
            # Check for required columns
            missing_cols = [col for col in self.config["feature_columns"] if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required feature columns: {missing_cols}")

            # Check data size
            if len(data) < self.config["sequence_length"]:
                raise ValueError(
                    f"Data length {len(data)} is less than sequence length {self.config['sequence_length']}"
                )

            # Handle NaN values
            if data.isnull().any().any():
                logger.warning("Input data contains NaN values, filling with forward fill")
                data = data.fillna(method='ffill').fillna(method='bfill')

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
                        self.config["feature_columns"].index(self.config["target_column"]),
                    ]
                )
                return X_seq, y
            else:
                return X_seq, None

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            logger.error(traceback.format_exc())
            raise ModelPredictionError(f"Data preparation failed: {str(e)}")

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
        if not self.available:
            print("âš ï¸ LSTMForecaster unavailable due to initialization failure")
            return {
                "train_loss": [0.0],
                "val_loss": [0.0],
                "cached": False,
                "error": "LSTMForecaster unavailable due to initialization failure"
            }
        
        try:
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
                error_msg = f"At least 20 data points are required after cleaning, got {len(X)}"
                logger.error(error_msg)
                raise ModelTrainingError(error_msg)

            # Check if data has been normalized
            if not self._check_data_normalization(X):
                self.logger.warning("Data appears to be unnormalized. Consider scaling before training.")

            # Log volatility and autocorrelation
            try:
                volatility = X.std().mean()
                autocorr = X.apply(lambda col: col.autocorr(lag=1)).mean()
                self.logger.info(f"Input volatility: {volatility:.6f}, autocorrelation: {autocorr:.6f}")
            except Exception as e:
                logger.warning(f"Failed to calculate data statistics: {e}")

            # Prepare data for LSTM (reshape to [samples, timesteps, features])
            try:
                seq_len = self.config["sequence_length"]
                X_seq = []
                y_seq = []
                for i in range(len(X) - seq_len):
                    X_seq.append(X.iloc[i:i+seq_len].values)
                    y_seq.append(y.iloc[i+seq_len])
                X_seq = np.array(X_seq)
                y_seq = np.array(y_seq)
            except Exception as e:
                logger.error(f"Failed to prepare sequences: {e}")
                raise ModelTrainingError(f"Sequence preparation failed: {str(e)}")

            # Train/validation split
            try:
                split_idx = int(len(X_seq) * (1 - validation_split))
                X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            except Exception as e:
                logger.error(f"Failed to split data: {e}")
                raise ModelTrainingError(f"Data splitting failed: {str(e)}")

            # Convert to PyTorch tensors
            try:
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
                X_val_tensor = torch.FloatTensor(X_val)
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
            except Exception as e:
                logger.error(f"Failed to convert to tensors: {e}")
                raise ModelTrainingError(f"Tensor conversion failed: {str(e)}")

            # Create data loaders
            try:
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            except Exception as e:
                logger.error(f"Failed to create data loaders: {e}")
                raise ModelTrainingError(f"DataLoader creation failed: {str(e)}")

            # Setup training
            try:
                optimizer = Adam(self.model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            except Exception as e:
                logger.error(f"Failed to setup training components: {e}")
                raise ModelTrainingError(f"Training setup failed: {str(e)}")

            # Training history
            history = {"train_loss": [], "val_loss": []}

            # Training loop with error handling
            try:
                self.model.train()
                for epoch in range(epochs):
                    try:
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

                    except Exception as e:
                        logger.error(f"Training failed at epoch {epoch+1}: {e}")
                        logger.error(traceback.format_exc())
                        raise ModelTrainingError(f"Training failed at epoch {epoch+1}: {str(e)}")

            except Exception as e:
                logger.error(f"Training loop failed: {e}")
                raise ModelTrainingError(f"Training loop failed: {str(e)}")

            # Cache the compiled model
            try:
                self.compiled_model = self.model
                self.last_input_hash = input_hash
                self._save_cached_model(input_hash)
            except Exception as e:
                logger.warning(f"Failed to cache model: {e}")

            return history

        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            logger.error(traceback.format_exc())
            raise ModelTrainingError(f"LSTM training failed: {str(e)}")

    def predict(self, data: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        """Predict using the LSTM model with input validation, logging, and batch-wise evaluation."""
        if not self.available:
            print("âš ï¸ LSTMForecaster unavailable due to initialization failure")
            # Return simple fallback prediction
            if "Close" in data.columns:
                return data["Close"].rolling(window=20, min_periods=1).mean().values
            else:
                return np.full(len(data), 1000.0)
        
        try:
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

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return simple fallback
            logger.info("Using simple fallback prediction")
            if 'Close' in data.columns:
                return data['Close'].rolling(window=20, min_periods=1).mean().values
            else:
                return np.full(len(data), 1000)  # Default value

    def _check_data_normalization(self, data: pd.DataFrame) -> bool:
        """Check if data appears to be normalized (mean close to 0, std close to 1)."""
        try:
            # Check if data is roughly in the range [-3cal for normalized data)
            data_range = data.abs().max().max()
            return data_range <= 3
        except Exception as e:
            logger.warning(f"Failed to check data normalization: {e}")
            return False

    def _log_input_statistics(self, X: pd.DataFrame, y: pd.Series):
        """Log comprehensive input statistics for monitoring."""
        try:
            self.logger.info(f"Input shape: {X.shape}")
            self.logger.info(f"Target shape: {y.shape}")
            self.logger.info(f"Input columns: {list(X.columns)}")
            self.logger.info(f"Input dtypes: {X.dtypes.to_dict()}")
            self.logger.info(f"Input range: {X.min().min():.4f} to {X.max().max():0.4f}")
            self.logger.info(f"Target range: {y.min():.4f} to {y.max():.4f}")
        except Exception as e:
            logger.warning(f"Failed to log input statistics: {e}")

    def save(self, path: str) -> None:
        """Save the model.

        Args:
            path (str): Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state and configuration
            torch.save({
               "model_state_dict": self.model.state_dict(),
            "config": self.config,
               "scaler": self.scaler,
                "device": self.device
            }, path)
            
            self.logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            logger.error(traceback.format_exc())
            raise ModelPredictionError(f"Model save failed: {str(e)}")

    def load(self, path: str) -> None:
        """Load the model.

        Args:
            path (str): Path to load the model from
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            # Load model state and configuration
            checkpoint = torch.load(path, map_location=self.device)
            
            # Update configuration
            if 'config' in checkpoint:
                self.config.update(checkpoint['config'])
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load scaler
            if 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
            
            self.logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise ModelPredictionError(f"Model load failed: {str(e)}")

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
        if not self.available:
            print("âš ï¸ LSTMForecaster unavailable due to initialization failure")
            # Return simple fallback forecast
            fallback_forecast = np.full(horizon, 1000.0)
            last_date = data.index[-1] if hasattr(data.index, "freq") else pd.Timestamp.now()
            forecast_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
            
            return {
                "forecast": fallback_forecast,
                "dates": forecast_dates,
                "confidence": np.full(horizon, 0.1),
                "model_type": "LSTM_Unavailable",
                "horizon": horizon,
                "error": "LSTMForecaster unavailable due to initialization failure"
            }
        
        try:
            # Validate inputs
            if data.empty:
                raise ModelPredictionError("Input data is empty")
            
            if horizon <= 0:
                raise ValueError("Forecast horizon must be positive")
            
            if horizon > 100:
                logger.warning(f"Large forecast horizon ({horizon}) may be unreliable")

            # Check if model is trained
            if not hasattr(self, "scaler") or self.scaler is None:
                raise ModelPredictionError("Model must be trained before forecasting")

            # Prepare data
            try:
                X_seq, _ = self._prepare_data(data, is_training=False)
            except Exception as e:
                logger.error(f"Data preparation failed: {e}")
                raise ModelPredictionError(f"Data preparation failed: {str(e)}")

            # Generate forecast
            try:
                self.model.eval()
                forecasts = []
                
                # Use the last sequence as starting point
                current_sequence = X_seq[-1:].clone()
                
                with torch.no_grad():
                    for _ in range(horizon):
                        # Make prediction
                        pred = self.model(current_sequence.to(self.device))
                        forecasts.append(pred.cpu().numpy())
                        
                        # Update sequence for next prediction (simple approach)
                        # In a more sophisticated implementation, you might want to use the prediction
                        # to update the sequence properly
                        current_sequence = torch.roll(current_sequence, -1, dims=1)
                        current_sequence[:, -1, :] = pred[0, 0]
                
                forecasts = np.concatenate(forecasts, axis=0).flatten()
                
                # Inverse transform forecasts
                if hasattr(self, "scaler") and self.scaler is not None:
                    forecasts = self.scaler.inverse_transform(forecasts.reshape(-1, 1))
                
                # Create forecast dates
                last_date = data.index[-1] if hasattr(data.index, "freq") else pd.Timestamp.now()
                forecast_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
                
                return {
                    "forecast": forecasts,
                    "dates": forecast_dates,
                    "confidence": np.full(horizon, 0.8), # Placeholder confidence
                    "model_type": "LSTM",
                    "horizon": horizon
                }

            except Exception as e:
                logger.error(f"Forecast generation failed: {e}")
                logger.error(traceback.format_exc())
                
                # Return fallback forecast
                logger.info("Using fallback forecast (extrapolation)")
                if 'Close' in data.columns:
                    last_value = data['Close'].iloc[-1]
                    trend = data['Close'].diff().mean()
                    fallback_forecast = [last_value + trend * (i+1) for i in range(horizon)]
                else:
                    fallback_forecast = np.full(horizon, 1000)
                
                last_date = data.index[-1] if hasattr(data.index, "freq") else pd.Timestamp.now()
                forecast_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
                
                return {
                  "forecast": fallback_forecast,
                   "dates": forecast_dates,
                  "confidence": np.full(horizon, 0.5),  # Lower confidence for fallback
                  "model_type": "LSTM_Fallback",
                    "horizon": horizon
                }

        except Exception as e:
            logger.error(f"LSTM forecast failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return simple fallback
            logger.info("Using simple fallback forecast")
            fallback_forecast = np.full(horizon, 1000)
            last_date = data.index[-1] if hasattr(data.index, "freq") else pd.Timestamp.now()
            forecast_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
            
            return {
              "forecast": fallback_forecast,
               "dates": forecast_dates,
              "confidence": np.full(horizon, 0.3), # Very low confidence
              "model_type": "LSTM_Simple_Fallback",
                "horizon": horizon
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
