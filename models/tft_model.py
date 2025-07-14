"""Temporal Fusion Transformer (TFT) Model for Evolve Trading Platform.

This module implements TFT using PyTorch Lightning for multivariate
time series forecasting with interpretable attention mechanisms.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.preprocessing import StandardScaler

# PyTorch Lightning imports
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    LIGHTNING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch Lightning not available: {e}")
    LIGHTNING_AVAILABLE = False

    # Create dummy base class for when Lightning is not available
    class DummyLightningModule:
        def __init__(self):
            pass

        def save_hyperparameters(self):
            pass

        def log(self, *args, **kwargs):
            pass

        def configure_optimizers(self):
            return {}

    pl = type("pl", (), {"LightningModule": DummyLightningModule})()

# Data handling
try:
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data."""

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        static_features: Optional[List[str]] = None,
        time_features: Optional[List[str]] = None,
    ):
        """Initialize the time series dataset."""
        self.data = data.copy()
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.static_features = static_features or []
        self.time_features = time_features or []
        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for TFT model."""
        # Ensure data is sorted by time
        if "timestamp" in self.data.columns:
            self.data = self.data.sort_values("timestamp").reset_index(drop=True)
        # Add time features if not present
        if not self.time_features:
            self.data["day_of_week"] = pd.to_datetime(self.data.index).dayofweek
            self.data["month"] = pd.to_datetime(self.data.index).month
            self.data["hour"] = pd.to_datetime(self.data.index).hour
            self.time_features = ["day_of_week", "month", "hour"]
        # Normalize numerical features
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != self.target_column:
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                if std_val > 0:
                    self.data[col] = (self.data[col] - mean_val) / std_val
        # Create sequences
        self.sequences = []
        self.targets = []
        for i in range(
            len(self.data) - self.sequence_length - self.prediction_horizon + 1
        ):
            # Input sequence
            seq_data = self.data.iloc[i : i + self.sequence_length]
            # Target sequence
            target_data = self.data.iloc[
                i
                + self.sequence_length : i
                + self.sequence_length
                + self.prediction_horizon
            ]
            # Extract features
            sequence = {
                "target": torch.FloatTensor(seq_data[self.target_column].values),
                "static": torch.FloatTensor(seq_data[self.static_features].values)
                if self.static_features
                else torch.zeros(1),
                "time": torch.FloatTensor(seq_data[self.time_features].values)
                if self.time_features
                else torch.zeros(1),
                "features": torch.FloatTensor(
                    seq_data.drop(
                        columns=[self.target_column]
                        + self.static_features
                        + self.time_features
                    ).values
                ),
            }
            target = torch.FloatTensor(target_data[self.target_column].values)
            self.sequences.append(sequence)
            self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class VariableSelectionNetwork(nn.Module):
    """Variable selection network for TFT."""

    def __init__(self, input_size: int, hidden_size: int, num_variables: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_variables = num_variables
        # Variable selection weights
        self.feature_selection = nn.Linear(input_size, num_variables)
        self.context_projection = nn.Linear(hidden_size, num_variables)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Calculate variable selection weights
        feature_weights = self.feature_selection(
            x
        )  # [batch_size, num_variables, num_variables]
        context_weights = self.context_projection(c).unsqueeze(
            1
        )  # [batch_size, 1, num_variables]
        # Combine weights
        selection_weights = F.softmax(feature_weights + context_weights, dim=-1)
        # Apply selection
        selected_features = torch.sum(
            x.unsqueeze(-1) * selection_weights.unsqueeze(-2), dim=1
        )
        return selected_features


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention for TFT."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"
        # Attention layers
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Attention weights for interpretability
        self.attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = query.shape
        # Project to query, key, value
        Q = (
            self.query_projection(query)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key_projection(key)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value_projection(value)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # Reshape and project output
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        output = self.output_projection(context)
        return output


class TemporalVariableSelection(nn.Module):
    """Temporal variable selection network."""

    def __init__(self, input_size: int, hidden_size: int, num_variables: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_variables = num_variables
        # GRU for temporal processing
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # Variable selection
        self.variable_selection = VariableSelectionNetwork(
            input_size, hidden_size, num_variables
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        batch_size, seq_len, num_vars, input_size = x.shape

        # Reshape for GRU
        x_reshaped = x.view(batch_size * num_vars, seq_len, input_size)

        # Process with GRU
        gru_out, hidden = self.gru(x_reshaped)

        # Reshape back
        gru_out = gru_out.view(batch_size, num_vars, seq_len, self.hidden_size)
        context = hidden.view(batch_size, num_vars, self.hidden_size)

        # Variable selection
        selected_features = self.variable_selection(x, context.mean(dim=1))

        return selected_features, context


class TFTModel(nn.Module):
    """Temporal Fusion Transformer model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        prediction_horizon: int = 5,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.prediction_horizon = prediction_horizon

        # Input projections
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Temporal variable selection
        self.temporal_selection = TemporalVariableSelection(input_size, hidden_size, 1)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_size, prediction_horizon)

        # Attention for interpretability
        self.attention = InterpretableMultiHeadAttention(
            hidden_size, num_heads, dropout
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            mask: Attention mask

        Returns:
            Predictions [batch_size, prediction_horizon]
        """
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self._positional_encoding(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_mask=mask)

        # Apply attention for interpretability
        x = self.attention(x, x, x, mask)

        # Global average pooling
        x = x.mean(dim=1)

        # Output projection
        predictions = self.output_projection(x)

        return predictions

    def _positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        batch_size, seq_len, d_model = x.shape

        position = torch.arange(seq_len, device=x.device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device).float()
            * -(np.log(10000.0) / d_model)
        )

        pos_encoding = torch.zeros(seq_len, d_model, device=x.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding.unsqueeze(0)


class TFTLightningModule(pl.LightningModule):
    """PyTorch Lightning module for TFT."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        prediction_horizon: int = 5,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TFTModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            prediction_horizon=prediction_horizon,
        )

        self.learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon

        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        data, targets = batch

        # Forward pass
        predictions = self(data["features"])

        # Calculate loss
        loss = self.criterion(predictions, targets)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        data, targets = batch

        # Forward pass
        predictions = self(data["features"])

        # Calculate loss
        loss = self.criterion(predictions, targets)

        # Calculate additional metrics
        mae = F.l1_loss(predictions, targets)
        mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))) * 100

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae)
        self.log("val_mape", mape)

        return loss

    def test_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        data, targets = batch

        # Forward pass
        predictions = self(data["features"])

        # Calculate metrics
        loss = self.criterion(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))) * 100

        # Calculate directional accuracy
        pred_direction = torch.sign(predictions[:, 1:] - predictions[:, :-1])
        true_direction = torch.sign(targets[:, 1:] - targets[:, :-1])
        directional_accuracy = (pred_direction == true_direction).float().mean()

        return {
            "test_loss": loss,
            "test_mae": mae,
            "test_mape": mape,
            "directional_accuracy": directional_accuracy,
        }

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class TFTForecaster:
    """TFT-based forecaster for trading."""

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        """Initialize TFT forecaster.

        Args:
            sequence_length: Length of input sequence
            prediction_horizon: Number of steps to predict
            hidden_size: Hidden layer size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            learning_rate: Learning rate
        """
        if not LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning is required for TFT model")

        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.model = None
        self.trainer = None
        self.scaler = None

        # Create output directory
        self.output_dir = Path("models/tft")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized TFT Forecaster with sequence_length={sequence_length}, "
            f"prediction_horizon={prediction_horizon}"
        )

    def _prepare_data(
        self, data: pd.DataFrame, target_column: str
    ) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """Prepare data for training."""
        # Split data
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.15)

        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size : train_size + val_size]
        test_data = data.iloc[train_size + val_size :]

        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data, target_column, self.sequence_length, self.prediction_horizon
        )
        val_dataset = TimeSeriesDataset(
            val_data, target_column, self.sequence_length, self.prediction_horizon
        )
        test_dataset = TimeSeriesDataset(
            test_data, target_column, self.sequence_length, self.prediction_horizon
        )

        return train_dataset, val_dataset, test_dataset

    def train(
        self,
        data: pd.DataFrame,
        target_column: str,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """Train TFT model.

        Args:
            data: Training data
            target_column: Target variable column
            batch_size: Batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience

        Returns:
            Training results
        """
        logger.info("Starting TFT model training...")

        # Prepare data
        train_dataset, val_dataset, test_dataset = self._prepare_data(
            data, target_column
        )

        # Determine input size
        sample_features = train_dataset[0][0]["features"]
        input_size = sample_features.shape[-1]

        # Initialize model
        self.model = TFTLightningModule(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            prediction_horizon=self.prediction_horizon,
            learning_rate=self.learning_rate,
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            ModelCheckpoint(
                dirpath=str(self.output_dir),
                filename="tft_best_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ]

        # Initialize trainer (tensorboard logging removed)
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=None,  # TensorBoard logging removed
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.5,
        )

        # Train model
        self.trainer.fit(self.model, train_loader, val_loader)

        # Test model
        test_results = self.trainer.test(self.model, test_loader)

        # Save model
        model_path = self.output_dir / "tft_final_model.ckpt"
        self.trainer.save_checkpoint(str(model_path))

        # Compile results
        results = {
            "model_path": str(model_path),
            "test_results": test_results[0],
            "training_completed": True,
            "model_config": {
                "sequence_length": self.sequence_length,
                "prediction_horizon": self.prediction_horizon,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
            },
        }

        logger.info(
            f"TFT training completed. Test loss: {test_results[0]['test_loss']:.4f}"
        )
        return results

    def predict(self, data: pd.DataFrame, target_column: str) -> np.ndarray:
        """Generate predictions.

        Args:
            data: Input data
            target_column: Target variable column

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create dataset
        dataset = TimeSeriesDataset(
            data, target_column, self.sequence_length, self.prediction_horizon
        )

        # Create data loader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Generate predictions
        predictions = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                data_batch, _ = batch
                batch_pred = self.model(data_batch["features"])
                predictions.append(batch_pred.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def get_attention_weights(
        self, data: pd.DataFrame, target_column: str
    ) -> np.ndarray:
        """Get attention weights for interpretability.

        Args:
            data: Input data
            target_column: Target variable column

        Returns:
            Attention weights array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create dataset
        dataset = TimeSeriesDataset(
            data, target_column, self.sequence_length, self.prediction_horizon
        )

        # Get attention weights from model
        attention_weights = []
        self.model.eval()

        with torch.no_grad():
            for i in range(min(len(dataset), 10)):  # Sample first 10 sequences
                data_sample, _ = dataset[i]
                _ = self.model(data_sample["features"].unsqueeze(0))

                if hasattr(self.model.model.attention, "attention_weights"):
                    attention_weights.append(
                        self.model.model.attention.attention_weights.cpu().numpy()
                    )

        return np.array(attention_weights) if attention_weights else np.array([])

    def save_model(self, path: str):
        """Save trained model."""
        if self.trainer is not None:
            self.trainer.save_checkpoint(path)
            logger.info(f"TFT model saved to {path}")

    def load_model(self, path: str):
        """Load trained model."""
        if LIGHTNING_AVAILABLE:
            self.model = TFTLightningModule.load_from_checkpoint(path)
            logger.info(f"TFT model loaded from {path}")


def create_tft_forecaster(
    data: pd.DataFrame,
    target_column: str = "close",
    sequence_length: int = 60,
    prediction_horizon: int = 5,
) -> Dict[str, Any]:
    """Create and train TFT forecaster.

    Args:
        data: Market data
        target_column: Target variable column
        sequence_length: Input sequence length
        prediction_horizon: Prediction horizon

    Returns:
        Training results
    """
    try:
        # Initialize forecaster
        forecaster = TFTForecaster(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            learning_rate=1e-3,
        )

        # Train model
        results = forecaster.train(
            data=data,
            target_column=target_column,
            batch_size=32,
            max_epochs=50,
            patience=10,
        )

        return results

    except Exception as e:
        logger.error(f"Error creating TFT forecaster: {e}")
        return {"error": str(e)}


def create_tft_model(config: Dict[str, Any] = None):
    class DummyTFT:
        pass

    return DummyTFT()
