from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from trading.models.base_model import BaseModel


class TemporalBlock(nn.Module):
    """Temporal block for TCN."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        """Initialize temporal block.

        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout rate
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.dropout1,
            self.conv2,
            self.bn2,
            nn.ReLU(),
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize TCN.

        Args:
            num_inputs: Number of input channels
            num_channels: List of number of channels in each layer
            kernel_size: Size of convolutional kernel
            dropout: Dropout rate
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_inputs)

        Returns:
            Output tensor of shape (batch_size, seq_len, num_channels[-1])
        """
        x = x.transpose(1, 2)  # (batch_size, num_inputs, seq_len)
        x = self.network(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_channels[-1])
        return x


class TCNModel(BaseModel):
    """Temporal Convolutional Network for time series forecasting."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize TCN forecaster.

        Args:
            config: Configuration dictionary containing:
                - input_size: Size of input features
                - output_size: Size of output features
                - num_channels: List of number of channels in each layer
                - kernel_size: Size of convolutional kernel
                - dropout: Dropout rate
                - sequence_length: Length of input sequences
                - feature_columns: List of feature column names
                - target_column: Name of target column
                - learning_rate: Learning rate
                - use_lr_scheduler: Whether to use learning rate scheduler
        """
        super().__init__(config)

        # Validate required config parameters
        required_params = [
            "input_size",
            "output_size",
            "num_channels",
            "kernel_size",
            "dropout",
            "sequence_length",
            "feature_columns",
            "target_column",
        ]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")

        # Set model parameters
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.num_channels = config["num_channels"]
        self.kernel_size = config["kernel_size"]
        self.dropout = config["dropout"]
        self.sequence_length = config["sequence_length"]
        self.feature_columns = config["feature_columns"]
        self.target_column = config["target_column"]

        # Validate sequence length
        if self.sequence_length < 2:
            raise ValueError("Sequence length must be at least 2")

        # Validate feature columns
        if len(self.feature_columns) != self.input_size:
            raise ValueError(
                f"Number of feature columns ({len(self.feature_columns)}) "
                f"must match input_size ({self.input_size})"
            )

        # Set up model architecture
        self._setup_model()

        # Initialize training state
        self.history = []
        self.optimizer = None
        self.scheduler = None

    def _setup_model(self):
        """Set up the TCN model architecture."""
        # Create TCN layers
        self.tcn = TemporalConvNet(
            num_inputs=self.input_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        ).to(self.device)

        # Create output layer
        self.fc = nn.Linear(self.num_channels[-1], self.output_size).to(self.device)

        # Initialize weights using Xavier uniform
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Pass through TCN layers
        x = self.tcn(x)

        # Take the last time step output
        x = x[:, -1, :]

        # Pass through output layer
        x = self.fc(x)
        return x

    def fit(self, data: pd.DataFrame, epochs: int = 10, batch_size: int = 32) -> None:
        """Train the model.

        Args:
            data: Input data as pandas DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.tcn.train()

        # Prepare data
        X, y = self._prepare_data(data, is_training=True)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters())

        # Initialize scheduler if enabled
        if self.config["use_lr_scheduler"]:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            )

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size]
                batch_y = y[i : i + batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                output = self(batch_X)
                loss = F.mse_loss(output, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Update learning rate if scheduler is enabled
            if self.scheduler is not None:
                self.scheduler.step(epoch_loss / (len(X) / batch_size))

            self.history.append(epoch_loss / (len(X) / batch_size))

    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions.

        Args:
            data: Input data as pandas DataFrame

        Returns:
            Dictionary containing predictions
        """
        self.tcn.eval()

        # Prepare data
        X, _ = self._prepare_data(data, is_training=False)

        with torch.no_grad():
            predictions = self(X).cpu().numpy()
            return {"predictions": predictions}

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
            predictions = self.predict(data)

            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()

            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data)
                forecast_values.append(
                    pred["predictions"][-1][0]
                )  # Extract scalar value

                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row[self.target_column] = pred["predictions"][-1][
                    0
                ]  # Update with prediction
                current_data = pd.concat(
                    [current_data, pd.DataFrame([new_row])], ignore_index=True
                )
                current_data = current_data.iloc[1:]  # Remove oldest row

            return {
                "forecast": np.array(forecast_values),
                "confidence": 0.8,  # TCN confidence
                "model": "TCN",
                "horizon": horizon,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
            }

        except Exception as e:
            import logging

            logging.error(f"Error in TCN model forecast: {e}")
            raise RuntimeError(f"TCN model forecasting failed: {e}")

    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> tuple:
        """Prepare data for training or prediction.

        Args:
            data: Input data as pandas DataFrame
            is_training: Whether data is for training

        Returns:
            Tuple of (X, y) tensors
        """
        # Convert to numpy arrays
        X = data[self.feature_columns].values
        y = data[self.target_column].values[1:]  # Predict next day's close

        # Normalize
        if is_training:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.y_mean = y.mean()
            self.y_std = y.std()

        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        # Convert to tensors
        X = torch.FloatTensor(X[:-1])  # Remove last row as we don't have target for it
        y = torch.FloatTensor(y).unsqueeze(-1)

        return X, y

    def save(self, path: str) -> Dict[str, Any]:
        """Save model state.

        Args:
            path: Path to save model state

        Returns:
            Dictionary with save status and metadata
        """
        try:
            state = {
                "model_state": self.state_dict(),
                "config": self.config,
                "history": self.history,
                "best_model_state": self.best_model_state,
                "best_val_loss": self.best_val_loss,
            }
            torch.save(state, path)
            return {
                "success": True,
                "path": path,
                "model_type": "TCN",
                "config_keys": list(self.config.keys()) if self.config else [],
                "history_length": len(self.history) if self.history else 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    def load(self, path: str) -> Dict[str, Any]:
        """Load model state.

        Args:
            path: Path to load model state from

        Returns:
            Dictionary with load status and metadata
        """
        try:
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state["model_state"])
            self.config = state["config"]
            self.history = state["history"]
            self.best_model_state = state["best_model_state"]
            self.best_val_loss = state["best_val_loss"]
            return {
                "success": True,
                "path": path,
                "model_type": "TCN",
                "config_keys": list(self.config.keys()) if self.config else [],
                "history_length": len(self.history) if self.history else 0,
                "best_val_loss": self.best_val_loss,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    def _train_step(self, data: torch.Tensor) -> float:
        """Perform a single training step.

        Args:
            data: Training data

        Returns:
            Training loss
        """
        self.optimizer.zero_grad()
        output = self(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _validate_step(self, data: torch.Tensor) -> float:
        """Perform a single validation step.

        Args:
            data: Validation data

        Returns:
            Validation loss
        """
        with torch.no_grad():
            output = self(data)
            loss = F.mse_loss(output, data)
            return loss.item()

    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self.optimizer is None:
            lr = self.config.get("learning_rate", 0.001)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        if self.config.get("use_lr_scheduler", False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )

    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> tuple:
        """Prepare data for training or prediction.

        Args:
            data: Input data as pandas DataFrame
            is_training: Whether data is for training

        Returns:
            Tuple of (X, y) tensors
        """
        # Convert to numpy arrays
        X = data[self.feature_columns].values
        y = data[self.target_column].values[1:]  # Predict next day's close

        # Normalize
        if is_training:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.y_mean = y.mean()
            self.y_std = y.std()

        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        # Convert to tensors
        X = torch.FloatTensor(X[:-1])  # Remove last row as we don't have target for it
        y = torch.FloatTensor(y).unsqueeze(-1)

        return X, y
