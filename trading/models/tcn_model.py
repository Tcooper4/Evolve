"""Temporal Convolutional Network for time series forecasting."""

# Standard library imports
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError as e:
    print("[WARN] PyTorch not available. Disabling TCN models.")
    print(f"   Missing: {e}")
    torch = None
    nn = None
    TORCH_AVAILABLE = False

# Local imports
from .base_model import BaseModel, ValidationError

logger = logging.getLogger(__name__)


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
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create TemporalBlock.")
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
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
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
        """Forward pass through temporal block.

        Args:
            x: Input tensor of shape (batch_size, n_inputs, seq_len)

        Returns:
            Output tensor of shape (batch_size, n_outputs, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        # Ensure residual and output have matching sequence length for addition
        if out.size(-1) > res.size(-1):
            out = out[..., -res.size(-1) :]
        elif out.size(-1) < res.size(-1):
            res = res[..., -out.size(-1) :]

        return self.relu(out + res)


# @ModelRegistry.register('TCN')


class TCNModel(BaseModel):
    """Temporal Convolutional Network for time series forecasting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create TCNModel.")
        """Initialize TCN model.

        Args:
            config: Model configuration dictionary with the following keys:
                - input_size: Number of input features (default: 2)
                - output_size: Number of output features (default: 1)
                - num_channels: List of channel sizes for each layer (default: [64, 128, 256])
                - kernel_size: Size of convolutional kernel (default: 3)
                - dropout: Dropout rate (default: 0.2)
                - sequence_length: Length of input sequences (default: 10)
                - feature_columns: List of column names to use as features (default: ['close', 'volume'])
                - target_column: Column name to predict (default: 'close')
                - learning_rate: Learning rate (default: 0.001)
                - use_lr_scheduler: Whether to use learning rate scheduler (default: True)
        """
        if config is None:
            config = {}

        # Set default configuration
        default_config = {
            "input_size": 2,
            "output_size": 1,
            "num_channels": [64, 128, 256],
            "kernel_size": 3,
            "dropout": 0.2,
            "sequence_length": 10,
            "feature_columns": ["close", "volume"],
            "target_column": "close",
            "learning_rate": 0.001,
            "use_lr_scheduler": True,
        }
        default_config.update(config)

        super().__init__(default_config)
        # BaseModel.__init__ already validates config and calls build_model;
        # we keep an explicit validation here for clarity.
        self._validate_config()
        # __init__ should not return anything

    def _validate_config(self) -> None:
        """Validate model configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate required parameters
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
            if param not in self.config:
                raise ValidationError(f"Missing required parameter: {param}")

        # Validate parameter values
        if self.config["input_size"] <= 0:
            raise ValidationError("input_size must be positive")
        if self.config["output_size"] <= 0:
            raise ValidationError("output_size must be positive")
        if not self.config["num_channels"]:
            raise ValidationError("num_channels cannot be empty")
        if self.config["kernel_size"] <= 0:
            raise ValidationError("kernel_size must be positive")
        if not 0 <= self.config["dropout"] <= 1:
            raise ValidationError("dropout must be between 0 and 1")
        if self.config["sequence_length"] < 2:
            raise ValidationError("sequence_length must be at least 2")
        if not self.config["feature_columns"]:
            raise ValidationError("feature_columns cannot be empty")
        # Set input_size dynamically from feature_columns instead of raising
        self.config["input_size"] = len(self.config["feature_columns"])

    def _setup_model(self) -> None:
        """Setup the TCN model architecture."""
        layers = []
        num_levels = len(self.config["num_channels"])
        for i in range(num_levels):
            dilation = 2**i
            in_channels = (
                self.config["input_size"]
                if i == 0
                else self.config["num_channels"][i - 1]
            )
            out_channels = self.config["num_channels"][i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    self.config["kernel_size"],
                    stride=1,
                    dilation=dilation,
                    padding=(self.config["kernel_size"] - 1) * dilation,
                    dropout=self.config["dropout"],
                )
            ]

        # TemporalConvNet followed by a linear readout on the last timestep
        self.tcn = nn.Sequential(*layers).to(self.device)
        self.linear = nn.Linear(
            self.config["num_channels"][-1], self.config["output_size"]
        ).to(self.device)

    def build_model(self):
        """Build the underlying TCN network to satisfy BaseModel.build_model."""
        self._setup_model()
        # Return a simple nn.Module view that mirrors predict/fit behaviour
        class _TCNWrapper(nn.Module):
            def __init__(self, tcn, linear):
                super().__init__()
                self.tcn = tcn
                self.linear = linear

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x expected as (batch, seq_len, input_size)
                x = x.transpose(1, 2)
                features = self.tcn(x)
                last_step = features[:, :, -1]
                return self.linear(last_step)

        return _TCNWrapper(self.tcn, self.linear).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        x = x.transpose(1, 2)  # Change to (batch_size, input_size, seq_len)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # Change back to (batch_size, seq_len, num_channels)
        x = self.linear(x[:, -1, :])  # Take last time step
        return x

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance lowercase columns to title case."""
        if df is None or df.empty:
            return df
        if "Close" not in df.columns and "close" in df.columns:
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return df

    def _prepare_data(
        self, data: pd.DataFrame, is_training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training or prediction.

        Args:
            data: Input data as pandas DataFrame
            is_training: Whether data is for training

        Returns:
            Tuple of (X, y) tensors where:
                X: Shape (batch_size, sequence_length, input_size)
                y: Shape (batch_size, output_size)
        """
        data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
        # Resolve column names (yfinance lowercase -> title case)
        fc = [c if c in data.columns else ("Close" if c == "close" else "Volume" if c == "volume" else c) for c in self.config["feature_columns"]]
        fc = [c for c in fc if c in data.columns]
        tc = self.config["target_column"] if self.config["target_column"] in data.columns else ("Close" if self.config["target_column"] == "close" else self.config["target_column"])
        if not fc:
            fc = ["Close"] if "Close" in data.columns else list(data.columns)[:1]
        if tc not in data.columns:
            tc = data.columns[0]
        # Validate data
        if data.isnull().any().any():
            raise ValidationError("Data contains missing values")

        # Check if all required columns exist
        missing_cols = [col for col in fc + [tc] if col not in data.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")

        # Convert to numpy arrays; train on next-period return when target is price
        X = data[fc].values
        if tc in ("Close", "close") and len(data) > self.config["sequence_length"] + 1:
            price_vals = data[tc].values
            returns = np.diff(price_vals) / (price_vals[:-1] + 1e-10)
            y = returns[self.config["sequence_length"] - 1:]  # align: sequence i -> return i to i+1
        else:
            y = data[tc].values[self.config["sequence_length"]:]
        if len(y) > len(X) - self.config["sequence_length"]:
            y = y[: len(X) - self.config["sequence_length"]]

        # Create sequences: shape (batch, seq_len, features)
        X_sequences = []
        for i in range(len(X) - self.config["sequence_length"]):
            X_sequences.append(X[i : i + self.config["sequence_length"]])
        X = np.array(X_sequences)

        # Reorder to (batch, features, seq_len) for Conv1d
        if X.ndim == 3:
            X = np.transpose(X, (0, 2, 1))

        # Normalize
        if is_training:
            self.X_mean = X.mean(axis=(0, 1))
            self.X_std = X.std(axis=(0, 1))
            self.y_mean = y.mean()
            self.y_std = y.std()

        X = (X - self.X_mean) / (self.X_std + 1e-10)
        y = (y - self.y_mean) / (self.y_std + 1e-10)

        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(-1)

        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)

        return X, y

    def summary(self):
        super().summary()

    def infer(self):
        super().infer()

    def shap_interpret(self, X_sample):
        """Run SHAP interpretability on a sample batch."""
        try:
            import shap
        except ImportError:
            logger.warning(
                "SHAP is not installed. Please install it with 'pip install shap'."
            )
            return
        explainer = shap.DeepExplainer(self.model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample.cpu().numpy())

    def test_synthetic(self):
        """Test model on synthetic data - DEPRECATED (P3: clearly marked).
        Use real market data for testing. Will be removed in a future release."""
        import warnings
        warnings.warn(
            "TCNModel.test_synthetic is deprecated. Use real market data for testing.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import streamlit as st
            st.warning(
                "Synthetic data testing is deprecated. Use real market data for testing."
            )
        except ImportError:
            pass
        return

    def fit(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, List[float]]:
        """Fit the TCN model to the data.

        Args:
            data: Training data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer

        Returns:
            Dictionary containing training history
        """
        try:
            # Prepare data
            X, y = self._prepare_data(data, is_training=True)

            # Setup optimizer and loss function
            params = list(self.tcn.parameters()) + list(self.linear.parameters())
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            criterion = nn.MSELoss()

            # Training history
            history = {"train_loss": []}

            # Training loop
            self.tcn.train()
            self.linear.train()
            for epoch in range(epochs):
                # Forward pass: TCN over sequence then linear on last timestep
                optimizer.zero_grad()
                features = self.tcn(X)            # (batch, channels, seq_len)
                last_step = features[:, :, -1]    # (batch, channels)
                outputs = self.linear(last_step)  # (batch, output_size)
                loss = criterion(outputs, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Record loss
                history["train_loss"].append(loss.item())

                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}"
                    )

            return history

        except Exception as e:
            logging.error(f"Error in TCN model fit: {e}")
            raise RuntimeError(f"TCN model fitting failed: {e}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted TCN model.

        Args:
            data: Input data DataFrame

        Returns:
            Numpy array of predictions
        """
        try:
            # Prepare data
            X, _ = self._prepare_data(data, is_training=False)

            # Make predictions
            self.tcn.eval()
            self.linear.eval()
            with torch.no_grad():
                features = self.tcn(X)            # (batch, channels, seq_len)
                last_step = features[:, :, -1]    # (batch, channels)
                predictions = self.linear(last_step)

            # Convert to numpy and denormalize
            predictions = predictions.cpu().numpy()
            predictions = predictions * self.y_std + self.y_mean

            return predictions.flatten()

        except Exception as e:
            logging.error(f"Error in TCN model predict: {e}")
            raise RuntimeError(f"TCN model prediction failed: {e}")

    def forecast(self, data: pd.DataFrame, horizon: int = 30, **kwargs) -> Dict[str, Any]:
        """Generate forecast: model predicts returns; build price path from last price. Returns already_denormalized=True."""
        try:
            data = self._normalize_columns(data.copy() if hasattr(data, "copy") else data)
            tc = self.config["target_column"]
            if tc not in data.columns:
                tc = "Close" if "Close" in data.columns else "close"
            current_ratio = float(data[tc].iloc[-1])
            forecast_values = []
            current_data = data.copy()

            for i in range(horizon):
                pred = self.predict(current_data)
                if len(pred) == 0:
                    break
                r = float(pred[-1])
                r = np.clip(r, -0.20, 0.20)
                next_ratio = current_ratio * (1.0 + r)
                forecast_values.append(next_ratio)
                if len(current_data) == 0:
                    break
                new_row = current_data.iloc[-1].copy()
                if tc in new_row.index:
                    new_row[tc] = next_ratio
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                if len(current_data) > len(data):
                    current_data = current_data.iloc[1:]
                current_ratio = next_ratio

            if len(forecast_values) == 0:
                # Fallback: flat forecast at last price (never return empty)
                forecast_values = [current_ratio] * int(horizon)

            forecast_arr = np.array(forecast_values, dtype="float64")
            payload = {
                "forecast": forecast_arr,
                "confidence": 0.8,
                "model": "TCN",
                "horizon": horizon,
                "already_denormalized": True,
            }
            # Compatibility with tests expecting {"result": {...}} wrapper
            return {"result": payload, **payload}

        except Exception as e:
            logging.error(f"Error in TCN model forecast: {e}")
            raise RuntimeError(f"TCN model forecasting failed: {e}")

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
            plt.title("TCN Model Predictions")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            logging.error(f"Error plotting TCN results: {e}")
            logger.error(f"Could not plot results: {e}")
