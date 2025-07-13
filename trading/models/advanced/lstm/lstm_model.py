"""LSTM model for time series forecasting."""

# Standard library imports
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from trading.models.base_model import BaseModel, ModelRegistry, ValidationError


@ModelRegistry.register('LSTM')
class LSTMForecaster(BaseModel):
    """LSTM model for time series forecasting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LSTM forecaster.

        Args:
            config: Configuration dictionary containing:
                - input_size: Size of input features (default: 2)
                - hidden_size: Size of hidden layers (default: 64)
                - num_layers: Number of LSTM layers (default: 2)
                - dropout: Dropout rate (default: 0.2)
                - bidirectional: Whether to use bidirectional LSTM (default: False)
                - sequence_length: Length of input sequences (default: 10)
                - feature_columns: List of feature column names (default: ['close', 'volume'])
                - target_column: Name of target column (default: 'close')
                - learning_rate: Learning rate (default: 0.001)
                - use_lr_scheduler: Whether to use learning rate scheduler (default: True)
        """
        if config is None:
            config = {}

        # Set default configuration
        default_config = {
            'input_size': 2,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': False,
            'sequence_length': 10,
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'learning_rate': 0.001,
            'use_lr_scheduler': True
        }
        default_config.update(config)

        super().__init__(default_config)
        self._validate_config()
        self._setup_model()

        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _validate_config(self) -> None:
        """Validate model configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate required parameters
        required_params = ['input_size', 'hidden_size', 'num_layers',
                         'dropout', 'sequence_length', 'feature_columns',
                         'target_column']
        for param in required_params:
            if param not in self.config:
                raise ValidationError(f"Missing required parameter: {param}")

        # Validate parameter values
        if self.config['input_size'] <= 0:
            raise ValidationError("input_size must be positive")
        if self.config['hidden_size'] <= 0:
            raise ValidationError("hidden_size must be positive")
        if self.config['num_layers'] <= 0:
            raise ValidationError("num_layers must be positive")
        if not 0 <= self.config['dropout'] <= 1:
            raise ValidationError("dropout must be between 0 and 1")
        if self.config['sequence_length'] < 2:
            raise ValidationError("sequence_length must be at least 2")
        if not self.config['feature_columns']:
            raise ValidationError("feature_columns cannot be empty")
        if len(self.config['feature_columns']) != self.config['input_size']:
            raise ValidationError(f"Number of feature columns ({len(self.config['feature_columns'])}) "
                                f"must match input_size ({self.config['input_size']})")

    def build_model(self) -> nn.Module:
        """Build the LSTM model architecture.

        Returns:
            PyTorch model
        """
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'] if self.config['num_layers'] > 1 else 0,
            bidirectional=self.config['bidirectional'],
            batch_first=True
        )

        # Calculate output size based on bidirectional flag
        output_size = self.config['hidden_size'] * 2 if self.config['bidirectional'] else self.config['hidden_size']

        # Output layer
        self.fc = nn.Linear(output_size, 1)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        return {'success': True, 'result': nn.ModuleList([self.lstm, self.fc]), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_hidden = lstm_out[:, -1]

        # Output layer
        out = self.fc(last_hidden)
        return out

    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training or prediction.

        Args:
            data: Input data as pandas DataFrame
            is_training: Whether data is for training

        Returns:
            Tuple of (X, y) tensors where:
                X: Shape (batch_size, sequence_length, input_size)
                y: Shape (batch_size, 1)
        """
        # Validate data
        if data.isnull().any().any():
            raise ValidationError("Data contains missing values")

        # Check if all required columns exist
        missing_cols = [col for col in self.config['feature_columns']
                       if col not in data.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")

        # Convert to numpy arrays
        X = data[self.config['feature_columns']].values
        y = data[self.config['target_column']].values[self.config['sequence_length']:]

        # Create sequences
        X_sequences = []
        for i in range(len(X) - self.config['sequence_length']):
            X_sequences.append(X[i:i + self.config['sequence_length']])
        X = np.array(X_sequences)

        # Normalize
        if is_training:
            self.X_mean = X.mean(axis=(0, 1))
            self.X_std = X.std(axis=(0, 1))
            self.y_mean = y.mean()
            self.y_std = y.std()

        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(-1)

        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)

        return X, y

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the LSTM model.

        Args:
            data: Input data as pandas DataFrame

        Returns:
            Predicted values as numpy array
        """
        try:
            # Prepare data
            X, _ = self._prepare_data(data, is_training=False)

            # Set model to evaluation mode
            self.eval()

            # Make predictions
            with torch.no_grad():
                predictions = self(X)

            # Convert to numpy and denormalize
            predictions = predictions.cpu().numpy()
            predictions = predictions * self.y_std + self.y_mean

            return predictions.flatten()

        except Exception as e:
            import logging
            logging.error(f"Error in LSTM model predict: {e}")
            raise RuntimeError(f"LSTM model prediction failed: {e}")

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
                forecast_values.append(pred[-1])

                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row[self.config.get('target_column', 'close')] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row

            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.85,  # LSTM confidence
                'model': 'LSTM',
                'horizon': horizon,
                'feature_columns': self.config.get('feature_columns', []),
                'target_column': self.config.get('target_column', 'close')
            }

        except Exception as e:
            import logging
            logging.error(f"Error in LSTM model forecast: {e}")
            raise RuntimeError(f"LSTM model forecasting failed: {e}")

    def summary(self) -> Dict[str, Any]:
        """Get model summary information.

        Returns:
            Dictionary containing model summary
        """
        return super().summary()

    def infer(self) -> Dict[str, Any]:
        """Run model inference.

        Returns:
            Dictionary containing inference results
        """
        return super().infer()

    def shap_interpret(self, X_sample) -> Dict[str, Any]:
        """Run SHAP interpretability on a sample batch.

        Args:
            X_sample: Sample input data for SHAP analysis

        Returns:
            Dictionary containing SHAP analysis results
        """
        try:
            import shap
        except ImportError:
            return {
                'success': False,
                'error': 'SHAP is not installed. Please install it with pip install shap.',
                'shap_values': None
            }

        try:
            explainer = shap.DeepExplainer(self.model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            shap.summary_plot(shap_values, X_sample.cpu().numpy())
            return {
                'success': True,
                'shap_values': shap_values,
                'explainer_type': 'DeepExplainer',
                'sample_shape': X_sample.shape
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'shap_values': None
            }

    def test_synthetic(self) -> Dict[str, Any]:
        """Test model on synthetic data.

        Returns:
            Dictionary containing test results
        """
        try:
            import numpy as np
            import pandas as pd
            n = 100
            df = pd.DataFrame({
                'close': np.sin(np.linspace(0, 10, n)),
                'volume': np.random.rand(n)
            })
            self.fit(df.iloc[:80], df.iloc[80:])
            y_pred = self.predict(df.iloc[80:])
            mse = ((y_pred.flatten() - df['close'].iloc[80:].values) ** 2).mean()
            return {
                'success': True,
                'mse': mse,
                'test_size': len(df.iloc[80:]),
                'synthetic_data_shape': df.shape
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'mse': None
            }
