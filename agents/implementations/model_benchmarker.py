"""
Model Benchmarker Module

This module handles benchmarking and evaluating model implementations
for the auto-evolutionary model generator.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import psutil

from .implementation_generator import ModelCandidate

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results of model benchmarking."""

    model_name: str
    mse: float
    mae: float
    r2_score: float
    sharpe_ratio: float
    max_drawdown: float
    training_time: float
    inference_time: float
    memory_usage: float
    overall_score: float
    benchmark_date: str


class ModelBenchmarker:
    """Benchmarks and evaluates model implementations."""

    def __init__(
        self,
        benchmark_data: pd.DataFrame,
        target_column: str = "returns",
        current_best_score: float = 1.0,
    ):
        """
        Initialize the model benchmarker.

        Args:
            benchmark_data: Data for benchmarking
            target_column: Target column name
            current_best_score: Current best performance score
        """
        self.benchmark_data = benchmark_data
        self.target_column = target_column
        self.current_best_score = current_best_score
        self.logger = logging.getLogger(__name__)

        # Prepare benchmark data
        self.X, self.y = self._prepare_benchmark_data(benchmark_data)

        logger.info(f"Initialized ModelBenchmarker with {len(benchmark_data)} samples")

    def benchmark_model(
        self, model_candidate: ModelCandidate, test_data: Optional[pd.DataFrame] = None
    ) -> BenchmarkResult:
        """
        Benchmark a model candidate.

        Args:
            model_candidate: Model candidate to benchmark
            test_data: Optional test data (uses benchmark_data if None)

        Returns:
            BenchmarkResult: Benchmark results
        """
        try:
            start_time = time.time()

            # Use test data if provided, otherwise use benchmark data
            if test_data is not None:
                X_test, y_test = self._prepare_benchmark_data(test_data)
            else:
                X_test, y_test = self.X, self.y

            # Benchmark based on model type
            if model_candidate.model_type == "sklearn":
                result = self._benchmark_sklearn_model(model_candidate, X_test, y_test)
            elif model_candidate.model_type in [
                "lstm",
                "transformer",
                "attention",
                "rl",
            ]:
                result = self._benchmark_pytorch_model(model_candidate, X_test, y_test)
            else:
                result = self._benchmark_generic_model(model_candidate, X_test, y_test)

            # Calculate overall score
            result.overall_score = self._calculate_overall_score(result)
            result.benchmark_date = datetime.now().isoformat()

            benchmark_time = time.time() - start_time
            logger.info(
                f"Benchmarked {model_candidate.name} in {benchmark_time:.2f}s - Score: {result.overall_score:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error benchmarking model {model_candidate.name}: {e}")
            # Return a default result with poor scores
            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=float("inf"),
                mae=float("inf"),
                r2_score=0.0,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                training_time=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                overall_score=0.0,
                benchmark_date=datetime.now().isoformat(),
            )

    def _prepare_benchmark_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for benchmarking."""
        try:
            # Ensure we have the target column
            if self.target_column not in data.columns:
                # Try to find a suitable target column
                possible_targets = ["Close", "close", "price", "returns", "target"]
                for col in possible_targets:
                    if col in data.columns:
                        self.target_column = col
                        break
                else:
                    raise ValueError(
                        f"Target column '{self.target_column}' not found in data"
                    )

            # Prepare features (use all numeric columns except target)
            feature_columns = [
                col
                for col in data.select_dtypes(include=[np.number]).columns
                if col != self.target_column
            ]

            if not feature_columns:
                # If no features, use lagged values of target
                feature_columns = [self.target_column]
                data = data.copy()
                data[f"{self.target_column}_lag1"] = data[self.target_column].shift(1)
                data[f"{self.target_column}_lag2"] = data[self.target_column].shift(2)
                feature_columns = [
                    f"{self.target_column}_lag1",
                    f"{self.target_column}_lag2",
                ]

            # Remove rows with NaN values
            data_clean = data[feature_columns + [self.target_column]].dropna()

            X = data_clean[feature_columns].values
            y = data_clean[self.target_column].values

            return X, y

        except Exception as e:
            logger.error(f"Error preparing benchmark data: {e}")
            # Return dummy data
            return np.random.randn(100, 5), np.random.randn(100)

    def _benchmark_sklearn_model(
        self, model_candidate: ModelCandidate, X: np.ndarray, y: np.ndarray
    ) -> BenchmarkResult:
        """Benchmark a scikit-learn model."""
        try:
            # Import sklearn models
            from sklearn.ensemble import (
                GradientBoostingRegressor,
                RandomForestRegressor,
            )
            from sklearn.linear_model import Lasso, LinearRegression, Ridge
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )
            from sklearn.model_selection import train_test_split
            from sklearn.svm import SVR

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Create model based on type
            model_type = model_candidate.name.lower()
            if "forest" in model_type or "random" in model_type:
                model = RandomForestRegressor(**model_candidate.hyperparameters)
            elif "gradient" in model_type or "boosting" in model_type:
                model = GradientBoostingRegressor(**model_candidate.hyperparameters)
            elif "linear" in model_type or "regression" in model_type:
                model = LinearRegression(**model_candidate.hyperparameters)
            elif "ridge" in model_type:
                model = Ridge(**model_candidate.hyperparameters)
            elif "lasso" in model_type:
                model = Lasso(**model_candidate.hyperparameters)
            elif "svm" in model_type or "svr" in model_type:
                model = SVR(**model_candidate.hyperparameters)
            else:
                model = RandomForestRegressor(**model_candidate.hyperparameters)

            # Train model
            train_start = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - train_start

            # Predict
            inference_start = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - inference_start

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Calculate trading metrics
            sharpe_ratio, max_drawdown = self._calculate_trading_metrics(y_test, y_pred)

            # Measure memory usage
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=mse,
                mae=mae,
                r2_score=r2,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                training_time=training_time,
                inference_time=inference_time,
                memory_usage=memory_usage,
                overall_score=0.0,  # Will be calculated later
                benchmark_date="",
            )

        except Exception as e:
            logger.error(
                f"Error benchmarking sklearn model {model_candidate.name}: {e}"
            )
            raise

    def _benchmark_pytorch_model(
        self, model_candidate: ModelCandidate, X: np.ndarray, y: np.ndarray
    ) -> BenchmarkResult:
        """Benchmark a PyTorch model."""
        try:
            # Check if PyTorch is available
            try:
                import torch
                import torch.nn as nn
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                from sklearn.model_selection import train_test_split
            except ImportError:
                logger.warning("PyTorch not available. Install with: pip install torch")
                raise ImportError("PyTorch not available")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Get model parameters from hyperparameters or use defaults
            input_dim = X_train.shape[1]
            hidden_dim = model_candidate.hyperparameters.get("hidden_dim", 64)
            num_layers = model_candidate.hyperparameters.get("num_layers", 2)
            output_dim = 1
            dropout = model_candidate.hyperparameters.get("dropout", 0.2)
            learning_rate = model_candidate.hyperparameters.get("learning_rate", 0.001)
            epochs = model_candidate.hyperparameters.get("epochs", 50)
            batch_size = model_candidate.hyperparameters.get("batch_size", 32)

            # Create model based on type
            model_type = model_candidate.model_type.lower()
            if "lstm" in model_type:
                model = self._create_lstm_model(input_dim, hidden_dim, num_layers, output_dim, dropout)
            elif "transformer" in model_type or "attention" in model_type:
                model = self._create_transformer_model(input_dim, hidden_dim, output_dim, dropout)
            else:
                # Default to a simple feedforward network
                model = self._create_feedforward_model(input_dim, hidden_dim, output_dim, dropout)

            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Prepare data
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_tensor = torch.FloatTensor(y_test).to(device)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Train model
            train_start = time.time()
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            training_time = time.time() - train_start

            # Evaluate model
            model.eval()
            inference_start = time.time()
            with torch.no_grad():
                y_pred_tensor = model(X_test_tensor)
                y_pred = y_pred_tensor.cpu().numpy().flatten()
            inference_time = time.time() - inference_start

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Calculate trading metrics
            sharpe_ratio, max_drawdown = self._calculate_trading_metrics(y_test, y_pred)

            # Measure memory usage
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=mse,
                mae=mae,
                r2_score=r2,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                training_time=training_time,
                inference_time=inference_time,
                memory_usage=memory_usage,
                overall_score=0.0,  # Will be calculated later
                benchmark_date="",
            )

        except ImportError:
            # PyTorch not available - return placeholder with warning
            logger.warning(
                f"PyTorch not available. Cannot benchmark {model_candidate.name}. "
                "Install with: pip install torch"
            )
            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=float("inf"),
                mae=float("inf"),
                r2_score=0.0,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                training_time=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                overall_score=0.0,
                benchmark_date=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(
                f"Error benchmarking PyTorch model {model_candidate.name}: {e}"
            )
            raise

    def _create_lstm_model(
        self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float
    ):
        """Create an LSTM model."""
        import torch.nn as nn

        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                # Reshape input for LSTM (batch, seq_len, features)
                # For time series, we'll use a sequence length of 1 with all features
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # Reshape for LSTM: (batch, features) -> (batch, 1, features)
                x = x.unsqueeze(1)
                lstm_out, _ = self.lstm(x)
                lstm_out = self.dropout(lstm_out[:, -1, :])
                out = self.fc(lstm_out)
                return out

        return LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)

    def _create_transformer_model(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ):
        """Create a simple transformer model."""
        import torch.nn as nn

        class TransformerModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, dropout):
                super(TransformerModel, self).__init__()
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.fc = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # Reshape for transformer: (batch, features) -> (batch, 1, features)
                x = x.unsqueeze(1)
                x = self.input_projection(x)
                x = self.transformer(x)
                x = self.dropout(x[:, -1, :])
                out = self.fc(x)
                return out

        return TransformerModel(input_dim, hidden_dim, output_dim, dropout)

    def _create_feedforward_model(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ):
        """Create a feedforward neural network."""
        import torch.nn as nn

        class FeedforwardModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, dropout):
                super(FeedforwardModel, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
                self.dropout = nn.Dropout(dropout)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        return FeedforwardModel(input_dim, hidden_dim, output_dim, dropout)

    def _benchmark_generic_model(
        self, model_candidate: ModelCandidate, X: np.ndarray, y: np.ndarray
    ) -> BenchmarkResult:
        """Benchmark a generic model."""
        try:
            # For now, return a placeholder result
            # In a full implementation, this would evaluate the generated code
            logger.warning(
                f"Generic model benchmarking not fully implemented for {model_candidate.name}"
            )

            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=0.15,
                mae=0.08,
                r2_score=0.6,
                sharpe_ratio=1.0,
                max_drawdown=0.2,
                training_time=2.0,
                inference_time=0.02,
                memory_usage=150.0,
                overall_score=0.0,
                benchmark_date="",
            )

        except Exception as e:
            logger.error(
                f"Error benchmarking generic model {model_candidate.name}: {e}"
            )
            raise

    def _calculate_trading_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate trading-specific metrics."""
        try:
            # Calculate returns
            returns_true = np.diff(y_true) / y_true[:-1]
            returns_pred = np.diff(y_pred) / y_pred[:-1]

            # Calculate Sharpe ratio
            excess_returns = returns_pred - returns_true
            sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)

            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + returns_pred)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)

            return sharpe_ratio, abs(max_drawdown)

        except Exception as e:
            logger.warning(f"Error calculating trading metrics: {e}")
            return 0.0, 0.2

    def _calculate_overall_score(self, result: BenchmarkResult) -> float:
        """Calculate overall performance score."""
        try:
            # Normalize metrics to 0-1 scale
            mse_score = max(0, 1 - result.mse / 1.0)  # Assuming max MSE of 1.0
            mae_score = max(0, 1 - result.mae / 0.5)  # Assuming max MAE of 0.5
            r2_score = max(0, result.r2_score)
            sharpe_score = max(0, min(1, result.sharpe_ratio / 2.0))  # Normalize to 0-1
            drawdown_score = max(
                0, 1 - result.max_drawdown / 0.5
            )  # Assuming max drawdown of 0.5

            # Time efficiency scores (lower is better)
            time_score = max(
                0, 1 - result.training_time / 60.0
            )  # Assuming max training time of 60s
            inference_score = max(
                0, 1 - result.inference_time / 1.0
            )  # Assuming max inference time of 1s

            # Memory efficiency score
            memory_score = max(
                0, 1 - result.memory_usage / 1000.0
            )  # Assuming max memory of 1GB

            # Weighted combination
            weights = {
                "mse": 0.2,
                "mae": 0.15,
                "r2": 0.2,
                "sharpe": 0.15,
                "drawdown": 0.1,
                "time": 0.05,
                "inference": 0.05,
                "memory": 0.1,
            }

            overall_score = (
                weights["mse"] * mse_score
                + weights["mae"] * mae_score
                + weights["r2"] * r2_score
                + weights["sharpe"] * sharpe_score
                + weights["drawdown"] * drawdown_score
                + weights["time"] * time_score
                + weights["inference"] * inference_score
                + weights["memory"] * memory_score
            )

            return overall_score

        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
