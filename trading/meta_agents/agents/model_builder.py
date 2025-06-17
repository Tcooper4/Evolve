"""
Model Builder Agent

This agent is responsible for:
1. Training and evaluating multiple forecasting models
2. Automatically selecting the best performing model
3. Generating forecasts and performance metrics
4. Managing model configurations and exports

The agent supports multiple model types and can combine them into hybrid models
for improved forecasting performance.
"""

import os
import json
import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import yaml
from pathlib import Path
import asyncio
import warnings
from enum import Enum
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import arch

# Local imports
from trading.agents.task_memory import Task, TaskMemory, TaskStatus

# Optional imports with guards
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Prophet models will be disabled.")

try:
    from sklearn.linear_model import Ridge
    RIDGE_AVAILABLE = True
except ImportError:
    RIDGE_AVAILABLE = False
    warnings.warn("Ridge not available. Ridge models will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    mse: float
    sharpe_ratio: float
    max_drawdown: float
    training_time: float
    forecast_steps: int
    timestamp: str

@dataclass
class ModelOutput:
    """Container for model output"""
    forecast: np.ndarray
    metrics: ModelMetrics
    model_name: str
    model_params: Dict
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None

class ModelBuilder:
    def __init__(self, config_path: str = "config/model_builder_config.json"):
        """
        Initialize the ModelBuilder.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.task_memory = TaskMemory()
        
        # Initialize model registry
        self.model_registry = {
            "ARIMA": self.run_arima,
            "LSTM": self.run_lstm,
            "XGBoost": self.run_xgboost,
        }
        
        # Add optional models if available
        if PROPHET_AVAILABLE:
            self.model_registry["Prophet"] = self.run_prophet
        if RIDGE_AVAILABLE:
            self.model_registry["Ridge"] = self.run_ridge
            
        # Add GARCH and Hybrid models
        self.model_registry.update({
            "GARCH": self.run_garch,
            "Hybrid": self.run_hybrid
        })
        
        # Initialize model storage
        self.model_dir = Path(self.config.get("model_dir", "models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = []
        
        logger.info("Initialized ModelBuilder")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}

    async def run_arima(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run ARIMA model.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="arima_training",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes="Training ARIMA model"
        )
        self.task_memory.add_task(task)

        try:
            start_time = datetime.now()
            
            # Prepare data
            series = data['value'].values
            train_size = int(len(series) * 0.8)
            train, test = series[:train_size], series[train_size:]
            
            # Fit ARIMA model
            model = ARIMA(train, order=(5,1,0))
            model_fit = model.fit()
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Calculate metrics
            mse = mean_squared_error(test, model_fit.predict(start=len(train), end=len(series)-1))
            returns = np.diff(forecast) / forecast[:-1]
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
            drawdown = np.min(np.cumsum(returns))
            
            metrics = ModelMetrics(
                mse=mse,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                training_time=(datetime.now() - start_time).total_seconds(),
                forecast_steps=forecast_steps,
                timestamp=datetime.now().isoformat()
            )
            
            output = ModelOutput(
                forecast=forecast,
                metrics=metrics,
                model_name="ARIMA",
                model_params={"order": (5,1,0)}
            )

            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully trained ARIMA model with MSE: {mse:.4f}",
                metadata={
                    'mse': mse,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': drawdown,
                    'training_time': metrics.training_time
                }
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in ARIMA model: {e}")
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to train ARIMA model: {str(e)}"
            )
            raise

    async def run_lstm(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run LSTM model.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="lstm_training",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes="Training LSTM model"
        )
        self.task_memory.add_task(task)

        try:
            start_time = datetime.now()
            
            # Prepare data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[['value']])
            
            # Create sequences
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:(i + seq_length)])
                    y.append(data[i + seq_length])
                return np.array(X), np.array(y)
            
            seq_length = self.config.get("lstm_seq_length", 10)
            X, y = create_sequences(scaled_data, seq_length)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(
                X_train, y_train,
                epochs=self.config.get("lstm_epochs", 50),
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            last_sequence = scaled_data[-seq_length:]
            forecast = []
            
            for _ in range(forecast_steps):
                next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
                forecast.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_pred
            
            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            
            # Calculate metrics
            mse = mean_squared_error(y_test, model.predict(X_test))
            returns = np.diff(forecast.flatten()) / forecast[:-1].flatten()
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
            drawdown = np.min(np.cumsum(returns))
            
            metrics = ModelMetrics(
                mse=mse,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                training_time=(datetime.now() - start_time).total_seconds(),
                forecast_steps=forecast_steps,
                timestamp=datetime.now().isoformat()
            )
            
            output = ModelOutput(
                forecast=forecast.flatten(),
                metrics=metrics,
                model_name="LSTM",
                model_params={
                    "seq_length": seq_length,
                    "epochs": self.config.get("lstm_epochs", 50)
                }
            )

            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully trained LSTM model with MSE: {mse:.4f}",
                metadata={
                    'mse': mse,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': drawdown,
                    'training_time': metrics.training_time
                }
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in LSTM model: {e}")
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to train LSTM model: {str(e)}"
            )
            raise

    async def run_xgboost(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run XGBoost model.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="xgboost_training",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes="Training XGBoost model"
        )
        self.task_memory.add_task(task)

        try:
            start_time = datetime.now()
            
            # Create features
            def create_features(df):
                df = df.copy()
                for i in range(1, 6):
                    df[f'lag_{i}'] = df['value'].shift(i)
                df['rolling_mean'] = df['value'].rolling(window=5).mean()
                df['rolling_std'] = df['value'].rolling(window=5).std()
                return df.dropna()
            
            # Prepare data
            df = create_features(data)
            X = df.drop(['value'], axis=1)
            y = df['value']
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
            model.fit(X_train, y_train)
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            last_data = df.iloc[-1:].copy()
            forecast = []
            
            for _ in range(forecast_steps):
                pred = model.predict(last_data.drop(['value'], axis=1))
                forecast.append(pred[0])
                
                # Update features for next prediction
                last_data['value'] = pred[0]
                last_data = create_features(last_data)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, model.predict(X_test))
            returns = np.diff(forecast) / forecast[:-1]
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
            drawdown = np.min(np.cumsum(returns))
            
            metrics = ModelMetrics(
                mse=mse,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                training_time=(datetime.now() - start_time).total_seconds(),
                forecast_steps=forecast_steps,
                timestamp=datetime.now().isoformat()
            )
            
            output = ModelOutput(
                forecast=np.array(forecast),
                metrics=metrics,
                model_name="XGBoost",
                model_params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5
                }
            )

            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully trained XGBoost model with MSE: {mse:.4f}",
                metadata={
                    'mse': mse,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': drawdown,
                    'training_time': metrics.training_time
                }
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in XGBoost model: {e}")
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to train XGBoost model: {str(e)}"
            )
            raise

    async def run_prophet(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run Prophet model.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available")
            
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="prophet_training",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes="Training Prophet model"
        )
        self.task_memory.add_task(task)

        try:
            start_time = datetime.now()
            
            # Prepare data
            df = data.reset_index()
            df.columns = ['ds', 'y']
            
            # Split data
            train_size = int(len(df) * 0.8)
            train_df = df[:train_size]
            test_df = df[train_size:]
            
            # Train model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            model.fit(train_df)
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            future = model.make_future_dataframe(periods=forecast_steps)
            forecast = model.predict(future)
            
            # Calculate metrics
            mse = mean_squared_error(test_df['y'], forecast['yhat'][train_size:])
            returns = np.diff(forecast['yhat'][-forecast_steps:]) / forecast['yhat'][-forecast_steps:-1]
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
            drawdown = np.min(np.cumsum(returns))
            
            metrics = ModelMetrics(
                mse=mse,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                training_time=(datetime.now() - start_time).total_seconds(),
                forecast_steps=forecast_steps,
                timestamp=datetime.now().isoformat()
            )
            
            output = ModelOutput(
                forecast=forecast['yhat'][-forecast_steps:].values,
                metrics=metrics,
                model_name="Prophet",
                model_params={
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": True
                }
            )

            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully trained Prophet model with MSE: {mse:.4f}",
                metadata={
                    'mse': mse,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': drawdown,
                    'training_time': metrics.training_time
                }
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in Prophet model: {e}")
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to train Prophet model: {str(e)}"
            )
            raise

    async def run_garch(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run GARCH model.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="garch_training",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes="Training GARCH model"
        )
        self.task_memory.add_task(task)

        try:
            start_time = datetime.now()
            
            # Prepare data
            returns = np.diff(data['value']) / data['value'][:-1]
            
            # Split data
            train_size = int(len(returns) * 0.8)
            train_returns = returns[:train_size]
            test_returns = returns[train_size:]
            
            # Train model
            model = arch.arch_model(
                train_returns,
                vol='GARCH',
                p=1,
                q=1
            )
            model_fit = model.fit(disp='off')
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            forecast = model_fit.forecast(horizon=forecast_steps)
            
            # Calculate metrics
            mse = mean_squared_error(test_returns, model_fit.conditional_volatility[train_size:])
            sharpe = np.sqrt(252) * np.mean(forecast.mean['h.1']) / np.std(forecast.mean['h.1'])
            drawdown = np.min(np.cumsum(forecast.mean['h.1']))
            
            metrics = ModelMetrics(
                mse=mse,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                training_time=(datetime.now() - start_time).total_seconds(),
                forecast_steps=forecast_steps,
                timestamp=datetime.now().isoformat()
            )
            
            output = ModelOutput(
                forecast=forecast.mean['h.1'].values,
                metrics=metrics,
                model_name="GARCH",
                model_params={"p": 1, "q": 1}
            )

            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully trained GARCH model with MSE: {mse:.4f}",
                metadata={
                    'mse': mse,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': drawdown,
                    'training_time': metrics.training_time
                }
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in GARCH model: {e}")
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to train GARCH model: {str(e)}"
            )
            raise

    async def run_ridge(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run Ridge model.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        if not RIDGE_AVAILABLE:
            raise ImportError("Ridge is not available")
            
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="ridge_training",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes="Training Ridge model"
        )
        self.task_memory.add_task(task)

        try:
            start_time = datetime.now()
            
            # Create features
            def create_features(df, n_lags=5):
                df = df.copy()
                for i in range(1, n_lags + 1):
                    df[f'lag_{i}'] = df['value'].shift(i)
                df['rolling_mean'] = df['value'].rolling(window=5).mean()
                df['rolling_std'] = df['value'].rolling(window=5).std()
                return df.dropna()
            
            # Prepare data
            df = create_features(data)
            X = df.drop(['value'], axis=1)
            y = df['value']
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train model
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            last_data = df.iloc[-1:].copy()
            forecast = []
            
            for _ in range(forecast_steps):
                pred = model.predict(last_data.drop(['value'], axis=1))
                forecast.append(pred[0])
                
                # Update features for next prediction
                last_data['value'] = pred[0]
                last_data = create_features(last_data)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, model.predict(X_test))
            returns = np.diff(forecast) / forecast[:-1]
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
            drawdown = np.min(np.cumsum(returns))
            
            metrics = ModelMetrics(
                mse=mse,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                training_time=(datetime.now() - start_time).total_seconds(),
                forecast_steps=forecast_steps,
                timestamp=datetime.now().isoformat()
            )
            
            output = ModelOutput(
                forecast=np.array(forecast),
                metrics=metrics,
                model_name="Ridge",
                model_params={"alpha": 1.0}
            )

            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully trained Ridge model with MSE: {mse:.4f}",
                metadata={
                    'mse': mse,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': drawdown,
                    'training_time': metrics.training_time
                }
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in Ridge model: {e}")
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to train Ridge model: {str(e)}"
            )
            raise

    async def run_hybrid(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run Hybrid model combining multiple models.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="hybrid_training",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes="Training Hybrid model"
        )
        self.task_memory.add_task(task)

        try:
            start_time = datetime.now()
            
            # Train individual models
            model_outputs = []
            for model_name, model_func in self.model_registry.items():
                if model_name != "Hybrid":
                    try:
                        output = await model_func(data)
                        model_outputs.append(output)
                    except Exception as e:
                        logger.warning(f"Model {model_name} failed: {e}")
            
            if not model_outputs:
                raise ValueError("No models successfully trained")
            
            # Combine forecasts using weighted average
            weights = np.array([1/output.metrics.mse for output in model_outputs])
            weights = weights / weights.sum()
            
            combined_forecast = np.zeros_like(model_outputs[0].forecast)
            for output, weight in zip(model_outputs, weights):
                combined_forecast += weight * output.forecast
            
            # Calculate combined metrics
            mse = np.average([output.metrics.mse for output in model_outputs], weights=weights)
            sharpe = np.average([output.metrics.sharpe_ratio for output in model_outputs], weights=weights)
            drawdown = np.average([output.metrics.max_drawdown for output in model_outputs], weights=weights)
            
            metrics = ModelMetrics(
                mse=mse,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                training_time=(datetime.now() - start_time).total_seconds(),
                forecast_steps=len(combined_forecast),
                timestamp=datetime.now().isoformat()
            )
            
            output = ModelOutput(
                forecast=combined_forecast,
                metrics=metrics,
                model_name="Hybrid",
                model_params={
                    "weights": weights.tolist(),
                    "models": [output.model_name for output in model_outputs]
                }
            )

            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully trained Hybrid model with MSE: {mse:.4f}",
                metadata={
                    'mse': mse,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': drawdown,
                    'training_time': metrics.training_time,
                    'models': [output.model_name for output in model_outputs]
                }
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in Hybrid model: {e}")
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to train Hybrid model: {str(e)}"
            )
            raise

    def _create_dummy_output(self, model_name: str, error: Exception) -> ModelOutput:
        """
        Create a dummy output for failed models.
        
        Args:
            model_name: Name of the failed model
            error: The error that occurred
            
        Returns:
            ModelOutput with dummy values and high MSE
        """
        forecast_steps = self.config.get("forecast_steps", 30)
        dummy_forecast = np.zeros(forecast_steps)
        
        metrics = ModelMetrics(
            mse=float('inf'),
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            training_time=0.0,
            forecast_steps=forecast_steps,
            timestamp=datetime.now().isoformat()
        )
        
        return ModelOutput(
            forecast=dummy_forecast,
            metrics=metrics,
            model_name=model_name,
            model_params={"error": str(error)}
        )

    async def select_best_model(self, data: pd.DataFrame) -> ModelOutput:
        """
        Select the best performing model from all available models.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput from the best performing model
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="model_selection",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes="Selecting best performing model"
        )
        self.task_memory.add_task(task)

        try:
            results = []
            for model_name, model_func in self.model_registry.items():
                try:
                    output = await model_func(data)
                    results.append(output)
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    results.append(self._create_dummy_output(model_name, e))
            
            # Select best model based on MSE
            best_output = min(results, key=lambda x: x.metrics.mse)
            
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Selected {best_output.model_name} as best model with MSE: {best_output.metrics.mse:.4f}",
                metadata={
                    'selected_model': best_output.model_name,
                    'mse': best_output.metrics.mse,
                    'sharpe_ratio': best_output.metrics.sharpe_ratio,
                    'max_drawdown': best_output.metrics.max_drawdown
                }
            )
            
            return best_output
            
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to select best model: {str(e)}"
            )
            raise

    def export_model_config(self, output: ModelOutput) -> Dict:
        """
        Export model configuration and results.
        
        Args:
            output: ModelOutput to export
            
        Returns:
            Dictionary containing model configuration
        """
        try:
            config = {
                "model_name": output.model_name,
                "model_params": output.model_params,
                "metrics": {
                    "mse": output.metrics.mse,
                    "sharpe_ratio": output.metrics.sharpe_ratio,
                    "max_drawdown": output.metrics.max_drawdown,
                    "training_time": output.metrics.training_time,
                    "forecast_steps": output.metrics.forecast_steps
                },
                "timestamp": output.metrics.timestamp,
                "forecast": output.forecast.tolist()
            }
            
            if output.confidence_intervals is not None:
                config["confidence_intervals"] = {
                    "lower": output.confidence_intervals[0].tolist(),
                    "upper": output.confidence_intervals[1].tolist()
                }
            
            # Save to file
            config_path = self.model_dir / f"{output.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return config
            
        except Exception as e:
            logger.error(f"Error exporting model config: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create sample data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        values = np.random.normal(0, 1, len(dates)).cumsum() + 100
        data = pd.DataFrame({'Close': values}, index=dates)
        
        # Initialize model builder
        builder = ModelBuilder()
        
        # Select best model
        best_model = await builder.select_best_model(data)
        
        # Export configuration
        config = builder.export_model_config(best_model)
        print(f"Best model: {config['model_name']}")
        print(f"MSE: {config['metrics']['mse']:.4f}")
        print(f"Sharpe Ratio: {config['metrics']['sharpe_ratio']:.4f}")

    asyncio.run(main()) 