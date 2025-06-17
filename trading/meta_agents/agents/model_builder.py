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
            
            return ModelOutput(
                forecast=forecast,
                metrics=metrics,
                model_name="ARIMA",
                model_params={"order": (5,1,0)}
            )
            
        except Exception as e:
            logger.error(f"Error in ARIMA model: {e}")
            raise

    async def run_lstm(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run LSTM model.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
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
            
            return ModelOutput(
                forecast=forecast.flatten(),
                metrics=metrics,
                model_name="LSTM",
                model_params={
                    "seq_length": seq_length,
                    "epochs": self.config.get("lstm_epochs", 50)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in LSTM model: {e}")
            raise

    async def run_xgboost(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run XGBoost model.
        
        Args:
            data: Input time series data
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        try:
            start_time = datetime.now()
            
            # Prepare data
            def create_features(df):
                df = df.copy()
                df['lag_1'] = df['value'].shift(1)
                df['lag_2'] = df['value'].shift(2)
                df['lag_3'] = df['value'].shift(3)
                df['rolling_mean'] = df['value'].rolling(window=7).mean()
                df['rolling_std'] = df['value'].rolling(window=7).std()
                return df
            
            data = create_features(data)
            data = data.dropna()
            
            # Split data
            train_size = int(len(data) * 0.8)
            train = data[:train_size]
            test = data[train_size:]
            
            # Prepare features
            feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'rolling_std']
            X_train = train[feature_cols]
            y_train = train['value']
            X_test = test[feature_cols]
            y_test = test['value']
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
            model.fit(X_train, y_train)
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            forecast = []
            last_data = data.iloc[-1:].copy()
            
            for _ in range(forecast_steps):
                # Update features
                last_data = create_features(last_data)
                X_pred = last_data[feature_cols]
                
                # Make prediction
                pred = model.predict(X_pred)[0]
                forecast.append(pred)
                
                # Update last_data for next iteration
                last_data['value'] = pred
            
            forecast = np.array(forecast)
            
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
            
            return ModelOutput(
                forecast=forecast,
                metrics=metrics,
                model_name="XGBoost",
                model_params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5
                }
            )
            
        except Exception as e:
            logger.error(f"Error in XGBoost model: {e}")
            raise

    async def run_prophet(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run Prophet model for time series forecasting.
        
        Args:
            data: DataFrame with 'Close' column and datetime index
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available")
            
        try:
            start_time = datetime.now()
            
            # Prepare data for Prophet
            df = data.reset_index()
            df.columns = ['ds', 'y']
            
            # Split data
            train_size = int(len(df) * 0.8)
            train = df[:train_size]
            test = df[train_size:]
            
            # Configure and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            # Add custom seasonality if needed
            if self.config.get("prophet_custom_seasonality"):
                model.add_seasonality(
                    name='custom',
                    period=365.25,
                    fourier_order=5
                )
            
            model.fit(train)
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            future = model.make_future_dataframe(periods=forecast_steps)
            forecast = model.predict(future)
            
            # Calculate metrics
            mse = mean_squared_error(test['y'], forecast['yhat'][train_size:])
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
            
            return ModelOutput(
                forecast=forecast['yhat'][-forecast_steps:].values,
                metrics=metrics,
                model_name="Prophet",
                model_params={
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": True,
                    "changepoint_prior_scale": 0.05,
                    "seasonality_prior_scale": 10.0
                },
                confidence_intervals=(
                    forecast['yhat_lower'][-forecast_steps:].values,
                    forecast['yhat_upper'][-forecast_steps:].values
                )
            )
            
        except Exception as e:
            logger.error(f"Error in Prophet model: {e}")
            # Return dummy forecast with high MSE
            return self._create_dummy_output("Prophet", e)

    async def run_garch(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run GARCH model for volatility forecasting.
        
        Args:
            data: DataFrame with 'Close' column and datetime index
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        try:
            start_time = datetime.now()
            
            # Calculate returns
            returns = np.diff(data['Close']) / data['Close'][:-1]
            
            # Split data
            train_size = int(len(returns) * 0.8)
            train = returns[:train_size]
            test = returns[train_size:]
            
            # Fit GARCH(1,1) model
            model = arch.arch_model(
                train,
                vol='GARCH',
                p=1,
                q=1,
                mean='Zero',
                dist='normal'
            )
            model_fit = model.fit(disp='off')
            
            # Generate volatility forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            vol_forecast = model_fit.forecast(horizon=forecast_steps)
            
            # Generate synthetic price forecast
            last_price = data['Close'].iloc[-1]
            price_forecast = [last_price]
            
            for i in range(forecast_steps):
                # Generate random return based on forecasted volatility
                vol = vol_forecast.variance['h.1'].iloc[i]
                ret = np.random.normal(0, np.sqrt(vol))
                next_price = price_forecast[-1] * (1 + ret)
                price_forecast.append(next_price)
            
            price_forecast = np.array(price_forecast[1:])  # Remove initial price
            
            # Calculate metrics
            mse = mean_squared_error(test, model_fit.forecast(horizon=len(test)).mean['h.1'])
            returns = np.diff(price_forecast) / price_forecast[:-1]
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
            
            return ModelOutput(
                forecast=price_forecast,
                metrics=metrics,
                model_name="GARCH",
                model_params={
                    "p": 1,
                    "q": 1,
                    "mean": "Zero",
                    "dist": "normal"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in GARCH model: {e}")
            return self._create_dummy_output("GARCH", e)

    async def run_ridge(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run Ridge regression model for time series forecasting.
        
        Args:
            data: DataFrame with 'Close' column and datetime index
            
        Returns:
            ModelOutput containing forecast and metrics
        """
        if not RIDGE_AVAILABLE:
            raise ImportError("Ridge is not available")
            
        try:
            start_time = datetime.now()
            
            # Create lag features
            def create_features(df, n_lags=5):
                df = df.copy()
                for i in range(1, n_lags + 1):
                    df[f'lag_{i}'] = df['Close'].shift(i)
                return df
            
            # Prepare data
            data = create_features(data)
            data = data.dropna()
            
            # Split data
            train_size = int(len(data) * 0.8)
            train = data[:train_size]
            test = data[train_size:]
            
            # Prepare features
            feature_cols = [f'lag_{i}' for i in range(1, 6)]
            X_train = train[feature_cols]
            y_train = train['Close']
            X_test = test[feature_cols]
            y_test = test['Close']
            
            # Train Ridge model
            model = Ridge(
                alpha=1.0,
                fit_intercept=True,
                normalize=True
            )
            model.fit(X_train, y_train)
            
            # Generate forecast
            forecast_steps = self.config.get("forecast_steps", 30)
            forecast = []
            last_data = data.iloc[-1:].copy()
            
            for _ in range(forecast_steps):
                # Update features
                last_data = create_features(last_data)
                X_pred = last_data[feature_cols]
                
                # Make prediction
                pred = model.predict(X_pred)[0]
                forecast.append(pred)
                
                # Update last_data for next iteration
                last_data['Close'] = pred
            
            forecast = np.array(forecast)
            
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
            
            return ModelOutput(
                forecast=forecast,
                metrics=metrics,
                model_name="Ridge",
                model_params={
                    "alpha": 1.0,
                    "fit_intercept": True,
                    "normalize": True,
                    "n_lags": 5
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Ridge model: {e}")
            return self._create_dummy_output("Ridge", e)

    async def run_hybrid(self, data: pd.DataFrame) -> ModelOutput:
        """
        Run hybrid model combining multiple models.
        
        Args:
            data: DataFrame with 'Close' column and datetime index
            
        Returns:
            ModelOutput containing combined forecast and metrics
        """
        try:
            start_time = datetime.now()
            
            # Run all available models
            model_outputs = []
            for model_name, model_func in self.model_registry.items():
                if model_name != "Hybrid":
                    try:
                        output = await model_func(data)
                        model_outputs.append(output)
                    except Exception as e:
                        logger.warning(f"Error running {model_name}: {e}")
            
            if not model_outputs:
                raise ValueError("No models successfully ran")
            
            # Calculate weights based on MSE
            mses = [output.metrics.mse for output in model_outputs]
            weights = 1 / np.array(mses)
            weights = weights / np.sum(weights)
            
            # Combine forecasts
            forecast_steps = self.config.get("forecast_steps", 30)
            combined_forecast = np.zeros(forecast_steps)
            
            for output, weight in zip(model_outputs, weights):
                combined_forecast += output.forecast * weight
            
            # Calculate combined metrics
            returns = np.diff(combined_forecast) / combined_forecast[:-1]
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
            drawdown = np.min(np.cumsum(returns))
            
            metrics = ModelMetrics(
                mse=np.mean(mses),
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                training_time=(datetime.now() - start_time).total_seconds(),
                forecast_steps=forecast_steps,
                timestamp=datetime.now().isoformat()
            )
            
            return ModelOutput(
                forecast=combined_forecast,
                metrics=metrics,
                model_name="Hybrid",
                model_params={
                    "weights": dict(zip(
                        [output.model_name for output in model_outputs],
                        weights.tolist()
                    ))
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Hybrid model: {e}")
            return self._create_dummy_output("Hybrid", e)

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
        Select the best performing model.
        
        Args:
            data: DataFrame with 'Close' column and datetime index
            
        Returns:
            ModelOutput from the best performing model
        """
        try:
            best_mse = float('inf')
            best_output = None
            
            for model_name, model_func in self.model_registry.items():
                try:
                    output = await model_func(data)
                    if output.metrics.mse < best_mse:
                        best_mse = output.metrics.mse
                        best_output = output
                except Exception as e:
                    logger.warning(f"Error running {model_name}: {e}")
            
            if best_output is None:
                raise ValueError("No models successfully ran")
            
            return best_output
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
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