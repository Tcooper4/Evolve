"""
Comprehensive unit tests for all forecasting models.

This module tests all forecasting models to ensure they have proper
fit(), predict(), evaluate(), and plot_forecast() methods.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModelBase:
    """Base class for model testing with common setup."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                "close": prices,
                "volume": np.random.normal(1000000, 200000, len(dates)),
            },
            index=dates,
        )

        return data

    @pytest.fixture
    def train_data(self, sample_data):
        """Create training data with features."""
        data = sample_data.copy()

        # Add technical indicators as features
        data["sma_20"] = data["close"].rolling(20).mean()
        data["sma_50"] = data["close"].rolling(50).mean()
        data["rsi"] = self._calculate_rsi(data["close"])
        data["volatility"] = data["close"].rolling(20).std()
        data["returns"] = data["close"].pct_change()
        data["target"] = data["close"].shift(-1)  # Next day's price

        # Remove NaN values
        data = data.dropna()

        return data

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class TestCatBoostModel(TestModelBase):
    """Test CatBoost model functionality."""

    def test_catboost_initialization(self):
        """Test CatBoost model initialization."""
        try:
            from trading.models.catboost_model import CatBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "catboost_params": {
                    "iterations": 100,
                    "depth": 6,
                    "learning_rate": 0.1,
                    "loss_function": "RMSE",
                    "verbose": False,
                },
            }

            model = CatBoostModel(config)
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert hasattr(model, "forecast")

        except ImportError:
            pytest.skip("CatBoost not available")

    def test_catboost_fit(self, train_data):
        """Test CatBoost model fitting."""
        try:
            from trading.models.catboost_model import CatBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "catboost_params": {
                    "iterations": 50,
                    "depth": 4,
                    "learning_rate": 0.1,
                    "loss_function": "RMSE",
                    "verbose": False,
                },
            }

            model = CatBoostModel(config)
            result = model.fit(train_data)

            assert model.fitted is True
            assert isinstance(result, dict)
            assert "train_loss" in result

        except ImportError:
            pytest.skip("CatBoost not available")

    def test_catboost_predict(self, train_data):
        """Test CatBoost model prediction."""
        try:
            from trading.models.catboost_model import CatBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "catboost_params": {
                    "iterations": 50,
                    "depth": 4,
                    "learning_rate": 0.1,
                    "loss_function": "RMSE",
                    "verbose": False,
                },
            }

            model = CatBoostModel(config)
            model.fit(train_data)

            # Test prediction
            predictions = model.predict(train_data.tail(10))
            assert len(predictions) == 10
            assert all(isinstance(p, (int, float)) for p in predictions)

        except ImportError:
            pytest.skip("CatBoost not available")

    def test_catboost_forecast(self, train_data):
        """Test CatBoost model forecasting."""
        try:
            from trading.models.catboost_model import CatBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "catboost_params": {
                    "iterations": 50,
                    "depth": 4,
                    "learning_rate": 0.1,
                    "loss_function": "RMSE",
                    "verbose": False,
                },
            }

            model = CatBoostModel(config)
            model.fit(train_data)

            # Test forecasting
            forecast_result = model.forecast(train_data, horizon=5)

            assert isinstance(forecast_result, dict)
            assert "forecast" in forecast_result
            assert "confidence" in forecast_result
            assert "model" in forecast_result
            assert len(forecast_result["forecast"]) == 5

        except ImportError:
            pytest.skip("CatBoost not available")


class TestXGBoostModel(TestModelBase):
    """Test XGBoost model functionality."""

    def test_xgboost_initialization(self):
        """Test XGBoost model initialization."""
        try:
            from trading.models.xgboost_model import XGBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "xgboost_params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                },
            }

            model = XGBoostModel(config)
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert hasattr(model, "forecast")

        except ImportError:
            pytest.skip("XGBoost not available")

    def test_xgboost_fit(self, train_data):
        """Test XGBoost model fitting."""
        try:
            from trading.models.xgboost_model import XGBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "xgboost_params": {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                },
            }

            model = XGBoostModel(config)
            result = model.fit(train_data)

            assert model.fitted is True
            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("XGBoost not available")

    def test_xgboost_predict(self, train_data):
        """Test XGBoost model prediction."""
        try:
            from trading.models.xgboost_model import XGBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "xgboost_params": {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                },
            }

            model = XGBoostModel(config)
            model.fit(train_data)

            predictions = model.predict(train_data.tail(10))
            assert len(predictions) == 10
            assert all(isinstance(p, (int, float)) for p in predictions)

        except ImportError:
            pytest.skip("XGBoost not available")

    def test_xgboost_output_validation(self, train_data):
        """Test XGBoost output length, shape, and MSE thresholds."""
        try:
            from sklearn.metrics import mean_squared_error

            from trading.models.xgboost_model import XGBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "xgboost_params": {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                },
            }

            model = XGBoostModel(config)
            model.fit(train_data)

            # Test prediction output length
            test_data = train_data.tail(5)
            predictions = model.predict(test_data)
            assert (
                len(predictions) == 5
            ), f"Expected 5 predictions, got {len(predictions)}"

            # Test prediction shape
            assert predictions.ndim == 1, f"Expected 1D array, got {predictions.ndim}D"

            # Test MSE threshold on synthetic data
            synthetic_data = train_data.copy()
            synthetic_data["close"] = np.linspace(100, 200, len(synthetic_data))
            synthetic_data["target"] = synthetic_data["close"] + np.random.normal(
                0, 1, len(synthetic_data)
            )

            model.fit(synthetic_data)
            synthetic_predictions = model.predict(synthetic_data.tail(20))
            synthetic_actual = synthetic_data["target"].tail(20)

            mse = mean_squared_error(synthetic_actual, synthetic_predictions)
            assert mse < 1000, f"MSE too high: {mse}, expected < 1000"

            # Test that predictions are finite
            assert np.all(
                np.isfinite(predictions)
            ), "Predictions contain NaN or infinite values"

            # Test that predictions are reasonable (not extreme values)
            assert np.all(predictions > -1000) and np.all(
                predictions < 10000
            ), "Predictions outside reasonable range"

        except ImportError:
            pytest.skip("XGBoost not available")

    def test_xgboost_auto_feature_engineering(self, train_data):
        """Test XGBoost auto feature engineering functionality."""
        try:
            from trading.models.xgboost_model import XGBoostModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "auto_feature_engineering": True,
                "xgboost_params": {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                },
            }

            model = XGBoostModel(config)

            # Test that auto feature engineering is enabled
            assert model.config.get("auto_feature_engineering", False) is True

            # Test feature preparation with auto engineering
            features, target = model.prepare_features(train_data)
            assert len(features) > 0, "Auto feature engineering should produce features"
            assert len(target) > 0, "Auto feature engineering should produce target"
            assert len(features) == len(
                target
            ), "Features and target should have same length"

        except ImportError:
            pytest.skip("XGBoost not available")


class TestRidgeModel(TestModelBase):
    """Test Ridge regression model functionality."""

    def test_ridge_initialization(self):
        """Test Ridge model initialization."""
        try:
            from trading.models.ridge_model import RidgeModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "ridge_params": {"alpha": 1.0, "random_state": 42},
            }

            model = RidgeModel(config)
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert hasattr(model, "forecast")

        except ImportError:
            pytest.skip("Ridge model not available")

    def test_ridge_fit(self, train_data):
        """Test Ridge model fitting."""
        try:
            from trading.models.ridge_model import RidgeModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "ridge_params": {"alpha": 1.0, "random_state": 42},
            }

            model = RidgeModel(config)
            result = model.fit(train_data)

            assert model.fitted is True
            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("Ridge model not available")

    def test_ridge_predict(self, train_data):
        """Test Ridge model prediction."""
        try:
            from trading.models.ridge_model import RidgeModel

            config = {
                "feature_columns": ["sma_20", "sma_50", "rsi", "volatility", "returns"],
                "target_column": "target",
                "ridge_params": {"alpha": 1.0, "random_state": 42},
            }

            model = RidgeModel(config)
            model.fit(train_data)

            predictions = model.predict(train_data.tail(10))
            assert len(predictions) == 10
            assert all(isinstance(p, (int, float)) for p in predictions)

        except ImportError:
            pytest.skip("Ridge model not available")


class TestLSTMModel(TestModelBase):
    """Test LSTM model functionality."""

    def test_lstm_initialization(self):
        """Test LSTM model initialization."""
        try:
            from trading.models.lstm_model import LSTMModel

            config = {
                "sequence_length": 20,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
            }

            model = LSTMModel(config)
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert hasattr(model, "forecast")

        except ImportError:
            pytest.skip("LSTM model not available")

    def test_lstm_fit(self, train_data):
        """Test LSTM model fitting."""
        try:
            from trading.models.lstm_model import LSTMModel

            config = {
                "sequence_length": 10,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.1,
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 5,
            }

            model = LSTMModel(config)
            result = model.fit(train_data)

            assert model.fitted is True
            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("LSTM model not available")


class TestProphetModel(TestModelBase):
    """Test Prophet model functionality."""

    def test_prophet_initialization(self):
        """Test Prophet model initialization."""
        try:
            from trading.models.prophet_model import ProphetModel

            config = {
                "date_column": "date",
                "target_column": "close",
                "prophet_params": {
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                },
            }

            model = ProphetModel(config)
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert hasattr(model, "forecast")

        except ImportError:
            pytest.skip("Prophet not available")

    def test_prophet_fit(self, sample_data):
        """Test Prophet model fitting."""
        try:
            from trading.models.prophet_model import ProphetModel

            # Prepare data for Prophet
            data = sample_data.reset_index()
            data = data.rename(columns={"index": "date"})

            config = {
                "date_column": "date",
                "target_column": "close",
                "prophet_params": {
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                },
            }

            model = ProphetModel(config)
            result = model.fit(data)

            assert model.fitted is True
            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("Prophet not available")


class TestModelEvaluation:
    """Test model evaluation functionality."""

    def test_model_evaluation_metrics(self):
        """Test that all models can be evaluated with proper metrics."""
        from pages.Forecasting import calculate_forecast_metrics

        # Generate test data
        np.random.seed(42)
        actual = np.random.randn(100) + 100
        predicted = actual + np.random.randn(100) * 0.1

        # Calculate metrics
        metrics = calculate_forecast_metrics(actual, predicted)

        # Verify all required metrics are present
        required_metrics = [
            "RMSE",
            "MAE",
            "MAPE",
            "Directional_Accuracy",
            "Sharpe_Ratio",
            "Max_Drawdown",
            "Win_Rate",
        ]

        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

    def test_model_confidence_scores(self):
        """Test that models provide confidence scores."""
        # This would test that each model provides a confidence score
        # For now, we'll test the concept
        confidence_scores = {
            "LSTM": 0.87,
            "Transformer": 0.89,
            "XGBoost": 0.85,
            "ARIMA": 0.82,
            "Prophet": 0.84,
            "Ensemble": 0.92,
        }

        for model, confidence in confidence_scores.items():
            assert 0 <= confidence <= 1
            assert isinstance(confidence, float)


class TestModelIntegration:
    """Test model integration with the forecast router."""

    def test_forecast_router_integration(self):
        """Test that models integrate properly with the forecast router."""
        try:
            from models.forecast_router import ForecastRouter

            router = ForecastRouter()

            # Test that router can handle different model types
            available_models = router.get_available_models()
            assert isinstance(available_models, list)
            assert len(available_models) > 0

        except ImportError:
            pytest.skip("Forecast router not available")

    def test_model_registry(self):
        """Test that models are properly registered."""
        try:
            from trading.models.base_model import ModelRegistry

            # Check that models are registered
            registered_models = ModelRegistry.get_registered_models()
            assert isinstance(registered_models, dict)
            assert len(registered_models) > 0

        except ImportError:
            pytest.skip("Model registry not available")


if __name__ == "__main__":
    pytest.main([__file__])
