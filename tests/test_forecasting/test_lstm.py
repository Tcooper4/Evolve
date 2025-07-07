"""Tests for the LSTM forecasting model."""

import pytest
import pandas as pd
import numpy as np
from trading.models.lstm_model import LSTMModel

class TestLSTMModel:
    @pytest.fixture
    def model(self):
        """Create an LSTM model instance for testing."""
        return LSTMModel(
            sequence_length=10,
            n_units=50,
            dropout_rate=0.2,
            learning_rate=0.001
        )

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_model_initialization(self, model):
        """Test that model initializes with correct parameters."""
        assert model.sequence_length == 10
        assert model.n_units == 50
        assert model.dropout_rate == 0.2
        assert model.learning_rate == 0.001
        assert model.name == 'LSTM'

    def test_data_preprocessing(self, model, sample_data):
        """Test that data is preprocessed correctly."""
        X, y = model.preprocess_data(sample_data['close'])
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == model.sequence_length
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_model_building(self, model):
        """Test that model architecture is built correctly."""
        model.build_model()
        
        assert hasattr(model, 'model')
        assert model.model is not None
        assert model.model.count_params() > 0

    def test_model_training(self, model, sample_data):
        """Test that model trains correctly."""
        model.fit(sample_data['close'], epochs=2, batch_size=32)
        
        assert model.is_fitted
        assert hasattr(model, 'history')
        assert 'loss' in model.history.history

    def test_forecast_generation(self, model, sample_data):
        """Test that forecasts are generated correctly."""
        model.fit(sample_data['close'], epochs=2, batch_size=32)
        forecast = model.forecast(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 5
        assert not forecast.isnull().any()

    def test_model_evaluation(self, model, sample_data):
        """Test that model evaluation metrics are calculated correctly."""
        model.fit(sample_data['close'], epochs=2, batch_size=32)
        metrics = model.evaluate(sample_data['close'])
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_parameter_validation(self):
        """Test that model parameters are validated."""
        with pytest.raises(ValueError):
            LSTMModel(sequence_length=0)  # Invalid sequence length
        with pytest.raises(ValueError):
            LSTMModel(n_units=0)  # Invalid number of units
        with pytest.raises(ValueError):
            LSTMModel(dropout_rate=1.5)  # Invalid dropout rate
        with pytest.raises(ValueError):
            LSTMModel(learning_rate=0)  # Invalid learning rate

    def test_empty_data_handling(self, model):
        """Test that model handles empty data correctly."""
        empty_data = pd.Series([])
        with pytest.raises(ValueError):
            model.fit(empty_data)

    def test_missing_data_handling(self, model):
        """Test that model handles missing data correctly."""
        data = pd.Series([100, np.nan, 101, 102])
        with pytest.raises(ValueError):
            model.fit(data)

    def test_forecast_horizon_validation(self, model, sample_data):
        """Test that forecast horizon is validated."""
        model.fit(sample_data['close'], epochs=2, batch_size=32)
        with pytest.raises(ValueError):
            model.forecast(steps=0)  # Invalid forecast horizon

    def test_model_persistence(self, model, sample_data, tmp_path):
        """Test that model can be saved and loaded."""
        # Fit model
        model.fit(sample_data['close'], epochs=2, batch_size=32)
        
        # Save model
        model_path = tmp_path / "lstm_model"
        model.save(model_path)
        
        # Load model
        loaded_model = LSTMModel.load(model_path)
        
        # Verify loaded model
        assert loaded_model.sequence_length == model.sequence_length
        assert loaded_model.n_units == model.n_units
        assert loaded_model.dropout_rate == model.dropout_rate
        assert loaded_model.learning_rate == model.learning_rate
        assert loaded_model.is_fitted

    def test_sequence_generation(self, model, sample_data):
        """Test that sequences are generated correctly."""
        X, y = model.preprocess_data(sample_data['close'])
        
        # Check sequence structure
        for i in range(len(X)):
            assert len(X[i]) == model.sequence_length
            assert y[i] == sample_data['close'].iloc[i + model.sequence_length]

    def test_training_history(self, model, sample_data):
        """Test that training history is recorded correctly."""
        history = model.fit(sample_data['close'], epochs=2, batch_size=32)
        
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert len(history.history['loss']) == 2
        assert len(history.history['val_loss']) == 2

    def test_early_stopping(self, model, sample_data):
        """Test that early stopping works correctly."""
        model.fit(
            sample_data['close'],
            epochs=10,
            batch_size=32,
            early_stopping=True,
            patience=2
        )
        
        assert model.is_fitted
        assert hasattr(model, 'history')
        assert 'val_loss' in model.history.history

    def test_learning_rate_scheduling(self, model, sample_data):
        """Test that learning rate scheduling works correctly."""
        model.fit(
            sample_data['close'],
            epochs=2,
            batch_size=32,
            learning_rate_schedule=True
        )
        
        assert model.is_fitted
        assert hasattr(model, 'history')
        assert 'lr' in model.history.history 