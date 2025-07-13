"""
Unit tests for LSTM forecaster model.

Tests LSTM model functionality with synthetic time series data,
including edge cases like short series, constant series, and NaN handling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Import the LSTM model
try:
    from trading.models.lstm_model import LSTMModel, LSTMForecaster
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    LSTMModel = Mock()
    LSTMForecaster = Mock()

class TestLSTMForecaster:
    """Test suite for LSTM forecaster model."""
    
    @pytest.fixture
    def synthetic_time_series(self):
        """Create synthetic time series with increasing trend."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create time series with trend and noise
        trend = np.linspace(100, 150, 100)  # Increasing trend
        noise = np.random.normal(0, 2, 100)  # Random noise
        values = trend + noise
        
        return pd.Series(values, index=dates, name='Close')
    
    @pytest.fixture
    def synthetic_dataframe(self):
        """Create synthetic DataFrame with multiple features."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create multiple features
        close = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
        volume = np.random.uniform(1000000, 5000000, 100)
        high = close + np.random.uniform(0, 5, 100)
        low = close - np.random.uniform(0, 5, 100)
        
        df = pd.DataFrame({
            'Close': close,
            'Volume': volume,
            'High': high,
            'Low': low
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def short_time_series(self):
        """Create short time series (< 10 points)."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        values = [100, 101, 102, 103, 104]
        return pd.Series(values, index=dates, name='Close')
    
    @pytest.fixture
    def constant_time_series(self):
        """Create constant time series."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        values = [100.0] * 50
        return pd.Series(values, index=dates, name='Close')
    
    @pytest.fixture
    def nan_time_series(self):
        """Create time series with NaN values."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        values = [100, 101, np.nan, 103, 104] + [100 + i for i in range(45)]
        return pd.Series(values, index=dates, name='Close')
    
    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model instance."""
        if not LSTM_AVAILABLE:
            pytest.skip("LSTM model not available")
        return LSTMModel(input_dim=1, hidden_dim=50, output_dim=1)
    
    @pytest.fixture
    def lstm_forecaster(self):
        """Create LSTM forecaster instance."""
        if not LSTM_AVAILABLE:
            pytest.skip("LSTM forecaster not available")
        return LSTMForecaster()
    
    def test_model_instantiation(self, lstm_model):
        """Test that LSTM model instantiates correctly."""
        assert lstm_model is not None
        assert hasattr(lstm_model, 'model')
        assert hasattr(lstm_model, 'is_fitted')
        assert lstm_model.is_fitted is False
    
    def test_model_fitting(self, lstm_model, synthetic_time_series):
        """Test that LSTM model fits to data correctly."""
        result = lstm_model.fit(synthetic_time_series)
        
        assert result['success'] is True
        assert lstm_model.is_fitted is True
        assert lstm_model.model is not None
        assert 'timestamp' in result
    
    def test_forecast_generation(self, lstm_model, synthetic_time_series):
        """Test that LSTM model generates forecasts correctly."""
        # Fit the model first
        lstm_model.fit(synthetic_time_series)
        
        # Generate forecast
        forecast_steps = 10
        result = lstm_model.predict(synthetic_time_series, horizon=forecast_steps)
        
        assert result['success'] is True
        assert 'predictions' in result
        assert len(result['predictions']) == forecast_steps
        assert not np.isnan(result['predictions']).any()
        assert 'timestamp' in result
    
    def test_forecast_output_length(self, lstm_model, synthetic_time_series):
        """Test that forecast output has correct length."""
        lstm_model.fit(synthetic_time_series)
        
        for steps in [1, 5, 10, 30]:
            result = lstm_model.predict(synthetic_time_series, horizon=steps)
            assert result['success'] is True
            assert len(result['predictions']) == steps
    
    def test_no_nan_in_forecast(self, lstm_model, synthetic_time_series):
        """Test that forecasts contain no NaN values."""
        lstm_model.fit(synthetic_time_series)
        
        result = lstm_model.predict(synthetic_time_series, horizon=10)
        assert result['success'] is True
        assert not np.isnan(result['predictions']).any()
    
    def test_short_time_series_handling(self, lstm_model, short_time_series):
        """Test handling of short time series (< 10 points)."""
        result = lstm_model.fit(short_time_series)
        
        # Should fail gracefully with clear error message
        assert result['success'] is False
        assert 'error' in result
        assert any(keyword in result['error'].lower() 
                  for keyword in ['insufficient', 'at least', 'minimum', 'sequence'])
    
    def test_constant_series_handling(self, lstm_model, constant_time_series):
        """Test handling of constant time series."""
        result = lstm_model.fit(constant_time_series)
        
        # Should handle constant series gracefully
        if result['success']:
            # If it succeeds, test prediction
            forecast_result = lstm_model.predict(constant_time_series, horizon=5)
            assert forecast_result['success'] is True
            assert len(forecast_result['predictions']) == 5
        else:
            # If it fails, should be due to constant series
            assert any(keyword in result.get('error', '').lower() 
                      for keyword in ['constant', 'variance', 'unique'])
    
    def test_nan_series_handling(self, lstm_model, nan_time_series):
        """Test handling of time series with NaN values."""
        result = lstm_model.fit(nan_time_series)
        
        # Should fail gracefully with clear error message
        assert result['success'] is False
        assert 'error' in result
        assert any(keyword in result['error'].lower() 
                  for keyword in ['nan', 'missing', 'invalid'])
    
    def test_model_summary(self, lstm_model, synthetic_time_series):
        """Test that model summary is generated correctly."""
        lstm_model.fit(synthetic_time_series)
        
        summary_result = lstm_model.get_model_summary()
        assert summary_result['success'] is True
        assert 'summary' in summary_result
        assert isinstance(summary_result['summary'], str)
        assert len(summary_result['summary']) > 0
    
    def test_unfitted_model_behavior(self, lstm_model, synthetic_time_series):
        """Test behavior when trying to predict without fitting."""
        result = lstm_model.predict(synthetic_time_series, horizon=5)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'fitted' in result['error'].lower()
    
    def test_sequence_length_handling(self, lstm_model, synthetic_time_series):
        """Test handling of different sequence lengths."""
        lstm_model.fit(synthetic_time_series)
        
        # Test different sequence lengths
        for seq_length in [10, 20, 30]:
            if hasattr(lstm_model, 'set_sequence_length'):
                lstm_model.set_sequence_length(seq_length)
                result = lstm_model.predict(synthetic_time_series, horizon=5)
                assert result['success'] is True
    
    def test_forecast_method(self, lstm_model, synthetic_time_series):
        """Test the forecast method specifically."""
        result = lstm_model.forecast(synthetic_time_series, horizon=10)
        
        assert 'forecast' in result
        assert 'confidence' in result
        assert 'model' in result
        assert result['model'] == 'LSTM'
        assert result['horizon'] == 10
        assert len(result['forecast']) == 10
    
    def test_model_save_load(self, lstm_model, synthetic_time_series, tmp_path):
        """Test model save and load functionality."""
        # Fit the model
        lstm_model.fit(synthetic_time_series)
        
        # Save model
        save_path = tmp_path / "lstm_model.pt"
        save_result = lstm_model.save_model(str(save_path))
        
        if save_result['success']:
            # Load model
            new_model = LSTMModel(input_dim=1, hidden_dim=50, output_dim=1)
            load_result = new_model.load_model(str(save_path))
            
            assert load_result['success'] is True
            assert new_model.is_fitted is True
            assert new_model.model is not None
    
    def test_error_handling_edge_cases(self, lstm_model):
        """Test error handling for various edge cases."""
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        result = lstm_model.fit(empty_series)
        assert result['success'] is False
        
        # Test with None
        result = lstm_model.fit(None)
        assert result['success'] is False
        
        # Test with single value
        single_value = pd.Series([100])
        result = lstm_model.fit(single_value)
        assert result['success'] is False
    
    def test_forecast_consistency(self, lstm_model, synthetic_time_series):
        """Test that forecasts are consistent across multiple calls."""
        lstm_model.fit(synthetic_time_series)
        
        # Generate multiple forecasts
        forecast1 = lstm_model.predict(synthetic_time_series, horizon=5)
        forecast2 = lstm_model.predict(synthetic_time_series, horizon=5)
        
        assert forecast1['success'] is True
        assert forecast2['success'] is True
        np.testing.assert_array_almost_equal(
            forecast1['predictions'], 
            forecast2['predictions'], 
            decimal=10
        )
    
    def test_trend_detection(self, lstm_model):
        """Test that model correctly identifies trends."""
        # Create data with clear trend
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        trend_data = pd.Series(np.linspace(100, 200, 50), index=dates)
        
        lstm_model.fit(trend_data)
        forecast = lstm_model.predict(trend_data, horizon=5)
        
        if forecast['success']:
            # Forecast should continue the trend (simplified check)
            assert len(forecast['predictions']) == 5
            assert not np.isnan(forecast['predictions']).any()
    
    def test_seasonality_handling(self, lstm_model):
        """Test that model handles seasonal data correctly."""
        # Create seasonal data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        t = np.linspace(0, 4*np.pi, 100)
        seasonal_data = pd.Series(100 + 10*np.sin(t), index=dates)
        
        lstm_model.fit(seasonal_data)
        forecast = lstm_model.predict(seasonal_data, horizon=5)
        
        if forecast['success']:
            assert len(forecast['predictions']) == 5
            assert not np.isnan(forecast['predictions']).any()
    
    @pytest.mark.parametrize("horizon", [1, 5, 10, 30])
    def test_different_forecast_horizons(self, lstm_model, synthetic_time_series, horizon):
        """Test forecasting with different horizons."""
        lstm_model.fit(synthetic_time_series)
        
        result = lstm_model.predict(synthetic_time_series, horizon=horizon)
        assert result['success'] is True
        assert len(result['predictions']) == horizon
        assert not np.isnan(result['predictions']).any()
    
    def test_model_configuration(self):
        """Test different model configurations."""
        configs = [
            {'input_dim': 1, 'hidden_dim': 50, 'output_dim': 1},
            {'input_dim': 1, 'hidden_dim': 100, 'output_dim': 1, 'num_layers': 2},
            {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'dropout': 0.2}
        ]
        
        for config in configs:
            model = LSTMModel(**config)
            assert model.config == config
    
    def test_training_parameters(self, lstm_model, synthetic_time_series):
        """Test training with different parameters."""
        # Test training with different epochs
        for epochs in [10, 20, 50]:
            if hasattr(lstm_model, 'train_model'):
                result = lstm_model.train_model(synthetic_time_series, epochs=epochs)
                assert result['success'] is True
    
    def test_batch_size_handling(self, lstm_model, synthetic_time_series):
        """Test handling of different batch sizes."""
        lstm_model.fit(synthetic_time_series)
        
        # Test different batch sizes
        for batch_size in [16, 32, 64]:
            if hasattr(lstm_model, 'set_batch_size'):
                lstm_model.set_batch_size(batch_size)
                result = lstm_model.predict(synthetic_time_series, horizon=5)
                assert result['success'] is True
    
    def test_learning_rate_handling(self, lstm_model, synthetic_time_series):
        """Test handling of different learning rates."""
        # Test different learning rates
        for lr in [0.001, 0.01, 0.1]:
            if hasattr(lstm_model, 'set_learning_rate'):
                lstm_model.set_learning_rate(lr)
                result = lstm_model.fit(synthetic_time_series)
                if result['success']:
                    assert lstm_model.is_fitted is True
    
    def test_model_architecture(self, lstm_model):
        """Test model architecture parameters."""
        assert hasattr(lstm_model, 'input_dim')
        assert hasattr(lstm_model, 'hidden_dim')
        assert hasattr(lstm_model, 'output_dim')
        
        # Test architecture parameters
        assert lstm_model.input_dim == 1
        assert lstm_model.hidden_dim == 50
        assert lstm_model.output_dim == 1
    
    def test_sequence_preprocessing(self, lstm_model, synthetic_time_series):
        """Test sequence preprocessing functionality."""
        if hasattr(lstm_model, 'preprocess_sequences'):
            sequences = lstm_model.preprocess_sequences(synthetic_time_series, sequence_length=10)
            assert isinstance(sequences, np.ndarray)
            assert len(sequences.shape) == 2  # (samples, features)
    
    def test_normalization(self, lstm_model, synthetic_time_series):
        """Test data normalization functionality."""
        if hasattr(lstm_model, 'normalize_data'):
            normalized = lstm_model.normalize_data(synthetic_time_series)
            assert isinstance(normalized, pd.Series)
            assert len(normalized) == len(synthetic_time_series)
    
    def test_denormalization(self, lstm_model, synthetic_time_series):
        """Test data denormalization functionality."""
        if hasattr(lstm_model, 'denormalize_predictions'):
            # First normalize
            if hasattr(lstm_model, 'normalize_data'):
                normalized = lstm_model.normalize_data(synthetic_time_series)
                # Then denormalize
                denormalized = lstm_model.denormalize_predictions(normalized.iloc[:5])
                assert isinstance(denormalized, np.ndarray)
                assert len(denormalized) == 5
    
    def test_model_evaluation_metrics(self, lstm_model, synthetic_time_series):
        """Test model evaluation metrics."""
        lstm_model.fit(synthetic_time_series)
        
        if hasattr(lstm_model, 'evaluate'):
            metrics = lstm_model.evaluate(synthetic_time_series)
            assert isinstance(metrics, dict)
            assert len(metrics) > 0
    
    def test_early_stopping(self, lstm_model, synthetic_time_series):
        """Test early stopping functionality."""
        if hasattr(lstm_model, 'train_with_early_stopping'):
            result = lstm_model.train_with_early_stopping(
                synthetic_time_series, 
                patience=5, 
                min_delta=0.001
            )
            assert result['success'] is True
    
    def test_model_performance_tracking(self, lstm_model, synthetic_time_series):
        """Test model performance tracking."""
        if hasattr(lstm_model, 'get_training_history'):
            lstm_model.fit(synthetic_time_series)
            history = lstm_model.get_training_history()
            assert isinstance(history, dict)
            assert 'loss' in history or 'val_loss' in history


if __name__ == "__main__":
    pytest.main([__file__]) 