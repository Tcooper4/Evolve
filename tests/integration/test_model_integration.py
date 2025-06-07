import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.lstm_model import LSTMForecaster
from trading.models.tcn_model import TCNModel
from pathlib import Path
import tempfile
from datetime import datetime, timedelta
from trading.data.preprocessing import DataPreprocessor, FeatureEngineering

class TestModelIntegration:
    """Integration tests for model interactions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Open': np.random.normal(100, 2, len(dates)),
            'High': np.random.normal(102, 2, len(dates)),
            'Low': np.random.normal(98, 2, len(dates)),
            'Close': np.random.normal(100, 2, len(dates)),
            'Volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1) + 1
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1) - 1
        
        return data
    
    @pytest.fixture
    def preprocessor(self):
        """Create data preprocessor with default settings."""
        return DataPreprocessor()
    
    @pytest.fixture
    def feature_engineering(self):
        """Create feature engineering with default settings."""
        return FeatureEngineering()
    
    @pytest.fixture
    def lstm_config(self):
        """LSTM model configuration."""
        return {
            'input_size': 2,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'sequence_length': 10,
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'learning_rate': 0.001,
            'use_lr_scheduler': True,
            'scheduler_patience': 1,
            'scheduler_factor': 0.1
        }
    
    @pytest.fixture
    def tcn_config(self):
        """TCN model configuration."""
        return {
            'input_size': 2,
            'output_size': 1,
            'num_channels': [64, 128, 256],
            'kernel_size': 3,
            'dropout': 0.2,
            'sequence_length': 10,
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'learning_rate': 0.001,
            'use_lr_scheduler': True
        }
    
    @pytest.fixture
    def tcn_model(self):
        """Create TCN model with default settings."""
        return TCNModel(config={
            'input_size': 20,  # Number of features
            'output_size': 1,
            'num_channels': [64, 32, 16],
            'kernel_size': 3,
            'dropout': 0.2,
            'sequence_length': 10,
            'feature_columns': [
                'Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower',
                'Volume_MA_5', 'Volume_MA_10', 'Fourier_Sin_7', 'Fourier_Cos_7',
                'Close_Lag_1', 'Close_Return_Lag_1', 'Volume_Lag_1',
                'Volume_Return_Lag_1', 'HL_Range_Lag_1', 'ROC_5', 'Momentum_5',
                'Stoch_K', 'Stoch_D', 'Volume_Trend'
            ],
            'target_column': 'Close'
        })
    
    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model with default settings."""
        return LSTMForecaster(config={
            'input_size': 20,  # Number of features
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 10,
            'feature_columns': [
                'Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower',
                'Volume_MA_5', 'Volume_MA_10', 'Fourier_Sin_7', 'Fourier_Cos_7',
                'Close_Lag_1', 'Close_Return_Lag_1', 'Volume_Lag_1',
                'Volume_Return_Lag_1', 'HL_Range_Lag_1', 'ROC_5', 'Momentum_5',
                'Stoch_K', 'Stoch_D', 'Volume_Trend'
            ],
            'target_column': 'Close'
        })
    
    def test_preprocessing_feature_engineering_integration(self, preprocessor, feature_engineering, sample_data):
        """Test integration between preprocessing and feature engineering."""
        # Preprocess data
        preprocessed_data = preprocessor.preprocess_data(sample_data)
        
        # Engineer features
        features = feature_engineering.engineer_features(preprocessed_data)
        
        # Verify data integrity
        assert isinstance(features, pd.DataFrame)
        assert not features.isna().any().any()
        assert len(features) == len(sample_data)
        
        # Verify feature engineering worked on preprocessed data
        assert 'RSI' in features.columns
        assert 'MACD' in features.columns
        assert 'BB_Upper' in features.columns
        assert 'BB_Lower' in features.columns
        
        # Verify preprocessing effects are preserved
        assert features['Close'].std() < 1  # Data should be normalized
        assert not features['Close'].isna().any()  # No missing values
    
    def test_feature_engineering_model_integration(self, feature_engineering, tcn_model, lstm_model, sample_data):
        """Test integration between feature engineering and models."""
        # Engineer features
        features = feature_engineering.engineer_features(sample_data)
        
        # Prepare data for models
        tcn_data = tcn_model._prepare_data(features)
        lstm_data = lstm_model._prepare_data(features)
        
        # Verify data shapes
        assert len(tcn_data['X']) == len(lstm_data['X'])
        assert len(tcn_data['y']) == len(lstm_data['y'])
        
        # Verify feature alignment
        assert tcn_data['X'].shape[2] == len(tcn_model.config['feature_columns'])
        assert lstm_data['X'].shape[2] == len(lstm_model.config['feature_columns'])
        
        # Verify sequence handling
        assert tcn_data['X'].shape[1] == tcn_model.config['sequence_length']
        assert lstm_data['X'].shape[1] == lstm_model.config['sequence_length']
    
    def test_full_pipeline_integration(self, preprocessor, feature_engineering, tcn_model, lstm_model, sample_data):
        """Test full pipeline integration from preprocessing to model prediction."""
        # Preprocess data
        preprocessed_data = preprocessor.preprocess_data(sample_data)
        
        # Engineer features
        features = feature_engineering.engineer_features(preprocessed_data)
        
        # Prepare data for models
        tcn_data = tcn_model._prepare_data(features)
        lstm_data = lstm_model._prepare_data(features)
        
        # Train models
        tcn_model.fit(tcn_data['X'], tcn_data['y'])
        lstm_model.fit(lstm_data['X'], lstm_data['y'])
        
        # Make predictions
        tcn_pred = tcn_model.predict(tcn_data['X'])
        lstm_pred = lstm_model.predict(lstm_data['X'])
        
        # Verify predictions
        assert len(tcn_pred) == len(lstm_pred)
        assert not np.isnan(tcn_pred).any()
        assert not np.isnan(lstm_pred).any()
        
        # Verify prediction shapes
        assert tcn_pred.shape == (len(tcn_data['X']), 1)
        assert lstm_pred.shape == (len(lstm_data['X']), 1)
    
    def test_error_handling_integration(self, preprocessor, feature_engineering, tcn_model, sample_data):
        """Test error handling across the pipeline."""
        # Test with missing required columns
        invalid_data = sample_data.drop('Close', axis=1)
        
        with pytest.raises(ValueError):
            preprocessor.preprocess_data(invalid_data)
        
        # Test with invalid data types
        invalid_data = sample_data.copy()
        invalid_data['Close'] = 'invalid'
        
        with pytest.raises(ValueError):
            feature_engineering.engineer_features(invalid_data)
        
        # Test with insufficient data for sequence
        small_data = sample_data.iloc[:5]
        
        with pytest.raises(ValueError):
            tcn_model._prepare_data(small_data)
    
    def test_data_consistency_integration(self, preprocessor, feature_engineering, tcn_model, sample_data):
        """Test data consistency across the pipeline."""
        # Preprocess data
        preprocessed_data = preprocessor.preprocess_data(sample_data)
        
        # Engineer features
        features = feature_engineering.engineer_features(preprocessed_data)
        
        # Prepare data for model
        model_data = tcn_model._prepare_data(features)
        
        # Verify data consistency
        assert len(model_data['X']) == len(model_data['y'])
        assert model_data['X'].shape[0] == len(features) - tcn_model.config['sequence_length']
        
        # Verify feature order consistency
        feature_cols = tcn_model.config['feature_columns']
        for i, col in enumerate(feature_cols):
            assert col in features.columns
            assert model_data['X'][0, 0, i] == features[col].iloc[0]
    
    def test_model_persistence_integration(self, preprocessor, feature_engineering, tcn_model, sample_data):
        """Test model persistence with preprocessed and engineered features."""
        # Run full pipeline
        preprocessed_data = preprocessor.preprocess_data(sample_data)
        features = feature_engineering.engineer_features(preprocessed_data)
        model_data = tcn_model._prepare_data(features)
        
        # Train model
        tcn_model.fit(model_data['X'], model_data['y'])
        
        # Save model
        tcn_model.save('test_model.pt')
        
        # Load model
        loaded_model = TCNModel.load('test_model.pt')
        
        # Verify model parameters
        assert loaded_model.config == tcn_model.config
        
        # Verify predictions match
        original_pred = tcn_model.predict(model_data['X'])
        loaded_pred = loaded_model.predict(model_data['X'])
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_ensemble_prediction_integration(self, preprocessor, feature_engineering, tcn_model, lstm_model, sample_data):
        """Test ensemble prediction using both models."""
        # Run full pipeline
        preprocessed_data = preprocessor.preprocess_data(sample_data)
        features = feature_engineering.engineer_features(preprocessed_data)
        
        # Prepare data for both models
        tcn_data = tcn_model._prepare_data(features)
        lstm_data = lstm_model._prepare_data(features)
        
        # Train both models
        tcn_model.fit(tcn_data['X'], tcn_data['y'])
        lstm_model.fit(lstm_data['X'], lstm_data['y'])
        
        # Make predictions
        tcn_pred = tcn_model.predict(tcn_data['X'])
        lstm_pred = lstm_model.predict(lstm_data['X'])
        
        # Calculate ensemble prediction (simple average)
        ensemble_pred = (tcn_pred + lstm_pred) / 2
        
        # Verify ensemble prediction
        assert len(ensemble_pred) == len(tcn_pred)
        assert not np.isnan(ensemble_pred).any()
        assert ensemble_pred.shape == (len(tcn_data['X']), 1)
        
        # Verify ensemble prediction is within bounds of individual predictions
        assert np.all(ensemble_pred >= np.minimum(tcn_pred, lstm_pred))
        assert np.all(ensemble_pred <= np.maximum(tcn_pred, lstm_pred))
    
    def test_model_ensemble_prediction(self, sample_data, lstm_config, tcn_config):
        """Test ensemble prediction using both models."""
        # Initialize models
        lstm_model = LSTMForecaster(config=lstm_config)
        tcn_model = TCNModel(config=tcn_config)
        
        # Train both models
        lstm_model.fit(sample_data, epochs=2, batch_size=4)
        tcn_model.fit(sample_data, epochs=2, batch_size=4)
        
        # Get predictions from both models
        lstm_pred = lstm_model.predict(sample_data)
        tcn_pred = tcn_model.predict(sample_data)
        
        # Check prediction shapes
        assert lstm_pred['predictions'].shape == tcn_pred['predictions'].shape
        
        # Create ensemble prediction (simple average)
        ensemble_pred = (lstm_pred['predictions'] + tcn_pred['predictions']) / 2
        
        # Verify ensemble prediction shape
        assert ensemble_pred.shape == lstm_pred['predictions'].shape
        
        # Verify ensemble prediction is within bounds of individual predictions
        assert np.all(ensemble_pred >= np.minimum(lstm_pred['predictions'], tcn_pred['predictions']))
        assert np.all(ensemble_pred <= np.maximum(lstm_pred['predictions'], tcn_pred['predictions']))
    
    def test_model_persistence_integration(self, sample_data, lstm_config, tcn_config):
        """Test model persistence and loading in an integrated workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize and train models
            lstm_model = LSTMForecaster(config=lstm_config)
            tcn_model = TCNModel(config=tcn_config)
            
            lstm_model.fit(sample_data, epochs=2, batch_size=4)
            tcn_model.fit(sample_data, epochs=2, batch_size=4)
            
            # Save models
            lstm_path = Path(tmpdir) / "lstm_model.pt"
            tcn_path = Path(tmpdir) / "tcn_model.pt"
            
            lstm_model.save(str(lstm_path))
            tcn_model.save(str(tcn_path))
            
            # Load models
            new_lstm = LSTMForecaster(config=lstm_config)
            new_tcn = TCNModel(config=tcn_config)
            
            new_lstm.load(str(lstm_path))
            new_tcn.load(str(tcn_path))
            
            # Compare predictions
            original_lstm_pred = lstm_model.predict(sample_data)
            loaded_lstm_pred = new_lstm.predict(sample_data)
            original_tcn_pred = tcn_model.predict(sample_data)
            loaded_tcn_pred = new_tcn.predict(sample_data)
            
            assert np.allclose(original_lstm_pred['predictions'], loaded_lstm_pred['predictions'])
            assert np.allclose(original_tcn_pred['predictions'], loaded_tcn_pred['predictions'])
    
    def test_model_data_preprocessing_consistency(self, sample_data, lstm_config, tcn_config):
        """Test consistency of data preprocessing between models."""
        lstm_model = LSTMForecaster(config=lstm_config)
        tcn_model = TCNModel(config=tcn_config)
        
        # Prepare data using both models
        lstm_X, lstm_y = lstm_model._prepare_data(sample_data, is_training=True)
        tcn_X, tcn_y = tcn_model._prepare_data(sample_data, is_training=True)
        
        # Check shapes
        assert lstm_X.shape == tcn_X.shape
        assert lstm_y.shape == tcn_y.shape
        
        # Check normalization
        assert torch.allclose(lstm_X.mean(dim=(0, 1)), torch.zeros(lstm_config['input_size']), atol=1e-6)
        assert torch.allclose(tcn_X.mean(dim=(0, 1)), torch.zeros(tcn_config['input_size']), atol=1e-6)
        assert torch.allclose(lstm_X.std(dim=(0, 1)), torch.ones(lstm_config['input_size']), atol=1e-6)
        assert torch.allclose(tcn_X.std(dim=(0, 1)), torch.ones(tcn_config['input_size']), atol=1e-6)
    
    def test_model_error_handling_integration(self, sample_data, lstm_config, tcn_config):
        """Test error handling in integrated workflow."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'close': np.random.randn(100),
            'invalid': np.random.randn(100)
        })
        
        lstm_model = LSTMForecaster(config=lstm_config)
        tcn_model = TCNModel(config=tcn_config)
        
        # Both models should raise ValueError for missing columns
        with pytest.raises(ValueError, match="Missing required columns"):
            lstm_model._prepare_data(invalid_data, is_training=True)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            tcn_model._prepare_data(invalid_data, is_training=True)
        
        # Test with too short sequence
        short_data = sample_data.iloc[:5]  # Less than sequence_length
        
        with pytest.raises(ValueError):
            lstm_model._prepare_data(short_data, is_training=True)
        
        with pytest.raises(ValueError):
            tcn_model._prepare_data(short_data, is_training=True)
    
    def test_model_training_integration(self, sample_data, lstm_config, tcn_config):
        """Test integrated training workflow."""
        # Initialize models
        lstm_model = LSTMForecaster(config=lstm_config)
        tcn_model = TCNModel(config=tcn_config)
        
        # Train both models
        lstm_model.fit(sample_data, epochs=2, batch_size=4)
        tcn_model.fit(sample_data, epochs=2, batch_size=4)
        
        # Verify training history
        assert len(lstm_model.history) > 0
        assert len(tcn_model.history) > 0
        
        # Verify model states
        assert lstm_model.model.training == False  # Should be in eval mode
        assert tcn_model.model.training == False  # Should be in eval mode
        
        # Verify optimizer states
        assert lstm_model.optimizer is not None
        assert tcn_model.optimizer is not None
        
        # Verify learning rate scheduling
        assert lstm_model.scheduler is not None
        assert tcn_model.scheduler is not None
    
    def test_model_prediction_integration(self, sample_data, lstm_config, tcn_config):
        """Test integrated prediction workflow."""
        # Initialize and train models
        lstm_model = LSTMForecaster(config=lstm_config)
        tcn_model = TCNModel(config=tcn_config)
        
        lstm_model.fit(sample_data, epochs=2, batch_size=4)
        tcn_model.fit(sample_data, epochs=2, batch_size=4)
        
        # Get predictions
        lstm_pred = lstm_model.predict(sample_data)
        tcn_pred = tcn_model.predict(sample_data)
        
        # Verify prediction shapes and types
        assert isinstance(lstm_pred, dict)
        assert isinstance(tcn_pred, dict)
        assert 'predictions' in lstm_pred
        assert 'predictions' in tcn_pred
        assert isinstance(lstm_pred['predictions'], np.ndarray)
        assert isinstance(tcn_pred['predictions'], np.ndarray)
        
        # Verify prediction ranges
        assert not np.isnan(lstm_pred['predictions']).any()
        assert not np.isnan(tcn_pred['predictions']).any()
        assert not np.isinf(lstm_pred['predictions']).any()
        assert not np.isinf(tcn_pred['predictions']).any() 