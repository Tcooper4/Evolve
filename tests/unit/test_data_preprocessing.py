import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading.data.preprocessing import (
    DataPreprocessor,
    FeatureEngineering,
    DataScaler,
    DataValidator
)

class TestDataPreprocessing:
    """Test suite for data preprocessing."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(10) * 10 + 100,
            'High': np.random.randn(10) * 10 + 105,
            'Low': np.random.randn(10) * 10 + 95,
            'Close': np.random.randn(10) * 10 + 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        return data
    
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()
    
    @pytest.fixture
    def feature_engineering(self):
        return FeatureEngineering()
    
    @pytest.fixture
    def scaler(self):
        return DataScaler()
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    def test_data_preprocessor_initialization(self, preprocessor):
        """Test data preprocessor initialization."""
        assert preprocessor is not None
        assert preprocessor.scaler is not None
        assert preprocessor.validator is not None
    
    def test_data_preprocessor_clean_data(self, preprocessor, sample_data):
        """Test data cleaning."""
        # Add some missing values and outliers
        dirty_data = sample_data.copy()
        dirty_data.loc[dirty_data.index[0], 'Close'] = np.nan
        dirty_data.loc[dirty_data.index[1], 'Volume'] = 1e12  # Outlier
        
        cleaned_data = preprocessor.clean_data(dirty_data)
        
        assert isinstance(cleaned_data, pd.DataFrame)
        assert not cleaned_data.isna().any().any()
        assert cleaned_data['Volume'].max() < 1e12
    
    def test_data_preprocessor_normalize_data(self, preprocessor, sample_data):
        """Test data normalization."""
        normalized_data = preprocessor.normalize_data(sample_data)
        
        assert isinstance(normalized_data, pd.DataFrame)
        assert not normalized_data.isna().any().any()
        assert normalized_data.shape == sample_data.shape
        
        # Check if data is normalized
        for col in normalized_data.columns:
            assert normalized_data[col].mean() < 1
            assert normalized_data[col].std() < 1
    
    def test_feature_engineering_technical_indicators(self, feature_engineering, sample_data):
        """Test technical indicator calculation."""
        features = feature_engineering.calculate_technical_indicators(sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert not features.isna().any().any()
        
        # Check for common technical indicators
        expected_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
        for indicator in expected_indicators:
            assert any(col.startswith(indicator) for col in features.columns)
    
    def test_feature_engineering_fourier_features(self, feature_engineering, sample_data):
        """Test Fourier feature calculation."""
        features = feature_engineering.calculate_fourier_features(sample_data, n_components=3)
        
        assert isinstance(features, pd.DataFrame)
        assert not features.isna().any().any()
        
        # Check for Fourier features
        expected_features = ['fourier_1', 'fourier_2', 'fourier_3']
        for feature in expected_features:
            assert any(col.startswith(feature) for col in features.columns)
    
    def test_feature_engineering_lag_features(self, feature_engineering, sample_data):
        """Test lag feature calculation."""
        features = feature_engineering.calculate_lag_features(sample_data, n_lags=3)
        
        assert isinstance(features, pd.DataFrame)
        assert not features.isna().any().any()
        
        # Check for lag features
        for col in sample_data.columns:
            for lag in range(1, 4):
                assert f'{col}_lag_{lag}' in features.columns
    
    def test_data_scaler_fit_transform(self, scaler, sample_data):
        """Test scaler fit and transform."""
        # Fit and transform
        scaled_data = scaler.fit_transform(sample_data)
        
        assert isinstance(scaled_data, pd.DataFrame)
        assert not scaled_data.isna().any().any()
        assert scaled_data.shape == sample_data.shape
        
        # Check if data is scaled
        for col in scaled_data.columns:
            assert abs(scaled_data[col].mean()) < 1
            assert scaled_data[col].std() < 1
    
    def test_data_scaler_inverse_transform(self, scaler, sample_data):
        """Test scaler inverse transform."""
        # Fit and transform
        scaled_data = scaler.fit_transform(sample_data)
        
        # Inverse transform
        original_data = scaler.inverse_transform(scaled_data)
        
        assert isinstance(original_data, pd.DataFrame)
        assert not original_data.isna().any().any()
        assert original_data.shape == sample_data.shape
        
        # Check if data is restored
        np.testing.assert_allclose(
            original_data.values,
            sample_data.values,
            rtol=1e-2, atol=1e-2
        )
    
    def test_data_validator_validate_data(self, validator, sample_data):
        """Test data validation."""
        # Test valid data
        assert validator.validate_data(sample_data)
        
        # Test invalid data
        invalid_data = sample_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Close'] = -1  # Invalid price
        assert not validator.validate_data(invalid_data)
    
    def test_data_validator_check_missing_values(self, validator, sample_data):
        """Test missing value check."""
        # Test data without missing values
        assert not validator.check_missing_values(sample_data)
        
        # Test data with missing values
        invalid_data = sample_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Close'] = np.nan
        assert validator.check_missing_values(invalid_data)
    
    def test_data_validator_check_outliers(self, validator, sample_data):
        """Test outlier check."""
        # Test data without outliers
        assert not validator.check_outliers(sample_data)
        
        # Test data with outliers
        invalid_data = sample_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Volume'] = 1e12  # Outlier
        assert validator.check_outliers(invalid_data)
    
    def test_data_validator_check_data_types(self, validator, sample_data):
        """Test data type check."""
        # Test valid data types
        assert validator.check_data_types(sample_data)
        
        # Test invalid data types
        invalid_data = sample_data.copy()
        invalid_data['Volume'] = invalid_data['Volume'].astype(str)
        assert not validator.check_data_types(invalid_data)
    
    def test_data_validator_check_date_index(self, validator, sample_data):
        """Test date index check."""
        # Test valid date index
        assert validator.check_date_index(sample_data)
        
        # Test invalid date index
        invalid_data = sample_data.copy()
        invalid_data.index = range(len(sample_data))
        assert not validator.check_date_index(invalid_data) 