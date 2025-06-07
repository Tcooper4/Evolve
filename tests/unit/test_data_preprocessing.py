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
        """Create preprocessor with default settings."""
        return DataPreprocessor()
    
    @pytest.fixture
    def custom_preprocessor(self):
        """Create preprocessor with custom settings."""
        return DataPreprocessor(config={
            'scaling_method': 'minmax',
            'outlier_method': 'zscore',
            'outlier_threshold': 2.5,
            'handle_missing': 'interpolate',
            'validation_level': 'strict'
        })
    
    @pytest.fixture
    def feature_engineering(self):
        """Create feature engineering with default settings."""
        return FeatureEngineering()
    
    @pytest.fixture
    def custom_feature_engineering(self):
        """Create feature engineering with custom settings."""
        return FeatureEngineering(config={
            'ma_windows': [10, 20, 50],
            'rsi_window': 10,
            'macd_params': {'fast': 8, 'slow': 21, 'signal': 5},
            'bb_window': 15,
            'bb_std': 2.5,
            'fourier_periods': [5, 10, 20],
            'lag_periods': [1, 3, 5],
            'volume_ma_windows': [3, 7, 14]
        })
    
    @pytest.fixture
    def scaler(self):
        return DataScaler()
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor is not None
        assert preprocessor.scaler is not None
        assert preprocessor.validator is not None
        assert not preprocessor.is_fitted
        assert preprocessor.feature_stats == {}
        
        # Check default settings
        assert preprocessor.scaling_method == 'standard'
        assert preprocessor.outlier_method == 'iqr'
        assert preprocessor.outlier_threshold == 1.5
        assert preprocessor.handle_missing == 'ffill'
        assert preprocessor.validation_level == 'strict'
    
    def test_custom_preprocessor_initialization(self, custom_preprocessor):
        """Test custom preprocessor initialization."""
        assert custom_preprocessor.scaling_method == 'minmax'
        assert custom_preprocessor.outlier_method == 'zscore'
        assert custom_preprocessor.outlier_threshold == 2.5
        assert custom_preprocessor.handle_missing == 'interpolate'
        assert custom_preprocessor.validation_level == 'strict'
    
    def test_input_validation(self, preprocessor, sample_data):
        """Test input validation."""
        # Test valid data
        preprocessor._validate_input(sample_data)  # Should not raise
        
        # Test invalid data types
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            preprocessor._validate_input(np.array([1, 2, 3]))
        
        # Test empty DataFrame
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            preprocessor._validate_input(pd.DataFrame())
        
        # Test non-numeric data
        non_numeric = pd.DataFrame({'text': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match="No numeric columns found"):
            preprocessor._validate_input(non_numeric)
        
        # Test infinite values
        infinite_data = sample_data.copy()
        infinite_data.loc[infinite_data.index[0], 'Close'] = np.inf
        with pytest.raises(ValueError, match="Input data contains infinite values"):
            preprocessor._validate_input(infinite_data)
        
        # Test invalid index
        invalid_index = sample_data.copy()
        invalid_index.index = range(len(invalid_index))
        with pytest.raises(ValueError, match="DataFrame must have a DatetimeIndex"):
            preprocessor._validate_input(invalid_index)
        
        # Test duplicate indices
        duplicate_index = sample_data.copy()
        duplicate_index.index = [duplicate_index.index[0]] * len(duplicate_index)
        with pytest.raises(ValueError, match="DataFrame contains duplicate indices"):
            preprocessor._validate_input(duplicate_index)
    
    def test_missing_value_handling(self, preprocessor, custom_preprocessor, sample_data):
        """Test missing value handling."""
        # Add missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[data_with_missing.index[0], 'Close'] = np.nan
        data_with_missing.loc[data_with_missing.index[1], 'Volume'] = np.nan
        
        # Test ffill method
        cleaned_data = preprocessor._handle_missing_values(data_with_missing)
        assert not cleaned_data.isna().any().any()
        
        # Test interpolate method
        cleaned_data = custom_preprocessor._handle_missing_values(data_with_missing)
        assert not cleaned_data.isna().any().any()
        
        # Test invalid method
        preprocessor.handle_missing = 'invalid'
        with pytest.raises(ValueError, match="Unknown missing value handling method"):
            preprocessor._handle_missing_values(data_with_missing)
    
    def test_outlier_handling(self, preprocessor, custom_preprocessor, sample_data):
        """Test outlier handling."""
        # Add outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers.loc[data_with_outliers.index[0], 'Close'] = 1000  # Extreme value
        
        # Test IQR method
        cleaned_data = preprocessor._handle_outliers(data_with_outliers)
        assert cleaned_data['Close'].max() < 1000
        
        # Test z-score method
        cleaned_data = custom_preprocessor._handle_outliers(data_with_outliers)
        assert cleaned_data['Close'].max() < 1000
        
        # Test invalid method
        preprocessor.outlier_method = 'invalid'
        with pytest.raises(ValueError, match="Unknown outlier handling method"):
            preprocessor._handle_outliers(data_with_outliers)
    
    def test_feature_statistics(self, preprocessor, sample_data):
        """Test feature statistics computation."""
        preprocessor._compute_feature_stats(sample_data)
        
        for col in sample_data.columns:
            assert col in preprocessor.feature_stats
            stats = preprocessor.feature_stats[col]
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            
            # Verify statistics
            assert abs(stats['mean'] - sample_data[col].mean()) < 1e-10
            assert abs(stats['std'] - sample_data[col].std()) < 1e-10
            assert abs(stats['min'] - sample_data[col].min()) < 1e-10
            assert abs(stats['max'] - sample_data[col].max()) < 1e-10
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit and transform workflow."""
        # Fit and transform
        transformed_data = preprocessor.fit_transform(sample_data)
        
        # Check results
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape == sample_data.shape
        assert not transformed_data.isna().any().any()
        assert preprocessor.is_fitted
        
        # Check normalization
        for col in transformed_data.columns:
            assert abs(transformed_data[col].mean()) < 1e-10
            assert abs(transformed_data[col].std() - 1) < 1e-10
    
    def test_inverse_transform(self, preprocessor, sample_data):
        """Test inverse transform."""
        # Fit and transform
        transformed_data = preprocessor.fit_transform(sample_data)
        
        # Inverse transform
        original_data = preprocessor.inverse_transform(transformed_data)
        
        # Check results
        assert isinstance(original_data, pd.DataFrame)
        assert original_data.shape == sample_data.shape
        assert not original_data.isna().any().any()
        
        # Check if data is restored
        np.testing.assert_allclose(
            original_data.values,
            sample_data.values,
            rtol=1e-2, atol=1e-2
        )
    
    def test_transform_without_fit(self, preprocessor, sample_data):
        """Test transform without fit."""
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.transform(sample_data)
    
    def test_inverse_transform_without_fit(self, preprocessor, sample_data):
        """Test inverse transform without fit."""
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.inverse_transform(sample_data)
    
    def test_get_feature_stats(self, preprocessor, sample_data):
        """Test getting feature statistics."""
        preprocessor.fit(sample_data)
        stats = preprocessor.get_feature_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) == len(sample_data.columns)
        
        for col in sample_data.columns:
            assert col in stats
            assert all(key in stats[col] for key in ['mean', 'std', 'min', 'max'])
    
    def test_get_set_params(self, preprocessor):
        """Test getting and setting parameters."""
        # Get parameters
        params = preprocessor.get_params()
        assert isinstance(params, dict)
        assert all(key in params for key in [
            'scaling_method', 'outlier_method', 'outlier_threshold',
            'handle_missing', 'validation_level'
        ])
        
        # Set parameters
        new_params = {
            'scaling_method': 'minmax',
            'outlier_method': 'zscore',
            'outlier_threshold': 2.5
        }
        preprocessor.set_params(**new_params)
        
        # Verify changes
        assert preprocessor.scaling_method == 'minmax'
        assert preprocessor.outlier_method == 'zscore'
        assert preprocessor.outlier_threshold == 2.5
        
        # Test invalid parameter
        with pytest.raises(ValueError, match="Unknown parameter"):
            preprocessor.set_params(invalid_param=1)
    
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

        # Check if the mean is approximately 0 and the standard deviation is approximately 1
        for col in normalized_data.columns:
            mean = normalized_data[col].mean()
            std = normalized_data[col].std()
            assert abs(mean) < 1e-6, f"Mean for {col} is not approximately 0: {mean}"
            assert abs(std - 1) < 1e-6, f"Standard deviation for {col} is not approximately 1: {std}"
    
    def test_feature_engineering_initialization(self, feature_engineering):
        """Test feature engineering initialization."""
        assert feature_engineering is not None
        
        # Check default settings
        assert feature_engineering.ma_windows == [20, 50, 200]
        assert feature_engineering.rsi_window == 14
        assert feature_engineering.macd_params == {'fast': 12, 'slow': 26, 'signal': 9}
        assert feature_engineering.bb_window == 20
        assert feature_engineering.bb_std == 2.0
        assert feature_engineering.fourier_periods == [7, 14, 30]
        assert feature_engineering.lag_periods == [1, 2, 3, 5, 10]
        assert feature_engineering.volume_ma_windows == [5, 10, 20]
    
    def test_custom_feature_engineering_initialization(self, custom_feature_engineering):
        """Test custom feature engineering initialization."""
        assert custom_feature_engineering.ma_windows == [10, 20, 50]
        assert custom_feature_engineering.rsi_window == 10
        assert custom_feature_engineering.macd_params == {'fast': 8, 'slow': 21, 'signal': 5}
        assert custom_feature_engineering.bb_window == 15
        assert custom_feature_engineering.bb_std == 2.5
        assert custom_feature_engineering.fourier_periods == [5, 10, 20]
        assert custom_feature_engineering.lag_periods == [1, 3, 5]
        assert custom_feature_engineering.volume_ma_windows == [3, 7, 14]
    
    def test_config_validation(self, feature_engineering):
        """Test configuration validation."""
        # Test invalid moving average windows
        with pytest.raises(ValueError, match="Moving average windows must be positive integers"):
            feature_engineering.ma_windows = [-1, 20, 50]
            feature_engineering._validate_config()
        
        # Test invalid RSI window
        with pytest.raises(ValueError, match="RSI window must be a positive integer"):
            feature_engineering.rsi_window = 0
            feature_engineering._validate_config()
        
        # Test invalid MACD parameters
        with pytest.raises(ValueError, match="MACD parameters must include"):
            feature_engineering.macd_params = {'fast': 12, 'slow': 26}
            feature_engineering._validate_config()
        
        # Test invalid Bollinger Bands parameters
        with pytest.raises(ValueError, match="Bollinger Bands window must be a positive integer"):
            feature_engineering.bb_window = -1
            feature_engineering._validate_config()
        
        with pytest.raises(ValueError, match="Bollinger Bands standard deviation must be positive"):
            feature_engineering.bb_std = 0
            feature_engineering._validate_config()
    
    def test_moving_averages(self, feature_engineering, sample_data):
        """Test moving average calculation."""
        data = feature_engineering.calculate_moving_averages(sample_data)
        
        # Check if all moving averages are calculated
        for window in feature_engineering.ma_windows:
            assert f'SMA_{window}' in data.columns
            assert f'EMA_{window}' in data.columns
            
            # Check if values are within reasonable range
            assert not data[f'SMA_{window}'].isna().all()
            assert not data[f'EMA_{window}'].isna().all()
    
    def test_rsi_calculation(self, feature_engineering, sample_data):
        """Test RSI calculation."""
        data = feature_engineering.calculate_rsi(sample_data)
        
        assert 'RSI' in data.columns
        assert not data['RSI'].isna().all()
        assert data['RSI'].min() >= 0
        assert data['RSI'].max() <= 100
    
    def test_macd_calculation(self, feature_engineering, sample_data):
        """Test MACD calculation."""
        data = feature_engineering.calculate_macd(sample_data)
        
        assert 'MACD' in data.columns
        assert 'MACD_Signal' in data.columns
        assert 'MACD_Hist' in data.columns
        
        # Check if values are within reasonable range
        assert not data['MACD'].isna().all()
        assert not data['MACD_Signal'].isna().all()
        assert not data['MACD_Hist'].isna().all()
    
    def test_bollinger_bands(self, feature_engineering, sample_data):
        """Test Bollinger Bands calculation."""
        data = feature_engineering.calculate_bollinger_bands(sample_data)
        
        assert 'BB_Middle' in data.columns
        assert 'BB_Upper' in data.columns
        assert 'BB_Lower' in data.columns
        assert 'BB_Width' in data.columns
        
        # Check if values are within reasonable range
        assert not data['BB_Middle'].isna().all()
        assert not data['BB_Upper'].isna().all()
        assert not data['BB_Lower'].isna().all()
        assert not data['BB_Width'].isna().all()
        
        # Check if upper band is always above lower band
        assert (data['BB_Upper'] >= data['BB_Lower']).all()
    
    def test_volume_indicators(self, feature_engineering, sample_data):
        """Test volume indicator calculation."""
        data = feature_engineering.calculate_volume_indicators(sample_data)
        
        # Check volume moving averages
        for window in feature_engineering.volume_ma_windows:
            assert f'Volume_MA_{window}' in data.columns
            assert not data[f'Volume_MA_{window}'].isna().all()
        
        # Check other volume indicators
        assert 'Volume_Trend' in data.columns
        assert 'VPT' in data.columns
        assert 'OBV' in data.columns
        
        assert not data['Volume_Trend'].isna().all()
        assert not data['VPT'].isna().all()
        assert not data['OBV'].isna().all()
    
    def test_fourier_features(self, feature_engineering, sample_data):
        """Test Fourier feature calculation."""
        data = feature_engineering.calculate_fourier_features(sample_data)
        
        for period in feature_engineering.fourier_periods:
            assert f'Fourier_Sin_{period}' in data.columns
            assert f'Fourier_Cos_{period}' in data.columns
            
            # Check if values are within [-1, 1] range
            assert data[f'Fourier_Sin_{period}'].min() >= -1
            assert data[f'Fourier_Sin_{period}'].max() <= 1
            assert data[f'Fourier_Cos_{period}'].min() >= -1
            assert data[f'Fourier_Cos_{period}'].max() <= 1
    
    def test_lag_features(self, feature_engineering, sample_data):
        """Test lag feature calculation."""
        data = feature_engineering.calculate_lag_features(sample_data)
        
        for lag in feature_engineering.lag_periods:
            assert f'Close_Lag_{lag}' in data.columns
            assert f'Close_Return_Lag_{lag}' in data.columns
            assert f'Volume_Lag_{lag}' in data.columns
            assert f'Volume_Return_Lag_{lag}' in data.columns
            assert f'HL_Range_Lag_{lag}' in data.columns
            
            # Check if values are properly lagged
            assert data[f'Close_Lag_{lag}'].equals(data['Close'].shift(lag))
            assert data[f'Volume_Lag_{lag}'].equals(data['Volume'].shift(lag))
    
    def test_momentum_indicators(self, feature_engineering, sample_data):
        """Test momentum indicator calculation."""
        data = feature_engineering.calculate_momentum_indicators(sample_data)
        
        # Check ROC
        for period in [5, 10, 20]:
            assert f'ROC_{period}' in data.columns
            assert f'Momentum_{period}' in data.columns
            assert not data[f'ROC_{period}'].isna().all()
            assert not data[f'Momentum_{period}'].isna().all()
        
        # Check Stochastic Oscillator
        assert 'Stoch_K' in data.columns
        assert 'Stoch_D' in data.columns
        assert not data['Stoch_K'].isna().all()
        assert not data['Stoch_D'].isna().all()
        
        # Check if values are within [0, 100] range
        assert data['Stoch_K'].min() >= 0
        assert data['Stoch_K'].max() <= 100
        assert data['Stoch_D'].min() >= 0
        assert data['Stoch_D'].max() <= 100
    
    def test_engineer_features(self, feature_engineering, sample_data):
        """Test complete feature engineering workflow."""
        data = feature_engineering.engineer_features(sample_data)
        
        # Check if all features are present
        feature_list = feature_engineering.get_feature_list()
        assert all(feature in data.columns for feature in feature_list)
        
        # Check if there are no NaN values
        assert not data.isna().any().any()
        
        # Check if data shape is correct
        assert len(data) > 0
    
    def test_missing_columns(self, feature_engineering, sample_data):
        """Test handling of missing columns."""
        # Remove required column
        invalid_data = sample_data.drop('Close', axis=1)
        
        with pytest.raises(ValueError, match="Input data must contain columns"):
            feature_engineering.engineer_features(invalid_data)
    
    def test_get_feature_list(self, feature_engineering):
        """Test getting feature list."""
        feature_list = feature_engineering.get_feature_list()
        
        assert isinstance(feature_list, list)
        assert len(feature_list) > 0
        
        # Check if all expected features are in the list
        expected_features = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
            'Volume_Trend', 'VPT', 'OBV',
            'Stoch_K', 'Stoch_D'
        ]
        assert all(feature in feature_list for feature in expected_features)
    
    def test_get_set_params(self, feature_engineering):
        """Test getting and setting parameters."""
        # Get parameters
        params = feature_engineering.get_params()
        assert isinstance(params, dict)
        assert all(key in params for key in [
            'ma_windows', 'rsi_window', 'macd_params', 'bb_window',
            'bb_std', 'fourier_periods', 'lag_periods', 'volume_ma_windows'
        ])
        
        # Set parameters
        new_params = {
            'ma_windows': [10, 20],
            'rsi_window': 10,
            'bb_std': 2.5
        }
        feature_engineering.set_params(**new_params)
        
        # Verify changes
        assert feature_engineering.ma_windows == [10, 20]
        assert feature_engineering.rsi_window == 10
        assert feature_engineering.bb_std == 2.5
        
        # Test invalid parameter
        with pytest.raises(ValueError, match="Unknown parameter"):
            feature_engineering.set_params(invalid_param=1)
    
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