"""
Comprehensive tests for XGBoost forecasting model.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'trading', 'models'))

def test_xgboost_empty_df():
    """Test XGBoost model with empty DataFrame."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        result = model.forecast(pd.DataFrame())
        
        # Should handle empty DataFrame gracefully
        assert result is not None
        assert hasattr(result, 'empty') or isinstance(result, (list, dict))
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_none_input():
    """Test XGBoost model with None input."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        result = model.forecast(None)
        
        # Should handle None input gracefully
        assert result is not None
        assert hasattr(result, 'empty') or isinstance(result, (list, dict))
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_missing_columns():
    """Test XGBoost model with missing required columns."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Create DataFrame without 'Close' column
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97]
        })
        
        result = model.forecast(data)
        
        # Should handle missing columns gracefully
        assert result is not None
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_correct_output_shape():
    """Test that XGBoost model produces correct output shape."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Create sample data
        data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
        result = model.forecast(data)
        
        # Check output structure
        assert result is not None
        
        # If result is DataFrame, check it has expected columns
        if isinstance(result, pd.DataFrame):
            assert len(result) > 0
            assert 'forecast' in result.columns or 'prediction' in result.columns
        
        # If result is dict, check it has expected keys
        elif isinstance(result, dict):
            assert 'forecast' in result or 'predictions' in result or 'values' in result
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_exception_fallback():
    """Test XGBoost model exception fallback logic."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Create problematic data that might cause exceptions
        data = pd.DataFrame({
            'Close': [np.nan, np.inf, -np.inf, 100, 101],  # Contains invalid values
            'Volume': [0, 0, 0, 1000, 1000]
        })
        
        result = model.forecast(data)
        
        # Should handle exceptions gracefully
        assert result is not None
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_validation():
    """Test XGBoost model input validation."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Test with various invalid inputs
        invalid_inputs = [
            "not a dataframe",
            123,
            [],
            {},
            pd.Series([1, 2, 3])
        ]
        
        for invalid_input in invalid_inputs:
            result = model.forecast(invalid_input)
            assert result is not None  # Should not crash
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_parameter_validation():
    """Test XGBoost model parameter validation."""
    try:
        from xgboost_model import XGBoostModel
        
        # Test with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            model = XGBoostModel(n_estimators=-1)
        
        with pytest.raises((ValueError, TypeError)):
            model = XGBoostModel(max_depth=0)
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_training_data():
    """Test XGBoost model with training data."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Create realistic training data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        result = model.forecast(data)
        
        # Should produce valid forecast
        assert result is not None
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_feature_importance():
    """Test XGBoost model feature importance."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Create sample data
        data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100),
            'RSI': np.random.uniform(0, 100, 100),
            'MACD': np.random.randn(100)
        })
        
        # Test feature importance if available
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance(data)
            assert importance is not None
            assert isinstance(importance, (dict, pd.Series))
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_hyperparameters():
    """Test XGBoost model hyperparameter handling."""
    try:
        from xgboost_model import XGBoostModel
        
        # Test with different hyperparameters
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }
        
        model = XGBoostModel(**hyperparams)
        
        # Verify hyperparameters are set correctly
        for param, value in hyperparams.items():
            assert hasattr(model, param) or hasattr(model, f'_{param}')
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

def test_xgboost_model_persistence():
    """Test XGBoost model save/load functionality."""
    try:
        from xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Test save functionality
        save_path = "test_xgboost_model.pkl"
        try:
            model.save(save_path)
            assert os.path.exists(save_path)
            
            # Test load functionality
            loaded_model = XGBoostModel.load(save_path)
            assert loaded_model is not None
            
        finally:
            # Cleanup
            if os.path.exists(save_path):
                os.remove(save_path)
        
    except ImportError as e:
        pytest.skip(f"XGBoostModel not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 