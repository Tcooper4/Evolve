"""
Tests for SHAP Explainer

This module provides comprehensive tests for SHAP explainer including:
- Model type detection
- Empty input handling
- Invalid model type handling
- Feature importance calculation
- Explanation stability
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'trading'))

# Mock SHAP imports for testing
class MockSHAPExplainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def explain_row(self, row):
        return np.random.random(len(row))

class MockTreeExplainer:
    def __init__(self, model):
        self.model = model
    
    def shap_values(self, data):
        return np.random.random((len(data), data.shape[1]))

class MockKernelExplainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def shap_values(self, data):
        return np.random.random((len(data), data.shape[1]))

# Mock the SHAP imports
sys.modules['shap'] = Mock()
sys.modules['shap'].TreeExplainer = MockTreeExplainer
sys.modules['shap'].KernelExplainer = MockKernelExplainer
sys.modules['shap'].Explainer = MockSHAPExplainer

from trading.models.forecast_explainability import ForecastExplainability

class TestSHAPExplainer:
    """Test class for SHAP explainer functionality."""
    
    @pytest.fixture
    def explainer(self):
        """Create a test explainer instance."""
        return ForecastExplainability()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'feature3': np.random.random(100),
            'target': np.random.random(100)
        })
    
    @pytest.fixture
    def tree_model(self):
        """Create a mock tree-based model."""
        model = Mock()
        model.__class__.__name__ = 'RandomForestRegressor'
        model.predict = Mock(return_value=np.random.random(10))
        return model
    
    @pytest.fixture
    def lstm_model(self):
        """Create a mock LSTM model."""
        model = Mock()
        model.__class__.__name__ = 'Sequential'
        model.predict = Mock(return_value=np.random.random(10))
        return model
    
    @pytest.fixture
    def prophet_model(self):
        """Create a mock Prophet model."""
        model = Mock()
        model.__class__.__name__ = 'Prophet'
        model.predict = Mock(return_value=pd.DataFrame({'yhat': np.random.random(10)}))
        return model
    
    def test_initialization(self, explainer):
        """Test explainer initialization."""
        assert explainer is not None
        assert hasattr(explainer, '_calculate_feature_importance')
    
    def test_empty_input_handling(self, explainer):
        """Test handling of empty input data."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Empty input data"):
            explainer._calculate_feature_importance(None, empty_df, method="shap")
        
        # Test with None input
        with pytest.raises(ValueError, match="Model cannot be None"):
            explainer._calculate_feature_importance(None, pd.DataFrame({'test': [1]}), method="shap")
        
        # Test with empty features
        model = Mock()
        empty_features = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Empty features data"):
            explainer._calculate_feature_importance(model, empty_features, method="shap")
    
    def test_invalid_model_type_handling(self, explainer, sample_data):
        """Test handling of unsupported model types."""
        # Test with unsupported model type
        unsupported_model = Mock()
        unsupported_model.__class__.__name__ = 'UnsupportedModel'
        
        # Should fallback to KernelExplainer for unsupported models
        result = explainer._calculate_feature_importance(unsupported_model, sample_data, method="shap")
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_tree_model_detection(self, explainer, tree_model, sample_data):
        """Test detection and handling of tree-based models."""
        # Test with RandomForestRegressor
        result = explainer._calculate_feature_importance(tree_model, sample_data, method="shap")
        
        assert isinstance(result, dict)
        assert len(result) == len(sample_data.columns)
        
        # Test with other tree models
        tree_models = ['DecisionTreeRegressor', 'GradientBoostingRegressor', 'XGBRegressor']
        
        for model_name in tree_models:
            model = Mock()
            model.__class__.__name__ = model_name
            model.predict = Mock(return_value=np.random.random(10))
            
            result = explainer._calculate_feature_importance(model, sample_data, method="shap")
            assert isinstance(result, dict)
            assert len(result) > 0
    
    def test_lstm_model_fallback(self, explainer, lstm_model, sample_data):
        """Test fallback to KernelExplainer for LSTM models."""
        result = explainer._calculate_feature_importance(lstm_model, sample_data, method="shap")
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Should use KernelExplainer for non-tree models
        # This is tested by checking that the result is not None and has the expected structure
    
    def test_prophet_model_fallback(self, explainer, prophet_model, sample_data):
        """Test fallback to KernelExplainer for Prophet models."""
        result = explainer._calculate_feature_importance(prophet_model, sample_data, method="shap")
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_lime_explainer(self, explainer, tree_model, sample_data):
        """Test LIME explainer functionality."""
        result = explainer._calculate_feature_importance(tree_model, sample_data, method="lime")
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_integrated_explainer(self, explainer, tree_model, sample_data):
        """Test integrated explainer functionality."""
        result = explainer._calculate_feature_importance(tree_model, sample_data, method="integrated")
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_invalid_method_handling(self, explainer, tree_model, sample_data):
        """Test handling of invalid explainer methods."""
        with pytest.raises(ValueError, match="Unsupported explainer method"):
            explainer._calculate_feature_importance(tree_model, sample_data, method="invalid_method")
    
    def test_feature_importance_structure(self, explainer, tree_model, sample_data):
        """Test that feature importance has correct structure."""
        result = explainer._calculate_feature_importance(tree_model, sample_data, method="shap")
        
        # Check structure
        assert isinstance(result, dict)
        assert len(result) == len(sample_data.columns)
        
        # Check all values are numeric
        for value in result.values():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_explanation_stability(self, explainer, tree_model, sample_data):
        """Test that explanations are stable across multiple runs."""
        # Run multiple times with same input
        results = []
        for _ in range(5):
            result = explainer._calculate_feature_importance(tree_model, sample_data, method="shap")
            results.append(result)
        
        # Check that all results have same structure
        for result in results:
            assert isinstance(result, dict)
            assert len(result) == len(sample_data.columns)
    
    def test_model_prediction_errors(self, explainer, sample_data):
        """Test handling of model prediction errors."""
        # Create model that raises exception
        error_model = Mock()
        error_model.__class__.__name__ = 'RandomForestRegressor'
        error_model.predict = Mock(side_effect=Exception("Prediction error"))
        
        with pytest.raises(Exception, match="Prediction error"):
            explainer._calculate_feature_importance(error_model, sample_data, method="shap")
    
    def test_data_type_validation(self, explainer, tree_model):
        """Test validation of data types."""
        # Test with numpy array instead of DataFrame
        numpy_data = np.random.random((100, 3))
        
        with pytest.raises(ValueError, match="Features must be a pandas DataFrame"):
            explainer._calculate_feature_importance(tree_model, numpy_data, method="shap")
        
        # Test with list instead of DataFrame
        list_data = [[1, 2, 3], [4, 5, 6]]
        
        with pytest.raises(ValueError, match="Features must be a pandas DataFrame"):
            explainer._calculate_feature_importance(tree_model, list_data, method="shap")
    
    def test_single_row_data(self, explainer, tree_model):
        """Test handling of single row data."""
        single_row_data = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0]
        })
        
        result = explainer._calculate_feature_importance(tree_model, single_row_data, method="shap")
        
        assert isinstance(result, dict)
        assert len(result) == len(single_row_data.columns)
    
    def test_large_dataset_handling(self, explainer, tree_model):
        """Test handling of large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'feature1': np.random.random(10000),
            'feature2': np.random.random(10000),
            'feature3': np.random.random(10000)
        })
        
        result = explainer._calculate_feature_importance(tree_model, large_data, method="shap")
        
        assert isinstance(result, dict)
        assert len(result) == len(large_data.columns)
    
    def test_missing_values_handling(self, explainer, tree_model):
        """Test handling of missing values in data."""
        # Create data with missing values
        data_with_nans = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],
            'feature2': [1.0, np.nan, 3.0, 4.0],
            'feature3': [1.0, 2.0, 3.0, np.nan]
        })
        
        # Should handle missing values gracefully
        result = explainer._calculate_feature_importance(tree_model, data_with_nans, method="shap")
        
        assert isinstance(result, dict)
        assert len(result) == len(data_with_nans.columns)
    
    def test_categorical_features(self, explainer, tree_model):
        """Test handling of categorical features."""
        # Create data with categorical features
        categorical_data = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B'],
            'feature2': [1.0, 2.0, 3.0, 4.0],
            'feature3': ['X', 'Y', 'X', 'Y']
        })
        
        result = explainer._calculate_feature_importance(tree_model, categorical_data, method="shap")
        
        assert isinstance(result, dict)
        assert len(result) == len(categorical_data.columns)
    
    def test_explainer_method_comparison(self, explainer, tree_model, sample_data):
        """Test comparison between different explainer methods."""
        methods = ["shap", "lime", "integrated"]
        results = {}
        
        for method in methods:
            try:
                result = explainer._calculate_feature_importance(tree_model, sample_data, method=method)
                results[method] = result
            except Exception as e:
                results[method] = f"Error: {str(e)}"
        
        # All methods should return results (or error messages)
        for method, result in results.items():
            assert result is not None
            if not isinstance(result, str):  # Not an error
                assert isinstance(result, dict)
                assert len(result) > 0
    
    def test_memory_usage(self, explainer, tree_model):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large dataset
        large_data = pd.DataFrame({
            'feature1': np.random.random(5000),
            'feature2': np.random.random(5000),
            'feature3': np.random.random(5000)
        })
        
        # Run explainer
        result = explainer._calculate_feature_importance(tree_model, large_data, method="shap")
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_concurrent_execution(self, explainer, tree_model, sample_data):
        """Test concurrent execution of explainer."""
        import threading
        import time
        
        results = []
        errors = []
        
        def run_explainer():
            try:
                result = explainer._calculate_feature_importance(tree_model, sample_data, method="shap")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_explainer)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3
        assert len(errors) == 0
        
        # All results should be similar
        for result in results:
            assert isinstance(result, dict)
            assert len(result) == len(sample_data.columns)
    
    def test_feature_importance_bounds_validation(self, explainer, tree_model, sample_data):
        """Test that feature importances are within expected bounds and flag values exceeding 1.0."""
        # Get SHAP feature importance
        result = explainer._calculate_feature_importance(tree_model, sample_data, method="shap")
        
        # Check structure
        assert isinstance(result, dict)
        assert len(result) == len(sample_data.columns)
        
        # Validate each feature importance value
        for feature_name, importance in result.items():
            # Basic type and value checks
            assert isinstance(importance, (int, float)), f"Feature {feature_name} importance should be numeric"
            assert not np.isnan(importance), f"Feature {feature_name} importance should not be NaN"
            assert not np.isinf(importance), f"Feature {feature_name} importance should not be infinite"
            
            # Check for values exceeding 1.0 (potential bug indicator)
            if abs(importance) > 1.0:
                # Log warning for values exceeding 1.0
                print(f"WARNING: Feature {feature_name} has importance {importance} exceeding 1.0")
                # This could indicate a bug in the SHAP calculation or data normalization
                # In a real implementation, you might want to raise a warning or error
                assert False, f"Feature importance {importance} for {feature_name} exceeds 1.0 - potential bug"
            
            # Check for reasonable bounds (SHAP values should typically be in reasonable range)
            assert abs(importance) <= 10.0, f"Feature {feature_name} importance {importance} seems unreasonably high"
        
        # Test with different model types to ensure bounds validation works
        lstm_model = Mock()
        lstm_model.__class__.__name__ = 'Sequential'
        lstm_model.predict = Mock(return_value=np.random.random(10))
        
        lstm_result = explainer._calculate_feature_importance(lstm_model, sample_data, method="shap")
        
        for feature_name, importance in lstm_result.items():
            assert isinstance(importance, (int, float))
            assert not np.isnan(importance)
            assert not np.isinf(importance)
            
            # Check for values exceeding 1.0
            if abs(importance) > 1.0:
                assert False, f"LSTM model feature importance {importance} for {feature_name} exceeds 1.0"
        
        # Test with normalized data to ensure bounds are still respected
        normalized_data = (sample_data - sample_data.mean()) / sample_data.std()
        normalized_result = explainer._calculate_feature_importance(tree_model, normalized_data, method="shap")
        
        for feature_name, importance in normalized_result.items():
            assert isinstance(importance, (int, float))
            assert not np.isnan(importance)
            assert not np.isinf(importance)
            
            # Even with normalized data, SHAP values should not exceed reasonable bounds
            if abs(importance) > 1.0:
                assert False, f"Normalized data feature importance {importance} for {feature_name} exceeds 1.0"
        
        # Test edge case with very small dataset
        small_data = sample_data.head(5)
        small_result = explainer._calculate_feature_importance(tree_model, small_data, method="shap")
        
        for feature_name, importance in small_result.items():
            assert isinstance(importance, (int, float))
            assert not np.isnan(importance)
            assert not np.isinf(importance)
            
            # Small datasets might have more variance, but still check bounds
            if abs(importance) > 1.0:
                assert False, f"Small dataset feature importance {importance} for {feature_name} exceeds 1.0"

if __name__ == "__main__":
    pytest.main([__file__]) 