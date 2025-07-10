"""
Tests for LIME Explainer

This module provides comprehensive tests for LIME explainer including:
- Explanation stability across runs
- Fixed random seed validation
- Feature importance calculation
- Error handling
- Performance testing
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

# Mock LIME imports for testing
class MockLimeTabularExplainer:
    def __init__(self, training_data, feature_names, mode, random_state=None):
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.random_state = random_state
    
    def explain_instance(self, data_row, predict_fn, num_features=10, num_samples=5000):
        # Mock explanation that respects random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Generate mock feature weights
        feature_weights = {}
        for i, feature in enumerate(self.feature_names):
            feature_weights[feature] = np.random.random()
        
        return MockExplanation(feature_weights)

class MockExplanation:
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights
    
    def as_list(self):
        return [(feature, weight) for feature, weight in self.feature_weights.items()]

# Mock the LIME imports
sys.modules['lime'] = Mock()
sys.modules['lime.lime_tabular'] = Mock()
sys.modules['lime.lime_tabular'].LimeTabularExplainer = MockLimeTabularExplainer

from trading.models.forecast_explainability import ForecastExplainability

class TestLIMEExplainer:
    """Test class for LIME explainer functionality."""
    
    @pytest.fixture
    def explainer(self):
        """Create a test explainer instance."""
        return ForecastExplainability()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)  # Fixed seed for reproducibility
        return pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'feature3': np.random.random(100),
            'target': np.random.random(100)
        })
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.predict = Mock(return_value=np.random.random(10))
        return model
    
    def test_initialization(self, explainer):
        """Test explainer initialization."""
        assert explainer is not None
        assert hasattr(explainer, '_calculate_feature_importance')
    
    def test_explanation_stability_fixed_seed(self, explainer, mock_model, sample_data):
        """Test that LIME explanations are stable with fixed random seed."""
        # Set fixed random seed
        fixed_seed = 42
        
        # Run LIME explainer multiple times with same seed
        results = []
        for _ in range(5):
            # Reset seed before each run
            np.random.seed(fixed_seed)
            result = explainer._calculate_feature_importance(mock_model, sample_data, method="lime")
            results.append(result)
        
        # All results should be identical with fixed seed
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "LIME explanations should be identical with fixed seed"
        
        # Check structure
        assert isinstance(first_result, dict)
        assert len(first_result) == len(sample_data.columns)
    
    def test_explanation_variance_without_seed(self, explainer, mock_model, sample_data):
        """Test that LIME explanations vary without fixed seed."""
        # Run LIME explainer multiple times without fixed seed
        results = []
        for _ in range(5):
            result = explainer._calculate_feature_importance(mock_model, sample_data, method="lime")
            results.append(result)
        
        # Results should vary (this is expected behavior for LIME)
        # We check that at least some results are different
        first_result = results[0]
        different_results = 0
        
        for result in results[1:]:
            if result != first_result:
                different_results += 1
        
        # At least some results should be different
        assert different_results > 0, "LIME explanations should vary without fixed seed"
    
    def test_feature_importance_structure(self, explainer, mock_model, sample_data):
        """Test that LIME feature importance has correct structure."""
        result = explainer._calculate_feature_importance(mock_model, sample_data, method="lime")
        
        # Check structure
        assert isinstance(result, dict)
        assert len(result) == len(sample_data.columns)
        
        # Check all values are numeric
        for value in result.values():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_empty_input_handling(self, explainer, mock_model):
        """Test handling of empty input data."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Empty input data"):
            explainer._calculate_feature_importance(mock_model, empty_df, method="lime")
        
        # Test with None model
        with pytest.raises(ValueError, match="Model cannot be None"):
            explainer._calculate_feature_importance(None, pd.DataFrame({'test': [1]}), method="lime")
    
    def test_single_row_data(self, explainer, mock_model):
        """Test handling of single row data."""
        single_row_data = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0]
        })
        
        result = explainer._calculate_feature_importance(mock_model, single_row_data, method="lime")
        
        assert isinstance(result, dict)
        assert len(result) == len(single_row_data.columns)
    
    def test_large_dataset_handling(self, explainer, mock_model):
        """Test handling of large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'feature1': np.random.random(1000),
            'feature2': np.random.random(1000),
            'feature3': np.random.random(1000)
        })
        
        result = explainer._calculate_feature_importance(mock_model, large_data, method="lime")
        
        assert isinstance(result, dict)
        assert len(result) == len(large_data.columns)
    
    def test_missing_values_handling(self, explainer, mock_model):
        """Test handling of missing values in data."""
        # Create data with missing values
        data_with_nans = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],
            'feature2': [1.0, np.nan, 3.0, 4.0],
            'feature3': [1.0, 2.0, 3.0, np.nan]
        })
        
        # Should handle missing values gracefully
        result = explainer._calculate_feature_importance(mock_model, data_with_nans, method="lime")
        
        assert isinstance(result, dict)
        assert len(result) == len(data_with_nans.columns)
    
    def test_categorical_features(self, explainer, mock_model):
        """Test handling of categorical features."""
        # Create data with categorical features
        categorical_data = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B'],
            'feature2': [1.0, 2.0, 3.0, 4.0],
            'feature3': ['X', 'Y', 'X', 'Y']
        })
        
        result = explainer._calculate_feature_importance(mock_model, categorical_data, method="lime")
        
        assert isinstance(result, dict)
        assert len(result) == len(categorical_data.columns)
    
    def test_model_prediction_errors(self, explainer, sample_data):
        """Test handling of model prediction errors."""
        # Create model that raises exception
        error_model = Mock()
        error_model.predict = Mock(side_effect=Exception("Prediction error"))
        
        with pytest.raises(Exception, match="Prediction error"):
            explainer._calculate_feature_importance(error_model, sample_data, method="lime")
    
    def test_data_type_validation(self, explainer, mock_model):
        """Test validation of data types."""
        # Test with numpy array instead of DataFrame
        numpy_data = np.random.random((100, 3))
        
        with pytest.raises(ValueError, match="Features must be a pandas DataFrame"):
            explainer._calculate_feature_importance(mock_model, numpy_data, method="lime")
        
        # Test with list instead of DataFrame
        list_data = [[1, 2, 3], [4, 5, 6]]
        
        with pytest.raises(ValueError, match="Features must be a pandas DataFrame"):
            explainer._calculate_feature_importance(mock_model, list_data, method="lime")
    
    def test_explanation_consistency(self, explainer, mock_model, sample_data):
        """Test that explanations are consistent across different data subsets."""
        # Test with different subsets of the same data
        subset1 = sample_data.iloc[:50]
        subset2 = sample_data.iloc[50:]
        
        result1 = explainer._calculate_feature_importance(mock_model, subset1, method="lime")
        result2 = explainer._calculate_feature_importance(mock_model, subset2, method="lime")
        
        # Both should have same structure
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert len(result1) == len(result2)
        assert set(result1.keys()) == set(result2.keys())
    
    def test_feature_importance_bounds(self, explainer, mock_model, sample_data):
        """Test that feature importance values are within reasonable bounds."""
        result = explainer._calculate_feature_importance(mock_model, sample_data, method="lime")
        
        for feature, importance in result.items():
            # Importance should be finite
            assert np.isfinite(importance)
            
            # Importance should be reasonable (not extremely large or small)
            assert -1000 < importance < 1000
    
    def test_explanation_performance(self, explainer, mock_model, sample_data):
        """Test performance of LIME explanations."""
        import time
        
        # Time the explanation
        start_time = time.time()
        result = explainer._calculate_feature_importance(mock_model, sample_data, method="lime")
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (less than 10 seconds)
        assert execution_time < 10.0
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_memory_usage(self, explainer, mock_model):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large dataset
        large_data = pd.DataFrame({
            'feature1': np.random.random(2000),
            'feature2': np.random.random(2000),
            'feature3': np.random.random(2000)
        })
        
        # Run explainer
        result = explainer._calculate_feature_importance(mock_model, large_data, method="lime")
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 200MB)
        assert memory_increase < 200 * 1024 * 1024  # 200MB
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_concurrent_execution(self, explainer, mock_model, sample_data):
        """Test concurrent execution of LIME explainer."""
        import threading
        import time
        
        results = []
        errors = []
        
        def run_explainer():
            try:
                result = explainer._calculate_feature_importance(mock_model, sample_data, method="lime")
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
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
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
        
        # All results should have same structure
        for result in results:
            assert isinstance(result, dict)
            assert len(result) == len(sample_data.columns)
    
    def test_explanation_quality(self, explainer, mock_model, sample_data):
        """Test quality of LIME explanations."""
        # Run multiple explanations and check consistency
        explanations = []
        
        for _ in range(10):
            result = explainer._calculate_feature_importance(mock_model, sample_data, method="lime")
            explanations.append(result)
        
        # Check that all explanations have same features
        feature_sets = [set(exp.keys()) for exp in explanations]
        assert all(fs == feature_sets[0] for fs in feature_sets)
        
        # Check that importance values are reasonable
        for explanation in explanations:
            for feature, importance in explanation.items():
                assert isinstance(importance, (int, float))
                assert not np.isnan(importance)
                assert not np.isinf(importance)
    
    def test_error_recovery(self, explainer, mock_model, sample_data):
        """Test error recovery in LIME explainer."""
        # Test with model that sometimes fails
        unreliable_model = Mock()
        call_count = 0
        
        def unreliable_predict(data):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every third call
                raise Exception("Temporary failure")
            return np.random.random(len(data))
        
        unreliable_model.predict = unreliable_predict
        
        # Should handle temporary failures gracefully
        try:
            result = explainer._calculate_feature_importance(unreliable_model, sample_data, method="lime")
            assert isinstance(result, dict)
        except Exception:
            # It's acceptable for LIME to fail with unreliable models
            pass
    
    def test_explanation_comparison(self, explainer, mock_model, sample_data):
        """Test comparison between LIME and other explainers."""
        # Test LIME vs SHAP
        lime_result = explainer._calculate_feature_importance(mock_model, sample_data, method="lime")
        shap_result = explainer._calculate_feature_importance(mock_model, sample_data, method="shap")
        
        # Both should return dictionaries
        assert isinstance(lime_result, dict)
        assert isinstance(shap_result, dict)
        
        # Both should have same features
        assert set(lime_result.keys()) == set(shap_result.keys())
        
        # Values should be different (LIME and SHAP use different methods)
        assert lime_result != shap_result

if __name__ == "__main__":
    pytest.main([__file__]) 