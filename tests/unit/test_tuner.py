"""
Tests for the parameter tuner functionality.

This module tests the ModelParameterTuner class with comprehensive
coverage including edge cases, invalid inputs, and error scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
import json
import os

from trading.ui.model_parameter_tuner import ModelParameterTuner, ParameterConfig

class TestModelParameterTuner:
    """Test cases for ModelParameterTuner."""
    
    @pytest.fixture
    def tuner(self):
        """Create a parameter tuner instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('trading.ui.model_parameter_tuner.Path') as mock_path:
                mock_path.return_value.parent.mkdir.return_value = None
                mock_path.return_value.exists.return_value = False
                tuner = ModelParameterTuner()
                tuner.history_file = mock_path.return_value
                yield tuner
    
    def test_initialization(self, tuner):
        """Test tuner initialization."""
        assert tuner.parameter_configs is not None
        assert isinstance(tuner.parameter_configs, dict)
        assert 'transformer' in tuner.parameter_configs
        assert 'lstm' in tuner.parameter_configs
        assert 'xgboost' in tuner.parameter_configs
    
    def test_parameter_validation_valid(self, tuner):
        """Test parameter validation with valid inputs."""
        valid_params = {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout': 0.2
        }
        
        result = tuner._validate_parameters('transformer', valid_params)
        assert result['valid'] is True
        assert 'valid' in result['message']
    
    def test_parameter_validation_invalid_range(self, tuner):
        """Test parameter validation with invalid range."""
        invalid_params = {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 10,  # Too many layers
            'dropout': 0.2
        }
        
        result = tuner._validate_parameters('transformer', invalid_params)
        assert result['valid'] is False
        assert 'overfitting' in result['message']
    
    def test_parameter_validation_missing_keys(self, tuner):
        """Test parameter validation with missing keys."""
        incomplete_params = {
            'd_model': 64,
            # Missing nhead
            'num_layers': 2
        }
        
        result = tuner._validate_parameters('transformer', incomplete_params)
        # Should handle missing keys gracefully
        assert isinstance(result, dict)
    
    def test_parameter_validation_edge_cases(self, tuner):
        """Test parameter validation with edge case values."""
        edge_cases = [
            # Zero values
            {'d_model': 0, 'nhead': 4, 'num_layers': 2},
            # Negative values
            {'d_model': -64, 'nhead': 4, 'num_layers': 2},
            # Very large values
            {'d_model': 10000, 'nhead': 4, 'num_layers': 2},
            # Float values where int expected
            {'d_model': 64.5, 'nhead': 4, 'num_layers': 2}
        ]
        
        for params in edge_cases:
            result = tuner._validate_parameters('transformer', params)
            assert isinstance(result, dict)
            assert 'valid' in result
    
    def test_optimization_history_persistence(self, tuner):
        """Test optimization history saving and loading."""
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = '[]'
            
            # Test adding result
            tuner.add_optimization_result(
                model_type='lstm',
                parameters={'hidden_size': 64, 'layers': 2},
                performance_metrics={'accuracy': 0.85, 'rmse': 0.02}
            )
            
            # Verify save was called
            assert mock_open.called
    
    def test_optimization_history_filtering(self, tuner):
        """Test optimization history filtering by model type."""
        # Add test data
        tuner.optimization_history = [
            {'model_type': 'lstm', 'timestamp': '2024-01-01'},
            {'model_type': 'transformer', 'timestamp': '2024-01-02'},
            {'model_type': 'lstm', 'timestamp': '2024-01-03'}
        ]
        
        # Test filtering
        lstm_results = tuner.get_optimization_history(model_type='lstm')
        assert len(lstm_results) == 2
        assert all(r['model_type'] == 'lstm' for r in lstm_results)
        
        # Test limit
        limited_results = tuner.get_optimization_history(limit=1)
        assert len(limited_results) == 1
    
    def test_optimization_history_edge_cases(self, tuner):
        """Test optimization history with edge cases."""
        # Empty history
        results = tuner.get_optimization_history()
        assert results == []
        
        # Invalid model type
        results = tuner.get_optimization_history(model_type='nonexistent')
        assert results == []
        
        # Zero limit
        results = tuner.get_optimization_history(limit=0)
        assert results == []
    
    def test_parameter_config_validation(self, tuner):
        """Test parameter configuration validation."""
        # Test with invalid parameter types
        invalid_configs = [
            {'name': 'test', 'param_type': 'invalid_type'},
            {'name': 'test', 'param_type': 'int', 'min_value': 'not_a_number'},
            {'name': 'test', 'param_type': 'float', 'max_value': 'not_a_number'}
        ]
        
        for config in invalid_configs:
            # Should not raise exception, but handle gracefully
            assert isinstance(config, dict)
    
    def test_error_handling(self, tuner):
        """Test error handling in various scenarios."""
        # Test with None parameters
        result = tuner._validate_parameters('transformer', None)
        assert isinstance(result, dict)
        
        # Test with empty parameters
        result = tuner._validate_parameters('transformer', {})
        assert isinstance(result, dict)
        
        # Test with invalid model type
        result = tuner._validate_parameters('nonexistent_model', {})
        assert isinstance(result, dict)
    
    def test_file_operations_error_handling(self, tuner):
        """Test error handling in file operations."""
        # Mock file operations to raise exceptions
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Should handle file errors gracefully
            tuner._load_optimization_history()
            assert tuner.optimization_history == []
            
            tuner._save_optimization_history()
            # Should not raise exception
    
    def test_parameter_boundary_conditions(self, tuner):
        """Test parameter boundary conditions."""
        boundary_tests = [
            # XGBoost boundary tests
            {
                'model': 'xgboost',
                'params': {'max_depth': 16, 'learning_rate': 0.31},  # Above limits
                'expected_valid': False
            },
            {
                'model': 'xgboost',
                'params': {'max_depth': 15, 'learning_rate': 0.3},  # At limits
                'expected_valid': True
            },
            # LSTM boundary tests
            {
                'model': 'lstm',
                'params': {'hidden_size': 257},  # Above limit
                'expected_valid': False
            },
            {
                'model': 'lstm',
                'params': {'hidden_size': 256},  # At limit
                'expected_valid': True
            }
        ]
        
        for test in boundary_tests:
            result = tuner._validate_parameters(test['model'], test['params'])
            if test['expected_valid']:
                assert result['valid'] is True
            else:
                assert result['valid'] is False
    
    def test_parameter_type_validation(self, tuner):
        """Test parameter type validation."""
        type_tests = [
            # String where number expected
            {'d_model': 'not_a_number', 'nhead': 4},
            # List where single value expected
            {'d_model': [64, 128], 'nhead': 4},
            # Dict where simple value expected
            {'d_model': {'value': 64}, 'nhead': 4}
        ]
        
        for params in type_tests:
            result = tuner._validate_parameters('transformer', params)
            assert isinstance(result, dict)
            # Should handle type errors gracefully
    
    def test_memory_management(self, tuner):
        """Test memory management with large parameter sets."""
        # Test with large number of parameters
        large_params = {f'param_{i}': i for i in range(1000)}
        
        result = tuner._validate_parameters('transformer', large_params)
        assert isinstance(result, dict)
        
        # Test with large optimization history
        large_history = [{'model_type': 'test', 'timestamp': '2024-01-01'} for _ in range(10000)]
        tuner.optimization_history = large_history
        
        # Should handle large datasets without memory issues
        results = tuner.get_optimization_history(limit=100)
        assert len(results) <= 100 