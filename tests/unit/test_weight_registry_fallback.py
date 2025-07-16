"""
Unit tests for weight registry fallback handling.
"""

import json
import tempfile
import os
import pytest
from unittest.mock import patch, mock_open

from memory.performance_weights import get_latest_weights


class TestWeightRegistryFallback:
    """Test weight registry fallback handling."""

    def test_get_latest_weights_file_not_found(self):
        """Test get_latest_weights when file doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = get_latest_weights()
            expected = {'LSTM': 0.25, 'XGB': 0.25, 'ARIMA': 0.25, 'Prophet': 0.25}
            assert result == expected

    def test_get_latest_weights_json_decode_error(self):
        """Test get_latest_weights when JSON is corrupted."""
        with patch('builtins.open', mock_open(read_data="invalid json")):
            result = get_latest_weights()
            expected = {'LSTM': 0.25, 'XGB': 0.25, 'ARIMA': 0.25, 'Prophet': 0.25}
            assert result == expected

    def test_get_latest_weights_valid_file(self):
        """Test get_latest_weights with valid JSON file."""
        valid_weights = {'LSTM': 0.3, 'XGB': 0.4, 'ARIMA': 0.2, 'Prophet': 0.1}
        with patch('builtins.open', mock_open(read_data=json.dumps(valid_weights))):
            result = get_latest_weights()
            assert result == valid_weights

    def test_get_latest_weights_empty_file(self):
        """Test get_latest_weights with empty file."""
        with patch('builtins.open', mock_open(read_data="")):
            result = get_latest_weights()
            expected = {'LSTM': 0.25, 'XGB': 0.25, 'ARIMA': 0.25, 'Prophet': 0.25}
            assert result == expected

    def test_get_latest_weights_malformed_json(self):
        """Test get_latest_weights with malformed JSON."""
        malformed_json = '{"LSTM": 0.3, "XGB": 0.4, "ARIMA": 0.2, "Prophet": 0.1'  # Missing closing brace
        with patch('builtins.open', mock_open(read_data=malformed_json)):
            result = get_latest_weights()
            expected = {'LSTM': 0.25, 'XGB': 0.25, 'ARIMA': 0.25, 'Prophet': 0.25}
            assert result == expected

    @patch('memory.performance_weights.logger')
    def test_logger_warning_on_fallback(self, mock_logger):
        """Test that logger.warning is called when using fallback weights."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            get_latest_weights()
            mock_logger.warning.assert_called_with(
                "Weight file not found or corrupted — using default weights."
            )

    def test_default_weights_structure(self):
        """Test that default weights have the correct structure."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = get_latest_weights()
            assert isinstance(result, dict)
            assert 'LSTM' in result
            assert 'XGB' in result
            assert 'ARIMA' in result
            assert 'Prophet' in result
            assert sum(result.values()) == 1.0  # Weights should sum to 1

    def test_default_weights_values(self):
        """Test that default weights have equal distribution."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = get_latest_weights()
            assert result['LSTM'] == 0.25
            assert result['XGB'] == 0.25
            assert result['ARIMA'] == 0.25
            assert result['Prophet'] == 0.25 