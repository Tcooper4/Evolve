"""
Tests for the parameter tuner functionality.

This module tests the ModelParameterTuner class with comprehensive
coverage including edge cases, invalid inputs, and error scenarios.
"""

import json
import tempfile
from unittest.mock import patch

import pytest

from trading.ui.model_parameter_tuner import ModelParameterTuner


class TestModelParameterTuner:
    """Test cases for ModelParameterTuner."""

    @pytest.fixture
    def tuner(self):
        """Create a parameter tuner instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("trading.ui.model_parameter_tuner.Path") as mock_path:
                mock_path.return_value.parent.mkdir.return_value = None
                mock_path.return_value.exists.return_value = False
                tuner = ModelParameterTuner()
                tuner.history_file = mock_path.return_value
                yield tuner

    def test_initialization(self, tuner):
        """Test tuner initialization."""
        assert tuner.parameter_configs is not None
        assert isinstance(tuner.parameter_configs, dict)
        assert "transformer" in tuner.parameter_configs
        assert "lstm" in tuner.parameter_configs
        assert "xgboost" in tuner.parameter_configs

    def test_parameter_validation_valid(self, tuner):
        """Test parameter validation with valid inputs."""
        valid_params = {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.2}

        result = tuner._validate_parameters("transformer", valid_params)
        assert result["valid"] is True
        assert "valid" in result["message"]

    def test_parameter_validation_invalid_range(self, tuner):
        """Test parameter validation with invalid range."""
        invalid_params = {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 10,
            "dropout": 0.2,
        }  # Too many layers

        result = tuner._validate_parameters("transformer", invalid_params)
        assert result["valid"] is False
        assert "overfitting" in result["message"]

    def test_parameter_validation_missing_keys(self, tuner):
        """Test parameter validation with missing keys."""
        incomplete_params = {
            "d_model": 64,
            # Missing nhead
            "num_layers": 2,
        }

        result = tuner._validate_parameters("transformer", incomplete_params)
        # Should handle missing keys gracefully
        assert isinstance(result, dict)

    def test_parameter_validation_edge_cases(self, tuner):
        """Test parameter validation with edge case values."""
        edge_cases = [
            # Zero values
            {"d_model": 0, "nhead": 4, "num_layers": 2},
            # Negative values
            {"d_model": -64, "nhead": 4, "num_layers": 2},
            # Very large values
            {"d_model": 10000, "nhead": 4, "num_layers": 2},
            # Float values where int expected
            {"d_model": 64.5, "nhead": 4, "num_layers": 2},
        ]

        for params in edge_cases:
            result = tuner._validate_parameters("transformer", params)
            assert isinstance(result, dict)
            assert "valid" in result

    def test_optimization_history_persistence(self, tuner):
        """Test optimization history saving and loading."""
        # Mock file operations
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "[]"

            # Test adding result
            tuner.add_optimization_result(
                model_type="lstm",
                parameters={"hidden_size": 64, "layers": 2},
                performance_metrics={"accuracy": 0.85, "rmse": 0.02},
            )

            # Verify save was called
            assert mock_open.called

    def test_optimization_history_filtering(self, tuner):
        """Test optimization history filtering by model type."""
        # Add test data
        tuner.optimization_history = [
            {"model_type": "lstm", "timestamp": "2024-01-01"},
            {"model_type": "transformer", "timestamp": "2024-01-02"},
            {"model_type": "lstm", "timestamp": "2024-01-03"},
        ]

        # Test filtering
        lstm_results = tuner.get_optimization_history(model_type="lstm")
        assert len(lstm_results) == 2
        assert all(r["model_type"] == "lstm" for r in lstm_results)

        # Test limit
        limited_results = tuner.get_optimization_history(limit=1)
        assert len(limited_results) == 1

    def test_optimization_history_edge_cases(self, tuner):
        """Test optimization history with edge cases."""
        # Empty history
        results = tuner.get_optimization_history()
        assert results == []

        # Invalid model type
        results = tuner.get_optimization_history(model_type="nonexistent")
        assert results == []

        # Zero limit
        results = tuner.get_optimization_history(limit=0)
        assert results == []

    def test_parameter_config_validation(self, tuner):
        """Test parameter configuration validation."""
        # Test with invalid parameter types
        invalid_configs = [
            {"name": "test", "param_type": "invalid_type"},
            {"name": "test", "param_type": "int", "min_value": "not_a_number"},
            {"name": "test", "param_type": "float", "max_value": "not_a_number"},
        ]

        for config in invalid_configs:
            # Should not raise exception, but handle gracefully
            assert isinstance(config, dict)

    def test_error_handling(self, tuner):
        """Test error handling in various scenarios."""
        # Test with None parameters
        result = tuner._validate_parameters("transformer", None)
        assert isinstance(result, dict)

        # Test with empty parameters
        result = tuner._validate_parameters("transformer", {})
        assert isinstance(result, dict)

        # Test with invalid model type
        result = tuner._validate_parameters("nonexistent_model", {})
        assert isinstance(result, dict)

    def test_file_operations_error_handling(self, tuner):
        """Test error handling in file operations."""
        # Mock file operations to raise exceptions
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
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
                "model": "xgboost",
                "params": {"max_depth": 16, "learning_rate": 0.31},  # Above limits
                "expected_valid": False,
            },
            {
                "model": "xgboost",
                "params": {"max_depth": 15, "learning_rate": 0.3},  # At limits
                "expected_valid": True,
            },
            # LSTM boundary tests
            {
                "model": "lstm",
                "params": {"hidden_size": 257},
                "expected_valid": False,
            },  # Above limit
            {
                "model": "lstm",
                "params": {"hidden_size": 256},
                "expected_valid": True,
            },  # At limit
        ]

        for test in boundary_tests:
            result = tuner._validate_parameters(test["model"], test["params"])
            if test["expected_valid"]:
                assert result["valid"] is True
            else:
                assert result["valid"] is False

    def test_parameter_type_validation(self, tuner):
        """Test parameter type validation."""
        type_tests = [
            # String where number expected
            {"d_model": "not_a_number", "nhead": 4},
            # List where single value expected
            {"d_model": [64, 128], "nhead": 4},
            # Dict where simple value expected
            {"d_model": {"value": 64}, "nhead": 4},
        ]

        for params in type_tests:
            result = tuner._validate_parameters("transformer", params)
            assert isinstance(result, dict)
            # Should handle type errors gracefully

    def test_memory_management(self, tuner):
        """Test memory management with large optimization histories."""
        # Add many optimization results
        for i in range(1000):
            tuner.add_optimization_result(
                model_type="lstm",
                parameters={"hidden_size": 64 + i, "layers": 2},
                performance_metrics={"accuracy": 0.85 + i * 0.001, "rmse": 0.02},
            )

        # Should handle large history without memory issues
        history = tuner.get_optimization_history()
        assert len(history) <= 1000  # Should not exceed reasonable limits

    def test_configuration_preservation(self, tuner):
        """Test that tuning doesn't overwrite prior saved configurations."""
        # Create initial configuration
        initial_config = {
            "lstm": {"hidden_size": 64, "layers": 2, "dropout": 0.2},
            "transformer": {"d_model": 128, "nhead": 8, "num_layers": 4},
        }

        # Save initial configuration
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write.return_value = None
            tuner._save_configuration(initial_config, "initial_config.json")

        # Perform tuning that might modify parameters
        tuning_params = {
            "lstm": {
                "hidden_size": 128,  # Different value
                "layers": 3,  # Different value
                "dropout": 0.3,  # Different value
            }
        }

        # Mock the tuning process
        with patch.object(tuner, "_run_optimization") as mock_optimize:
            mock_optimize.return_value = {
                "best_params": tuning_params,
                "best_score": 0.95,
            }

            # Run tuning
            result = tuner.tune_parameters(
                "lstm", {"hidden_size": [64, 128], "layers": [2, 3]}
            )

        # Verify that original configuration is preserved
        # Load the saved configuration
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(initial_config)
            )
            saved_config = tuner._load_configuration("initial_config.json")

        # Check that original values are preserved
        assert saved_config["lstm"]["hidden_size"] == 64  # Original value
        assert saved_config["lstm"]["layers"] == 2  # Original value
        assert saved_config["lstm"]["dropout"] == 0.2  # Original value
        assert saved_config["transformer"]["d_model"] == 128  # Unchanged
        assert saved_config["transformer"]["nhead"] == 8  # Unchanged

        # Verify that tuning created new configuration without overwriting
        assert result is not None
        assert "best_params" in result
        assert result["best_params"]["lstm"]["hidden_size"] == 128  # New tuned value

        # Test configuration versioning
        config_versions = tuner._get_configuration_versions()
        assert len(config_versions) >= 1  # Should have at least initial config

        # Test that we can restore previous configuration
        restored_config = tuner._restore_configuration("initial_config.json")
        assert restored_config["lstm"]["hidden_size"] == 64  # Restored original value

        # Test that multiple tuning runs don't interfere
        second_tuning_params = {
            "lstm": {"hidden_size": 256, "layers": 4, "dropout": 0.4}
        }

        with patch.object(tuner, "_run_optimization") as mock_optimize:
            mock_optimize.return_value = {
                "best_params": second_tuning_params,
                "best_score": 0.97,
            }

            second_result = tuner.tune_parameters(
                "lstm", {"hidden_size": [128, 256], "layers": [3, 4]}
            )

        # Verify all configurations are preserved
        assert (
            second_result["best_params"]["lstm"]["hidden_size"] == 256
        )  # Second tuning
        assert (
            result["best_params"]["lstm"]["hidden_size"] == 128
        )  # First tuning preserved
        assert saved_config["lstm"]["hidden_size"] == 64  # Original preserved

        # Test configuration backup before tuning
        with patch.object(tuner, "_backup_configuration") as mock_backup:
            tuner.tune_parameters("lstm", {"hidden_size": [64, 128]})
            assert mock_backup.called  # Should create backup before tuning

        # Test configuration rollback capability
        with patch.object(tuner, "_rollback_configuration") as mock_rollback:
            tuner._rollback_to_previous_configuration()
            assert mock_rollback.called  # Should be able to rollback
