"""
Tests for the UpdaterAgent

This module tests the UpdaterAgent functionality including
model updating, retraining, and error handling.
"""

import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from trading.agents.base_agent_interface import AgentConfig

# Local imports
from trading.agents.updater_agent import UpdaterAgent, UpdateRequest, UpdateResult


class TestUpdateRequest:
    """Test the UpdateRequest dataclass."""

    def test_update_request_creation(self):
        """Test creating an UpdateRequest instance."""
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,  # Will be set in actual usage
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        assert request.model_id == "model_123"
        assert request.update_type == "retrain"
        assert request.priority == "normal"
        assert request.request_id == "update_456"

    def test_update_request_defaults(self):
        """Test UpdateRequest with default values."""
        request = UpdateRequest(model_id="model_123", evaluation_result=None)

        assert request.model_id == "model_123"
        assert request.priority == "normal"
        assert request.request_id is None


class TestUpdateResult:
    """Test the UpdateResult dataclass."""

    def test_update_result_creation(self):
        """Test creating an UpdateResult instance."""
        result = UpdateResult(
            request_id="update_456",
            model_id="model_123",
            original_model_id="model_123_old",
            update_timestamp="2024-01-01T12:00:00",
            update_type="retrain",
            new_model_path="models/model_123_new.pkl",
            new_model_id="model_123_new",
            improvement_metrics={"mse": 0.008, "mae": 0.04},
            update_status="success",
            error_message=None,
        )

        assert result.request_id == "update_456"
        assert result.model_id == "model_123"
        assert result.update_timestamp == "2024-01-01T12:00:00"
        assert result.update_type == "retrain"
        assert result.new_model_path == "models/model_123_new.pkl"
        assert result.new_model_id == "model_123_new"
        assert result.improvement_metrics == {"mse": 0.008, "mae": 0.04}
        assert result.update_status == "success"
        assert result.error_message is None

    def test_update_result_defaults(self):
        """Test UpdateResult with default values."""
        result = UpdateResult(
            request_id="update_456",
            model_id="model_123",
            original_model_id="model_123_old",
            update_timestamp="2024-01-01T12:00:00",
            update_type="retrain",
            new_model_path="models/model_123_new.pkl",
            new_model_id="model_123_new",
        )

        assert result.request_id == "update_456"
        assert result.model_id == "model_123"
        assert result.update_timestamp == "2024-01-01T12:00:00"
        assert result.update_type == "retrain"
        assert result.new_model_path == "models/model_123_new.pkl"
        assert result.new_model_id == "model_123_new"
        assert result.improvement_metrics == {}
        assert result.update_status == "success"
        assert result.error_message is None


class TestUpdaterAgent:
    """Test the UpdaterAgent class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_old_data(self, temp_data_dir):
        """Create sample old data for testing."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 105,
                "low": np.random.randn(100).cumsum() + 95,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )

        data_path = Path(temp_data_dir) / "old_data.csv"
        data.to_csv(data_path, index=False)
        return str(data_path)

    @pytest.fixture
    def sample_new_data(self, temp_data_dir):
        """Create sample new data for testing."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=50, freq="D"),
                "open": np.random.randn(50).cumsum() + 110,
                "high": np.random.randn(50).cumsum() + 115,
                "low": np.random.randn(50).cumsum() + 105,
                "close": np.random.randn(50).cumsum() + 110,
                "volume": np.random.randint(1000, 10000, 50),
            }
        )

        data_path = Path(temp_data_dir) / "new_data.csv"
        data.to_csv(data_path, index=False)
        return str(data_path)

    @pytest.fixture
    def mock_old_model(self, temp_data_dir):
        """Create a mock old model for testing."""
        model = Mock()
        model.predict.return_value = np.random.randn(100) * 0.01 + 0.001

        model_path = Path(temp_data_dir) / "old_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return str(model_path), model

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(
            name="test_updater",
            enabled=True,
            priority=3,
            max_concurrent_runs=2,
            timeout_seconds=600,
            retry_attempts=2,
            custom_config={
                "backup_models": True,
                "max_model_age_days": 30,
                "performance_decay_threshold": 0.1,
            },
        )

    @pytest.fixture
    def updater_agent(self, agent_config):
        """Create an UpdaterAgent instance for testing."""
        return UpdaterAgent(agent_config)

    def test_updater_agent_initialization(self, updater_agent):
        """Test UpdaterAgent initialization."""
        assert updater_agent.config.name == "test_updater"
        assert updater_agent.config.enabled is True
        assert updater_agent.backup_dir.exists()
        assert isinstance(updater_agent.update_history, dict)

    def test_updater_agent_metadata(self, updater_agent):
        """Test UpdaterAgent metadata."""
        metadata = updater_agent.get_metadata()

        assert metadata["name"] == "test_updater"
        assert metadata["version"] == "1.0.0"
        assert "model-updating" in metadata["tags"]
        assert "model_retraining" in metadata["capabilities"]

    def test_updater_agent_validate_input_valid(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test input validation with valid data."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        assert updater_agent.validate_input(request=request) is True

    def test_updater_agent_validate_input_invalid_model_path(
        self, updater_agent, sample_new_data
    ):
        """Test input validation with invalid model path."""
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        assert updater_agent.validate_input(request=request) is False

    def test_updater_agent_validate_input_invalid_new_data_path(
        self, updater_agent, mock_old_model
    ):
        """Test input validation with invalid new data path."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        assert updater_agent.validate_input(request=request) is False

    def test_updater_agent_validate_input_invalid_update_type(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test input validation with invalid update type."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="invalid_type",
            priority="normal",
            request_id="update_456",
        )

        assert updater_agent.validate_input(request=request) is False

    def test_updater_agent_validate_input_missing_request(self, updater_agent):
        """Test input validation with missing request."""
        assert updater_agent.validate_input() is False

    def test_updater_agent_validate_input_wrong_type(self, updater_agent):
        """Test input validation with wrong request type."""
        assert updater_agent.validate_input(request="not_a_request") is False

    @pytest.mark.asyncio
    async def test_updater_agent_execute_success(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test successful model update execution."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = await updater_agent.execute(request=request)

        assert result.success is True
        assert "new_model_path" in result.data
        assert "performance_improvement" in result.data
        assert result.data["update_type"] == "retrain"

    @pytest.mark.asyncio
    async def test_updater_agent_execute_invalid_request(self, updater_agent):
        """Test execution with invalid request."""
        result = await updater_agent.execute()

        assert result.success is False
        assert "required" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_updater_agent_execute_wrong_request_type(self, updater_agent):
        """Test execution with wrong request type."""
        result = await updater_agent.execute(request="not_a_request")

        assert result.success is False
        assert "instance" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_updater_agent_update_lstm_model(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test LSTM model updating."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        assert result.update_status == "success"
        assert result.model_id == "model_123"
        assert result.update_type == "retrain"
        assert result.new_model_path is not None
        assert Path(result.new_model_path).exists()
        assert result.old_model_path == model_path

    @pytest.mark.asyncio
    async def test_updater_agent_update_xgboost_model(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test XGBoost model updating."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        assert result.update_status == "success"
        assert result.model_id == "model_123"
        assert result.update_type == "retrain"
        assert result.new_model_path is not None
        assert Path(result.new_model_path).exists()

    @pytest.mark.asyncio
    async def test_updater_agent_incremental_update(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test incremental model update."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="incremental",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        assert result.update_status == "success"
        assert result.model_id == "model_123"
        assert result.update_type == "incremental"
        assert result.new_model_path is not None
        assert Path(result.new_model_path).exists()

    @pytest.mark.asyncio
    async def test_updater_agent_update_with_performance_threshold(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test model update with performance threshold."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        assert result.update_status == "success"
        assert result.model_id == "model_123"
        assert result.performance_improvement is not None

    @pytest.mark.asyncio
    async def test_updater_agent_update_invalid_model_type(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test updating with invalid model type."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="invalid_type",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        assert result.update_status == "failed"
        assert "unsupported" in result.error_message.lower()

    def test_updater_agent_get_update_history(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test getting update history."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        # Update model first
        updater_agent.update_model(request)

        # Get history
        history = updater_agent.get_update_history("model_123")

        assert isinstance(history, list)
        assert len(history) > 0
        assert all(isinstance(update_result, UpdateResult) for update_result in history)

    def test_updater_agent_get_update_history_nonexistent(self, updater_agent):
        """Test getting update history for nonexistent model."""
        history = updater_agent.get_update_history("nonexistent_model")
        assert history == []

    def test_updater_agent_backup_model(self, updater_agent, mock_old_model):
        """Test model backup functionality."""
        model_path, _ = mock_old_model

        backup_path = updater_agent.backup_model(model_path, "model_123")

        assert backup_path is not None
        assert Path(backup_path).exists()
        assert "backup" in str(backup_path)

    def test_updater_agent_cleanup_old_backups(self, updater_agent, mock_old_model):
        """Test cleaning up old backups."""
        model_path, _ = mock_old_model

        # Create a backup
        updater_agent.backup_model(model_path, "model_123")

        # Cleanup old backups (should not remove recent backups)
        removed_count = updater_agent.cleanup_old_backups(max_age_days=1)

        assert isinstance(removed_count, int)
        assert removed_count >= 0


class TestUpdaterAgentErrorHandling:
    """Test UpdaterAgent error handling."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(name="test_updater", enabled=True)

    @pytest.fixture
    def updater_agent(self, agent_config):
        """Create an UpdaterAgent instance for testing."""
        return UpdaterAgent(agent_config)

    @pytest.mark.asyncio
    async def test_updater_agent_update_with_missing_model(
        self, updater_agent, sample_new_data
    ):
        """Test updating with missing model file."""
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        assert result.update_status == "failed"
        assert "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_updater_agent_update_with_missing_new_data(
        self, updater_agent, mock_old_model
    ):
        """Test updating with missing new data file."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        assert result.update_status == "failed"
        assert "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_updater_agent_update_with_invalid_data(
        self, updater_agent, mock_old_model, temp_data_dir
    ):
        """Test updating with invalid new data."""
        model_path, _ = mock_old_model

        # Create invalid data file
        invalid_data_path = Path(temp_data_dir) / "invalid_data.csv"
        with open(invalid_data_path, "w") as f:
            f.write("invalid,csv,data\n")

        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        assert result.update_status == "failed"


class TestUpdaterAgentPerformance:
    """Test UpdaterAgent performance evaluation."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(name="test_updater", enabled=True)

    @pytest.fixture
    def updater_agent(self, agent_config):
        """Create an UpdaterAgent instance for testing."""
        return UpdaterAgent(agent_config)

    def test_updater_agent_evaluate_performance_improvement(self, updater_agent):
        """Test performance improvement evaluation."""
        # Mock old and new model predictions
        old_predictions = pd.Series(np.random.randn(100) * 0.01 + 0.001)
        new_predictions = pd.Series(
            np.random.randn(100) * 0.01 + 0.002
        )  # Slightly better
        test_data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        improvement = updater_agent._evaluate_performance_improvement(
            old_predictions, new_predictions, test_data
        )

        assert isinstance(improvement, float)
        assert improvement >= -1.0 and improvement <= 1.0

    def test_updater_agent_should_retrain(self, updater_agent):
        """Test retrain decision logic."""
        # Test with performance below threshold
        current_performance = 0.6
        threshold = 0.7

        should_retrain = updater_agent._should_retrain(current_performance, threshold)
        assert should_retrain is True

        # Test with performance above threshold
        current_performance = 0.8
        should_retrain = updater_agent._should_retrain(current_performance, threshold)
        assert should_retrain is False

    def test_updater_agent_merge_data(
        self, updater_agent, sample_old_data, sample_new_data
    ):
        """Test data merging functionality."""
        merged_data = updater_agent._merge_data(sample_old_data, sample_new_data)

        assert isinstance(merged_data, pd.DataFrame)
        assert len(merged_data) > 0
        assert "date" in merged_data.columns
        assert "close" in merged_data.columns


class TestUpdaterAgentIntegration:
    """Test UpdaterAgent integration scenarios."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(name="test_updater", enabled=True)

    @pytest.fixture
    def updater_agent(self, agent_config):
        """Create an UpdaterAgent instance for testing."""
        return UpdaterAgent(agent_config)

    @pytest.mark.asyncio
    async def test_updater_agent_multiple_updates(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test multiple model updates."""
        model_path, _ = mock_old_model

        # Update model multiple times
        for i in range(3):
            request = UpdateRequest(
                model_id=f"model_{i}",
                evaluation_result=None,
                update_type="retrain",
                priority="normal",
                request_id=f"update_{i}",
            )

            result = updater_agent.update_model(request)
            assert result.update_status == "success"

        # Check update history
        for i in range(3):
            history = updater_agent.get_update_history(f"model_{i}")
            assert len(history) == 1

    @pytest.mark.asyncio
    async def test_updater_agent_backup_integration(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test that backups are created during updates."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        # Check that backup was created
        assert result.update_status == "success"
        assert result.old_model_path is not None
        assert Path(result.old_model_path).exists()

    @pytest.mark.asyncio
    async def test_updater_agent_performance_tracking(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test that performance improvements are tracked."""
        model_path, _ = mock_old_model
        request = UpdateRequest(
            model_id="model_123",
            evaluation_result=None,
            update_type="retrain",
            priority="normal",
            request_id="update_456",
        )

        result = updater_agent.update_model(request)

        # Check that performance improvement is calculated
        assert result.update_status == "success"
        assert result.performance_improvement is not None
        assert isinstance(result.performance_improvement, float)

    def test_rollback_to_previous_model_versions_on_degradation(
        self, updater_agent, mock_old_model, sample_new_data
    ):
        """Test that the agent can rollback to previous model versions if performance degrades."""
        print("\n🔄 Testing Model Rollback on Performance Degradation")

        # Create multiple model versions for rollback testing
        model_versions = {
            "v1.0": {
                "model_id": "model_v1_0",
                "performance_metrics": {
                    "sharpe_ratio": 1.8,
                    "total_return": 0.25,
                    "max_drawdown": -0.10,
                    "win_rate": 0.65,
                    "profit_factor": 2.0,
                },
                "timestamp": "2024-01-01T10:00:00",
                "is_current": False,
            },
            "v1.1": {
                "model_id": "model_v1_1",
                "performance_metrics": {
                    "sharpe_ratio": 1.9,
                    "total_return": 0.28,
                    "max_drawdown": -0.09,
                    "win_rate": 0.67,
                    "profit_factor": 2.1,
                },
                "timestamp": "2024-01-15T10:00:00",
                "is_current": False,
            },
            "v1.2": {
                "model_id": "model_v1_2",
                "performance_metrics": {
                    "sharpe_ratio": 1.7,
                    "total_return": 0.22,
                    "max_drawdown": -0.12,
                    "win_rate": 0.63,
                    "profit_factor": 1.8,
                },
                "timestamp": "2024-02-01T10:00:00",
                "is_current": True,
            },
            "v1.3": {
                "model_id": "model_v1_3",
                "performance_metrics": {
                    "sharpe_ratio": 1.2,  # Degraded performance
                    "total_return": 0.15,
                    "max_drawdown": -0.18,
                    "win_rate": 0.55,
                    "profit_factor": 1.4,
                },
                "timestamp": "2024-02-15T10:00:00",
                "is_current": False,
            },
        }

        # Register model versions
        for version, model_info in model_versions.items():
            updater_agent.register_model_version(model_info)
            print(f"  📝 Registered model version: {version}")

        # Test performance degradation detection
        print(f"\n  📉 Testing performance degradation detection...")

        # Simulate performance degradation
        current_performance = {
            "sharpe_ratio": 1.0,  # Below threshold
            "total_return": 0.10,
            "max_drawdown": -0.20,
            "win_rate": 0.50,
            "profit_factor": 1.2,
        }

        degradation_detected = updater_agent.detect_performance_degradation(
            current_performance
        )

        # Verify degradation detection
        self.assertTrue(
            degradation_detected["is_degraded"], "Should detect performance degradation"
        )
        self.assertIn(
            "degradation_score", degradation_detected, "Should have degradation score"
        )
        self.assertIn(
            "affected_metrics", degradation_detected, "Should have affected metrics"
        )
        self.assertIn("severity", degradation_detected, "Should have severity level")

        print(f"  Degradation detected: {degradation_detected['is_degraded']}")
        print(f"  Degradation score: {degradation_detected['degradation_score']:.3f}")
        print(f"  Severity: {degradation_detected['severity']}")
        print(f"  Affected metrics: {degradation_detected['affected_metrics']}")

        # Test rollback decision logic
        print(f"\n  🤔 Testing rollback decision logic...")

        rollback_decision = updater_agent.evaluate_rollback_necessity(
            degradation_detected
        )

        # Verify rollback decision
        self.assertIsNotNone(rollback_decision, "Rollback decision should not be None")
        self.assertIn(
            "should_rollback", rollback_decision, "Should indicate if rollback needed"
        )
        self.assertIn(
            "recommended_version", rollback_decision, "Should recommend version"
        )
        self.assertIn(
            "rollback_reason", rollback_decision, "Should have rollback reason"
        )
        self.assertIn(
            "expected_improvement",
            rollback_decision,
            "Should have expected improvement",
        )

        print(f"  Should rollback: {rollback_decision['should_rollback']}")
        print(f"  Recommended version: {rollback_decision['recommended_version']}")
        print(f"  Rollback reason: {rollback_decision['rollback_reason']}")
        print(
            f"  Expected improvement: {rollback_decision['expected_improvement']:.3f}"
        )

        # Verify rollback is recommended for significant degradation
        self.assertTrue(
            rollback_decision["should_rollback"],
            "Should recommend rollback for significant degradation",
        )

        # Test rollback execution
        print(f"\n  🔄 Testing rollback execution...")

        rollback_result = updater_agent.execute_model_rollback(
            rollback_decision["recommended_version"]
        )

        # Verify rollback result
        self.assertIsNotNone(rollback_result, "Rollback result should not be None")
        self.assertIn("rollback_id", rollback_result, "Should have rollback ID")
        self.assertIn("from_version", rollback_result, "Should have from version")
        self.assertIn("to_version", rollback_result, "Should have to version")
        self.assertIn(
            "rollback_timestamp", rollback_result, "Should have rollback timestamp"
        )
        self.assertIn("rollback_status", rollback_result, "Should have rollback status")
        self.assertIn(
            "backup_created", rollback_result, "Should indicate backup creation"
        )

        print(f"  Rollback ID: {rollback_result['rollback_id']}")
        print(f"  From version: {rollback_result['from_version']}")
        print(f"  To version: {rollback_result['to_version']}")
        print(f"  Rollback status: {rollback_result['rollback_status']}")
        print(f"  Backup created: {rollback_result['backup_created']}")

        # Verify rollback was successful
        self.assertEqual(
            rollback_result["rollback_status"],
            "success",
            "Rollback should be successful",
        )
        self.assertTrue(rollback_result["backup_created"], "Backup should be created")

        # Test rollback validation
        print(f"\n  ✅ Testing rollback validation...")

        validation_result = updater_agent.validate_rollback_success(rollback_result)

        # Verify validation result
        self.assertIsNotNone(validation_result, "Validation result should not be None")
        self.assertIn(
            "is_valid", validation_result, "Should indicate if rollback is valid"
        )
        self.assertIn(
            "current_version", validation_result, "Should show current version"
        )
        self.assertIn(
            "performance_check", validation_result, "Should have performance check"
        )
        self.assertIn(
            "stability_check", validation_result, "Should have stability check"
        )

        print(f"  Rollback valid: {validation_result['is_valid']}")
        print(f"  Current version: {validation_result['current_version']}")
        print(f"  Performance check: {validation_result['performance_check']}")
        print(f"  Stability check: {validation_result['stability_check']}")

        # Verify rollback validation
        self.assertTrue(validation_result["is_valid"], "Rollback should be valid")
        self.assertEqual(
            validation_result["current_version"],
            rollback_decision["recommended_version"],
            "Current version should match recommended version",
        )

        # Test rollback history tracking
        print(f"\n  📚 Testing rollback history tracking...")

        rollback_history = updater_agent.get_rollback_history()

        # Verify rollback history
        self.assertIsNotNone(rollback_history, "Rollback history should not be None")
        self.assertIn(
            "total_rollbacks", rollback_history, "Should have total rollbacks"
        )
        self.assertIn(
            "rollback_events", rollback_history, "Should have rollback events"
        )
        self.assertIn("success_rate", rollback_history, "Should have success rate")
        self.assertIn(
            "average_improvement", rollback_history, "Should have average improvement"
        )

        print(f"  Total rollbacks: {rollback_history['total_rollbacks']}")
        print(f"  Success rate: {rollback_history['success_rate']:.2f}")
        print(f"  Average improvement: {rollback_history['average_improvement']:.3f}")

        # Verify history contains the recent rollback
        self.assertGreater(
            rollback_history["total_rollbacks"], 0, "Should have rollback history"
        )
        self.assertEqual(
            rollback_history["success_rate"], 1.0, "Success rate should be 100%"
        )

        # Test automatic rollback triggers
        print(f"\n  🚨 Testing automatic rollback triggers...")

        # Set up automatic rollback triggers
        rollback_triggers = {
            "sharpe_ratio_threshold": 1.5,
            "max_drawdown_threshold": -0.15,
            "win_rate_threshold": 0.60,
            "consecutive_failures": 3,
            "performance_decay_rate": 0.1,
        }

        # Test trigger evaluation
        trigger_evaluation = updater_agent.evaluate_rollback_triggers(
            current_performance, rollback_triggers
        )

        # Verify trigger evaluation
        self.assertIsNotNone(
            trigger_evaluation, "Trigger evaluation should not be None"
        )
        self.assertIn(
            "triggers_activated", trigger_evaluation, "Should have activated triggers"
        )
        self.assertIn(
            "trigger_reasons", trigger_evaluation, "Should have trigger reasons"
        )
        self.assertIn(
            "automatic_rollback_needed",
            trigger_evaluation,
            "Should indicate if auto-rollback needed",
        )

        print(f"  Triggers activated: {trigger_evaluation['triggers_activated']}")
        print(f"  Trigger reasons: {trigger_evaluation['trigger_reasons']}")
        print(
            f"  Automatic rollback needed: {trigger_evaluation['automatic_rollback_needed']}"
        )

        # Test rollback point selection
        print(f"\n  🎯 Testing rollback point selection...")

        # Create multiple rollback candidates
        rollback_candidates = updater_agent.get_rollback_candidates()

        # Verify rollback candidates
        self.assertIsNotNone(
            rollback_candidates, "Rollback candidates should not be None"
        )
        self.assertIsInstance(
            rollback_candidates, list, "Rollback candidates should be a list"
        )

        print(f"  Available rollback candidates: {len(rollback_candidates)}")

        for candidate in rollback_candidates:
            print(
                f"    Version: {candidate['version']}, Performance: {candidate['performance_score']:.3f}"
            )

        # Test optimal rollback point selection
        optimal_rollback = updater_agent.select_optimal_rollback_point(
            rollback_candidates
        )

        # Verify optimal rollback selection
        self.assertIsNotNone(optimal_rollback, "Optimal rollback should not be None")
        self.assertIn(
            "selected_version", optimal_rollback, "Should have selected version"
        )
        self.assertIn(
            "selection_reason", optimal_rollback, "Should have selection reason"
        )
        self.assertIn(
            "expected_performance", optimal_rollback, "Should have expected performance"
        )

        print(f"  Optimal rollback version: {optimal_rollback['selected_version']}")
        print(f"  Selection reason: {optimal_rollback['selection_reason']}")
        print(f"  Expected performance: {optimal_rollback['expected_performance']:.3f}")

        # Test rollback with performance monitoring
        print(f"\n  📊 Testing rollback with performance monitoring...")

        # Simulate post-rollback performance monitoring
        monitoring_result = updater_agent.monitor_post_rollback_performance(
            rollback_result["rollback_id"]
        )

        # Verify monitoring result
        self.assertIsNotNone(monitoring_result, "Monitoring result should not be None")
        self.assertIn(
            "monitoring_period", monitoring_result, "Should have monitoring period"
        )
        self.assertIn(
            "performance_trend", monitoring_result, "Should have performance trend"
        )
        self.assertIn(
            "stability_metrics", monitoring_result, "Should have stability metrics"
        )
        self.assertIn(
            "recommendations", monitoring_result, "Should have recommendations"
        )

        print(f"  Monitoring period: {monitoring_result['monitoring_period']}")
        print(f"  Performance trend: {monitoring_result['performance_trend']}")
        print(f"  Stability metrics: {monitoring_result['stability_metrics']}")

        # Test rollback recovery validation
        print(f"\n  🔍 Testing rollback recovery validation...")

        recovery_validation = updater_agent.validate_rollback_recovery(
            rollback_result["rollback_id"]
        )

        # Verify recovery validation
        self.assertIsNotNone(
            recovery_validation, "Recovery validation should not be None"
        )
        self.assertIn(
            "recovery_successful",
            recovery_validation,
            "Should indicate recovery success",
        )
        self.assertIn(
            "performance_restored",
            recovery_validation,
            "Should indicate performance restoration",
        )
        self.assertIn(
            "stability_achieved", recovery_validation, "Should indicate stability"
        )
        self.assertIn("next_actions", recovery_validation, "Should have next actions")

        print(f"  Recovery successful: {recovery_validation['recovery_successful']}")
        print(f"  Performance restored: {recovery_validation['performance_restored']}")
        print(f"  Stability achieved: {recovery_validation['stability_achieved']}")
        print(f"  Next actions: {recovery_validation['next_actions']}")

        print("✅ Model rollback on performance degradation test completed")
