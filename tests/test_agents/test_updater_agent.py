"""
Tests for the UpdaterAgent

This module tests the UpdaterAgent functionality including
model updating, retraining, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import pickle
from datetime import datetime, timedelta

# Local imports
from trading.agents.updater_agent import (
    UpdaterAgent, ModelUpdateRequest, ModelUpdateResult
)
from trading.agents.base_agent_interface import AgentConfig


class TestModelUpdateRequest:
    """Test the ModelUpdateRequest dataclass."""
    
    def test_model_update_request_creation(self):
        """Test creating a ModelUpdateRequest instance."""
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path="models/model_123.pkl",
            model_type="lstm",
            new_data_path="data/new_data.csv",
            update_type="retrain",
            retrain_threshold=0.1,
            performance_threshold=0.8,
            request_id="update_456"
        )
        
        assert request.model_id == "model_123"
        assert request.model_path == "models/model_123.pkl"
        assert request.model_type == "lstm"
        assert request.new_data_path == "data/new_data.csv"
        assert request.update_type == "retrain"
        assert request.retrain_threshold == 0.1
        assert request.performance_threshold == 0.8
        assert request.request_id == "update_456"
    
    def test_model_update_request_defaults(self):
        """Test ModelUpdateRequest with default values."""
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path="models/model_123.pkl",
            model_type="lstm",
            new_data_path="data/new_data.csv"
        )
        
        assert request.model_id == "model_123"
        assert request.model_path == "models/model_123.pkl"
        assert request.model_type == "lstm"
        assert request.new_data_path == "data/new_data.csv"
        assert request.update_type == "incremental"
        assert request.retrain_threshold == 0.05
        assert request.performance_threshold == 0.7
        assert request.request_id is None


class TestModelUpdateResult:
    """Test the ModelUpdateResult dataclass."""
    
    def test_model_update_result_creation(self):
        """Test creating a ModelUpdateResult instance."""
        result = ModelUpdateResult(
            request_id="update_456",
            model_id="model_123",
            update_timestamp="2024-01-01T12:00:00",
            update_type="retrain",
            old_model_path="models/model_123_old.pkl",
            new_model_path="models/model_123_new.pkl",
            performance_improvement=0.15,
            training_metrics={"mse": 0.008, "mae": 0.04},
            update_status="success",
            error_message=None
        )
        
        assert result.request_id == "update_456"
        assert result.model_id == "model_123"
        assert result.update_timestamp == "2024-01-01T12:00:00"
        assert result.update_type == "retrain"
        assert result.old_model_path == "models/model_123_old.pkl"
        assert result.new_model_path == "models/model_123_new.pkl"
        assert result.performance_improvement == 0.15
        assert result.training_metrics == {"mse": 0.008, "mae": 0.04}
        assert result.update_status == "success"
        assert result.error_message is None
    
    def test_model_update_result_defaults(self):
        """Test ModelUpdateResult with default values."""
        result = ModelUpdateResult(
            request_id="update_456",
            model_id="model_123",
            update_timestamp="2024-01-01T12:00:00",
            update_type="retrain",
            old_model_path="models/model_123_old.pkl",
            new_model_path="models/model_123_new.pkl"
        )
        
        assert result.request_id == "update_456"
        assert result.model_id == "model_123"
        assert result.update_timestamp == "2024-01-01T12:00:00"
        assert result.update_type == "retrain"
        assert result.old_model_path == "models/model_123_old.pkl"
        assert result.new_model_path == "models/model_123_new.pkl"
        assert result.performance_improvement == 0.0
        assert result.training_metrics is None
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
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        data_path = Path(temp_data_dir) / "old_data.csv"
        data.to_csv(data_path, index=False)
        return str(data_path)
    
    @pytest.fixture
    def sample_new_data(self, temp_data_dir):
        """Create sample new data for testing."""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'open': np.random.randn(50).cumsum() + 110,
            'high': np.random.randn(50).cumsum() + 115,
            'low': np.random.randn(50).cumsum() + 105,
            'close': np.random.randn(50).cumsum() + 110,
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        data_path = Path(temp_data_dir) / "new_data.csv"
        data.to_csv(data_path, index=False)
        return str(data_path)
    
    @pytest.fixture
    def mock_old_model(self, temp_data_dir):
        """Create a mock old model for testing."""
        model = Mock()
        model.predict.return_value = np.random.randn(100) * 0.01 + 0.001
        
        model_path = Path(temp_data_dir) / "old_model.pkl"
        with open(model_path, 'wb') as f:
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
                "performance_decay_threshold": 0.1
            }
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
    
    def test_updater_agent_validate_input_valid(self, updater_agent, mock_old_model, sample_new_data):
        """Test input validation with valid data."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data
        )
        
        assert updater_agent.validate_input(request=request) is True
    
    def test_updater_agent_validate_input_invalid_model_path(self, updater_agent, sample_new_data):
        """Test input validation with invalid model path."""
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path="nonexistent_model.pkl",
            model_type="lstm",
            new_data_path=sample_new_data
        )
        
        assert updater_agent.validate_input(request=request) is False
    
    def test_updater_agent_validate_input_invalid_new_data_path(self, updater_agent, mock_old_model):
        """Test input validation with invalid new data path."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path="nonexistent_data.csv"
        )
        
        assert updater_agent.validate_input(request=request) is False
    
    def test_updater_agent_validate_input_invalid_update_type(self, updater_agent, mock_old_model, sample_new_data):
        """Test input validation with invalid update type."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data,
            update_type="invalid_type"
        )
        
        assert updater_agent.validate_input(request=request) is False
    
    def test_updater_agent_validate_input_missing_request(self, updater_agent):
        """Test input validation with missing request."""
        assert updater_agent.validate_input() is False
    
    def test_updater_agent_validate_input_wrong_type(self, updater_agent):
        """Test input validation with wrong request type."""
        assert updater_agent.validate_input(request="not_a_request") is False
    
    @pytest.mark.asyncio
    async def test_updater_agent_execute_success(self, updater_agent, mock_old_model, sample_new_data):
        """Test successful model update execution."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data,
            update_type="retrain"
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
    async def test_updater_agent_update_lstm_model(self, updater_agent, mock_old_model, sample_new_data):
        """Test LSTM model updating."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data,
            update_type="retrain"
        )
        
        result = updater_agent.update_model(request)
        
        assert result.update_status == "success"
        assert result.model_id == "model_123"
        assert result.update_type == "retrain"
        assert result.new_model_path is not None
        assert Path(result.new_model_path).exists()
        assert result.old_model_path == model_path
    
    @pytest.mark.asyncio
    async def test_updater_agent_update_xgboost_model(self, updater_agent, mock_old_model, sample_new_data):
        """Test XGBoost model updating."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="xgboost",
            new_data_path=sample_new_data,
            update_type="retrain"
        )
        
        result = updater_agent.update_model(request)
        
        assert result.update_status == "success"
        assert result.model_id == "model_123"
        assert result.update_type == "retrain"
        assert result.new_model_path is not None
        assert Path(result.new_model_path).exists()
    
    @pytest.mark.asyncio
    async def test_updater_agent_incremental_update(self, updater_agent, mock_old_model, sample_new_data):
        """Test incremental model update."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data,
            update_type="incremental"
        )
        
        result = updater_agent.update_model(request)
        
        assert result.update_status == "success"
        assert result.model_id == "model_123"
        assert result.update_type == "incremental"
        assert result.new_model_path is not None
        assert Path(result.new_model_path).exists()
    
    @pytest.mark.asyncio
    async def test_updater_agent_update_with_performance_threshold(self, updater_agent, mock_old_model, sample_new_data):
        """Test model update with performance threshold."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data,
            update_type="retrain",
            performance_threshold=0.9  # High threshold
        )
        
        result = updater_agent.update_model(request)
        
        assert result.update_status == "success"
        assert result.model_id == "model_123"
        assert result.performance_improvement is not None
    
    @pytest.mark.asyncio
    async def test_updater_agent_update_invalid_model_type(self, updater_agent, mock_old_model, sample_new_data):
        """Test updating with invalid model type."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="invalid_type",
            new_data_path=sample_new_data
        )
        
        result = updater_agent.update_model(request)
        
        assert result.update_status == "failed"
        assert "unsupported" in result.error_message.lower()
    
    def test_updater_agent_get_update_history(self, updater_agent, mock_old_model, sample_new_data):
        """Test getting update history."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data,
            update_type="retrain"
        )
        
        # Update model first
        result = updater_agent.update_model(request)
        
        # Get history
        history = updater_agent.get_update_history("model_123")
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert all(isinstance(update_result, ModelUpdateResult) for update_result in history)
    
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
        backup_path = updater_agent.backup_model(model_path, "model_123")
        
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
    async def test_updater_agent_update_with_missing_model(self, updater_agent, sample_new_data):
        """Test updating with missing model file."""
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path="nonexistent_model.pkl",
            model_type="lstm",
            new_data_path=sample_new_data
        )
        
        result = updater_agent.update_model(request)
        
        assert result.update_status == "failed"
        assert "error" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_updater_agent_update_with_missing_new_data(self, updater_agent, mock_old_model):
        """Test updating with missing new data file."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path="nonexistent_data.csv"
        )
        
        result = updater_agent.update_model(request)
        
        assert result.update_status == "failed"
        assert "error" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_updater_agent_update_with_invalid_data(self, updater_agent, mock_old_model, temp_data_dir):
        """Test updating with invalid new data."""
        model_path, _ = mock_old_model
        
        # Create invalid data file
        invalid_data_path = Path(temp_data_dir) / "invalid_data.csv"
        with open(invalid_data_path, 'w') as f:
            f.write("invalid,csv,data\n")
        
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=str(invalid_data_path)
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
        new_predictions = pd.Series(np.random.randn(100) * 0.01 + 0.002)  # Slightly better
        test_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        })
        
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
    
    def test_updater_agent_merge_data(self, updater_agent, sample_old_data, sample_new_data):
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
    async def test_updater_agent_multiple_updates(self, updater_agent, mock_old_model, sample_new_data):
        """Test multiple model updates."""
        model_path, _ = mock_old_model
        
        # Update model multiple times
        for i in range(3):
            request = ModelUpdateRequest(
                model_id=f"model_{i}",
                model_path=model_path,
                model_type="lstm",
                new_data_path=sample_new_data,
                update_type="retrain"
            )
            
            result = updater_agent.update_model(request)
            assert result.update_status == "success"
        
        # Check update history
        for i in range(3):
            history = updater_agent.get_update_history(f"model_{i}")
            assert len(history) == 1
    
    @pytest.mark.asyncio
    async def test_updater_agent_backup_integration(self, updater_agent, mock_old_model, sample_new_data):
        """Test that backups are created during updates."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data,
            update_type="retrain"
        )
        
        result = updater_agent.update_model(request)
        
        # Check that backup was created
        assert result.update_status == "success"
        assert result.old_model_path is not None
        assert Path(result.old_model_path).exists()
    
    @pytest.mark.asyncio
    async def test_updater_agent_performance_tracking(self, updater_agent, mock_old_model, sample_new_data):
        """Test that performance improvements are tracked."""
        model_path, _ = mock_old_model
        request = ModelUpdateRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            new_data_path=sample_new_data,
            update_type="retrain"
        )
        
        result = updater_agent.update_model(request)
        
        # Check that performance improvement is calculated
        assert result.update_status == "success"
        assert result.performance_improvement is not None
        assert isinstance(result.performance_improvement, float) 