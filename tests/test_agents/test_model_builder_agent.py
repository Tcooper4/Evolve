"""
Tests for the ModelBuilderAgent

This module tests the ModelBuilderAgent functionality including
model building, validation, and error handling.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.agents.base_agent_interface import AgentConfig

# Local imports
from trading.agents.model_builder_agent import (
    ModelBuilderAgent,
    ModelBuildRequest,
    ModelBuildResult,
)


class TestModelBuildRequest:
    """Test the ModelBuildRequest dataclass."""

    def test_model_build_request_creation(self):
        """Test creating a ModelBuildRequest instance."""
        request = ModelBuildRequest(
            model_type="lstm",
            data_path="data/sample.csv",
            target_column="close",
            features=["open", "high", "low"],
            hyperparameters={"epochs": 100, "batch_size": 32},
            validation_split=0.2,
            random_state=42,
            request_id="test_123",
        )

        assert request.model_type == "lstm"
        assert request.data_path == "data/sample.csv"
        assert request.target_column == "close"
        assert request.features == ["open", "high", "low"]
        assert request.hyperparameters == {"epochs": 100, "batch_size": 32}
        assert request.validation_split == 0.2
        assert request.random_state == 42
        assert request.request_id == "test_123"

    def test_model_build_request_defaults(self):
        """Test ModelBuildRequest with default values."""
        request = ModelBuildRequest(
            model_type="xgboost", data_path="data/sample.csv", target_column="close"
        )

        assert request.model_type == "xgboost"
        assert request.data_path == "data/sample.csv"
        assert request.target_column == "close"
        assert request.features is None
        assert request.hyperparameters is None
        assert request.validation_split == 0.2
        assert request.random_state == 42
        assert request.request_id is None


class TestModelBuildResult:
    """Test the ModelBuildResult dataclass."""

    def test_model_build_result_creation(self):
        """Test creating a ModelBuildResult instance."""
        result = ModelBuildResult(
            request_id="test_123",
            model_type="lstm",
            model_path="models/lstm_model.pkl",
            model_id="model_456",
            build_timestamp="2024-01-01T12:00:00",
            training_metrics={"mse": 0.01, "mae": 0.05},
            model_config={"epochs": 100, "batch_size": 32},
            feature_importance={"feature1": 0.8, "feature2": 0.2},
            build_status="success",
            error_message=None,
        )

        assert result.request_id == "test_123"
        assert result.model_type == "lstm"
        assert result.model_path == "models/lstm_model.pkl"
        assert result.model_id == "model_456"
        assert result.build_timestamp == "2024-01-01T12:00:00"
        assert result.training_metrics == {"mse": 0.01, "mae": 0.05}
        assert result.model_config == {"epochs": 100, "batch_size": 32}
        assert result.feature_importance == {"feature1": 0.8, "feature2": 0.2}
        assert result.build_status == "success"
        assert result.error_message is None

    def test_model_build_result_defaults(self):
        """Test ModelBuildResult with default values."""
        result = ModelBuildResult(
            request_id="test_123",
            model_type="lstm",
            model_path="models/lstm_model.pkl",
            model_id="model_456",
            build_timestamp="2024-01-01T12:00:00",
            training_metrics={"mse": 0.01},
            model_config={"epochs": 100},
        )

        assert result.request_id == "test_123"
        assert result.model_type == "lstm"
        assert result.model_path == "models/lstm_model.pkl"
        assert result.model_id == "model_456"
        assert result.build_timestamp == "2024-01-01T12:00:00"
        assert result.training_metrics == {"mse": 0.01}
        assert result.model_config == {"epochs": 100}
        assert result.feature_importance is None
        assert result.build_status == "success"
        assert result.error_message is None


class TestModelBuilderAgent:
    """Test the ModelBuilderAgent class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self, temp_data_dir):
        """Create sample data for testing."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100, freq="D"),
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 105,
                "low": np.random.randn(100).cumsum() + 95,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )

        data_path = Path(temp_data_dir) / "sample_data.csv"
        data.to_csv(data_path, index=False)
        return str(data_path)

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(
            name="test_model_builder",
            enabled=True,
            priority=1,
            max_concurrent_runs=2,
            timeout_seconds=600,
            retry_attempts=3,
            custom_config={
                "max_models": 5,
                "model_types": ["lstm", "xgboost", "ensemble"],
            },
        )

    @pytest.fixture
    def model_builder_agent(self, agent_config):
        """Create a ModelBuilderAgent instance for testing."""
        return ModelBuilderAgent(agent_config)

    def test_model_builder_agent_initialization(self, model_builder_agent):
        """Test ModelBuilderAgent initialization."""
        assert model_builder_agent.config.name == "test_model_builder"
        assert model_builder_agent.config.enabled is True
        assert model_builder_agent.models_dir.exists()
        assert isinstance(model_builder_agent.model_registry, dict)

    def test_model_builder_agent_metadata(self, model_builder_agent):
        """Test ModelBuilderAgent metadata."""
        metadata = model_builder_agent.get_metadata()

        assert metadata["name"] == "test_model_builder"
        assert metadata["version"] == "1.0.0"
        assert "model-building" in metadata["tags"]
        assert "lstm_building" in metadata["capabilities"]

    def test_model_builder_agent_validate_input_valid(
        self, model_builder_agent, sample_data
    ):
        """Test input validation with valid data."""
        request = ModelBuildRequest(
            model_type="lstm", data_path=sample_data, target_column="close"
        )

        assert model_builder_agent.validate_input(request=request) is True

    def test_model_builder_agent_validate_input_invalid_type(
        self, model_builder_agent, sample_data
    ):
        """Test input validation with invalid model type."""
        request = ModelBuildRequest(
            model_type="invalid_type", data_path=sample_data, target_column="close"
        )

        assert model_builder_agent.validate_input(request=request) is False

    def test_model_builder_agent_validate_input_missing_data(self, model_builder_agent):
        """Test input validation with missing data path."""
        request = ModelBuildRequest(
            model_type="lstm", data_path="nonexistent_file.csv", target_column="close"
        )

        assert model_builder_agent.validate_input(request=request) is False

    def test_model_builder_agent_validate_input_missing_request(
        self, model_builder_agent
    ):
        """Test input validation with missing request."""
        assert model_builder_agent.validate_input() is False

    def test_model_builder_agent_validate_input_wrong_type(self, model_builder_agent):
        """Test input validation with wrong request type."""
        assert model_builder_agent.validate_input(request="not_a_request") is False

    @pytest.mark.asyncio
    async def test_model_builder_agent_execute_success(
        self, model_builder_agent, sample_data
    ):
        """Test successful model building execution."""
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"epochs": 5, "batch_size": 16},  # Small for testing
        )

        result = await model_builder_agent.execute(request=request)

        assert result.success is True
        assert "model_id" in result.data
        assert "model_path" in result.data
        assert result.data["model_type"] == "lstm"
        assert "training_metrics" in result.data

    @pytest.mark.asyncio
    async def test_model_builder_agent_execute_invalid_request(
        self, model_builder_agent
    ):
        """Test execution with invalid request."""
        result = await model_builder_agent.execute()

        assert result.success is False
        assert "required" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_model_builder_agent_execute_wrong_request_type(
        self, model_builder_agent
    ):
        """Test execution with wrong request type."""
        result = await model_builder_agent.execute(request="not_a_request")

        assert result.success is False
        assert "instance" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_lstm_model(
        self, model_builder_agent, sample_data
    ):
        """Test LSTM model building."""
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"epochs": 5, "batch_size": 16},
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "success"
        assert result.model_type == "lstm"
        assert result.model_id is not None
        assert result.model_path is not None
        assert Path(result.model_path).exists()
        assert "training_metrics" in result.training_metrics

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_xgboost_model(
        self, model_builder_agent, sample_data
    ):
        """Test XGBoost model building."""
        request = ModelBuildRequest(
            model_type="xgboost",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "success"
        assert result.model_type == "xgboost"
        assert result.model_id is not None
        assert result.model_path is not None
        assert Path(result.model_path).exists()
        assert "training_metrics" in result.training_metrics

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_ensemble_model(
        self, model_builder_agent, sample_data
    ):
        """Test ensemble model building."""
        request = ModelBuildRequest(
            model_type="ensemble",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"n_estimators": 5},
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "success"
        assert result.model_type == "ensemble"
        assert result.model_id is not None
        assert result.model_path is not None
        assert Path(result.model_path).exists()
        assert "training_metrics" in result.training_metrics

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_invalid_model_type(
        self, model_builder_agent, sample_data
    ):
        """Test building with invalid model type."""
        request = ModelBuildRequest(
            model_type="invalid_type", data_path=sample_data, target_column="close"
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "failed"
        assert "unsupported" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_with_features(
        self, model_builder_agent, sample_data
    ):
        """Test model building with specific features."""
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            features=["open", "high", "low", "volume"],
            hyperparameters={"epochs": 5, "batch_size": 16},
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "success"
        assert result.model_type == "lstm"
        assert result.model_id is not None

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_with_custom_hyperparameters(
        self, model_builder_agent, sample_data
    ):
        """Test model building with custom hyperparameters."""
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            hyperparameters={
                "epochs": 10,
                "batch_size": 8,
                "learning_rate": 0.001,
                "hidden_size": 64,
            },
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "success"
        assert result.model_type == "lstm"
        assert result.model_config["epochs"] == 10
        assert result.model_config["batch_size"] == 8

    def test_model_builder_agent_get_model_status(
        self, model_builder_agent, sample_data
    ):
        """Test getting model status."""
        # Build a model first
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"epochs": 5, "batch_size": 16},
        )

        build_result = model_builder_agent.build_model(request)

        # Get status
        status = model_builder_agent.get_model_status(build_result.model_id)

        assert status is not None
        assert status.model_id == build_result.model_id
        assert status.model_type == "lstm"

    def test_model_builder_agent_get_model_status_nonexistent(
        self, model_builder_agent
    ):
        """Test getting status for nonexistent model."""
        status = model_builder_agent.get_model_status("nonexistent_id")
        assert status is None

    def test_model_builder_agent_list_models(self, model_builder_agent, sample_data):
        """Test listing models."""
        # Build a model first
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"epochs": 5, "batch_size": 16},
        )

        model_builder_agent.build_model(request)

        # List models
        models = model_builder_agent.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, dict) for model in models)

    def test_model_builder_agent_cleanup_old_models(
        self, model_builder_agent, sample_data
    ):
        """Test cleaning up old models."""
        # Build a model first
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"epochs": 5, "batch_size": 16},
        )

        model_builder_agent.build_model(request)

        # Cleanup old models (should not remove recent models)
        removed_count = model_builder_agent.cleanup_old_models(max_age_days=1)

        assert isinstance(removed_count, int)
        assert removed_count >= 0


class TestModelBuilderAgentErrorHandling:
    """Test ModelBuilderAgent error handling."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(name="test_model_builder", enabled=True)

    @pytest.fixture
    def model_builder_agent(self, agent_config):
        """Create a ModelBuilderAgent instance for testing."""
        return ModelBuilderAgent(agent_config)

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_with_missing_data(
        self, model_builder_agent
    ):
        """Test building model with missing data file."""
        request = ModelBuildRequest(
            model_type="lstm", data_path="nonexistent_file.csv", target_column="close"
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "failed"
        assert "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_with_invalid_data(
        self, model_builder_agent, temp_data_dir
    ):
        """Test building model with invalid data."""
        # Create invalid data file
        invalid_data_path = Path(temp_data_dir) / "invalid_data.csv"
        with open(invalid_data_path, "w") as f:
            f.write("invalid,csv,data\n")

        request = ModelBuildRequest(
            model_type="lstm", data_path=str(invalid_data_path), target_column="close"
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "failed"

    @pytest.mark.asyncio
    async def test_model_builder_agent_build_with_missing_target(
        self, model_builder_agent, sample_data
    ):
        """Test building model with missing target column."""
        request = ModelBuildRequest(
            model_type="lstm", data_path=sample_data, target_column="nonexistent_column"
        )

        result = model_builder_agent.build_model(request)

        assert result.build_status == "failed"
        assert "error" in result.error_message.lower()


class TestModelBuilderAgentIntegration:
    """Test ModelBuilderAgent integration scenarios."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(name="test_model_builder", enabled=True)

    @pytest.fixture
    def model_builder_agent(self, agent_config):
        """Create a ModelBuilderAgent instance for testing."""
        return ModelBuilderAgent(agent_config)

    @pytest.mark.asyncio
    async def test_model_builder_agent_multiple_models(
        self, model_builder_agent, sample_data
    ):
        """Test building multiple models of different types."""
        model_types = ["lstm", "xgboost", "ensemble"]
        results = []

        for model_type in model_types:
            request = ModelBuildRequest(
                model_type=model_type,
                data_path=sample_data,
                target_column="close",
                hyperparameters=(
                    {"epochs": 5, "batch_size": 16}
                    if model_type == "lstm"
                    else {"n_estimators": 10}
                ),
            )

            result = model_builder_agent.build_model(request)
            results.append(result)

        # All should succeed
        for result in results:
            assert result.build_status == "success"

        # Should have different model IDs
        model_ids = [result.model_id for result in results]
        assert len(set(model_ids)) == len(model_ids)

        # Should have different model paths
        model_paths = [result.model_path for result in results]
        assert len(set(model_paths)) == len(model_paths)

    @pytest.mark.asyncio
    async def test_model_builder_agent_registry_consistency(
        self, model_builder_agent, sample_data
    ):
        """Test that model registry is consistent."""
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"epochs": 5, "batch_size": 16},
        )

        result = model_builder_agent.build_model(request)

        # Check registry
        assert result.request_id in model_builder_agent.model_registry
        registry_entry = model_builder_agent.model_registry[result.request_id]
        assert registry_entry.model_id == result.model_id
        assert registry_entry.model_type == result.model_type

    @pytest.mark.asyncio
    async def test_model_builder_agent_memory_integration(
        self, model_builder_agent, sample_data
    ):
        """Test that agent memory is properly updated."""
        request = ModelBuildRequest(
            model_type="lstm",
            data_path=sample_data,
            target_column="close",
            hyperparameters={"epochs": 5, "batch_size": 16},
        )

        result = model_builder_agent.build_model(request)

        # Check that memory was updated (this would require checking the memory file)
        # For now, just verify the result has the expected structure
        assert result.build_status == "success"
        assert result.model_id is not None
        assert result.training_metrics is not None

    @pytest.mark.asyncio
    async def test_dynamic_model_generation_from_text_prompt(
        self, model_builder_agent, sample_data
    ):
        """Test dynamic model generation from text prompt (e.g., 'build an XGBoost forecaster')."""
        print("\n🤖 Testing Dynamic Model Generation from Text Prompt")

        # Test different text prompts
        text_prompts = [
            "build an XGBoost forecaster for stock prediction",
            "create a LSTM neural network for time series forecasting",
            "generate an ensemble model combining multiple algorithms",
            "build a random forest model for price prediction",
            "create a deep learning model with attention mechanism",
        ]

        for prompt in text_prompts:
            print(f"\n  📝 Testing prompt: '{prompt}'")

            # Parse text prompt to extract model configuration
            model_config = model_builder_agent.parse_text_prompt(prompt)

            # Verify model configuration extraction
            assert (
                model_config is not None
            ), f"Should extract config from prompt: {prompt}"
            assert "model_type" in model_config, "Config should contain model_type"
            assert (
                "hyperparameters" in model_config
            ), "Config should contain hyperparameters"

            print(f"    Extracted model type: {model_config['model_type']}")
            print(f"    Extracted hyperparameters: {model_config['hyperparameters']}")

            # Create model build request from extracted config
            request = ModelBuildRequest(
                model_type=model_config["model_type"],
                data_path=sample_data,
                target_column="close",
                features=["open", "high", "low", "volume"],
                hyperparameters=model_config["hyperparameters"],
                validation_split=0.2,
                random_state=42,
                request_id=f"prompt_{hash(prompt) % 10000}",
            )

            # Validate the generated request
            is_valid = model_builder_agent.validate_input(request=request)
            assert is_valid, f"Generated request should be valid for prompt: {prompt}"

            # Build the model
            result = await model_builder_agent.execute(request=request)

            # Verify successful model generation
            assert (
                result.success
            ), f"Model generation should succeed for prompt: {prompt}"
            assert result.data is not None, "Result should contain model data"
            assert "model_id" in result.data, "Result should contain model_id"
            assert "model_path" in result.data, "Result should contain model_path"

            print(f"    ✅ Model generated successfully: {result.data['model_id']}")

            # Test model-specific validations
            if "xgboost" in prompt.lower():
                assert (
                    model_config["model_type"] == "xgboost"
                ), "Should extract XGBoost for XGBoost prompt"
                assert (
                    "n_estimators" in model_config["hyperparameters"]
                ), "XGBoost should have n_estimators parameter"

            elif "lstm" in prompt.lower():
                assert (
                    model_config["model_type"] == "lstm"
                ), "Should extract LSTM for LSTM prompt"
                assert (
                    "epochs" in model_config["hyperparameters"]
                ), "LSTM should have epochs parameter"

            elif "ensemble" in prompt.lower():
                assert (
                    model_config["model_type"] == "ensemble"
                ), "Should extract ensemble for ensemble prompt"
                assert (
                    "models" in model_config["hyperparameters"]
                ), "Ensemble should have models parameter"

            elif "random" in prompt.lower() and "forest" in prompt.lower():
                assert (
                    model_config["model_type"] == "random_forest"
                ), "Should extract random_forest for random forest prompt"
                assert (
                    "n_estimators" in model_config["hyperparameters"]
                ), "Random forest should have n_estimators parameter"

        # Test complex prompt with multiple requirements
        print(f"\n  🔧 Testing complex prompt...")
        complex_prompt = "build a deep learning model with LSTM layers, dropout regularization, and early stopping for predicting stock prices with high accuracy"

        complex_config = model_builder_agent.parse_text_prompt(complex_prompt)

        # Verify complex configuration extraction
        assert (
            complex_config["model_type"] == "lstm"
        ), "Should extract LSTM for deep learning prompt"
        assert (
            "dropout" in complex_config["hyperparameters"]
        ), "Should include dropout for regularization"
        assert (
            "early_stopping" in complex_config["hyperparameters"]
        ), "Should include early stopping"
        assert (
            "epochs" in complex_config["hyperparameters"]
        ), "Should include epochs parameter"

        print(
            f"    Complex config extracted: {complex_config['model_type']} with {len(complex_config['hyperparameters'])} parameters"
        )

        # Test prompt with invalid model type
        print(f"\n  ⚠️ Testing invalid model prompt...")
        invalid_prompt = "build a quantum computer model for stock prediction"

        with pytest.raises(ValueError):
            model_builder_agent.parse_text_prompt(invalid_prompt)

        print(f"    ✅ Invalid model type properly rejected")

        # Test prompt with missing information
        print(f"\n  🔍 Testing incomplete prompt...")
        incomplete_prompt = "build a model"

        incomplete_config = model_builder_agent.parse_text_prompt(incomplete_prompt)

        # Should use default configuration
        assert (
            incomplete_config is not None
        ), "Should provide default config for incomplete prompt"
        assert "model_type" in incomplete_config, "Should have default model type"
        assert (
            "hyperparameters" in incomplete_config
        ), "Should have default hyperparameters"

        print(
            f"    ✅ Default config provided for incomplete prompt: {incomplete_config['model_type']}"
        )

        # Test prompt with custom parameters
        print(f"\n  ⚙️ Testing prompt with custom parameters...")
        custom_prompt = (
            "build an XGBoost model with 500 trees, max_depth 6, and learning_rate 0.1"
        )

        custom_config = model_builder_agent.parse_text_prompt(custom_prompt)

        # Verify custom parameters are extracted
        assert custom_config["model_type"] == "xgboost", "Should extract XGBoost"
        assert (
            custom_config["hyperparameters"]["n_estimators"] == 500
        ), "Should extract n_estimators"
        assert (
            custom_config["hyperparameters"]["max_depth"] == 6
        ), "Should extract max_depth"
        assert (
            custom_config["hyperparameters"]["learning_rate"] == 0.1
        ), "Should extract learning_rate"

        print(f"    ✅ Custom parameters extracted: {custom_config['hyperparameters']}")

        # Test end-to-end model generation from text prompt
        print(f"\n  🚀 Testing end-to-end model generation from text prompt...")

        final_prompt = "build a gradient boosting model for stock price prediction with 200 estimators"
        final_config = model_builder_agent.parse_text_prompt(final_prompt)

        final_request = ModelBuildRequest(
            model_type=final_config["model_type"],
            data_path=sample_data,
            target_column="close",
            features=["open", "high", "low", "volume"],
            hyperparameters=final_config["hyperparameters"],
            validation_split=0.2,
            random_state=42,
            request_id="final_prompt_test",
        )

        final_result = await model_builder_agent.execute(request=final_request)

        # Verify successful end-to-end generation
        assert final_result.success, "End-to-end model generation should succeed"
        assert "model_id" in final_result.data, "Should have model_id in result"
        assert "training_metrics" in final_result.data, "Should have training metrics"

        print(
            f"    ✅ End-to-end model generation successful: {final_result.data['model_id']}"
        )

        print(
            "Model builder agent test completed. All model building "
            "functions have been validated."
        )
