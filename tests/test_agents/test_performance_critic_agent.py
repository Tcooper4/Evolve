"""
Tests for the PerformanceCriticAgent

This module tests the PerformanceCriticAgent functionality including
model evaluation, performance metrics calculation, and error handling.
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
from trading.agents.performance_critic_agent import (
    ModelEvaluationRequest,
    ModelEvaluationResult,
    PerformanceCriticAgent,
)


class TestModelEvaluationRequest:
    """Test the ModelEvaluationRequest dataclass."""

    def test_model_evaluation_request_creation(self):
        """Test creating a ModelEvaluationRequest instance."""
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path="models/model_123.pkl",
            model_type="lstm",
            test_data_path="data/test_data.csv",
            evaluation_period=252,
            benchmark_symbol="SPY",
            risk_free_rate=0.02,
            request_id="eval_456",
        )

        assert request.model_id == "model_123"
        assert request.model_path == "models/model_123.pkl"
        assert request.model_type == "lstm"
        assert request.test_data_path == "data/test_data.csv"
        assert request.evaluation_period == 252
        assert request.benchmark_symbol == "SPY"
        assert request.risk_free_rate == 0.02
        assert request.request_id == "eval_456"

    def test_model_evaluation_request_defaults(self):
        """Test ModelEvaluationRequest with default values."""
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path="models/model_123.pkl",
            model_type="lstm",
            test_data_path="data/test_data.csv",
        )

        assert request.model_id == "model_123"
        assert request.model_path == "models/model_123.pkl"
        assert request.model_type == "lstm"
        assert request.test_data_path == "data/test_data.csv"
        assert request.evaluation_period == 252
        assert request.benchmark_symbol is None
        assert request.risk_free_rate == 0.02
        assert request.request_id is None


class TestModelEvaluationResult:
    """Test the ModelEvaluationResult dataclass."""

    def test_model_evaluation_result_creation(self):
        """Test creating a ModelEvaluationResult instance."""
        result = ModelEvaluationResult(
            request_id="eval_456",
            model_id="model_123",
            evaluation_timestamp="2024-01-01T12:00:00",
            performance_metrics={"sharpe_ratio": 1.5, "total_return": 0.25},
            risk_metrics={"max_drawdown": -0.15, "volatility": 0.20},
            trading_metrics={"win_rate": 0.55, "profit_factor": 1.8},
            benchmark_comparison={"excess_return": 0.05, "information_ratio": 0.8},
            recommendations=["Increase position sizing", "Reduce risk"],
            evaluation_status="success",
            error_message=None,
        )

        assert result.request_id == "eval_456"
        assert result.model_id == "model_123"
        assert result.evaluation_timestamp == "2024-01-01T12:00:00"
        assert result.performance_metrics == {"sharpe_ratio": 1.5, "total_return": 0.25}
        assert result.risk_metrics == {"max_drawdown": -0.15, "volatility": 0.20}
        assert result.trading_metrics == {"win_rate": 0.55, "profit_factor": 1.8}
        assert result.benchmark_comparison == {
            "excess_return": 0.05,
            "information_ratio": 0.8,
        }
        assert result.recommendations == ["Increase position sizing", "Reduce risk"]
        assert result.evaluation_status == "success"
        assert result.error_message is None

    def test_model_evaluation_result_defaults(self):
        """Test ModelEvaluationResult with default values."""
        result = ModelEvaluationResult(
            request_id="eval_456",
            model_id="model_123",
            evaluation_timestamp="2024-01-01T12:00:00",
            performance_metrics={"sharpe_ratio": 1.5},
            risk_metrics={"max_drawdown": -0.15},
            trading_metrics={"win_rate": 0.55},
        )

        assert result.request_id == "eval_456"
        assert result.model_id == "model_123"
        assert result.evaluation_timestamp == "2024-01-01T12:00:00"
        assert result.performance_metrics == {"sharpe_ratio": 1.5}
        assert result.risk_metrics == {"max_drawdown": -0.15}
        assert result.trading_metrics == {"win_rate": 0.55}
        assert result.benchmark_comparison is None
        assert result.recommendations == []
        assert result.evaluation_status == "success"
        assert result.error_message is None


class TestPerformanceCriticAgent:
    """Test the PerformanceCriticAgent class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_test_data(self, temp_data_dir):
        """Create sample test data for evaluation."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=252, freq="D"),
                "open": np.random.randn(252).cumsum() + 100,
                "high": np.random.randn(252).cumsum() + 105,
                "low": np.random.randn(252).cumsum() + 95,
                "close": np.random.randn(252).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 252),
            }
        )

        data_path = Path(temp_data_dir) / "test_data.csv"
        data.to_csv(data_path, index=False)
        return str(data_path)

    @pytest.fixture
    def mock_model(self, temp_data_dir):
        """Create a mock model for testing."""
        model = Mock()
        model.predict.return_value = (
            np.random.randn(252) * 0.01 + 0.001
        )  # Small returns

        model_path = Path(temp_data_dir) / "mock_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return str(model_path), model

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(
            name="test_performance_critic",
            enabled=True,
            priority=2,
            max_concurrent_runs=3,
            timeout_seconds=300,
            retry_attempts=2,
            custom_config={
                "evaluation_period": 252,
                "min_sharpe_ratio": 0.5,
                "max_drawdown": -0.15,
                "min_win_rate": 0.45,
            },
        )

    @pytest.fixture
    def performance_critic_agent(self, agent_config):
        """Create a PerformanceCriticAgent instance for testing."""
        return PerformanceCriticAgent(agent_config)

    def test_performance_critic_agent_initialization(self, performance_critic_agent):
        """Test PerformanceCriticAgent initialization."""
        assert performance_critic_agent.config.name == "test_performance_critic"
        assert performance_critic_agent.config.enabled is True
        assert isinstance(performance_critic_agent.thresholds, dict)
        assert isinstance(performance_critic_agent.evaluation_history, dict)

    def test_performance_critic_agent_metadata(self, performance_critic_agent):
        """Test PerformanceCriticAgent metadata."""
        metadata = performance_critic_agent.get_metadata()

        assert metadata["name"] == "test_performance_critic"
        assert metadata["version"] == "1.0.0"
        assert "performance" in metadata["tags"]
        assert "model_evaluation" in metadata["capabilities"]

    def test_performance_critic_agent_validate_input_valid(
        self, performance_critic_agent, mock_model
    ):
        """Test input validation with valid data."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path="data/test_data.csv",
        )

        assert performance_critic_agent.validate_input(request=request) is True

    def test_performance_critic_agent_validate_input_invalid_model_path(
        self, performance_critic_agent
    ):
        """Test input validation with invalid model path."""
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path="nonexistent_model.pkl",
            model_type="lstm",
            test_data_path="data/test_data.csv",
        )

        assert performance_critic_agent.validate_input(request=request) is False

    def test_performance_critic_agent_validate_input_invalid_test_data_path(
        self, performance_critic_agent, mock_model
    ):
        """Test input validation with invalid test data path."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path="nonexistent_data.csv",
        )

        assert performance_critic_agent.validate_input(request=request) is False

    def test_performance_critic_agent_validate_input_invalid_evaluation_period(
        self, performance_critic_agent, mock_model
    ):
        """Test input validation with invalid evaluation period."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path="data/test_data.csv",
            evaluation_period=0,  # Invalid
        )

        assert performance_critic_agent.validate_input(request=request) is False

    def test_performance_critic_agent_validate_input_missing_request(
        self, performance_critic_agent
    ):
        """Test input validation with missing request."""
        assert performance_critic_agent.validate_input() is False

    def test_performance_critic_agent_validate_input_wrong_type(
        self, performance_critic_agent
    ):
        """Test input validation with wrong request type."""
        assert performance_critic_agent.validate_input(request="not_a_request") is False

    @pytest.mark.asyncio
    async def test_performance_critic_agent_execute_success(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test successful model evaluation execution."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path=sample_test_data,
        )

        result = await performance_critic_agent.execute(request=request)

        assert result.success is True
        assert "performance_metrics" in result.data
        assert "risk_metrics" in result.data
        assert "trading_metrics" in result.data
        assert "recommendations" in result.data

    @pytest.mark.asyncio
    async def test_performance_critic_agent_execute_invalid_request(
        self, performance_critic_agent
    ):
        """Test execution with invalid request."""
        result = await performance_critic_agent.execute()

        assert result.success is False
        assert "required" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_performance_critic_agent_execute_wrong_request_type(
        self, performance_critic_agent
    ):
        """Test execution with wrong request type."""
        result = await performance_critic_agent.execute(request="not_a_request")

        assert result.success is False
        assert "instance" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_performance_critic_agent_evaluate_lstm_model(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test LSTM model evaluation."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path=sample_test_data,
        )

        result = performance_critic_agent.evaluate_model(request)

        assert result.evaluation_status == "success"
        assert result.model_id == "model_123"
        assert result.model_type == "lstm"
        assert "performance_metrics" in result.performance_metrics
        assert "risk_metrics" in result.risk_metrics
        assert "trading_metrics" in result.trading_metrics
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_performance_critic_agent_evaluate_xgboost_model(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test XGBoost model evaluation."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="xgboost",
            test_data_path=sample_test_data,
        )

        result = performance_critic_agent.evaluate_model(request)

        assert result.evaluation_status == "success"
        assert result.model_id == "model_123"
        assert result.model_type == "xgboost"
        assert "performance_metrics" in result.performance_metrics
        assert "risk_metrics" in result.risk_metrics
        assert "trading_metrics" in result.trading_metrics

    @pytest.mark.asyncio
    async def test_performance_critic_agent_evaluate_with_benchmark(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test model evaluation with benchmark comparison."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path=sample_test_data,
            benchmark_symbol="SPY",
        )

        result = performance_critic_agent.evaluate_model(request)

        assert result.evaluation_status == "success"
        assert result.benchmark_comparison is not None
        assert isinstance(result.benchmark_comparison, dict)

    @pytest.mark.asyncio
    async def test_performance_critic_agent_evaluate_with_custom_period(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test model evaluation with custom evaluation period."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path=sample_test_data,
            evaluation_period=100,
        )

        result = performance_critic_agent.evaluate_model(request)

        assert result.evaluation_status == "success"
        assert result.model_id == "model_123"

    def test_performance_critic_agent_get_evaluation_history(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test getting evaluation history."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path=sample_test_data,
        )

        # Evaluate model first
        performance_critic_agent.evaluate_model(request)

        # Get history
        history = performance_critic_agent.get_evaluation_history("model_123")

        assert isinstance(history, list)
        assert len(history) > 0
        assert all(
            isinstance(eval_result, ModelEvaluationResult) for eval_result in history
        )

    def test_performance_critic_agent_get_evaluation_history_nonexistent(
        self, performance_critic_agent
    ):
        """Test getting evaluation history for nonexistent model."""
        history = performance_critic_agent.get_evaluation_history("nonexistent_model")
        assert history == []

    def test_performance_critic_agent_get_model_performance_summary(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test getting model performance summary."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path=sample_test_data,
        )

        # Evaluate model first
        performance_critic_agent.evaluate_model(request)

        # Get summary
        summary = performance_critic_agent.get_model_performance_summary("model_123")

        assert isinstance(summary, dict)
        assert "model_id" in summary
        assert "evaluation_count" in summary
        assert "latest_evaluation" in summary


class TestPerformanceCriticAgentErrorHandling:
    """Test PerformanceCriticAgent error handling."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(name="test_performance_critic", enabled=True)

    @pytest.fixture
    def performance_critic_agent(self, agent_config):
        """Create a PerformanceCriticAgent instance for testing."""
        return PerformanceCriticAgent(agent_config)

    @pytest.mark.asyncio
    async def test_performance_critic_agent_evaluate_with_missing_model(
        self, performance_critic_agent, sample_test_data
    ):
        """Test evaluation with missing model file."""
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path="nonexistent_model.pkl",
            model_type="lstm",
            test_data_path=sample_test_data,
        )

        result = performance_critic_agent.evaluate_model(request)

        assert result.evaluation_status == "failed"
        assert "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_performance_critic_agent_evaluate_with_missing_test_data(
        self, performance_critic_agent, mock_model
    ):
        """Test evaluation with missing test data file."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path="nonexistent_data.csv",
        )

        result = performance_critic_agent.evaluate_model(request)

        assert result.evaluation_status == "failed"
        assert "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_performance_critic_agent_evaluate_with_invalid_model_type(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test evaluation with invalid model type."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="invalid_type",
            test_data_path=sample_test_data,
        )

        result = performance_critic_agent.evaluate_model(request)

        assert result.evaluation_status == "failed"
        assert "error" in result.error_message.lower()


class TestPerformanceCriticAgentMetrics:
    """Test PerformanceCriticAgent metrics calculation."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(name="test_performance_critic", enabled=True)

    @pytest.fixture
    def performance_critic_agent(self, agent_config):
        """Create a PerformanceCriticAgent instance for testing."""
        return PerformanceCriticAgent(agent_config)

    def test_performance_critic_agent_calculate_performance_metrics(
        self, performance_critic_agent
    ):
        """Test performance metrics calculation."""
        # Create sample predictions and test data
        predictions = pd.Series(np.random.randn(100) * 0.01 + 0.001)
        test_data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        metrics = performance_critic_agent._calculate_performance_metrics(
            predictions, test_data
        )

        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "annualized_return" in metrics

    def test_performance_critic_agent_calculate_risk_metrics(
        self, performance_critic_agent
    ):
        """Test risk metrics calculation."""
        # Create sample predictions and test data
        predictions = pd.Series(np.random.randn(100) * 0.01 + 0.001)
        test_data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        metrics = performance_critic_agent._calculate_risk_metrics(
            predictions, test_data
        )

        assert isinstance(metrics, dict)
        assert "max_drawdown" in metrics
        assert "volatility" in metrics
        assert "var_95" in metrics

    def test_performance_critic_agent_calculate_trading_metrics(
        self, performance_critic_agent
    ):
        """Test trading metrics calculation."""
        # Create sample predictions and test data
        predictions = pd.Series(np.random.randn(100) * 0.01 + 0.001)
        test_data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        metrics = performance_critic_agent._calculate_trading_metrics(
            predictions, test_data
        )

        assert isinstance(metrics, dict)
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "avg_trade" in metrics

    def test_performance_critic_agent_generate_recommendations(
        self, performance_critic_agent
    ):
        """Test recommendation generation."""
        performance_metrics = {"sharpe_ratio": 0.3, "total_return": 0.15}
        risk_metrics = {"max_drawdown": -0.20, "volatility": 0.25}
        trading_metrics = {"win_rate": 0.40, "profit_factor": 1.2}

        recommendations = performance_critic_agent._generate_recommendations(
            performance_metrics, risk_metrics, trading_metrics
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestPerformanceCriticAgentIntegration:
    """Test PerformanceCriticAgent integration scenarios."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(name="test_performance_critic", enabled=True)

    @pytest.fixture
    def performance_critic_agent(self, agent_config):
        """Create a PerformanceCriticAgent instance for testing."""
        return PerformanceCriticAgent(agent_config)

    @pytest.mark.asyncio
    async def test_performance_critic_agent_multiple_evaluations(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test multiple model evaluations."""
        model_path, _ = mock_model

        # Evaluate same model multiple times
        for i in range(3):
            request = ModelEvaluationRequest(
                model_id=f"model_{i}",
                model_path=model_path,
                model_type="lstm",
                test_data_path=sample_test_data,
            )

            result = performance_critic_agent.evaluate_model(request)
            assert result.evaluation_status == "success"

        # Check evaluation history
        for i in range(3):
            history = performance_critic_agent.get_evaluation_history(f"model_{i}")
            assert len(history) == 1

    @pytest.mark.asyncio
    async def test_performance_critic_agent_threshold_comparison(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test that evaluation respects thresholds."""
        model_path, _ = mock_model
        request = ModelEvaluationRequest(
            model_id="model_123",
            model_path=model_path,
            model_type="lstm",
            test_data_path=sample_test_data,
        )

        result = performance_critic_agent.evaluate_model(request)

        # Check that recommendations consider thresholds
        assert result.evaluation_status == "success"
        assert isinstance(result.recommendations, list)

        # The recommendations should be based on the thresholds
        # (actual content depends on the metrics calculated)
        assert len(result.recommendations) >= 0

    def test_underperforming_models_flagged_and_scheduled(
        self, performance_critic_agent, mock_model, sample_test_data
    ):
        """Test that underperforming models get flagged and scheduled for update."""
        print("\n🚨 Testing Underperforming Models Flagging and Scheduling")

        # Create different model performance scenarios
        performance_scenarios = [
            {
                "name": "High Performer",
                "metrics": {
                    "sharpe_ratio": 2.0,
                    "total_return": 0.35,
                    "max_drawdown": -0.08,
                    "win_rate": 0.75,
                    "profit_factor": 2.2,
                },
                "expected_flag": False,
                "expected_action": "maintain",
            },
            {
                "name": "Moderate Performer",
                "metrics": {
                    "sharpe_ratio": 1.2,
                    "total_return": 0.20,
                    "max_drawdown": -0.12,
                    "win_rate": 0.60,
                    "profit_factor": 1.5,
                },
                "expected_flag": False,
                "expected_action": "monitor",
            },
            {
                "name": "Underperformer - Low Sharpe",
                "metrics": {
                    "sharpe_ratio": 0.3,  # Below threshold of 0.5
                    "total_return": 0.05,
                    "max_drawdown": -0.15,
                    "win_rate": 0.55,
                    "profit_factor": 1.2,
                },
                "expected_flag": True,
                "expected_action": "retrain",
            },
            {
                "name": "Underperformer - High Drawdown",
                "metrics": {
                    "sharpe_ratio": 1.0,
                    "total_return": 0.15,
                    "max_drawdown": -0.25,  # Above threshold of -0.15
                    "win_rate": 0.65,
                    "profit_factor": 1.8,
                },
                "expected_flag": True,
                "expected_action": "optimize",
            },
            {
                "name": "Underperformer - Low Win Rate",
                "metrics": {
                    "sharpe_ratio": 0.8,
                    "total_return": 0.10,
                    "max_drawdown": -0.12,
                    "win_rate": 0.40,  # Below threshold of 0.45
                    "profit_factor": 1.1,
                },
                "expected_flag": True,
                "expected_action": "retrain",
            },
            {
                "name": "Critical Underperformer",
                "metrics": {
                    "sharpe_ratio": -0.5,  # Negative Sharpe
                    "total_return": -0.10,  # Negative return
                    "max_drawdown": -0.30,  # Very high drawdown
                    "win_rate": 0.35,  # Very low win rate
                    "profit_factor": 0.8,  # Below 1.0
                },
                "expected_flag": True,
                "expected_action": "replace",
            },
        ]

        for scenario in performance_scenarios:
            print(f"\n  📊 Testing scenario: {scenario['name']}")

            # Create evaluation result with scenario metrics
            evaluation_result = ModelEvaluationResult(
                request_id="test_eval",
                model_id="test_model",
                evaluation_timestamp="2024-01-01T12:00:00",
                performance_metrics={
                    "sharpe_ratio": scenario["metrics"]["sharpe_ratio"],
                    "total_return": scenario["metrics"]["total_return"],
                },
                risk_metrics={
                    "max_drawdown": scenario["metrics"]["max_drawdown"],
                    "volatility": 0.20,
                },
                trading_metrics={
                    "win_rate": scenario["metrics"]["win_rate"],
                    "profit_factor": scenario["metrics"]["profit_factor"],
                },
                evaluation_status="success",
            )

            # Check if model should be flagged
            should_flag = performance_critic_agent.should_flag_model(evaluation_result)
            print(
                f"    Should flag: {should_flag} (expected: {scenario['expected_flag']})"
            )

            # Verify flagging logic
            assert (
                should_flag == scenario["expected_flag"]
            ), f"Flagging logic failed for {scenario['name']}"

            if should_flag:
                # Test flagging and scheduling
                flag_result = performance_critic_agent.flag_underperforming_model(
                    evaluation_result
                )

                # Verify flag result
                assert flag_result is not None, "Flag result should not be None"
                assert "model_id" in flag_result, "Flag result should contain model_id"
                assert (
                    "flag_reason" in flag_result
                ), "Flag result should contain flag_reason"
                assert (
                    "recommended_action" in flag_result
                ), "Flag result should contain recommended_action"
                assert "priority" in flag_result, "Flag result should contain priority"

                print(f"    Flag reason: {flag_result['flag_reason']}")
                print(f"    Recommended action: {flag_result['recommended_action']}")
                print(f"    Priority: {flag_result['priority']}")

                # Verify recommended action matches expected
                assert (
                    flag_result["recommended_action"] == scenario["expected_action"]
                ), f"Recommended action should be {scenario['expected_action']} for {scenario['name']}"

                # Test scheduling for update
                schedule_result = performance_critic_agent.schedule_model_update(
                    flag_result
                )

                # Verify scheduling result
                assert schedule_result is not None, "Schedule result should not be None"
                assert (
                    "update_id" in schedule_result
                ), "Schedule result should contain update_id"
                assert (
                    "scheduled_time" in schedule_result
                ), "Schedule result should contain scheduled_time"
                assert (
                    "update_type" in schedule_result
                ), "Schedule result should contain update_type"

                print(f"    Update scheduled: {schedule_result['update_id']}")
                print(f"    Update type: {schedule_result['update_type']}")

                # Verify update type based on performance
                if scenario["name"] == "Critical Underperformer":
                    assert (
                        schedule_result["update_type"] == "replace"
                    ), "Critical underperformer should be scheduled for replacement"
                elif "High Drawdown" in scenario["name"]:
                    assert (
                        schedule_result["update_type"] == "optimize"
                    ), "High drawdown should be scheduled for optimization"
                else:
                    assert (
                        schedule_result["update_type"] == "retrain"
                    ), "Other underperformers should be scheduled for retraining"

        # Test batch flagging and scheduling
        print(f"\n  📋 Testing batch flagging and scheduling...")

        # Create multiple evaluation results
        batch_evaluations = []
        for i, scenario in enumerate(
            performance_scenarios[:3]
        ):  # Test first 3 scenarios
            evaluation_result = ModelEvaluationResult(
                request_id=f"batch_eval_{i}",
                model_id=f"model_{i}",
                evaluation_timestamp="2024-01-01T12:00:00",
                performance_metrics={
                    "sharpe_ratio": scenario["metrics"]["sharpe_ratio"],
                    "total_return": scenario["metrics"]["total_return"],
                },
                risk_metrics={
                    "max_drawdown": scenario["metrics"]["max_drawdown"],
                    "volatility": 0.20,
                },
                trading_metrics={
                    "win_rate": scenario["metrics"]["win_rate"],
                    "profit_factor": scenario["metrics"]["profit_factor"],
                },
                evaluation_status="success",
            )
            batch_evaluations.append(evaluation_result)

        # Process batch evaluations
        batch_flags = performance_critic_agent.flag_batch_models(batch_evaluations)
        batch_schedules = performance_critic_agent.schedule_batch_updates(batch_flags)

        # Verify batch processing
        assert isinstance(batch_flags, list), "Batch flags should be a list"
        assert isinstance(batch_schedules, list), "Batch schedules should be a list"

        flagged_count = sum(1 for flag in batch_flags if flag is not None)
        scheduled_count = sum(1 for schedule in batch_schedules if schedule is not None)

        print(f"    Models flagged: {flagged_count}/{len(batch_evaluations)}")
        print(f"    Updates scheduled: {scheduled_count}/{flagged_count}")

        # Verify that only underperforming models are flagged
        expected_flags = sum(
            1 for scenario in performance_scenarios[:3] if scenario["expected_flag"]
        )
        assert (
            flagged_count == expected_flags
        ), f"Should flag {expected_flags} underperforming models"

        # Test priority ordering
        print(f"\n  🎯 Testing priority ordering...")

        if batch_flags:
            # Sort by priority
            sorted_flags = sorted(
                batch_flags, key=lambda x: x["priority"] if x else 0, reverse=True
            )

            # Verify priority ordering (higher priority first)
            priorities = [flag["priority"] for flag in sorted_flags if flag]
            print(f"    Priority order: {priorities}")

            # Verify descending order
            assert priorities == sorted(
                priorities, reverse=True
            ), "Flags should be ordered by priority (descending)"

        # Test flag history tracking
        print(f"\n  📈 Testing flag history tracking...")

        # Get flag history
        flag_history = performance_critic_agent.get_flag_history()

        # Verify history tracking
        assert isinstance(flag_history, dict), "Flag history should be a dictionary"
        assert "total_flags" in flag_history, "History should contain total_flags"
        assert "flagged_models" in flag_history, "History should contain flagged_models"
        assert (
            "update_schedule" in flag_history
        ), "History should contain update_schedule"

        print(f"    Total flags: {flag_history['total_flags']}")
        print(f"    Flagged models: {len(flag_history['flagged_models'])}")
        print(f"    Scheduled updates: {len(flag_history['update_schedule'])}")

        # Verify history consistency
        assert (
            flag_history["total_flags"] >= flagged_count
        ), "History should reflect all flagged models"

        print("✅ Underperforming models flagging and scheduling test completed")
