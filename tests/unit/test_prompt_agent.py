"""
Unit tests for Prompt Agent.

Tests prompt agent functionality with simulated prompts and mocked LLM responses,
including action selection, model selection, and strategy selection.
"""

import json
import warnings
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import prompt agent modules
try:
    from trading.agents.agent_manager import AgentManager
    from trading.agents.prompt_agent import PromptAgent

    PROMPT_AGENT_AVAILABLE = True
except ImportError:
    PROMPT_AGENT_AVAILABLE = False
    PromptAgent = Mock()
    AgentManager = Mock()


class TestPromptAgent:
    """Test suite for Prompt Agent."""

    @pytest.fixture
    def prompt_agent(self):
        """Create Prompt Agent instance."""
        if not PROMPT_AGENT_AVAILABLE:
            pytest.skip("Prompt Agent not available")
        return PromptAgent()

    @pytest.fixture
    def agent_manager(self):
        """Create Agent Manager instance."""
        if not PROMPT_AGENT_AVAILABLE:
            pytest.skip("Agent Manager not available")
        return AgentManager()

    @pytest.fixture
    def sample_prompts(self):
        """Sample prompts for testing."""
        return [
            "forecast AAPL for the next 10 days",
            "apply RSI strategy to TSLA",
            "use MACD signals for MSFT",
            "run Bollinger Bands on GOOGL",
            "analyze market sentiment for NVDA",
            "get technical indicators for AMZN",
            "predict price movement for META",
            "generate trading signals for NFLX",
        ]

    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for different scenarios."""
        return {
            "forecast": {"action": "forecast", "model": "arima", "symbol": "AAPL", "horizon": 10, "confidence": 0.85},
            "rsi_strategy": {
                "action": "apply_strategy",
                "strategy": "rsi",
                "symbol": "TSLA",
                "parameters": {"period": 14, "oversold": 30, "overbought": 70},
                "confidence": 0.78,
            },
            "macd_strategy": {
                "action": "apply_strategy",
                "strategy": "macd",
                "symbol": "MSFT",
                "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "confidence": 0.82,
            },
            "bollinger_strategy": {
                "action": "apply_strategy",
                "strategy": "bollinger",
                "symbol": "GOOGL",
                "parameters": {"period": 20, "std_dev": 2},
                "confidence": 0.75,
            },
            "sentiment_analysis": {
                "action": "analyze_sentiment",
                "symbol": "NVDA",
                "sources": ["news", "social_media", "earnings"],
                "confidence": 0.70,
            },
            "technical_indicators": {
                "action": "get_indicators",
                "symbol": "AMZN",
                "indicators": ["rsi", "macd", "bollinger_bands", "moving_averages"],
                "confidence": 0.88,
            },
        }

    def test_agent_instantiation(self, prompt_agent):
        """Test that Prompt Agent instantiates correctly."""
        assert prompt_agent is not None
        assert hasattr(prompt_agent, "process_prompt")
        assert hasattr(prompt_agent, "select_action")
        assert hasattr(prompt_agent, "select_model")
        assert hasattr(prompt_agent, "select_strategy")

    def test_prompt_processing(self, prompt_agent, sample_prompts):
        """Test that prompts are processed correctly."""
        for prompt in sample_prompts:
            result = prompt_agent.process_prompt(prompt)

            assert isinstance(result, dict)
            assert "success" in result
            assert "action" in result
            assert "timestamp" in result

    def test_forecast_prompt_handling(self, prompt_agent):
        """Test handling of forecast prompts."""
        forecast_prompts = [
            "forecast AAPL for the next 10 days",
            "predict TSLA price movement",
            "forecast MSFT stock price",
            "predict GOOGL for 5 days",
        ]

        for prompt in forecast_prompts:
            result = prompt_agent.process_prompt(prompt)

            assert result["success"] is True
            assert result["action"] == "forecast"
            assert "symbol" in result
            assert "horizon" in result
            assert "model" in result

    def test_strategy_prompt_handling(self, prompt_agent):
        """Test handling of strategy prompts."""
        strategy_prompts = [
            "apply RSI strategy to TSLA",
            "use MACD signals for MSFT",
            "run Bollinger Bands on GOOGL",
            "apply moving average strategy",
        ]

        for prompt in strategy_prompts:
            result = prompt_agent.process_prompt(prompt)

            assert result["success"] is True
            assert result["action"] == "apply_strategy"
            assert "strategy" in result
            assert "symbol" in result

    def test_action_selection(self, prompt_agent, sample_prompts):
        """Test that correct actions are selected for different prompts."""
        for prompt in sample_prompts:
            result = prompt_agent.process_prompt(prompt)

            assert result["success"] is True
            assert result["action"] in ["forecast", "apply_strategy", "analyze_sentiment", "get_indicators"]

    def test_model_selection(self, prompt_agent):
        """Test that correct models are selected for forecasting."""
        forecast_prompts = [
            "forecast AAPL using ARIMA",
            "predict TSLA with LSTM",
            "forecast MSFT using Prophet",
            "predict GOOGL with XGBoost",
        ]

        for prompt in forecast_prompts:
            result = prompt_agent.process_prompt(prompt)

            if result["action"] == "forecast":
                assert "model" in result
                assert result["model"] in ["arima", "lstm", "prophet", "xgboost", "hybrid"]

    def test_strategy_selection(self, prompt_agent):
        """Test that correct strategies are selected."""
        strategy_prompts = [
            "apply RSI strategy",
            "use MACD signals",
            "run Bollinger Bands",
            "apply moving average crossover",
        ]

        for prompt in strategy_prompts:
            result = prompt_agent.process_prompt(prompt)

            if result["action"] == "apply_strategy":
                assert "strategy" in result
                assert result["strategy"] in ["rsi", "macd", "bollinger", "moving_average"]

    @patch("trading.agents.prompt_agent.openai.ChatCompletion.create")
    def test_openai_integration(self, mock_openai, prompt_agent):
        """Test OpenAI integration with mocked responses."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {"action": "forecast", "model": "arima", "symbol": "AAPL", "horizon": 10, "confidence": 0.85}
        )
        mock_openai.return_value = mock_response

        result = prompt_agent.process_prompt("forecast AAPL for the next 10 days")

        assert result["success"] is True
        assert result["action"] == "forecast"
        assert result["model"] == "arima"
        assert result["symbol"] == "AAPL"
        assert result["horizon"] == 10

    @patch("trading.agents.prompt_agent.huggingface_hub.InferenceClient")
    def test_huggingface_integration(self, mock_hf, prompt_agent):
        """Test HuggingFace integration with mocked responses."""
        # Mock HuggingFace response
        mock_client = Mock()
        mock_client.text_generation.return_value = json.dumps(
            {
                "action": "apply_strategy",
                "strategy": "rsi",
                "symbol": "TSLA",
                "parameters": {"period": 14},
                "confidence": 0.78,
            }
        )
        mock_hf.return_value = mock_client

        result = prompt_agent.process_prompt("apply RSI strategy to TSLA")

        assert result["success"] is True
        assert result["action"] == "apply_strategy"
        assert result["strategy"] == "rsi"
        assert result["symbol"] == "TSLA"

    def test_symbol_extraction(self, prompt_agent):
        """Test that stock symbols are correctly extracted from prompts."""
        symbol_tests = [
            ("forecast AAPL", "AAPL"),
            ("apply RSI to TSLA", "TSLA"),
            ("use MACD for MSFT", "MSFT"),
            ("analyze GOOGL", "GOOGL"),
            ("predict NVDA price", "NVDA"),
            ("get indicators for AMZN", "AMZN"),
        ]

        for prompt, expected_symbol in symbol_tests:
            result = prompt_agent.process_prompt(prompt)

            if result["success"]:
                assert "symbol" in result
                assert result["symbol"] == expected_symbol

    def test_parameter_extraction(self, prompt_agent):
        """Test that parameters are correctly extracted from prompts."""
        parameter_tests = [
            ("forecast AAPL for 10 days", {"horizon": 10}),
            ("apply RSI with period 21", {"period": 21}),
            ("use MACD fast=8 slow=21", {"fast_period": 8, "slow_period": 21}),
            ("Bollinger Bands std=3", {"std_dev": 3}),
        ]

        for prompt, expected_params in parameter_tests:
            result = prompt_agent.process_prompt(prompt)

            if result["success"]:
                for param, value in expected_params.items():
                    assert param in result
                    assert result[param] == value

    def test_confidence_calculation(self, prompt_agent):
        """Test that confidence scores are calculated correctly."""
        result = prompt_agent.process_prompt("forecast AAPL for 10 days")

        if result["success"]:
            assert "confidence" in result
            confidence = result["confidence"]
            assert isinstance(confidence, (int, float))
            assert 0 <= confidence <= 1

    def test_error_handling(self, prompt_agent):
        """Test error handling for invalid prompts."""
        invalid_prompts = [
            "",  # Empty prompt
            "invalid command",  # Unrecognized command
            "forecast",  # Incomplete prompt
            "apply strategy",  # Missing symbol
        ]

        for prompt in invalid_prompts:
            result = prompt_agent.process_prompt(prompt)

            assert result["success"] is False
            assert "error" in result
            assert isinstance(result["error"], str)

    def test_ambiguous_prompt_handling(self, prompt_agent):
        """Test handling of ambiguous prompts."""
        ambiguous_prompts = [
            "analyze stock",  # No symbol specified
            "apply strategy",  # No strategy specified
            "forecast price",  # No symbol or horizon
            "get indicators",  # No symbol specified
        ]

        for prompt in ambiguous_prompts:
            result = prompt_agent.process_prompt(prompt)

            # Should either fail gracefully or use defaults
            if result["success"]:
                assert "symbol" in result or "error" in result
            else:
                assert "error" in result

    def test_context_awareness(self, prompt_agent):
        """Test that agent maintains context across multiple prompts."""
        # First prompt
        result1 = prompt_agent.process_prompt("forecast AAPL for 10 days")

        # Second prompt should maintain context
        result2 = prompt_agent.process_prompt("apply RSI strategy")

        if result1["success"] and result2["success"]:
            # Second prompt should inherit symbol from context
            assert result2["symbol"] == "AAPL"

    def test_parameter_validation(self, prompt_agent):
        """Test that parameters are validated correctly."""
        invalid_parameter_tests = [
            ("forecast AAPL for -5 days", "horizon"),  # Negative horizon
            ("apply RSI with period 0", "period"),  # Zero period
            ("use MACD fast=0 slow=26", "fast_period"),  # Zero fast period
        ]

        for prompt, invalid_param in invalid_parameter_tests:
            result = prompt_agent.process_prompt(prompt)

            if result["success"]:
                # Should use default values for invalid parameters
                assert invalid_param in result
                assert result[invalid_param] > 0
            else:
                # Should fail with validation error
                assert "error" in result
                assert "invalid" in result["error"].lower() or "validation" in result["error"].lower()

    def test_model_compatibility(self, prompt_agent):
        """Test that model selection is compatible with data."""
        compatibility_tests = [
            ("forecast AAPL with ARIMA", "arima"),
            ("predict TSLA using LSTM", "lstm"),
            ("forecast MSFT with Prophet", "prophet"),
            ("predict GOOGL using XGBoost", "xgboost"),
        ]

        for prompt, expected_model in compatibility_tests:
            result = prompt_agent.process_prompt(prompt)

            if result["success"] and result["action"] == "forecast":
                assert result["model"] == expected_model

    def test_strategy_compatibility(self, prompt_agent):
        """Test that strategy selection is compatible with data."""
        compatibility_tests = [
            ("apply RSI strategy", "rsi"),
            ("use MACD signals", "macd"),
            ("run Bollinger Bands", "bollinger"),
            ("apply moving average", "moving_average"),
        ]

        for prompt, expected_strategy in compatibility_tests:
            result = prompt_agent.process_prompt(prompt)

            if result["success"] and result["action"] == "apply_strategy":
                assert result["strategy"] == expected_strategy

    def test_response_format(self, prompt_agent):
        """Test that responses are formatted correctly."""
        result = prompt_agent.process_prompt("forecast AAPL for 10 days")

        # Check required fields
        required_fields = ["success", "action", "timestamp"]
        for field in required_fields:
            assert field in result

        # Check data types
        assert isinstance(result["success"], bool)
        assert isinstance(result["action"], str)
        assert isinstance(result["timestamp"], str)

        # Check timestamp format
        try:
            datetime.fromisoformat(result["timestamp"])
        except ValueError:
            pytest.fail("Timestamp is not in ISO format")

    def test_agent_manager_integration(self, agent_manager, prompt_agent):
        """Test integration with agent manager."""
        # Test agent registration
        agent_manager.register_agent("prompt_agent", prompt_agent)

        # Test agent retrieval
        retrieved_agent = agent_manager.get_agent("prompt_agent")
        assert retrieved_agent is not None
        assert retrieved_agent == prompt_agent

        # Test agent execution
        result = agent_manager.execute_agent("prompt_agent", "forecast AAPL for 10 days")
        assert isinstance(result, dict)
        assert "success" in result

    def test_concurrent_prompt_processing(self, prompt_agent):
        """Test concurrent prompt processing."""
        import threading

        results = []
        errors = []

        def process_prompt(prompt):
            try:
                result = prompt_agent.process_prompt(prompt)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        prompts = ["forecast AAPL for 10 days", "apply RSI to TSLA", "use MACD for MSFT", "analyze GOOGL"]

        for prompt in prompts:
            thread = threading.Thread(target=process_prompt, args=(prompt,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == len(prompts)
        assert len(errors) == 0

        for result in results:
            assert isinstance(result, dict)
            assert "success" in result

    def test_memory_management(self, prompt_agent):
        """Test memory management for long conversations."""
        # Process many prompts to test memory usage
        for i in range(100):
            prompt = f"forecast AAPL for {i+1} days"
            result = prompt_agent.process_prompt(prompt)

            assert result["success"] is True
            assert result["action"] == "forecast"
            assert result["symbol"] == "AAPL"
            assert result["horizon"] == i + 1

    def test_custom_llm_provider(self, prompt_agent):
        """Test custom LLM provider integration."""
        # Mock custom LLM provider
        custom_llm = Mock()
        custom_llm.generate.return_value = {
            "action": "forecast",
            "model": "custom_model",
            "symbol": "AAPL",
            "horizon": 10,
            "confidence": 0.90,
        }

        # Test with custom provider
        with patch.object(prompt_agent, "llm_provider", custom_llm):
            result = prompt_agent.process_prompt("forecast AAPL for 10 days")

            assert result["success"] is True
            assert result["model"] == "custom_model"
            assert result["confidence"] == 0.90


if __name__ == "__main__":
    pytest.main([__file__])
