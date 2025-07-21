"""
Tests for Prompt Examples functionality

This module tests the prompt examples system that uses semantic similarity
to find relevant examples for new prompts.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.llm.agent import PromptAgent


@pytest.fixture
def sample_prompt_examples():
    """Create sample prompt examples for testing."""
    return {
        "examples": [
            {
                "id": "test_forecast_001",
                "prompt": "Forecast AAPL stock price for the next 30 days",
                "category": "forecasting",
                "symbols": ["AAPL"],
                "timeframe": "30 days",
                "strategy_type": "Forecasting",
                "parsed_output": {
                    "action": "forecast",
                    "symbol": "AAPL",
                    "timeframe": "30 days",
                },
                "success": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "performance_score": 0.92,
            },
            {
                "id": "test_strategy_001",
                "prompt": "Create RSI strategy for TSLA",
                "category": "strategy_creation",
                "symbols": ["TSLA"],
                "timeframe": "unknown",
                "strategy_type": "RSI",
                "parsed_output": {
                    "action": "create_strategy",
                    "symbol": "TSLA",
                    "strategy": "RSI",
                },
                "success": True,
                "timestamp": "2024-01-15T11:15:00Z",
                "performance_score": 0.88,
            },
        ],
        "metadata": {
            "version": "1.0",
            "created": "2024-01-15T10:00:00Z",
            "last_updated": "2024-01-15T11:15:00Z",
            "total_examples": 2,
            "categories": ["forecasting", "strategy_creation"],
            "symbols": ["AAPL", "TSLA"],
        },
    }


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for testing."""
    with patch("agents.llm.agent.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        with patch("agents.llm.agent.SentenceTransformer") as mock_transformer:
            # Mock the encode method to return dummy embeddings
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_transformer.return_value = mock_model
            yield mock_transformer


class TestPromptExamples:
    """Test cases for prompt examples functionality."""

    def test_load_prompt_examples(self, sample_prompt_examples, tmp_path):
        """Test loading prompt examples from JSON file."""
        # Create temporary prompt examples file
        examples_file = tmp_path / "prompt_examples.json"
        with open(examples_file, "w") as f:
            json.dump(sample_prompt_examples, f)

        # Mock the file path
        with patch("agents.llm.agent.Path") as mock_path:
            mock_path.return_value.__truediv__.return_value = examples_file

            # Create agent (this will trigger loading)
            agent = PromptAgent()

            # Check that examples were loaded
            assert agent.prompt_examples is not None
            assert len(agent.prompt_examples["examples"]) == 2
            assert agent.prompt_examples["metadata"]["total_examples"] == 2

    def test_extract_symbols_from_prompt(self):
        """Test symbol extraction from prompts."""
        agent = PromptAgent()

        # Test various prompt formats
        assert agent._extract_symbols_from_prompt("Forecast AAPL price") == ["AAPL"]
        assert agent._extract_symbols_from_prompt("Analyze TSLA and GOOGL") == [
            "TSLA",
            "GOOGL",
        ]
        assert agent._extract_symbols_from_prompt("No symbols here") == []
        assert agent._extract_symbols_from_prompt("AAPL, TSLA, and MSFT") == [
            "AAPL",
            "TSLA",
            "MSFT",
        ]

    def test_extract_timeframe_from_prompt(self):
        """Test timeframe extraction from prompts."""
        agent = PromptAgent()

        # Test various timeframe formats
        assert agent._extract_timeframe_from_prompt("next 30 days") == "30 days"
        assert agent._extract_timeframe_from_prompt("next 2 weeks") == "14 days"
        assert agent._extract_timeframe_from_prompt("next 3 months") == "90 days"
        assert agent._extract_timeframe_from_prompt("last 6 months") == "180 days"
        assert agent._extract_timeframe_from_prompt("no timeframe") == "unknown"

    def test_extract_strategy_type_from_prompt(self):
        """Test strategy type extraction from prompts."""
        agent = PromptAgent()

        # Test various strategy keywords
        assert agent._extract_strategy_type_from_prompt("RSI strategy") == "RSI"
        assert agent._extract_strategy_type_from_prompt("MACD analysis") == "MACD"
        assert (
            agent._extract_strategy_type_from_prompt("Bollinger Bands")
            == "Bollinger_Bands"
        )
        assert (
            agent._extract_strategy_type_from_prompt("forecast price") == "Forecasting"
        )
        assert (
            agent._extract_strategy_type_from_prompt("backtest strategy")
            == "Backtesting"
        )
        assert agent._extract_strategy_type_from_prompt("no strategy") == "unknown"

    def test_compute_example_embeddings(
        self, sample_prompt_examples, mock_sentence_transformer
    ):
        """Test computing embeddings for examples."""
        agent = PromptAgent()
        agent.prompt_examples = sample_prompt_examples

        # Test embedding computation
        embeddings = agent._compute_example_embeddings()

        assert embeddings is not None
        assert embeddings.shape == (2, 3)  # 2 examples, 3-dimensional embeddings

    def test_find_similar_examples(
        self, sample_prompt_examples, mock_sentence_transformer
    ):
        """Test finding similar examples using cosine similarity."""
        agent = PromptAgent()
        agent.prompt_examples = sample_prompt_examples
        agent.sentence_transformer = mock_sentence_transformer.return_value
        agent.example_embeddings = agent._compute_example_embeddings()

        # Test finding similar examples
        similar_examples = agent._find_similar_examples("Forecast TSLA price", top_k=2)

        assert len(similar_examples) == 2
        assert all("similarity_score" in ex for ex in similar_examples)
        assert all("example" in ex for ex in similar_examples)

    def test_create_few_shot_prompt(self, sample_prompt_examples):
        """Test creating few-shot prompts with examples."""
        agent = PromptAgent()

        similar_examples = [
            {
                "example": sample_prompt_examples["examples"][0],
                "similarity_score": 0.85,
                "prompt": sample_prompt_examples["examples"][0]["prompt"],
                "parsed_output": sample_prompt_examples["examples"][0]["parsed_output"],
                "category": "forecasting",
                "performance_score": 0.92,
            }
        ]

        # Test creating few-shot prompt
        enhanced_prompt = agent._create_few_shot_prompt(
            "Forecast GOOGL price", similar_examples
        )

        assert "Here are some similar examples" in enhanced_prompt
        assert "Forecast AAPL stock price" in enhanced_prompt
        assert "Forecast GOOGL price" in enhanced_prompt
        assert "JSON format" in enhanced_prompt

    def test_get_prompt_examples_stats(self, sample_prompt_examples):
        """Test getting prompt examples statistics."""
        agent = PromptAgent()
        agent.prompt_examples = sample_prompt_examples

        # Test getting statistics
        stats = agent.get_prompt_examples_stats()

        assert stats["total_examples"] == 2
        assert "forecasting" in stats["categories"]
        assert "strategy_creation" in stats["categories"]
        assert "AAPL" in stats["unique_symbols"]
        assert "TSLA" in stats["unique_symbols"]
        assert "Forecasting" in stats["unique_strategy_types"]
        assert "RSI" in stats["unique_strategy_types"]
        assert "average_performance_score" in stats

    def test_save_successful_example(self, sample_prompt_examples, tmp_path):
        """Test saving successful prompt examples."""
        # Create temporary file
        examples_file = tmp_path / "prompt_examples.json"
        with open(examples_file, "w") as f:
            json.dump(sample_prompt_examples, f)

        # Mock the file path
        with patch("agents.llm.agent.Path") as mock_path:
            mock_path.return_value.__truediv__.return_value = examples_file

            agent = PromptAgent()
            agent.prompt_examples = sample_prompt_examples

            # Test saving successful example
            parsed_output = {
                "action": "forecast",
                "symbol": "MSFT",
                "timeframe": "15 days",
            }

            agent._save_successful_example(
                "Forecast MSFT price for next 15 days",
                parsed_output,
                "forecasting",
                0.95,
            )

            # Check that example was added
            assert len(agent.prompt_examples["examples"]) == 3
            assert agent.prompt_examples["metadata"]["total_examples"] == 3

            # Check the new example
            new_example = agent.prompt_examples["examples"][-1]
            assert new_example["prompt"] == "Forecast MSFT price for next 15 days"
            assert new_example["category"] == "forecasting"
            assert new_example["symbols"] == ["MSFT"]
            assert new_example["performance_score"] == 0.95

    def test_no_sentence_transformers_available(self):
        """Test behavior when sentence transformers are not available."""
        with patch("agents.llm.agent.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            agent = PromptAgent()

            # Should not have sentence transformer
            assert agent.sentence_transformer is None
            assert agent.example_embeddings is None

            # Should return empty list for similar examples
            similar_examples = agent._find_similar_examples("test prompt")
            assert similar_examples == []

    def test_process_prompt_with_examples(
        self, sample_prompt_examples, mock_sentence_transformer
    ):
        """Test processing prompt with few-shot examples."""
        agent = PromptAgent()
        agent.prompt_examples = sample_prompt_examples
        agent.sentence_transformer = mock_sentence_transformer.return_value
        agent.example_embeddings = agent._compute_example_embeddings()

        # Mock the response methods to return success
        with patch.object(agent, "_handle_forecast_request") as mock_forecast:
            mock_forecast.return_value = {
                "success": True,
                "result": {"forecast": "test"},
                "message": "Success",
                "timestamp": "2024-01-15T12:00:00Z",
            }

            # Test processing prompt
            response = agent.process_prompt("Forecast AAPL price for next 30 days")

            # Should find similar examples and process successfully
            assert response is not None
            assert response.get("success", False)


if __name__ == "__main__":
    pytest.main([__file__])
