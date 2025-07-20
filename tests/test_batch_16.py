"""
Tests for Batch 16 improvements:
- Soft-matching sentiment analysis with embeddings
- Rolling score decay with exponential weighting
- Enhanced error logging for prompt handling
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import logging
import hashlib

# Import the modules we're testing
from trading.nlp.sentiment_processor import SentimentProcessor
from trading.utils.metrics.scorer import ModelScorer
from trading.services.launch_prompt_router import PromptValidationHandler


class TestBatch16Improvements(unittest.TestCase):
    """Test suite for Batch 16 improvements."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging for tests
        logging.basicConfig(level=logging.DEBUG)
        
        # Create test data
        self.test_texts = [
            "This stock is performing excellently with strong growth",
            "The market is crashing and investors are panicking",
            "The company reported mixed results with uncertain outlook",
            "Bullish sentiment is driving the price higher",
            "Bearish signals suggest a potential downturn"
        ]
        
        self.test_predictions = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        self.test_actuals = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        self.test_returns = np.array([0.01, -0.02, 0.005, -0.01, 0.015])
        
        # Test prompts
        self.test_prompts = [
            "Forecast AAPL stock price for next week",
            "Analyze the technical indicators for TSLA",
            "Create a momentum strategy for SPY",
            "Calculate portfolio risk metrics",
            "Short prompt",  # Should fail validation
            "A" * 1500,  # Too long prompt
        ]

    @patch('trading.nlp.sentiment_processor.EMBEDDINGS_AVAILABLE', True)
    @patch('trading.nlp.sentiment_processor.SentenceTransformer', create=True)
    @patch('trading.nlp.sentiment_processor.cosine_similarity', create=True)
    def test_sentiment_soft_matching_initialization(self, mock_cosine, mock_transformer):
        """Test soft-matching initialization in sentiment processor."""
        # Mock the transformer
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.random.rand(10, 384)  # Mock embeddings
        
        processor = SentimentProcessor(enable_soft_matching=True)
        
        # Check that soft-matching was initialized
        self.assertTrue(processor.enable_soft_matching)
        self.assertIsNotNone(processor.embedding_model)
        self.assertEqual(processor.soft_match_threshold, 0.7)
        
        # Test with embeddings not available
        with patch('trading.nlp.sentiment_processor.EMBEDDINGS_AVAILABLE', False):
            processor = SentimentProcessor(enable_soft_matching=True)
            self.assertFalse(processor.enable_soft_matching)

    @patch('trading.nlp.sentiment_processor.EMBEDDINGS_AVAILABLE', True)
    @patch('trading.nlp.sentiment_processor.SentenceTransformer', create=True)
    @patch('trading.nlp.sentiment_processor.cosine_similarity', create=True)
    def test_sentiment_soft_matching_functionality(self, mock_cosine, mock_transformer):
        """Test soft-matching functionality for rare words."""
        # Mock the embedding model
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        # Prepare a fake lexicon and embeddings
        lexicon_words = ['bullish', 'bearish', 'positive', 'negative']
        lexicon_embeddings = {w: np.random.rand(384) for w in lexicon_words}
        mock_embeddings = [np.random.rand(384) for _ in lexicon_words]
        
        # Patch encode to return embeddings for lexicon words, and for the input word
        def encode_side_effect(words):
            if isinstance(words, list) and len(words) == len(lexicon_words):
                return mock_embeddings
            else:
                return [np.random.rand(384)]
        mock_model.encode.side_effect = encode_side_effect

        # Mock similarity: first word is above threshold
        def cosine_side_effect(a, b):
            # Always return 0.85 for the first word, below for others
            return np.array([[0.85]])
        mock_cosine.side_effect = cosine_side_effect

        processor = SentimentProcessor(enable_soft_matching=True)
        processor.embedding_model = mock_model
        processor.lexicon_embeddings = lexicon_embeddings
        processor.sentiment_lexicons = {
            "basic": {
                "bullish": 1.0,
                "bearish": -1.0,
                "positive": 0.8,
                "negative": -0.8
            }
        }

        # Test soft-matching for a rare word
        soft_matches = processor._find_soft_matches("bullish_synonym")

        # Should find matches above threshold
        self.assertGreater(len(soft_matches), 0)
        self.assertGreaterEqual(soft_matches[0][1], processor.soft_match_threshold)

    def test_rolling_score_decay_initialization(self):
        """Test rolling score decay initialization."""
        config = {
            "enable_rolling_decay": True,
            "decay_span": 10,
            "min_window_size": 5
        }
        
        scorer = ModelScorer(config)
        
        self.assertTrue(scorer.enable_rolling_decay)
        self.assertEqual(scorer.decay_span, 10)
        self.assertEqual(scorer.min_window_size, 5)
        self.assertEqual(len(scorer.score_history), 0)

    def test_rolling_score_decay_functionality(self):
        """Test rolling score decay with exponential weighting."""
        scorer = ModelScorer({"enable_rolling_decay": True, "decay_span": 3})
        
        model_name = "test_model"
        timestamps = [
            datetime.now() - timedelta(minutes=4),
            datetime.now() - timedelta(minutes=3),
            datetime.now() - timedelta(minutes=2),
            datetime.now() - timedelta(minutes=1),
            datetime.now()
        ]
        
        # Add some score history
        for i, ts in enumerate(timestamps):
            scores = {"mse": 1.0 + i * 0.1, "sharpe": 0.5 - i * 0.05}
            scorer.score_history[model_name] = scorer.score_history.get(model_name, [])
            scorer.score_history[model_name].append((ts, scores))
        
        # Test applying rolling decay
        current_scores = {"mse": 1.5, "sharpe": 0.3}
        decayed_scores = scorer._apply_rolling_decay(model_name, current_scores, datetime.now())
        
        # Should have decayed scores
        self.assertIn("mse", decayed_scores)
        self.assertIn("sharpe", decayed_scores)
        self.assertIsInstance(decayed_scores["mse"], float)
        self.assertIsInstance(decayed_scores["sharpe"], float)

    def test_score_trend_analysis(self):
        """Test score trend analysis functionality."""
        scorer = ModelScorer()
        model_name = "trend_test_model"
        
        # Create some score history
        base_time = datetime.now()
        for i in range(10):
            ts = base_time + timedelta(hours=i)
            scores = {"mse": 1.0 + i * 0.1}  # Increasing trend
            scorer.score_history[model_name] = scorer.score_history.get(model_name, [])
            scorer.score_history[model_name].append((ts, scores))
        
        # Test trend analysis
        trend_info = scorer.get_score_trend(model_name, "mse", window=5)
        
        self.assertIn("trend_slope", trend_info)
        self.assertIn("current_value", trend_info)
        self.assertIn("data_points", trend_info)
        self.assertEqual(trend_info["data_points"], 10)
        self.assertGreater(trend_info["trend_slope"], 0)  # Should be increasing

    def test_enhanced_error_logging_initialization(self):
        """Test enhanced error logging initialization."""
        handler = PromptValidationHandler()
        
        # Test prompt hash generation
        test_prompt = "Test prompt for hashing"
        hash1 = handler._generate_prompt_hash(test_prompt)
        hash2 = handler._generate_prompt_hash(test_prompt)
        
        # Same prompt should generate same hash
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 8)  # Should be 8 characters
        
        # Different prompts should generate different hashes
        different_prompt = "Different test prompt"
        hash3 = handler._generate_prompt_hash(different_prompt)
        self.assertNotEqual(hash1, hash3)

    def test_enhanced_error_logging_validation_failures(self):
        """Test enhanced error logging for validation failures."""
        handler = PromptValidationHandler()
        
        # Test empty prompt
        result = handler.validate_prompt("")
        self.assertFalse(result["valid"])
        self.assertIn("prompt_hash", result)
        self.assertIn("timestamp", result)
        self.assertIn("error", result)
        
        # Test short prompt
        result = handler.validate_prompt("Short")
        self.assertFalse(result["valid"])
        self.assertIn("prompt_hash", result)
        self.assertIn("timestamp", result)
        
        # Test long prompt
        long_prompt = "A" * 1500
        result = handler.validate_prompt(long_prompt)
        self.assertFalse(result["valid"])
        self.assertIn("prompt_hash", result)
        self.assertIn("timestamp", result)

    def test_routing_failure_handling(self):
        """Test routing failure handling with detailed logging."""
        handler = PromptValidationHandler()
        
        test_prompt = "Test prompt for routing failure"
        test_error = ValueError("Test routing error")
        test_context = {"agent": "test_agent", "confidence": 0.5}
        
        result = handler.handle_routing_failure(test_prompt, test_error, test_context)
        
        self.assertFalse(result["success"])
        self.assertIn("prompt_hash", result)
        self.assertIn("timestamp", result)
        self.assertIn("traceback", result)
        self.assertIn("error_type", result)
        self.assertEqual(result["error_type"], "ValueError")
        self.assertIn("suggestions", result)

    def test_gpt_fallback_failure_handling(self):
        """Test GPT fallback failure handling with detailed logging."""
        handler = PromptValidationHandler()
        
        test_prompt = "Test prompt for GPT fallback failure"
        test_error = ConnectionError("API connection failed")
        fallback_attempts = 3
        
        result = handler.handle_gpt_fallback_failure(test_prompt, test_error, fallback_attempts)
        
        self.assertFalse(result["success"])
        self.assertIn("prompt_hash", result)
        self.assertIn("timestamp", result)
        self.assertIn("traceback", result)
        self.assertIn("error_type", result)
        self.assertEqual(result["error_type"], "ConnectionError")
        self.assertEqual(result["fallback_attempts"], 3)
        self.assertIn("suggestions", result)

    def test_successful_prompt_validation(self):
        """Test successful prompt validation with logging."""
        handler = PromptValidationHandler()
        
        valid_prompt = "Forecast the price of AAPL for the next 7 days"
        result = handler.validate_prompt(valid_prompt)
        
        self.assertTrue(result["valid"])
        self.assertIn("prompt_hash", result)
        self.assertIn("timestamp", result)
        self.assertIn("intent", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["intent"], "forecast")

    def test_model_scoring_with_decay(self):
        """Test model scoring with rolling decay enabled."""
        scorer = ModelScorer({"enable_rolling_decay": True})
        model_name = "test_model"
        timestamp = datetime.now()
        
        # Test scoring with decay
        scores = scorer.model_score(
            y_true=self.test_actuals,
            y_pred=self.test_predictions,
            returns=self.test_returns,
            model_name=model_name,
            timestamp=timestamp
        )
        
        self.assertIn("mse", scores)
        self.assertIn("sharpe", scores)
        self.assertIsInstance(scores["mse"], float)
        self.assertIsInstance(scores["sharpe"], float)
        
        # Check that history was updated
        self.assertIn(model_name, scorer.score_history)
        self.assertEqual(len(scorer.score_history[model_name]), 1)

    def test_clear_score_history(self):
        """Test clearing score history functionality."""
        scorer = ModelScorer()
        model_name = "test_model"
        
        # Add some history
        scorer.score_history[model_name] = [(datetime.now(), {"mse": 1.0})]
        scorer.score_history["other_model"] = [(datetime.now(), {"mse": 2.0})]
        
        # Clear specific model
        scorer.clear_history(model_name)
        self.assertNotIn(model_name, scorer.score_history)
        self.assertIn("other_model", scorer.score_history)
        
        # Clear all history
        scorer.clear_history()
        self.assertEqual(len(scorer.score_history), 0)

    def test_error_logging_with_exception_handling(self):
        """Test error logging with exception handling."""
        handler = PromptValidationHandler()
        
        # Test with an exception during validation
        with patch.object(handler, '_has_action_words', side_effect=Exception("Test exception")):
            result = handler.validate_prompt("Test prompt")
            
            self.assertFalse(result["valid"])
            self.assertIn("prompt_hash", result)
            self.assertIn("timestamp", result)
            self.assertIn("traceback", result)
            self.assertIn("error", result)

    @patch('trading.nlp.sentiment_processor.EMBEDDINGS_AVAILABLE', True)
    @patch('trading.nlp.sentiment_processor.SentenceTransformer', create=True)
    @patch('trading.nlp.sentiment_processor.cosine_similarity', create=True)
    def test_soft_matching_threshold_behavior(self, mock_cosine, mock_transformer):
        """Test soft-matching threshold behavior."""
        # Mock low similarity scores
        mock_cosine.return_value = np.array([[0.3, 0.2, 0.1]])  # Below threshold
        
        processor = SentimentProcessor(enable_soft_matching=True)
        processor.soft_match_threshold = 0.7
        
        # Test with low similarity
        soft_matches = processor._find_soft_matches("test_word")
        
        # Should return empty list for low similarity
        self.assertEqual(len(soft_matches), 0)

    def test_rolling_decay_with_insufficient_history(self):
        """Test rolling decay behavior with insufficient history."""
        scorer = ModelScorer({"enable_rolling_decay": True, "min_window_size": 10})
        model_name = "test_model"
        
        # Add only a few scores (less than min_window_size)
        for i in range(5):
            ts = datetime.now() + timedelta(minutes=i)
            scores = {"mse": 1.0 + i * 0.1}
            scorer.score_history[model_name] = scorer.score_history.get(model_name, [])
            scorer.score_history[model_name].append((ts, scores))
        
        # Test applying decay with insufficient history
        current_scores = {"mse": 1.5}
        decayed_scores = scorer._apply_rolling_decay(model_name, current_scores, datetime.now())
        
        # Should return current scores unchanged
        self.assertEqual(decayed_scores["mse"], current_scores["mse"])


if __name__ == "__main__":
    unittest.main()
