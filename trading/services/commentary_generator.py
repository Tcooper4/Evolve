"""
Commentary Generator Module

Handles generation of GPT-powered and fallback commentary for trading analysis results.
"""

import json
import logging
from typing import Any, Dict, Optional

import openai

logger = logging.getLogger(__name__)


class CommentaryGenerator:
    """
    Generates commentary on trading analysis results.

    Supports both GPT-powered commentary and fallback text-based commentary
    when GPT is not available.
    """

    def __init__(self, openai_api_key: str = None, max_tokens: int = 8000):
        """
        Initialize the commentary generator.

        Args:
            openai_api_key: OpenAI API key for GPT commentary
            max_tokens: Maximum tokens for response truncation
        """
        self.openai_api_key = openai_api_key
        self.max_tokens = max_tokens
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            import os

            openai.api_key = os.getenv("OPENAI_API_KEY")

    def truncate_response(self, response: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate response to specified token limit.

        Args:
            response: Response text to truncate
            max_tokens: Maximum tokens (uses instance default if None)

        Returns:
            Truncated response
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            # Try to use tokenizer if available
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 tokenizer
            tokens = tokenizer.encode(response)

            if len(tokens) <= max_tokens:
                return response

            # Truncate to max_tokens
            truncated_tokens = tokens[:max_tokens]
            truncated_response = tokenizer.decode(truncated_tokens)

            logger.info(
                f"Response truncated from {len(tokens)} to {len(truncated_tokens)} tokens"
            )
            return truncated_response

        except ImportError:
            # Fallback to character-based estimation (rough approximation)
            # Assume ~4 characters per token on average
            estimated_tokens = len(response) // 4

            if estimated_tokens <= max_tokens:
                return response

            # Truncate to estimated token limit
            max_chars = max_tokens * 4
            truncated_response = response[:max_chars]

            logger.info(
                f"Response truncated from ~{estimated_tokens} to ~{max_tokens} tokens (estimated)"
            )
            return truncated_response

        except Exception as e:
            logger.warning(
                f"Error in token counting, using character-based truncation: {e}"
            )
            # Fallback to simple character truncation
            max_chars = max_tokens * 4
            return response[:max_chars]

    def generate_commentary(
        self, query: str, parsed: Dict[str, Any], result: Dict[str, Any]
    ) -> str:
        """
        Generate commentary on the analysis results.

        Args:
            query: Original user query
            parsed: Parsed query parameters
            result: Analysis results

        Returns:
            Generated commentary
        """
        if not openai.api_key:
            return self._generate_fallback_commentary(query, parsed, result)

        try:
            commentary = self._generate_gpt_commentary(query, parsed, result)
            # Apply token truncation
            return self.truncate_response(commentary)
        except Exception as e:
            logger.error(f"Error generating GPT commentary: {e}")
            return self._generate_fallback_commentary(query, parsed, result)

    def _generate_gpt_commentary(
        self, query: str, parsed: Dict[str, Any], result: Dict[str, Any]
    ) -> str:
        """Generate GPT commentary on the analysis results."""
        # Prepare context for GPT
        context = {
            "query": query,
            "intent": parsed.get("intent"),
            "symbol": parsed.get("symbol"),
            "timeframe": parsed.get("timeframe"),
            "period": parsed.get("period"),
            "results": result,
        }

        system_prompt = """
        You are a quantitative trading analyst providing commentary on trading decisions and model recommendations.

        Your role is to:
        1. Explain the analysis results in clear, professional language
        2. Provide insights into why certain decisions were made
        3. Highlight key metrics and their significance
        4. Offer actionable recommendations
        5. Mention any risks or limitations

        Be concise but comprehensive. Use financial terminology appropriately.
        Focus on the most important findings and their implications for trading decisions.
        """

        user_prompt = f"""
        User Query: {query}

        Analysis Results:
        {json.dumps(context, indent=2)}

        Please provide a comprehensive commentary on these results, explaining the key findings and their implications for trading decisions.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        return response.choices[0].message.content

    def _generate_fallback_commentary(
        self, query: str, parsed: Dict[str, Any], result: Dict[str, Any]
    ) -> str:
        """Generate fallback commentary without GPT."""
        intent = parsed.get("intent")
        symbol = parsed.get("symbol")

        if intent == "model_recommendation":
            best_model = result.get("best_model")
            if best_model:
                return f"Based on our analysis, the {
                    best_model['model_type'].upper()} model shows the best performance for {symbol} with an overall score of {
                    best_model['evaluation'].get(
                        'overall_score',
                        0):.2f}. This model has been evaluated across multiple metrics including Sharpe ratio, maximum drawdown, and win rate."
            else:
                return f"Analysis completed for {symbol}, but no suitable model was found. Consider adjusting parameters or timeframes."

        elif intent == "trading_signal":
            signal = result.get("signal", {})
            if signal:
                return f"Trading signal for {symbol}: {
                    signal['signal']} ({
                    signal['strength']} confidence). {
                    signal['reasoning']} Model performance score: {
                    signal['model_score']:.2f}"
            else:
                return f"Unable to generate trading signal for {symbol}. Please check model availability and data quality."

        elif intent == "market_analysis":
            return f"Comprehensive market analysis completed for {symbol}. The analysis includes model performance evaluation, market data trends, and technical indicators. Review the generated plots for visual insights."

        else:
            return f"Analysis completed for {symbol}. The system processed your query and generated relevant insights based on available data and models."
