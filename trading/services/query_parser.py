"""
Query Parser Module

Handles natural language query parsing for the QuantGPT interface.
Supports both GPT-based and regex-based parsing with fallback mechanisms.
"""

import json
import logging
import re
from typing import Any, Dict

import openai

logger = logging.getLogger(__name__)


class QueryParser:
    """
    Natural language query parser for trading system queries.

    Supports intent classification, parameter extraction, and validation
    with both GPT and regex-based parsing methods.
    """

    def __init__(self, openai_api_key: str = None):
        """
        Initialize the query parser.

        Args:
            openai_api_key: OpenAI API key for GPT parsing
        """
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            import os

            openai.api_key = os.getenv("OPENAI_API_KEY")

        # Trading system context
        self.trading_context = {
            "available_symbols": ["BTCUSDT", "ETHUSDT", "NVDA", "TSLA", "AAPL", "GOOGL", "MSFT", "AMZN"],
            "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "available_periods": ["7d", "14d", "30d", "90d", "180d", "1y"],
            "available_models": ["lstm", "xgboost", "ensemble", "transformer", "tcn"],
        }

        # Intent keywords mapping
        self.intent_keywords = {
            "model_recommendation": ["best", "model", "recommend", "which model", "top model"],
            "trading_signal": ["long", "short", "buy", "sell", "trade", "signal", "position"],
            "market_analysis": ["analyze", "analysis", "market", "trend", "technical"],
            "general_query": ["what", "how", "when", "why", "explain", "show"],
        }

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to extract intent and parameters.

        Args:
            query: Natural language query

        Returns:
            Parsed query with intent and parameters
        """
        try:
            # Use OpenAI to parse the query if available
            if openai.api_key:
                return self._parse_with_gpt(query)
            else:
                return self._parse_with_regex(query)
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return self._parse_with_regex(query)

    def _parse_with_gpt(self, query: str) -> Dict[str, Any]:
        """Parse query using GPT for better understanding."""
        try:
            system_prompt = """
            You are a trading system query parser. Extract the following information from user queries:
            - intent: The main action requested (model_recommendation, trading_signal, market_analysis, general_query)
            - symbol: The trading symbol/asset (e.g., NVDA, TSLA, BTCUSDT)
            - timeframe: The time interval (1m, 5m, 15m, 1h, 4h, 1d)
            - period: The analysis period (7d, 14d, 30d, 90d, 180d, 1y)
            - model_type: Specific model type if mentioned (lstm, xgboost, ensemble, etc.)
            - confidence: Confidence score (0-1) for the parsing

            Available symbols: BTCUSDT, ETHUSDT, NVDA, TSLA, AAPL, GOOGL, MSFT, AMZN
            Available timeframes: 1m, 5m, 15m, 1h, 4h, 1d
            Available periods: 7d, 14d, 30d, 90d, 180d, 1y
            Available models: lstm, xgboost, ensemble, transformer, tcn

            Return only valid JSON with the extracted information.
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0.1,
                max_tokens=200,
            )

            parsed = json.loads(response.choices[0].message.content)

            # Validate and clean the parsed data
            return self._validate_parsed_data(parsed)

        except Exception as e:
            logger.error(f"GPT parsing failed: {e}")
            return self._parse_with_regex(query)

    def _parse_with_regex(self, query: str) -> Dict[str, Any]:
        """Parse query using regex patterns as fallback."""
        query_lower = query.lower()

        # Extract symbol
        symbol_pattern = r"\b(BTCUSDT|ETHUSDT|NVDA|TSLA|AAPL|GOOGL|MSFT|AMZN)\b"
        symbol_match = re.search(symbol_pattern, query_lower)
        symbol = symbol_match.group(1) if symbol_match else None

        # Extract timeframe
        timeframe_pattern = r"\b(1m|5m|15m|1h|4h|1d)\b"
        timeframe_match = re.search(timeframe_pattern, query_lower)
        timeframe = timeframe_match.group(1) if timeframe_match else "1h"

        # Extract period
        period_pattern = r"\b(7d|14d|30d|90d|180d|1y)\b"
        period_match = re.search(period_pattern, query_lower)
        period = period_match.group(1) if period_match else "30d"

        # Extract model type
        model_pattern = r"\b(lstm|xgboost|ensemble|transformer|tcn)\b"
        model_match = re.search(model_pattern, query_lower)
        model_type = model_match.group(1) if model_match else None

        # Determine intent
        intent = self._classify_intent(query_lower)

        return {
            "intent": intent,
            "symbol": symbol,
            "timeframe": timeframe,
            "period": period,
            "model_type": model_type,
            "confidence": 0.7,
        }

    def _classify_intent(self, query_lower: str) -> str:
        """Classify the intent of the query based on keywords."""
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        return "general_query"

    def _validate_parsed_data(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean parsed data."""
        # Ensure required fields exist
        required_fields = ["intent", "symbol", "timeframe", "period"]
        for field in required_fields:
            if field not in parsed:
                parsed[field] = None

        # Validate symbol
        if parsed["symbol"] and parsed["symbol"] not in self.trading_context["available_symbols"]:
            parsed["symbol"] = None

        # Validate timeframe
        if parsed["timeframe"] and parsed["timeframe"] not in self.trading_context["available_timeframes"]:
            parsed["timeframe"] = "1h"

        # Validate period
        if parsed["period"] and parsed["period"] not in self.trading_context["available_periods"]:
            parsed["period"] = "30d"

        # Validate model type
        if parsed.get("model_type") and parsed["model_type"] not in self.trading_context["available_models"]:
            parsed["model_type"] = None

        # Set default confidence if not provided
        if "confidence" not in parsed:
            parsed["confidence"] = 0.8

        return parsed
