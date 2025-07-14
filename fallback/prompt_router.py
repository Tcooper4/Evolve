"""
Fallback Prompt Router Implementation

Provides fallback functionality for prompt interpretation and routing
when the primary prompt router is unavailable.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FallbackPromptRouter:
    """
    Fallback implementation of the Prompt Router.

    Provides basic prompt interpretation and routing functionality when
    the primary prompt router is unavailable. Uses keyword-based routing
    and simple intent detection.
    """

    def __init__(self) -> None:
        """
        Initialize the fallback prompt router.

        Sets up basic logging and initializes routing patterns for
        fallback operations.
        """
        self._routing_patterns = self._initialize_routing_patterns()
        self._status = "fallback"
        logger.info("FallbackPromptRouter initialized")

    def _initialize_routing_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize routing patterns for keyword-based routing.

        Returns:
            Dict[str, Dict[str, Any]]: Routing patterns and their configurations
        """
        return {
            "forecast": {
                "keywords": [
                    "forecast",
                    "predict",
                    "prediction",
                    "model",
                    "future",
                    "trend",
                ],
                "confidence": 0.7,
                "agent": "forecast_agent",
                "description": "Price forecasting and prediction requests",
            },
            "strategy": {
                "keywords": ["strategy", "trade", "signal", "buy", "sell", "position"],
                "confidence": 0.7,
                "agent": "strategy_agent",
                "description": "Trading strategy and signal requests",
            },
            "portfolio": {
                "keywords": ["portfolio", "position", "risk", "allocation", "balance"],
                "confidence": 0.6,
                "agent": "portfolio_agent",
                "description": "Portfolio management and risk requests",
            },
            "analysis": {
                "keywords": ["analyze", "analysis", "report", "performance", "metrics"],
                "confidence": 0.5,
                "agent": "analysis_agent",
                "description": "Market analysis and reporting requests",
            },
            "backtest": {
                "keywords": [
                    "backtest",
                    "historical",
                    "test",
                    "simulate",
                    "performance",
                ],
                "confidence": 0.6,
                "agent": "backtest_agent",
                "description": "Backtesting and historical analysis requests",
            },
        }

    def route_prompt(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a prompt to appropriate agents (fallback implementation).

        Args:
            prompt: The user prompt to route
            context: Optional context information

        Returns:
            Dict[str, Any]: Routing result with agent and confidence information
        """
        try:
            logger.info(f"Routing prompt: {prompt[:50]}...")

            # Extract intent and entities
            intent = self._extract_intent(prompt)
            entities = self._extract_entities(prompt)

            # Determine best route
            route = self._determine_route(intent, entities, context)

            # Log the routing decision
            routing_result = {
                "prompt": prompt,
                "intent": intent,
                "entities": entities,
                "route": route,
                "confidence": route.get("confidence", 0.3),
                "timestamp": datetime.now().isoformat(),
                "fallback_mode": True,
            }

            logger.info(
                f"Routed to {route.get('agent', 'unknown')} with confidence {route.get('confidence', 0.3)}"
            )
            return routing_result

        except Exception as e:
            logger.error(f"Error in fallback prompt routing: {e}")
            return {
                "prompt": prompt,
                "intent": "unknown",
                "entities": {},
                "route": {
                    "agent": "general_agent",
                    "confidence": 0.1,
                    "reasoning": "Fallback routing due to error",
                },
                "confidence": 0.1,
                "timestamp": datetime.now().isoformat(),
                "fallback_mode": True,
                "error": str(e),
            }

    def _extract_intent(self, prompt: str) -> str:
        """
        Extract intent from prompt using keyword matching.

        Args:
            prompt: The user prompt

        Returns:
            str: Detected intent
        """
        try:
            prompt_lower = prompt.lower()

            # Score each pattern
            scores = {}
            for intent, pattern in self._routing_patterns.items():
                score = 0
                for keyword in pattern["keywords"]:
                    if keyword in prompt_lower:
                        score += 1
                scores[intent] = score

            # Return highest scoring intent
            if scores:
                best_intent = max(scores, key=scores.get)
                if scores[best_intent] > 0:
                    return best_intent

            return "general"

        except Exception as e:
            logger.error(f"Error extracting intent: {e}")
            return "general"

    def _extract_entities(self, prompt: str) -> Dict[str, Any]:
        """
        Extract entities from prompt using regex patterns.

        Args:
            prompt: The user prompt

        Returns:
            Dict[str, Any]: Extracted entities
        """
        try:
            entities = {}

            # Extract stock symbols (uppercase letters, typically 1-5 chars)
            symbol_pattern = r"\b[A-Z]{1,5}\b"
            symbols = re.findall(symbol_pattern, prompt)
            if symbols:
                entities["symbols"] = symbols

            # Extract time periods
            time_patterns = {
                "days": r"(\d+)\s*days?",
                "weeks": r"(\d+)\s*weeks?",
                "months": r"(\d+)\s*months?",
                "years": r"(\d+)\s*years?",
            }

            for time_unit, pattern in time_patterns.items():
                matches = re.findall(pattern, prompt.lower())
                if matches:
                    entities[time_unit] = [int(m) for m in matches]

            # Extract model types
            model_keywords = ["lstm", "xgboost", "prophet", "arima", "ensemble"]
            found_models = [
                model for model in model_keywords if model in prompt.lower()
            ]
            if found_models:
                entities["models"] = found_models

            # Extract strategy types
            strategy_keywords = ["rsi", "macd", "bollinger", "sma", "ema"]
            found_strategies = [
                strategy for strategy in strategy_keywords if strategy in prompt.lower()
            ]
            if found_strategies:
                entities["strategies"] = found_strategies

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}

    def _determine_route(
        self, intent: str, entities: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determine the best route based on intent and entities.

        Args:
            intent: The detected intent
            entities: Extracted entities
            context: Optional context information

        Returns:
            Dict[str, Any]: Route information
        """
        try:
            if intent in self._routing_patterns:
                pattern = self._routing_patterns[intent]
                return {
                    "agent": pattern["agent"],
                    "confidence": pattern["confidence"],
                    "reasoning": pattern["description"],
                    "entities": entities,
                }
            else:
                return {
                    "agent": "general_agent",
                    "confidence": 0.3,
                    "reasoning": "Default routing for unknown intent",
                    "entities": entities,
                }

        except Exception as e:
            logger.error(f"Error determining route: {e}")
            return {
                "agent": "general_agent",
                "confidence": 0.1,
                "reasoning": "Error in route determination",
                "entities": entities,
            }

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback prompt router.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            return {
                "status": self._status,
                "available_patterns": len(self._routing_patterns),
                "patterns": list(self._routing_patterns.keys()),
                "fallback_mode": True,
                "message": "Using fallback prompt router",
            }
        except Exception as e:
            logger.error(f"Error getting fallback prompt router health: {e}")
            return {
                "status": "error",
                "available_patterns": 0,
                "fallback_mode": True,
                "error": str(e),
            }
