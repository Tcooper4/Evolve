#!/usr/bin/env python3
"""
Prompt Router Service Launcher

Launches the PromptRouterService as a standalone process.
Enhanced with graceful handling of malformed user prompts and example suggestions.
"""

import logging
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.prompt_router_service import PromptRouterService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/prompt_router_service.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class PromptValidationHandler:
    """Handles validation and suggestions for malformed user prompts."""

    def __init__(self):
        self.example_prompts = {
            "forecast": [
                "Forecast the price of AAPL for the next 7 days",
                "Predict TSLA stock movement for tomorrow",
                "What will be the closing price of MSFT next Friday?",
            ],
            "analysis": [
                "Analyze the technical indicators for NVDA",
                "What are the key support and resistance levels for AMD?",
                "Provide a fundamental analysis of GOOGL",
            ],
            "strategy": [
                "Create a momentum trading strategy for SPY",
                "Design a mean reversion strategy for QQQ",
                "Suggest a volatility-based strategy for VIX",
            ],
            "portfolio": [
                "Optimize my portfolio allocation for maximum Sharpe ratio",
                "Rebalance my portfolio with 60% stocks and 40% bonds",
                "Calculate the optimal position sizes for my holdings",
            ],
            "risk": [
                "Calculate the Value at Risk for my portfolio",
                "What is the maximum drawdown risk for AAPL?",
                "Analyze the correlation risk between my holdings",
            ],
        }

        self.prompt_patterns = {
            "forecast": r"\b(forecast|predict|price|movement|tomorrow|next|future)\b",
            "analysis": r"\b(analyze|analysis|technical|fundamental|support|resistance|indicator)\b",
            "strategy": r"\b(strategy|strategy|momentum|mean reversion|volatility|trading)\b",
            "portfolio": r"\b(portfolio|allocation|rebalance|position|sharpe|optimize)\b",
            "risk": r"\b(risk|var|drawdown|correlation|volatility|exposure)\b",
        }

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate user prompt and provide suggestions if malformed."""
        if not prompt or not isinstance(prompt, str):
            return {
                "valid": False,
                "error": "Prompt is empty or not a string",
                "suggestions": self._get_general_suggestions(),
            }

        prompt = prompt.strip()
        if len(prompt) < 10:
            return {
                "valid": False,
                "error": "Prompt is too short (minimum 10 characters)",
                "suggestions": self._get_general_suggestions(),
            }

        if len(prompt) > 1000:
            return {
                "valid": False,
                "error": "Prompt is too long (maximum 1000 characters)",
                "suggestions": ["Please provide a more concise prompt focusing on one specific request."],
            }

        # Check for basic structure
        if not self._has_action_words(prompt):
            return {
                "valid": False,
                "error": "Prompt lacks clear action words",
                "suggestions": self._get_action_suggestions(),
            }

        # Check for specific intent
        intent = self._detect_intent(prompt)
        if not intent:
            return {
                "valid": False,
                "error": "Unable to determine prompt intent",
                "suggestions": self._get_general_suggestions(),
            }

        return {"valid": True, "intent": intent, "confidence": self._calculate_confidence(prompt, intent)}

    def _has_action_words(self, prompt: str) -> bool:
        """Check if prompt contains action words."""
        action_words = [
            "forecast",
            "predict",
            "analyze",
            "calculate",
            "create",
            "design",
            "optimize",
            "rebalance",
            "suggest",
            "provide",
            "show",
            "get",
            "what",
            "how",
            "when",
            "where",
            "why",
        ]
        return any(word in prompt.lower() for word in action_words)

    def _detect_intent(self, prompt: str) -> Optional[str]:
        """Detect the intent of the prompt."""
        prompt_lower = prompt.lower()

        # Check patterns for each intent
        for intent, pattern in self.prompt_patterns.items():
            if re.search(pattern, prompt_lower):
                return intent

        return None

    def _calculate_confidence(self, prompt: str, intent: str) -> float:
        """Calculate confidence score for intent detection."""
        prompt_lower = prompt.lower()
        pattern = self.prompt_patterns[intent]
        matches = len(re.findall(pattern, prompt_lower))

        # Simple confidence calculation based on pattern matches
        if matches >= 2:
            return 0.9
        elif matches == 1:
            return 0.7
        else:
            return 0.5

    def _get_general_suggestions(self) -> List[str]:
        """Get general prompt suggestions."""
        suggestions = []
        for intent, examples in self.example_prompts.items():
            suggestions.extend(examples[:1])  # Add one example from each category
        return suggestions

    def _get_action_suggestions(self) -> List[str]:
        """Get suggestions for adding action words."""
        return [
            "Try starting with: 'Forecast', 'Analyze', 'Calculate', 'Create', or 'What is'",
            "Be specific about what you want to know or do",
            "Include the asset or topic you want to analyze",
        ]

    def get_examples_for_intent(self, intent: str) -> List[str]:
        """Get example prompts for a specific intent."""
        return self.example_prompts.get(intent, [])

    def format_error_response(self, validation_result: Dict[str, Any]) -> str:
        """Format a user-friendly error response with suggestions."""
        error_msg = f"❌ {validation_result['error']}\n\n"
        error_msg += "💡 Here are some example prompts you can try:\n\n"

        suggestions = validation_result.get("suggestions", [])
        for i, suggestion in enumerate(suggestions, 1):
            error_msg += f"{i}. {suggestion}\n"

        error_msg += "\n🔧 Tips for better prompts:\n"
        error_msg += "• Be specific about what you want to analyze\n"
        error_msg += "• Include the asset symbol or name\n"
        error_msg += "• Specify the time frame if relevant\n"
        error_msg += "• Use clear action words like 'forecast', 'analyze', 'calculate'\n"

        return error_msg


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, "service"):
        signal_handler.service.stop()
    sys.exit(0)


def main():
    """Main function to launch the PromptRouterService."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        logger.info("Starting PromptRouterService...")

        # Initialize prompt validation handler
        prompt_handler = PromptValidationHandler()

        # Initialize the service with prompt validation
        service = PromptRouterService(
            redis_host="localhost", redis_port=6379, redis_db=0, prompt_validator=prompt_handler
        )

        # Store service reference for signal handler
        signal_handler.service = service

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the service
        service.start()

        logger.info("PromptRouterService started successfully")
        logger.info(f"Listening on channels: {service.input_channel}, {service.control_channel}")
        logger.info("Prompt validation and suggestions enabled")

        # Keep the service running
        try:
            while service.is_running:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            service.stop()

    except Exception as e:
        logger.error(f"Error starting PromptRouterService: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
