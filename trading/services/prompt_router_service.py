"""
Prompt Router Service

Service wrapper for the PromptRouterAgent, handling prompt routing and intent detection
via Redis pub/sub communication.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from trading.agents.prompt_router_agent import PromptRouterAgent
from trading.memory.agent_memory import AgentMemory
from trading.services.base_service import BaseService

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


logger = logging.getLogger(__name__)


class PromptRouterService(BaseService):
    """
    Service wrapper for PromptRouterAgent.

    Handles prompt routing and intent detection requests and communicates results via Redis.
    """

    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0
    ):
        """Initialize the PromptRouterService."""
        super().__init__("prompt_router", redis_host, redis_port, redis_db)

        # Initialize the agent
        self.agent = PromptRouterAgent()
        self.memory = AgentMemory()

        logger.info("PromptRouterService initialized")

    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming prompt routing requests.

        Args:
            data: Message data containing prompt routing request

        Returns:
            Response with routing results or error
        """
        try:
            message_type = data.get("type", "")

            if message_type == "route_prompt":
                return self._handle_route_request(data)
            elif message_type == "detect_intent":
                return self._handle_intent_detection(data)
            elif message_type == "parse_arguments":
                return self._handle_argument_parsing(data)
            elif message_type == "get_routing_history":
                return self._handle_history_request(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {
                    "type": "error",
                    "error": f"Unknown message type: {message_type}",
                    "original_message": data,
                }

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"type": "error", "error": str(e), "original_message": data}

    def _handle_route_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompt routing request."""
        try:
            route_data = data.get("data", {})

            # Extract routing parameters
            user_prompt = route_data.get("user_prompt")
            context = route_data.get("context", {})
            available_agents = route_data.get("available_agents", [])

            if not user_prompt:
                return {
                    "type": "error",
                    "error": "user_prompt is required",
                    "suggestions": [
                        "Show me the latest trades for AAPL",
                        "What is the Sharpe ratio of my portfolio?",
                        "Switch to momentum strategy",
                        "Summarize today's market news",
                        "Plot the equity curve for TSLA",
                    ],
                }

            logger.info(f"Routing prompt: {user_prompt[:50]}...")

            # Route the prompt using the agent
            result = self.agent.route_prompt(
                user_prompt=user_prompt,
                context=context,
                available_agents=available_agents,
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="prompt_router",
                decision_type="route_prompt",
                details={
                    "user_prompt_length": len(user_prompt),
                    "detected_intent": result.get("intent"),
                    "target_agent": result.get("target_agent"),
                    "confidence": result.get("confidence", 0),
                },
            )

            return {"type": "prompt_routed", "result": result}

        except Exception as e:
            logger.error(f"Error routing prompt: {e}")
            return {"type": "error", "error": str(e)}

    def _handle_intent_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle intent detection request."""
        try:
            intent_data = data.get("data", {})

            # Extract parameters
            user_prompt = intent_data.get("user_prompt")
            use_openai = intent_data.get("use_openai", True)

            if not user_prompt:
                return {
                    "type": "error",
                    "error": "user_prompt is required",
                    "suggestions": [
                        "Show me the latest trades for AAPL",
                        "What is the Sharpe ratio of my portfolio?",
                        "Switch to momentum strategy",
                        "Summarize today's market news",
                        "Plot the equity curve for TSLA",
                    ],
                }

            logger.info(f"Detecting intent for: {user_prompt[:50]}...")

            # Detect intent using the agent
            result = self.agent.detect_intent(
                user_prompt=user_prompt, use_openai=use_openai
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="prompt_router",
                decision_type="detect_intent",
                details={
                    "user_prompt_length": len(user_prompt),
                    "detected_intent": result.get("intent"),
                    "confidence": result.get("confidence", 0),
                    "use_openai": use_openai,
                },
            )

            return {"type": "intent_detected", "result": result}

        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            return {"type": "error", "error": str(e)}

    def _handle_argument_parsing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle argument parsing request."""
        try:
            parse_data = data.get("data", {})

            # Extract parameters
            user_prompt = parse_data.get("user_prompt")
            intent = parse_data.get("intent")
            use_openai = parse_data.get("use_openai", True)

            if not user_prompt or not intent:
                return {
                    "type": "error",
                    "error": "user_prompt and intent are required",
                    "suggestions": [
                        "Show me the latest trades for AAPL",
                        "What is the Sharpe ratio of my portfolio?",
                        "Switch to momentum strategy",
                        "Summarize today's market news",
                        "Plot the equity curve for TSLA",
                    ],
                }

            logger.info(f"Parsing arguments for intent: {intent}")

            # Parse arguments using the agent
            result = self.agent.parse_arguments(
                user_prompt=user_prompt, intent=intent, use_openai=use_openai
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="prompt_router",
                decision_type="parse_arguments",
                details={
                    "user_prompt_length": len(user_prompt),
                    "intent": intent,
                    "parsed_arguments": result.get("arguments", {}),
                    "use_openai": use_openai,
                },
            )

            return {"type": "arguments_parsed", "result": result}

        except Exception as e:
            logger.error(f"Error parsing arguments: {e}")
            return {"type": "error", "error": str(e)}

    def _handle_history_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle routing history request."""
        try:
            history_data = data.get("data", {})

            # Extract parameters
            intent = history_data.get("intent")
            limit = history_data.get("limit", 10)

            # Get routing history
            history = self.agent.get_routing_history(intent=intent, limit=limit)

            return {
                "type": "routing_history",
                "history": history,
                "intent": intent,
                "count": len(history),
            }

        except Exception as e:
            logger.error(f"Error getting routing history: {e}")
            return {"type": "error", "error": str(e)}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            memory_stats = self.memory.get_stats()

            # Get recent routing activities
            recent_routing = [
                entry
                for entry in memory_stats.get("recent_decisions", [])
                if entry.get("agent_name") == "prompt_router"
            ]

            # Count by type
            routing_types = {}
            for routing in recent_routing:
                routing_type = routing.get("decision_type", "unknown")
                routing_types[routing_type] = routing_types.get(routing_type, 0) + 1

            # Count by intent
            intent_counts = {}
            for routing in recent_routing:
                intent = routing.get("details", {}).get("detected_intent", "unknown")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            return {
                "total_routing_activities": len(recent_routing),
                "routing_types": routing_types,
                "intent_counts": intent_counts,
                "memory_entries": memory_stats.get("total_entries", 0),
                "recent_routing": recent_routing[:5],
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"error": str(e)}
