"""
Fallback Agent Hub Implementation

Provides fallback functionality for agent routing and management when
the primary agent hub is unavailable.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FallbackAgentHub:
    """
    Fallback implementation of the Agent Hub.

    Provides basic agent routing and management functionality when the
    primary agent hub is unavailable. This ensures the system remains
    functional in degraded conditions.
    """

    def __init__(self) -> None:
        """
        Initialize the fallback agent hub.

        Sets up basic logging and initializes internal state for
        fallback operations.
        """
        self._interactions: List[Dict[str, Any]] = []
        self._status = "fallback"
        logger.info("FallbackAgentHub initialized")

    def route(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Route a prompt to appropriate agents (fallback implementation).

        Args:
            prompt: The user prompt to route

        Returns:
            Optional[Dict[str, Any]]: Basic routing result or None if routing fails
        """
        try:
            logger.debug(f"Routing prompt: {prompt[:50]}...")

            # Basic keyword-based routing
            prompt_lower = prompt.lower()

            if any(word in prompt_lower for word in ["forecast", "predict", "model"]):
                return {
                    "agent": "forecast_agent",
                    "confidence": 0.5,
                    "reasoning": "Keyword-based routing to forecast agent",
                }
            elif any(word in prompt_lower for word in ["strategy", "trade", "signal"]):
                return {
                    "agent": "strategy_agent",
                    "confidence": 0.5,
                    "reasoning": "Keyword-based routing to strategy agent",
                }
            elif any(
                word in prompt_lower for word in ["portfolio", "position", "risk"]
            ):
                return {
                    "agent": "portfolio_agent",
                    "confidence": 0.5,
                    "reasoning": "Keyword-based routing to portfolio agent",
                }
            else:
                return {
                    "agent": "general_agent",
                    "confidence": 0.3,
                    "reasoning": "Default routing to general agent",
                }

        except Exception as e:
            logger.error(f"Error in fallback routing: {e}")
            return None

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback agent hub.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            return {
                "status": self._status,
                "available_agents": 4,  # forecast, strategy, portfolio, general
                "last_interaction": datetime.now().isoformat(),
                "fallback_mode": True,
                "message": "Using fallback agent hub",
            }
        except Exception as e:
            logger.error(f"Error getting fallback system health: {e}")
            return {
                "status": "error",
                "available_agents": 0,
                "fallback_mode": True,
                "error": str(e),
            }

    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent agent interactions (fallback implementation).

        Args:
            limit: Maximum number of interactions to return

        Returns:
            List[Dict[str, Any]]: List of recent interactions
        """
        try:
            logger.debug(f"Getting recent interactions (limit: {limit})")

            # Return mock interactions for fallback mode
            mock_interactions = [
                {
                    "timestamp": datetime.now().isoformat(),
                    "agent": "fallback_agent",
                    "action": "routing",
                    "status": "success",
                    "message": "Fallback routing completed",
                }
            ]

            return mock_interactions[:limit]

        except Exception as e:
            logger.error(f"Error getting recent interactions: {e}")
            return []

    def log_interaction(self, interaction: Dict[str, Any]) -> None:
        """
        Log an agent interaction (fallback implementation).

        Args:
            interaction: The interaction data to log
        """
        try:
            interaction["timestamp"] = datetime.now().isoformat()
            interaction["fallback_mode"] = True
            self._interactions.append(interaction)
            logger.debug(
                f"Logged fallback interaction: {interaction.get('action', 'unknown')}"
            )
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")

    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """
        Get status of a specific agent (fallback implementation).

        Args:
            agent_name: Name of the agent to check

        Returns:
            Dict[str, Any]: Agent status information
        """
        try:
            return {
                "name": agent_name,
                "status": "fallback",
                "available": True,
                "last_activity": datetime.now().isoformat(),
                "message": f"Agent {agent_name} running in fallback mode",
            }
        except Exception as e:
            logger.error(f"Error getting agent status for {agent_name}: {e}")
            return {
                "name": agent_name,
                "status": "error",
                "available": False,
                "error": str(e),
            }
