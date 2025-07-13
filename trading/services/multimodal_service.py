"""
Multimodal Service

Service wrapper for the MultimodalAgent, handling plotting and vision analysis requests
via Redis pub/sub communication.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trading.agents.multimodal_agent import MultimodalAgent
from trading.memory.agent_memory import AgentMemory
from trading.services.base_service import BaseService

logger = logging.getLogger(__name__)


class MultimodalService(BaseService):
    """
    Service wrapper for MultimodalAgent.

    Handles plotting and vision analysis requests and communicates results via Redis.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        """Initialize the MultimodalService."""
        super().__init__("multimodal", redis_host, redis_port, redis_db)

        # Initialize the agent
        self.agent = MultimodalAgent()
        self.memory = AgentMemory()

        logger.info("MultimodalService initialized")

    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming multimodal requests.

        Args:
            data: Message data containing multimodal request

        Returns:
            Response with analysis results or error
        """
        try:
            message_type = data.get("type", "")

            if message_type == "generate_plot":
                return self._handle_plot_request(data)
            elif message_type == "analyze_image":
                return self._handle_image_analysis(data)
            elif message_type == "generate_insights":
                return self._handle_insights_request(data)
            elif message_type == "get_plot_history":
                return self._handle_history_request(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {"type": "error", "error": f"Unknown message type: {message_type}", "original_message": data}

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"type": "error", "error": str(e), "original_message": data}

    def _handle_plot_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plot generation request."""
        try:
            plot_data = data.get("data", {})

            # Extract plot parameters
            plot_type = plot_data.get("plot_type")
            data_source = plot_data.get("data_source")
            plot_config = plot_data.get("plot_config", {})
            save_path = plot_data.get("save_path")

            if not plot_type or not data_source:
                return {"type": "error", "error": "plot_type and data_source are required"}

            logger.info(f"Generating {plot_type} plot")

            # Generate plot using the agent
            result = self.agent.generate_plot(
                plot_type=plot_type, data_source=data_source, plot_config=plot_config, save_path=save_path
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="multimodal",
                decision_type="generate_plot",
                details={
                    "plot_type": plot_type,
                    "data_source": data_source,
                    "save_path": save_path,
                    "success": result.get("success", False),
                },
            )

            return {
                "type": "plot_generated",
                "result": result,
                "status": "success" if result.get("success") else "failed",
            }

        except Exception as e:
            logger.error(f"Error generating plot: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def _handle_image_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle image analysis request."""
        try:
            analysis_data = data.get("data", {})

            # Extract analysis parameters
            image_path = analysis_data.get("image_path")
            analysis_type = analysis_data.get("analysis_type", "general")
            model_name = analysis_data.get("model_name", "gpt-4v")

            if not image_path:
                return {"type": "error", "error": "image_path is required"}

            logger.info(f"Analyzing image: {image_path}")

            # Analyze image using the agent
            result = self.agent.analyze_image(image_path=image_path, analysis_type=analysis_type, model_name=model_name)

            # Log to memory
            self.memory.log_decision(
                agent_name="multimodal",
                decision_type="analyze_image",
                details={
                    "image_path": image_path,
                    "analysis_type": analysis_type,
                    "model_name": model_name,
                    "analysis_length": len(result.get("analysis", "")),
                },
            )

            return {"type": "image_analyzed", "result": result}

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"type": "error", "error": str(e)}

    def _handle_insights_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle insights generation request."""
        try:
            insights_data = data.get("data", {})

            # Extract parameters
            data_source = insights_data.get("data_source")
            insight_type = insights_data.get("insight_type", "performance")
            include_plots = insights_data.get("include_plots", True)

            if not data_source:
                return {"type": "error", "error": "data_source is required"}

            logger.info(f"Generating {insight_type} insights")

            # Generate insights using the agent
            result = self.agent.generate_insights(
                data_source=data_source, insight_type=insight_type, include_plots=include_plots
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="multimodal",
                decision_type="generate_insights",
                details={
                    "data_source": data_source,
                    "insight_type": insight_type,
                    "include_plots": include_plots,
                    "insights_count": len(result.get("insights", [])),
                },
            )

            return {"type": "insights_generated", "result": result}

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"type": "error", "error": str(e)}

    def _handle_history_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plot history request."""
        try:
            history_data = data.get("data", {})

            # Extract parameters
            plot_type = history_data.get("plot_type")
            limit = history_data.get("limit", 10)

            # Get plot history
            history = self.agent.get_plot_history(plot_type=plot_type, limit=limit)

            return {"type": "plot_history", "history": history, "plot_type": plot_type, "count": len(history)}

        except Exception as e:
            logger.error(f"Error getting plot history: {e}")
            return {"type": "error", "error": str(e)}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            memory_stats = self.memory.get_stats()

            # Get recent multimodal activities
            recent_activities = [
                entry for entry in memory_stats.get("recent_decisions", []) if entry.get("agent_name") == "multimodal"
            ]

            # Count by type
            activity_types = {}
            for activity in recent_activities:
                activity_type = activity.get("decision_type", "unknown")
                activity_types[activity_type] = activity_types.get(activity_type, 0) + 1

            return {
                "total_multimodal_activities": len(recent_activities),
                "activity_types": activity_types,
                "memory_entries": memory_stats.get("total_entries", 0),
                "recent_activities": recent_activities[:5],
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"error": str(e)}
