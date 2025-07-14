"""
Performance Critic Service

Service wrapper for the PerformanceCriticAgent, handling model evaluation requests
via Redis pub/sub communication.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trading.agents.performance_critic_agent import PerformanceCriticAgent
from trading.memory.agent_memory import AgentMemory
from trading.services.base_service import BaseService

logger = logging.getLogger(__name__)


class PerformanceCriticService(BaseService):
    """
    Service wrapper for PerformanceCriticAgent.

    Handles model evaluation requests and communicates results via Redis.
    """

    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0
    ):
        """Initialize the PerformanceCriticService."""
        super().__init__("performance_critic", redis_host, redis_port, redis_db)

        # Initialize the agent
        self.agent = PerformanceCriticAgent()
        self.memory = AgentMemory()

        logger.info("PerformanceCriticService initialized")

    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming model evaluation requests.

        Args:
            data: Message data containing evaluation request

        Returns:
            Response with evaluation results or error
        """
        try:
            message_type = data.get("type", "")

            if message_type == "evaluate_model":
                return self._handle_evaluate_request(data)
            elif message_type == "compare_models":
                return self._handle_compare_request(data)
            elif message_type == "get_evaluation_history":
                return self._handle_history_request(data)
            elif message_type == "get_best_models":
                return self._handle_best_models_request(data)
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

    def _handle_evaluate_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model evaluation request."""
        try:
            eval_data = data.get("data", {})

            # Extract evaluation parameters
            model_id = eval_data.get("model_id")
            symbol = eval_data.get("symbol", "BTCUSDT")
            timeframe = eval_data.get("timeframe", "1h")
            period = eval_data.get("period", "30d")
            metrics = eval_data.get("metrics", ["sharpe", "drawdown", "win_rate"])

            if not model_id:
                return {"type": "error", "error": "model_id is required"}

            logger.info(f"Evaluating model {model_id} for {symbol}")

            # Evaluate the model using the agent
            evaluation = self.agent.evaluate_model(
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                metrics=metrics,
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="performance_critic",
                decision_type="evaluate_model",
                details={
                    "model_id": model_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "period": period,
                    "metrics": metrics,
                    "evaluation_score": evaluation.get("overall_score", 0),
                },
            )

            return {
                "type": "model_evaluated",
                "evaluation": evaluation,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def _handle_compare_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model comparison request."""
        try:
            compare_data = data.get("data", {})

            # Extract comparison parameters
            model_ids = compare_data.get("model_ids", [])
            symbol = compare_data.get("symbol", "BTCUSDT")
            timeframe = compare_data.get("timeframe", "1h")
            period = compare_data.get("period", "30d")

            if not model_ids or len(model_ids) < 2:
                return {
                    "type": "error",
                    "error": "At least 2 model_ids are required for comparison",
                }

            logger.info(f"Comparing models: {model_ids}")

            # Compare models using the agent
            comparison = self.agent.compare_models(
                model_ids=model_ids, symbol=symbol, timeframe=timeframe, period=period
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="performance_critic",
                decision_type="compare_models",
                details={
                    "model_ids": model_ids,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "period": period,
                    "winner_id": comparison.get("winner", {}).get("model_id"),
                },
            )

            return {
                "type": "models_compared",
                "comparison": comparison,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def _handle_history_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evaluation history request."""
        try:
            history_data = data.get("data", {})

            # Extract parameters
            model_id = history_data.get("model_id")
            limit = history_data.get("limit", 10)

            if not model_id:
                return {"type": "error", "error": "model_id is required"}

            # Get evaluation history
            history = self.agent.get_evaluation_history(model_id=model_id, limit=limit)

            return {
                "type": "evaluation_history",
                "history": history,
                "model_id": model_id,
            }

        except Exception as e:
            logger.error(f"Error getting evaluation history: {e}")
            return {"type": "error", "error": str(e)}

    def _handle_best_models_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle best models request."""
        try:
            best_data = data.get("data", {})

            # Extract parameters
            symbol = best_data.get("symbol", "BTCUSDT")
            timeframe = best_data.get("timeframe", "1h")
            limit = best_data.get("limit", 5)
            metric = best_data.get("metric", "overall_score")

            # Get best models
            best_models = self.agent.get_best_models(
                symbol=symbol, timeframe=timeframe, limit=limit, metric=metric
            )

            return {
                "type": "best_models",
                "models": best_models,
                "symbol": symbol,
                "timeframe": timeframe,
                "metric": metric,
            }

        except Exception as e:
            logger.error(f"Error getting best models: {e}")
            return {"type": "error", "error": str(e)}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            memory_stats = self.memory.get_stats()

            # Get recent evaluations
            recent_evaluations = [
                entry
                for entry in memory_stats.get("recent_decisions", [])
                if entry.get("agent_name") == "performance_critic"
                and entry.get("decision_type") == "evaluate_model"
            ]

            return {
                "total_evaluations": len(recent_evaluations),
                "average_score": sum(
                    eval.get("details", {}).get("evaluation_score", 0)
                    for eval in recent_evaluations
                )
                / max(len(recent_evaluations), 1),
                "memory_entries": memory_stats.get("total_entries", 0),
                "recent_evaluations": recent_evaluations[:5],
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"error": str(e)}
