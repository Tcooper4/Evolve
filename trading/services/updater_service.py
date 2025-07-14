"""
Updater Service

Service wrapper for the UpdaterAgent, handling model update and retraining requests
via Redis pub/sub communication.
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trading.agents.updater_agent import UpdaterAgent
from trading.memory.agent_memory import AgentMemory
from trading.services.base_service import BaseService

logger = logging.getLogger(__name__)


class UpdaterService(BaseService):
    """
    Service wrapper for UpdaterAgent.

    Handles model update and retraining requests and communicates results via Redis.
    """

    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0
    ):
        """Initialize the UpdaterService."""
        super().__init__("updater", redis_host, redis_port, redis_db)

        # Initialize the agent
        self.agent = UpdaterAgent()
        self.memory = AgentMemory()

        logger.info("UpdaterService initialized")

    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming model update requests.

        Args:
            data: Message data containing update request

        Returns:
            Response with update results or error
        """
        try:
            message_type = data.get("type", "")

            if message_type == "retrain_model":
                return self._handle_retrain_request(data)
            elif message_type == "tune_model":
                return self._handle_tune_request(data)
            elif message_type == "replace_model":
                return self._handle_replace_request(data)
            elif message_type == "get_update_history":
                return self._handle_history_request(data)
            elif message_type == "auto_update":
                return self._handle_auto_update_request(data)
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

    def _handle_retrain_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model retraining request."""
        try:
            retrain_data = data.get("data", {})

            # Extract retraining parameters
            model_id = retrain_data.get("model_id")
            new_data_period = retrain_data.get("new_data_period", "30d")
            retrain_type = retrain_data.get("retrain_type", "incremental")

            if not model_id:
                return {"type": "error", "error": "model_id is required"}

            logger.info(f"Retraining model {model_id} with {retrain_type} training")

            # Retrain the model using the agent
            result = self.agent.retrain_model(
                model_id=model_id,
                new_data_period=new_data_period,
                retrain_type=retrain_type,
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="updater",
                decision_type="retrain_model",
                details={
                    "model_id": model_id,
                    "retrain_type": retrain_type,
                    "new_data_period": new_data_period,
                    "success": result.get("success", False),
                    "new_performance": result.get("new_performance", {}),
                },
            )

            return {
                "type": "model_retrained",
                "result": result,
                "status": "success" if result.get("success") else "failed",
            }

        except Exception as e:
            tb = traceback.format_exc()
            ts = datetime.utcnow().isoformat() + "Z"
            logger.error(
                f"Error retraining model: {e}\nTraceback:\n{tb}\nTimestamp: {ts}"
            )
            return {
                "type": "error",
                "error": str(e),
                "traceback": tb,
                "timestamp": ts,
                "status": "failed",
            }

    def _handle_tune_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model tuning request."""
        try:
            tune_data = data.get("data", {})

            # Extract tuning parameters
            model_id = tune_data.get("model_id")
            tuning_params = tune_data.get("tuning_params", {})
            optimization_method = tune_data.get("optimization_method", "bayesian")

            if not model_id:
                return {"type": "error", "error": "model_id is required"}

            logger.info(
                f"Tuning model {model_id} with {optimization_method} optimization"
            )

            # Tune the model using the agent
            result = self.agent.tune_model(
                model_id=model_id,
                tuning_params=tuning_params,
                optimization_method=optimization_method,
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="updater",
                decision_type="tune_model",
                details={
                    "model_id": model_id,
                    "optimization_method": optimization_method,
                    "tuning_params": tuning_params,
                    "success": result.get("success", False),
                    "best_params": result.get("best_params", {}),
                    "improvement": result.get("improvement", 0),
                },
            )

            return {
                "type": "model_tuned",
                "result": result,
                "status": "success" if result.get("success") else "failed",
            }

        except Exception as e:
            logger.error(f"Error tuning model: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def _handle_replace_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model replacement request."""
        try:
            replace_data = data.get("data", {})

            # Extract replacement parameters
            old_model_id = replace_data.get("old_model_id")
            new_model_id = replace_data.get("new_model_id")
            replacement_reason = replace_data.get("reason", "performance_improvement")

            if not old_model_id or not new_model_id:
                return {
                    "type": "error",
                    "error": "Both old_model_id and new_model_id are required",
                }

            logger.info(f"Replacing model {old_model_id} with {new_model_id}")

            # Replace the model using the agent
            result = self.agent.replace_model(
                old_model_id=old_model_id,
                new_model_id=new_model_id,
                reason=replacement_reason,
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="updater",
                decision_type="replace_model",
                details={
                    "old_model_id": old_model_id,
                    "new_model_id": new_model_id,
                    "reason": replacement_reason,
                    "success": result.get("success", False),
                },
            )

            return {
                "type": "model_replaced",
                "result": result,
                "status": "success" if result.get("success") else "failed",
            }

        except Exception as e:
            logger.error(f"Error replacing model: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def _handle_history_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update history request."""
        try:
            history_data = data.get("data", {})

            # Extract parameters
            model_id = history_data.get("model_id")
            limit = history_data.get("limit", 10)

            if not model_id:
                return {"type": "error", "error": "model_id is required"}

            # Get update history
            history = self.agent.get_update_history(model_id=model_id, limit=limit)

            return {"type": "update_history", "history": history, "model_id": model_id}

        except Exception as e:
            logger.error(f"Error getting update history: {e}")
            return {"type": "error", "error": str(e)}

    def _handle_auto_update_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle automatic update request."""
        try:
            auto_data = data.get("data", {})

            # Extract parameters
            symbol = auto_data.get("symbol", "BTCUSDT")
            timeframe = auto_data.get("timeframe", "1h")
            update_type = auto_data.get("update_type", "smart")

            logger.info(
                f"Starting auto-update for {symbol} with {update_type} strategy"
            )

            # Perform automatic update
            result = self.agent.auto_update_models(
                symbol=symbol, timeframe=timeframe, update_type=update_type
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="updater",
                decision_type="auto_update",
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "update_type": update_type,
                    "models_updated": result.get("models_updated", []),
                    "models_replaced": result.get("models_replaced", []),
                    "overall_improvement": result.get("overall_improvement", 0),
                },
            )

            return {
                "type": "auto_update_completed",
                "result": result,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error in auto-update: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            memory_stats = self.memory.get_stats()

            # Get recent updates
            recent_updates = [
                entry
                for entry in memory_stats.get("recent_decisions", [])
                if entry.get("agent_name") == "updater"
            ]

            # Count by type
            update_types = {}
            for update in recent_updates:
                update_type = update.get("decision_type", "unknown")
                update_types[update_type] = update_types.get(update_type, 0) + 1

            return {
                "total_updates": len(recent_updates),
                "update_types": update_types,
                "memory_entries": memory_stats.get("total_entries", 0),
                "recent_updates": recent_updates[:5],
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"error": str(e)}
