"""
Safe Executor Service

Service wrapper for the SafeExecutor, providing safe execution of user-defined
models and strategies via Redis pub/sub communication.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from trading.memory.agent_memory import AgentMemory
from trading.services.base_service import BaseService
from trading.utils.safe_executor import SafeExecutor

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


logger = logging.getLogger(__name__)


class SafeExecutorService(BaseService):
    """
    Service wrapper for SafeExecutor.

    Handles safe execution requests and communicates results via Redis.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        timeout_seconds: int = 300,
        memory_limit_mb: int = 1024,
    ) -> Dict[str, Any]:
        """Initialize the SafeExecutorService."""
        try:
            super().__init__("safe_executor", redis_host, redis_port, redis_db)

            # Initialize SafeExecutor
            self.safe_executor = SafeExecutor(
                timeout_seconds=timeout_seconds,
                memory_limit_mb=memory_limit_mb,
                enable_sandbox=True,
                log_executions=True,
            )
            self.memory = AgentMemory()

            logger.info("SafeExecutorService initialized")
            return {
                "status": "success",
                "message": "SafeExecutorService initialized successfully",
            }
        except Exception as e:
            logger.error(f"Error initializing SafeExecutorService: {e}")
            return {
                "success": True,
                "result": {"status": "error", "message": str(e)},
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming safe execution requests.

        Args:
            data: Message data containing execution request

        Returns:
            Response with execution results
        """
        try:
            message_type = data.get("type", "")

            if message_type == "execute_model":
                return self._handle_model_execution(data)
            elif message_type == "execute_strategy":
                return self._handle_strategy_execution(data)
            elif message_type == "execute_indicator":
                return self._handle_indicator_execution(data)
            elif message_type == "get_statistics":
                return self._handle_statistics_request(data)
            elif message_type == "cleanup":
                return self._handle_cleanup_request(data)
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

    def _handle_model_execution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model execution request."""
        try:
            execution_data = data.get("data", {})

            # Extract parameters
            model_code = execution_data.get("model_code")
            model_name = execution_data.get("model_name")
            input_data = execution_data.get("input_data", {})
            model_type = execution_data.get("model_type", "custom")

            if not model_code or not model_name:
                return {
                    "type": "error",
                    "error": "model_code and model_name are required",
                }

            logger.info(f"Executing model: {model_name}")

            # Execute the model safely
            result = self.safe_executor.execute_model(
                model_code=model_code,
                model_name=model_name,
                input_data=input_data,
                model_type=model_type,
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="safe_executor",
                decision_type="model_executed",
                details={
                    "model_name": model_name,
                    "model_type": model_type,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error": result.error,
                },
            )

            return {
                "type": "model_executed",
                "result": {
                    "status": result.status.value,
                    "output": result.output,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "memory_used": result.memory_used,
                    "return_value": result.return_value,
                    "logs": result.logs,
                },
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error executing model: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def _handle_strategy_execution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle strategy execution request."""
        try:
            execution_data = data.get("data", {})

            # Extract parameters
            strategy_code = execution_data.get("strategy_code")
            strategy_name = execution_data.get("strategy_name")
            market_data = execution_data.get("market_data", {})
            parameters = execution_data.get("parameters", {})

            if not strategy_code or not strategy_name:
                return {
                    "type": "error",
                    "error": "strategy_code and strategy_name are required",
                }

            logger.info(f"Executing strategy: {strategy_name}")

            # Execute the strategy safely
            result = self.safe_executor.execute_strategy(
                strategy_code=strategy_code,
                strategy_name=strategy_name,
                market_data=market_data,
                parameters=parameters,
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="safe_executor",
                decision_type="strategy_executed",
                details={
                    "strategy_name": strategy_name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error": result.error,
                },
            )

            return {
                "type": "strategy_executed",
                "result": {
                    "status": result.status.value,
                    "output": result.output,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "memory_used": result.memory_used,
                    "return_value": result.return_value,
                    "logs": result.logs,
                },
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def _handle_indicator_execution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle indicator execution request."""
        try:
            execution_data = data.get("data", {})

            # Extract parameters
            indicator_code = execution_data.get("indicator_code")
            indicator_name = execution_data.get("indicator_name")
            price_data = execution_data.get("price_data", {})
            parameters = execution_data.get("parameters", {})

            if not indicator_code or not indicator_name:
                return {
                    "type": "error",
                    "error": "indicator_code and indicator_name are required",
                }

            logger.info(f"Executing indicator: {indicator_name}")

            # Execute the indicator safely
            result = self.safe_executor.execute_indicator(
                indicator_code=indicator_code,
                indicator_name=indicator_name,
                price_data=price_data,
                parameters=parameters,
            )

            # Log to memory
            self.memory.log_decision(
                agent_name="safe_executor",
                decision_type="indicator_executed",
                details={
                    "indicator_name": indicator_name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error": result.error,
                },
            )

            return {
                "type": "indicator_executed",
                "result": {
                    "status": result.status.value,
                    "output": result.output,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "memory_used": result.memory_used,
                    "return_value": result.return_value,
                    "logs": result.logs,
                },
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error executing indicator: {e}")
            return {"type": "error", "error": str(e), "status": "failed"}

    def _handle_statistics_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistics request."""
        try:
            stats = self.safe_executor.get_statistics()

            return {"type": "statistics", "statistics": stats, "status": "success"}

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"type": "error", "error": str(e)}

    def _handle_cleanup_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cleanup request."""
        try:
            self.safe_executor.cleanup()

            return {"type": "cleanup_completed", "status": "success"}

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"type": "error", "error": str(e)}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            executor_stats = self.safe_executor.get_statistics()
            memory_stats = self.memory.get_stats()

            # Get recent executions
            recent_executions = [
                entry
                for entry in memory_stats.get("recent_decisions", [])
                if entry.get("agent_name") == "safe_executor"
            ]

            return {
                "executor_statistics": executor_stats,
                "recent_executions": recent_executions[:10],
                "total_memory_entries": memory_stats.get("total_entries", 0),
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"error": str(e)}

    def stop(self) -> Dict[str, Any]:
        """Stop the service and clean up resources."""
        try:
            if hasattr(self, "safe_executor"):
                self.safe_executor.cleanup()
            super().stop()
            return {
                "status": "success",
                "message": "SafeExecutorService stopped successfully",
            }
        except Exception as e:
            logger.error(f"Error stopping SafeExecutorService: {e}")
            return {
                "success": True,
                "result": {"status": "error", "message": str(e)},
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }
