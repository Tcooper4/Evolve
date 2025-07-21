"""
Reasoning Service

Redis pub/sub service for real-time reasoning updates and decision monitoring.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import redis

from utils.reasoning_logger import (
    AgentDecision,
    ConfidenceLevel,
    DecisionType,
    ReasoningLogger,
)

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


class ReasoningService:
    """
    Redis pub/sub service for real-time reasoning updates.

    Monitors agent decisions and provides real-time updates to clients.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        service_name: str = "reasoning_service",
    ):
        """
        Initialize the ReasoningService.

        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database
            service_name: Service name for logging
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.service_name = service_name

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )

        # Initialize reasoning logger
        self.reasoning_logger = ReasoningLogger(
            redis_host=redis_host, redis_port=redis_port, redis_db=redis_db
        )

        # Service status
        self.running = False
        self.last_heartbeat = time.time()

        # Channels to listen to
        self.channels = [
            "agent_decisions",
            "forecast_completed",
            "strategy_completed",
            "model_evaluation_completed",
        ]

        # Decision cache for real-time updates
        self.recent_decisions = []
        self.max_cache_size = 100

        logger.info(f"ReasoningService initialized: {service_name}")

    def start(self):
        """Start the reasoning service."""
        try:
            self.running = True
            logger.info(f"Starting {self.service_name}")

            # Start heartbeat
            self._start_heartbeat()

            # Start listening for events
            self._listen_for_events()

        except Exception as e:
            logger.error(f"Error starting {self.service_name}: {e}")
            self.running = False
            raise

    def stop(self):
        """Stop the reasoning service."""
        self.running = False
        logger.info(f"Stopping {self.service_name}")

    def _start_heartbeat(self):
        """Start heartbeat monitoring."""

        def heartbeat():
            while self.running:
                try:
                    self.last_heartbeat = time.time()
                    self.redis_client.set(
                        f"service:{self.service_name}:heartbeat",
                        self.last_heartbeat,
                        ex=60,
                    )
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    time.sleep(30)

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

    def _listen_for_events(self):
        """Listen for Redis events and process decisions."""
        pubsub = self.redis_client.pubsub()

        try:
            # Subscribe to channels
            for channel in self.channels:
                pubsub.subscribe(channel)
                logger.info(f"Subscribed to {channel}")

            # Listen for messages
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    try:
                        self._handle_event(message["channel"], message["data"])
                    except Exception as e:
                        logger.error(f"Error handling event: {e}")

        except Exception as e:
            logger.error(f"Error in event listener: {e}")
        finally:
            pubsub.close()

    def _handle_event(self, channel: str, data: str):
        """Handle incoming events."""
        try:
            event_data = json.loads(data)
            logger.info(
                f"Received event on {channel}: {event_data.get('event_id', 'unknown')}"
            )

            if channel == "agent_decisions":
                self._handle_agent_decision(event_data)
            elif channel == "forecast_completed":
                self._handle_forecast_completed(event_data)
            elif channel == "strategy_completed":
                self._handle_strategy_completed(event_data)
            elif channel == "model_evaluation_completed":
                self._handle_model_evaluation_completed(event_data)

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding event data: {e}")
        except Exception as e:
            logger.error(f"Error handling event on {channel}: {e}")

    def _handle_agent_decision(self, event_data: Dict[str, Any]):
        """Handle agent decision event."""
        try:
            decision_id = event_data.get("decision_id")
            if decision_id:
                # Get the full decision
                decision = self.reasoning_logger.get_decision(decision_id)
                if decision:
                    # Add to cache
                    self._add_to_cache(decision)

                    # Publish reasoning update
                    self._publish_reasoning_update(decision)

                    logger.info(f"Processed agent decision: {decision_id}")

        except Exception as e:
            logger.error(f"Error handling agent decision: {e}")

    def _handle_forecast_completed(self, event_data: Dict[str, Any]):
        """Handle forecast completion event."""
        try:
            # Extract data from event
            forecast_data = event_data.get("forecast_data", {})
            symbol = event_data.get("symbol", "Unknown")
            timeframe = event_data.get("timeframe", "Unknown")
            agent_name = event_data.get("agent_name", "ForecastAgent")

            # Create reasoning data
            reasoning = {
                "primary_reason": f"Generated forecast for {symbol} using {forecast_data.get('model_name', 'ML model')}",
                "supporting_factors": [
                    f"Historical data analysis for {symbol}",
                    f"Technical indicators: {', '.join(forecast_data.get('indicators', []))}",
                    f"Market conditions: {forecast_data.get('market_conditions', 'Unknown')}",
                ],
                "alternatives_considered": [
                    "Different model architectures",
                    "Alternative timeframes",
                    "Various feature combinations",
                ],
                "risks_assessed": [
                    "Market volatility",
                    "Model uncertainty",
                    "Data quality issues",
                ],
                "confidence_explanation": f"Confidence based on model performance and market conditions",
                "expected_outcome": f"Forecast indicates {forecast_data.get('trend', 'neutral')} movement for {symbol}",
            }

            # Log the decision
            decision_id = self.reasoning_logger.log_decision(
                agent_name=agent_name,
                decision_type=DecisionType.FORECAST,
                action_taken=f"Generated forecast for {symbol}: {forecast_data.get('prediction', 'N/A')}",
                context={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "market_conditions": forecast_data.get("market_conditions", {}),
                    "available_data": forecast_data.get("features", []),
                    "constraints": forecast_data.get("constraints", {}),
                    "user_preferences": forecast_data.get("preferences", {}),
                },
                reasoning=reasoning,
                confidence_level=(
                    ConfidenceLevel.HIGH
                    if forecast_data.get("confidence", 0) > 0.7
                    else ConfidenceLevel.MEDIUM
                ),
                metadata={"forecast_data": forecast_data},
            )

            logger.info(f"Logged forecast decision: {decision_id}")

        except Exception as e:
            logger.error(f"Error handling forecast completed: {e}")

    def _handle_strategy_completed(self, event_data: Dict[str, Any]):
        """Handle strategy completion event."""
        try:
            # Extract data from event
            strategy_data = event_data.get("strategy_data", {})
            trade_data = event_data.get("trade_data", {})
            symbol = event_data.get("symbol", "Unknown")
            timeframe = event_data.get("timeframe", "Unknown")
            agent_name = event_data.get("agent_name", "StrategyAgent")

            # Create reasoning data
            reasoning = {
                "primary_reason": f"Executed {strategy_data.get('strategy_name', 'trading strategy')} for {symbol}",
                "supporting_factors": [
                    f"Technical analysis signals",
                    f"Market conditions: {strategy_data.get('market_conditions', {}).get('trend', 'Unknown')}",
                    f"Risk management rules applied",
                ],
                "alternatives_considered": [
                    "Different entry/exit points",
                    "Alternative position sizes",
                    "Various stop-loss levels",
                ],
                "risks_assessed": [
                    "Market volatility risk",
                    "Execution slippage",
                    "Position sizing risk",
                ],
                "confidence_explanation": f"Confidence based on strategy backtest performance and current market conditions",
                "expected_outcome": f"Expected {strategy_data.get('performance', {}).get('expected_return', 'positive')} return",
            }

            # Log the decision
            decision_id = self.reasoning_logger.log_decision(
                agent_name=agent_name,
                decision_type=DecisionType.STRATEGY,
                action_taken=f"Executed {strategy_data.get('strategy_name', 'strategy')} on {symbol}",
                context={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "market_conditions": strategy_data.get("market_conditions", {}),
                    "available_data": strategy_data.get("signals", []),
                    "constraints": strategy_data.get("constraints", {}),
                    "user_preferences": strategy_data.get("preferences", {}),
                },
                reasoning=reasoning,
                confidence_level=(
                    ConfidenceLevel.HIGH
                    if strategy_data.get("confidence", 0) > 0.7
                    else ConfidenceLevel.MEDIUM
                ),
                metadata={"strategy_data": strategy_data, "trade_data": trade_data},
            )

            logger.info(f"Logged strategy decision: {decision_id}")

        except Exception as e:
            logger.error(f"Error handling strategy completed: {e}")

    def _handle_model_evaluation_completed(self, event_data: Dict[str, Any]):
        """Handle model evaluation completion event."""
        try:
            # Extract data from event
            evaluation_data = event_data.get("evaluation_data", {})
            symbol = event_data.get("symbol", "Unknown")
            timeframe = event_data.get("timeframe", "Unknown")
            agent_name = event_data.get("agent_name", "EvaluationAgent")

            # Create reasoning data
            reasoning = {
                "primary_reason": f"Evaluated model performance for {symbol}",
                "supporting_factors": [
                    f"Model accuracy: {evaluation_data.get('metrics', {}).get('accuracy', 'N/A')}",
                    f"Sharpe ratio: {evaluation_data.get('metrics', {}).get('sharpe_ratio', 'N/A')}",
                    f"Backtest period: {evaluation_data.get('period', 'N/A')}",
                ],
                "alternatives_considered": [
                    "Different evaluation metrics",
                    "Alternative time periods",
                    "Various model configurations",
                ],
                "risks_assessed": [
                    "Overfitting risk",
                    "Data leakage risk",
                    "Market regime changes",
                ],
                "confidence_explanation": f"Confidence based on comprehensive evaluation metrics",
                "expected_outcome": f"Model expected to perform {evaluation_data.get('performance', 'adequately')}",
            }

            # Log the decision
            decision_id = self.reasoning_logger.log_decision(
                agent_name=agent_name,
                decision_type=DecisionType.MODEL_SELECTION,
                action_taken=f"Evaluated model for {symbol}",
                context={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "market_conditions": evaluation_data.get("market_conditions", {}),
                    "available_data": evaluation_data.get("features", []),
                    "constraints": evaluation_data.get("constraints", {}),
                    "user_preferences": evaluation_data.get("preferences", {}),
                },
                reasoning=reasoning,
                confidence_level=(
                    ConfidenceLevel.HIGH
                    if evaluation_data.get("confidence", 0) > 0.7
                    else ConfidenceLevel.MEDIUM
                ),
                metadata={"evaluation_data": evaluation_data},
            )

            logger.info(f"Logged model evaluation decision: {decision_id}")

        except Exception as e:
            logger.error(f"Error handling model evaluation completed: {e}")

    def _add_to_cache(self, decision: AgentDecision):
        """Add decision to cache."""
        self.recent_decisions.append(decision)

        # Maintain cache size
        if len(self.recent_decisions) > self.max_cache_size:
            self.recent_decisions.pop(0)

    def _publish_reasoning_update(self, decision: AgentDecision):
        """Publish reasoning update to clients."""
        try:
            update_data = {
                "type": "reasoning_update",
                "decision_id": decision.decision_id,
                "agent_name": decision.agent_name,
                "decision_type": decision.decision_type.value,
                "symbol": decision.context.symbol,
                "action_taken": decision.action_taken,
                "confidence_level": decision.confidence_level.value,
                "timestamp": decision.timestamp,
                "summary": self.reasoning_logger.get_summary(decision.decision_id),
                "explanation": self.reasoning_logger.get_explanation(
                    decision.decision_id
                ),
            }

            self.redis_client.publish("reasoning_updates", json.dumps(update_data))

        except Exception as e:
            logger.error(f"Error publishing reasoning update: {e}")

    def get_recent_decisions(self, limit: int = 20) -> List[AgentDecision]:
        """Get recent decisions from cache."""
        return self.recent_decisions[-limit:] if self.recent_decisions else []

    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning statistics."""
        return self.reasoning_logger.get_statistics()

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "service_name": self.service_name,
            "running": self.running,
            "last_heartbeat": self.last_heartbeat,
            "channels": self.channels,
            "redis_connected": self.redis_client.ping(),
            "cache_size": len(self.recent_decisions),
        }


def main():
    """Main function to run the reasoning service."""
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning Service")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database")
    parser.add_argument(
        "--service-name", default="reasoning_service", help="Service name"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize and start service
    service = ReasoningService(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        service_name=args.service_name,
    )

    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        service.stop()
    except Exception as e:
        logger.error(f"Service error: {e}")
        service.stop()


if __name__ == "__main__":
    main()
