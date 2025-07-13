"""
Commentary Agent

This agent provides LLM-based commentary for trading decisions and analysis.
It wraps the Commentary Engine and follows the BaseAgent interface for
seamless integration with the agent system.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from trading.commentary import (
    CommentaryRequest,
    CommentaryType,
    create_commentary_engine,
)
from trading.memory.agent_memory import AgentMemory
from trading.utils.reasoning_logger import (
    ConfidenceLevel,
    DecisionType,
    ReasoningLogger,
)

from .base_agent_interface import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class CommentaryAgentRequest:
    """Request for commentary agent."""

    commentary_type: str
    symbol: str
    trade_data: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None
    market_data: Optional[Any] = None
    portfolio_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class CommentaryAgent(BaseAgent):
    """
    Commentary Agent that provides LLM-based explanations and insights.

    This agent wraps the Commentary Engine and provides a standardized interface
    for generating commentary across the trading system.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the commentary agent."""
        if config is None:
            config = AgentConfig(
                name="CommentaryAgent",
                enabled=True,
                priority=2,
                max_concurrent_runs=5,
                timeout_seconds=60,
                retry_attempts=2,
                custom_config={},
            )
        super().__init__(config)

        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.reasoning_logger = ReasoningLogger()

        # Initialize commentary engine
        engine_config = self.config.custom_config.get("engine_config", {})
        self.commentary_engine = create_commentary_engine(engine_config)

        # Commentary type mapping
        self.commentary_type_mapping = {
            "trade": CommentaryType.TRADE_EXPLANATION,
            "performance": CommentaryType.PERFORMANCE_ANALYSIS,
            "regime": CommentaryType.MARKET_REGIME,
            "risk": CommentaryType.RISK_ASSESSMENT,
            "strategy": CommentaryType.STRATEGY_RECOMMENDATION,
            "counterfactual": CommentaryType.COUNTERFACTUAL_ANALYSIS,
            "daily": CommentaryType.DAILY_SUMMARY,
            "portfolio": CommentaryType.PORTFOLIO_OVERVIEW,
        }

        self.logger.info("CommentaryAgent initialized successfully")

    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the commentary agent's main logic.

        Args:
            **kwargs: Arguments for commentary generation

        Returns:
            AgentResult: Result of the execution
        """
        try:
            # Extract parameters from kwargs
            commentary_type = kwargs.get("commentary_type", "trade")
            symbol = kwargs.get("symbol", "UNKNOWN")
            trade_data = kwargs.get("trade_data")
            performance_data = kwargs.get("performance_data")
            market_data = kwargs.get("market_data")
            portfolio_data = kwargs.get("portfolio_data")
            context = kwargs.get("context")

            # Create request
            request = CommentaryAgentRequest(
                commentary_type=commentary_type,
                symbol=symbol,
                trade_data=trade_data,
                performance_data=performance_data,
                market_data=market_data,
                portfolio_data=portfolio_data,
                context=context,
            )

            # Generate commentary with retry logic
            return await self._generate_commentary_with_retry(request)

        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters for commentary generation.

        Args:
            **kwargs: Input parameters to validate

        Returns:
            bool: True if input is valid
        """
        try:
            # Check required parameters
            if "symbol" not in kwargs:
                self.logger.error("Missing required parameter: symbol")
                return False

            symbol = kwargs.get("symbol")
            if not symbol or not isinstance(symbol, str):
                self.logger.error("Symbol must be a non-empty string")
                return False

            # Check commentary type
            commentary_type = kwargs.get("commentary_type", "trade")
            if commentary_type not in self.commentary_type_mapping:
                self.logger.error(f"Invalid commentary type: {commentary_type}")
                return False

            # Validate data structures
            for data_key in ["trade_data", "performance_data", "portfolio_data", "context"]:
                if data_key in kwargs and kwargs[data_key] is not None:
                    if not isinstance(kwargs[data_key], dict):
                        self.logger.error(f"{data_key} must be a dictionary")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating input: {e}")
            return False

    def validate_config(self) -> bool:
        """
        Validate the agent's configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check basic configuration
            if not self.config.name:
                self.logger.error("Agent name is required")
                return False

            if self.config.timeout_seconds <= 0:
                self.logger.error("Timeout must be positive")
                return False

            if self.config.max_concurrent_runs <= 0:
                self.logger.error("Max concurrent runs must be positive")
                return False

            # Check commentary engine configuration
            if not self.commentary_engine:
                self.logger.error("Commentary engine not initialized")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    def handle_error(self, error: Exception) -> AgentResult:
        """
        Handle errors during execution with comprehensive error categorization.

        Args:
            error: Exception that occurred

        Returns:
            AgentResult: Error result with appropriate error information
        """
        import traceback

        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()

        # Categorize errors
        if isinstance(error, asyncio.TimeoutError):
            error_category = "timeout"
            user_message = f"Commentary generation timed out after {self.config.timeout_seconds} seconds"
        elif isinstance(error, ConnectionError):
            error_category = "connection"
            user_message = "Failed to connect to commentary service"
        elif isinstance(error, ValueError):
            error_category = "validation"
            user_message = f"Invalid input: {error_message}"
        elif isinstance(error, KeyError):
            error_category = "data"
            user_message = f"Missing required data: {error_message}"
        else:
            error_category = "unknown"
            user_message = f"Unexpected error: {error_message}"

        # Log error details
        self.logger.error(f"CommentaryAgent error ({error_category}): {error_message}")
        self.logger.debug(f"Error traceback: {error_traceback}")

        # Update status
        self.status.failed_runs += 1
        self.status.current_error = user_message
        self.status.is_running = False
        self.status.last_failure = datetime.now()

        return AgentResult(
            success=False,
            error_message=user_message,
            error_type=error_type,
            metadata={"error_category": error_category, "traceback": error_traceback, "agent_name": self.config.name},
        )

    def _setup(self) -> None:
        """Setup method called during initialization."""
        try:
            # Initialize commentary engine if not already done
            if not hasattr(self, "commentary_engine") or self.commentary_engine is None:
                engine_config = self.config.custom_config.get("engine_config", {})
                self.commentary_engine = create_commentary_engine(engine_config)

            # Initialize memory and reasoning logger
            if not hasattr(self, "memory"):
                self.memory = AgentMemory()
            if not hasattr(self, "reasoning_logger"):
                self.reasoning_logger = ReasoningLogger()

            # Validate setup
            if not self.commentary_engine:
                raise RuntimeError("Failed to initialize commentary engine")

            self.logger.info("CommentaryAgent setup completed successfully")

        except Exception as e:
            self.logger.error(f"Error during setup: {e}")
            raise

    def get_capabilities(self) -> List[str]:
        """
        Get the agent's capabilities.

        Returns:
            List[str]: List of capability names
        """
        return [
            "trade_explanation",
            "performance_analysis",
            "risk_assessment",
            "market_regime_analysis",
            "strategy_recommendation",
            "counterfactual_analysis",
            "daily_summary",
            "portfolio_overview",
        ]

    def get_requirements(self) -> Dict[str, Any]:
        """
        Get the agent's requirements.

        Returns:
            Dict[str, Any]: Dictionary of requirements
        """
        return {
            "dependencies": ["trading.commentary", "trading.memory.agent_memory", "trading.utils.reasoning_logger"],
            "system_requirements": {"python_version": ">=3.8", "async_support": True},
            "external_services": ["commentary_engine", "llm_service"],
        }

    async def _generate_commentary_with_retry(self, request: CommentaryAgentRequest) -> AgentResult:
        """
        Generate commentary with retry logic and comprehensive error handling.

        Args:
            request: Commentary request

        Returns:
            AgentResult: Result of commentary generation
        """
        max_retries = self.config.retry_attempts
        retry_delay = self.config.retry_delay_seconds

        for attempt in range(max_retries + 1):
            try:
                self.logger.info(
                    f"Generating {request.commentary_type} commentary for {request.symbol} (attempt {attempt + 1})"
                )

                # Map commentary type
                commentary_type = self.commentary_type_mapping.get(
                    request.commentary_type.lower(), CommentaryType.TRADE_EXPLANATION
                )

                # Create commentary request
                commentary_request = CommentaryRequest(
                    commentary_type=commentary_type,
                    symbol=request.symbol,
                    timestamp=datetime.now(),
                    trade_data=request.trade_data,
                    performance_data=request.performance_data,
                    market_data=request.market_data,
                    portfolio_data=request.portfolio_data,
                    context=request.context,
                )

                # Generate commentary with timeout
                response = await asyncio.wait_for(
                    self.commentary_engine.generate_commentary(commentary_request), timeout=self.config.timeout_seconds
                )

                # Log decision
                self._log_commentary_decision(response)

                # Store in memory
                self._store_commentary(response)

                # Update success status
                self.status.successful_runs += 1
                self.status.last_success = datetime.now()
                self.status.current_error = None

                return AgentResult(
                    success=True,
                    message=f"Generated {request.commentary_type} commentary",
                    data={
                        "commentary": response,
                        "type": request.commentary_type,
                        "symbol": request.symbol,
                        "confidence": response.confidence_score,
                        "attempt": attempt + 1,
                    },
                )

            except asyncio.TimeoutError as e:
                self.logger.warning(f"Commentary generation timed out (attempt {attempt + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    return self.handle_error(e)

            except ConnectionError as e:
                self.logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return self.handle_error(e)

            except Exception as e:
                self.logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    return self.handle_error(e)

        # This should never be reached, but just in case
        return AgentResult(
            success=False,
            error_message="Failed to generate commentary after all retry attempts",
            error_type="MaxRetriesExceeded",
        )

    async def generate_commentary(self, request: CommentaryAgentRequest) -> AgentResult:
        """
        Generate commentary based on the request.

        Args:
            request: Commentary request

        Returns:
            AgentResult with commentary data
        """
        try:
            self.logger.info(f"Generating {request.commentary_type} commentary for {request.symbol}")

            # Map commentary type
            commentary_type = self.commentary_type_mapping.get(
                request.commentary_type.lower(), CommentaryType.TRADE_EXPLANATION
            )

            # Create commentary request
            commentary_request = CommentaryRequest(
                commentary_type=commentary_type,
                symbol=request.symbol,
                timestamp=datetime.now(),
                trade_data=request.trade_data,
                performance_data=request.performance_data,
                market_data=request.market_data,
                portfolio_data=request.portfolio_data,
                context=request.context,
            )

            # Generate commentary
            response = await self.commentary_engine.generate_commentary(commentary_request)

            # Log decision
            self._log_commentary_decision(response)

            # Store in memory
            self._store_commentary(response)

            return AgentResult(
                success=True,
                message=f"Generated {request.commentary_type} commentary",
                data={
                    "commentary": response,
                    "type": request.commentary_type,
                    "symbol": request.symbol,
                    "confidence": response.confidence_score,
                },
            )

        except Exception as e:
            self.logger.error(f"Error generating commentary: {str(e)}")
            return AgentResult(
                success=False, message=f"Failed to generate commentary: {str(e)}", data={"error": str(e)}
            )

    def explain_trade(self, symbol: str, trade_data: Dict[str, Any], market_data: Optional[Any] = None) -> AgentResult:
        """
        Explain a trading decision.

        Args:
            symbol: Trading symbol
            trade_data: Trade details
            market_data: Market data

        Returns:
            AgentResult with trade explanation
        """
        request = CommentaryAgentRequest(
            commentary_type="trade", symbol=symbol, trade_data=trade_data, market_data=market_data
        )

        # Run async function in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                task = asyncio.create_task(self.generate_commentary(request))
                # This is a simplified approach - in practice you'd handle this differently
                return AgentResult(
                    success=False,
                    message="Async commentary generation not supported in sync context",
                    data={"suggestion": "Use generate_commentary_async for async operation"},
                )
            else:
                return loop.run_until_complete(self.generate_commentary(request))
        except RuntimeError:
            # No event loop running
            return AgentResult(
                success=False,
                message="No event loop available for async commentary generation",
                data={"suggestion": "Use generate_commentary_async for async operation"},
            )

    def analyze_performance(self, symbol: str, performance_data: Dict[str, Any]) -> AgentResult:
        """
        Analyze trading performance.

        Args:
            symbol: Trading symbol
            performance_data: Performance metrics

        Returns:
            AgentResult with performance analysis
        """
        request = CommentaryAgentRequest(
            commentary_type="performance", symbol=symbol, performance_data=performance_data
        )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return AgentResult(
                    success=False,
                    message="Async commentary generation not supported in sync context",
                    data={"suggestion": "Use generate_commentary_async for async operation"},
                )
            else:
                return loop.run_until_complete(self.generate_commentary(request))
        except RuntimeError:
            return AgentResult(
                success=False,
                message="No event loop available for async commentary generation",
                data={"suggestion": "Use generate_commentary_async for async operation"},
            )

    def assess_risk(
        self, symbol: str, trade_data: Optional[Dict[str, Any]] = None, portfolio_data: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Assess trading risks.

        Args:
            symbol: Trading symbol
            trade_data: Trade details (optional)
            portfolio_data: Portfolio data (optional)

        Returns:
            AgentResult with risk assessment
        """
        request = CommentaryAgentRequest(
            commentary_type="risk", symbol=symbol, trade_data=trade_data, portfolio_data=portfolio_data
        )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return AgentResult(
                    success=False,
                    message="Async commentary generation not supported in sync context",
                    data={"suggestion": "Use generate_commentary_async for async operation"},
                )
            else:
                return loop.run_until_complete(self.generate_commentary(request))
        except RuntimeError:
            return AgentResult(
                success=False,
                message="No event loop available for async commentary generation",
                data={"suggestion": "Use generate_commentary_async for async operation"},
            )

    def _log_commentary_decision(self, response):
        """Log commentary decision."""
        self.reasoning_logger.log_decision(
            agent_name="CommentaryAgent",
            decision_type=DecisionType.COMMENTARY_GENERATION,
            action_taken=f"Generated {response.commentary_type.value} commentary",
            context={
                "commentary_type": response.commentary_type.value,
                "symbol": response.metadata.get("symbol", "Unknown"),
                "confidence": response.confidence_score,
                "insights_count": len(response.key_insights),
                "recommendations_count": len(response.recommendations),
            },
            reasoning={
                "primary_reason": f"Generated {response.commentary_type.value} commentary",
                "supporting_factors": response.key_insights,
                "recommendations": response.recommendations,
                "risk_warnings": response.risk_warnings,
            },
            confidence_level=ConfidenceLevel.HIGH if response.confidence_score > 0.8 else ConfidenceLevel.MEDIUM,
            metadata=response.metadata,
        )

    def _store_commentary(self, response):
        """Store commentary in memory."""
        try:
            self.memory.store(
                "commentary_history",
                {"response": response.__dict__, "timestamp": datetime.now(), "agent": "CommentaryAgent"},
            )
        except Exception as e:
            self.logger.error(f"Error storing commentary: {str(e)}")

    def get_commentary_statistics(self) -> Dict[str, Any]:
        """Get commentary generation statistics."""
        return self.commentary_engine.get_commentary_statistics()


# Convenience function for creating commentary agent


def create_commentary_agent(config: Optional[AgentConfig] = None) -> CommentaryAgent:
    """Create a configured commentary agent."""
    return CommentaryAgent(config)
