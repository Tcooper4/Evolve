"""
Router agent for the financial forecasting system.

This module handles routing of requests to appropriate agents based on intent detection.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from .base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

class IntentType(str, Enum):
    """Types of intents that can be detected."""
    FORECAST = "forecast"
    STRATEGY = "strategy"
    BACKTEST = "backtest"
    COMMENTARY = "commentary"
    EXPLANATION = "explanation"
    MULTI_STEP = "multi_step"
    UNCERTAIN = "uncertain"

class RouterAgent(BaseAgent):
    """Agent responsible for routing requests to appropriate handlers."""

    def __init__(self, name: str = "router", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the router agent.
        
        Args:
            name: Name of the agent
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.intent_handlers: Dict[IntentType, Callable] = {}
        self.pipelines: Dict[str, List[IntentType]] = {}
        self.agent_registry: Dict[str, Any] = {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
    def _setup(self) -> None:
        """Setup the router agent."""
        self._register_default_handlers()
        self._register_default_pipelines()
        self.logger.info("Router agent setup completed")
    
    def _register_default_handlers(self) -> None:
        """Register default intent handlers."""
        self.register_handler(IntentType.FORECAST, self._handle_forecast)
        self.register_handler(IntentType.STRATEGY, self._handle_strategy)
        self.register_handler(IntentType.BACKTEST, self._handle_backtest)
        self.register_handler(IntentType.COMMENTARY, self._handle_commentary)
        self.register_handler(IntentType.EXPLANATION, self._handle_explanation)
        self.register_handler(IntentType.MULTI_STEP, self._handle_multi_step)
        self.register_handler(IntentType.UNCERTAIN, self._handle_uncertain)
    
    def _register_default_pipelines(self) -> None:
        """Register default processing pipelines."""
        self.register_pipeline("forecast_and_backtest", 
                             [IntentType.FORECAST, IntentType.BACKTEST])
        self.register_pipeline("strategy_and_backtest", 
                             [IntentType.STRATEGY, IntentType.BACKTEST])
        self.register_pipeline("forecast_and_explain", 
                             [IntentType.FORECAST, IntentType.EXPLANATION])
    
    def register_handler(self, intent: IntentType, handler: Callable) -> None:
        """
        Register a handler for an intent type.
        
        Args:
            intent: The intent type to handle
            handler: The handler function
        """
        self.intent_handlers[intent] = handler
        self.logger.info(f"Registered handler for intent: {intent}")
    
    def register_pipeline(self, name: str, intents: List[IntentType]) -> None:
        """
        Register a pipeline of intents.
        
        Args:
            name: Pipeline name
            intents: List of intents in sequence
        """
        self.pipelines[name] = intents
        self.logger.info(f"Registered pipeline: {name}")
    
    def register_agent(self, name: str, agent: Any) -> None:
        """
        Register an agent for routing.
        
        Args:
            name: Agent name
            agent: Agent instance
        """
        self.agent_registry[name] = agent
        self.logger.info(f"Registered agent: {name}")
    
    def get_handler(self, intent: IntentType) -> Optional[Callable]:
        """
        Get the handler for an intent type.
        
        Args:
            intent: The intent type
            
        Returns:
            The handler function or None if not found
        """
        return self.intent_handlers.get(intent)
    
    def get_pipeline(self, name: str) -> Optional[List[IntentType]]:
        """
        Get a pipeline by name.
        
        Args:
            name: Pipeline name
            
        Returns:
            List of intents or None if not found
        """
        return self.pipelines.get(name)
    
    def detect_intent(self, prompt: str) -> IntentType:
        """
        Detect intent from a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Detected intent type
        """
        prompt_lower = prompt.lower()
        
        # Simple keyword-based intent detection
        if any(word in prompt_lower for word in ['forecast', 'predict', 'price prediction']):
            return IntentType.FORECAST
        elif any(word in prompt_lower for word in ['strategy', 'signal', 'trading strategy']):
            return IntentType.STRATEGY
        elif any(word in prompt_lower for word in ['backtest', 'back test', 'historical']):
            return IntentType.BACKTEST
        elif any(word in prompt_lower for word in ['explain', 'explanation', 'why']):
            return IntentType.EXPLANATION
        elif any(word in prompt_lower for word in ['commentary', 'analysis', 'insight']):
            return IntentType.COMMENTARY
        elif any(word in prompt_lower for word in ['multi', 'pipeline', 'workflow']):
            return IntentType.MULTI_STEP
        else:
            return IntentType.UNCERTAIN
    
    def run(self, prompt: str, **kwargs) -> AgentResult:
        """
        Process a routing request.
        
        Args:
            prompt: Input prompt to route
            **kwargs: Additional arguments
            
        Returns:
            AgentResult: Result of the routing process
        """
        try:
            if not self.validate_input(prompt):
                return AgentResult(
                    success=False,
                    message="Invalid input provided"
                )
            
            # Detect intent
            intent = self.detect_intent(prompt)
            self.logger.info(f"Detected intent: {intent} for prompt: {prompt[:50]}...")
            
            # Get handler
            handler = self.get_handler(intent)
            if not handler:
                return AgentResult(
                    success=False,
                    message=f"No handler registered for intent: {intent}"
                )
            
            # Process with handler
            result = handler(prompt, intent, **kwargs)
            
            # Log execution
            self.log_execution(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in routing: {e}")
            return self.handle_error(e)
    
    def _handle_forecast(self, prompt: str, intent: IntentType, **kwargs) -> AgentResult:
        """
        Handle forecasting requests.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            **kwargs: Additional arguments
            
        Returns:
            AgentResult: Forecast result
        """
        try:
            # Extract ticker from prompt (simple implementation)
            ticker = self._extract_ticker(prompt)
            
            # Get forecaster agent
            forecaster = self.agent_registry.get('forecaster')
            if not forecaster:
                return AgentResult(
                    success=False,
                    message="Forecaster agent not available"
                )
            
            # Execute forecast
            forecast_result = forecaster.run(prompt, **kwargs)
            
            return AgentResult(
                success=True,
                message=f"Forecast completed for {ticker}",
                data={
                    'intent': intent.value,
                    'ticker': ticker,
                    'forecast_result': forecast_result.data if forecast_result.success else None
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error in forecast handling: {str(e)}"
            )
    
    def _handle_strategy(self, prompt: str, intent: IntentType, **kwargs) -> AgentResult:
        """
        Handle strategy requests.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            **kwargs: Additional arguments
            
        Returns:
            AgentResult: Strategy result
        """
        try:
            ticker = self._extract_ticker(prompt)
            
            strategy_agent = self.agent_registry.get('strategy_manager')
            if not strategy_agent:
                return AgentResult(
                    success=False,
                    message="Strategy manager not available"
                )
            
            strategy_result = strategy_agent.run(prompt, **kwargs)
            
            return AgentResult(
                success=True,
                message=f"Strategy analysis completed for {ticker}",
                data={
                    'intent': intent.value,
                    'ticker': ticker,
                    'strategy_result': strategy_result.data if strategy_result.success else None
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error in strategy handling: {str(e)}"
            )
    
    def _handle_backtest(self, prompt: str, intent: IntentType, **kwargs) -> AgentResult:
        """
        Handle backtesting requests.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            **kwargs: Additional arguments
            
        Returns:
            AgentResult: Backtest result
        """
        try:
            ticker = self._extract_ticker(prompt)
            
            backtester = self.agent_registry.get('backtester')
            if not backtester:
                return AgentResult(
                    success=False,
                    message="Backtester not available"
                )
            
            backtest_result = backtester.run(prompt, **kwargs)
            
            return AgentResult(
                success=True,
                message=f"Backtest completed for {ticker}",
                data={
                    'intent': intent.value,
                    'ticker': ticker,
                    'backtest_result': backtest_result.data if backtest_result.success else None
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error in backtest handling: {str(e)}"
            )
    
    def _handle_commentary(self, prompt: str, intent: IntentType, **kwargs) -> AgentResult:
        """
        Handle commentary requests.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            **kwargs: Additional arguments
            
        Returns:
            AgentResult: Commentary result
        """
        try:
            commentary_agent = self.agent_registry.get('commentary_agent')
            if not commentary_agent:
                return AgentResult(
                    success=False,
                    message="Commentary agent not available"
                )
            
            commentary_result = commentary_agent.run(prompt, **kwargs)
            
            return AgentResult(
                success=True,
                message="Commentary generated",
                data={
                    'intent': intent.value,
                    'commentary': commentary_result.data if commentary_result.success else None
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error in commentary handling: {str(e)}"
            )
    
    def _handle_explanation(self, prompt: str, intent: IntentType, **kwargs) -> AgentResult:
        """
        Handle explanation requests.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            **kwargs: Additional arguments
            
        Returns:
            AgentResult: Explanation result
        """
        try:
            # For now, use commentary agent for explanations
            return self._handle_commentary(prompt, intent, **kwargs)
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error in explanation handling: {str(e)}"
            )
    
    def _handle_multi_step(self, prompt: str, intent: IntentType, **kwargs) -> AgentResult:
        """
        Handle multi-step processing requests.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            **kwargs: Additional arguments
            
        Returns:
            AgentResult: Multi-step result
        """
        try:
            # Simple multi-step: forecast + commentary
            forecast_result = self._handle_forecast(prompt, IntentType.FORECAST, **kwargs)
            commentary_result = self._handle_commentary(prompt, IntentType.COMMENTARY, **kwargs)
            
            return AgentResult(
                success=forecast_result.success and commentary_result.success,
                message="Multi-step processing completed",
                data={
                    'intent': intent.value,
                    'forecast': forecast_result.data,
                    'commentary': commentary_result.data
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error in multi-step handling: {str(e)}"
            )
    
    def _handle_uncertain(self, prompt: str, intent: IntentType, **kwargs) -> AgentResult:
        """
        Handle uncertain intent requests.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            **kwargs: Additional arguments
            
        Returns:
            AgentResult: Uncertain intent result
        """
        return AgentResult(
            success=False,
            message="Unable to determine intent. Please rephrase your request.",
            data={
                'intent': intent.value,
                'suggestions': [
                    "Try asking for a forecast: 'Forecast AAPL price'",
                    "Try asking for strategy: 'Show me trading strategy for AAPL'",
                    "Try asking for backtest: 'Backtest this strategy'"
                ]
            }
        )
    
    def _extract_ticker(self, prompt: str) -> str:
        """
        Extract ticker symbol from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Extracted ticker symbol or default
        """
        # Simple ticker extraction (can be enhanced with NLP)
        words = prompt.upper().split()
        common_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        
        for word in words:
            if word in common_tickers:
                return word
        
        return 'AAPL'  # Default ticker
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get router status information.
        
        Returns:
            Dictionary containing router status
        """
        base_status = super().get_status()
        base_status.update({
            'registered_handlers': list(self.intent_handlers.keys()),
            'registered_pipelines': list(self.pipelines.keys()),
            'registered_agents': list(self.agent_registry.keys()),
            'confidence_threshold': self.confidence_threshold
        })
        return base_status

if __name__ == "__main__":
    # Test router
    router = RouterAgent()
    result = router.run("Forecast the price of AAPL for the next 30 days")
    print(f"Router result: {result}")
