# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
from ..llm.llm_interface import LLMInterface
from ..utils.prompt_parser import PromptParser
from ..utils.exceptions import RouterError, IntentDetectionError, AgentError

class IntentType(Enum):
    """Types of intents that can be detected."""
    FORECAST = "forecast"
    STRATEGY = "strategy"
    BACKTEST = "backtest"
    COMMENTARY = "commentary"
    EXPLANATION = "explanation"
    MULTI_STEP = "multi_step"
    UNCERTAIN = "uncertain"

@dataclass
class Intent:
    """Represents a detected intent with confidence score."""
    type: IntentType
    confidence: float
    raw_intent: str
    processed_intent: str
    sub_intents: Optional[List['Intent']] = None

@dataclass
class AgentResult:
    """Standardized result from any agent."""
    data: Any
    visual: Optional[go.Figure] = None
    explanation: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentRouter:
    """Route user prompts to the appropriate trading agent based on intent."""

    def __init__(
        self,
        llm: LLMInterface,
        forecaster: Any,
        strategy_manager: Any,
        backtester: Any,
        commentary_agent: Optional[Any] = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize the router.
        
        Args:
            llm: LLM interface for intent detection
            forecaster: Forecasting agent
            strategy_manager: Strategy management agent
            backtester: Backtesting agent
            commentary_agent: Optional commentary agent
            confidence_threshold: Minimum confidence score for intent detection
        """
        self.llm = llm
        self.forecaster = forecaster
        self.strategy_manager = strategy_manager
        self.backtester = backtester
        self.commentary_agent = commentary_agent or llm
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.prompt_parser = PromptParser()

        # Initialize intent handlers
        self.intent_handlers: Dict[IntentType, callable] = {
            IntentType.FORECAST: self._handle_forecast,
            IntentType.STRATEGY: self._handle_strategy,
            IntentType.BACKTEST: self._handle_backtest,
            IntentType.COMMENTARY: self._handle_commentary,
            IntentType.EXPLANATION: self._handle_commentary,
            IntentType.MULTI_STEP: self._handle_multi_step,
        }

        # Initialize pipeline definitions
        self.pipelines: Dict[str, List[IntentType]] = {
            "forecast_and_backtest": [IntentType.FORECAST, IntentType.BACKTEST],
            "strategy_and_backtest": [IntentType.STRATEGY, IntentType.BACKTEST],
            "forecast_and_explain": [IntentType.FORECAST, IntentType.EXPLANATION],
        }

    def route(self, prompt: str, **kwargs) -> AgentResult:
        """Route the prompt to the correct agent based on intent.
        
        Args:
            prompt: User prompt to process
            **kwargs: Additional arguments for the handlers
            
        Returns:
            AgentResult containing the processed response
            
        Raises:
            RouterError: If routing fails
            IntentDetectionError: If intent detection fails
        """
        try:
            # Parse prompt
            parsed_prompt = self.prompt_parser.parse(prompt)
            self.logger.debug("Parsed prompt: %s", parsed_prompt)
            
            # Detect intent
            intent = self._detect_intent(prompt, parsed_prompt)
            self.logger.info("Detected intent: %s (confidence: %.2f)", 
                           intent.type.value, intent.confidence)
            
            # Handle uncertain intent
            if intent.confidence < self.confidence_threshold:
                self.logger.warning("Low confidence intent detected, falling back to explanation")
                return self._handle_commentary(prompt, parsed_prompt, **kwargs)
            
            # Route to appropriate handler
            handler = self.intent_handlers.get(intent.type)
            if not handler:
                raise RouterError(f"No handler found for intent: {intent.type}")
            
            return handler(prompt, parsed_prompt, **kwargs)
            
        except Exception as e:
            self.logger.error("Error routing prompt: %s", str(e))
            return AgentResult(
                data=None,
                status="error",
                error=str(e)
            )

    def _detect_intent(self, prompt: str, parsed_prompt: Dict) -> Intent:
        """Detect intent from prompt with confidence score.
        
        Args:
            prompt: Raw user prompt
            parsed_prompt: Parsed prompt information
            
        Returns:
            Intent object with type and confidence score
            
        Raises:
            IntentDetectionError: If intent detection fails
        """
        try:
            # Get raw intent from LLM
            raw_intent = self.llm.prompt_processor.extract_intent(prompt)
            
            # Process intent and get confidence
            processed_intent, confidence = self.llm.prompt_processor.get_intent_confidence(
                prompt, raw_intent
            )
            
            # Check for multi-step intent
            if "and" in processed_intent.lower():
                pipeline = self._identify_pipeline(processed_intent)
                if pipeline:
                    return Intent(
                        type=IntentType.MULTI_STEP,
                        confidence=confidence,
                        raw_intent=raw_intent,
                        processed_intent=processed_intent,
                        sub_intents=[Intent(
                            type=IntentType(intent_type),
                            confidence=confidence,
                            raw_intent=raw_intent,
                            processed_intent=processed_intent
                        ) for intent_type in pipeline]
                    )
            
            # Map to intent type
            intent_type = self._map_intent_type(processed_intent)
            
            return Intent(
                type=intent_type,
                confidence=confidence,
                raw_intent=raw_intent,
                processed_intent=processed_intent
            )
            
        except Exception as e:
            raise IntentDetectionError(f"Failed to detect intent: {str(e)}")

    def _map_intent_type(self, intent: str) -> IntentType:
        """Map processed intent string to IntentType enum."""
        intent = intent.lower()
        if "forecast" in intent:
            return IntentType.FORECAST
        elif "strategy" in intent or "recommend" in intent:
            return IntentType.STRATEGY
        elif "backtest" in intent:
            return IntentType.BACKTEST
        elif "explain" in intent:
            return IntentType.EXPLANATION
        else:
            return IntentType.COMMENTARY

    def _identify_pipeline(self, intent: str) -> Optional[List[IntentType]]:
        """Identify if intent matches a known pipeline."""
        intent = intent.lower()
        for pipeline_name, steps in self.pipelines.items():
            if all(step.value in intent for step in steps):
                return steps
        return None

    def _handle_multi_step(self, prompt: str, parsed_prompt: Dict, **kwargs) -> AgentResult:
        """Handle multi-step intents by chaining agent calls."""
        if not parsed_prompt.get('sub_intents'):
            raise RouterError("No sub-intents found for multi-step request")
        
        results = []
        for sub_intent in parsed_prompt['sub_intents']:
            handler = self.intent_handlers.get(sub_intent.type)
            if not handler:
                raise RouterError(f"No handler found for sub-intent: {sub_intent.type}")
            
            result = handler(prompt, parsed_prompt, **kwargs)
            results.append(result)
        
        # Combine results
        return self._combine_results(results)

    def _handle_forecast(self, prompt: str, parsed_prompt: Dict, **kwargs) -> AgentResult:
        """Handle forecasting requests."""
        try:
            # Extract parameters
            ticker = parsed_prompt.get('ticker')
            duration = parsed_prompt.get('duration', '30d')
            output_format = parsed_prompt.get('output_format', 'chart')
            
            # Get data
            data = kwargs.get('data')
            if data is None:
                raise AgentError("No data provided for forecasting")
            
            # Generate forecast
            forecast = self.forecaster.predict(data)
            
            # Generate visualization
            visual = self._create_forecast_visual(forecast, data)
            
            # Generate explanation
            explanation = self.commentary_agent.process_prompt(
                f"Explain the forecast for {ticker} over the next {duration}"
            )
            
            return AgentResult(
                data=forecast,
                visual=visual,
                explanation=explanation,
                metadata={
                    'ticker': ticker,
                    'duration': duration,
                    'output_format': output_format
                }
            )
            
        except Exception as e:
            self.logger.error("Error in forecast handler: %s", str(e))
            return AgentResult(
                data=None,
                status="error",
                error=str(e)
            )

    def _handle_strategy(self, prompt: str, parsed_prompt: Dict, **kwargs) -> AgentResult:
        """Handle strategy requests."""
        try:
            # Extract parameters
            strategy_name = parsed_prompt.get('strategy')
            ticker = parsed_prompt.get('ticker')
            
            # Get data
            data = kwargs.get('data')
            if data is None:
                raise AgentError("No data provided for strategy")
            
            # Generate signals
            signals = self.strategy_manager.generate_signals(data)
            
            # Generate visualization
            visual = self._create_strategy_visual(signals, data)
            
            # Generate explanation
            explanation = self.commentary_agent.process_prompt(
                f"Explain the {strategy_name} strategy signals for {ticker}"
            )
            
            return AgentResult(
                data=signals,
                visual=visual,
                explanation=explanation,
                metadata={
                    'strategy': strategy_name,
                    'ticker': ticker
                }
            )
            
        except Exception as e:
            self.logger.error("Error in strategy handler: %s", str(e))
            return AgentResult(
                data=None,
                status="error",
                error=str(e)
            )

    def _handle_backtest(self, prompt: str, parsed_prompt: Dict, **kwargs) -> AgentResult:
        """Handle backtesting requests."""
        try:
            # Extract parameters
            strategy_name = parsed_prompt.get('strategy')
            ticker = parsed_prompt.get('ticker')
            
            # Get parameters
            params = kwargs.get('params', {})
            
            # Run backtest
            results = self.backtester.run_backtest(**params)
            
            # Generate visualization
            visual = self._create_backtest_visual(results)
            
            # Generate explanation
            explanation = self.commentary_agent.process_prompt(
                f"Explain the backtest results for {strategy_name} strategy on {ticker}"
            )
            
            return AgentResult(
                data=results,
                visual=visual,
                explanation=explanation,
                metadata={
                    'strategy': strategy_name,
                    'ticker': ticker
                }
            )
            
        except Exception as e:
            self.logger.error("Error in backtest handler: %s", str(e))
            return AgentResult(
                data=None,
                status="error",
                error=str(e)
            )

    def _handle_commentary(self, prompt: str, parsed_prompt: Dict, **kwargs) -> AgentResult:
        """Handle commentary and explanation requests."""
        try:
            # Generate explanation
            explanation = self.commentary_agent.process_prompt(prompt)
            
            return AgentResult(
                data=None,
                explanation=explanation
            )
            
        except Exception as e:
            self.logger.error("Error in commentary handler: %s", str(e))
            return AgentResult(
                data=None,
                status="error",
                error=str(e)
            )

    def _create_forecast_visual(self, forecast: pd.DataFrame, data: pd.DataFrame) -> go.Figure:
        """Create visualization for forecast results."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                name='Historical',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add forecast
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast['forecast'],
                name='Forecast',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast['upper'],
                name='Upper Bound',
                line=dict(color='gray', dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast['lower'],
                name='Lower Bound',
                line=dict(color='gray', dash='dot'),
                fill='tonexty',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.update_layout(
            title='Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price',
            showlegend=True
        )
        
        return fig

    def _create_strategy_visual(self, signals: pd.DataFrame, data: pd.DataFrame) -> go.Figure:
        """Create visualization for strategy signals."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Add price data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                name='Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add signals
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['signal'],
                name='Signal',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Strategy Signals',
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True
        )
        
        return fig

    def _create_backtest_visual(self, results: Dict) -> go.Figure:
        """Create visualization for backtest results."""
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        
        # Add portfolio value
        fig.add_trace(
            go.Scatter(
                y=results['portfolio_values'],
                name='Portfolio Value',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                y=results['drawdowns'],
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Add returns
        fig.add_trace(
            go.Scatter(
                y=results['returns'],
                name='Returns',
                line=dict(color='green')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Backtest Results',
            xaxis_title='Time',
            yaxis_title='Value',
            showlegend=True,
            height=900
        )
        
        return fig

    def _combine_results(self, results: List[AgentResult]) -> AgentResult:
        """Combine multiple agent results into a single result."""
        combined_data = {}
        combined_visual = None
        combined_explanation = []
        combined_metadata = {}
        
        for result in results:
            if result.data is not None:
                combined_data.update(result.data)
            if result.visual is not None:
                if combined_visual is None:
                    combined_visual = result.visual
                else:
                    # Combine visualizations if needed
                    pass
            if result.explanation:
                combined_explanation.append(result.explanation)
            if result.metadata:
                combined_metadata.update(result.metadata)
        
        return AgentResult(
            data=combined_data,
            visual=combined_visual,
            explanation="\n\n".join(combined_explanation) if combined_explanation else None,
            metadata=combined_metadata
        )
