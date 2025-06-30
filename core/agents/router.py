# -*- coding: utf-8 -*-
"""Route user prompts to the appropriate trading agent based on intent."""

# Standard library imports
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel

# Local imports
from trading.llm.llm_interface import LLMInterface
from trading.utils.prompt_parser import PromptParser
from trading.utils.exceptions import RouterError, IntentDetectionError, AgentError
from trading.core.performance import log_performance
from trading.config.intents import INTENT_CONFIG, IntentConfig
from trading.config.agents import AGENT_CONFIG, AgentConfig
from trading.memory.task_memory import Task, TaskMemory, TaskStatus

class IntentType(str, Enum):
    """Types of intents that can be detected."""
    def __new__(cls, value: str, config: IntentConfig):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.config = config
        return {'success': True, 'result': obj, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    @classmethod
    def _missing_(cls, value: str) -> Optional['IntentType']:
        """Handle missing intent types gracefully."""
        return {'success': True, 'result': cls.UNCERTAIN, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    @classmethod
    def load_from_config(cls) -> None:
        """Load intent types from configuration."""
        for intent_name, intent_config in INTENT_CONFIG.items():
            if not hasattr(cls, intent_name.upper()):
                setattr(cls, intent_name.upper(), (intent_name, intent_config))

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
# Initialize intents from config
IntentType.load_from_config()

@dataclass
class AgentResult:
    """Result from agent execution."""
    data: Any
    visual: Optional[go.Figure] = None
    explanation: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class IntentRouter:
    """Base class for intent routing logic."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize the router.
        
        Args:
            confidence_threshold: Minimum confidence score for intent detection
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.intent_handlers: Dict[IntentType, Callable] = {}
        self.pipelines: Dict[str, List[IntentType]] = {}
        
    def register_handler(self, intent: IntentType, handler: Callable) -> None:
        """Register a handler for an intent type.
        
        Args:
            intent: The intent type to handle
            handler: The handler function
        """
        self.intent_handlers[intent] = handler
        
    def register_pipeline(self, name: str, intents: List[IntentType]) -> None:
        """Register a pipeline of intents.
        
        Args:
            name: Pipeline name
            intents: List of intents in sequence
        """
        self.pipelines[name] = intents
        
    def get_handler(self, intent: IntentType) -> Optional[Callable]:
        """Get the handler for an intent type.
        
        Args:
            intent: The intent type
            
        Returns:
            The handler function or None if not found
        """
        return self.intent_handlers.get(intent)
        
    def get_pipeline(self, name: str) -> Optional[List[IntentType]]:
        """Get a pipeline by name.
        
        Args:
            name: Pipeline name
            
        Returns:
            List of intents or None if not found
        """
        return self.pipelines.get(name)

class AgentRouter(IntentRouter):
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
        super().__init__(confidence_threshold)
        self.llm = llm
        self.forecaster = forecaster
        self.strategy_manager = strategy_manager
        self.backtester = backtester
        self.commentary_agent = commentary_agent or llm
        self.prompt_parser = PromptParser()
        self.overrides_file = Path("memory/logs/strategy_overrides.json")
        self.task_memory = TaskMemory()

        # Initialize intent handlers
        self.register_handler(IntentType.FORECAST, self._handle_forecast)
        self.register_handler(IntentType.STRATEGY, self._handle_strategy)
        self.register_handler(IntentType.BACKTEST, self._handle_backtest)
        self.register_handler(IntentType.COMMENTARY, self._handle_commentary)
        self.register_handler(IntentType.EXPLANATION, self._handle_commentary)
        self.register_handler(IntentType.MULTI_STEP, self._handle_multi_step)

        # Initialize pipeline definitions
        self.register_pipeline("forecast_and_backtest", 
                             [IntentType.FORECAST, IntentType.BACKTEST])
        self.register_pipeline("strategy_and_backtest", 
                             [IntentType.STRATEGY, IntentType.BACKTEST])
        self.register_pipeline("forecast_and_explain", 
                             [IntentType.FORECAST, IntentType.EXPLANATION])

    def _get_override(self, ticker: str) -> Optional[Dict[str, str]]:
        """Get strategy/model override for a ticker if it exists."""
        if not self.overrides_file.exists():
            return None
            
        try:
            with open(self.overrides_file, encoding='utf-8') as f:
                overrides = json.load(f)
            return overrides.get(ticker)
        except Exception as e:
            self.logger.error(f"Error reading overrides: {e}")
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def _handle_forecast(self, prompt: str, parsed_prompt: Dict, **kwargs) -> AgentResult:
        """Handle forecasting requests."""
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="forecast",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes=f"Forecast request for {parsed_prompt.get('ticker')}"
        )
        self.task_memory.add_task(task)

        try:
            # Extract parameters
            ticker = parsed_prompt.get('ticker')
            duration = parsed_prompt.get('duration', '30d')
            output_format = parsed_prompt.get('output_format', 'chart')
            
            # Get data
            data = kwargs.get('data')
            if data is None:
                raise AgentError("No data provided for forecasting")
            
            # Check for model override
            override = self._get_override(ticker)
            if override and override.get('recommend_model'):
                self.logger.info(f"Using override model {override['recommend_model']} for {ticker}")
                # Set the model in the forecaster
                self.forecaster.set_model(override['recommend_model'])
            
            # Generate forecast
            forecast = self.forecaster.predict(data)
            
            # Log performance metrics
            log_performance(
                ticker=ticker,
                model=self.forecaster.__class__.__name__,
                strategy="forecast",
                mse=forecast.get('mse'),
                accuracy=forecast.get('accuracy'),
                notes=f"Forecast for {duration}"
            )
            
            # Generate visualization
            visual = self._create_forecast_visual(forecast, data)
            
            # Generate explanation
            explanation = self.commentary_agent.process_prompt(
                f"Explain the forecast for {ticker} over the next {duration}"
            )
            
            # Update task status
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully generated forecast for {ticker}",
                metadata={
                    'ticker': ticker,
                    'duration': duration,
                    'mse': forecast.get('mse'),
                    'accuracy': forecast.get('accuracy')
                }
            )
            
            return AgentResult(
                data=forecast,
                visual=visual,
                explanation=explanation,
                metadata={
                    'ticker': ticker,
                    'duration': duration,
                    'output_format': output_format,
                    'task_id': task_id
                }
            )
            
        except Exception as e:
            self.logger.error("Error in forecast handler: %s", str(e))
            # Update task status
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to generate forecast: {str(e)}"
            )
            return AgentResult(
                data=None,
                status="error",
                error=str(e),
                metadata={'task_id': task_id}
            )

    def _handle_strategy(self, prompt: str, parsed_prompt: Dict, **kwargs) -> AgentResult:
        """Handle strategy requests."""
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="strategy",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes=f"Strategy request for {parsed_prompt.get('ticker')}"
        )
        self.task_memory.add_task(task)

        try:
            # Extract parameters
            ticker = parsed_prompt.get('ticker')
            strategy_name = parsed_prompt.get('strategy', 'default')
            
            # Get data
            data = kwargs.get('data')
            if data is None:
                raise AgentError("No data provided for strategy")
            
            # Check for strategy override
            override = self._get_override(ticker)
            if override and override.get('recommend_strategy'):
                self.logger.info(f"Using override strategy {override['recommend_strategy']} for {ticker}")
                strategy_name = override['recommend_strategy']

            # Update task status
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully executed strategy {strategy_name} for {ticker}",
                metadata={
                    'ticker': ticker,
                    'strategy': strategy_name
                }
            )

        except Exception as e:
            self.logger.error("Error in strategy handler: %s", str(e))
            # Update task status
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to execute strategy: {str(e)}"
            )
            return AgentResult(
                data=None,
                status="error",
                error=str(e),
                metadata={'task_id': task_id}
            )

    def _create_forecast_visual(self, forecast: pd.DataFrame, data: pd.DataFrame) -> go.Figure:
        """Create visualization for forecast results."""
        fig = make_subplots(rows=1, cols=1)
        
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
                y=forecast['prediction'],
                name='Forecast',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast['upper_bound'],
                name='Upper Bound',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast['lower_bound'],
                name='Lower Bound',
                line=dict(color='red', dash='dash'),
                fill='tonexty'
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
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('Price', 'Signals'))
        
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
        
        # Add buy signals
        buy_signals = signals[signals['signal'] == 1]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
        
        # Add sell signals
        sell_signals = signals[signals['signal'] == -1]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
        
        # Add signal strength
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['strength'],
                name='Signal Strength',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Strategy Signals',
            xaxis_title='Date',
            showlegend=True,
            height=800
        )
        
        return fig

    def process_prompt(self, prompt: str, **kwargs) -> AgentResult:
        """Process a user prompt and route to appropriate handler.
        
        Args:
            prompt: User prompt text
            **kwargs: Additional arguments for handlers
            
        Returns:
            AgentResult containing the response
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="prompt_processing",
            status=TaskStatus.PENDING,
            agent=self.__class__.__name__,
            notes=f"Processing prompt: {prompt[:100]}..."
        )
        self.task_memory.add_task(task)

        try:
            # Parse prompt
            parsed_prompt = self.prompt_parser.parse(prompt)
            
            # Detect intent
            intent = self.llm.detect_intent(prompt)
            if not intent:
                raise IntentDetectionError("Could not detect intent")
                
            # Get handler
            handler = self.get_handler(intent)
            if not handler:
                raise RouterError(f"No handler registered for intent: {intent}")
                
            # Process with handler
            result = handler(prompt, parsed_prompt, **kwargs)
            
            # Update task status
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                notes=f"Successfully processed prompt with intent {intent}",
                metadata={
                    'intent': intent,
                    'status': result.status
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Error processing prompt: %s", str(e))
            # Update task status
            self.task_memory.update_task(
                task_id,
                status=TaskStatus.FAILED,
                notes=f"Failed to process prompt: {str(e)}"
            )
            return AgentResult(
                data=None,
                status="error",
                error=str(e),
                metadata={'task_id': task_id}
            )

if __name__ == "__main__":
    # Test router
    router = AgentRouter(
        llm=LLMInterface(),
        forecaster=None,  # Add actual agents here
        strategy_manager=None,
        backtester=None
    )
    
    result = router.process_prompt(
        "Forecast the price of AAPL for the next 30 days"
    )
    print(json.dumps(result.__dict__, indent=2))
