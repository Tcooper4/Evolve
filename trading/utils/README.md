# Reasoning Logger System

A comprehensive system for recording, displaying, and analyzing agent decisions in plain language for transparency and explainability.

## Features

- **Decision Logging**: Record every decision agents make with detailed context
- **Plain Language Summaries**: Human-readable summaries of complex decisions
- **Chat-Style Explanations**: Conversational explanations of why actions were taken
- **Real-Time Updates**: Live monitoring of agent decisions via Redis
- **Multiple Display Formats**: Terminal and Streamlit interfaces
- **Search & Filter**: Find and analyze specific decisions
- **Statistics & Analytics**: Comprehensive decision analytics

## Quick Start

### Basic Usage

```python
from trading.utils.reasoning_logger import ReasoningLogger, DecisionType, ConfidenceLevel

# Initialize logger
logger = ReasoningLogger()

# Log a decision
decision_id = logger.log_decision(
    agent_name='LSTMForecaster',
    decision_type=DecisionType.FORECAST,
    action_taken='Predicted AAPL will reach $185.50 in 7 days',
    context={
        'symbol': 'AAPL',
        'timeframe': '1h',
        'market_conditions': {'trend': 'bullish', 'rsi': 65},
        'available_data': ['price', 'volume', 'rsi', 'macd'],
        'constraints': {'max_forecast_days': 30},
        'user_preferences': {'risk_tolerance': 'medium'}
    },
    reasoning={
        'primary_reason': 'Strong technical indicators showing bullish momentum',
        'supporting_factors': [
            'RSI indicates bullish momentum (65)',
            'MACD shows positive crossover',
            'Volume is above average'
        ],
        'alternatives_considered': [
            'Conservative forecast of $180.00',
            'Aggressive forecast of $190.00'
        ],
        'risks_assessed': [
            'Market volatility could increase',
            'Earnings announcement next week'
        ],
        'confidence_explanation': 'High confidence due to strong technical signals',
        'expected_outcome': 'AAPL expected to continue bullish trend'
    },
    confidence_level=ConfidenceLevel.HIGH,
    metadata={'model_name': 'LSTM_v2', 'forecast_value': 185.50}
)

print(f"Decision logged: {decision_id}")
```

### Convenience Functions

```python
from trading.utils.reasoning_logger import log_forecast_decision, log_strategy_decision

# Log forecast decision
forecast_id = log_forecast_decision(
    agent_name='LSTMForecaster',
    symbol='AAPL',
    timeframe='1h',
    forecast_value=185.50,
    confidence=0.85,
    reasoning={
        'primary_reason': 'Technical analysis shows bullish momentum',
        'supporting_factors': ['RSI oversold', 'MACD positive'],
        'alternatives_considered': ['Wait for confirmation'],
        'risks_assessed': ['Market volatility'],
        'confidence_explanation': 'High confidence due to clear signals',
        'expected_outcome': 'Expected 5% upside'
    }
)

# Log strategy decision
strategy_id = log_strategy_decision(
    agent_name='RSIStrategy',
    symbol='AAPL',
    action='BUY 100 shares at $182.30',
    strategy_name='RSI Mean Reversion',
    reasoning={
        'primary_reason': 'RSI oversold condition with strong support',
        'supporting_factors': ['RSI below 40', 'Price near support'],
        'alternatives_considered': ['Wait for lower entry'],
        'risks_assessed': ['Support could break'],
        'confidence_explanation': 'Medium confidence due to clear setup',
        'expected_outcome': 'Expect 3-5% upside'
    }
)
```

## Display Components

### Terminal Display

```python
from trading.utils.reasoning_display import ReasoningDisplay

# Initialize display
display = ReasoningDisplay(logger)

# Display recent decisions
display.display_recent_decisions_terminal(limit=10)

# Display specific decision
decision = logger.get_decision(decision_id)
display.display_decision_terminal(decision)

# Display statistics
display.display_statistics_terminal()
```

### Streamlit Display

```python
# Create complete reasoning page
from trading.utils.reasoning_display import create_reasoning_page_streamlit

# In your Streamlit app
create_reasoning_page_streamlit()
```

Or use individual components:

```python
# Display decision in Streamlit
display.display_decision_streamlit(decision)

# Display recent decisions
display.display_recent_decisions_streamlit(limit=10)

# Display statistics
display.display_statistics_streamlit()

# Create sidebar controls
filters = display.create_streamlit_sidebar()
```

## Real-Time Service

### Start the Reasoning Service

```bash
# Start reasoning service
python trading/utils/launch_reasoning_service.py

# Or with custom configuration
python trading/utils/launch_reasoning_service.py \
    --redis-host localhost \
    --redis-port 6379 \
    --service-name reasoning_service
```

### Service Integration

The reasoning service automatically listens for:
- `agent_decisions` events
- `forecast_completed` events
- `strategy_completed` events
- `model_evaluation_completed` events

And publishes `reasoning_updates` events with decision summaries.

## Decision Types

### DecisionType Enum

- `FORECAST`: Price predictions and forecasts
- `STRATEGY`: Trading strategy execution
- `MODEL_SELECTION`: Model selection and evaluation
- `PARAMETER_TUNING`: Hyperparameter optimization
- `RISK_MANAGEMENT`: Risk assessment and management
- `PORTFOLIO_ALLOCATION`: Portfolio allocation decisions
- `SIGNAL_GENERATION`: Trading signal generation
- `DATA_SELECTION`: Data source and feature selection
- `FEATURE_ENGINEERING`: Feature engineering decisions
- `BACKTEST`: Backtesting results and analysis
- `OPTIMIZATION`: Strategy optimization decisions
- `ALERT`: System alerts and notifications

### ConfidenceLevel Enum

- `VERY_LOW`: Very low confidence (0-20%)
- `LOW`: Low confidence (20-40%)
- `MEDIUM`: Medium confidence (40-60%)
- `HIGH`: High confidence (60-80%)
- `VERY_HIGH`: Very high confidence (80-100%)

## Data Structures

### DecisionContext

```python
@dataclass
class DecisionContext:
    symbol: str
    timeframe: str
    timestamp: str
    market_conditions: Dict[str, Any]
    available_data: List[str]
    constraints: Dict[str, Any]
    user_preferences: Dict[str, Any]
```

### DecisionReasoning

```python
@dataclass
class DecisionReasoning:
    primary_reason: str
    supporting_factors: List[str]
    alternatives_considered: List[str]
    risks_assessed: List[str]
    confidence_explanation: str
    expected_outcome: str
```

### AgentDecision

```python
@dataclass
class AgentDecision:
    decision_id: str
    agent_name: str
    decision_type: DecisionType
    action_taken: str
    context: DecisionContext
    reasoning: DecisionReasoning
    confidence_level: ConfidenceLevel
    timestamp: str
    metadata: Dict[str, Any]
```

## API Reference

### ReasoningLogger

#### Methods

- `log_decision()`: Log a new decision
- `get_decision()`: Retrieve a specific decision
- `get_agent_decisions()`: Get decisions by agent
- `get_decisions_by_type()`: Get decisions by type
- `get_summary()`: Get plain language summary
- `get_explanation()`: Get chat-style explanation
- `get_statistics()`: Get decision statistics
- `clear_old_decisions()`: Clean up old decisions

### ReasoningDisplay

#### Methods

- `display_decision_terminal()`: Display decision in terminal
- `display_recent_decisions_terminal()`: Display recent decisions in terminal
- `display_statistics_terminal()`: Display statistics in terminal
- `display_decision_streamlit()`: Display decision in Streamlit
- `display_recent_decisions_streamlit()`: Display recent decisions in Streamlit
- `display_statistics_streamlit()`: Display statistics in Streamlit
- `create_streamlit_sidebar()`: Create Streamlit sidebar controls
- `display_live_feed_streamlit()`: Display live decision feed

### ReasoningService

#### Methods

- `start()`: Start the service
- `stop()`: Stop the service
- `get_status()`: Get service status
- `get_recent_decisions()`: Get recent decisions from cache
- `get_statistics()`: Get reasoning statistics

## Examples

### Integration with Trading System

```python
# After making a forecast
def on_forecast_completed(forecast_result):
    """Handle forecast completion."""
    
    decision_id = log_forecast_decision(
        agent_name='LSTMForecaster',
        symbol=forecast_result['symbol'],
        timeframe=forecast_result['timeframe'],
        forecast_value=forecast_result['prediction'],
        confidence=forecast_result['confidence'],
        reasoning={
            'primary_reason': forecast_result['reasoning']['primary_reason'],
            'supporting_factors': forecast_result['reasoning']['factors'],
            'alternatives_considered': forecast_result['reasoning']['alternatives'],
            'risks_assessed': forecast_result['reasoning']['risks'],
            'confidence_explanation': forecast_result['reasoning']['confidence_explanation'],
            'expected_outcome': forecast_result['reasoning']['expected_outcome']
        }
    )
    
    return decision_id

# After executing a strategy
def on_strategy_executed(strategy_result):
    """Handle strategy execution."""
    
    decision_id = log_strategy_decision(
        agent_name=strategy_result['agent_name'],
        symbol=strategy_result['symbol'],
        action=strategy_result['action'],
        strategy_name=strategy_result['strategy_name'],
        reasoning=strategy_result['reasoning']
    )
    
    return decision_id
```

### Custom Decision Types

```python
# Log a custom decision type
decision_id = logger.log_decision(
    agent_name='CustomAgent',
    decision_type=DecisionType.RISK_MANAGEMENT,
    action_taken='Reduced position size by 50% due to high volatility',
    context={
        'symbol': 'AAPL',
        'timeframe': '1h',
        'market_conditions': {'volatility': 'high', 'vix': 25},
        'available_data': ['price', 'volatility', 'vix'],
        'constraints': {'max_risk_per_trade': 0.02},
        'user_preferences': {'risk_averse': True}
    },
    reasoning={
        'primary_reason': 'High market volatility detected',
        'supporting_factors': ['VIX above 20', 'Price volatility increased'],
        'alternatives_considered': ['Close position', 'Hedge with options'],
        'risks_assessed': ['Further volatility increase', 'Gap risk'],
        'confidence_explanation': 'High confidence in risk assessment',
        'expected_outcome': 'Reduced exposure while maintaining position'
    },
    confidence_level=ConfidenceLevel.HIGH,
    metadata={'original_position': 200, 'new_position': 100}
)
```

### Search and Filter

```python
# Get decisions by agent
lstm_decisions = logger.get_agent_decisions('LSTMForecaster', limit=20)

# Get decisions by type
forecast_decisions = logger.get_decisions_by_type(DecisionType.FORECAST, limit=10)

# Get explanations
for decision in lstm_decisions:
    explanation = logger.get_explanation(decision.decision_id)
    summary = logger.get_summary(decision.decision_id)
    print(f"Decision: {decision.decision_id}")
    print(f"Summary: {summary[:100]}...")
    print(f"Explanation: {explanation[:100]}...")
```

## Configuration

### Environment Variables

```bash
# Redis configuration
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_DB="0"

# OpenAI for enhanced explanations (optional)
export OPENAI_API_KEY="your_openai_api_key"

# Service configuration
export REASONING_SERVICE_NAME="reasoning_service"
```

### Redis Channels

The system uses the following Redis channels:
- `agent_decisions`: New agent decisions
- `reasoning_updates`: Real-time reasoning updates
- `forecast_completed`: Forecast completion events
- `strategy_completed`: Strategy completion events
- `model_evaluation_completed`: Model evaluation events

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python trading/utils/test_reasoning.py

# Run specific test class
python -m unittest trading.utils.test_reasoning.TestReasoningLogger

# Run with coverage
coverage run trading/utils/test_reasoning.py
coverage report
```

## Demo

Run the demo to see the system in action:

```bash
python trading/utils/demo_reasoning.py
```

This will:
1. Create sample decisions
2. Demonstrate display components
3. Show real-time updates
4. Test search and filtering
5. Display statistics

## Integration with Existing Services

The reasoning logger integrates with the existing service infrastructure:

```python
# Add to ServiceManager
from utils.reasoning_service import ReasoningService

# Add reasoning service to services dictionary
self.services['reasoning'] = {
    'script': 'launch_reasoning_service.py',
    'description': 'Reasoning Logger Service',
    'status': 'stopped',
    'process': None,
    'pid': None
}

# Add to ServiceClient
def get_reasoning_decisions(self, agent_name: str = None, limit: int = 10):
    """Get reasoning decisions."""
    # Implementation here
```

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```
   Error: Redis connection failed
   Solution: Ensure Redis is running and accessible
   ```

2. **Missing Dependencies**
   ```
   Error: Module not found
   Solution: Install required packages: pip install redis openai jinja2
   ```

3. **File Permission Error**
   ```
   Error: Cannot write to log directory
   Solution: Ensure write permissions to logs directory
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific components
logging.getLogger('trading.utils.reasoning_logger').setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 