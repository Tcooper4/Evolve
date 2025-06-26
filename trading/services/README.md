# Trading Agent Services

This directory contains the service architecture for the Evolve trading system, converting each major agent into its own independent service that communicates via Redis pub/sub.

## Architecture Overview

The services architecture provides:

- **Independent Service Execution**: Each agent runs as a separate Python process
- **Redis Pub/Sub Communication**: Services communicate via Redis channels
- **Parallel Processing**: Services can run simultaneously and independently
- **Persistent Messaging**: Messages persist between service runs
- **Centralized Management**: ServiceManager provides unified control

## Service Components

### Core Services

1. **ModelBuilderService** (`model_builder_service.py`)
   - Wraps ModelBuilderAgent functionality
   - Handles model building requests
   - Supports LSTM, XGBoost, ensemble models
   - Channels: `model_builder_input`, `model_builder_output`, `model_builder_control`

2. **PerformanceCriticService** (`performance_critic_service.py`)
   - Wraps PerformanceCriticAgent functionality
   - Evaluates models on performance metrics
   - Compares models and identifies best performers
   - Channels: `performance_critic_input`, `performance_critic_output`, `performance_critic_control`

3. **UpdaterService** (`updater_service.py`)
   - Wraps UpdaterAgent functionality
   - Handles model retraining and updates
   - Manages model replacement and tuning
   - Channels: `updater_input`, `updater_output`, `updater_control`

### Specialized Services

4. **ResearchService** (`research_service.py`)
   - Wraps ResearchAgent functionality
   - Searches GitHub and arXiv for new models/strategies
   - Summarizes papers and generates code suggestions
   - Channels: `research_input`, `research_output`, `research_control`

5. **MetaTunerService** (`meta_tuner_service.py`)
   - Wraps MetaTunerAgent functionality
   - Auto-tunes model hyperparameters
   - Supports Bayesian optimization and grid search
   - Channels: `meta_tuner_input`, `meta_tuner_output`, `meta_tuner_control`

6. **MultimodalService** (`multimodal_service.py`)
   - Wraps MultimodalAgent functionality
   - Generates plots and visualizations
   - Analyzes images with vision models
   - Channels: `multimodal_input`, `multimodal_output`, `multimodal_control`

7. **PromptRouterService** (`prompt_router_service.py`)
   - Wraps PromptRouterAgent functionality
   - Routes user prompts to appropriate services
   - Detects intent and parses arguments
   - Channels: `prompt_router_input`, `prompt_router_output`, `prompt_router_control`

8. **QuantGPTService** (`quant_gpt_service.py`)
   - Natural language interface for the trading system
   - Processes queries like "Give me the best model for NVDA over 90 days"
   - Provides GPT-powered commentary on trading decisions
   - Integrates with all other services for comprehensive analysis
   - Channels: `quant_gpt_input`, `quant_gpt_output`, `quant_gpt_control`

9. **SafeExecutorService** (`safe_executor_service.py`)
   - Safe execution environment for user-defined models and strategies
   - Provides timeout protection, memory limits, and isolated execution
   - Blocks dangerous code and validates user input
   - Supports models, strategies, and technical indicators
   - Channels: `safe_executor_input`, `safe_executor_output`, `safe_executor_control`

### Infrastructure

10. **BaseService** (`base_service.py`)
    - Abstract base class for all services
    - Provides Redis pub/sub infrastructure
    - Handles message routing and error handling

11. **ServiceManager** (`service_manager.py`)
    - Centralized service orchestration
    - Manages service lifecycle (start/stop/monitor)
    - Provides unified interface for all services

12. **ServiceClient** (`service_client.py`)
    - Client library for interacting with services
    - Provides high-level API for service communication
    - Handles request/response patterns

## Installation and Setup

### Prerequisites

1. **Redis Server**
   ```bash
   # Install Redis (Ubuntu/Debian)
   sudo apt-get install redis-server
   
   # Install Redis (macOS)
   brew install redis
   
   # Start Redis
   redis-server
   ```

2. **Python Dependencies**
   ```bash
   pip install redis
   ```

### Service Launchers

Each service has its own launcher script:

- `launch_model_builder.py` - Model Builder Service
- `launch_performance_critic.py` - Performance Critic Service
- `launch_updater.py` - Updater Service
- `launch_research.py` - Research Service
- `launch_meta_tuner.py` - Meta Tuner Service
- `launch_multimodal.py` - Multimodal Service
- `launch_prompt_router.py` - Prompt Router Service
- `launch_quant_gpt.py` - QuantGPT Natural Language Interface
- `launch_safe_executor.py` - Safe Executor for User-Defined Models

## Usage

### Starting Services

#### Using ServiceManager (Recommended)

```bash
# Start all services
python service_manager.py --action start-all

# Start specific service
python service_manager.py --action start --service model_builder

# Check service status
python service_manager.py --action status

# Stop all services
python service_manager.py --action stop-all
```

#### Manual Service Launch

```bash
# Start individual services
python launch_model_builder.py
python launch_performance_critic.py
python launch_updater.py
# ... etc
```

### Using ServiceClient

```python
from services.service_client import ServiceClient

# Initialize client
client = ServiceClient()

# Build a model
result = client.build_model('lstm', 'BTCUSDT', '1h')
print(result)

# Evaluate a model
result = client.evaluate_model('model_123', 'BTCUSDT', '1h')
print(result)

# Retrain a model
result = client.retrain_model('model_123')
print(result)

# Search for new models
result = client.search_github('trading bot machine learning')
print(result)

# Tune hyperparameters
result = client.tune_hyperparameters('lstm')
print(result)

# Generate plots
result = client.generate_plot('equity_curve', 'backtest_results')
print(result)

# Route a prompt
result = client.route_prompt('Build me an LSTM model for Bitcoin prediction')
print(result)

# Process natural language queries with QuantGPT
result = client.process_natural_language_query('Give me the best model for NVDA over 90 days')
print(result)

# Get query history
result = client.get_query_history(limit=10, symbol='NVDA')
print(result)

# Get available symbols and parameters
result = client.get_available_symbols()
print(result)

# Execute user-defined models safely
result = client.execute_model_safely(
    model_code='def main(input_data): return {"prediction": 100.5}',
    model_name="custom_model",
    input_data={"prices": [100, 101, 102]},
    model_type="custom"
)
print(result)

# Execute user-defined strategies safely
result = client.execute_strategy_safely(
    strategy_code='def main(input_data): return {"signal": "BUY", "confidence": 0.8}',
    strategy_name="custom_strategy",
    market_data={"prices": [100, 101, 102]},
    parameters={"threshold": 0.5}
)
print(result)

# Execute user-defined indicators safely
result = client.execute_indicator_safely(
    indicator_code='def main(input_data): return {"ma": 101.0, "signal": "BULLISH"}',
    indicator_name="custom_indicator",
    price_data={"prices": [100, 101, 102]},
    parameters={"window": 3}
)
print(result)

# Get SafeExecutor statistics
result = client.get_safe_executor_statistics()
print(result)

# Clean up SafeExecutor resources
result = client.cleanup_safe_executor()
print(result)

client.close()
```

### Command Line Interface

```bash
# Build a model
python service_client.py --action build

# Evaluate a model
python service_client.py --action evaluate --service model_123

# Retrain a model
python service_client.py --action retrain --service model_123

# Search GitHub
python service_client.py --action search

# Tune hyperparameters
python service_client.py --action tune

# Generate plots
python service_client.py --action plot

# Route prompts
python service_client.py --action route
```

## Message Format

### Request Messages

```json
{
  "type": "message_type",
  "data": {
    "param1": "value1",
    "param2": "value2"
  },
  "timestamp": 1640995200.0
}
```

### Response Messages

```json
{
  "type": "response_type",
  "service": "service_name",
  "data": {
    "result": "response_data"
  },
  "status": "success",
  "timestamp": 1640995200.0
}
```

## Service Communication Flow

1. **Client sends request** to service input channel
2. **Service processes request** using underlying agent
3. **Service logs decision** to persistent memory
4. **Service sends response** to output channel
5. **Client receives response** and processes result

## QuantGPT Natural Language Interface

QuantGPT provides a natural language interface to the entire trading system, allowing users to interact with all services using plain English queries.

### Features

- **Natural Language Processing**: Understands queries like "Give me the best model for NVDA over 90 days"
- **GPT-Powered Commentary**: Provides intelligent analysis and explanations of trading decisions
- **Multi-Service Integration**: Automatically routes queries to appropriate services
- **Persistent Memory**: Remembers past queries and decisions for context
- **Real-time Analysis**: Processes queries and returns results with GPT commentary

### Example Queries

```python
# Model recommendations
"Give me the best model for NVDA over 90 days"
"Find the optimal model for TSLA on 1h timeframe"
"What model should I use for BTCUSDT?"

# Trading signals
"Should I long TSLA this week?"
"What's the trading signal for AAPL?"
"Should I buy GOOGL now?"

# Market analysis
"Analyze BTCUSDT market conditions"
"What's happening with ETHUSDT?"
"Give me a market overview for MSFT"

# General queries
"What's the best strategy for crypto trading?"
"Which stocks are performing well?"
"Show me the latest model performance"
```

### Usage Examples

#### Direct QuantGPT Usage

```python
from services.quant_gpt import QuantGPT

# Initialize QuantGPT
quant_gpt = QuantGPT(openai_api_key='your-api-key')

# Process a query
result = quant_gpt.process_query("Give me the best model for NVDA over 90 days")

# Display results
if result['status'] == 'success':
    print(f"Intent: {result['parsed_intent']['intent']}")
    print(f"Symbol: {result['parsed_intent']['symbol']}")
    print(f"GPT Commentary: {result['gpt_commentary']}")
```

#### Interactive Mode

```bash
# Launch interactive QuantGPT
python launch_quant_gpt.py

# Enter queries interactively
💬 Query: Give me the best model for NVDA over 90 days
💬 Query: Should I long TSLA this week?
💬 Query: quit
```

#### Service Integration

```python
from services.service_client import ServiceClient

client = ServiceClient()

# Process natural language query
result = client.process_natural_language_query("Should I buy AAPL now?")

# Get query history
history = client.get_query_history(limit=10, symbol='AAPL')

# Get available symbols
symbols = client.get_available_symbols()
```

### Query Processing Flow

1. **Query Parsing**: Uses GPT or regex to extract intent and parameters
2. **Intent Detection**: Identifies the type of request (model_recommendation, trading_signal, etc.)
3. **Service Routing**: Routes to appropriate services (ModelBuilder, PerformanceCritic, etc.)
4. **Result Aggregation**: Combines results from multiple services
5. **GPT Commentary**: Generates intelligent analysis and explanations
6. **Memory Logging**: Stores query and results for future reference

### Supported Parameters

- **Symbols**: BTCUSDT, ETHUSDT, NVDA, TSLA, AAPL, GOOGL, MSFT, AMZN
- **Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d
- **Periods**: 7d, 14d, 30d, 90d, 180d, 1y
- **Models**: lstm, xgboost, ensemble, transformer, tcn

### Configuration

```python
# Initialize with custom settings
quant_gpt = QuantGPT(
    openai_api_key='your-api-key',
    redis_host='localhost',
    redis_port=6379,
    redis_db=0
)
```

### Error Handling

- **GPT Unavailable**: Falls back to regex parsing
- **Service Unavailable**: Returns error with helpful message
- **Invalid Queries**: Provides suggestions for valid queries
- **Network Issues**: Handles Redis and API connection problems

## Safe Executor for User-Defined Models

The SafeExecutor provides a secure environment for executing user-defined models, strategies, and technical indicators with comprehensive safety features.

### Security Features

- **Code Validation**: Blocks dangerous imports and functions (os, subprocess, eval, exec, etc.)
- **Timeout Protection**: Automatically terminates long-running executions
- **Memory Limits**: Prevents memory exhaustion attacks
- **Isolated Execution**: Runs code in separate processes with resource limits
- **Sandboxed Environment**: Restricts file system and network access
- **Error Logging**: Comprehensive logging of all executions and errors

### Supported Execution Types

1. **Models**: Custom prediction and forecasting models
2. **Strategies**: Trading strategies with market data
3. **Indicators**: Technical analysis indicators

### Example Usage

#### Safe Model Execution

```python
from services.service_client import ServiceClient

client = ServiceClient()

# Execute a custom model safely
model_code = '''
import numpy as np

def main(input_data):
    prices = input_data.get('prices', [100, 101, 102, 103, 104])
    window = input_data.get('window', 3)
    
    if len(prices) < window:
        return {"error": "Not enough data"}
    
    ma = np.mean(prices[-window:])
    prediction = ma * 1.01
    
    return {
        "prediction": prediction,
        "moving_average": ma,
        "confidence": 0.7
    }
'''

result = client.execute_model_safely(
    model_code=model_code,
    model_name="custom_ma_model",
    input_data={"prices": [100, 101, 102, 103, 104, 105], "window": 3},
    model_type="custom"
)

if result and result.get('type') == 'model_executed':
    execution_result = result.get('result', {})
    if execution_result.get('status') == 'success':
        print(f"Prediction: {execution_result['return_value']['prediction']}")
```

#### Safe Strategy Execution

```python
# Execute a custom strategy safely
strategy_code = '''
def main(input_data):
    market_data = input_data.get('market_data', {})
    rsi = market_data.get('rsi', 65)
    
    if rsi > 70:
        signal = "SELL"
        confidence = 0.8
    elif rsi < 30:
        signal = "BUY"
        confidence = 0.8
    else:
        signal = "HOLD"
        confidence = 0.5
    
    return {
        "signal": signal,
        "confidence": confidence,
        "rsi": rsi
    }
'''

result = client.execute_strategy_safely(
    strategy_code=strategy_code,
    strategy_name="rsi_strategy",
    market_data={"rsi": 75},
    parameters={"oversold": 30, "overbought": 70}
)
```

#### Safe Indicator Execution

```python
# Execute a custom indicator safely
indicator_code = '''
import numpy as np

def main(input_data):
    price_data = input_data.get('price_data', {})
    prices = price_data.get('prices', [])
    fast_period = input_data.get('parameters', {}).get('fast_period', 12)
    
    if len(prices) < fast_period:
        return {"error": "Not enough data"}
    
    fast_ma = np.mean(prices[-fast_period:])
    return {
        "fast_ma": fast_ma,
        "signal": "BULLISH" if fast_ma > prices[0] else "BEARISH"
    }
'''

result = client.execute_indicator_safely(
    indicator_code=indicator_code,
    indicator_name="fast_ma_indicator",
    price_data={"prices": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]},
    parameters={"fast_period": 12}
)
```

### Security Validation

The SafeExecutor automatically blocks dangerous code:

```python
# This will be blocked
dangerous_code = '''
import os
import subprocess

def main(input_data):
    os.system("rm -rf /")  # Blocked!
    return {"status": "dangerous"}
'''

result = client.execute_model_safely(
    model_code=dangerous_code,
    model_name="dangerous_model",
    input_data={}
)

# Result will have status: 'validation_error'
```

### Configuration

```python
# Configure SafeExecutor with custom limits
from utils.safe_executor import SafeExecutor

executor = SafeExecutor(
    timeout_seconds=300,      # 5 minutes
    memory_limit_mb=1024,     # 1GB
    max_output_size=10485760, # 10MB
    enable_sandbox=True,
    log_executions=True
)
```

### Monitoring and Statistics

```python
# Get execution statistics
stats = client.get_safe_executor_statistics()
print(f"Total Executions: {stats['total_executions']}")
print(f"Success Rate: {stats['success_rate']:.1%}")
print(f"Average Execution Time: {stats['average_execution_time']:.2f}s")

# Clean up resources
client.cleanup_safe_executor()
```

### Error Handling

- **Validation Errors**: Dangerous code is blocked before execution
- **Timeout Errors**: Long-running code is automatically terminated
- **Memory Errors**: Memory-intensive code is killed
- **Execution Errors**: Runtime errors are caught and logged
- **System Errors**: Infrastructure issues are handled gracefully

## Monitoring and Logging

### Log Files

Each service creates its own log file:
- `logs/model_builder_service.log`
- `logs/performance_critic_service.log`
- `logs/updater_service.log`
- `logs/research_service.log`
- `logs/meta_tuner_service.log`
- `logs/multimodal_service.log`
- `logs/prompt_router_service.log`
- `logs/quant_gpt_service.log`
- `logs/safe_executor_service.log`

### Redis Monitoring

Monitor Redis channels for real-time communication:
```bash
# Monitor all service channels
redis-cli monitor

# Subscribe to specific service output
redis-cli subscribe model_builder_output
```

## Configuration

### Redis Configuration

Default Redis settings:
- Host: `localhost`
- Port: `6379`
- Database: `0`

Customize in service initialization:
```python
service = ModelBuilderService(
    redis_host='your-redis-host',
    redis_port=6379,
    redis_db=0
)
```

### Service Configuration

Each service can be configured via environment variables:
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export LOG_LEVEL=INFO
```

## Error Handling

### Service Errors

Services handle errors gracefully:
- Invalid requests return error responses
- Service failures are logged
- Redis connection issues are handled with retries
- Graceful shutdown on signals

### Client Errors

Client handles common errors:
- Timeout waiting for responses
- Redis connection failures
- Invalid service names
- Malformed requests

## Scaling and Deployment

### Horizontal Scaling

Services can be scaled horizontally:
- Run multiple instances of the same service
- Use Redis clustering for high availability
- Load balance requests across instances

### Container Deployment

Services can be containerized:
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "services/launch_model_builder.py"]
```

### Kubernetes Deployment

Deploy services to Kubernetes:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-builder-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-builder
  template:
    metadata:
      labels:
        app: model-builder
    spec:
      containers:
      - name: model-builder
        image: evolve/model-builder:latest
        ports:
        - containerPort: 6379
```

## Performance Considerations

### Redis Performance

- Use Redis persistence for message durability
- Configure appropriate memory limits
- Monitor Redis performance metrics
- Consider Redis clustering for high throughput

### Service Performance

- Services run independently for parallel processing
- Each service can be optimized separately
- Monitor service resource usage
- Scale services based on demand

## Security

### Redis Security

- Configure Redis authentication
- Use SSL/TLS for Redis connections
- Restrict Redis network access
- Monitor Redis access logs

### Service Security

- Validate all incoming messages
- Sanitize user inputs
- Implement rate limiting
- Log security events

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis server is running
   - Verify host/port configuration
   - Check network connectivity

2. **Service Not Responding**
   - Check service is running
   - Verify Redis channels
   - Check service logs

3. **Message Timeout**
   - Increase timeout settings
   - Check service processing time
   - Verify message format

### Debug Commands

```bash
# Check Redis status
redis-cli ping

# List Redis channels
redis-cli pubsub channels

# Monitor Redis traffic
redis-cli monitor

# Check service logs
tail -f logs/model_builder_service.log
```