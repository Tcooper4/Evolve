# Live Market Runner

The Live Market Runner is a comprehensive system for streaming live market data, triggering agents periodically, and tracking forecast vs actual results. It operates autonomously and separately from backtesting.

## Features

### 📊 Live Data Streaming
- **Real-time Market Data**: Streams live data from multiple sources
- **Symbol Management**: Tracks multiple symbols simultaneously
- **Price History**: Maintains rolling price and volume history
- **Market Metrics**: Calculates volatility, correlations, and market regime

### 🤖 Agent Triggering
- **Time-based Triggers**: Execute agents at regular intervals
- **Price Move Triggers**: Trigger agents on significant price movements
- **Volume Spike Triggers**: Trigger agents on unusual volume activity
- **Volatility Triggers**: Trigger agents on volatility spikes
- **Manual Triggers**: Support for manual agent execution

### 📈 Forecast Tracking
- **Forecast Storage**: Stores all agent forecasts with metadata
- **Accuracy Tracking**: Compares forecasts with actual results
- **Performance Metrics**: Calculates accuracy, PnL, and success rates
- **Historical Analysis**: Maintains forecast history for analysis

### 🔄 Autonomous Operation
- **Self-contained**: Operates independently of backtesting
- **Continuous Monitoring**: Runs 24/7 with automatic error recovery
- **Performance Monitoring**: Tracks execution times and error rates
- **State Persistence**: Saves state and results to disk

## Architecture

### Core Components

#### LiveMarketRunner
- **Main Controller**: Orchestrates all live market operations
- **Data Management**: Handles live data streaming and updates
- **Agent Coordination**: Manages agent triggering and execution
- **Forecast Tracking**: Monitors and updates forecast accuracy

#### Market Data Integration
- **Multiple Sources**: Integrates with existing market data providers
- **Real-time Updates**: Continuously updates symbol data
- **Caching**: Efficient data caching and management
- **Error Handling**: Robust error handling and fallback mechanisms

#### Agent Integration
- **Agent Manager**: Uses existing agent manager for execution
- **Trigger System**: Flexible trigger configuration system
- **Execution Monitoring**: Tracks agent execution and performance
- **Result Processing**: Processes and stores agent results

## Configuration

### Basic Configuration
```python
from trading.live_market_runner import create_live_market_runner

config = {
    'symbols': ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'],
    'market_data_config': {
        'cache_size': 1000,
        'update_threshold': 5,
        'max_retries': 3
    },
    'triggers': {
        'model_builder': {
            'trigger_type': 'time_based',
            'interval_seconds': 3600,
            'enabled': True
        },
        'execution_agent': {
            'trigger_type': 'price_move',
            'price_move_threshold': 0.005,
            'enabled': True
        }
    }
}

runner = create_live_market_runner(config)
```

### Trigger Types

#### Time-based Triggers
```python
{
    'trigger_type': 'time_based',
    'interval_seconds': 3600,  # Every hour
    'enabled': True
}
```

#### Price Move Triggers
```python
{
    'trigger_type': 'price_move',
    'price_move_threshold': 0.01,  # 1% price move
    'enabled': True
}
```

#### Volume Spike Triggers
```python
{
    'trigger_type': 'volume_spike',
    'volume_spike_threshold': 2.0,  # 2x average volume
    'enabled': True
}
```

#### Volatility Triggers
```python
{
    'trigger_type': 'volatility_spike',
    'volatility_threshold': 0.02,  # 2% volatility
    'enabled': True
}
```

## Usage

### Basic Usage
```python
import asyncio
from trading.live_market_runner import create_live_market_runner

async def main():
    # Create runner
    runner = create_live_market_runner()
    
    # Start the runner
    await runner.start()
    
    # Keep running
    try:
        while runner.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await runner.stop()

asyncio.run(main())
```

### Advanced Usage
```python
import asyncio
from trading.live_market_runner import create_live_market_runner

async def main():
    # Custom configuration
    config = {
        'symbols': ['AAPL', 'TSLA', 'NVDA'],
        'update_interval': 30,  # 30 seconds
        'trigger_interval': 10,  # 10 seconds
        'triggers': {
            'model_builder': {
                'trigger_type': 'time_based',
                'interval_seconds': 1800,  # 30 minutes
                'enabled': True
            },
            'execution_agent': {
                'trigger_type': 'price_move',
                'price_move_threshold': 0.003,  # 0.3%
                'enabled': True
            }
        }
    }
    
    # Create and start runner
    runner = create_live_market_runner(config)
    await runner.start()
    
    # Monitor state
    while runner.running:
        state = runner.get_current_state()
        print(f"Symbols: {list(state['symbols'].keys())}")
        print(f"Forecasts: {state['forecast_count']}")
        await asyncio.sleep(60)

asyncio.run(main())
```

### State Monitoring
```python
# Get current state
state = runner.get_current_state()
print(f"Timestamp: {state['timestamp']}")
print(f"Running: {state['running']}")
print(f"Symbols: {list(state['symbols'].keys())}")

# Get symbol data
for symbol, data in state['symbols'].items():
    print(f"{symbol}: ${data['price']:.2f} ({data['price_change']:+.2%})")

# Get forecast accuracy
accuracy = runner.get_forecast_accuracy()
print(f"Average accuracy: {accuracy['avg_accuracy']:.2%}")
```

## Forecast Tracking

### Forecast Structure
```python
@dataclass
class ForecastResult:
    timestamp: datetime
    symbol: str
    forecast_price: float
    forecast_direction: str  # 'up', 'down', 'sideways'
    confidence: float
    model_name: str
    actual_price: Optional[float] = None
    accuracy: Optional[float] = None
```

### Accuracy Calculation
```python
# Get overall accuracy
accuracy = runner.get_forecast_accuracy()
print(f"Total forecasts: {accuracy['total_forecasts']}")
print(f"Completed: {accuracy['completed_forecasts']}")
print(f"Average accuracy: {accuracy['avg_accuracy']:.2%}")

# Get symbol-specific accuracy
aapl_accuracy = runner.get_forecast_accuracy("AAPL")
print(f"AAPL accuracy: {aapl_accuracy['avg_accuracy']:.2%}")
```

### Forecast Storage
- **File**: `trading/live/forecast_results.json`
- **Format**: JSON with timestamp and forecast data
- **Retention**: Last 1000 forecasts kept in memory
- **Backup**: Automatic saving every 10 forecasts

## Agent Integration

### Agent Execution
```python
# Agents are executed with live market data
market_data = {
    'live_data': runner.live_data,
    'price_history': dict(runner.price_history),
    'global_metrics': runner.global_metrics
}

result = await agent_manager.execute_agent(
    agent_name, 
    market_data=market_data,
    live_mode=True
)
```

### Trigger Management
```python
# Check if agent should be triggered
should_trigger = await runner._should_trigger_agent(
    agent_name, 
    trigger_config, 
    current_time
)

if should_trigger:
    await runner._execute_agent(agent_name)
```

## Performance Monitoring

### Execution Metrics
```python
# Monitor execution times
execution_times = runner.execution_times
for agent, times in execution_times.items():
    avg_time = np.mean(times) if times else 0
    print(f"{agent}: {avg_time:.2f}s average")

# Monitor error rates
error_counts = runner.error_counts
for agent, count in error_counts.items():
    print(f"{agent}: {count} errors")
```

### System Metrics
- **Uptime**: Continuous operation tracking
- **Data Quality**: Market data freshness and completeness
- **Agent Performance**: Success rates and execution times
- **Forecast Accuracy**: Historical accuracy tracking

## File Structure

```
trading/
├── live_market_runner.py          # Main implementation
├── demo_live_market_runner.py     # Demo script
├── test_live_market_runner.py     # Test script
├── launch_live_market_runner.py   # Launcher script
├── README_LIVE_MARKET_RUNNER.md   # This documentation
└── live/
    ├── config.json                # Configuration file
    ├── forecast_results.json      # Forecast storage
    ├── current_state.json         # Current state
    ├── performance_metrics.json   # Performance data
    └── logs/
        ├── live_market_runner.log # Main logs
        └── launcher.log           # Launcher logs
```

## Configuration File

### Example Configuration
```json
{
  "symbols": ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
  "update_interval": 30,
  "trigger_interval": 10,
  "tracking_interval": 300,
  "monitoring_interval": 600,
  "market_data_config": {
    "cache_size": 1000,
    "update_threshold": 5,
    "max_retries": 3
  },
  "triggers": {
    "model_builder": {
      "trigger_type": "time_based",
      "interval_seconds": 3600,
      "enabled": true
    },
    "performance_critic": {
      "trigger_type": "time_based",
      "interval_seconds": 1800,
      "enabled": true
    },
    "execution_agent": {
      "trigger_type": "price_move",
      "price_move_threshold": 0.005,
      "enabled": true
    }
  }
}
```

## Running the System

### Command Line
```bash
# Run the launcher
python trading/launch_live_market_runner.py

# Run the demo
python trading/demo_live_market_runner.py

# Run tests
python trading/test_live_market_runner.py
```

### Programmatic
```python
import asyncio
from trading.live_market_runner import create_live_market_runner

async def main():
    runner = create_live_market_runner()
    await runner.start()
    
    # Keep running
    while runner.running:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Monitoring and Logging

### Log Files
- **Main Log**: `trading/live/logs/live_market_runner.log`
- **Launcher Log**: `trading/live/logs/launcher.log`
- **Agent Logs**: Individual agent logs in their respective directories

### Log Levels
- **INFO**: Normal operation messages
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors and failures
- **DEBUG**: Detailed debugging information

### Key Metrics
- **Data Updates**: Frequency and success of market data updates
- **Agent Executions**: Success rates and execution times
- **Forecast Accuracy**: Historical accuracy tracking
- **System Health**: Uptime and error rates

## Error Handling

### Data Errors
- **Network Failures**: Automatic retry with exponential backoff
- **Data Validation**: Validation of received market data
- **Fallback Sources**: Multiple data source fallbacks
- **Cache Usage**: Use cached data when live data unavailable

### Agent Errors
- **Execution Failures**: Logged and tracked
- **Timeout Handling**: Configurable execution timeouts
- **Error Recovery**: Automatic retry mechanisms
- **Status Tracking**: Agent status monitoring

### System Errors
- **Graceful Shutdown**: Proper cleanup on shutdown
- **State Persistence**: Save state before shutdown
- **Error Logging**: Comprehensive error logging
- **Recovery Mechanisms**: Automatic recovery from errors

## Future Enhancements

### Planned Features
- **Real-time Data Sources**: Integration with real-time data feeds
- **Advanced Triggers**: Machine learning-based trigger optimization
- **Distributed Operation**: Multi-instance coordination
- **Web Dashboard**: Real-time monitoring dashboard
- **Alert System**: Configurable alerts and notifications

### Integration Opportunities
- **Trading APIs**: Direct integration with trading platforms
- **Risk Management**: Real-time risk monitoring
- **Portfolio Management**: Live portfolio tracking
- **Performance Analytics**: Advanced performance analysis

---

**🚀 Live Market Runner** - Your autonomous live market data streaming and agent triggering system. 