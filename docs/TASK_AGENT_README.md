# TaskAgent Documentation

## Overview

The `TaskAgent` is a sophisticated autonomous agent that implements recursive task execution with performance monitoring and automatic retry logic. It can handle various task types and will recursively attempt to improve performance until success is achieved or maximum depth is reached.

## Key Features

- **Recursive Task Execution**: Automatically retries with different approaches until performance threshold is met
- **Performance Monitoring**: Tracks performance metrics and makes intelligent decisions about next actions
- **Memory Logging**: Logs all task results to memory using `memory/prompt_log.py`
- **Builder-Evaluator-Updater Chain**: For forecast, strategy, and backtest tasks, automatically calls:
  1. **Builder** (`ModelBuilderAgent`) - builds models
  2. **Evaluator** (`PerformanceCriticAgent`) - evaluates performance
  3. **Updater** (`UpdaterAgent`) - updates models if needed
- **Task Type Support**: Supports forecast, strategy, backtest, and other task types
- **Configurable Thresholds**: Performance thresholds can be set per task (e.g., Sharpe < 1)

## Task Types

### Primary Task Types

1. **FORECAST** - Forecasting tasks (e.g., stock price prediction)
2. **STRATEGY** - Trading strategy development
3. **BACKTEST** - Strategy backtesting and validation

### Additional Task Types

4. **MODEL_BUILD** - Model building and initialization
5. **MODEL_EVALUATE** - Model evaluation and analysis
6. **MODEL_UPDATE** - Model updates and optimization
7. **STRATEGY_OPTIMIZE** - Strategy optimization
8. **DATA_ANALYSIS** - Data analysis tasks
9. **FORECAST_GENERATE** - Forecast generation
10. **TRADE_EXECUTE** - Trade execution
11. **RISK_ASSESS** - Risk assessment
12. **GENERAL** - General tasks

## Architecture

### Action Types

The TaskAgent uses a strategy pattern with the following action types:

- **BUILDER** - Builds models using ModelBuilderAgent
- **EVALUATOR** - Evaluates models using PerformanceCriticAgent
- **UPDATER** - Updates models using UpdaterAgent
- **RUN_MODEL** - Runs existing models
- **SCORE_PERFORMANCE** - Scores performance metrics
- **UPDATE_PARAMETERS** - Updates model parameters
- **RETRY_WITH_DIFFERENT_APPROACH** - Retries with different approach
- **FALLBACK_TO_BASELINE** - Falls back to baseline model
- **STOP_AND_REPORT** - Stops execution and reports results

### Execution Flow

1. **Task Initialization**: Creates task context and maps parameters
2. **Action Selection**: Determines initial action based on task type
3. **Action Execution**: Executes the selected action strategy
4. **Performance Evaluation**: Evaluates performance against threshold
5. **Decision Making**: Decides next action based on performance
6. **Recursive Loop**: Continues until success or max depth reached
7. **Memory Logging**: Logs all results to prompt memory

## Usage

### Basic Usage

```python
from agents.task_agent import execute_forecast_task, execute_strategy_task, execute_backtest_task

# Execute a forecast task
result = await execute_forecast_task(
    prompt="Build a forecasting model for AAPL stock price prediction",
    parameters={
        "symbol": "AAPL",
        "data_path": "data/aapl_data.csv",
        "hyperparameters": {"epochs": 100, "batch_size": 32}
    },
    max_depth=3,
    performance_threshold=0.7  # Sharpe ratio threshold
)

print(f"Success: {result.success}")
print(f"Performance Score: {result.performance_score}")
print(f"Message: {result.message}")
```

### Advanced Usage

```python
from agents.task_agent import TaskAgent, TaskType

agent = TaskAgent()

# Execute a custom task
result = await agent.execute_task(
    prompt="Build and evaluate a custom ensemble model",
    task_type=TaskType.MODEL_BUILD,
    parameters={
        "model_type": "ensemble",
        "data_path": "data/custom_data.csv",
        "target_column": "target",
        "hyperparameters": {"n_estimators": 100}
    },
    max_depth=5,
    performance_threshold=0.8,
    parent_task_id="parent_123"
)

# Get task history
task_history = agent.get_task_history(result.task_id)
all_tasks = agent.get_all_tasks()
```

## Parameter Mapping

The TaskAgent automatically maps parameters based on task type:

### Forecast Tasks
```python
{
    "model_type": "lstm",
    "data_path": "data/forecast_data.csv",
    "target_column": "close",
    "hyperparameters": {
        "epochs": 100,
        "batch_size": 32,
        "lookback_window": 60
    },
    "test_data_path": "data/forecast_test.csv"
}
```

### Strategy Tasks
```python
{
    "model_type": "ensemble",
    "data_path": "data/strategy_data.csv",
    "target_column": "returns",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1
    },
    "test_data_path": "data/strategy_test.csv"
}
```

### Backtest Tasks
```python
{
    "model_type": "xgboost",
    "data_path": "data/backtest_data.csv",
    "target_column": "signal",
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 8,
        "subsample": 0.8
    },
    "test_data_path": "data/backtest_test.csv"
}
```

## Performance Thresholds

The TaskAgent uses performance thresholds to determine when to stop retrying:

- **Sharpe Ratio**: Normalized to 0-1 range, where 1.0 = excellent performance
- **Total Return**: Annualized return percentage
- **Max Drawdown**: Maximum drawdown percentage
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Default Thresholds

- **Forecast Tasks**: 0.7 (Sharpe ratio)
- **Strategy Tasks**: 0.8 (Sharpe ratio)
- **Backtest Tasks**: 0.6 (Sharpe ratio)
- **Model Build Tasks**: 0.75 (Sharpe ratio)

## Memory Integration

The TaskAgent integrates with the memory system to log all task interactions:

### Prompt Memory
- Logs all prompts and results using `memory/prompt_log.py`
- Stores execution time, success status, and metadata
- Maintains history for analysis and debugging

### Agent Memory
- Tracks agent actions and decisions
- Stores performance history and improvement metrics
- Enables learning from past experiences

## Error Handling

The TaskAgent includes comprehensive error handling:

1. **Action Failures**: Automatically retries with different approaches
2. **Model Failures**: Falls back to baseline models
3. **Data Issues**: Handles missing or corrupted data gracefully
4. **Timeout Handling**: Respects execution timeouts
5. **Recovery Logic**: Attempts to recover from various failure modes

## Configuration

The TaskAgent can be configured with various parameters:

```python
config = {
    "max_depth": 5,
    "performance_threshold": 0.7,
    "timeout_seconds": 300,
    "retry_attempts": 3,
    "memory_backend": "json",  # or "redis"
    "log_level": "INFO"
}

agent = TaskAgent(config=config)
```

## Examples

See `examples/task_agent_example.py` for comprehensive usage examples including:

- Forecast task execution
- Strategy task execution
- Backtest task execution
- Custom task execution
- Task history retrieval
- Performance analysis

## Integration with Other Components

The TaskAgent integrates with several other system components:

### ModelBuilderAgent
- Builds LSTM, XGBoost, and ensemble models
- Handles hyperparameter tuning
- Manages model storage and registration

### PerformanceCriticAgent
- Evaluates model performance using financial metrics
- Calculates Sharpe ratio, drawdown, win rate, etc.
- Provides performance recommendations

### UpdaterAgent
- Updates models based on performance feedback
- Handles model retraining and tuning
- Manages model replacement and optimization

### Memory System
- `memory/prompt_log.py` for prompt history
- `trading/memory/agent_memory.py` for agent memory
- `trading/memory/performance_memory.py` for performance tracking

## Best Practices

1. **Set Appropriate Thresholds**: Choose performance thresholds based on task requirements
2. **Monitor Task History**: Regularly review task history for insights
3. **Use Meaningful Prompts**: Write clear, specific prompts for better results
4. **Configure Parameters**: Provide relevant parameters for each task type
5. **Handle Results**: Always check success status and handle errors appropriately
6. **Log Important Events**: Use the logging system for debugging and monitoring

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required agents are properly installed
2. **Data Path Issues**: Verify data file paths exist and are accessible
3. **Memory Issues**: Check available memory for large model training
4. **Timeout Issues**: Increase timeout for complex tasks
5. **Performance Issues**: Adjust thresholds or model parameters

### Debugging

1. **Enable Debug Logging**: Set log level to DEBUG for detailed output
2. **Check Task History**: Review task history for failure patterns
3. **Monitor Memory Usage**: Check memory consumption during execution
4. **Validate Parameters**: Ensure all required parameters are provided
5. **Test with Simple Tasks**: Start with simple tasks to verify setup

## Future Enhancements

Planned improvements include:

1. **Multi-Agent Coordination**: Better coordination between multiple agents
2. **Advanced Performance Metrics**: More sophisticated performance evaluation
3. **Automated Hyperparameter Optimization**: Automatic parameter tuning
4. **Real-time Monitoring**: Live performance monitoring and alerts
5. **Distributed Execution**: Support for distributed task execution
6. **Custom Action Strategies**: User-defined action strategies
7. **Performance Prediction**: Predict task success probability
8. **Resource Optimization**: Better resource allocation and management 