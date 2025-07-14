# Task Orchestrator Documentation

## Overview

The Task Orchestrator is a centralized task management and scheduling system for the Evolve trading platform. It coordinates the execution of all platform agents, manages dependencies, monitors performance, and ensures system reliability.

## Architecture

### Core Components

1. **TaskOrchestrator**: Main orchestrator class that manages all tasks and agents
2. **TaskConfig**: Configuration for individual tasks including scheduling and execution parameters
3. **TaskExecution**: Records of task execution including status, timing, and results
4. **AgentStatus**: Status tracking for each agent including health and performance metrics

### Key Features

- **Modular Design**: Each agent is independent and configurable
- **Dependency Management**: Tasks can depend on other tasks
- **Conditional Execution**: Tasks execute based on market conditions and system state
- **Performance Monitoring**: Real-time tracking of task performance and system health
- **Error Handling**: Robust error handling with retry mechanisms and failure thresholds
- **Configurable Scheduling**: Flexible scheduling with different intervals and priorities

## Configuration

### Task Schedule Configuration (`config/task_schedule.yaml`)

The orchestrator uses a YAML configuration file to define all tasks and their parameters:

```yaml
orchestrator:
  enabled: true
  max_concurrent_tasks: 5
  default_timeout_minutes: 15
  health_check_interval_minutes: 5
  performance_monitoring: true
  error_alerting: true

tasks:
  model_innovation:
    enabled: true
    interval_minutes: 1440  # 24 hours
    priority: "medium"
    max_duration_minutes: 60
    retry_count: 3
    retry_delay_minutes: 5
    dependencies: ["data_sync"]
    conditions:
      market_hours: false
      system_health: 0.8
    parameters:
      innovation_mode: "exploratory"
      model_types: ["lstm", "transformer"]
```

### Configuration Parameters

#### Orchestrator Level
- `enabled`: Enable/disable the orchestrator
- `max_concurrent_tasks`: Maximum number of tasks running simultaneously
- `default_timeout_minutes`: Default timeout for task execution
- `health_check_interval_minutes`: How often to check system health
- `performance_monitoring`: Enable performance monitoring
- `error_alerting`: Enable error alerting

#### Task Level
- `enabled`: Enable/disable the task
- `interval_minutes`: How often to execute the task
- `priority`: Task priority (critical, high, medium, low)
- `max_duration_minutes`: Maximum allowed execution time
- `retry_count`: Number of retry attempts on failure
- `retry_delay_minutes`: Delay between retry attempts
- `dependencies`: List of tasks that must complete first
- `conditions`: Conditions that must be met for execution
- `parameters`: Parameters passed to the task
- `timeout_minutes`: Task-specific timeout
- `concurrent_execution`: Allow concurrent execution
- `error_threshold`: Maximum errors before task is disabled
- `performance_threshold`: Minimum performance score required

## Supported Agents

The Task Orchestrator manages the following agents:

### 1. Model Innovation Agent
- **Purpose**: Continuously improve and innovate trading models
- **Frequency**: Daily (1440 minutes)
- **Dependencies**: Data sync, performance analysis
- **Conditions**: Off-market hours, good system health

### 2. Strategy Research Agent
- **Purpose**: Research and develop new trading strategies
- **Frequency**: Every 12 hours (720 minutes)
- **Dependencies**: Sentiment fetch, data sync
- **Conditions**: Normal market volatility, no major news events

### 3. Sentiment Fetcher
- **Purpose**: Collect and analyze market sentiment
- **Frequency**: Every 30 minutes
- **Dependencies**: None
- **Conditions**: Market hours, reasonable system health

### 4. Meta Controller
- **Purpose**: System-wide coordination and control
- **Frequency**: Every 5 minutes
- **Dependencies**: None
- **Conditions**: Good system health

### 5. Risk Manager
- **Purpose**: Monitor and manage portfolio risk
- **Frequency**: Every 15 minutes
- **Dependencies**: Meta control
- **Conditions**: Market hours, active positions

### 6. Execution Agent
- **Purpose**: Execute trading orders
- **Frequency**: Every minute
- **Dependencies**: Risk management
- **Conditions**: Market hours, pending orders

### 7. Explainer Agent
- **Purpose**: Generate explanations for decisions
- **Frequency**: Hourly
- **Dependencies**: Execution
- **Conditions**: New trades, reasonable system health

### 8. System Health Monitor
- **Purpose**: Monitor system health and performance
- **Frequency**: Every 5 minutes
- **Dependencies**: None
- **Conditions**: None

### 9. Data Synchronization
- **Purpose**: Sync market data and system data
- **Frequency**: Every 10 minutes
- **Dependencies**: None
- **Conditions**: Market hours

### 10. Performance Analysis
- **Purpose**: Analyze trading performance
- **Frequency**: Every 2 hours
- **Dependencies**: Execution
- **Conditions**: Trading activity

## Usage Examples

### Basic Usage

```python
from core.task_orchestrator import start_orchestrator

async def main():
    # Start orchestrator with default configuration
    orchestrator = await start_orchestrator()
    
    try:
        # Keep running
        await asyncio.sleep(3600)  # Run for 1 hour
    finally:
        await orchestrator.stop()

asyncio.run(main())
```

### Custom Configuration

```python
from core.task_orchestrator import create_task_orchestrator

# Create orchestrator with custom config
orchestrator = create_task_orchestrator("config/custom_schedule.yaml")

# Start orchestrator
await orchestrator.start()

# Execute specific task
await orchestrator.execute_task_now("model_innovation", {
    "innovation_mode": "aggressive",
    "model_types": ["transformer"]
})

# Get system status
status = orchestrator.get_system_status()
print(f"System health: {status['performance_metrics']['overall_health']}")

# Stop orchestrator
await orchestrator.stop()
```

### Task Management

```python
# Get task status
task_status = orchestrator.get_task_status("model_innovation")
print(f"Task enabled: {task_status['enabled']}")
print(f"Success rate: {task_status['success_rate']}")

# Update task configuration
orchestrator.update_task_config("model_innovation", {
    "interval_minutes": 720,  # Change to 12 hours
    "priority": "high"
})

# Export status report
report_path = orchestrator.export_status_report()
print(f"Report exported to: {report_path}")
```

## Execution Conditions

The orchestrator supports various execution conditions:

### Market Conditions
- `market_hours`: Check if currently market hours
- `market_volatility`: Check market volatility level
- `news_events`: Check for significant news events

### System Conditions
- `system_health`: Check overall system health score
- `position_count`: Check number of active positions
- `pending_orders`: Check for pending orders
- `new_trades`: Check for recent trades
- `trading_activity`: Check for trading activity

### Custom Conditions
You can add custom conditions by implementing methods in the TaskOrchestrator class:

```python
async def _check_custom_condition(self, condition: str, value: Any) -> bool:
    if condition == 'custom_condition':
        return await self._evaluate_custom_condition(value)
    return True
```

## Performance Monitoring

### Metrics Tracked
- **Success Rates**: Percentage of successful task executions
- **Error Counts**: Number of consecutive failures
- **Execution Duration**: Average execution time per task
- **System Health**: Overall system health score
- **Agent Status**: Individual agent health and performance

### Performance Thresholds
Tasks can be configured with performance thresholds:
- `performance_threshold`: Minimum success rate required
- `error_threshold`: Maximum consecutive errors allowed

### Health Scoring
System health is calculated as the average health score of all agents:
- Health score = Success count / (Success count + Failure count)
- Range: 0.0 (unhealthy) to 1.0 (healthy)

## Error Handling

### Retry Mechanism
- **Retry Count**: Number of retry attempts on failure
- **Retry Delay**: Time between retry attempts
- **Exponential Backoff**: Increasing delays for repeated failures

### Failure Thresholds
- **Error Threshold**: Maximum errors before task is disabled
- **Performance Threshold**: Minimum performance required
- **Timeout Handling**: Tasks are killed if they exceed timeout

### Error Recovery
- **Automatic Restart**: Failed agents can be automatically restarted
- **Graceful Degradation**: System continues operating with reduced functionality
- **Alert System**: Notifications for critical failures

## Best Practices

### 1. Task Design
- Keep tasks focused and single-purpose
- Use appropriate timeouts and retry counts
- Implement proper error handling in task methods
- Use dependencies to ensure proper execution order

### 2. Configuration
- Start with conservative intervals and adjust based on performance
- Monitor system resources and adjust concurrent task limits
- Use conditions to avoid unnecessary task execution
- Set appropriate performance thresholds

### 3. Monitoring
- Regularly check system status and performance metrics
- Monitor error rates and adjust thresholds as needed
- Export status reports for analysis
- Set up alerts for critical failures

### 4. Performance Optimization
- Use concurrent execution for independent tasks
- Optimize task execution time
- Implement caching where appropriate
- Monitor resource usage

### 5. Security
- Validate all configuration parameters
- Implement proper access controls
- Log all system activities
- Regular security audits

## Troubleshooting

### Common Issues

#### 1. Tasks Not Executing
- Check if task is enabled in configuration
- Verify dependencies are satisfied
- Check execution conditions
- Review error logs

#### 2. High Error Rates
- Increase retry counts
- Extend timeouts
- Check system resources
- Review task implementation

#### 3. Performance Degradation
- Reduce concurrent task limits
- Optimize task execution
- Check system resources
- Review dependencies

#### 4. System Health Issues
- Check individual agent health scores
- Review error history
- Monitor resource usage
- Restart unhealthy agents

### Debugging

#### Enable Debug Logging
```python
import logging
logging.getLogger('core.task_orchestrator').setLevel(logging.DEBUG)
```

#### Check Task Status
```python
status = orchestrator.get_task_status("task_name")
print(json.dumps(status, indent=2))
```

#### Monitor Execution
```python
# Get recent executions
executions = orchestrator.task_executions
for execution_id, execution in list(executions.items())[-10:]:
    print(f"{execution.task_name}: {execution.status} - {execution.error_message}")
```

## Integration

### With Existing Agents
The orchestrator integrates with existing Evolve platform agents:

```python
# Agents are automatically discovered and initialized
from agents.model_innovation_agent import ModelInnovationAgent
from trading.risk.risk_manager import RiskManager
# ... other imports

# Orchestrator will automatically find and manage these agents
```

### With External Systems
- **Monitoring**: Export metrics to external monitoring systems
- **Alerting**: Integrate with email, Slack, or webhook systems
- **Logging**: Centralized logging with external log aggregation
- **Metrics**: Export performance metrics to time-series databases

### API Integration
The orchestrator provides a clean API for external integration:

```python
# Get system status
status = orchestrator.get_system_status()

# Execute tasks programmatically
await orchestrator.execute_task_now("task_name", parameters)

# Update configurations
orchestrator.update_task_config("task_name", updates)
```

## Future Enhancements

### Planned Features
- **Dynamic Scheduling**: AI-powered task scheduling optimization
- **Resource Management**: Advanced resource allocation and management
- **Distributed Execution**: Support for distributed task execution
- **Advanced Analytics**: Deep performance analytics and insights
- **Machine Learning**: ML-powered failure prediction and prevention

### Extension Points
- **Custom Conditions**: Framework for custom execution conditions
- **Plugin System**: Plugin architecture for custom agents
- **API Gateway**: REST API for external integration
- **Web Dashboard**: Web-based monitoring and control interface

## Conclusion

The Task Orchestrator provides a robust, scalable foundation for managing the Evolve trading platform's complex agent ecosystem. With its modular design, comprehensive monitoring, and flexible configuration, it ensures reliable operation while maintaining high performance and system health.

For more information, see the API documentation and example implementations in the `examples/` directory. 