# Pluggable Agents System

This directory contains the new pluggable agents system that allows for dynamic enable/disable functionality, agent registration, and easy swapping of agent implementations.

## Overview

The pluggable agents system provides:

- **Standalone Components**: Each agent is a self-contained, pluggable component
- **Dynamic Configuration**: Enable/disable agents via configuration files
- **Easy Swapping**: Replace or upgrade agents without system changes
- **Clean Interfaces**: Standardized `.run()` and `.execute()` methods
- **Error Handling**: Comprehensive error handling and status tracking
- **Metrics**: Built-in execution metrics and monitoring
- **Performance Tracking**: Agent leaderboard with automatic deprecation
- **Visualization**: Interactive dashboard for performance analysis

## Architecture

### Base Agent Interface

All agents implement the `BaseAgent` interface defined in `base_agent_interface.py`:

```python
class BaseAgent(ABC):
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the agent's main logic."""
        pass
    
    def enable(self) -> None:
        """Enable the agent."""
        pass
    
    def disable(self) -> None:
        """Disable the agent."""
        pass
    
    def get_status(self) -> AgentStatus:
        """Get the current status of the agent."""
        pass
```

### Agent Manager

The `AgentManager` class provides centralized management of all agents:

- Agent registration and discovery
- Dynamic enable/disable functionality
- Configuration management
- Execution coordination
- Metrics collection
- Performance tracking and leaderboard management

### Agent Leaderboard

The `AgentLeaderboard` class tracks agent performance and provides:

- Performance metrics tracking (Sharpe ratio, drawdown, win rate, etc.)
- Automatic deprecation of underperforming agents
- Leaderboard ranking and sorting
- Performance history tracking
- Export capabilities for dashboards and reports

## Available Agents

### 1. ModelBuilderAgent

**Purpose**: Builds and initializes various ML models from scratch

**Capabilities**:
- LSTM model building
- XGBoost model building
- Ensemble model building
- Hyperparameter tuning

**Usage**:
```python
from trading.agents.model_builder_agent import ModelBuildRequest

request = ModelBuildRequest(
    model_type="lstm",
    data_path="data/sample_data.csv",
    target_column="close",
    hyperparameters={"epochs": 100, "batch_size": 32}
)

result = await execute_agent("model_builder", request=request)
```

### 2. PerformanceCriticAgent

**Purpose**: Evaluates model performance based on financial metrics

**Capabilities**:
- Performance metrics calculation (Sharpe ratio, drawdown, etc.)
- Risk assessment
- Benchmark comparison
- Trading metrics analysis

**Usage**:
```python
from trading.agents.performance_critic_agent import ModelEvaluationRequest

request = ModelEvaluationRequest(
    model_id="model_123",
    model_path="models/model_123.pkl",
    model_type="lstm",
    test_data_path="data/test_data.csv"
)

result = await execute_agent("performance_critic", request=request)
```

### 3. UpdaterAgent

**Purpose**: Updates models based on performance feedback

**Capabilities**:
- Model retraining
- Hyperparameter tuning
- Model replacement
- Ensemble weight adjustment

**Usage**:
```python
# The updater agent processes evaluation results automatically
result = await execute_agent("updater", evaluation_result=eval_result)
```

## Agent Leaderboard System

### Overview

The Agent Leaderboard system automatically tracks agent performance and provides insights into which agents are performing best. It includes:

- **Performance Tracking**: Monitor Sharpe ratio, drawdown, win rate, and total return
- **Automatic Deprecation**: Automatically deprecate underperforming agents
- **Leaderboard Ranking**: Sort agents by various performance metrics
- **Interactive Dashboard**: Visualize performance with charts and filters
- **Export Capabilities**: Export data for external analysis

### Performance Metrics

The system tracks the following metrics for each agent:

- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall return percentage
- **Extra Metrics**: Custom metrics like volatility, Calmar ratio, profit factor

### Deprecation Thresholds

Agents are automatically deprecated when they fall below these thresholds:

```python
default_thresholds = {
    'sharpe_ratio': 0.5,    # Sharpe ratio below 0.5
    'max_drawdown': 0.25,   # Drawdown above 25%
    'win_rate': 0.45        # Win rate below 45%
}
```

### Basic Usage

```python
from trading.agents.agent_manager import get_agent_manager

# Get the agent manager (includes leaderboard)
manager = get_agent_manager()

# Log agent performance (automatically updates leaderboard)
manager.log_agent_performance(
    agent_name="model_builder_v1",
    sharpe_ratio=1.8,
    max_drawdown=0.12,
    win_rate=0.68,
    total_return=0.35,
    extra_metrics={
        "volatility": 0.18,
        "calmar_ratio": 2.5,
        "profit_factor": 2.1
    }
)

# Get leaderboard data
top_agents = manager.get_leaderboard(top_n=10, sort_by="sharpe_ratio")
print("Top 10 agents by Sharpe ratio:")
for i, agent in enumerate(top_agents, 1):
    print(f"{i}. {agent['agent_name']}: Sharpe={agent['sharpe_ratio']:.2f}")

# Get deprecated agents
deprecated = manager.get_deprecated_agents()
print(f"Deprecated agents: {deprecated}")

# Get active agents
active = manager.get_active_agents()
print(f"Active agents: {active}")
```

### Advanced Usage

```python
# Custom deprecation thresholds
from trading.agents.agent_leaderboard import AgentLeaderboard

custom_thresholds = {
    'sharpe_ratio': 1.0,    # Higher threshold
    'max_drawdown': 0.20,   # Lower drawdown tolerance
    'win_rate': 0.50        # Higher win rate requirement
}

leaderboard = AgentLeaderboard(deprecation_thresholds=custom_thresholds)

# Update performance with custom thresholds
leaderboard.update_performance(
    agent_name="conservative_agent",
    sharpe_ratio=0.8,  # Below custom threshold
    max_drawdown=0.15,
    win_rate=0.55,
    total_return=0.25
)

# Export to DataFrame for analysis
df = leaderboard.as_dataframe()
print(df.head())

# Get performance history
history = leaderboard.get_history(limit=50)
print(f"Recent performance updates: {len(history)}")
```

### Dashboard Usage

Launch the interactive dashboard:

```bash
# Basic launch
python trading/agents/launch_leaderboard_dashboard.py

# Custom port
python trading/agents/launch_leaderboard_dashboard.py --port 8502

# Remote access
python trading/agents/launch_leaderboard_dashboard.py --host 0.0.0.0 --port 8503

# Headless mode
python trading/agents/launch_leaderboard_dashboard.py --headless

# Check dependencies
python trading/agents/launch_leaderboard_dashboard.py --check-deps
```

The dashboard provides:

- **Interactive Leaderboard**: Sort and filter agents by performance
- **Performance Charts**: Visualize Sharpe ratios, returns, and risk metrics
- **Status Management**: View active vs deprecated agents
- **Deprecation Controls**: Manually deprecate or reactivate agents
- **Performance History**: Track performance over time
- **Export Options**: Download data as CSV or JSON
- **Summary Reports**: Generate performance summaries

### Integration with Reports

The leaderboard data can be integrated into the reporting system:

```python
from trading.report.report_generator import ReportGenerator

# Create report with leaderboard data
report_gen = ReportGenerator()

# Add leaderboard section to report
leaderboard_data = manager.get_leaderboard(top_n=5)
report_gen.add_section("Agent Performance", {
    "top_performers": leaderboard_data,
    "deprecated_agents": manager.get_deprecated_agents(),
    "performance_summary": {
        "total_agents": len(manager.leaderboard.leaderboard),
        "active_agents": len(manager.get_active_agents()),
        "avg_sharpe": sum(a['sharpe_ratio'] for a in leaderboard_data) / len(leaderboard_data)
    }
})

# Generate report
report_gen.generate_report("agent_performance_report.md")
```

## Configuration

### Agent Configuration File

The system uses `agent_config.json` for configuration:

```json
{
  "agents": {
    "model_builder": {
      "enabled": true,
      "priority": 1,
      "max_concurrent_runs": 2,
      "timeout_seconds": 600,
      "retry_attempts": 3,
      "custom_config": {
        "max_models": 10,
        "model_types": ["lstm", "xgboost", "ensemble"]
      }
    }
  },
  "manager": {
    "auto_start": true,
    "max_concurrent_agents": 5,
    "execution_timeout": 300,
    "enable_logging": true,
    "enable_metrics": true
  }
}
```

### Configuration Options

#### Agent-Level Configuration

- `enabled`: Whether the agent is enabled
- `priority`: Execution priority (lower numbers = higher priority)
- `max_concurrent_runs`: Maximum concurrent executions
- `timeout_seconds`: Execution timeout
- `retry_attempts`: Number of retry attempts on failure
- `custom_config`: Agent-specific configuration

#### Manager-Level Configuration

- `auto_start`: Automatically start enabled agents
- `max_concurrent_agents`: Maximum concurrent agents
- `execution_timeout`: Default execution timeout
- `enable_logging`: Enable detailed logging
- `enable_metrics`: Enable metrics collection

## Usage Examples

### Basic Usage

```python
from trading.agents.agent_manager import get_agent_manager, execute_agent

# Get the agent manager
manager = get_agent_manager()

# List all agents
agents = manager.list_agents()
print(f"Available agents: {[agent['name'] for agent in agents]}")

# Enable/disable agents
manager.enable_agent("model_builder")
manager.disable_agent("performance_critic")

# Execute an agent
result = await execute_agent("model_builder", request=build_request)
```

### Custom Agent Registration

```python
from trading.agents.base_agent_interface import BaseAgent, AgentConfig
from trading.agents.agent_manager import register_agent

class CustomAgent(BaseAgent):
    async def execute(self, **kwargs) -> AgentResult:
        # Custom logic here
        return AgentResult(success=True, data={"result": "custom"})

# Register custom agent
config = AgentConfig(
    name="custom_agent",
    enabled=True,
    priority=1,
    custom_config={"param": "value"}
)

register_agent("custom_agent", CustomAgent, config)
```

### Agent Workflow with Performance Tracking

```python
# Complete workflow example with performance tracking
async def run_agent_workflow():
    # 1. Build model
    build_result = await execute_agent("model_builder", request=build_request)
    
    # 2. Evaluate model
    eval_result = await execute_agent("performance_critic", request=eval_request)
    
    # 3. Log performance to leaderboard
    manager = get_agent_manager()
    manager.log_agent_performance(
        agent_name="model_builder_v1",
        sharpe_ratio=eval_result.data["sharpe_ratio"],
        max_drawdown=eval_result.data["max_drawdown"],
        win_rate=eval_result.data["win_rate"],
        total_return=eval_result.data["total_return"]
    )
    
    # 4. Update if needed
    update_result = await execute_agent("updater", evaluation_result=eval_result.data)
    
    return update_result
```

### Configuration Management

```python
# Update agent configuration
manager = get_agent_manager()
manager.update_agent_config("model_builder", {
    "max_models": 15,
    "model_types": ["lstm", "xgboost", "ensemble", "transformer"]
})

# Save configuration
manager.save_config()

# Get metrics
metrics = manager.get_execution_metrics()
print(f"Total executions: {metrics['total_executions']}")

# Get leaderboard data
leaderboard_data = manager.get_leaderboard(top_n=5)
print("Top 5 performing agents:")
for agent in leaderboard_data:
    print(f"- {agent['agent_name']}: Sharpe={agent['sharpe_ratio']:.2f}")
```

## Agent Status and Monitoring

### Status Information

Each agent provides status information:

```python
status = manager.get_agent_status("model_builder")
print(f"Enabled: {status.enabled}")
print(f"Running: {status.is_running}")
print(f"Total runs: {status.total_runs}")
print(f"Successful runs: {status.successful_runs}")
print(f"Failed runs: {status.failed_runs}")
```

### Execution Metrics

The system tracks execution metrics:

```python
metrics = manager.get_execution_metrics()
for agent_name, agent_metrics in metrics["agent_metrics"].items():
    print(f"{agent_name}:")
    print(f"  Total executions: {agent_metrics['total_executions']}")
    print(f"  Success rate: {agent_metrics['successful_executions'] / agent_metrics['total_executions']:.2%}")
    print(f"  Avg execution time: {agent_metrics['avg_execution_time']:.2f}s")
```

### Performance Monitoring

Monitor agent performance through the leaderboard:

```python
# Get performance summary
leaderboard_data = manager.get_leaderboard()
active_agents = manager.get_active_agents()
deprecated_agents = manager.get_deprecated_agents()

print(f"Performance Summary:")
print(f"  Total Agents: {len(leaderboard_data)}")
print(f"  Active Agents: {len(active_agents)}")
print(f"  Deprecated Agents: {len(deprecated_agents)}")

# Check for underperforming agents
for agent in leaderboard_data:
    if agent['sharpe_ratio'] < 0.5:
        print(f"⚠️  {agent['agent_name']} has low Sharpe ratio: {agent['sharpe_ratio']:.2f}")
```

## Error Handling

The system provides comprehensive error handling:

```python
result = await execute_agent("model_builder", request=request)

if result.success:
    print(f"Success: {result.data}")
else:
    print(f"Error: {result.error_message}")
    print(f"Execution time: {result.execution_time}s")
```

## Best Practices

### 1. Agent Design

- Implement the `BaseAgent` interface
- Provide clear metadata (version, description, capabilities)
- Use proper error handling and logging
- Validate input parameters

### 2. Configuration

- Use configuration files for agent settings
- Provide sensible defaults
- Document custom configuration options
- Use environment-specific configurations

### 3. Error Handling

- Always check execution results
- Implement proper retry logic
- Log errors with context
- Provide meaningful error messages

### 4. Monitoring

- Monitor agent status regularly
- Track execution metrics
- Set up alerts for failures
- Review performance periodically

### 5. Performance Tracking

- Log performance metrics after each evaluation
- Set appropriate deprecation thresholds
- Monitor leaderboard regularly
- Use the dashboard for visual analysis
- Export data for external analysis

## Migration from Old System

If you're migrating from the old agent system:

1. **Update imports**: Use the new agent manager
2. **Update configuration**: Use the new config format
3. **Update execution**: Use `execute_agent()` function
4. **Add performance tracking**: Log performance to leaderboard
5. **Test thoroughly**: Verify all functionality works

## Troubleshooting

### Common Issues

1. **Agent not found**: Check if agent is registered
2. **Agent disabled**: Enable the agent via configuration
3. **Execution timeout**: Increase timeout in configuration
4. **Configuration errors**: Validate JSON syntax
5. **Dashboard not loading**: Check dependencies with `--check-deps`

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

- Check agent status: `manager.get_agent_status(agent_name)`
- Review logs: Check log files for detailed information
- Test individual agents: Use the demo script
- Validate configuration: Check JSON syntax and required fields
- View performance: Use the leaderboard dashboard

## Demo and Testing

### Run Demo

```bash
# Run the leaderboard demo
python trading/agents/demo_leaderboard.py

# Run agent tests
pytest tests/test_agents/test_agent_leaderboard.py -v

# Run all agent tests
pytest tests/test_agents/ -v
```

### Test Coverage

The test suite covers:

- Agent performance tracking
- Deprecation logic
- Leaderboard functionality
- Edge cases and error handling
- Integration scenarios
- Concurrent updates

## Future Enhancements

Planned improvements:

- **Plugin System**: Load agents from external plugins
- **Distributed Execution**: Run agents across multiple nodes
- **Advanced Scheduling**: Sophisticated execution scheduling
- **Web Interface**: Web-based agent management
- **API Integration**: REST API for agent management
- **Advanced Analytics**: Machine learning for performance prediction
- **Automated Optimization**: Auto-tuning of deprecation thresholds
- **Real-time Alerts**: Notifications for performance issues 