# Task Orchestrator Integration Guide

## Overview

This guide explains how to integrate the Task Orchestrator with your existing Evolve trading platform. The Task Orchestrator provides centralized task management, scheduling, and monitoring for all platform agents.

## Quick Start

### 1. Check System Requirements

```bash
python start_orchestrator.py --check
```

This will verify that all required components are available.

### 2. Start Standalone Orchestrator

```bash
python start_orchestrator.py --standalone
```

This starts the Task Orchestrator in standalone mode, managing only the orchestrator components.

### 3. Start Integrated System

```bash
python start_orchestrator.py --integrated
```

This starts the Task Orchestrator integrated with your existing Evolve platform components.

### 4. Monitor System

```bash
python start_orchestrator.py --monitor --duration 30
```

This starts the orchestrator with real-time monitoring for 30 minutes.

## Integration Methods

### Method 1: Direct Integration (Recommended)

Use the system integration module to connect with existing components:

```python
from system.orchestrator_integration import start_integrated_system

# Start integrated system
integration = await start_integrated_system()

# Get system status
status = integration.get_system_status()
print(f"System health: {status['orchestrator_status']['performance_metrics']['overall_health']}")

# Execute system commands
result = await integration.execute_system_command("forecast AAPL 7d")
```

### Method 2: Manual Integration

Manually connect orchestrator with existing agents:

```python
from core.task_orchestrator import TaskOrchestrator
from agents.model_innovation_agent import ModelInnovationAgent
from trading.risk.risk_manager import RiskManager

# Create orchestrator
orchestrator = TaskOrchestrator("config/task_schedule.yaml")

# Connect existing agents
orchestrator.agents['model_innovation'] = ModelInnovationAgent()
orchestrator.agents['risk_management'] = RiskManager()

# Start orchestrator
await orchestrator.start()
```

### Method 3: Integration Script

Use the provided integration script:

```bash
python scripts/integrate_orchestrator.py --start --monitor
```

## Configuration

### Task Schedule Configuration

The orchestrator uses `config/task_schedule.yaml` for configuration:

```yaml
orchestrator:
  enabled: true
  max_concurrent_tasks: 5
  default_timeout_minutes: 15
  health_check_interval_minutes: 5

tasks:
  model_innovation:
    enabled: true
    interval_minutes: 1440  # Daily
    priority: "medium"
    dependencies: ["data_sync"]
    conditions:
      market_hours: false
      system_health: 0.8
```

### Environment-Specific Configuration

Different environments can have different configurations:

```yaml
environments:
  development:
    max_concurrent_tasks: 3
    error_threshold_multiplier: 2
    
  production:
    max_concurrent_tasks: 5
    error_threshold_multiplier: 1
```

## Agent Integration

### Supported Agents

The orchestrator automatically discovers and integrates with these agents:

1. **ModelInnovationAgent** (`agents/model_innovation_agent.py`)
   - Method: `innovate_models()`
   - Schedule: Daily
   - Purpose: Model improvement and innovation

2. **StrategyResearchAgent** (`agents/strategy_research_agent.py`)
   - Method: `research_strategies()`
   - Schedule: Every 12 hours
   - Purpose: Strategy research and development

3. **SentimentAnalyzer** (`trading/nlp/sentiment_analyzer.py`)
   - Method: `fetch_sentiment()`
   - Schedule: Every 30 minutes
   - Purpose: Market sentiment analysis

4. **RiskManager** (`trading/risk/risk_manager.py`)
   - Method: `manage_risk()`
   - Schedule: Every 15 minutes
   - Purpose: Risk monitoring and management

5. **ExecutionAgent** (`execution/execution_agent.py`)
   - Method: `execute_orders()`
   - Schedule: Every minute
   - Purpose: Order execution

6. **ExplainerAgent** (`reporting/explainer_agent.py`)
   - Method: `generate_explanations()`
   - Schedule: Hourly
   - Purpose: Decision explanations

### Custom Agent Integration

To integrate a custom agent:

```python
class CustomAgent:
    async def custom_method(self, **kwargs):
        # Your agent logic here
        return {"status": "success", "result": "custom_result"}

# Register with orchestrator
orchestrator.agents['custom_task'] = CustomAgent()

# Update task configuration
orchestrator.tasks['custom_task'] = TaskConfig(
    name='custom_task',
    task_type=TaskType.CUSTOM,
    enabled=True,
    interval_minutes=60,
    priority=TaskPriority.MEDIUM
)
```

## System Integration

### Integration with Existing Components

The orchestrator integrates with these existing system components:

1. **Data Management**
   ```python
   from trading.data.data_manager import DataManager
   orchestrator.data_manager = DataManager()
   ```

2. **Risk Management**
   ```python
   from trading.risk.risk_manager import RiskManager
   orchestrator.risk_manager = RiskManager()
   ```

3. **Portfolio Management**
   ```python
   from trading.portfolio.portfolio_manager import PortfolioManager
   orchestrator.portfolio_manager = PortfolioManager()
   ```

4. **Market Analysis**
   ```python
   from market_analysis.market_analyzer import MarketAnalyzer
   orchestrator.market_analyzer = MarketAnalyzer()
   ```

5. **Reporting System**
   ```python
   from reporting.report_generator import ReportGenerator
   orchestrator.report_generator = ReportGenerator()
   ```

### Health Monitoring Integration

```python
from system.health_monitor import SystemHealthMonitor
orchestrator.health_monitor = SystemHealthMonitor()

# Monitor system health
health_status = await orchestrator.health_monitor.get_status()
```

## Monitoring and Control

### Real-time Monitoring

```python
# Get system status
status = orchestrator.get_system_status()

print(f"Orchestrator Running: {status['orchestrator_running']}")
print(f"Total Tasks: {status['total_tasks']}")
print(f"Enabled Tasks: {status['enabled_tasks']}")
print(f"Overall Health: {status['performance_metrics']['overall_health']}")
```

### Task Monitoring

```python
# Get specific task status
task_status = orchestrator.get_task_status("model_innovation")

print(f"Task Enabled: {task_status['enabled']}")
print(f"Success Rate: {task_status['success_rate']}")
print(f"Error Count: {task_status['error_count']}")
```

### Performance Metrics

```python
# Export performance report
report_path = orchestrator.export_status_report()

# Get performance metrics
metrics = orchestrator.performance_metrics
print(f"Success Rates: {metrics['success_rates']}")
print(f"Error Counts: {metrics['error_counts']}")
```

## Web Interface Integration

### Streamlit App Integration

The Task Orchestrator is integrated into the existing Streamlit app:

1. **Navigation**: Added "🤖 Orchestrator" to the advanced navigation
2. **Status Display**: Real-time orchestrator status and health
3. **Task Monitoring**: Live task execution status
4. **Quick Actions**: Start/stop, configure, and monitor tasks

### API Integration

```python
from fastapi import FastAPI
from system.orchestrator_integration import EvolveSystemIntegration

app = FastAPI()
integration = EvolveSystemIntegration()

@app.get("/orchestrator/status")
async def get_orchestrator_status():
    return integration.get_system_status()

@app.post("/orchestrator/execute")
async def execute_task(task_name: str, parameters: dict = None):
    return await integration.execute_system_command(task_name, parameters)
```

## Error Handling and Recovery

### Automatic Recovery

The orchestrator includes automatic error recovery:

1. **Retry Mechanism**: Failed tasks are retried automatically
2. **Error Thresholds**: Tasks are disabled after too many failures
3. **Health Monitoring**: System health is continuously monitored
4. **Graceful Degradation**: System continues operating with reduced functionality

### Manual Recovery

```python
# Restart failed task
await orchestrator.execute_task_now("failed_task")

# Update task configuration
orchestrator.update_task_config("failed_task", {
    "retry_count": 5,
    "error_threshold": 10
})

# Reset error counts
orchestrator.error_counts["failed_task"] = 0
```

## Performance Optimization

### Task Scheduling Optimization

1. **Priority-based Scheduling**: Critical tasks run first
2. **Dependency Management**: Tasks wait for dependencies
3. **Resource Management**: Limit concurrent executions
4. **Conditional Execution**: Tasks run only when conditions are met

### Performance Monitoring

```python
# Monitor task performance
for task_name, task in orchestrator.tasks.items():
    if task_name in orchestrator.success_rates:
        success_rate = orchestrator.success_rates[task_name]
        if success_rate < task.performance_threshold:
            logger.warning(f"Task {task_name} below performance threshold: {success_rate}")
```

## Security Considerations

### Access Control

1. **Configuration Validation**: All configuration is validated
2. **Error Logging**: All errors are logged securely
3. **Resource Limits**: Execution time and resource limits are enforced
4. **Audit Trail**: All actions are logged for audit purposes

### Best Practices

1. **Environment Separation**: Use different configurations for dev/test/prod
2. **Monitoring**: Set up alerts for critical failures
3. **Backup**: Regularly backup configuration and status data
4. **Updates**: Keep orchestrator and dependencies updated

## Troubleshooting

### Common Issues

1. **Agent Not Found**
   ```
   Error: Agent 'model_innovation' not found
   Solution: Ensure the agent module is properly installed and importable
   ```

2. **Configuration Error**
   ```
   Error: Invalid configuration format
   Solution: Validate YAML syntax and required fields
   ```

3. **Permission Error**
   ```
   Error: Cannot write to logs directory
   Solution: Ensure proper file permissions
   ```

4. **Import Error**
   ```
   Error: Module not found
   Solution: Install missing dependencies or check Python path
   ```

### Debug Mode

Enable debug logging:

```bash
python start_orchestrator.py --standalone --log-level DEBUG
```

### Status Check

Check system status:

```bash
python start_orchestrator.py --status
```

## Migration Guide

### From Manual Agent Management

If you're currently managing agents manually:

1. **Identify Current Agents**: List all agents currently running
2. **Create Task Configurations**: Define schedules and dependencies
3. **Test Integration**: Run in test environment first
4. **Gradual Migration**: Migrate agents one by one
5. **Monitor Performance**: Ensure no performance degradation

### From Other Schedulers

If migrating from other scheduling systems:

1. **Export Current Schedules**: Export existing schedules
2. **Map to Task Configurations**: Convert to orchestrator format
3. **Test Compatibility**: Ensure all dependencies are met
4. **Parallel Running**: Run both systems in parallel initially
5. **Switch Over**: Switch to orchestrator when confident

## Support and Maintenance

### Regular Maintenance

1. **Health Checks**: Run health checks regularly
2. **Performance Reviews**: Review performance metrics monthly
3. **Configuration Updates**: Update configurations as needed
4. **Log Rotation**: Rotate logs to prevent disk space issues

### Monitoring Setup

Set up monitoring for:

1. **System Health**: Overall system health score
2. **Task Success Rates**: Individual task performance
3. **Error Rates**: Error frequency and patterns
4. **Resource Usage**: CPU, memory, and disk usage

### Backup Strategy

1. **Configuration Backup**: Backup task configurations
2. **Status Backup**: Backup execution status and metrics
3. **Log Backup**: Backup execution logs
4. **Recovery Plan**: Document recovery procedures

## Conclusion

The Task Orchestrator provides a robust, scalable foundation for managing your Evolve trading platform. With proper integration and configuration, it ensures reliable operation while maintaining high performance and system health.

For additional support, refer to the main documentation in `docs/TASK_ORCHESTRATOR.md` or contact the development team. 