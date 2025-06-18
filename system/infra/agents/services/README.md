# Services Module

This module provides core services for the agentic forecasting platform.

## Features

- Automation monitoring
- Workflow management
- Service health checks
- Performance tracking
- Resource monitoring
- Alert generation

## Structure

- `automation_monitoring.py`: Service monitoring and health checks
- `automation_workflows.py`: Workflow execution and management
- `alert_manager.py`: Alert generation and handling
- `performance_tracker.py`: Performance metrics collection
- `resource_monitor.py`: Resource usage monitoring

## Usage

```python
from services import AutomationMonitor, WorkflowManager

# Initialize services
monitor = AutomationMonitor()
workflow_manager = WorkflowManager()

# Start monitoring
await monitor.start_monitoring()

# Create and execute workflow
workflow = workflow_manager.create_workflow({
    "id": "test_workflow",
    "steps": [...]
})
await workflow_manager.execute_workflow(workflow.id)
```

## Testing

Run service tests:
```bash
pytest tests/services/
```

## Monitoring

Services expose metrics for:
- Health status
- Performance metrics
- Resource usage
- Workflow execution
- Alert generation

## Configuration

Services can be configured via:
- Environment variables
- Configuration files
- Command-line arguments 