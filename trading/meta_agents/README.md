# Meta Agents

This directory contains agents responsible for system self-maintenance, monitoring, and quality assurance.

## Agent Overview

### Data Quality Agent (`data_quality_agent.py`)
- **Purpose**: Monitors and maintains data quality across the trading system
- **Trigger**: Scheduled (daily) and reactive (on data updates)
- **Orchestrator Interaction**: Reports data quality metrics and issues

### Performance Monitor Agent (`performance_monitor_agent.py`)
- **Purpose**: Tracks system performance metrics and resource usage
- **Trigger**: Scheduled (hourly) and reactive (on performance alerts)
- **Orchestrator Interaction**: Provides performance reports and alerts

### Test Repair Agent (`test_repair_agent.py`)
- **Purpose**: Automatically fixes failing tests and maintains test coverage
- **Trigger**: Reactive (on test failures) and scheduled (weekly maintenance)
- **Orchestrator Interaction**: Reports test status and repair actions

### Code Review Agent (`code_review_agent.py`)
- **Purpose**: Reviews code changes and suggests improvements
- **Trigger**: Reactive (on code changes) and scheduled (weekly reviews)
- **Orchestrator Interaction**: Provides code review feedback

### Code Generator Agent (`code_generator.py`)
- **Purpose**: Generates boilerplate code and documentation
- **Trigger**: User-invoked and scheduled (documentation updates)
- **Orchestrator Interaction**: Reports generation status

## Orchestrator Integration

The `orchestrator.py` file coordinates all meta agents by:
1. Scheduling regular maintenance tasks
2. Managing agent dependencies
3. Handling inter-agent communication
4. Providing a unified interface for system maintenance

## Usage

Meta agents can be triggered in three ways:
1. **Scheduled**: Regular maintenance tasks run on predefined schedules
2. **Reactive**: Agents respond to system events and issues
3. **User-invoked**: Manual triggering through the orchestrator

Example usage:
```python
from trading.meta_agents.orchestrator import MetaAgentOrchestrator

# Initialize orchestrator
orchestrator = MetaAgentOrchestrator()

# Run scheduled maintenance
orchestrator.run_scheduled_tasks()

# Trigger specific agent
orchestrator.trigger_agent('data_quality')
```

## Best Practices

1. Always use the orchestrator to trigger agents
2. Monitor agent logs for issues
3. Review agent reports regularly
4. Keep agent configurations up to date
5. Test agent interactions before deployment 