# Core Agents

This directory contains the core AI agents responsible for cognitive tasks, decision-making, and intelligent behavior in the system.

## Structure

- `base_agent.py` - Abstract base class for all cognitive agents
- `router.py` - Intelligent task routing and planning
- `self_improving_agent.py` - Agent capable of self-improvement and learning
- `goal_planner.py` - Goal-oriented planning and execution
- `agent_manager.py` - Management and coordination of cognitive agents

## Agent Types

1. **Cognitive Agents**
   - Task planning and execution
   - Decision making
   - Learning and adaptation
   - Goal-oriented behavior

2. **Meta Agents**
   - Code generation and review
   - Documentation
   - Testing and repair
   - Performance monitoring
   - Data quality assessment

3. **Specialized Agents**
   - Strategy selection
   - Alert handling
   - Monitoring and orchestration

## Usage

These agents are designed to work together through the `AgentManager` class, which handles:
- Agent registration and discovery
- Task routing and coordination
- State management
- Performance monitoring

## Integration

The core agents integrate with:
- LLM services for natural language understanding
- Memory systems for state persistence
- Monitoring systems for performance tracking
- Logging systems for observability 