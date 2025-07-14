# Refactored Prompt Router Documentation

## Overview

The refactored prompt router is a modular, memory-aware system that intelligently routes user prompts to specialized agents. It maintains memory of past interactions, automatically updates agent weights based on performance, and can detect when to search for better models.

## Key Features

### 🧠 Memory Management
- **Persistent Memory**: Stores all prompt interactions in JSON format
- **Similarity Detection**: Finds similar past prompts using fuzzy matching
- **Performance Tracking**: Maintains detailed performance metrics for each agent
- **Context Awareness**: Remembers user preferences and system state

### 🤖 Modular Agent Architecture
- **BasePromptAgent**: Abstract base class for all agents
- **Specialized Agents**: Each request type has its own agent class
- **Intelligent Selection**: Agents are selected based on performance and capability
- **Fallback Handling**: General agent handles unknown requests

### ⚖️ Automatic Weight Updates
- **Performance-Based Weights**: Agent weights are updated based on success rate, response time, and user satisfaction
- **Dynamic Selection**: Better performing agents are more likely to be selected
- **Automatic Optimization**: System continuously improves routing decisions

### 🔍 Model Search Detection
- **Performance Monitoring**: Automatically detects when agents are underperforming
- **Search Recommendations**: Suggests when to search for better models
- **Threshold-Based Alerts**: Triggers based on success rate, response time, and user satisfaction

## Architecture

### Core Classes

#### `RefactoredPromptRouter`
Main router class that orchestrates prompt handling and agent selection.

```python
router = get_prompt_router()
result = router.handle_prompt("What stocks should I buy today?")
```

#### `BasePromptAgent`
Abstract base class for all prompt handling agents.

```python
class CustomAgent(BasePromptAgent):
    def can_handle(self, prompt: str, processed: ProcessedPrompt) -> bool:
        # Return True if this agent can handle the prompt
        pass
    
    def handle(self, prompt: str, processed: ProcessedPrompt) -> Dict[str, Any]:
        # Handle the prompt and return response
        pass
```

#### `PromptMemoryManager`
Manages persistent memory of all prompt interactions.

```python
memory_manager = PromptMemoryManager()
similar_prompts = memory_manager.find_similar_prompts("What stocks should I buy?")
```

#### `EnhancedPromptProcessor`
Enhanced processor with memory integration and learning capabilities.

### Agent Types

#### `InvestmentAgent`
Handles investment-related queries:
- "What stocks should I buy today?"
- "Which stocks are recommended?"
- "Where should I invest?"

#### `ForecastAgent`
Handles forecasting requests:
- "Forecast AAPL for next week"
- "Predict TSLA price movement"
- "What will the market do tomorrow?"

#### `StrategyAgent`
Handles trading strategy queries:
- "What's the best RSI strategy?"
- "How should I trade MACD signals?"
- "Which strategy works for volatile markets?"

#### `GeneralAgent`
Fallback agent for unknown or general queries:
- "How is the weather?"
- "What time is it?"
- General system questions

## Usage Examples

### Basic Usage

```python
from agents.prompt_router_refactored import get_prompt_router, PromptContext

# Get router instance
router = get_prompt_router()

# Simple prompt handling
result = router.handle_prompt("What stocks should I buy today?")
print(f"Agent used: {result['agent_used']}")
print(f"Message: {result['message']}")
```

### With Context

```python
# Create context with user preferences
context = PromptContext(
    user_id="user_123",
    user_preferences={
        "risk_tolerance": "moderate",
        "preferred_sectors": ["technology", "healthcare"]
    },
    system_state={
        "market_condition": "bull",
        "volatility": "medium"
    }
)

# Handle prompt with context
result = router.handle_prompt("What should I invest in?", context)
```

### Performance Monitoring

```python
# Get performance report
report = router.get_performance_report()

for agent_name, stats in report["agents"].items():
    print(f"{agent_name}:")
    print(f"  Success Rate: {stats['success_rate']:.3f}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.3f}s")
    print(f"  Weight: {stats['weight']:.3f}")
```

## Configuration

### Memory Settings

The memory manager can be configured with different settings:

```python
# Custom memory file location
memory_manager = PromptMemoryManager("custom/path/memory.json")

# Memory settings
memory_manager.max_memories = 10000  # Maximum number of memories to keep
memory_manager.similarity_threshold = 0.8  # Similarity threshold for matching
```

### Agent Weights

Agent weights are automatically updated, but you can also manually adjust them:

```python
# Get agent performance
performance = memory_manager.get_agent_performance("InvestmentAgent")

# Update weight manually if needed
performance.weight = 0.9
```

## Performance Metrics

### Agent Performance Tracking

Each agent tracks the following metrics:

- **Success Rate**: Percentage of successful requests
- **Average Response Time**: Mean response time in seconds
- **User Satisfaction**: Average user feedback score (0-1)
- **Total Requests**: Number of requests handled
- **Weight**: Dynamic weight used for agent selection

### Weight Calculation

Agent weights are calculated using:

```python
weight = (success_rate * 0.6 + user_satisfaction * 0.4) * response_time_penalty
```

Where `response_time_penalty` is 0.8 if average response time > 2.0 seconds.

### Model Search Triggers

The system recommends model search when:

- Success rate < 0.6
- Average response time > 3.0 seconds
- User satisfaction < 0.5

## Memory Structure

### PromptMemory

```python
@dataclass
class PromptMemory:
    prompt_hash: str              # MD5 hash of normalized prompt
    original_prompt: str          # Original user prompt
    request_type: RequestType     # Classified request type
    agent_used: str              # Agent that handled the request
    success: bool                # Whether the request was successful
    response_time: float         # Response time in seconds
    user_feedback: Optional[float]  # User satisfaction score
    timestamp: datetime          # When the interaction occurred
    parameters: Dict[str, Any]   # Extracted parameters
```

### Memory File Format

The memory is stored in JSON format:

```json
{
  "memories": [
    {
      "prompt_hash": "abc123...",
      "original_prompt": "What stocks should I buy?",
      "request_type": "investment",
      "agent_used": "InvestmentAgent",
      "success": true,
      "response_time": 0.5,
      "user_feedback": 0.8,
      "timestamp": "2024-01-15T10:30:00",
      "parameters": {"symbol": "AAPL"}
    }
  ]
}
```

## Extending the System

### Adding New Agents

1. Create a new agent class inheriting from `BasePromptAgent`:

```python
class CustomAgent(BasePromptAgent):
    def __init__(self):
        super().__init__("CustomAgent")
        self.keywords = ["custom", "special", "unique"]
    
    def can_handle(self, prompt: str, processed: ProcessedPrompt) -> bool:
        return any(keyword in prompt.lower() for keyword in self.keywords)
    
    def handle(self, prompt: str, processed: ProcessedPrompt) -> Dict[str, Any]:
        # Implement custom handling logic
        return {
            "success": True,
            "message": "Custom response",
            "agent_used": self.name
        }
```

2. Register the agent in the router:

```python
router = get_prompt_router()
router.agents.append(CustomAgent())
```

### Custom Performance Metrics

You can extend the performance tracking by adding custom metrics:

```python
class CustomAgent(BasePromptAgent):
    def __init__(self):
        super().__init__("CustomAgent")
        self.custom_metric = 0.0
    
    def update_performance(self, success: bool, response_time: float, user_feedback: Optional[float] = None):
        super().update_performance(success, response_time, user_feedback)
        
        # Add custom metric update
        if success:
            self.custom_metric += 1
```

## Testing

### Running Tests

Use the provided test script:

```bash
python test_refactored_prompt_router.py
```

### Test Coverage

The test suite covers:

- Basic functionality testing
- Memory management
- Performance tracking
- Context awareness
- Agent selection accuracy

### Performance Testing

To test performance under load:

```python
import time
from concurrent.futures import ThreadPoolExecutor

def stress_test():
    router = get_prompt_router()
    prompts = ["What stocks should I buy?"] * 100
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(router.handle_prompt, prompts))
    
    return results
```

## Troubleshooting

### Common Issues

1. **Memory File Corruption**
   - Delete the memory file to reset: `rm data/prompt_memory.json`
   - The system will create a new memory file automatically

2. **Poor Agent Performance**
   - Check agent weights in the performance report
   - Consider adding more training data
   - Review agent selection logic

3. **Slow Response Times**
   - Monitor agent performance metrics
   - Check if model search is recommended
   - Consider optimizing agent implementations

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Regular monitoring tasks:

1. Check performance reports weekly
2. Review agent weights monthly
3. Monitor memory file size
4. Analyze user satisfaction trends

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - Use ML models for better prompt classification
   - Predictive agent selection
   - Automated parameter optimization

2. **Advanced Memory Features**
   - Semantic similarity using embeddings
   - Conversation context tracking
   - User behavior modeling

3. **Real-time Optimization**
   - Dynamic weight updates
   - A/B testing for agent selection
   - Real-time performance monitoring

4. **Multi-modal Support**
   - Image and voice prompt handling
   - Multi-language support
   - Context-aware responses

## API Reference

### Main Functions

- `get_prompt_router()`: Get singleton router instance
- `router.handle_prompt(prompt, context)`: Handle a user prompt
- `router.get_performance_report()`: Get performance statistics

### Data Classes

- `PromptContext`: Context information for prompt processing
- `ProcessedPrompt`: Processed prompt with metadata
- `PromptMemory`: Memory entry for past interactions
- `AgentPerformance`: Performance metrics for an agent

### Enums

- `RequestType`: Types of user requests (FORECAST, STRATEGY, etc.)

## Contributing

When contributing to the refactored prompt router:

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility
5. Add performance monitoring for new agents

## License

This refactored prompt router is part of the Evolve Trading Platform and follows the same licensing terms. 