# Prompt Examples System

## Overview

The Prompt Examples System is an intelligent few-shot learning mechanism that uses semantic similarity to find relevant examples for new prompts. This system enhances the PromptAgent's ability to understand and process user requests by providing context from similar successful interactions.

## Features

- **Semantic Similarity Search**: Uses SentenceTransformers to find similar prompts
- **Automatic Example Storage**: Saves successful prompt-response pairs
- **Few-Shot Learning**: Enhances prompts with relevant examples
- **Performance Tracking**: Tracks and scores prompt performance
- **Metadata Extraction**: Automatically extracts symbols, timeframes, and strategy types

## Architecture

### Components

1. **Prompt Examples JSON**: Stores successful prompt-response pairs
2. **Sentence Transformer**: Computes semantic embeddings for similarity search
3. **Similarity Engine**: Finds top-k similar examples using cosine similarity
4. **Few-Shot Generator**: Creates enhanced prompts with examples
5. **Performance Tracker**: Monitors and scores prompt success

### Data Structure

```json
{
  "examples": [
    {
      "id": "forecast_aapl_001",
      "prompt": "Forecast AAPL stock price for the next 30 days using technical analysis",
      "category": "forecasting",
      "symbols": ["AAPL"],
      "timeframe": "30 days",
      "strategy_type": "technical_analysis",
      "parsed_output": {
        "action": "forecast",
        "symbol": "AAPL",
        "timeframe": "30 days",
        "method": "technical_analysis",
        "confidence": 0.85
      },
      "success": true,
      "timestamp": "2024-01-15T10:30:00Z",
      "performance_score": 0.92
    }
  ],
  "metadata": {
    "version": "1.0",
    "total_examples": 15,
    "categories": ["forecasting", "strategy_creation", "backtesting"],
    "symbols": ["AAPL", "TSLA", "SPY", "NVDA"]
  }
}
```

## Usage

### Basic Usage

```python
from agents.llm.agent import PromptAgent

# Initialize agent (automatically loads examples)
agent = PromptAgent()

# Process a prompt (automatically uses few-shot learning)
response = agent.process_prompt("Forecast TSLA price for next 30 days")
```

### Manual Similarity Search

```python
# Find similar examples
similar_examples = agent._find_similar_examples(
    "Create RSI strategy for AAPL", 
    top_k=3
)

# Create few-shot prompt
enhanced_prompt = agent._create_few_shot_prompt(
    "Create RSI strategy for AAPL", 
    similar_examples
)
```

### Statistics and Monitoring

```python
# Get system statistics
stats = agent.get_prompt_examples_stats()
print(f"Total examples: {stats['total_examples']}")
print(f"Categories: {stats['categories']}")
print(f"Average performance: {stats['average_performance_score']:.3f}")
```

## How It Works

### 1. Example Loading

When the PromptAgent initializes:
- Loads `prompt_examples.json` from the agents directory
- Computes embeddings for all example prompts using SentenceTransformers
- Stores embeddings for fast similarity search

### 2. Similarity Search

For each new prompt:
- Encodes the prompt using the same sentence transformer
- Computes cosine similarity with all stored example embeddings
- Returns top-k most similar examples with similarity scores

### 3. Few-Shot Prompt Creation

Creates an enhanced prompt that includes:
- Original user prompt
- Similar examples with their successful outputs
- Instructions to follow the example format

### 4. Automatic Example Saving

After successful prompt processing:
- Extracts metadata (symbols, timeframes, strategy types)
- Calculates performance score
- Saves to JSON file with timestamp
- Updates embeddings for future searches

## Configuration

### Dependencies

```bash
pip install sentence-transformers
```

### Model Selection

The system uses `all-MiniLM-L6-v2` by default, which provides:
- Fast inference (384-dimensional embeddings)
- Good semantic understanding
- Small model size (~90MB)

### Customization

```python
# Use different sentence transformer model
from sentence_transformers import SentenceTransformer

agent = PromptAgent()
agent.sentence_transformer = SentenceTransformer('all-mpnet-base-v2')  # Larger, more accurate
```

## Performance Optimization

### Embedding Caching

- Embeddings are computed once and cached
- Automatic re-computation when examples are added
- Memory-efficient storage using numpy arrays

### Similarity Thresholds

```python
# Adjust similarity thresholds
similar_examples = agent._find_similar_examples(
    prompt, 
    top_k=3,
    min_similarity=0.5  # Only use examples above threshold
)
```

### Batch Processing

For multiple prompts:
```python
prompts = ["prompt1", "prompt2", "prompt3"]
all_similar = agent._find_similar_examples_batch(prompts, top_k=2)
```

## Categories and Types

### Supported Categories

- **forecasting**: Price prediction requests
- **strategy_creation**: Strategy development
- **backtesting**: Historical strategy testing
- **analysis**: Market analysis requests
- **portfolio_optimization**: Portfolio management
- **risk_assessment**: Risk analysis
- **signal_generation**: Trading signal creation
- **market_regime**: Market condition analysis

### Strategy Types

- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger_Bands**: Bollinger Bands strategy
- **Moving_Average**: Moving average strategies
- **Forecasting**: Time series forecasting
- **Backtesting**: Historical testing
- **Optimization**: Parameter optimization

## Best Practices

### 1. Example Quality

- Only save high-quality, successful examples
- Include diverse prompt types and categories
- Maintain balanced representation across strategies

### 2. Performance Scoring

- Use meaningful performance metrics
- Consider user feedback in scoring
- Regularly review and update scores

### 3. Similarity Thresholds

- Set appropriate similarity thresholds
- Avoid using low-similarity examples
- Monitor false positive matches

### 4. Regular Maintenance

- Periodically clean old examples
- Update embeddings when adding examples
- Monitor system performance

## Troubleshooting

### Common Issues

1. **SentenceTransformers Not Available**
   ```
   Warning: SentenceTransformers not available. Prompt examples will be disabled.
   ```
   **Solution**: Install sentence-transformers package

2. **No Similar Examples Found**
   - Check if examples file exists
   - Verify example quality and diversity
   - Adjust similarity thresholds

3. **Low Similarity Scores**
   - Review example prompts
   - Consider adding more examples
   - Check model performance

### Debug Mode

```python
import logging
logging.getLogger("agents.llm.agent").setLevel(logging.DEBUG)

# Process prompt with debug output
response = agent.process_prompt("test prompt")
```

## Examples

### Example 1: Forecasting Request

**Input**: "Forecast AAPL price for next 30 days"

**Similar Examples Found**:
1. "Forecast AAPL stock price for the next 30 days using technical analysis" (similarity: 0.95)
2. "Predict TSLA price for next month" (similarity: 0.82)

**Enhanced Prompt**:
```
Here are some similar examples to help guide your response:

Example 1:
Input: Forecast AAPL stock price for the next 30 days using technical analysis
Output: {
  "action": "forecast",
  "symbol": "AAPL",
  "timeframe": "30 days",
  "method": "technical_analysis",
  "confidence": 0.85
}
Category: forecasting
Performance Score: 0.92

Now, please process this request:
Forecast AAPL price for next 30 days

Based on the examples above, provide a structured response in JSON format.
```

### Example 2: Strategy Creation

**Input**: "Create RSI strategy for TSLA"

**Similar Examples Found**:
1. "Create a bullish RSI strategy for TSLA with 14-period lookback" (similarity: 0.98)
2. "Build RSI mean reversion strategy" (similarity: 0.85)

**Result**: Enhanced prompt with RSI strategy examples

## Integration

### With Existing Systems

The prompt examples system integrates seamlessly with:
- **ForecastRouter**: Enhanced forecasting requests
- **StrategyGatekeeper**: Better strategy selection
- **TradeExecutionSimulator**: Improved trade requests
- **SelfTuningOptimizer**: Optimized parameter suggestions

### API Endpoints

```python
# Get examples statistics
GET /api/prompt-examples/stats

# Find similar examples
POST /api/prompt-examples/similar
{
  "prompt": "Forecast AAPL price",
  "top_k": 3
}

# Save new example
POST /api/prompt-examples/save
{
  "prompt": "user prompt",
  "response": "parsed response",
  "category": "forecasting"
}
```

## Future Enhancements

### Planned Features

1. **Dynamic Weighting**: Weight examples by recency and performance
2. **Multi-modal Examples**: Support for images and charts
3. **User Feedback Integration**: Learn from user corrections
4. **A/B Testing**: Compare different example sets
5. **Clustering**: Group similar examples for better organization

### Research Areas

- **Better Embedding Models**: Domain-specific transformers
- **Active Learning**: Intelligent example selection
- **Cross-lingual Support**: Multi-language prompt examples
- **Contextual Similarity**: Consider market conditions and time

## Contributing

### Adding Examples

1. Create high-quality prompt-response pairs
2. Include relevant metadata
3. Test similarity matching
4. Update documentation

### Testing

```bash
# Run prompt examples tests
pytest tests/test_prompt_examples.py

# Run demo
python examples/prompt_examples_demo.py
```

### Code Style

- Follow existing code patterns
- Add comprehensive docstrings
- Include type hints
- Write unit tests for new features 