# Routing Module

This module provides a clean, modular interface for handling prompt routing in the Evolve Trading Platform.

## Overview

The routing module consolidates all prompt processing and routing logic from `app.py` into a clean, reusable interface. It handles:

- Natural language prompt analysis and classification
- Intelligent routing to appropriate agents and services
- Model/strategy selection and backtest decision logic
- Result formatting and forwarding
- Error handling and fallback mechanisms

## Key Components

### `PromptRouter` Class

The main router class that handles all prompt processing and routing logic.

**Features:**
- Automatic component initialization with error handling
- Multiple fallback mechanisms for robustness
- Navigation information extraction
- Comprehensive logging and monitoring
- Standardized response formatting

### `route_prompt()` Function

A convenience function that provides a simple interface for prompt routing.

**Usage:**
```python
from routing.prompt_router import route_prompt

result = route_prompt("Forecast SPY using the most accurate model", llm_type="default")
```

## API Reference

### `route_prompt(prompt: str, llm_type: str = "default") -> Dict[str, Any]`

Routes a user prompt and returns a structured response.

**Parameters:**
- `prompt`: User's input prompt
- `llm_type`: Type of LLM to use (default, gpt4, claude, etc.)

**Returns:**
A dictionary containing:
- `success`: Boolean indicating if processing was successful
- `message`: Response message from the system
- `data`: Additional data from processing
- `navigation_info`: Navigation suggestions for the UI
- `processing_time`: Time taken to process the prompt
- `timestamp`: ISO timestamp of processing
- `strategy_name`: Strategy used (if applicable)
- `model_used`: Model used (if applicable)
- `confidence`: Confidence score (if applicable)
- `signal`: Trading signal (if applicable)

## Integration with app.py

The new routing module replaces the inline prompt handling logic in `app.py`. The changes include:

1. **Cleaner Code**: Removed complex inline logic from app.py
2. **Better Error Handling**: Centralized error handling and fallback mechanisms
3. **Modular Design**: Easy to test, maintain, and extend
4. **Standardized Interface**: Consistent response format across the application

## Error Handling

The module includes comprehensive error handling:

- **Import Errors**: Graceful fallback when dependencies are missing
- **Agent Errors**: Fallback responses when agents fail
- **Processing Errors**: Detailed error messages and logging
- **Validation**: Input validation and sanitization

## Testing

Use the provided test script to verify functionality:

```bash
python test_prompt_router.py
```

## Future Enhancements

Potential improvements for the routing module:

1. **Caching**: Add response caching for similar prompts
2. **Rate Limiting**: Implement rate limiting for API calls
3. **Metrics**: Add performance metrics and monitoring
4. **Plugins**: Support for custom routing plugins
5. **A/B Testing**: Support for testing different routing strategies 