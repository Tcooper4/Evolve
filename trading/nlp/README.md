# NLP Module for Agentic Forecasting System

This module provides natural language processing (NLP) capabilities for the forecasting platform, including entity extraction, intent classification, LLM integration, and agentic routing.

## Structure

- `prompt_processor.py` — Entity extraction, intent classification, agentic routing, and memory logging.
- `llm_processor.py` — LLM (OpenAI) integration with moderation, streaming, and JSON validation.
- `configs/entity_patterns.json` — Patterns for extracting trading-related entities.
- `sample_prompts.json` — Real-world prompts for testing and development.
- `memory_log.jsonl` — Memory log of prompt ➜ entities ➜ routed action.
- `logs/nlp_debug.log` — Debug log for all NLP operations.
- `sandbox_nlp.py` — Terminal-based sandbox for interactive NLP testing.

## Usage

### 1. Streamlit UI
- Use the `pages/nlp_tester.py` page to test entity extraction, intent classification, and LLM responses interactively.

### 2. Terminal Sandbox
- Run `python trading/nlp/sandbox_nlp.py` to test prompts, see parsed entities, intent, routing, and LLM output.

### 3. Automated Tests
- Run `pytest tests/nlp/` to execute unit tests for both processors.

## Integration Points
- `PromptProcessor.route_to_agent()` can be connected to your agentic routing system (e.g., ForecastAgent).
- Memory logs and debug logs are available for diagnostics and audit trails.

## Dependencies
- `openai` (for LLMProcessor)
- `streamlit` (for UI)
- `pytest` or `unittest` (for tests)

## Extending
- Add new entity types or patterns in `entity_patterns.json`.
- Enhance intent classification logic in `PromptProcessor`.
- Plug in your own agent or task router in `route_to_agent()`.

## Example Prompt
```
Show me the forecast for AAPL next week using the LSTM model.
```

---

**Modular, agentic, and ready for integration with the main forecasting pipeline.** 