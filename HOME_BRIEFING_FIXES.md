# Home Briefing and Model Registry Fixes

This document summarizes three fixes applied to align the Home briefing with the global LLM selector, fix model registry loading, and improve fallback briefing readability.

---

## 1. Home briefing uses active LLM (not hardcoded Anthropic)

### Problem
`trading/services/home_briefing_service.py` was hardcoded to use the Anthropic SDK directly (`from anthropic import Anthropic`, `client.messages.create`). The briefing ignored the Admin LLM selection (Claude, GPT-4, Gemini, Ollama, etc.), and the fallback message told users to "Install the anthropic package".

### Solution
- **Removed** all direct use of the `anthropic` package and `client.messages.create`.
- **Use** `call_active_llm_chat()` from `agents.llm.active_llm_calls` for the briefing request. The briefing now uses whatever LLM the user has selected in Admin (Claude, GPT-4, Gemini, Ollama, Kimi, HuggingFace).
- **Fallback** when no provider is configured or the call fails:
  - Check `get_active_llm()`; on exception or missing provider, return the readable fallback (see §3).
  - Message text: **"AI briefing unavailable — configure an LLM provider in Admin"** (with instruction to use Admin → AI Model Settings and Refresh briefing).
  - Removed the "Install the anthropic package" message.

### Code changes
- `generate_briefing()`:
  - Imports: `from agents.llm.active_llm_calls import get_active_llm, call_active_llm_chat`.
  - Calls `get_active_llm()` first; if that fails or provider is missing, returns `_make_fallback_briefing(...)`.
  - Main AI call: `call_active_llm_chat(system_prompt=..., context_block=..., conversation_messages=[], user_message="Generate today's briefing and cards.", max_tokens=2048)`.
  - On any exception from the LLM call, returns `_make_fallback_briefing(...)` with the same user-facing message.
- New helper `_make_fallback_briefing(portfolio_summary)` builds the fallback dict with readable text for memory, market, and risk (see §3).

**Result:** Switching to GPT-4 (or any other provider) in Admin → AI Model Settings makes the Home briefing use that provider.

---

## 2. Model registry loader: correct import paths (no `models.forecast_router`)

### Problem
Ten model classes failed to load with:
`No module named 'models.forecast_router'`.

The registry loader in `trading/models/forecast_router.py` was using top-level paths like `models.arima_model`, `models.lstm_model`, etc. Importing `models.arima_model` loads the top-level `models/` package, whose `__init__.py` does `from .forecast_router import ForecastRouter`. There is no `models/forecast_router.py` (ForecastRouter lives in `trading/models/forecast_router.py`), so the import failed and broke model discovery.

### Solution
- **Discovery** now uses `trading.models.*` only:
  - Added `_DISCOVERY_CLASS_MAP`: stem → `(module_path, class_name)` with correct paths, e.g. `arima` → `("trading.models.arima_model", "ARIMAModel")`, `lstm` → `("trading.models.lstm_model", "LSTMModel")`, etc.
  - `_discover_available_models()` builds the class path as `f"{module_path}.{class_name}"` and calls `_get_model_class(...)` with that path (no more `models.{stem}_model.{stem.title()}Model`).
- **`_get_model_class(class_path)`**:
  - If `class_path` starts with `"models."` and not `"trading.models."`, rewrite to `"trading.models." + class_path[7:]` so configs or legacy paths using `models.*` resolve to `trading.models.*`.
  - Added `_CLASS_NAME_ALIASES` for wrong casing (e.g. `ArimaModel` → `ARIMAModel`, `LstmModel` → `LSTMModel`) so configs that specify the wrong class name still load.
- **Registry attribute** in `_load_model_registry()`: use `getattr(registry, "registry", None) or getattr(registry, "_models", None) or {}` so both `ModelRegistry` (with `.registry` property) and objects with only `_models` work.

### Config
- `config/model_registry.yaml` already uses `trading.models.*` class paths (e.g. `trading.models.arima_model.ARIMAModel`). No change required.

**Result:** All 10 model classes (ARIMA, LSTM, Prophet, XGBoost, CatBoost, GARCH, Ridge, TCN, Ensemble, BaseModel) load via `trading.models.*` without triggering the missing `models.forecast_router` module.

---

## 3. Fallback briefing: readable text instead of raw JSON

### Problem
In the fallback path (when no LLM is configured or the call fails), the briefing dumped raw JSON for the risk snapshot (e.g. `{"total_positions": 0, ...}`) and used `json.dumps(market_data)` for market data. Memory was already text but could be truncated; the overall fallback was not plain-English.

### Solution
- **Risk snapshot:** New helper `_format_risk_snapshot_readable(risk_snapshot, summary)`:
  - If `summary` is a dict: format as **"Equity: $100,000 | Open positions: 0 | P&L: $0"** (using `current_equity`/`equity`, `open_positions`, `total_pnl`).
  - Otherwise falls back to a short snippet of `risk_snapshot` or "No portfolio data available."
- **Market data:** New helper `_format_market_data_readable(market_data)`:
  - Plain-English lines per symbol, e.g. **"SPY: $450.25 (vs week ago: +1.2%)"**, or "No market data available."
- **Memory:** Already plain text; fallback uses it truncated to 2000 chars with "..." if longer, or "No trading history or preferences in memory yet."
- **Fallback card:** Replaced "Set up AI briefing" / "pip install anthropic" with **"Configure AI briefing"** and detail: "Go to Admin → AI Model Settings, choose a provider and set its API key, then Save. Use Refresh briefing to try again."

`_make_fallback_briefing(portfolio_summary)` uses these helpers so the fallback section shows only readable text—no raw JSON or Python dict dumps.

---

## Files touched

| File | Changes |
|------|--------|
| `trading/services/home_briefing_service.py` | Use `get_active_llm` and `call_active_llm_chat`; remove Anthropic imports and `client.messages.create`; add `_make_fallback_briefing`, `_format_market_data_readable`, `_format_risk_snapshot_readable`; fallback message and card text updated. |
| `trading/models/forecast_router.py` | Add `_DISCOVERY_CLASS_MAP` and `_CLASS_NAME_ALIASES`; `_discover_available_models()` uses `trading.models.*` paths; `_get_model_class()` normalizes `models.` → `trading.models.` and uses aliases; `_load_model_registry()` uses `registry` or `_models` safely. |
| `HOME_BRIEFING_FIXES.md` | This document. |

---

## References

- **LLM selector:** `LLM_SELECTOR.md` — global LLM selection via Admin, `get_active_llm()`, `call_active_llm_chat()` in `agents/llm/active_llm_calls.py`.
- **Config:** `config/model_registry.yaml` uses `trading.models.*` class paths; no updates needed for this fix.
