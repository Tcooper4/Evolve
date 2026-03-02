# Agent Wiring — MemoryStore and Situational Awareness

This document describes how major page outputs are written to MemoryStore and how the main Chat agent (PromptAgent) and page assistants get full situational awareness across the app.

---

## Overview

- **Write path:** Strategy Testing, Forecasting, Trade Execution, and Risk Management pages write structured results to MemoryStore (long-term) under fixed namespaces and categories.
- **Read path:** The Chat flow (via `trading.services.chat_nl_service`) builds a "Current trading context" block from MemoryStore and injects it into the context sent to the LLM. Page assistants use `get_full_context_summary()` so they also see cross-page context.

All MemoryStore access is wrapped in try/except so failures never block the UI or Chat.

---

## (1) Strategy Testing → MemoryStore

**File:** `pages/3_Strategy_Testing.py`

**When:** Immediately after a successful quick backtest run, when `results` and `st.session_state.backtest_results` are set.

**Call:** `store.add(MemoryType.LONG_TERM, namespace="backtests", value={...}, category="results")`

**Content keys:** `strategy_name`, `symbol`, `total_return`, `sharpe_ratio`, `max_drawdown`, `win_rate`, `start_date`, `end_date`, `timestamp`

**Details:** Uses existing `_get_memory_store()`. Start/end dates come from the loaded data index (`data.index[0]`, `data.index[-1]`). Wrapped in try/except so a MemoryStore failure does not block the UI.

---

## (2) Forecasting → MemoryStore

**File:** `pages/2_Forecasting.py`

**When:** Right after storing the forecast in session state (`current_forecast`, `current_model`, `current_model_instance`) following a successful quick forecast.

**Call:** `store.add(MemoryType.LONG_TERM, namespace="forecasts", value={...}, category="results")`

**Content keys:** `symbol`, `model_name`, `horizon`, `forecast_first`, `forecast_last`, `confidence` (optional dict with `lower`/`upper` if available), `timestamp`

**Details:** First and last predicted values are taken from the forecast array. Confidence is set from `forecast_result['lower_bound']`/`upper_bound` when present. Wrapped in try/except.

---

## (3) Trade Execution → MemoryStore

**File:** `pages/4_Trade_Execution.py`

**When:** Right after a successful order submit and after appending the order to `active_orders` and `order_history`.

**Call:** `store.add(MemoryType.LONG_TERM, namespace="trades", value={...}, category="orders")`

**Content keys:** `symbol`, `action` (buy/sell), `quantity`, `price`, `mode` (paper/live from `st.session_state.execution_mode`), `timestamp`

**Details:** Price uses the limit price when present; otherwise 0. Mode comes from `st.session_state.get("execution_mode", "paper")`. Wrapped in try/except.

---

## (4) Risk Management → MemoryStore

**File:** `pages/6_Risk_Management.py`

**When:** After risk metrics are computed and appended to `risk_history`, and before rendering the Risk Gauge.

**Call:** `store.add(MemoryType.LONG_TERM, namespace="risk", value={...}, category="snapshots")` only when allowed by the throttle.

**Content keys:** `var_95`, `volatility`, `max_drawdown`, `sharpe_ratio`, `timestamp`

**Throttle:** At most once per hour. Before adding, the code lists the latest snapshot (`namespace="risk"`, `category="snapshots"`, `limit=1`) and checks its `created_at`. If it is within the last hour, no new snapshot is written. This avoids flooding MemoryStore on every Risk page refresh.

---

## (5) PromptAgent / Chat context enrichment

**Where:** The Chat page (`pages/1_Chat.py`) uses `trading.services.chat_nl_service` to build the context block and call the LLM. The system prompt and context are built there, not inside `agents/llm/agent.py`. PromptAgent is used for agent actions (`run_agent_action`); the actual LLM call is `chat_nl_service.call_claude(system_prompt, context_block, ...)`.

**Change:** In `chat_nl_service.py`:

- **`get_trading_context_summary(store)`**  
  Queries MemoryStore for:
  - Last 3 backtests: `list(LONG_TERM, namespace="backtests", category="results", limit=3)`
  - Last forecast: `list(LONG_TERM, namespace="forecasts", category="results", limit=1)`
  - Last 3 trades: `list(LONG_TERM, namespace="trades", category="orders", limit=3)`
  - Current risk snapshot: `list(LONG_TERM, namespace="risk", category="snapshots", limit=1)`  

  Formats these as a concise plain-English block (e.g. “Last backtest: RSI on AAPL returned 12.3%, Sharpe 1.4 (3 days ago). Last forecast: LSTM on AAPL predicts +2.1% over 5 days. Last trade: Bought 10 AAPL at $264 (paper, yesterday). Risk: VaR95=1.2%, volatility=15%, max drawdown=5%, Sharpe=1.2.”). Uses a small helper `_relative_time(ts)` for “X days ago” style strings.

- **`build_context_block(..., store=None)`**  
  If `store` is provided, it prepends a “Current trading context:” section (the result of `get_trading_context_summary(store)`) to the existing memory and agent-output blocks. The whole MemoryStore read/format step is wrapped in try/except so a failure never blocks Chat.

**Chat page:** `pages/1_Chat.py` now passes `store=store` into `build_context_block(...)`, so every Chat message gets the current trading context in the context block sent to the LLM.

---

## (6) Page assistant cross-page awareness

**File:** `trading/services/page_assistant_service.py`

- **`get_full_context_summary()`**  
  Returns a single plain-English paragraph by calling `get_memory_store()` and then `chat_nl_service.get_trading_context_summary(store)`. Returns the same style of summary as Chat (backtests, forecast, trades, risk). On any exception, returns `""`.

**File:** `ui/page_assistant.py`

- **`render_page_assistant(page_name)`**  
  Builds the page-specific context with `get_page_context(page_name, st.session_state)` and then appends cross-app context with `get_full_context_summary()`. The combined string is passed into the system prompt as “Here is their current context: …” so page assistants also benefit from backtests, forecasts, trades, and risk snapshot without duplicating query logic.

---

## Namespaces and categories summary

| Namespace   | Category   | Written by        | Consumed by                    |
|------------|------------|-------------------|---------------------------------|
| `backtests`  | `results`   | Strategy Testing  | Chat, page assistants          |
| `forecasts`  | `results`   | Forecasting       | Chat, page assistants          |
| `trades`     | `orders`    | Trade Execution   | Chat, page assistants          |
| `risk`       | `snapshots` | Risk Management   | Chat, page assistants (throttled) |

---

## Error handling

- Every MemoryStore write on the four pages is inside a try/except that catches all exceptions and passes (or logs at debug). No write failure should block or break the UI.
- Chat’s trading context is built inside a try/except in `build_context_block`; on failure, the rest of the context (memory + agent output) is still sent.
- `get_full_context_summary()` catches all exceptions and returns `""`, so page assistants always get at least the page-local context.

---

## References

- **AGENT_ARCHITECTURE.md** — Active agents and roles.
- **PAGE_ASSISTANT.md** — Page assistant component and context.
- **trading/memory/memory_store.py** — MemoryStore API (`add`, `list`, namespaces, categories).
