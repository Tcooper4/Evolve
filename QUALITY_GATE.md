# Quality Gate for MemoryStore Writes

This document describes the quality filters applied before writing to MemoryStore at the four page locations defined in AGENT_WIRING.md, and how the Chat agent is informed when data may be unreliable.

---

## Overview

Before each situational-awareness write to MemoryStore (backtests, forecasts, trades, risk snapshots), the code now enforces **quality gates** so that only valid, meaningful data is persisted. When data is written but is known to be anomalous or degenerate, a **flag** is added so that the Chat agent can surface a plain-English **caveat** to the user.

---

## 1. Strategy Testing (`pages/3_Strategy_Testing.py`)

**Namespace/category:** `backtests` / `results`

**Quality gate:**

- Write **only** if the backtest completed without errors **and** `sharpe_ratio` is not `None` **and** `total_return` is not `None`. This applies to both MemoryStore writes: the `StreamlitStrategyTesting` / `backtest` write and the `backtests` / `results` write used for Chat context.
- If the result looks **anomalous** (`sharpe_ratio < -2` **or** `total_return < -0.8`), the `backtests`/`results` entry is still written (so the agent has context) but the value dict includes **`"anomalous": True`**.

**Effect:** Prevents writing backtest results that are missing key metrics; flags extreme results so the Chat agent can advise the user to treat them with caution.

---

## 2. Forecasting (`pages/2_Forecasting.py`)

**Namespace/category:** `forecasts` / `results`

**Quality gate:**

- Write only if there are forecast values (at least `forecast_first` or `forecast_last` is set).
- If the forecast looks **degenerate** (all values the same — flat line, indicating possible model failure), the value dict includes **`"model_failed": True`**. Degeneracy is detected via `np.unique(...).size <= 1` on the forecast array.

**Effect:** Avoids writing completely empty forecasts; when the forecast is flat, the Chat agent can warn that the last forecast may indicate model failure.

---

## 3. Trade Execution (`pages/4_Trade_Execution.py`)

**Namespace/category:** `trades` / `orders`

**Quality gate:**

- Write **only** if the order was **accepted**. The code checks `order_info.get("status")` and writes only when status is one of: `"submitted"`, `"filled"`, `"accepted"`, `"executed"`. Rejected or errored orders are not written.

**Effect:** Only successful or in-progress orders appear in Chat’s trading context; failed orders do not pollute the summary.

---

## 4. Risk Management (`pages/6_Risk_Management.py`)

**Namespace/category:** `risk` / `snapshots`

**Existing behavior:** Throttle of at most one write per hour is unchanged.

**Quality gate added:**

- Write **only** if **both** `volatility` and `sharpe_ratio` from the risk metrics are **not `None`** (zero is allowed). If either is missing, the snapshot is skipped.

**Effect:** Ensures the Chat agent only sees risk snapshots with at least volatility and Sharpe ratio; avoids incomplete or invalid snapshots.

---

## 5. Chat Context Caveats (`trading/services/chat_nl_service.py`)

**Function:** `get_trading_context_summary(store)`

When building the context block for the Chat agent:

- **Backtests:** For each backtest record, if the stored value has **`anomalous: True`**, an extra sentence is appended:  
  **"Note: last backtest result was anomalous — treat with caution."**
- **Forecasts:** For the last forecast record, if the stored value has **`model_failed: True`**, an extra sentence is appended:  
  **"Note: last forecast may indicate model failure — treat with caution."**

These strings are included in the same plain-English context block that is sent to the LLM, so the agent can be honest with the user about data quality without changing the rest of the flow.

---

## Summary Table

| Page              | Namespace   | Gate condition                          | Optional flag in value   | Caveat text (when flag set) |
|-------------------|------------|------------------------------------------|--------------------------|-----------------------------|
| Strategy Testing  | backtests  | `sharpe_ratio` and `total_return` non-None | `anomalous: True`        | "Last backtest result was anomalous — treat with caution." |
| Forecasting       | forecasts  | Has forecast_first/forecast_last          | `model_failed: True`     | "Last forecast may indicate model failure — treat with caution." |
| Trade Execution   | trades     | Order status in submitted/filled/accepted/executed | —                    | —                           |
| Risk Management   | risk       | `volatility` and `sharpe_ratio` non-None  | —                        | —                           |

All write paths remain wrapped in try/except so MemoryStore failures do not block the UI.
