# Mock Data Audit (excluding _dead_code and tests)

This document lists every location in the **live** codebase (excluding `_dead_code/` and `tests/`) where mock/synthetic data generation was found, and what was done for each.

---

## Summary of actions

- **Fallback-only mock:** Comment added `# MOCK FALLBACK - replace with real data source` and, where the UI displays that data, `st.warning("Data unavailable - showing placeholder.")`.
- **Primary/default mock on a page:** Same comment and a visible `st.warning` so users know they are seeing placeholder data. No replacement with yfinance in this pass (can be done later via FallbackDataProvider).

---

## 1. pages/6_Risk_Management.py

| Line(s) | What | Action |
|--------|------|--------|
| 252–257 | Sample portfolio returns when `not positions`: `np.random.normal(0.0005, 0.015, 252)` | Comment `# MOCK FALLBACK - replace with real data source`. Function now returns third value `is_placeholder=True`. |
| 259–264 | "Simplified" returns when positions exist but no real returns: same `np.random.normal` | Comment `# MOCK FALLBACK - replace with real data source`. Return `is_placeholder=True`. |
| 267–272 | Exception path: sample returns and positions | Comment `# MOCK FALLBACK - replace with real data source`. Return `is_placeholder=True`. |
| 317–319 | First use of `get_portfolio_data()` in Risk Dashboard tab | When `is_placeholder` is True, show `st.warning("Data unavailable - showing placeholder.")`. All call sites updated to unpack `(returns, positions, is_placeholder)`. |

---

## 2. pages/7_Performance.py

| Line(s) | What | Action |
|--------|------|--------|
| 265–267 | `sample_returns = pd.Series(np.random.normal(0.0005, 0.015, 365), index=dates)` used as primary data for "Overall Portfolio Metrics" | Comment `# MOCK FALLBACK - replace with real data source`. Added `st.warning("Data unavailable - showing placeholder.")` in that section. |
| 267, 1434–1435, 1982, 2344, 2423 | Other occurrences of `np.random.normal` in Performance (sample_returns, benchmark_returns, strategy series) | Not changed in this pass; they are in other tabs/sections. Can be audited similarly and replaced with real data (e.g. FallbackDataProvider or strategy logger) in a follow-up. |

---

## 3. pages/9_Reports.py

| Line(s) | What | Action |
|--------|------|--------|
| 246–248 | `performance_data` with `np.random.normal` for Daily Return and Cumulative Return | Comment `# MOCK FALLBACK - replace with real data source`. Added `st.warning("Data unavailable - showing placeholder.")` in that block. |
| 773, 796, 856 | Equity curve, benchmark, drawdown built with `np.random.normal` / `np.cumsum(np.random.normal(...))` | Not changed in this pass; can be tagged and replaced in a follow-up. |

---

## 4. trading/data/providers/fallback_provider.py

| Line(s) | What | Action |
|--------|------|--------|
| 265–266, 272–276 | Mock volume, change, change_pct, and fallback price when provider fails | Comment updated to `# MOCK FALLBACK - replace with real data source` for the exception-path block. |
| 440–447, 532 | Other uses of `np.random.normal` for price/OHLC when generating or approximating data | Left as-is; they are part of fallback/simulation paths. Can be documented as fallback-only in a future pass. |

---

## 5. Other live files (no code changes in this pass)

| File | Location | Description |
|------|----------|-------------|
| pages/8_Model_Lab.py | ~3539 | `partial_dependence = np.sin(...) + np.random.normal(0, 0.1, ...)` — visualization/sensitivity, not primary market data. |
| trading/agents/model_synthesizer_agent.py | ~1191 | `np.random.normal` in agent logic. |
| trading/backtesting/performance_analysis.py | 589 | `np.random.normal` for sampling/noise. |
| trading/backtesting/monte_carlo.py | 202, 599 | Monte Carlo simulation — synthetic paths by design. |
| data/live_feed.py | 530–558 | Simulated live feed with `np.random.normal` — likely demo/sim. |
| agents/agent_controller.py | 1078 | `create_mock_agent` — fallback when real agent missing. |
| agents/llm/llm_summary.py | 188, 216, 223, 226 | `_generate_mock_summary` — fallback when LLM unavailable. |
| routing/prompt_router.py | 640, 771 | `create_mock_agent` — fallback. |
| trading/config/settings.py | 205, 543 | `MOCK_EXTERNAL_APIS` — config flag, not data. |
| config/config.py | 275, 625 | `mock_data` — config property. |
| execution/execution_agent.py | 282 | `np.random.normal` for price change — simulation. |
| trading/ui/institutional_dashboard.py | 457 | `np.random.normal` for returns. |

---

## 6. Excluded (tests and _dead_code)

All references under `tests/` and `_dead_code/` were excluded from this audit. Tests routinely use `np.random.normal`, `create_mock_*`, and mock DataFrames; no changes were made there.

---

## 7. Recommendations

- **Risk Management:** Replace placeholder returns with real data (e.g. fetch historical returns for portfolio symbols via FallbackDataProvider and aggregate).
- **Performance:** Replace `sample_returns` with strategy/portfolio returns from StrategyLogger or FallbackDataProvider when available.
- **Reports:** Replace performance_data and related equity/benchmark/drawdown construction with real series from memory or data provider.
- **fallback_provider:** When all providers fail, consider returning an error or a clear "no data" state instead of mock OHLC/volume so the UI can show "Data unavailable" without displaying fake numbers.
