# Empty States – Mock Data Removal

This document describes the replacement of all remaining mock/placeholder data with proper empty states across the dashboard. No fake numbers are displayed as if they are real.

---

## Summary

- **Portfolio (5)**: Fallback equity curve, rolling metrics, and strategy performance that used `np.random.normal()` when no positions exist now show a shared empty-state message and do not render charts.
- **Risk (6)**: `get_portfolio_data()` no longer returns synthetic returns; when there is no real data, the UI shows an empty state and skips VaR, volatility, drawdown, gauges, and charts. Placeholder warning removed.
- **Performance (7)**: Overall Portfolio Metrics, Performance Trend, Strategy Comparison, Attribution (tab4), and Advanced Analytics (tab5) no longer use `sample_returns` or other random data as primary display; when no real data, a shared empty state is shown and metrics/charts/tables are not rendered. Placeholder warning removed.
- **Reports (9)**: Quick Reports and Custom Report Builder no longer build performance/equity/drawdown from `np.random`; when no real report data, a shared empty state is shown and charts/tables with fake data are not rendered. Placeholder warning removed.
- **All four pages**: A consistent `_empty_state(message, icon)` helper is defined at the top of each file and used for every empty state (centered, padded div with icon and message).
- **data/live_feed.py**: Module-level `SIMULATION_MODE = True` and a clear comment were added; fallback historical and live simulation paths are wrapped in `if SIMULATION_MODE:` (simulation logic unchanged).

---

## 1. pages/5_Portfolio.py

| What was removed | What was added |
|------------------|----------------|
| `plot_equity_curve`: fallback DataFrame built with `np.random.normal(0, 100, len(dates))` when no positions or empty df | Function returns `None` when no positions or empty df; caller shows `_empty_state("No trading history yet. Make your first trade or run a backtest to see portfolio performance here.", "📊")` and does not render the chart |
| `plot_rolling_metrics`: fallback dates/returns with `np.random.normal(0, 0.02, ...)` and exception-path `np.random.normal(1.0, 0.3, ...)` | Function returns `None` when no positions or empty df or on calculation error; caller shows same empty state and does not render the chart |
| `plot_strategy_performance`: fallback strategy DataFrames with `np.random.normal(1000, 500, ...)` and `np.random.normal(0.05, 0.02, ...)`, and exception-path fallback metrics | Function returns `None` when no positions or empty df or on error; caller shows same empty state and does not render the chart |

**Helper:** `_empty_state(message: str, icon: str = "📊")` added at top of file; used in Performance Visualization tabs (Equity Curve, Rolling Metrics, Strategy Performance) when the corresponding plot function returns `None`.

---

## 2. pages/6_Risk_Management.py

| What was removed | What was added |
|------------------|----------------|
| `get_portfolio_data()`: all paths that returned `pd.Series(np.random.normal(0.0005, 0.015, 252), ...)` and mock positions | When no real data, function returns `(None, None, True)`. No synthetic returns are ever returned. Real data path left unchanged for when actual returns are provided later |
| `st.warning("Data unavailable - showing placeholder.")` in Risk Dashboard tab | Removed |
| Risk Dashboard: rendering of metrics, gauge, and charts when `returns` was from placeholder | When `has_data` is False (no returns or placeholder), `_empty_state("No portfolio data yet. Add positions or run a backtest to see risk metrics.", "📊")` is shown and the rest of the tab (auto-refresh, limits, metrics, gauge, charts) is skipped |
| Same pattern in VaR Analysis, Monte Carlo, Stress Testing, Advanced Risk Analytics tabs | Each tab uses `has_data = not is_placeholder and returns is not None and not returns.empty`; when False, shows the same empty state and skips all metrics/charts |

**Helper:** `_empty_state(message, icon)` added at top of file; used in all tabs that depend on `get_portfolio_data()` when there is no real data.

---

## 3. pages/7_Performance.py

| What was removed | What was added |
|------------------|----------------|
| **Tab 1 (Performance Summary):** `sample_returns = pd.Series(np.random.normal(0.0005, 0.015, 365), ...)` and `st.warning("Data unavailable - showing placeholder.")` | `sample_returns = None` (TODO: load from strategy logger). When None or empty, `_empty_state("No performance data yet. Run a backtest or make trades to populate this page.", "📈")` is shown and the entire block (Overall Portfolio Metrics, Strategy Comparison, Performance Trend, Narrative, Commentary, Top Trades) is not rendered |
| **Tab 4 (Attribution Analysis):** `sample_returns` and `benchmark_returns` from `np.random.normal(...)` | `sample_returns = None`; when None or empty, same empty state; else block contains all attribution content (with real data when available) |
| **Tab 5 (Advanced Analytics):** `sample_returns = pd.Series(np.random.normal(...))` | `sample_returns = None`; when None or empty, same empty state; else block contains configuration, rolling metrics, drawdown, regime analysis, correlation, performance dashboard, etc. |

**Helper:** `_empty_state(message, icon)` added at top of file; used in tabs 1, 4, and 5 when there is no performance data. Sections that use real data from MemoryStore or strategy logs are unchanged.

---

## 4. pages/9_Reports.py

| What was removed | What was added |
|------------------|----------------|
| Quick Reports: `performance_data` built with `np.random.normal(0.001, 0.02, ...)` and `np.cumsum(np.random.normal(...))`; `st.warning("Data unavailable - showing placeholder.")`; equity curve chart and performance table using that data | `has_report_data = False` (TODO: set from backtest/logs). When False, `_empty_state("No report data yet. Complete a backtest or trading session to generate reports.", "📋")` is shown and Executive Summary, Performance Metrics, charts, perf table, Risk Metrics, Trade Details, Attribution, Store report, Success, and Export Options are not rendered |
| Custom Report Builder – "Equity Curve Chart" section: `equity_values = 100000 + np.cumsum(np.random.normal(500, 2000, ...))` and `benchmark_values = 100000 + np.cumsum(np.random.normal(300, 1500, ...))` | Replaced with `_empty_state("No report data yet. Complete a backtest or trading session to generate reports.", "📋")`; no chart rendered |
| Custom Report Builder – "Drawdown Chart" section: `drawdown = -np.abs(np.random.normal(0, 2, len(dates)))` | Replaced with the same empty state; no chart rendered |

**Helper:** `_empty_state(message, icon)` added at top of file; used in Quick Reports when `has_report_data` is False and in Custom Report Builder for Equity Curve Chart and Drawdown Chart sections.

---

## 5. Shared empty-state helper (all four pages)

Each of the four pages defines the same helper at the top of the file:

```python
def _empty_state(message: str, icon: str = "📊"):
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px;color:#888;border:1px dashed #ddd;border-radius:8px;margin:20px 0">
        <div style="font-size:48px;margin-bottom:12px">{icon}</div>
        <div style="font-size:16px">{message}</div>
    </div>
    """, unsafe_allow_html=True)
```

This is used consistently so empty states look intentional and polished rather than broken.

---

## 6. data/live_feed.py

| What was added | Purpose |
|----------------|---------|
| Module-level `SIMULATION_MODE = True` | Makes it explicit that fallback data is simulation |
| Comment: `# SIMULATION MODE - replace with real broker feed when connecting to live data` | Documents intent to replace with real broker feed |
| `_get_fallback_historical_data`: at start of method, `if not SIMULATION_MODE: return pd.DataFrame()` | When not in simulation mode, no synthetic historical data is generated |
| `_get_fallback_live_data`: `if not SIMULATION_MODE: return {..., "price": None, "volume": None, "source": "unavailable"}` | When not in simulation mode, no synthetic live data is returned |

Simulation logic (e.g. `np.random.normal` usage) was not changed; it is only gated by `SIMULATION_MODE`.

---

## 7. Placeholder warnings removed

- **pages/6_Risk_Management.py**: Removed `st.warning("Data unavailable - showing placeholder.")` and the pattern `if returns is None or returns.empty: st.warning("No portfolio data available...")` in favor of the empty-state message and skipping content.
- **pages/7_Performance.py**: Removed `st.warning("Data unavailable - showing placeholder.")` from the Performance Summary tab (replaced by empty state).
- **pages/9_Reports.py**: Removed `st.warning("Data unavailable - showing placeholder.")` from the Quick Reports performance block (replaced by empty state).

No fake numbers are shown as real; when data is missing, only the shared empty-state message and optional icon are shown, and no charts or tables with placeholder data are rendered.
