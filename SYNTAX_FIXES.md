# Syntax and Import Fixes

This document summarizes fixes applied for IndentationErrors from mock data cleanup, stub modules for missing agents, and the Optuna tuner import correction.

---

## 1. pages/7_Performance.py

### Empty / mis-indented blocks fixed

- **~line 1482 (Attribution tab)**  
  `with col_ab1` … `with col_ab4` blocks: bodies (e.g. `st.metric(...)`) were at the same indent as the `with`. Indented so the body is clearly under each `with`.

- **Alpha/Beta figure**  
  Indented `fig_ab.add_trace` and `fig_ab.update_layout` under the correct scope.

- **~1651 – Factor Attribution**  
  `if attribution_method in ["Factor Attribution", "Comprehensive"]:` had no indented body. The whole block (subheader, factor_attribution, col_fact1/col_fact2, dataframes, chart) was indented as the body of that `if`.

- **~1688 – Time-Based Attribution**  
  `if attribution_method in ["Time-Based Attribution", "Comprehensive"]:` had no body. The block (subheader, time_period selectbox, time_returns, `if not time_returns.empty` and its content) was indented as the body of that `if`.

- **~1772 – Benchmark comparison**  
  `with col_bench1` and `with col_bench2` had no indented bodies. Their contents (cumulative returns chart and comparison table / excess return) were indented under the respective `with`.

- **Helper functions (rolling metrics, drawdown, recovery, regime)**  
  Corrected indentation for:
  - `calculate_rolling_sharpe` (function body and `if ... return pd.Series()`)
  - `calculate_rolling_max_drawdown` (body, `if` return, and `for` loop body)
  - `analyze_drawdown_periods` (body, `if` return, and `for`/`if`/`elif` bodies)
  - `calculate_recovery_time` (body, both `if` returns and final `return` dict)
  - `classify_market_regime` (full body: `regimes = []`, `for` loop, and `return pd.Series(...)`)
  - `calculate_regime_performance` (full body: early returns, common_index, aligned_returns, `for regime`, `regime_perf`, `return regime_perf`)
  - `calculate_strategy_correlation` (full body: early return, strategies, corr_matrix, return DataFrame)

- **~2040 – Rolling metrics `except ImportError`**  
  The `except ImportError` block had no body. The fallback block (comment, `fig_rolling = make_subplots(...)`, add_trace, update_layout, `st.plotly_chart`) was indented as the body of that `except`.

- **~2103 – Drawdown periods `if not drawdown_periods.empty`**  
  The `if` had no body. The block (display_dd, dataframe, try/except for drawdown chart, and the inner `else: st.info("No significant drawdown periods")` with its try/except) was indented under the `if`, and the outer `else` for “no periods” was aligned correctly.  
  **~2156 – `with col_dd2`**  
  Body (`st.markdown`, risk metrics, `st.metric` calls) was indented under `with col_dd2`.

- **~2172 – Trade distribution `if not trade_history.empty`**  
  The `if` had no body. The block (`col_dist1/2/3`, `with col_dist1/2/3` and their contents) was indented under the `if`; the `else: st.info("No trade history...")` was aligned with the `if`.

- **~2269 – Regime-based performance `if regime_perf`**  
  The `if` had no body. The block (col_regime1/2, `with col_regime1/2`, regime timeline `if not regimes.empty`, and the outer `else: st.warning("Unable to calculate regime-based performance")`) was indented under the `if regime_perf`.

- **~2349 – Strategy correlation `if not strategy_df.empty and len(strategy_df) > 1`**  
  The `if` had no body. The block (`strategy_corr = ...`, `if not strategy_corr.empty` with col_corr1/2 and their contents, and the outer `else: st.info("Need at least 2 strategies...")`) was indented under the first `if`.

- **~2432 – Advanced Performance Visualization `try`**  
  The `try` had no body. The block (import `PerformanceVisualizer`, `visualizer`, equity_curve, benchmark_curve, trade history prep, button and spinner with inner try/except) was indented under the `try`; the `except ImportError` body was indented under the `except`.

- **~2520 – Execution Replay `try`**  
  The `try` had no body. The block (import `ExecutionReplay`, replay, col1/col2 date inputs, `if st.button("Load Execution History")` with spinner, inner `try`/`if executions`/`else`/`except Exception`) was indented under the outer `try`. The inner `try`'s `except Exception` was aligned with that inner `try`. The two `else` blocks (“Execution ID column not found” and “No executions found”) were aligned with their respective `if` statements. The outer `except ImportError` and `except Exception` bodies were indented under their clauses.

---

## 2. pages/9_Reports.py

### Empty / mis-indented blocks fixed

- **~286 – Risk metrics**  
  `if include_risk_metrics:` had no body. The entire block (st.markdown, risk_metrics dict, col1/col2/col3, st.metric calls) was indented as the body of that `if`.

- **~361 – Export report columns**  
  - `with export_col1:` was over-indented and its body was mis-indented; corrected so the `with` is aligned with the columns assignment and the body (comment, `if st.button("📄 Export PDF")`, `try`/`except` block) is properly nested.  
  - `with export_col2:`: same pattern for Excel export (`if st.button`, `try`/`except`).  
  - `with export_col3:`: same pattern for HTML export; `with export_col3` indent was fixed and the `if st.button`/`try`/`except` body indented.  
  - `with export_col4:`: same pattern for Email report; `if st.button` body (`email_address = st.text_input`, `if email_address and st.button("Send")` …) was indented under the `if`.

---

## 3. Stub modules (warnings: “Could not initialize model creator” / “Could not initialize prompt router”)

- **trading/agents/model_creator_agent.py**  
  New stub defining `get_model_creator_agent()` returning `None`, so callers (e.g. `agents/llm/agent.py`) can import it without “No module named 'trading.agents.model_creator_agent'”.

- **trading/agents/prompt_router_agent.py**  
  New stub defining:
  - `create_prompt_router()` returning `None`
  - A minimal `PromptRouterAgent` class (for code that imports the class)  
  So “No module named 'trading.agents.prompt_router_agent'” is resolved.

---

## 4. Optuna tuner import (warning: “cannot import name 'OptunaTuner'”)

- **trading/optimization/optuna_tuner**  
  The module exposes `SharpeOptunaTuner` and `get_sharpe_optuna_tuner`, not `OptunaTuner`.

- **pages/8_Model_Lab.py**  
  - Import changed from `OptunaTuner` to `SharpeOptunaTuner, get_sharpe_optuna_tuner` (from `trading.optimization.optuna_tuner`).  
  - Session state initialization for the tuner now uses `get_sharpe_optuna_tuner()` instead of `OptunaTuner()`.

---

## 5. Verification

- Both files compile with zero errors:
  ```bash
  py -3.10 -m py_compile pages/7_Performance.py pages/9_Reports.py
  ```
