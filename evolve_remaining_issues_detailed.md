# 🛠️ Remaining Issues to Reach 100% Completion for Evolve

This checklist includes ONLY the unresolved issues and TODOs as of the latest zip (`Evolve-main (1).zip`). It is structured for execution by Cursor or a developer without introducing regressions.

---

## 1. 🔁 Agent Feedback Loop & Evolution

- [x] Add a `ModelImproverAgent` that reviews backtest performance and automatically updates model configs (dynamic hyperparameter tuning).
- [x] Add a `StrategyImproverAgent` that monitors strategy performance and adjusts logic (e.g. Bollinger band width, RSI thresholds).
- [x] Create agent memory logs (`logs/agent_thoughts.json`) to record decision rationale and enable recall.
- [x] Add goal-tracking framework (`goals/`) where agents can read/write completion progress (optional Redis/MongoDB for persistence).
- [x] Implement task delegation across multiple agents with roles (e.g., forecaster, optimizer, reviewer).

---

## 2. ⚙️ Forecasting Models – Completion Gaps

- [x] Validate `TransformerForecaster` if included; otherwise integrate and test with same ensemble pipeline.
- [x] Improve error handling for missing data during forecast execution (currently logs error but no UI fallback).
- [x] Enable per-model parameter tuning from UI (e.g. `n_estimators` for XGBoost, `epochs` for LSTM).
- [x] Add confidence interval display logic for all models on plots.

---

## 3. 📈 Strategy Engine – Optimization Missing

- [x] Add dynamic strategy tuning agent or function that adjusts RSI/Bollinger/MACD values based on recent returns.
- [x] Implement multi-strategy comparison matrix (table comparing Sharpe, Win Rate, Drawdown, etc.).
- [x] Enable strategy stacking (applying multiple strategies at once, e.g., RSI + MACD crossover).
- [x] Build guardrails for invalid parameter ranges (e.g. RSI period < 1 crashes strategy).

---

## 4. 📊 Backtesting & Reporting

- [x] Add performance attribution module: calculate returns per signal type and strategy.
- [x] Enable downloadable trade logs from UI (CSV/Excel).
- [x] Add rolling metrics (e.g. rolling Sharpe, volatility) over time.
- [x] Fix report edge case where empty buy/sell signals yield broken chart.
- [x] Create summary report per run (markdown or HTML): include charts, stats, and strategy summary.

---

## 5. 🧠 Prompt Intelligence & Routing

- [x] Improve regex fallback matching for ambiguous prompts (e.g. "show me analysis of AAPL this week").
- [x] Add named intent templates for: `compare_strategies`, `optimize_model`, `debug_forecast`, etc.
- [x] Improve parser reliability when OpenAI is unavailable (add Hugging Face model caching).
- [x] Let PromptRouter return full JSON spec of inferred action for debugging (optional toggle).

---

## 6. 🧪 Testing Coverage

- [x] Add test cases for new strategy combinations.
- [x] Create tests for agent registry and fallback loading.
- [x] Test prompt template formatter against malformed or missing variables.
- [x] Add ensemble voting test for hybrid forecasting model.

---

## 7. 🎛️ UI Enhancements

- [x] Add dynamic sliders for all model and strategy parameters (with min/max validation).
- [x] Display confidence score and model selection breakdown for hybrid model.
- [x] Add session summary bar showing selected ticker, date range, active strategy, and last run.
- [x] Enable log view or expandable error viewer inside UI.

---

## 8. 🧱 System Resilience

- [x] Add global exception logging to capture and log all unhandled errors (`core/error_handler.py`).
- [x] Add fallback if any model fails during ensemble voting (e.g. fallback to last good forecast).
- [x] Warn user if no buy/sell signals are found (e.g. "Strategy inactive for selected period").

---

## 9. 🔁 Modularity & Cleanup

- [x] Verify no agent logic exists outside `agents/` or `trading/agents/`.
- [x] Move all prompt strings to `prompt_templates.py` if any remain hardcoded.
- [x] Add comment blocks to all agent init methods (`__init__`) describing expected args and purpose.
- [x] Ensure every module uses absolute imports only (Cursor safe).

---

## ✅ Once All Above Are Done:

- Rerun final tests in `tests/`.
- Confirm working state of all UI controls.
- Run a full end-to-end prompt (e.g., "Forecast AAPL with RSI strategy and show me the best model") and verify correct execution.

