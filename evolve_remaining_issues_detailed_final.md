
# 🧠 Evolve System: Final Detailed Remaining Issues Checklist (Post-Zip Review)

This checklist reflects all remaining issues found in the most recent codebase (900 files, Evolve-main (2).zip).
Each section includes actionable items that can be directly implemented in Cursor or another IDE.

---

## 🔁 1. AGENT SYSTEM

### Prompt Routing & Agents
- [ ] **Agent Discovery Fallback**: `agents/registry.py` should fail gracefully if an agent class is missing.
- [ ] **Voice Agent Modularization**: `voice_prompt_agent.py` still has some partially duplicated routing logic. Route through `PromptRouterAgent` directly and confirm centralized handling.
- [ ] **Agent Config Templates**: Agents lack a shared config file for routing behavior (OpenAI fallback, timeout, max tokens). Create `agents/agent_config.py`.

---

## 📊 2. FORECASTING MODELS

### Core Models
- [ ] **LSTM Tuning**: `lstm_model.py` - still missing dropout regularization, dynamic sequence length control, and batch normalization.
- [ ] **XGBoost Model Metrics**: `xgboost_model.py` - lacks MSE and confidence interval return to UI. Add standardized interface like Prophet.
- [ ] **Ensemble Confidence Aggregation**: Hybrid model must weight predictions based on model error over time. Currently static weights.

---

## 📈 3. STRATEGY ENGINE

### Strategy Logic
- [ ] **Vectorization Pass**: Re-check `rsi_signals.py`, `macd_signals.py`, and `bollinger_signals.py` for any lingering loops or `.iterrows()`.
- [ ] **Strategy Naming in Reports**: Trade reports still omit strategy name used for signal generation.
- [ ] **Custom Strategy Upload**: No current way to define and run user-defined indicators. Add support for code injection or modular config YAML.

---

## 🧪 4. TESTING & STABILITY

- [ ] **Unit Coverage Report**: Run `pytest --cov` and export HTML report for coverage review.
- [ ] **Backtest Precision Tests**: Some strategies show +/-1 trade mismatch when compared to signals chart. Add tolerance tests.

---

## 📊 5. UI/UX & EXPORTS

- [ ] **Chart Alignment Bug**: Buy/Sell markers occasionally misalign with index after slider is adjusted rapidly.
- [ ] **Confidence Intervals Chart**: Forecast charts lack shaded CI band for LSTM/XGBoost.
- [ ] **Equity Curve with Benchmark**: Show SPY benchmark alongside custom strategy equity curve.

---

## 🤖 6. AGENTIC BEHAVIOR

- [ ] **Autonomous Mode Toggle**: Add sidebar checkbox to allow full autonomous execution without user dropdowns (e.g. trigger all forecasts, tune models, run backtest).
- [ ] **Task History Memory**: PromptRouterAgent should store task history + results to avoid repeating same command.
- [ ] **Self-Improving Agent**: (Future) Implement agent loop that runs strategy → backtest → adjusts hyperparameters automatically.

---

## 🧼 7. MISC

- [ ] **Requirements Auto-Repair**: Add CLI util (`fix_env.py`) that auto-detects and fixes PyTorch, NumPy, pandas-ta conflicts.
- [ ] **Logging Unification**: Some modules use print(), some use `logging`. Convert to centralized logger (e.g., `core/logger.py`).

---

### ✅ Next Step: Give Cursor This Prompt

```
Go through each item in evolve_remaining_issues_detailed_final.md and implement it exactly.
Do not invent new logic unless required.
If any fix is ambiguous, comment and halt.
Use absolute imports. Route all LLMs through PromptRouterAgent. Centralize all templates.
```

---

