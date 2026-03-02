# Minor Fixes

Summary of three minor fixes applied to reduce startup noise and prevent placeholder recommendations.

---

## 1. Startup warnings: model creator / prompt router

**Issue:** Startup showed warnings: "Could not initialize model creator" and "Could not initialize prompt router" even though no live code uses these agents.

**Change:** In `agents/llm/agent.py`, the initialization block for the model builder and prompt router already runs in `try/except`. The failure messages were downgraded from visible warnings to debug-only logs:

- **Before:** `self.logger.warning(f"Could not initialize model builder: {e}")`  
  **After:** `self.logger.debug(f"Could not initialize model builder: {e}")`

- **Before:** `self.logger.warning(f"Could not initialize prompt router: {e}")`  
  **After:** `self.logger.debug(f"Could not initialize prompt router: {e}")`

**Result:** Normal startup no longer shows these messages. They still appear when debug logging is enabled (e.g. `logging.DEBUG`).

---

## 2. OptunaTuner warning: "Some model modules not available: cannot import name OptunaTuner"

**Issue:** Code or transitive imports expected a class named `OptunaTuner`; the module only exposes `SharpeOptunaTuner`.

**Changes:**

- **`trading/optimization/optuna_tuner.py`**  
  Added a backward-compatibility alias at the bottom of the file:
  ```python
  # Backward-compatibility alias for code that expects OptunaTuner
  OptunaTuner = SharpeOptunaTuner
  ```
  Any import of `OptunaTuner` from this module (or code that references it after a generic import) now resolves to `SharpeOptunaTuner`.

- **Live code**  
  No files outside `_dead_code/` were importing `OptunaTuner`; they already use `SharpeOptunaTuner` and `get_sharpe_optuna_tuner` from `trading.optimization.optuna_tuner`. The alias ensures any remaining or future references to `OptunaTuner` work without changing call sites.

- **`tests/unit/test_optuna_tuner.py`**  
  Docstrings/comments updated to mention SharpeOptunaTuner (OptunaTuner alias) for clarity.

**Result:** The "cannot import name OptunaTuner" error is resolved; startup and Model Lab no longer hit this when loading model modules.

---

## 3. Monitoring tools: no placeholder degradation recommendations on startup

**Issue:** The monitoring_tools service was writing "Model degradation: model" and "Strategy degradation: strategy" to MemoryStore on startup when there was no real degradation data (e.g. no model_id/strategy_name, or only default/empty metrics).

**Change:** In `trading/services/monitoring_tools.py`:

- **`check_model_degradation()`**  
  - Existing guard kept: if both `recent_sharpe` and `recent_drawdown` are `None`, return immediately without writing.  
  - New guard: if degradation is detected but `model_id` is `None`, do **not** write to MemoryStore; return a result with `written_to_memory: False` and a message that the write was skipped (no model_id).  
  - Recommendation title is now always `f"Model degradation: {model_id}"` when we do write (no fallback to `"model"`).

- **`check_strategy_degradation()`**  
  - Same idea: if both performance metrics are `None`, return without writing (already in place).  
  - New guard: if degradation is detected but `strategy_name` is `None`, do **not** write; return with `written_to_memory: False` and skip message.  
  - Title when writing is `f"Strategy degradation: {strategy_name}"` (no fallback to `"strategy"`).

**Result:** Placeholder recommendations ("model", "strategy") are no longer written when there is no real identifier. Recommendations are only written when actual metrics show degradation **and** a concrete model_id or strategy_name is provided.

---

## Files touched

| File | Change |
|------|--------|
| `agents/llm/agent.py` | Model builder / prompt router init failures logged at `debug` instead of `warning`. |
| `trading/optimization/optuna_tuner.py` | Added `OptunaTuner = SharpeOptunaTuner` alias at end of file. |
| `trading/services/monitoring_tools.py` | Guards in `check_model_degradation` and `check_strategy_degradation` to skip writing when `model_id` or `strategy_name` is missing. |
| `tests/unit/test_optuna_tuner.py` | Docstrings updated to refer to SharpeOptunaTuner (OptunaTuner alias). |
