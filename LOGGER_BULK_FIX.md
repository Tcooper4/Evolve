# Logger ordering bulk fix

For each file, `logger = logging.getLogger(__name__)` was moved to be the **first line after all import statements**, before any other code. Only files that had that exact line at module level were modified.

---

## 1. memory/logger_utils.py

- **Before:** No module-level `logger = logging.getLogger(__name__)` (only `logger = UnifiedLogger()` at line 311 and `logger = logging.getLogger(name)` inside methods).
- **After:** No change.
- **py_compile:** OK (exit 0).

---

## 2. nlp/natural_language_insights.py

- **Before:** `logger = logging.getLogger(__name__)` was on **line 56** (after try/except blocks that already used `logger`, causing a use-before-define bug).
- **After:** `logger = logging.getLogger(__name__)` is on **line 16**, immediately after `from typing import Any, Dict, List, Optional` (line 15).
- **py_compile:** OK (exit 0).

---

## 3. trading/evaluation/model_evaluator.py

- **Before:** `logger = logging.getLogger(__name__)` was on **line 53** (after try/except blocks that already used `logger`).
- **After:** `logger = logging.getLogger(__name__)` is on **line 17**, immediately after `import pandas as pd` (line 16).
- **py_compile:** OK (exit 0).

---

## 4. trading/models/advanced/transformer/time_series_transformer.py

- **Before:** `logger = logging.getLogger(__name__)` was on **line 52** (after multiple try/except blocks that already used `logger`).
- **After:** `logger = logging.getLogger(__name__)` is on **line 7**, immediately after `from typing import Any, Dict, Optional, Tuple` (line 6), so it is defined before any try/except that uses it.
- **py_compile:** OK (exit 0).

---

## 5. trading/risk/risk_logger.py

- **Before:** `logger = logging.getLogger(__name__)` was on **line 21** (after the `# Configure logging` block and `logging.basicConfig(...)`).
- **After:** `logger = logging.getLogger(__name__)` is on **line 12**, immediately after `from .risk_metrics import calculate_advanced_metrics, calculate_rolling_metrics` (line 11). The `# Configure logging` / `logging.basicConfig(...)` block follows.
- **py_compile:** OK (exit 0).

---

## 6. trading/services/launch_agent_api.py

- **Before:** No module-level `logger = logging.getLogger(__name__)` (only `root_logger` and `api_logger` inside code).
- **After:** No change.
- **py_compile:** OK (exit 0).

---

## 7. trading/utils/error_logger.py

- **Before:** No module-level `logger = logging.getLogger(__name__)` (only `self.logger = logging.getLogger("trading_errors")` inside the class).
- **After:** No change.
- **py_compile:** OK (exit 0).

---

## 8. trading/utils/memory_logger.py

- **Before:** No module-level `logger = logging.getLogger(__name__)` (only `self.logger = logging.getLogger(__name__)` inside the class).
- **After:** No change.
- **py_compile:** OK (exit 0).

---

## 9. trading/utils/monitor.py

- **Before:** No module-level `logger = logging.getLogger(__name__)` (only `self.logger = logging.getLogger(...)` inside the class).
- **After:** No change.
- **py_compile:** OK (exit 0).

---

## Summary table

| File | Before (line) | After (line) | Changed? |
|------|---------------|--------------|----------|
| memory/logger_utils.py | — | — | No |
| nlp/natural_language_insights.py | 56 | 16 | Yes |
| trading/evaluation/model_evaluator.py | 53 | 17 | Yes |
| trading/models/advanced/transformer/time_series_transformer.py | 52 | 7 | Yes |
| trading/risk/risk_logger.py | 21 | 12 | Yes |
| trading/services/launch_agent_api.py | — | — | No |
| trading/utils/error_logger.py | — | — | No |
| trading/utils/memory_logger.py | — | — | No |
| trading/utils/monitor.py | — | — | No |

All 9 files compile with `py -3.10 -m py_compile` with zero errors.
