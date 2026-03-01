# Admin Page Runtime Fixes (ADMIN_FIXES_2)

This document describes fixes for four specific runtime errors that appeared in Admin page logs. No trading logic was changed.

---

## (1) NameError: name 'Any' is not defined (pages/11_Admin.py line 147)

**Cause:** The return type of `run_admin_self_test()` was `Dict[str, Dict[str, Any]]`, but `Any` was not imported from `typing`. Only `Dict, List, Optional` were imported.

**Fix:** Updated the top-level typing import to include `Any` and `Tuple`:

```python
# Before
from typing import Dict, List, Optional

# After
from typing import Any, Dict, List, Optional, Tuple
```

**File:** `pages/11_Admin.py` (line 25).

---

## (2) No module named 'config.market_analysis_config'

**Cause:** The `config` package was cached in `sys.modules` from an earlier import (e.g. when the app or another page loaded). That cached package was created before `config/market_analysis_config.py` existed, so subsequent imports of `config` or `config.llm_config` still saw the old broken package.

**Fix:**

1. **Force clean reload of config in Admin**  
   In `pages/11_Admin.py`, immediately after ensuring the project root is on `sys.path`, clear any cached `config` package so the next import loads the current code (including `config.market_analysis_config`):

   ```python
   import sys
   # Force reload of config package to pick up new market_analysis_config module
   for key in list(sys.modules.keys()):
       if key == "config" or key.startswith("config."):
           del sys.modules[key]
   ```

   This block runs before any Admin code imports `config` or `config.llm_config`.

2. **Module and syntax**  
   `config/market_analysis_config.py` already exists and defines `MarketAnalysisConfig`; it was verified with:

   ```bash
   python -c "from config.market_analysis_config import MarketAnalysisConfig; print('OK')"
   ```

   No syntax changes were required.

**Files:** `pages/11_Admin.py` (early block after path setup). `config/market_analysis_config.py` unchanged.

---

## (3) No module named 'agents.model_generator_agent'

**Cause:** During agent rationalization, `agents.model_generator_agent` (and related agent modules) were moved to `_dead_code`. The top-level `agents/__init__.py` still imported them unconditionally, so loading the `agents` package (e.g. when Admin or Chat used `from agents.llm.active_llm_calls import ...`) failed and the LLM interface fell back to a stub.

**Fix:** In `agents/__init__.py`, all imports of removed/relocated agents were wrapped in `try/except ImportError` and the names are set to `None` when the module is missing:

- `from .model_generator_agent import ...` → try/except; on ImportError, set `ArxivResearchFetcher`, `AutoEvolutionaryModelGenerator`, `BenchmarkResult`, `ModelBenchmarker`, `ModelCandidate`, `MIGenerator`, `ResearchPaper`, `run_model_evolution` to `None`.
- `from .model_innovation_agent import ...` → try/except; on ImportError, set `InnovationConfig`, `InnovationModelCandidate`, `ModelEvaluation`, `ModelInnovationAgent`, `create_model_innovation_agent` to `None`.
- `from .prompt_agent import ...` → try/except; on ImportError, set `PromptAgent`, `create_prompt_agent` to `None`.
- `from .strategy_research_agent import ...` → try/except; on ImportError, set `StrategyResearchAgent` to `None`.

The registry and `agents.llm` do not depend on these legacy agents for basic operation, so the package and LLM interface load successfully without them.

**File:** `agents/__init__.py`.

---

## (4) cannot import name 'get_fallback_provider' from 'trading.data.providers'

**Cause:** `trading/data/providers/__init__.py` listed `get_fallback_provider` in `__all__` but did not import it at module level. The function exists in `trading/data/providers/fallback_provider.py` and was only used inside `ProviderManager._initialize_providers()`. So `from trading.data.providers import get_fallback_provider` raised ImportError and the Home briefing market data fetch failed silently.

**Fix:**

1. **Export `get_fallback_provider` from the package**  
   In `trading/data/providers/__init__.py`, at top level (after importing `BaseDataProvider`, `ProviderConfig`), add:

   ```python
   try:
       from .fallback_provider import get_fallback_provider
   except ImportError:
       def get_fallback_provider():
           return None
   ```

   So `get_fallback_provider` is always defined and the name in `__all__` is valid.

2. **Home briefing service**  
   In `trading/services/home_briefing_service.py`, the market data fetch was updated to use the existing public API that returns the fallback provider when no name is given, so it works even if `get_fallback_provider` is the stub:

   - Replaced `from trading.data.providers import get_fallback_provider` and `provider = get_fallback_provider()`  
   - With `from trading.data.providers import get_data_provider` and `provider = get_data_provider()` (which returns the fallback provider when `name` is `None`).

**Files:** `trading/data/providers/__init__.py`, `trading/services/home_briefing_service.py`.

---

## Verification

After these changes, the following log lines should no longer appear on startup when opening/running the app and Admin:

- `NameError: name 'Any'` (or similar) from `pages/11_Admin.py`.
- `No module named 'config.market_analysis_config'`.
- `No module named 'agents.model_generator_agent'`.
- `cannot import name 'get_fallback_provider' from 'trading.data.providers'`.

**Summary of changed files**

| File | Change |
|------|--------|
| `pages/11_Admin.py` | Add `Any`, `Tuple` to typing import; add sys.modules config-clean block before other imports. |
| `config/market_analysis_config.py` | No change; verified present and importable. |
| `agents/__init__.py` | Wrap model_generator_agent, model_innovation_agent, prompt_agent, strategy_research_agent imports in try/except ImportError; set names to None when missing. |
| `trading/data/providers/__init__.py` | Top-level try/except import of `get_fallback_provider` from `.fallback_provider` with stub on ImportError. |
| `trading/services/home_briefing_service.py` | Use `get_data_provider()` instead of `get_fallback_provider()` for market data fetch. |

No trading logic was modified; only imports, type hints, and the choice of provider accessor were adjusted.
