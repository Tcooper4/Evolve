# Admin Page Audit and Fixes (pages/11_Admin.py)

## 1. AUDIT: Imports, Tabs, Sections, Buttons, Dependencies

### Top-level / early imports

| Section | Dependency | Import path | Exists? | Issue |
|--------|------------|-------------|---------|--------|
| (global) | logging, sys, pathlib, datetime, typing | stdlib | Yes | — |
| (global) | pandas, plotly.graph_objects, streamlit | third-party | Yes | — |
| Backend | EnhancedSettings | trading.config.enhanced_settings | Yes | Optional; can fail at module load if env validation fails → placeholder used |
| Backend | AgentRegistry | trading.agents.agent_registry | Yes | Optional; placeholder if ImportError |
| Backend | SystemHealthMonitor | trading.monitoring.health_check (else monitoring.health_check) | Yes (trading) | Optional; placeholder if ImportError |
| Backend | SystemStatus | trading.utils.system_status | Yes | Optional; placeholder if ImportError |

### Config package (used by app and LLM section)

| Section | Dependency | Import path | Exists? | Issue |
|--------|------------|-------------|---------|--------|
| config package | MarketAnalysisConfig | config.market_analysis_config | **No (was missing)** | config/__init__.py imported it; only .yaml existed → **Fixed** with config/market_analysis_config.py |
| Config tab | get_active_llm, set_active_llm, etc. | config.llm_config | Yes | Resolved only when config package loads; **Fixed** by adding market_analysis_config and optional retry after clearing sys.modules |
| Config tab | get_provider_status, test_active_llm | agents.llm.active_llm_calls | Yes | Depends on config.llm_config; works after config fix |

### Per-tab / inline imports

| Section | Dependency | Import path | Exists? | Issue |
|--------|------------|-------------|---------|--------|
| API / WebSocket | WebSocketClient | utils.websocket_client | Yes | Wrapped in try/except; fallback if ImportError |
| AI Assistant | LLMProcessor | trading.nlp.llm_processor | Yes | Wrapped in try/except; **Updated** to show "Feature not available" on ImportError |
| Task Orchestrator | TaskOrchestrator, TaskScheduler, TaskMonitor, task_models | core.orchestrator.* | Yes | Wrapped in try/except; ORCHESTRATOR_AVAILABLE = False on ImportError |
| Logs/Debug | psutil, numpy, gc, plotly.express | third-party / stdlib | Yes | Optional / local imports |

### Tabs and primary actions

| Tab | Primary action | Backend | Status |
|-----|----------------|--------|--------|
| 1. System Dashboard | Health gauge, metrics, automation workflows | session_state, health_monitor, automation_core (session) | Uses placeholders if backends missing |
| 2. Configuration | Save system_config, **AI Model Settings** (Save / Test Connection) | st.session_state.system_config, config.llm_config, MemoryStore | **Fixed**: LLM section loads; Save/Test use set_active_llm and test_active_llm |
| 3. AI Agents | List agents, enable/disable | AgentRegistry (optional) | Placeholder if registry missing |
| 4. System Monitoring | Metrics, WebSocket, AI Assistant | SystemStatus, WebSocketClient, LLMProcessor | All optional with fallbacks |
| 5. Logs & Debugging | View logs, Clear cache, Reset session, Force GC, Test DB | session_state | — |
| 6. Maintenance | Backup DB, Optimize DB, **Run self-test** | run_admin_self_test() | **Added** self-test expander and button |

---

## 2. FIX — config.market_analysis_config

- **Cause:** `config/__init__.py` did `from .market_analysis_config import MarketAnalysisConfig` but only `config/market_analysis_config.yaml` existed; no `.py` module.
- **Change:** Added **config/market_analysis_config.py** that:
  - Defines `MarketAnalysisConfig` and loads `config/market_analysis_config.yaml` via `yaml.safe_load`.
  - Exposes `get()`, `market_conditions`, `analysis_settings`, `visualization_settings`, `pipeline_settings`, `to_dict()`.
- **Resilience:** In **config/__init__.py**, the import is wrapped in `try/except ImportError`; on failure `MarketAnalysisConfig = None` so the rest of the config package still loads.

---

## 3. FIX — LLM selector in Admin

- **Cause:** The Configuration tab’s “AI Model Settings” block failed because:
  1. The **config** package failed to load (missing `config.market_analysis_config`), so `from config.llm_config import ...` never succeeded.
  2. In some runs, **config** could resolve to **trading.config** (wrong package), so `config.llm_config` did not exist.
- **Change:**
  1. **config.market_analysis_config** added (see above), so `config` loads and `config.llm_config` is importable.
  2. In **pages/11_Admin.py**, the LLM block now:
     - Ensures project root is on `sys.path`.
     - Tries `from config.llm_config import ...` and `from agents.llm.active_llm_calls import ...` first.
     - On `ImportError` or `ModuleNotFoundError`, clears all `config` and `config.*` entries from `sys.modules` and retries once, so the next import uses the project-root **config**.
- **Result:** Provider dropdown, model input, status indicator, **Save**, and **Test Connection** work when Admin is opened, using `get_active_llm()`, `set_active_llm()`, `get_provider_status()`, `test_active_llm()`, and MemoryStore.

---

## 4. FIX — Other broken imports and fallbacks

- **config.market_analysis_config:** Fixed by new module and optional import in config/__init__.py.
- **EnhancedSettings, AgentRegistry, SystemHealthMonitor, SystemStatus:** Already behind try/except with placeholders; no change.
- **LLMProcessor (AI Assistant):** ImportError message updated to: “Feature not available. LLM Processor (trading.nlp.llm_processor) is not available.”
- **WebSocketClient, core.orchestrator:** Already behind try/except; no change.
- No additional bare imports that could crash the page; remaining optional imports are already wrapped.

---

## 5. FIX — Admin tabs functional check

- **Tab 2 (Configuration):** System config is stored in `st.session_state.system_config`; persistence is session-scoped. AI Model Settings now load and **Save** / **Test Connection** call `set_active_llm()` and `test_active_llm()`; persistence is via MemoryStore (preference `active_llm`).
- **Tab 1 (Dashboard):** Uses session_state and optional health_monitor; no broken backend.
- **Tabs 3–6:** Use optional backends with fallbacks; primary actions (e.g. agent list, orchestrator, logs, maintenance) either work when backends exist or show “Feature not available” / warnings.

---

## 6. FIX — EnhancedSettings

- **Observation:** Startup log “EnhancedSettings not found, using placeholder” occurs when `from trading.config.enhanced_settings import EnhancedSettings` fails. The module exists; failure is usually due to **trading.config.enhanced_settings** running `create_enhanced_settings()` at import time, which validates env vars (e.g. API key patterns) and can raise.
- **Current behavior:** Admin already uses a try/except and sets `EnhancedSettings = None` and `settings_manager = None`, so the page does not crash.
- **No code change:** Keeping the optional import and placeholder is sufficient; no stub or app_config.yaml persistence was added for EnhancedSettings in this pass.

---

## 7. Self-test and debug button

- **Added** `run_admin_self_test()` in **pages/11_Admin.py** (after backend init). It:
  1. Ensures project root is on `sys.path`.
  2. Checks **config.llm_config**: imports `get_active_llm`, gets current provider/model.
  3. Checks **MemoryStore**: `get_memory_store()`, `get_preference("active_llm")`.
  4. Checks **get_provider_status**: calls `get_provider_status("claude")`.
  5. Checks **test_active_llm**: calls `test_active_llm()` (may fail if no API key).
- **Return value:** Dict of check name → `{ "ok": bool, "message": str }`.
- **UI:** In **Maintenance** tab (tab6), added an expander **“🧪 Run self-test”** with a **“Run self-test”** button that calls `run_admin_self_test()` and displays results with `st.success` / `st.error` and `st.json()`.

---

## Summary of file changes

| File | Change |
|------|--------|
| **config/market_analysis_config.py** | **New.** Defines `MarketAnalysisConfig` and loads `market_analysis_config.yaml`. |
| **config/__init__.py** | Optional import of `MarketAnalysisConfig`; on ImportError set to `None`. |
| **pages/11_Admin.py** | LLM block: try import first, on failure clear `config` from `sys.modules` and retry. Added `run_admin_self_test()`. Added “Run self-test” expander and button in Maintenance tab. AI Assistant ImportError message set to “Feature not available …”. |

---

## How to verify

1. **Start app:** `streamlit run app.py`.
2. **Open Admin** (e.g. sidebar → Admin).
3. **Configuration tab:** Confirm “🤖 AI Model Settings” shows provider dropdown, model field, status, **Save**, **Test Connection**; choose a provider, click Save, then Test Connection.
4. **Maintenance tab:** Open “🧪 Run self-test”, click “Run self-test”; confirm all checks (or expected failures, e.g. test_active_llm without key) and JSON output.
