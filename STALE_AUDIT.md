# Stale Reference Audit

This document records the audit of the **live** codebase (excluding `_dead_code/`) for references that pointed to modules, classes, or paths that no longer exist, and the fixes applied.

---

## Scope

- **In scope:** All live code under `agents/`, `trading/`, `pages/`, `config/`, `data/`.
- **Excluded:** `_dead_code/` (not scanned for "stale" refs; it is legacy by design).

---

## 1. `__init__.py` Exports and Imports

### 1.1 `trading/agents/__init__.py`

| Finding | Fix |
|--------|-----|
| **ExecutionAgent** was loaded via `_lazy_import(".execution_agent", "ExecutionAgent")`. The module `trading.agents.execution_agent` does not exist; the class lives in `trading/agents/execution/execution_agent.py`. | Updated to `_lazy_import(".execution.execution_agent", "ExecutionAgent")` in both `get_execution_agent()` and the `module_mapping` inside `_import_core_agents()`. |

All other `_lazy_import` targets in this file refer to modules that exist under `trading/agents/` (e.g. `model_builder_agent`, `commentary_agent`). Those that point to removed or moved agents (e.g. `prompt_router_agent`, `nlp_agent`, `multimodal_agent`) already return `None` on import failure and do not break the app.

### 1.2 `agents/__init__.py`

- Legacy agents (e.g. `model_generator_agent`, `strategy_research_agent`) are imported behind try/except; failures are acceptable.
- Direct import from `.registry` is valid; `agents/registry.py` exists and was verified.

**No code changes** were required in `agents/__init__.py`.

---

## 2. Config YAML (config/app_config.yaml and Others)

- **config/app_config.yaml:** No paths or names reference `_dead_code` or renamed modules. `fallback_agent: "commentary"` refers to CommentaryAgent, which exists. Template paths under `trading/nlp/config/` (`response_templates.json`, `viz_settings.json`, `entity_patterns.json`, `intent_patterns.json`) were verified to exist.
- **config/strategies.yaml**, **config/model_registry.yaml**, and other YAML under `config/`: Strategy and model class paths (e.g. `trading.models.lstm_model.LSTMForecaster`) align with the current layout; no stale references to moved or dead code were found.

**No config changes** were required.

---

## 3. Data JSON (data/*.json)

- **data/agent_registry.json:** Already trimmed in a prior pass. Checked again: all listed agents reference existing paths (e.g. `trading/agents/execution/execution_agent.py` for ExecutionAgent, `agents/llm/agent.py` for PromptAgent, `trading/agents/enhanced_prompt_router.py` for EnhancedPromptRouterAgent). No stale entries found.
- No other JSON files in `data/` reference class names or module paths that were audited.

**No data JSON changes** were required.

---

## 4. Imports in pages/, trading/, agents/ Referencing _dead_code or Removed Modules

- Grep for `_dead_code` and `dead_code` in live code found only **comments** or **guarded usage** (e.g. setting an agent to `None` with a comment "rationalized to _dead_code") in:
  - `trading/services/prompt_router_service.py`
  - `trading/services/multimodal_service.py`
  - `trading/services/meta_tuner_service.py`
  - `trading/validation/walk_forward_utils.py`
  - `trading/integration/institutional_grade_system.py`
  - `agents/__init__.py`
  - `scripts/trim_agent_registry.py`
- There are **no** direct imports from `_dead_code` in live pages, trading, or agents code; optional/legacy agents are either tried with try/except or set to `None` when the module is missing.

**No additional guards** were required.

---

## 5. agents/registry.py Fallbacks

| Finding | Fix |
|--------|-----|
| **PromptRouterAgent** fallback used `from trading.agents.prompt_router_agent import PromptRouterAgent`. The module `trading.agents.prompt_router_agent` does not exist; the live router is `EnhancedPromptRouterAgent` in `trading.agents.enhanced_prompt_router`. | Replaced the fallback with `from trading.agents.enhanced_prompt_router import EnhancedPromptRouterAgent` and return an instance of that class. Fallback remains inside try/except. |

- **BaseAgent** fallbacks: `trading.agents.base_agent_interface.BaseAgent` and `trading.base_agent.BaseAgent` both exist; no change.

---

## Summary of Fixes

| Location | Issue | Action |
|----------|--------|--------|
| `trading/agents/__init__.py` | `get_execution_agent()` and `_import_core_agents()` used `.execution_agent` for ExecutionAgent | Switched to `.execution.execution_agent` so the class loads from `trading/agents/execution/execution_agent.py`. |
| `agents/registry.py` | `_try_prompt_router_fallback` imported non-existent `PromptRouterAgent` | Now imports and uses `EnhancedPromptRouterAgent` from `trading.agents.enhanced_prompt_router`. |

All other checked areas (remaining `__init__.py` exports, config YAML, data JSON, and live imports) either already pointed to valid targets or were safely optional (try/except or lazy None).
