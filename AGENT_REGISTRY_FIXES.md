# Agent Registry Fixes

Summary of changes to the agent registry and Admin display so the registry reflects the actual active agents from AGENT_ARCHITECTURE.md and phantom/stale entries are clearly marked.

---

## (1) Clean up `data/agent_registry.json`

**Removed:** The three phantom entries that had no corresponding implementation in the live codebase:

- **ModelSelectorAgent**
- **OptimizerAgent**
- **RiskAnalyzerAgent**

(Also removed all other legacy entries; the file was replaced with a single source of truth containing only the active agents.)

**Added:** Exactly the **9 active agents** defined in AGENT_ARCHITECTURE.md, each with:

- **name** — Agent class name (e.g. `PromptAgent`).
- **type** — Short label for the UI (e.g. Chat, Router, Execution).
- **status** — `active` (or `available`); used by Admin and for validation.
- **description** — One-sentence role from AGENT_ARCHITECTURE.md.
- **location** — File path of the class, relative to project root (used for “file exists” validation).

Minimal fields required by `trading.agents.agent_registry.AgentInfo` were also added so `AgentRegistry._load_registry()` still works: `class_name`, `module_path`, `capabilities`, `dependencies`, `category`, `version`, `author`, `tags`, `config_schema`, `created_at`, `updated_at`.

**Final 9 entries:**

| Agent | Type | Location |
|-------|------|----------|
| PromptAgent | Chat | `agents/llm/agent.py` |
| EnhancedPromptRouterAgent | Router | `trading/agents/enhanced_prompt_router.py` |
| ModelBuilderAgent | Model Builder | `trading/agents/model_builder_agent.py` |
| PerformanceCriticAgent | Critic | `trading/agents/performance_critic_agent.py` |
| ExecutionAgent | Execution | `trading/agents/execution/execution_agent.py` |
| CommentaryAgent | Commentary | `trading/agents/commentary_agent.py` |
| UpdaterAgent | Updater | `trading/agents/updater_agent.py` |
| AgentLeaderboard | Utility | `trading/agents/agent_leaderboard.py` |
| MockAgent | Fallback | `agents/mock_agent.py` |

---

## (2) Admin Agent Registry display (`pages/11_Admin.py`)

**Load from file:** The Agent Registry table no longer uses a hardcoded dict. When `st.session_state.agent_registry` is empty, the Admin page now:

1. Loads **`data/agent_registry.json`** from the project root (`Path(__file__).resolve().parent.parent / "data" / "agent_registry.json"`).
2. Reads the `agents` object and uses it as the in-memory registry for the tab.
3. Ensures each agent has defaults for table columns: `last_run`, `performance_score`, `enabled`, `configuration`, `execution_history`, `performance_metrics`.

**Location validation:** After loading (and on each render), each entry is validated:

- If the agent has a **`location`** path, the code checks whether that file exists under the project root.
- If the file **does not exist**, the agent’s **`status`** is set to **`orphaned`**.
- So phantom or stale entries (e.g. old paths or removed agents) are always shown as orphaned.

**Display:**

- **Status filter:** The “Filter by Status” dropdown now includes **`orphaned`** in addition to `All`, `active`, `paused`, `error`.
- **Status text:** The table status column uses:
  - **active** → “Active”
  - **paused** → “Paused”
  - **error** → “Error”
  - **orphaned** → “Orphaned”  
  Orphaned and error both use the same red marker so phantom/stale entries are visually distinct from real, active agents.

**Imports:** `json` was added to the Admin page imports for loading the registry file.

---

## (3) `trading/agents/agent_registry.py` compatibility

The new `data/agent_registry.json` includes extra keys used only by the Admin UI (`type`, `location`). `AgentInfo.from_dict()` only accepts fields that exist on the `AgentInfo` dataclass, so passing the raw agent dict would raise.

**Change:** In `_load_registry()`, before calling `AgentInfo.from_dict(agent_data)`, the code now builds a dict that contains only the known `AgentInfo` keys. Any other keys (e.g. `type`, `location`) are ignored when constructing `AgentInfo`. This keeps the single JSON file valid for both:

- Admin (which uses `type`, `location`, and status for display and validation).
- `AgentRegistry` (which uses `AgentInfo` for discovery and registration).

---

## Verification

- **Registry file:** `data/agent_registry.json` contains only the 9 agents listed above; the three phantom agents (ModelSelectorAgent, OptimizerAgent, RiskAnalyzerAgent) are removed.
- **Admin:** Opening the AI Agents tab loads the registry from the file, validates `location` for each agent, and shows “Orphaned” in red for any entry whose file is missing; “Active” remains for entries whose file exists.
- **AgentRegistry:** Loading the same file in `trading.agents.agent_registry` no longer raises; extra keys are stripped before `from_dict`.

No trading or execution logic was changed; only registry data, Admin display, and registry loading were updated.
