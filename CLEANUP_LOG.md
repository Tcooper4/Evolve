# Cleanup Log — _dead_code Removal (March 2026)

## Summary

Before deleting `_dead_code/`, valuable reference code was extracted to `docs/future_features/`. The entire `_dead_code/` directory was then removed.

---

## STEP 1 — Create docs/future_features/

**Done.** Directory `docs/future_features/` was created.

---

## STEP 2 — Copy files with prefix note

The following files were copied into `docs/future_features/` with this header added at the top of each file:

```
# FUTURE FEATURE — Not active. Preserved for reference.
# See docs/future_features/README.md for context.
```

| Source | Destination |
|--------|-------------|
| _dead_code/agents/multimodal_agent.py | docs/future_features/multimodal_agent.py |
| _dead_code/agents/prompt_clarification_agent.py | docs/future_features/prompt_clarification_agent.py |
| _dead_code/agents/strategy_research_agent.py | docs/future_features/strategy_research_agent.py |
| _dead_code/agents/meta_learner.py | docs/future_features/meta_learner.py |
| _dead_code/agents/self_tuning_optimizer_agent.py | docs/future_features/self_tuning_optimizer_agent.py |
| _dead_code/fallback/fallback_llm.py | docs/future_features/fallback_llm_negation.py |
| _dead_code/pages_archive/HybridModel.py | docs/future_features/hybrid_model_weight_ui.py |

---

## STEP 3 — Create docs/future_features/README.md

**Done.** README.md was created with the feature table and priority notes as specified.

---

## STEP 4 — Delete _dead_code/

**Done.** The entire `_dead_code/` directory was removed with:

```powershell
Remove-Item -Recurse -Force _dead_code
```

Removed contents included:
- _dead_code/agents/ (all agent modules, updater, optimization, llm)
- _dead_code/archive/ (legacy_tests)
- _dead_code/examples/ (all example scripts and README)
- _dead_code/fallback/ (all fallback implementations)
- _dead_code/interface/ (unified_interface.py)
- _dead_code/pages_archive/ (including pre_streamline_20250101)

---

## STEP 5 — Confirm deletion and list docs/future_features/

**Confirmed.** `_dead_code/` no longer exists. Contents of `docs/future_features/`:

| File | Size (bytes) |
|------|--------------|
| fallback_llm_negation.py | 19,624 |
| hybrid_model_weight_ui.py | 18,170 |
| meta_learner.py | 16,437 |
| multimodal_agent.py | 27,254 |
| prompt_clarification_agent.py | 13,567 |
| README.md | 885 |
| self_tuning_optimizer_agent.py | 34,869 |
| strategy_research_agent.py | 42,842 |

**Total:** 7 Python reference files + 1 README. No integration into the active codebase; reference only.

---

## Strategy Testing page — ensemble weight history check

The active Strategy Testing page (`pages/3_Strategy_Testing.py`) uses `WeightedEnsembleStrategy` and `EnsembleConfig` but does **not** include ensemble weight history visualization or real-time composition display. Therefore `HybridModel.py` was preserved as `docs/future_features/hybrid_model_weight_ui.py` for future reference when adding weight history / composition UI.

---

# Orphan File Cleanup (March 2026)

## Summary

Safe cleanup of orphaned files only: no import path changes, no consolidation. Only files with zero references elsewhere were deleted.

---

## STEP 1 — _dead_code/

Skipped (already deleted in previous cleanup).

---

## STEP 2 — trading/agents/ and trading/nlp/

For each file, `grep -rn "filename_stem" --include="*.py" .` was run (excluding the file itself). Deleted only when result was zero.

| File | Action | Reason |
|------|--------|--------|
| trading/agents/launch_execution_agent.py | **Deleted** | Zero imports elsewhere |
| trading/agents/launch_leaderboard_dashboard.py | **Deleted** | Zero imports elsewhere |
| trading/agents/demo_risk_controls.py | **Deleted** | Zero imports (only self-refs in file) |
| trading/agents/demo_leaderboard.py | **Deleted** | Zero imports (only self-refs in file) |
| trading/agents/demo_pluggable_agents.py | **Deleted** | Zero imports elsewhere |
| trading/agents/test_integration.py | **Kept** | References: tests/__init__.py lists "test_integration"; other test_integration matches |
| trading/agents/test_execution_agent.py | **Kept** | Reference: trading/services/test_service_integration.py contains "test_execution_agent" |
| trading/nlp/sandbox_nlp.py | **Deleted** | Zero imports (only self-ref in comment) |

---

## STEP 3 — tests/ audit and one-off scripts

Verified zero imports for each; only deleted when no other file imported.

| File | Action | Reason |
|------|--------|--------|
| tests/dump_model_forecasts.py | **Deleted** | No imports found |
| tests/comprehensive_audit.py | Kept | Imported in tests/__init__.py |
| tests/comprehensive_return_fix.py | Kept | Imported in tests/__init__.py |
| tests/quick_fix.py | Kept | Imported in tests/__init__.py |
| tests/simple_audit.py | Kept | Imported in tests/__init__.py |
| tests/targeted_audit.py | Kept | Imported in tests/__init__.py |
| tests/audit_return_statements.py | Kept | Imported in tests/__init__.py |
| tests/check_system.py | Kept | Imported in tests/__init__.py |
| fix_*.py, post_upgrade_*.py, demo_*.py in tests/ | N/A | No fix_*.py found; post_upgrade_* and demo_* are imported (tests/__init__.py, test_return_statements.py) — not deleted |

---

## STEP 4 — scripts/ directory

`grep -rn "from scripts\|import scripts" --include="*.py" .` was run.

**Result: refs exist — scripts/ NOT deleted.**

References found:
- tests/unit/test_pandera_migration.py: `from scripts.manage_data_quality import DataQualityManager`
- tests/test_enhancements.py: `from scripts.run_live_dashboard import DashboardRunner`
- tests/test_signal_collector_and_kube_deploy.py: `from scripts.deploy_to_kube_batch import ...` and `from scripts.deploy_to_kube_batch import main`

---

## STEP 5 — Compile check and smoke test

### Compile

```powershell
py -3.10 -m py_compile app.py pages/0_Home.py pages/1_Chat.py pages/2_Forecasting.py
```

**Exit code: 0** — all compiled successfully.

### Smoke test

```powershell
py -3.10 tests/model_smoke_test.py
```

Smoke test was run; it loads the app and runs model checks (XGBoost, ARIMA, etc.). Run exceeded 60s timeout in the session; captured output showed no failures (INFO/WARNING only, models initializing and training). No syntax or import errors. To confirm full pass, run locally:

```powershell
py -3.10 tests/model_smoke_test.py
```
