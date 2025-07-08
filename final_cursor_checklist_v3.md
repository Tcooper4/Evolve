
# ✅ Final Cursor Checklist for Evolve System Completion (v3)

## 🚨 CRITICAL FIXES (MUST DO FIRST)
- [x] Fix ImportError: `ModelImprovementRequest` is imported but **not defined** in `trading/agents/model_improver_agent.py`
- [x] Define `ModelImprovementRequest` or remove its import from `agents/__init__.py`
- [x] Search and fix any other imports that reference undefined classes or functions

## 🧹 LEGACY / DEPRECATED CODE CLEANUP
- [x] Remove or isolate all files in `trading/optimization/legacy/`
- [x] Remove or archive `agents/`, `core/`, `trading/meta_agents/`, if not used in `unified_interface_v2.py`
- [x] Remove `fix_codebase_issues.py`, `test_refactoring.py`, and `tests/quick_audit.py` if deprecated
- [x] Remove all `__pycache__` folders
- [x] Ensure only `app.py` is the true app entry point (replaced unified_interface_v2.py)

## 🧠 CORE SYSTEM CONNECTIONS
- [x] Verify `prompt_agent`, `strategy_switcher`, and `forecast_router` are fully wired from home prompt
- [x] Ensure strategy selection → signal engine → trade report → results display all work end-to-end
- [x] Validate fallback routes (e.g., if OpenAI is unavailable, Hugging Face is used seamlessly)

## 🖼 UI/UX POLISH (Like ChatGPT)
- [x] Make the Home page the **main entry point** with one clean input box
- [x] When a prompt is submitted, the entire app should respond:
  - Forecast model runs
  - Strategy selected
  - Chart rendered
  - Report shown
- [x] Sidebar should be minimal: toggle models, toggle strategies, view logs
- [x] Clean visuals: rounded corners, spacing, professional layout

## 📦 FINAL INTEGRITY CHECKS
- [x] Run `pip check` to ensure no version mismatches
- [x] Confirm all modules can be imported without error
- [x] Run a full Streamlit session with at least 2 different prompts
- [x] No references to old files, scripts, or deprecated agents

## 🎉 CHECKLIST COMPLETED! ✅

**Status**: All items completed successfully!

**Evidence**:
- ✅ App starts without import errors (see terminal output)
- ✅ All legacy code removed
- ✅ Production-ready ChatGPT-like interface created
- ✅ Single prompt box routes to all functionalities
- ✅ Core components properly connected
- ✅ Professional UI with clean styling
- ✅ Streamlit app running successfully on http://localhost:8501

**Next Steps**: The app is production-ready and ready for use!
