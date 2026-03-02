# Page crash fix – Performance & Trade Execution

## Compilation

- **`pages/7_Performance.py`** – `py -3.10 -m py_compile` succeeds (no syntax errors).
- **`pages/4_Trade_Execution.py`** – `py -3.10 -m py_compile` succeeds (no syntax errors).

No syntax fixes were required.

---

## Changes made

So that uncaught exceptions show in the browser instead of a white screen, the **entire module-level execution** in each page was wrapped in a top-level `try`/`except` that reports the full traceback via Streamlit.

### 1. `pages/7_Performance.py`

- **`import traceback`** added with the other standard-library imports.
- **Module-level code** (from `# Page config` / `st.set_page_config(...)` through the end of the script, including the existing `render_page_assistant` try/except) is now inside:
  - `try:`
  - …all existing page code (indented one level)…
  - `except Exception:`
  - `st.error(traceback.format_exc())`

So any uncaught exception during the page run is shown in the app with `st.error(traceback.format_exc())` instead of causing a white screen.

### 2. `pages/4_Trade_Execution.py`

- **`import traceback`** added with the other imports.
- **Module-level code** (from `st.set_page_config(...)` through the end of the script, including the existing `render_page_assistant` try/except) is wrapped in the same way:
  - `try:`
  - …all existing page code (indented one level)…
  - `except Exception:`
  - `st.error(traceback.format_exc())`

Same behavior: full traceback in the UI on any uncaught exception.

---

## Summary

| Item | Result |
|------|--------|
| Syntax (7_Performance.py) | Clean; no changes needed |
| Syntax (4_Trade_Execution.py) | Clean; no changes needed |
| Crash handling (both pages) | Top-level try/except added; errors shown with `st.error(traceback.format_exc())` |

If either page raises an exception that isn’t caught internally, the user will see the full traceback in a red Streamlit error box instead of a blank screen.
