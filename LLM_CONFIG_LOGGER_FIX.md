# LLM config logger fix and Admin AI Model Settings

## 1. config/llm_config.py — logger usage and placement

### Every occurrence of `logger` in the file (after changes)

| Line | Content |
|------|--------|
| 16 | `logger = logging.getLogger(__name__)` |
| 107 | `logger.warning(f"Could not load app config for LLM, using env only: {e}")` |
| 143 | `logger.warning(f"get_active_llm failed, defaulting to Claude: {e}")` |
| 164 | `logger.error(f"set_active_llm failed: {e}")` |

### Change made

- **Placement:** `logger = logging.getLogger(__name__)` is now the **very first line after all import statements**, with no blank line between the last import and the logger assignment.
- **Before:** There was a blank line between `from typing import ...` and `logger = ...`.
- **After:** Line 15 is the last import; line 16 is `logger = logging.getLogger(__name__)`; then constants and code follow. No other code appears before the logger definition.

---

## 2. pages/11_Admin.py — AI Model Settings except block

### Change made

- **Removed:** `st.code(str(e), language="text")` from the `except Exception as e:` block for the "AI Model Settings" section (previously around line 1723).
- **Reason:** The error is already logged with `logging.warning("AI Model Settings not available: %s", e)`. Showing the exception text in the UI is redundant and can expose internal details; the error is now handled silently from the user’s perspective (only the generic `st.info` message is shown).
- **Result:** The except block now only calls `logging.warning(...)` and `st.info("Configure LLM in config/llm_config.py ...")`; it no longer displays the exception via `st.code(...)`.

---

## Summary

| File | Change |
|------|--------|
| `config/llm_config.py` | Logger definition moved to the first line after all imports (no blank line before it). |
| `pages/11_Admin.py` | Removed `st.code(str(e), language="text")` from the AI Model Settings except block; errors are logged only, not shown on screen. |
