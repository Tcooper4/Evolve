# Admin Fixes (4) — pages/11_Admin.py

Two bugs fixed; only `pages/11_Admin.py` was changed. No trading logic or other files modified.

---

## (1) Logger defined after all imports

**Issue:** `logger` could be used before it was defined (e.g. at ~1720) if execution order or imports ever ran code that referenced `logger` before the line where it was created.

**Change:** `logger = logging.getLogger(__name__)` was moved so it is the **first non-import line** after **all** imports.

- **Removed** `logger = logging.getLogger(__name__)` from line 27 (previously right after the stdlib imports).
- **Inserted** `logger = logging.getLogger(__name__)` immediately after `import streamlit as st` (now line 41), before `# Configure logging` and any other code.

**Result:** Every use of `logger` in the file (lines 50, 56, 65, 71, 131, 140, 149, 969, 1022, 1033, 1162, 1720, 3158, 3862) is after the definition at line 41. No `logger` reference appears before line 41.

---

## (2) StreamlitAPIException from ternary with non-bool `is_set`

**Issue:** At ~1902, `st.markdown("✅") if is_set else st.markdown("—")` could raise `StreamlitAPIException: _repr_html_() is not a valid Streamlit command` because the ternary condition might not be a plain `bool`, and evaluating it could trigger `_repr_html_()` (e.g. from a pandas or env result).

**Changes:**

1. **AI Provider Keys loop (env_var, label, provider in _ai_keys)**  
   - **Line ~1853–1856:** `if is_set:` → `if bool(is_set):` so the condition is always a plain bool.  
   - **Line ~1862:** `placeholder="•••••••• (set)" if is_set else "Enter key..."` → `placeholder="•••••••• (set)" if bool(is_set) else "Enter key..."`.  
   - **Line ~1851:** `is_set` was already set with `bool(os.getenv(env_var) or (...))`; left as is.

2. **Trading keys loop (env_var, label in _trading_keys)**  
   - **Line ~1900:** `st.markdown("✅") if is_set else st.markdown("—")` → `st.markdown("✅" if bool(is_set) else "—")` so a single string is passed to `st.markdown` and the condition is a plain bool.  
   - **Line ~1907:** `placeholder="•••••••• (set)" if is_set else "Enter key..."` → `placeholder="•••••••• (set)" if bool(is_set) else "Enter key..."`.  
   - **Line ~1899:** `is_set = bool(os.getenv(env_var))` was already a bool; left as is.

**Result:** All uses of `is_set` in Streamlit calls or placeholders now go through `bool(is_set)` or use an already bool-assigned `is_set`. No remaining ternary of the form `st.xxx(...) if is_set else st.xxx(...)` that could receive a non-bool.

---

## Summary of line-level changes

| Location | Before | After |
|----------|--------|--------|
| After `from typing import ...` | `logger = logging.getLogger(__name__)` + blank + `# Ensure project root` | `# Ensure project root` (logger removed here) |
| After `import streamlit as st` | `# Configure logging` | `logger = logging.getLogger(__name__)` + `# Configure logging` |
| AI Provider Keys, col_status | `if is_set:` | `if bool(is_set):` |
| AI Provider Keys, placeholder | `... if is_set else "Enter key..."` | `... if bool(is_set) else "Enter key..."` |
| Trading keys, col_status | `st.markdown("✅") if is_set else st.markdown("—")` | `st.markdown("✅" if bool(is_set) else "—")` |
| Trading keys, placeholder | `... if is_set else "Enter key..."` | `... if bool(is_set) else "Enter key..."` |

---

## Verification

- Grep over the file: `logger` is defined at line 41; every `logger.warning`, `logger.error`, and `logger.debug` is on a line number &gt; 41.  
- Grep for `st.(success|markdown|caption)(...) if ... else`: no remaining ternary Streamlit calls using a raw condition; all relevant conditionals use `bool(is_set)` or a single `st.markdown("✅" if bool(is_set) else "—")`.
