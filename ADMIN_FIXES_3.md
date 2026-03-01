# Admin and Home Briefing Fixes (3)

Three small bugs fixed in `pages/11_Admin.py` and one debug addition in `trading/services/home_briefing_service.py`.

---

## 1. `logger` not defined (Admin.py)

**Problem:** `name 'logger' is not defined` at lines ~1657 and ~1719. The sys.modules cache-clearing block and the LLM import block run at module level; if any code path used `logger` before it was bound, it would raise.

**Fix:** Moved the logger definition to the very top of the file, immediately after the stdlib imports (`logging`, `sys`, `pathlib`, `datetime`, `typing`) and before any other code.

- Added `logger = logging.getLogger(__name__)` right after the `from typing import ...` line.
- Removed the duplicate `logger = logging.getLogger(__name__)` that was after `logging.basicConfig(level=logging.INFO)` (kept `logging.basicConfig`).

**File:** `pages/11_Admin.py`

---

## 2. StreamlitAPIException: `_repr_html_()` is not a valid Streamlit command (Admin.py)

**Problem:** At line 1901, `st.success("✅")` caused `StreamlitAPIException: _repr_html_() is not a valid Streamlit command`. The emoji string was being interpreted as an HTML object.

**Fix:** In the API Keys section, replaced status-indicator calls so they are not treated as Streamlit commands that expect specific types:

- All `st.success("✅")` in the API Keys section → `st.markdown("✅")`.
- All `st.caption("—")` used for status indicators in the API Keys section → `st.markdown("—")`.

**Locations updated:**

- AI Provider Keys: status column `st.success("✅")` / `st.caption("—")` → `st.markdown("✅")` / `st.markdown("—")`.
- Trading keys (Alpaca): `st.success("✅") if is_set else st.caption("—")` → `st.markdown("✅") if is_set else st.markdown("—")`.

**File:** `pages/11_Admin.py`

---

## 3. Debug log for briefing LLM (home_briefing_service.py)

**Purpose:** Confirm in startup/logs which provider and model are used for the Home briefing when `get_active_llm()` is used correctly.

**Fix:** Added a debug log line immediately before the `call_active_llm_chat()` call:

```python
logger.debug("Calling LLM with provider: %s, model: %s", provider, _model)
```

`provider` and `_model` are already in scope from the earlier `get_active_llm()` call in the same function. No other logic changes.

**File:** `trading/services/home_briefing_service.py`

---

## Summary

| Item | File | Change |
|------|------|--------|
| 1. Logger defined early | `pages/11_Admin.py` | `logger = logging.getLogger(__name__)` moved to top after stdlib imports; duplicate removed. |
| 2. API Keys status widgets | `pages/11_Admin.py` | In API Keys section: `st.success("✅")` → `st.markdown("✅")`, `st.caption("—")` → `st.markdown("—")`. |
| 3. Briefing LLM debug log | `trading/services/home_briefing_service.py` | `logger.debug("Calling LLM with provider: %s, model: %s", provider, _model)` before `call_active_llm_chat()`. |

No other changes were made.
