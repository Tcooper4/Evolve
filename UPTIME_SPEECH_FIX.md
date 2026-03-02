# Uptime and Speech Fixes

This document describes two fixes applied to prevent runtime errors when the System Dashboard or Trade Execution (Chat) page load.

---

## 1. Admin System Dashboard — NameError: name 'uptime' is not defined

### Problem

In `pages/11_Admin.py`, the **System Dashboard** tab (tab1) called `_get_system_dashboard_data()` and used its return value as `dash` in some places, but in the **System Health Details** expander (around line 1659) the code used bare variables `uptime`, `agents_active`, and `health_score`. The name `uptime` was never defined (the function returns `uptime_str`), causing:

```text
NameError: name 'uptime' is not defined
```

### Solution

- **Single call at top of tab:** The tab already called `_get_system_dashboard_data()` once at the start. The return value is now stored in a variable named `dashboard_data` (renamed from `dash`) for clarity.
- **Use return value everywhere:** All references in the System Dashboard tab that depended on that data now use `dashboard_data`:
  - **Quick Stats:** `dashboard_data["uptime_str"]`, `dashboard_data["trades_today"]`, `dashboard_data["active_strategies"]`.
  - **Recent events:** `dashboard_data.get("recent_events")` and `dashboard_data["recent_events"]`.
  - **Component status:** `dashboard_data.get("agents_active", 0)` for the “AI Agents” row in the expander.
  - **System Metrics (expander):** `dashboard_data.get('uptime_str', '—')` for Uptime, `dashboard_data.get('agents_active', 0)` for Active Agents. `health_score` and `cpu_usage` are still taken from session state as before.

### Files changed

- **pages/11_Admin.py**
  - Renamed `dash` → `dashboard_data` at the top of the System Dashboard tab.
  - Replaced every use of `dash` with `dashboard_data` in that tab.
  - In the System Health Details expander, replaced bare `uptime` and `agents_active` with `dashboard_data.get('uptime_str', '—')` and `dashboard_data.get('agents_active', 0)`.

---

## 2. Trade Execution / Chat — pyttsx3 missing (page not loading)

### Problem

The Trade Execution page (and any UI that imports the chatbox agent) could fail to load if **pyttsx3** was not installed. The module was imported at top level; when the import failed, the name `pyttsx3` was never defined. Later, `TextToSpeech` called `pyttsx3.init()` and other code used `pyttsx3` without checking availability, causing `NameError` or import errors and preventing the page from loading.

### Solution

In `ui/chatbox_agent.py`:

- **Optional import with flag:** The pyttsx3 import is wrapped in a try/except, and a single availability flag is set:
  - `try: import pyttsx3; PYTTSX3_AVAILABLE = True`
  - `except ImportError: pyttsx3 = None; PYTTSX3_AVAILABLE = False`
- **Backward compatibility:** `TTS_AVAILABLE = PYTTSX3_AVAILABLE` is set so existing checks on TTS availability still work.
- **Guarded use:** Every use of pyttsx3 is guarded by `PYTTSX3_AVAILABLE` (and, where relevant, `pyttsx3 is not None` and `self.engine is not None`):
  - **TextToSpeech.__init__:** If `not PYTTSX3_AVAILABLE or pyttsx3 is None`, the initializer returns early and sets `self.engine = None`. Otherwise it calls `pyttsx3.init()` inside a try/except and sets `self.engine = None` on failure.
  - **TextToSpeech.speak:** If `not PYTTSX3_AVAILABLE or pyttsx3 is None or self.engine is None`, the method returns without speaking.
  - **ChatboxAgent:** TTS is only created when both `enable_tts` and `PYTTSX3_AVAILABLE` are true: `self.tts = TextToSpeech() if (enable_tts and PYTTSX3_AVAILABLE) else None`.

With this, the Trade Execution page and chatbox agent load even when **speech_recognition** or **pyttsx3** (or both) are missing; voice/TTS features are simply disabled.

### Files changed

- **ui/chatbox_agent.py**
  - Try/except around `import pyttsx3`; set `pyttsx3 = None` and `PYTTSX3_AVAILABLE` in the except.
  - Added `TTS_AVAILABLE = PYTTSX3_AVAILABLE`.
  - TextToSpeech: init and speak guard on `PYTTSX3_AVAILABLE` and `pyttsx3`/`self.engine`; init uses try/except around `pyttsx3.init()`.
  - ChatboxAgent: `self.tts = TextToSpeech() if (enable_tts and PYTTSX3_AVAILABLE) else None`.

---

## Summary

| Issue | Location | Fix |
|-------|----------|-----|
| NameError `uptime` | pages/11_Admin.py System Dashboard tab | Use `dashboard_data = _get_system_dashboard_data()` once; reference `dashboard_data['uptime_str']`, `dashboard_data.get('agents_active', 0)`, etc. in the expander and elsewhere. |
| Page not loading (pyttsx3) | ui/chatbox_agent.py | Optional import with `pyttsx3 = None` and `PYTTSX3_AVAILABLE`; guard all pyttsx3 use; create TextToSpeech only when `PYTTSX3_AVAILABLE`. |
