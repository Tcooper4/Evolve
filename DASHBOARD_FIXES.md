# Dashboard Fixes

Three fixes applied across the codebase.

---

## 1. StreamlitAPIException in ui/page_assistant.py (session_state after widget)

### Problem
`st.session_state[allow_suggestions_key] = allow_suggestions` was set **after** the checkbox widget with that key was rendered. Streamlit does not allow modifying a widget’s session state key after the widget has been created in the same run.

### Fix
- **Initialize all keys before any widget:**  
  `input_key`, `history_key`, `allow_suggestions_key` are initialized at the top (e.g. `if key not in st.session_state: st.session_state[key] = ...`).  
  `allow_suggestions_key` is set to `False` by default before the expander.
- **Let the checkbox own its state:**  
  The checkbox is rendered with only `key=allow_suggestions_key` (no `value=`). No assignment to `st.session_state[allow_suggestions_key]` after the checkbox.
- **Reading the value:**  
  In the Send button handler, the “allow suggestions” value is read with `st.session_state.get(allow_suggestions_key, False)` when building the system prompt.
- **Clearing the text input:**  
  Assigning `st.session_state[input_key] = ""` after the text input widget in the same run can also trigger Streamlit. So a **clear flag** is used: before `st.rerun()`, set `st.session_state[clear_input_key] = True`. At the **start** of the next run, if `clear_input_key` is set, clear `input_key` and remove the flag, then render widgets. The input is cleared on the next run without writing to the widget’s key after it’s rendered.

### Files
- `ui/page_assistant.py`: init `input_key` and `clear_input_key`; remove post-checkbox assignment; use clear flag for input; read `allow_suggestions_key` from session in handler.

---

## 2. Admin System Dashboard — real data and “Not configured” badges

### Problem
The System Dashboard tab used hardcoded/mock data: “5 days, 12 hours” uptime, “47” trades today, “8” active strategies, “3” active agents, and fake system events. Health Monitoring, Automation, Agent API, and WebSocket sections showed warning boxes when services were not available, which looked like errors.

### Fix

**Real data**
- **Uptime:** `app_start_time` is stored in `st.session_state` on first load (`time.time()`). A helper `_get_system_dashboard_data()` computes uptime as `time.time() - app_start_time` and formats it as “X days, Y hours” (or “X hours”).
- **Trades today:** MemoryStore is queried for `namespace="trades"`, `category="orders"`; records with `timestamp` containing today’s date are counted.
- **Active strategies:** `config/strategies.yaml` is loaded; the number of enabled strategy definitions under `strategies.definitions` is counted.
- **Active agents:** `data/agent_registry.json` is loaded; agents with `status == "active"` are counted.
- **System events:** If `logs/app.log` exists, the last lines are read and the last 5 lines that contain WARNING/ERROR/INFO are shown as events. Otherwise the existing session `system_events` list is used (or “No recent events”).

**“Not configured” badges**
- **Health monitoring not available:** Replaced `st.warning("⚠️ Health monitoring not available")` with a grey badge “Not configured” and caption: “Health monitoring is not enabled. Enable SystemHealthMonitor in app to use this section.”
- **Automation not available:** Same pattern — grey badge and one-line caption.
- **Agent API not running / timeout / requests missing:** Replaced warning boxes with the same grey badge and short captions (e.g. “Agent API is not running. Start it with: python scripts/launch_agent_api.py”).
- **WebSocket not available / real-time updates disabled:** Grey badge and caption (e.g. “WebSocket client not available. Install: pip install websockets” and “Real-time updates are disabled. Click Connect to enable.”).
- **Task Orchestrator not available:** Grey badge and caption instead of warning.

Badge markup:  
`st.markdown('<span style="background:#e0e0e0;color:#555;padding:4px 10px;border-radius:4px;">Not configured</span>', unsafe_allow_html=True)`  
plus `st.caption("...")` for the explanation.

### Files
- `pages/11_Admin.py`: Added `import time` and `app_start_time` in session state; added `_get_system_dashboard_data()`; Quick Stats use its return values; system events use log-based or session events; “Not configured” badges replace the listed warnings; System Uptime in the backup section uses `_get_system_dashboard_data()["uptime_str"]`.

---

## 3. Trade Execution / UI loading — speech_recognition optional in chatbox_agent

### Problem
`ui/chatbox_agent.py` had a top-level `import speech_recognition as sr`. If `speech_recognition` is not installed, importing `ui` (e.g. via `ui.page_assistant` or other components that trigger `ui/__init__.py` and then `ui.chatbox_agent`) could fail and block pages that depend on the UI package (e.g. Trade Execution).

### Fix
- **Optional import:**  
  `try: import speech_recognition as sr; SPEECH_AVAILABLE = True`  
  `except ImportError: sr = None; SPEECH_AVAILABLE = False`
- **Guarded use of `sr`:**  
  - In `SpeechRecognizer.__init__`: only create `self.recognizer = sr.Recognizer()` and set its attributes when `SPEECH_AVAILABLE` and `sr` is not None; otherwise `self.recognizer = None`.
  - In `listen_for_speech`: return `None` immediately if `not SPEECH_AVAILABLE or self.recognizer is None or sr is None`; otherwise use `sr.Microphone()` and `sr.WaitTimeoutError` etc.
  - In `_transcribe_with_sphinx`: return `None` if `not SPEECH_AVAILABLE or sr is None or self.recognizer is None`; otherwise use `sr.AudioData` and `self.recognizer.recognize_sphinx`.
  - In `ChatboxAgent.__init__`: create `SpeechRecognizer` only when `enable_voice and SPEECH_AVAILABLE`, so `self.speech_recognizer` can be `None`.

With this, `ui.chatbox_agent` loads even when `speech_recognition` is not installed, and the rest of the UI (and Trade Execution) can load.

### Files
- `ui/chatbox_agent.py`: Optional `speech_recognition` import and `SPEECH_AVAILABLE`; all `sr` and recognizer usage guarded; `SpeechRecognizer` only created when `SPEECH_AVAILABLE`.

---

## Summary

| Issue | Location | Change |
|-------|----------|--------|
| StreamlitAPIException (session_state after widget) | ui/page_assistant.py | Init keys before widgets; no post-widget assignment; clear-input flag; read checkbox from session in handler. |
| Admin mock data | pages/11_Admin.py | Real uptime, trades today, active strategies, active agents, and log-based system events via `_get_system_dashboard_data()`. |
| Admin “not available” look | pages/11_Admin.py | Grey “Not configured” badge + one-line caption for Health, Automation, Agent API, WebSocket, Task Orchestrator. |
| speech_recognition blocking UI | ui/chatbox_agent.py | Optional import, SPEECH_AVAILABLE, and guards so `sr` is only used when available. |
