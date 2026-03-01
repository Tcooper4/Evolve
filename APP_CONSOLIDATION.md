# App Consolidation (app.py)

## Summary

`app.py` was the original single-page Streamlit entry point with its own navigation, prompt box, and page renderer. The app now uses Streamlit’s native multipage layout: **pages/0_Home.py** is the front door, **pages/1_Chat.py** is the chat interface, and the remaining pages cover Forecasting, Strategy Testing, Trade Execution, Portfolio, Risk, Performance, Model Lab, Reports, Alerts, Admin, and Memory. `app.py` has been reduced to a **minimal launcher** that only handles startup and branding.

---

## What app.py Contained (Before)

### 1. Startup and config
- TensorFlow/keras warning suppression  
- Project root added to `sys.path`  
- `atexit` hook to close DB and memory store  
- Logging (config.logging_config or basicConfig fallback)  
- `dotenv` load  
- API key debug logs (Alpha Vantage, Finnhub, Polygon)  
- `st.set_page_config` (wide layout, collapsed sidebar)

### 2. UI imports (ui.page_renderer)
- `render_sidebar`, `render_top_navigation`, `render_voice_input`  
- `render_prompt_interface`, `render_prompt_result`, `render_conversation_history`, `render_agent_logs`  
- Page renderers: `render_home_page`, `render_forecasting_page`, `render_strategy_page`, `render_model_page`, `render_reports_page`  
- Advanced: `render_settings_page`, `render_system_monitor_page`, `render_performance_analytics_page`, `render_risk_management_page`, `render_orchestrator_page`  
- `render_footer`  
- Fallback stub implementations when the UI module failed to import

### 3. Lazy initializers (session state)
- `_ensure_orchestrator()` → TaskOrchestrator, TaskScheduler, TaskMonitor  
- `_ensure_agent_controller()` → AgentController, TaskRouter, registry  
- `_ensure_notification_service()`  
- `_ensure_audit_logger()`  
- `_ensure_monitoring()` → health_monitor, system_monitor  
- `_ensure_automation()` → automation_core, workflow_manager  
- `_ensure_model_monitoring()` → model_log, perf_logger, model_monitor  
- `_ensure_commentary_service()`  
- `_ensure_phase2_models()`, `_ensure_phase_visualization()`, `_ensure_phase_nlp()`, `_ensure_phase_analytics()`, `_ensure_phase_execution()`

### 4. Custom CSS
- Large block of CSS for header, prompt container, buttons, result cards, sidebar, nav items, status indicators, conversation styling, tooltips

### 5. Main flow
- Sidebar and top nav from `render_sidebar()` / `render_top_navigation()`  
- Voice input from `render_voice_input()`  
- **Prompt input** from `render_prompt_interface()`; on submit, `routing.prompt_router.route_prompt(prompt)` and `render_prompt_result(result)`  
- Conversation history and agent logs  
- **Main content** by `primary_nav`: Home & Chat, Forecasting, Strategy Lab, Model Lab, Reports  
- **Advanced** by `advanced_nav`: Settings, Monitor, Analytics, Risk, Orchestrator (with lazy orchestrator init)  
- Footer from `render_footer()`

---

## Where Those Features Live Now

| Former app.py feature | Now covered by |
|------------------------|----------------|
| **Prompt input + routing + result** | **pages/1_Chat.py** – Chat UI and NL routing |
| **Sidebar / top nav** | Streamlit multipage sidebar (app + pages/0_Home, 1_Chat, …) |
| **Home & Chat** | **pages/0_Home.py** (briefing, cards), **pages/1_Chat.py** (chat) |
| **Forecasting** | **pages/2_Forecasting.py** |
| **Strategy Lab** | **pages/3_Strategy_Testing.py** |
| **Model Lab** | **pages/8_Model_Lab.py** |
| **Reports** | **pages/9_Reports.py** |
| **Settings** | **pages/11_Admin.py** (Configuration, API Keys, Broker, etc.) |
| **Monitor** | **pages/11_Admin.py** (System Monitoring tab) |
| **Analytics** | **pages/7_Performance.py** |
| **Risk** | **pages/6_Risk_Management.py** |
| **Orchestrator** | **pages/11_Admin.py** (Maintenance tab – Task Orchestrator) |
| **Lazy inits (orchestrator, agent controller, etc.)** | Used only when the relevant page is opened; can be re-attached in those pages or left as optional |
| **Custom CSS** | Removed; pages use Streamlit defaults and any page-specific styling |
| **ui.page_renderer** | No longer used by app.py; individual pages implement their own UI |

---

## What app.py Does Now (Minimal launcher)

1. **Path** – Inserts project root into `sys.path`.  
2. **Shutdown** – `atexit` to close DB and memory store.  
3. **Logging** – config logging or basicConfig.  
4. **Env** – `load_dotenv()`.  
5. **Page config** – `st.set_page_config` (title, wide, sidebar).  
6. **Sidebar** – Short branding block (“Evolve AI”, caption).  
7. **Main body** – When `app.py` is the active “page”, shows a short welcome and tells users to use the sidebar to open Home, Chat, etc.

No prompt box, no custom nav, no render_* calls, no lazy inits, no big CSS block. All feature entry points are the Streamlit pages under `pages/`.

---

## How to Run

```bash
streamlit run app.py
```

The sidebar lists the main script and all pages (0_Home, 1_Chat, 2_Forecasting, …). Choosing **Home** or **Chat** gives the main experience; the **app** entry is the minimal welcome view.
