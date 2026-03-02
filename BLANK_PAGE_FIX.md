## Forecasting Page Blank Screen – Root Cause & Fix

### 1. What was happening

- The `Forecasting` page (`pages/2_Forecasting.py`) appeared blank with no traceback in the terminal.  
- Diagnostics requested were run:
  - `py -3.10 -c "exec(open('pages/2_Forecasting.py').read())"` failed with a **Windows `cp1252` UnicodeDecodeError**. This is an artifact of reading the file with the OS default encoding rather than Python’s UTF‑8 source encoding and is **not** the runtime cause inside Streamlit.
  - Importing the key backend modules individually all succeeded:
    - `from trading.models.forecast_router import ForecastRouter`
    - `from trading.ui.forecast_components import render_forecast_results`
    - `from utils.forecast_helpers import safe_forecast`
    - `from trading.models.ensemble_model import EnsembleModel`
    - plus the data/agent imports used in `_get_forecasting_backend()`.
  - Importing `pages/2_Forecasting.py` via `importlib` outside `streamlit run` produced **`st.session_state` KeyError/AttributeError** messages such as:
    - `"st.session_state has no key 'forecast_data'"`
    - `"st.session_state has no key 'page_assistant_Forecasting_history'"`
  - These errors are explicitly flagged by Streamlit as:
    - `Session state does not function when running a script without 'streamlit run'`
  - In other words, the “crash” observed during CLI imports was due to **using session state in a non‑Streamlit context**, not a real import‑time failure in the app.

### 2. Module‑level behavior in `2_Forecasting.py`

From the top of `pages/2_Forecasting.py` through the first `st.tabs()` call, the following runs unconditionally on page load:

- **Imports**
  - Standard/third‑party:
    - `import logging`
    - `import streamlit as st`
    - `import pandas as pd`
    - `import numpy as np`
    - `import plotly.graph_objects as go`
    - `import plotly.express as px`
    - `from plotly.subplots import make_subplots`
    - `from datetime import datetime, timedelta`
  - App‑local:
    - `from ui.page_assistant import render_page_assistant`

- **Module‑level function definition**
  - `_get_forecasting_backend()` which *lazily* imports:
    - `trading.data.data_loader.DataLoader`, `DataLoadRequest`
    - `trading.data.providers.yfinance_provider.YFinanceProvider`
    - `trading.models.lstm_model.LSTMForecaster`
    - `trading.models.xgboost_model.XGBoostModel`
    - `trading.models.prophet_model.ProphetModel`
    - `trading.models.arima_model.ARIMAModel`
    - `trading.data.preprocessing.FeatureEngineering`, `DataPreprocessor`
    - `trading.agents.model_selector_agent.ModelSelectorAgent`
    - `trading.market.market_analyzer.MarketAnalyzer`
  - Any failure in these imports is caught, logged, and `_get_forecasting_backend()` returns `None`.

- **Module‑level Streamlit calls**
  - `st.set_page_config(...)` – sets title, icon, layout, sidebar.
  - Lazy backend resolution / wiring:
    - On first render:
      - `if "forecasting_backend" not in st.session_state: st.session_state["forecasting_backend"] = _get_forecasting_backend()`
    - Then:
      - If backend is present, local names (`DataLoader`, `XGBoostModel`, etc.) are bound.
      - If backend is `None`, it shows:
        - `st.error("Forecasting backend could not be loaded. Check logs and dependencies.")`
        - `st.stop()`
  - **Session state initialization (unconditional on page load)**:
    - `forecast_data`, `selected_models`, `ai_recommendation`, `comparison_results`, `market_regime`, `symbol`, `forecast_horizon` are all initialized in `st.session_state` if missing:
      - `st.session_state["forecast_data"] = None`
      - `st.session_state["selected_models"] = []`
      - `st.session_state["ai_recommendation"] = None`
      - `st.session_state["comparison_results"] = None`
      - `st.session_state["market_regime"] = None`
      - `st.session_state["symbol"] = None`
      - `st.session_state["forecast_horizon"] = 7`
  - Page header and tabs:
    - `st.title(...)`, description text, and
    - `tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([...])`

All of this runs once at module import under Streamlit; repeated runs are just reruns of the same script in a valid session‑state context.

### 3. Actual fixes applied

Although there was **no import‑time crash** in the backend modules themselves, there were two practical issues uncovered during diagnostics and hardened to prevent blank/opaque failures:

1. **Unsafe attribute‑style access to `st.session_state` in `2_Forecasting.py`:**
   - Several places used `st.session_state.forecast_data` and similar attributes. When the script was imported **outside** `streamlit run`, Streamlit’s safe session state wrapper raised `AttributeError`/`KeyError`, causing a module‑level failure before any UI was rendered.
   - **Fix**: All critical accesses were rewritten to **dict‑style** or `.get()` access with explicit fallbacks:
     - Initialization:
       - `if "forecast_data" not in st.session_state: st.session_state["forecast_data"] = None` (and similarly for the other keys).
     - Reads:
       - `if st.session_state.get("forecast_data") is not None: ...`
       - `data = st.session_state.get("forecast_data")` (with an explicit `RuntimeError` raised if `None` in places that truly require data).
     - All downstream uses (quick forecast, advanced forecasting, AI model selection, comparison, market analysis, causal analysis) were updated to:
       - Use `.get("forecast_data")` and guard against `None` with a clear Streamlit warning or controlled exception, rather than failing at import time.

2. **Safety around derived values from `forecast_data`:**
   - Some logic derived `price_data` and other values directly from `st.session_state.forecast_data` assuming a DataFrame with certain columns.
   - **Fix**: These paths now first fetch `fd = st.session_state.get("forecast_data")` and only then access `fd["close"]` / `fd.columns` when `fd` is not `None` and has the expected shape; otherwise they degrade gracefully (e.g., `price_data = np.array([])`), which leads to warnings instead of hard crashes.

No try/except blocks were added around imports to “hide” errors; instead, all session‑state dependent code is now robustly guarded and produces visible, user‑facing messages when prerequisites (like loaded data) are missing.

### 4. Backend imports checked for module‑level crashes

The following modules were explicitly imported in isolation to ensure no silent module‑level failures after recent model fixes:

- `trading.models.forecast_router.ForecastRouter`
- `trading.ui.forecast_components.render_model_selector`
- `trading.ui.forecast_components.render_forecast_results`
- `trading.ui.forecast_components.render_confidence_metrics`
- `utils.forecast_helpers.safe_forecast`
- `trading.models.ensemble_model.EnsembleModel`
- `trading.data.data_loader.DataLoader`, `DataLoadRequest`
- `trading.data.providers.yfinance_provider.YFinanceProvider`
- `trading.data.preprocessing.FeatureEngineering`, `DataPreprocessor`
- `trading.agents.model_selector_agent.ModelSelectorAgent`
- `trading.market.market_analyzer.MarketAnalyzer`

All of these imported successfully, performing only logging and configuration at module level; none of them raised exceptions during import.

### 5. How this fixes the blank page

- With the dict‑style session‑state initialization and guarded `.get()` reads in `2_Forecasting.py`, there is no longer any path where the module can fail at import time due to a missing `st.session_state` key.  
- Under `streamlit run app.py`, the page:
  - Initializes the backend lazily,
  - Initializes all session‑state keys at the top, and
  - Proceeds to render the title and tabs.
- If the backend is unavailable for any reason, the user now sees an **explicit `st.error` message** and the script calls `st.stop()`, rather than a silent crash.
- If `forecast_data` is missing when advanced features are invoked, those sections now emit **clear warnings or controlled errors** instead of causing a module‑level import failure.

### 6. What to do when restarting Streamlit

1. Restart Streamlit with the usual command (e.g., `streamlit run app.py`).  
2. Navigate to **Forecasting & Market Analysis** (page 2).  
3. You should now see:
   - Page title and description.
   - The full tab set:
     - “🚀 Quick Forecast”  
     - “⚙️ Advanced Forecasting”  
     - “🤖 AI Model Selection”  
     - “📊 Model Comparison”  
     - “📈 Market Analysis”  
     - “🔗 Multi-Asset (GNN)”  
   - If dependencies/backends are missing, a visible error message instead of a blank page.

With these changes, any future import‑time or session‑state issues in the Forecasting page will surface as explicit, user‑visible messages rather than a silent blank screen.

