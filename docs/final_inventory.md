## PAGE: 2_Forecasting.py
### Tabs
[UNCONFIRMED] Full tab list pending complete read of `pages/2_Forecasting.py` in chunks.
### Expanders  
[UNCONFIRMED]
### Sidebar
[UNCONFIRMED]
### Major Features
[UNCONFIRMED]
### Buttons with Actions
[UNCONFIRMED]
### Session State Keys
[UNCONFIRMED]
### External Integrations
[UNCONFIRMED]
### Stubs & Incomplete Features
[UNCONFIRMED]
---
## PAGE: 3_Strategy_Testing.py
### Tabs
[UNCONFIRMED]
### Expanders  
[UNCONFIRMED]
### Sidebar
[UNCONFIRMED]
### Major Features
[UNCONFIRMED]
### Buttons with Actions
[UNCONFIRMED]
### Session State Keys
[UNCONFIRMED]
### External Integrations
[UNCONFIRMED]
### Stubs & Incomplete Features
[UNCONFIRMED]
---
## PAGE: 4_Trade_Execution.py
### Tabs
[UNCONFIRMED]
### Expanders  
[UNCONFIRMED]
### Sidebar
[UNCONFIRMED]
### Major Features
[UNCONFIRMED]
### Buttons with Actions
[UNCONFIRMED]
### Session State Keys
[UNCONFIRMED]
### External Integrations
[UNCONFIRMED]
### Stubs & Incomplete Features
[UNCONFIRMED]
---
## PAGE: 5_Portfolio.py
### Tabs
[UNCONFIRMED]
### Expanders  
[UNCONFIRMED]
### Sidebar
[UNCONFIRMED]
### Major Features
[UNCONFIRMED]
### Buttons with Actions
[UNCONFIRMED]
### Session State Keys
[UNCONFIRMED]
### External Integrations
[UNCONFIRMED]
### Stubs & Incomplete Features
[UNCONFIRMED]
---
## PAGE: 6_Risk_Management.py
### Tabs
[UNCONFIRMED]
### Expanders  
[UNCONFIRMED]
### Sidebar
[UNCONFIRMED]
### Major Features
[UNCONFIRMED]
### Buttons with Actions
[UNCONFIRMED]
### Session State Keys
[UNCONFIRMED]
### External Integrations
[UNCONFIRMED]
### Stubs & Incomplete Features
[UNCONFIRMED]
---
## PAGE: 7_Performance.py
### Tabs
[UNCONFIRMED]
### Expanders  
[UNCONFIRMED]
### Sidebar
[UNCONFIRMED]
### Major Features
[UNCONFIRMED]
### Buttons with Actions
[UNCONFIRMED]
### Session State Keys
[UNCONFIRMED]
### External Integrations
[UNCONFIRMED]
### Stubs & Incomplete Features
[UNCONFIRMED]
---
## PAGE: 11_Admin.py
### Tabs
[UNCONFIRMED]
### Expanders  
[UNCONFIRMED]
### Sidebar
[UNCONFIRMED]
### Major Features
[UNCONFIRMED]
### Buttons with Actions
[UNCONFIRMED]
### Session State Keys
[UNCONFIRMED]
### External Integrations
[UNCONFIRMED]
### Stubs & Incomplete Features
[UNCONFIRMED]
---
## COMPONENT: watchlist_widget.py
### Public Functions
- `render_watchlist() -> None`: renders a full watchlist UI.
  - Inputs: none directly; uses Streamlit inputs for symbol, note, price/RSI alert thresholds.
  - Renders:
    - Symbol & note inputs.
    - Numeric inputs for price-above, price-below, RSI-below, RSI-above thresholds.
    - “Add / Update Ticker” button that persists watchlist entries via `WatchlistManager.add_ticker`.
    - Main watchlist table showing Symbol, Price, Change %, RSI(14), Alert Status, Note, and a badge emoji column.
    - Per-row “Remove {SYM}” buttons.
- `_fetch_price_and_rsi(symbols: Dict[str, None]) -> Dict[str, Dict[str, Optional[float]]]`:
  - Internal helper; for each symbol, pulls 3 months of history from `yfinance`, computes latest price and `safe_rsi`(14), returns a mapping `{symbol: {"price": float|None, "rsi": float|None}}`.
### Session State
- Reads:
  - `st.session_state["watchlist_symbol_input"]` via `st.text_input` (symbol).
  - `st.session_state["watchlist_note_input"]` (note).
  - `st.session_state["watchlist_price_above"]` / `["watchlist_price_below"]` (price thresholds).
  - `st.session_state["watchlist_rsi_below"]` / `["watchlist_rsi_above"]` (RSI thresholds).
- Writes:
  - None directly; all writes go through `WatchlistManager` and Streamlit widgets.
### Backend Calls
- `WatchlistManager` from `trading.data.watchlist`:
  - `add_ticker(symbol, alert_price_above, alert_price_below, alert_rsi_below, alert_rsi_above, note)` to insert/update records.
  - `get_all()` to fetch all watchlist rows; expected to be list of dicts with keys `symbol`, `alert_price_above`, `alert_price_below`, `alert_rsi_below`, `alert_rsi_above`, `note`.
  - `remove_ticker(symbol)` to delete entries.
- `yfinance.Ticker(symbol).history` for both `_fetch_price_and_rsi` and for 2-day history to compute previous close.
- `safe_rsi` from `trading.utils.safe_indicators` to compute RSI series.
### Persistence
- Persists watchlist data via `WatchlistManager` (likely a SQLite-backed manager; persistence is not in this file but is implied by the docstring and class name).
- Uses Streamlit session state only for transient widget values; no direct file I/O or DB code here.
### Notes
- Error handling:
  - `_fetch_price_and_rsi` wraps each symbol fetch in `try/except` and silently skips failures.
  - Additional history fetch for change % is also wrapped in `try/except`.
- Alert logic:
  - Price triggers:
    - `🔴 TRIGGERED (price ≥ target)` when `price >= alert_price_above`.
    - `🔴 TRIGGERED (price ≤ target)` when `price <= alert_price_below`.
    - `🟡 NEAR price alert` when price is within ~2% of any price threshold.
  - RSI triggers:
    - `🔴 TRIGGERED (RSI ≤ target)` or `🔴 TRIGGERED (RSI ≥ target)` based on RSI thresholds.
  - Default status is `✅ OK`.
- Status
  - **Working**: full code path is implemented and makes external calls to yfinance and `WatchlistManager` with consistent error handling.
---
## COMPONENT: multi_timeframe_chart.py
### Public Functions
- `render_multi_timeframe_chart(symbol: str, hist_daily: Optional[pd.DataFrame] = None, show_volume: bool = True, show_sma: bool = True, height: int = 500) -> None`:
  - Fetches 3-month daily, 1-year weekly, and 5-year monthly OHLCV data for `symbol` using `yfinance` (unless `hist_daily` is provided for the daily view).
  - Renders three Streamlit tabs:
    - “📅 3-Month Daily”
    - “📆 1-Year Weekly”
    - “🗓️ 5-Year Monthly”
  - For each tab, calls `_render_ohlc_panel(data, symbol, label, show_volume, show_sma, height)` to draw a multi-pane Plotly candlestick chart with optional SMA(20/50) overlays and volume bars.
### Session State
- Reads/writes:
  - Uses `st.tabs` and `st.plotly_chart`; no direct `st.session_state` keys are read or written in this component.
### Backend Calls
- `yfinance.Ticker(symbol).history` to retrieve daily, weekly, and monthly OHLCV data.
- Plotly Graph Objects (`go.Figure`, `go.Candlestick`, `go.Scatter`, `go.Bar`) and `plotly.subplots.make_subplots` to construct the figures.
### Persistence
- No persistence. The function is purely presentational and fetches data on demand.
### Notes
- `_render_ohlc_panel`:
  - Normalizes column names to standard OHLCV.
  - Checks for required OHLC columns; renders a caption and returns if missing.
  - Uses 2-row layout when `show_volume` and “Volume” is present; otherwise 1 row.
  - Adds SMA20 and SMA50 when there is enough history.
  - Adds volume bars colored green/red depending on up/down closes.
  - Computes simple metrics for Last Close, SMA20, 52-week High, and a 20-day average volume metric row.
- Status
  - **Working**: all code paths are complete with robust handling for missing data; errors surfaced via `st.error` and `st.caption`.
---
## COMPONENT: news_candle_chart.py
### Public Functions
- `render_news_candle_chart(symbol: str = "AAPL", period: str = "3mo", interval: str = "1d", volume_threshold: float = 2.0, price_threshold: float = 0.02, show_annotations: bool = True) -> None`:
  - Fetches OHLCV history via `_fetch_ohlcv(symbol, period, interval)` (5-minute cached ticker history).
  - Calls `trading.analysis.volume_news_linker.detect_significant_candles(hist, volume_threshold, price_threshold)` to mark “significant” candles with a boolean `is_significant` and `volume_ratio`/`price_change_pct`.
  - Builds a two-row Plotly figure:
    - Row 1: candlestick chart with up/down colors.
    - Row 2: volume bars with colors based on candle direction and gold for significant candles; includes a 20-day rolling average volume line.
  - When `show_annotations`:
    - Calls `build_chart_annotations(df, symbol, max_annotations=8)` from `volume_news_linker` to obtain marker positions and hover text, then adds an “📰” marker scatter to the price chart.
    - Builds an “Significant Events” expandable list via `st.expander` for up to 5 top volume-ratio days, and for each calls `get_news_for_date(symbol, date_str)` to display news headlines, sources, URLs, and summaries.
### Session State
- Uses only Streamlit widgets (`st.expander`, `st.markdown`, `st.caption`, `st.plotly_chart`); no explicit `st.session_state` reads or writes in this file.
### Backend Calls
- `yfinance.Ticker(symbol).history` (wrapped in `_fetch_ohlcv`).
- `trading.analysis.volume_news_linker.detect_significant_candles`, `build_chart_annotations`, and `get_news_for_date` for event detection and news retrieval.
### Persistence
- No persistence beyond the cached `_fetch_ohlcv` result (`st.cache_data`); no disk or DB writes here.
### Notes
- Error handling:
  - `_fetch_ohlcv` returns empty DataFrame on failure; caller warns when not enough data.
  - Exceptions in data fetching and annotations/news workload show `st.error`/`st.caption` but do not crash the app.
- Status
  - **Working**: fully implemented visualization component with external analysis/news integration and explicit error handling.
---
## COMPONENT: onboarding.py
### Public Functions
- `check_onboarding() -> Optional[str]`:
  - Main entry point; orchestrates onboarding for API keys and preferred LLM.
  - Ensures a persistent `session_id` via `_ensure_session_id()` and writes it to:
    - `st.session_state["evolve_session_id"]`.
    - URL query params `sid`.
    - Browser `localStorage` via `_persist_session_id_to_local_storage`.
  - Logic:
    - If `st.session_state["evolve_force_onboarding"]` is set, resets `evolve_onboarding_done` and `evolve_show_form` and clears the force flag.
    - If `evolve_onboarding_done` is already `True`, returns the current `session_id`.
    - If `evolve_show_form` is not set and `load_user_keys(session_id)` returns stored keys with OpenAI or Anthropic entries, returns `session_id` without showing the form.
    - Otherwise renders the onboarding form via `_render_onboarding_form(session_id)` and:
      - On successful submission, marks `evolve_onboarding_done = True` and returns `session_id`.
      - If not submitted or invalid, renders a small JS snippet to post `session_id` to parent window and a “Reset my keys” button that sets `evolve_force_onboarding` and `evolve_onboarding_done` and calls `st.rerun()`. Returns `None`.
- `_ensure_session_id() -> str`:
  - Reads `evolve_session_id` from `st.session_state` or `sid` from `st.query_params`.
  - If absent, generates a new 16-byte hex token, sets it to both `st.session_state["evolve_session_id"]` and `st.query_params["sid"]`.
- `_persist_session_id_to_local_storage(session_id: str) -> None`:
  - Embeds a small `<script>` snippet via `streamlit.components.v1.html` to store `session_id` in browser `localStorage` under key `evolve_session_id`.
- `_render_onboarding_form(session_id: str) -> bool`:
  - Renders a `st.form("onboarding_keys")` with:
    - OpenAI API Key (required if no Anthropic key).
    - Anthropic API Key (optional).
    - News API Key (optional).
    - Preferred LLM provider select (openai/anthropic).
  - On submit:
    - Validates that at least one of OpenAI/Anthropic keys is supplied.
    - Calls `save_user_keys(session_id, keys)` and `save_user_preferences(session_id, {"preferred_llm_provider": preferred_llm})`.
    - Updates query param `sid`, sets `st.session_state["evolve_show_form"] = False` and `["evolve_onboarding_done"] = True`, shows success message.
    - Returns True on success; False on missing keys or if not submitted.
### Session State
- Reads:
  - `st.session_state["evolve_session_id"]`.
  - `st.session_state["evolve_force_onboarding"]`.
  - `st.session_state["evolve_onboarding_done"]`.
  - `st.session_state["evolve_show_form"]`.
- Writes:
  - `st.session_state["evolve_session_id"]`.
  - `st.session_state["evolve_force_onboarding"]`.
  - `st.session_state["evolve_onboarding_done"]`.
  - `st.session_state["evolve_show_form"]`.
### Backend Calls
- `load_user_keys`, `save_user_keys`, and `save_user_preferences` from `config.user_store`; they encapsulate persistence of credentials and preferences (likely in SQLite or local files, defined elsewhere).
- `streamlit.components.v1.html` to run small client-side JavaScript.
### Persistence
- Keys and preferences are persisted via the `user_store` backend keyed by `session_id`.
- `session_id` is stored in three places:
  - Streamlit session state.
  - URL query parameter `sid`.
  - Browser `localStorage`.
### Notes
- Error/edge handling:
  - Enforces at least one API key before accepting the form.
  - Supports a “Reset my keys” button to force re-onboarding.
- Status
  - **Working**: fully implemented onboarding flow with clear state transitions and persistence hooks.
---
## FEATURE FLAGS
### Flag Inventory

| flag name                      | default | controls              | checked by        | how to enable                               |
|--------------------------------|---------|-----------------------|-------------------|---------------------------------------------|
| `FEATURE_REAL_TIME_STREAMING` | False   | [UNCONFIRMED]         | [UNCONFIRMED]     | `FEATURE_REAL_TIME_STREAMING=true` in env  |
| `FEATURE_ADVANCED_ORDERS`     | False   | [UNCONFIRMED]         | [UNCONFIRMED]     | `FEATURE_ADVANCED_ORDERS=true` in env      |
| `FEATURE_RL_TRADING`          | False   | [UNCONFIRMED]         | [UNCONFIRMED]     | `FEATURE_RL_TRADING=true` in env           |
| `FEATURE_AUTO_REBALANCE`      | False   | [UNCONFIRMED]         | [UNCONFIRMED]     | `FEATURE_AUTO_REBALANCE=true` in env       |
| `FEATURE_GPU_ACCELERATION`    | False   | [UNCONFIRMED]         | [UNCONFIRMED]     | `FEATURE_GPU_ACCELERATION=true` in env     |
| `FEATURE_DISASTER_RECOVERY`   | False   | [UNCONFIRMED]         | [UNCONFIRMED]     | `FEATURE_DISASTER_RECOVERY=true` in env    |
| `FEATURE_BROKER_REDUNDANCY`   | False   | [UNCONFIRMED]         | [UNCONFIRMED]     | `FEATURE_BROKER_REDUNDANCY=true` in env    |

- **Defaults**:
  - All flags are initialized from environment variables with default string `'false'` and treated as enabled only when value is one of `("true", "1", "yes", "on")` (case-insensitive).
- **Control & usage**:
  - The `FeatureFlags` class and `is_feature_enabled` helper are defined in `utils/feature_flags.py`. From this file alone:
    - The **names above are the complete list of environment-backed flags**.
    - The **exact features they control and which modules check them are not visible here** and must be located by searching for `is_feature_enabled` or specific flag names elsewhere in the codebase [UNCONFIRMED in this file].
- **How to enable**:
  - Set corresponding environment variables before launching the app; e.g. in your shell or `.env`:
    - `FEATURE_REAL_TIME_STREAMING=true`
    - `FEATURE_ADVANCED_ORDERS=1`
    - etc.

