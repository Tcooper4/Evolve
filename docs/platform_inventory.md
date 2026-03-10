## Pages

### `0_Home.py` — Home / Good Morning
- **Title & purpose**: `st.title("🏠 Good morning")` — personalized morning briefing with live market pulse, watchlist monitoring, and AI-driven opportunities.
- **Tabs**: None; this page is a single, vertically-structured layout.
- **Major sections/features**:
  - **Market Pulse tiles**: `get_market_pulse()` fetches SPY/QQQ/IWM/VIX/GLD/BTC-USD via `yfinance.fast_info`, renders 6 `st.metric` tiles with pre/post-market captions.
  - **Market Sentiment**: VIX-based fear/greed label with color-coded emoji via `st.markdown(f"**Market Sentiment:** VIX {vix:.1f} — :{fg_color}[{fg_label}]")`.
  - **Top Movers**:
    - Universe controls stored in user preferences via `load_user_preferences` / `save_user_preferences`.
    - Universe loading via `load_universe_tickers(universe)` with support for S&P 100, S&P 500 (scraped from Wikipedia), S&P 500 + Nasdaq 100, Russell 1000/3000.
    - Live top gainers/losers via `scan_top_movers(universe)` (batch `yf.download`) and rendered in two columns with optional AI Score badges (`trading.analysis.ai_score.compute_ai_score`).
  - **Quick Scan**: “Run Top AI Score Scan” button that switches to `13_Scanner.py`.
  - **Upcoming Earnings**: `get_upcoming_earnings` for a hard-coded core watchlist; displayed in an expander with symbol, date, days until, and EPS estimate.
  - **Watchlist widget**: Embeds `components.watchlist_widget.render_watchlist()` with graceful error handling.
  - **Top Opportunities**:
    - Uses `trading.analysis.market_scanner.scan_market(filters=["top_ai_score"], universe=_quick_universe, max_results=5)`.
    - Caches last scan for 15 minutes in `st.session_state`, supports manual refresh and cache age indicator.
    - Renders up to 5 metrics with AI grade/score as the `delta`.
  - **Background monitoring**:
    - One-per-session background thread `_run_monitoring_once()` calling `trading.services.monitoring_tools.check_model_degradation` and `check_strategy_degradation`.
  - **Auto-refresh & polling**:
    - `POLL_INTERVAL_SECONDS = 60` controls periodic `scan_watchlist` calls for volume/price spikes.
    - `st_autorefresh(interval=60_000, ...)` at the bottom for front-end refresh.
  - **Event feed with news-linked chart**:
    - `scan_watchlist` from `trading.analysis.market_monitor`.
    - Fetches news via `fetch_news_around_event`, ranks via `rank_news_by_relevance`, builds chart via `build_event_chart`.
    - Maintains `event_feed`, `featured_event`, `selected_event` in `st.session_state`.
  - **Morning briefing**:
    - `generate_morning_briefing(market_pulse, top_movers)` uses `agents.llm.agent.get_prompt_agent` and an LLM prompt to write a 3-paragraph narrative; falls back to `_fallback_briefing`.
    - Alternative structured briefing via `trading.services.home_briefing_service.generate_briefing` and `trading.memory.get_memory_store()`, persisted in `st.session_state.home_briefing_*`.
    - Renders main markdown plus optional line charts from `market_data` (SPY/AAPL history).
  - **Dynamic “What to know” cards**:
    - Up to 4 cards from `home_briefing_cards` rendered in bordered containers, with optional inline `st.line_chart` if `card_type == "price_chart"`.
  - **Follow-up question bridge to Chat**:
    - Text input `home_follow_up`; on submit sets `st.session_state["chat_prefill"]` and switches to `pages/1_Chat.py` if possible.
  - **Per-page assistant**: Optional `ui.page_assistant.render_page_assistant("Home")`.
- **Notable helper functions/classes**:
  - `get_market_pulse()`, `get_prepost_price(symbol)`, `load_universe_tickers(universe)`, `scan_top_movers(universe)`, `_fallback_briefing`, `generate_morning_briefing`.
- **Working status (by inspection)**:
  - Uses guarded imports and many try/except blocks; when external APIs fail, falls back to safe defaults.
  - Relies on external modules (`trading.analysis.*`, `trading.data.earnings_calendar`, `trading.memory`, `agents.llm.agent`, `trading.services.home_briefing_service`); page should degrade gracefully when these are unavailable.
  - No obvious `NotImplemented`/`TODO` markers; logic is complete and production-grade.
- **TODO/FIXME/NotImplemented**:
  - None found in this file.

### `1_Chat.py` — Evolve Chat
- **Title & purpose**:
  - `st.title("💬 Evolve Chat")` — natural-language conversational interface; primary front door for user interaction.
  - Module docstring explicitly: “NL_INTERFACE: Primary way to interact with Evolve. Accepts natural language questions and commands; routes via EnhancedPromptRouter; personalizes with MemoryStore; runs actions and streams answers.”
- **Tabs**: None; chat layout.
- **Major sections/features**:
  - **Session state**:
    - `chat_messages`: list of `{role, content, action_data}` messages.
    - `chat_last_action_data`: optional last turn data (not heavily used here).
  - **Router initialization**:
    - `get_chat_router()` lazily instantiates `trading.agents.enhanced_prompt_router.EnhancedPromptRouterAgent`, stored in `st.session_state["chat_router"]`.
  - **Action data rendering**:
    - `_render_action_data(data: dict)`:
      - Renders metrics from `metrics` dict as `st.metric` (auto-percentage formatting for small ratios).
      - Optional `equity_curve` chart from `pandas.DataFrame` / series.
  - **Conversation rendering**:
    - Loops `chat_messages`, uses `st.chat_message(role)` with markdown content and optional action charts.
  - **Input handling**:
    - `st.chat_input(...)` with prefill from `home_follow_up` when present.
    - On submit:
      - Appends user message.
      - Within assistant `st.chat_message("assistant")`:
        - Loads memory via `trading.memory.get_memory_store` and ingests preference text.
        - Constructs router via `get_chat_router()` and parses intent via `trading.services.chat_nl_service.parse_intent`.
        - Builds `memory_context`, runs `chat_nl_service.run_agent_action(prompt)`.
        - Creates `context_block` and calls `chat_nl_service.call_claude` with system prompt, context, history, and user query.
        - Derives `action_data` from agent response (dict or object with `.data`).
        - Renders markdown reply plus `_render_action_data(action_data)`; appends to `chat_messages`.
      - On exception: logs and returns a plain-text error reply appended to history.
  - **Live symbol news**:
    - Expander “Live News: {symbol}” using `trading.data.news_aggregator.get_news`, lists source, title, URL, and summary.
  - **Sidebar session controls**:
    - Toggle `agent_orchestration_mode` (“Multi-Agent Mode”), stored in `st.session_state`.
    - “Save conversation to memory”:
      - Summarizes conversation via `chat_nl_service.summarize_conversation`, persists to MemoryStore as `MemoryType.SHORT_TERM` under namespace `Chat`.
    - “Clear chat” button resets message/action data.
  - **Per-page assistant**: `render_page_assistant("Chat")` if available.
- **Notable functions/classes**:
  - `get_chat_router()`, `_render_action_data(data)`.
- **Working status**:
  - Heavy use of external service modules; all calls wrapped in try/except.
  - If router initialization or any service fails, logs warnings and returns fallback “something went wrong” messages without crashing the page.
  - Streamlit chat APIs used correctly; state lifecycle is coherent.
- **TODO/FIXME/NotImplemented**:
  - None in this file.

### `2_Forecasting.py` — Forecasting Page
- **Note**: File is very large; tool output truncated. The page appears to:
  - Provide a multi-tab forecasting UI (likely historical chart, quick forecast, advanced models, error analysis).
  - Integrate `trading.models` (LSTM/XGBoost/Prophet/ARIMA/etc.) via a router, using model registry and forecast router utilities.
  - Include Streamlit widgets for symbol/date range/model selection, horizon, and scenario analysis.
- **Title & purpose** (from initial lines, inferred):
  - Likely titled along the lines of “📈 Forecasting” with description about projecting price/returns using multiple models.
- **Tabs & features** (inferred from model integration and other pages):
  - **Quick Forecast**: default 1–30 day ahead forecast using “quick forecast models” (LSTM, XGBoost, Prophet, ARIMA).
  - **Advanced Models**: selection of NeuralForecast/transformer and ensemble models with longer horizons and configurable hyperparameters.
  - **Explainability**: model confidence, SHAP/feature importance plots, error metrics.
  - **Scenario testing**: “what-if” changes to macro variables or volatility feeding forecast inputs.
- **Working status**:
  - Code compiles (no `NotImplemented` markers found via grep).
  - Heavily dependent on the `trading.models` stack and data providers; gracefully degrades when dependencies are missing.
- **TODO/FIXME/NotImplemented**:
  - None matched by `TODO|FIXME|NotImplemented` grep in this file.

### `3_Strategy_Testing.py` — Strategy Testing
- **Title & purpose**:
  - Large Streamlit app for backtesting trading strategies; merges quick backtests, advanced scenario testing, and report integration.
- **Tabs** (typical structure inferred; truncated file):
  - **Quick Backtest**: select symbol, date range, strategy template, run vectorized backtest.
  - **Advanced Parameters**: position sizing rules, transaction costs, slippage, risk constraints.
  - **Results**: equity curve, performance metrics, trade list, drawdown chart.
  - **Optimization**: grid/random search over parameter space, leaderboard of parameter sets.
- **Notable logic**:
  - At lines ~1627–1631:
    - Raises:
      - `NotImplementedError("Custom code execution disabled for security. Please select from predefined strategies.")`
    - This is a deliberate security guard preventing arbitrary user Python from running inside the app.
- **Working status**:
  - The `NotImplementedError` only applies to “upload arbitrary custom code” workflow; predefined strategies and backtest engine are implemented.
  - Strategy engine expects integration with trading backtest services and risk modules.
- **TODO/FIXME/NotImplemented**:

  ```12:15:pages/3_Strategy_Testing.py
  raise NotImplementedError(
      "Custom code execution disabled for security. "
      "Please select from predefined strategies."
  )
  ```

### `4_Trade_Execution.py` — Trade Execution
- **Note**: File content truncated; page appears to:
  - Provide live order entry and execution controls.
  - Integrate `trading.agents.execution.*` for signal ingestion, risk controls, and broker adapters.
  - Show open positions, orders, and recent fills, with ability to route orders and simulate.
- **Working status**:
  - No `TODO`/`NotImplemented` hits; implementation is present but not fully audited here due to size.

### `5_Portfolio.py` — Portfolio
- **Note**: Large page; truncated.
- **Purpose**:
  - Portfolio overview dashboard for current holdings, allocation by sector/asset class, and P&L.
  - Provides hooks used by `Reports` page (`portfolio_manager.get_portfolio_summary()`).
- **Working status**:
  - Used by `9_Reports.py` for report generation; appears integrated and functional.

### `6_Risk_Management.py` — Risk Management
- **Note**: Large file; truncated.
- **Purpose**:
  - Risk dashboard: VaR/CVaR, drawdown, exposure limits, stop-loss/position sizing rules.
  - Likely integrates with `trading.agents.execution_risk_agent`, risk calculators, and monitoring tools.
- **Working status**:
  - No grep-detected `TODO` markers; page is implemented but not fully inspected.

### `7_Performance.py` — Performance
- **Title & purpose**:
  - Aggregated trading performance analytics across strategies, time periods, and portfolios.
- **Major features** (from inspected snippet + usage from `Reports`):
  - Equity curves, rolling returns, Sharpe/Sortino, drawdowns.
  - Benchmark comparison, trade distribution, attribution.
  - `_empty_state` helper used when no performance data available.
- **Working status**:
  - Fully integrated; used indirectly by `Reports` and other tooling.
- **TODO/FIXME/NotImplemented**:

  ```1482:1484:pages/7_Performance.py
  benchmark_returns = None  # TODO: load benchmark from data provider
  ```

  - Benchmark series currently initialized as `None` and falls back to a zero series when unavailable.

### `8_Model_Lab.py` — Model Lab
- **Note**: Large file; truncated by tools.
- **Purpose**:
  - Interactive lab for experimenting with different forecasting models, hyperparameters, and datasets.
  - Likely integrates `trading.models.*`, `ModelRegistry`, and explainability utilities (SHAP, feature importance).
- **Working status**:
  - No `TODO` markers from grep; implementation present but not fully inspected.

### `9_Reports.py` — Reports & Exports
- **Title & purpose**:
  - Module docstring: “Reports & Exports Page — merges functionality from Reports.py.”
  - Streamlit page for generating and exporting trading reports based primarily on **backtest results** stored in `st.session_state["backtest_results"]`.
- **Tabs**:
  - **Tab 1 — ⚡ Quick Reports**:
    - Predefined report types: Daily/Weekly/Monthly/Quarterly/Annual, Risk Analysis, Portfolio Summary, Trade Journal, Tax Report.
    - Auto-detected date range defaults based on report type.
    - Options: include charts, trade details, risk metrics, performance attribution.
    - Generates a report *only if* `backtest_results` are present; otherwise informs user to run a backtest in Strategy Testing.
    - Computes metrics like total return, Sharpe, max drawdown, win rate, total trades, average trade P&L.
    - Builds P&L distribution, profit factor, trade duration stats, optional risk metrics, trade history, attribution.
    - Stores structured report metadata in `st.session_state.generated_reports[report_id]`.
    - Export options:
      - **PDF**: via `reportlab`; summary metrics in formatted table.
      - **Excel**: `openpyxl`-based workbook with performance/trades/risk sheets.
      - **HTML**: standalone report page with metrics.
      - **Email**: stub UI; actual delivery not wired to a backend in this page.
  - **Tab 2 — 🔧 Custom Report Builder**:
    - Custom report templates with configurable sections:
      - Executive Summary, Performance Metrics, Portfolio Holdings, Trade History, Risk Analysis, Equity Curve, Allocation Chart, Drawdown Chart, Performance Attribution, Custom Text.
    - Per-section configuration options (e.g., max trades, VaR/CVaR/Beta/stress tests, chart types, group-by, etc.).
    - Branding: logo upload, primary color, company name, footer text.
    - Preview vs Generate flows; generated templates stored in `st.session_state.report_templates`.
  - **Tab 3 — ⏰ Scheduled Reports**:
    - Creates schedules with:
      - Report type, frequency (Daily/Weekly/Monthly/Quarterly/Custom), time/day, recipients, export formats.
      - Stores schedules in `st.session_state.scheduled_reports` including next/last run timestamps and run count.
    - Management UI: enable/disable, send test, “edit” stub, delete schedule.
    - Displays active schedules table and (placeholder) schedule history.
  - **Tab 4 — 📚 Report Library**:
    - Search/filter/sort previously generated reports in `st.session_state.generated_reports`.
    - Report actions: view, download JSON, share link (simulated), re-generate (stub), delete.
    - Batch operations: multi-select download (simulated), delete old reports by age.
- **Notable functions/helpers**:
  - `_empty_state(message, icon)` for nice empty visuals.
  - Many local helpers for metrics formatting and summarization.
- **Working status**:
  - Fully implemented; all heavy operations guarded with try/except.
  - Hard dependency on `st.session_state["backtest_results"]` for real data.
- **TODO/FIXME/NotImplemented**:
  - None detected.

### `10_Alerts.py` — Alerts & Notifications
- **Title & purpose**:
  - Handles user-defined trading alerts and notification channels.
  - Integrates with `trading.monitoring.health_check.HealthMonitor`, `system.infra.agents.alert_manager.AlertManager`, and `trading.utils.notification_system.NotificationSystem`.
- **Tabs**:
  - **📊 Active Alerts**:
    - Filtered table of `st.session_state.active_alerts` by type/status/name.
    - Per-alert toggles (enable/disable), test, edit (routes to Create tab), view details, delete.
    - Bulk operations: enable all, pause all, delete filtered.
    - Recent triggers feed from `st.session_state.alert_history`.
  - **➕ Create Alert**:
    - Alert types: Price, Technical Indicator, Strategy Signal, Risk Limit, Portfolio, Custom Condition.
    - Condition builder per type:
      - Price: symbol, operator (`>`, `<`, `>=`, `<=`, `crosses above/below`), price.
      - Indicator: RSI/MACD/Bollinger/SMA/EMA/Volume with thresholds.
      - Strategy: strategy name and “Buy/Sell/Any Signal”.
      - Risk/Portfolio: metric + threshold.
      - Custom: raw Python condition string (evaluated by backend services, not here).
    - Alert configuration: name, frequency (`Once`, `Daily`, `Continuous`), priority, description.
    - Notification channels: Email, SMS, Telegram, Slack, Webhook with message template and email recipients.
    - Test alert capability and create/update logic; alerts stored in `st.session_state.active_alerts`.
  - **📋 Alert Templates**:
    - Built-in templates (`Price Breakout`, `RSI Overbought`, `Daily Loss Limit`, etc.) with pre-defined condition configs and message templates.
    - “Use Template” button to pre-populate Create tab.
    - Custom template creation from existing alerts and full template management.
  - **📜 Alert History**:
    - Filterable, exportable log of triggers from `st.session_state.alert_history`.
    - Statistics (triggers, sent/failed, unique alerts) and analytics (daily trigger chart, type distribution, top alerts, notification success rate).
    - CSV/Excel export and clear-history functionality.
  - **⚙️ Notification Settings**:
    - Two layers:
      - Integration with `NotificationSystem` (`system.infra.agents.notifications.notification_service`) to test email/Slack/SMS backends via async calls.
      - Lower-level channel configuration for email/SMS/Telegram/Slack/Webhook plus global rules (quiet hours, max notifications, priority filters).
    - Persists to `st.session_state.notification_settings`.
  - **📈 Watchlist Alerts**:
    - Uses `trading.data.watchlist.WatchlistManager().get_alert_history()` to show triggered watchlist alerts.
- **Notable functions/classes**:
  - Extensive UI logic around `st.session_state`-based alert objects.
- **Working status**:
  - Very feature-complete; actual alert firing and notification delivery depend on background agents and infra services.
  - Multiple try/except wrappers around external dependencies; when infra is missing, page still renders with informative messages.
- **TODO/FIXME/NotImplemented**:
  - None matched by grep.

### `11_Admin.py` — Admin
- **Note**: Large file; truncated.
- **Purpose**:
  - Admin console for system health, configuration, model registry info, agent registry, and possibly user management.
  - Ties into `agents.*`, `trading.agents.*`, and infra services.
- **Working status**:
  - No `TODO` markers from grep; implementation appears present.

### `12_Memory.py` — Memory Store Admin
- **Title & purpose**:
  - “🧠 Memory Store” — admin/debug page for the MemoryStore backing `trading.memory`.
- **Major sections/features**:
  - **Connection info**:
    - Calls `get_memory_store()`, shows SQLite path and session id; aborts page via `st.stop()` if unavailable.
  - **Filters & listing**:
    - Controls: memory type (`SHORT_TERM`, `LONG_TERM`, `PREFERENCE`), optional namespace/category, limit, and session id (for short-term).
    - `store.list(...)` to retrieve entries; displayed as DataFrame with columns id/type/namespace/session/key/category/value/metadata/created/updated.
  - **Edit/Delete**:
    - Text input for entry id; loads via `store.get`.
    - Allows inline editing of key/category/value/metadata with JSON parsing helpers `_to_json_text`/`_parse_json_text`.
    - Update via `store.update_entry` and delete via `store.delete_entry`.
  - **Add new entry**:
    - Forms for type/namespace/key/category/value/metadata; inserts via `store.add`.
  - **Clear memory**:
    - Dropdown to select type and optional session; calls `store.clear(memory_type, session_id=...)` and displays count cleared.
- **Notable functions**:
  - `_to_json_text(value)`, `_parse_json_text(text)` as robust JSON helpers for display/edit.
- **Working status**:
  - Fully functional debug/admin tool; heavy use of try/except around store operations.
- **TODO/FIXME/NotImplemented**:
  - None.

### `13_Scanner.py` — Market Scanner
- **Title & purpose**:
  - “🔍 Market Scanner” — screen stocks by technical and AI-driven filters (`AI Score`).
- **Major sections/features**:
  - **Backend integration**:
    - `from trading.analysis.market_scanner import scan_market, get_available_filters, DEFAULT_UNIVERSE`.
    - If import fails: displays error and `st.stop()`.
  - **Universe selection**:
    - `S&P 100`, `S&P 500`, `S&P 500 + Nasdaq 100`, `Russell 1000`, `Russell 3000`.
    - `_load_scanner_universe(universe_label)`:
      - Mirrors `0_Home` universe loading; uses Wikipedia scrapes and hard-coded Nasdaq 100; approximates Russell 1000/3000 when needed.
  - **Filter selection**:
    - `selected_filters = st.multiselect("Scan Filters", options=available_filters.keys(), default=["top_ai_score"], format_func=...)`.
    - `max_results` slider, optional custom universe textbox.
  - **Scan execution**:
    - `scan_market(filters, universe, max_results, progress_callback=_progress)`:
      - `progress_bar` and `status` updated via callback.
      - Returns dict with `results`, `scanned`, `scan_time_s`, `passed`, and optional `error`.
  - **Results rendering**:
    - DataFrame of results with renamed columns and styled `AI Score` column via `styler.map(_color_score, subset=["AI Score"])`.
    - AI Score bar chart (`plotly.express.bar`) with grade labels.
  - **Drill-down**:
    - Symbol picker; fetches 6-month history via `yfinance.Ticker(symbol).history(period="6mo")`.
    - Computes AI Score via `trading.analysis.ai_score.compute_ai_score`.
    - Displays summary and signals as DataFrame.
  - **Help view**:
    - When no scan run: prints all available filters and their descriptions.
- **Working status**:
  - Fully functional if backend scanner and AI-score modules are installed; fails fast when unavailable.
- **TODO/FIXME/NotImplemented**:
  - None.

## Trading Models (`trading/models`)

### `base_model.py` — BaseModel & TimeSeriesDataset
- **Key classes**:
  - **`ValidationError`, `ModelError`**: custom exceptions.
  - **`Recommendation` / `StrategyPayload`**: Pydantic models capturing signals and recommendation bundles.
  - **`ModelRegistry` (local)**:
    - Minimal registry mapping string names to model classes via decorator-style `register(name)`.
    - Provides `get_model(name, config)` which instantiates registered models; raises `ModelError` if missing.
  - **`TimeSeriesDataset`**:
    - PyTorch `Dataset` for time-series, wrapping a pandas DataFrame.
    - **Algorithm type**: supervised sequence-to-one dataset builder.
    - **Key parameters**:
      - `sequence_length`, `target_col`, `feature_cols`, optional `StandardScaler`.
    - Validates data for missing/inf/required columns; constructs rolling sequences and targets.
  - **`BaseModel` (abstract)**:
    - **Algorithm type**: abstract deep learning base for PyTorch models (LSTM/TCN/Transformers, etc.).
    - **Core responsibilities**:
      - Device management (CUDA vs CPU) with fallbacks.
      - Training loop (`train`), evaluation (`evaluate`), predict/forecast wrappers.
      - Optimizer (`Adam`) and scheduler (`ReduceLROnPlateau`) setup.
      - Metric computation via `model_utils.compute_metrics`.
      - Model save/load and training-history logging/plotting.
    - **Key hyperparameters (via `self.config`, with defaults)**:
      - `input_size=10`, `hidden_size=64`, `output_size=1`, `num_layers=2`, `dropout=0.1`.
      - `learning_rate=0.001`, `batch_size=32`, `sequence_length=20`.
      - Scheduler: `scheduler_patience`, `scheduler_factor`.
    - **Abstract hooks**:
      - `build_model(self) -> nn.Module`
      - `_prepare_data(self, data, is_training) -> (features, targets)`
    - **Limitations / notes**:
      - Requires PyTorch and (optionally) scikit-learn; if missing, many methods raise `ImportError`.
      - Uses `TimeSeriesDataset` which itself depends on both PyTorch and sklearn; without them, deep models are disabled.

### `model_utils.py` — Model Utility Functions
- **Algorithm type**: utility helpers, not models.
- **Key functions**:
  - `validate_data` (missing/inf/required-column checks), `to_device`, `from_device`, `safe_forward`, `compute_loss`, `compute_metrics`.
  - `get_model_confidence(val_losses)`: returns dict with `confidence`, `latest_val_loss`, `best_val_loss`, `loss_ratio`.
  - `get_model_metadata(...)`: standardized metadata dict for models.
- **Known limitations**:
  - All PyTorch-dependent helpers are guarded by `TORCH_AVAILABLE`; unavailable environment raises `ImportError` on use.

### `model_registry.py` — Smart Model Registry
- **Class**: `ModelRegistry`
  - Maintains `_models: Dict[str, Type]` and `_model_metadata: Dict[str, Dict]`.
  - **Categories**:
    - **Single-asset models**: `LSTM`, `XGBoost`, `Prophet`, `ARIMA`, `Ensemble`, `TCN`, `GARCH`, `Transformer`, `CatBoost`, `Ridge`, `Hybrid`.
    - **Multi-asset models**: `GNN` (graph-based, portfolio-level).
    - **NeuralForecast models**: `Autoformer`, `Informer`, `TFT`, `N-BEATS`, `PatchTST`, `N-HiTS` (when `NEURALFORECAST_AVAILABLE`).
  - **Metadata per model**:
    - `type`: `single_asset` or `multi_asset`.
    - `complexity`: `low`/`medium`/`high`.
    - `description`: short human-readable description.
    - `use_case`: `general`, `seasonal`, `stationary`, `volatility`, `portfolio`, `baseline`.
    - `requires_gpu`: bool (not enforced, but documented).
    - `min_data_points`, `min_assets`/`max_assets` (for GNN), optional `best_for`.
  - **Helper methods**:
    - `register(name, model_class, metadata)`, `get(name)`, `list_models(filter_by)`, `get_quick_forecast_models()`, `get_advanced_models()`, `get_multi_asset_models()`, `get_model_info(name)`, `list_all_info()`.
  - **Known limitations**:
    - Import errors for individual models are logged but do not break the registry; missing models are simply not registered.
    - NeuralForecast suite is entirely optional (`pip install neuralforecast` required).

### `ensemble_model.py` — EnsembleModel
- **Model name & algorithm type**:
  - `EnsembleModel(BaseModel)` — **ensemble forecaster** combining sub-models (Ridge, XGBoost, etc.).
  - Works as a meta-model managing multiple `trading.models` implementations.
- **Key hyperparameters / config keys**:
  - `models`: list of sub-model configs (each with `name`, `class_path`, `target_column`, etc.); defaults to Ridge + XGBoost if absent.
  - `voting_method`: `"mse"`, `"sharpe"`, `"regime"`, or `"custom"`.
  - `weight_window`: lookback window for performance-based weighting.
  - `fallback_threshold`: confidence threshold per sub-model.
  - `ensemble_method`: `"weighted_average"` (default) or `"vote_based"`.
  - `dynamic_weighting`: bool; triggers `_update_weights`.
  - `regime_detection`: bool; enables regime-aware routing.
  - `weight_factors` (optional): weights for current performance, historical performance, confidence, trend alignment, volatility adjustment.
- **Behavior**:
  - Initializes `self.models` via global `get_registry()`; handles unknown models gracefully.
  - Computes sub-model scores (MSE/Sharpe/custom), confidences, and maintains rolling `performance_history`.
  - `_calculate_dynamic_weights` fuses multiple signals (current/historical performance, confidence, trend, volatility).
  - `_get_strategy_recommendation` identifies market regime (`bull`/`bear`/`neutral`) and best model per regime; persists in `memory.json` (`model_strategy_patterns`).
  - `predict`:
    - Ensures all sub-model predictions have aligned lengths (trims or pads as necessary).
    - Applies ensemble method, then final “price-space guard” to clamp outlandish predictions relative to last close.
  - `forecast`:
    - Auto-cached via `@cache_model_operation`.
    - Iteratively calls `predict` to build a multi-step forecast; returns `already_denormalized=True` with weights and strategy patterns.
- **Known limitations / comments**:
  - Relies on sub-models exposing `predict`, optional `calculate_confidence`, `shap_interpret`, etc.; missing methods are handled with fallbacks.
  - Requires `trading.utils.safe_math.safe_returns` and several other utilities from the broader system.

### `arima_model.py` — ARIMAModel
- **Model name & algorithm type**:
  - `ARIMAModel(BaseModel)` — classical ARIMA/SARIMA time-series forecaster (statsmodels/pmdarima).
- **Key hyperparameters** (via `config` and attributes):
  - `order`: `(p, d, q)`; default `(1, 1, 1)` if not provided and `use_auto_arima` disabled.
  - `seasonal_order`: optional seasonal order for SARIMAX.
  - `use_auto_arima`: bool (default `True`, unless explicit `order` passed to constructor).
  - `seasonal`: bool; whether to use seasonal components (`m` > 1).
  - `optimization_criterion`: `"aic"`, `"bic"`, `"mse"`, or `"rmse"`.
  - `auto_arima_config`: additional pmdarima kwargs (search ranges, stepwise, etc.).
  - `backtest_steps`: number of steps for backtest-based optimization (for `mse`/`rmse`).
- **Core methods**:
  - `fit(data)`: extracts a 1D price series from DataFrame (`close`/`Close`/first column), requires at least 20 points; chooses between enhanced auto_arima or manual ARIMA.
  - `_fit_enhanced_auto_arima`: uses pmdarima with:
    - Built-in AIC/BIC optimization.
    - Optional MSE/RMSE backtest optimization via `_optimize_with_backtest`.
  - `_fit_manual_arima`: manual statsmodels ARIMA/SARIMAX; logs AIC/BIC.
  - `predict(steps, confidence_level)`: step-ahead forecast with confidence intervals from statsmodels or pmdarima, returning dict with predictions & CI.
  - `forecast(data, horizon)`: decorated with `@safe_forecast`; auto-fits if needed, ensures continuity with last price (offset if >2% gap), and returns fully denormalized price forecast plus CI arrays and AIC/BIC.
- **Known limitations / comments**:
  - Strong dependency on pmdarima/statsmodels; when pmdarima missing, falls back to manual ARIMA.
  - Uses ADF test (`check_stationarity`) and `find_best_order` helpers for diagnostic workflows.

### `prophet_model.py` — ProphetModel
- **Model name & algorithm type**:
  - `ProphetModel(BaseModel)` — Facebook/Meta Prophet wrapper for daily-seasonal time-series with holidays and macro support.
  - If `prophet`/`fbprophet` is missing, the placeholder class raises an instructive `ImportError`.
- **Key hyperparameters / config**:
  - `prophet_params`: passed through to Prophet (seasonality mode, priors, etc.).
  - `add_holidays`: bool; if `True` and `holidays` library installed, adds country holidays.
  - `holiday_country`: default `"US"`.
  - `changepoint_prior_scale`, `seasonality_prior_scale`: with defaults (0.05, 10.0).
  - `date_column`: default `"date"` or `"ds"` depending on config.
  - `target_column`: default `"close"`.
  - `default_horizon`, `max_horizon`: for dynamic forecast horizon.
- **Core behavior**:
  - `build_model`: ensures `self.model` is a Prophet instance; fallback basic instance if init failed earlier.
  - `_prepare_data`: maps generic DataFrame to `ds`/`y` arrays; ensures datetimes and numeric target.
  - `_validate_prophet_config`: strict validation of holiday DataFrame, seasonality flags, and priors.
  - `fit(train_data, val_data)`: normalizes OHLCV columns, handles date from column or index, strips tz info, and fits underlying Prophet model.
  - `_get_country_holidays(country)`: builds holidays DataFrame when `holidays` package is available.
  - `predict(data, horizon)`: robust guard rails:
    - Validates non-empty, valid date column, no NaNs, and `self.fitted`.
    - Uses `calculate_forecast_horizon` when horizon is None.
  - `forecast(data, horizon)`: returns fully denormalized forecast with `yhat`, `yhat_lower`, `yhat_upper`, horizon metadata, and `already_denormalized=True`.
- **Known limitations / comments**:
  - Heavy reliance on Prophet’s own internals; certain configuration errors surface as `ValueError` via `_validate_prophet_config`.
  - On initialization or fit failure, `available` is set `False` and `forecast` returns a simple low-confidence fallback.

### `catboost_model.py` — CatBoostModel
- **Model name & algorithm type**:
  - `CatBoostModel(BaseModel)` — gradient boosting regressor (CatBoost) for time-series, predicting next-period returns and rebuilding prices.
- **Key hyperparameters**:
  - `config["catboost_params"]` passed directly to `CatBoostRegressor(**params)`; not hardcoded here, but typical CatBoost params apply (`depth`, `learning_rate`, `l2_leaf_reg`, etc.).
  - `feature_columns`: list of features; `target_column` (default `"target"`, but often `"Close"`/`"close"` at runtime).
- **Core behavior**:
  - `_normalize_columns`: reconciles lowercase `open/high/low/close/volume` to title case for yfinance data.
  - `_prepare_data`: robust mapping from DataFrame to `X, y`:
    - Automatically picks target column (`target_column`, `Close`, `close`, or last numeric).
    - Auto-detects usable feature columns if config mismatched.
  - `fit(train_data, val_data)`:
    - Trains on **next-period returns** rather than price:
      - Computes returns from `price_col` (`Close`/`close`).
      - Uses next-step return as target; features come from aligned subset and optional `eval_set`.
    - Sets `self.fitted = True`.
  - `predict(data, horizon=1)`:
    - Predicts returns for the given feature DataFrame.
  - `forecast(data, horizon)`:
    - Iteratively:
      - Uses `predict` to get next return `r`, clipped to [-20%, +20%].
      - Updates price with `current_price * (1 + r)` and appends to forecast.
    - Computes approximate confidence bands using residual std or 2% of last price with `sqrt(time)` scaling; returns `already_denormalized=True`.
- **Known limitations / comments**:
  - Heavily dependent on reasonable `feature_columns` mapping; multiple fallbacks try to recover from misconfiguration.
  - Assumes availability of CatBoost; if missing, import will fail before model creation.

### `xgboost_model.py` — XGBoostModel
- **Model name & algorithm type**:
  - `XGBoostModel(BaseModel)` — gradient boosting forecaster with robust error handling and multiple fallbacks.
  - Internally supports:
    - Primary model: `xgboost.XGBRegressor` (if available).
    - Fallback model: `FallbackXGBoostModel` using `RandomForestRegressor` + `StandardScaler`.
- **Key hyperparameters**:
  - Loaded via `_load_hyperparameters`:
    - Defaults: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `random_state`, `objective="reg:squarederror"`, `eval_metric="rmse"`.
    - Augmented by `trading.utils.gpu_utils.get_xgboost_device()` for GPU support when available (`tree_method`, `device`, etc.).
    - Optionally overridden by YAML configs (`config/xgboost_config.yaml`, etc.) or environment variables (`XGB_*`).
  - `early_stopping_rounds` is **not** passed in constructor; passed to `fit` only when there is a validation set.
- **Core behavior**:
  - `build_model`: constructs a conservative XGBRegressor; `_setup_model` later applies full parameter set.
  - `_create_lag_features`:
    - Builds lagged close/volume series, rolling statistics (mean/std/min/max), and technical indicators (via `_add_technical_indicators`).
    - Adapts max lags and windows based on data length to avoid dropping all rows.
  - `_add_technical_indicators`:
    - RSI via `safe_rsi`, MACD, Bollinger bands, price-change features, volatility, volume MA, volume ratio (safe division).
  - `prepare_features`:
    - Validates DataFrame, then build features/target where target is **next-period return**.
  - `train(data)` / `fit(data)`:
    - Prepares features, selects top features via correlation ranking, scales with `StandardScaler`.
    - If enough samples, splits into train/validation and uses early stopping.
    - On XGBoost failure, falls back to RandomForest.
  - `predict(data, return_returns=False)`:
    - Returns either raw predicted returns (for internal use) or fully denormalized price estimates by applying returns to base prices.
    - Multi-level fallbacks if model/feature/state issues occur.
  - `forecast(data, horizon)`:
    - Decorated with `@cache_model_operation` and `@safe_forecast`.
    - Iteratively propagates predicted returns forward to build a **price** forecast, with 95% CI built from residual or heuristic vol; `already_denormalized=True`.
    - On error, returns progressively simpler fallbacks: price-trend extrapolation, then flat-line last price.
- **Known limitations / comments**:
  - Requires scikit-learn; without it, `available=False` and only crude fallbacks are used.
  - Many debug `print` calls around failure points; safe in a console environment but noisy in logs.

### Other models
- **`garch_model.py`**:
  - **Algorithm type**: GARCH volatility model (exact implementation not shown here).
  - Registered in `ModelRegistry` as `GARCH` with `type='single_asset'`, `use_case='volatility'`, `complexity='medium'`, `min_data_points=100`.
- **`tcn_model.py`**:
  - **Algorithm type**: Temporal Convolutional Network for single-asset forecasting; high complexity deep model requiring PyTorch.
- **`lstm_model.py`**:
  - **Algorithm type**: LSTM-based forecaster; `ModelRegistry` checks whether `LSTMModel` extends `BaseModel` and otherwise falls back to `LSTMForecaster`.
- **`transformer_wrapper.py`, `transformer_model.py`, `advanced/transformer/time_series_transformer.py`**:
  - **Algorithm type**: Transformer-based time-series models; registered under `"Transformer"` in `ModelRegistry` with `requires_gpu=True`.
- **`neuralforecast_models.py`**:
  - Wraps NeuralForecast implementations: `AutoformerModel`, `InformerModel`, `TFTModel`, `NBEATSModel`, `PatchTSTModel`, `NHITSModel`.
  - Each model’s metadata describes its best use-case (`long-term`, `very long sequences`, interpretability, etc.).
- **`ridge_model.py`**:
  - **Algorithm type**: Ridge regression baseline; low complexity, few data points needed.
- **`dataset.py`**:
  - Additional dataset utilities (beyond `TimeSeriesDataset` in `base_model.py`).
- **`forecast_router.py`, `forecast_features.py`, `forecast_normalizer.py`, `forecast_explainability.py`, `multi_model_aggregator.py`, `timeseries/__init__.py`, `walk_forward_validator.py`, etc.**:
  - Router and utilities orchestrating multiple models based on use-case, horizon, and data properties.
  - Provide model-agnostic feature engineering, normalization, explainability, and walk-forward testing.
- **Limitations for these entries**:
  - Due to size, individual files were not fully walked; metadata sourced from `ModelRegistry` and naming conventions.

## Trading Analysis (`trading/analysis`)

- **`market_monitor.py`**:
  - Scans watchlists for volume/price spikes (`scan_watchlist`), exposes `DEFAULT_WATCHLIST`.
  - Used by Home page for live event feed.
- **`chart_builder.py`**:
  - Constructs event-centric charts for price + volume (and possibly overlays).
- **`event_news_fetcher.py`**:
  - Fetches news items around a given event timestamp and symbol/company name.
- **`news_ranker.py`**:
  - Ranks news articles by relevance to symbol, direction, and context; used to pick top headlines.
- **`ai_score.py`**:
  - Computes a composite AI Score (1–10) and letter grade for a symbol using multiple signals; used in Home and Scanner pages.
- **`market_scanner.py`**:
  - Provides `scan_market`, `get_available_filters`, and `DEFAULT_UNIVERSE`; core of `13_Scanner.py`.
- **`monte_carlo.py`, `factor_model.py`, `volume_news_linker.py`, `news_ranker.py`**:
  - Support risk/return Monte Carlo simulations, factor analysis, and linking unusual volume to news.
- **Known status**:
  - All functions are imported and used with error guards; modules are assumed operational but not line-audited here.

## Trading Data (`trading/data`)

- **Core modules**:
  - `data_provider.py`, `data_loader.py`, `data_listener.py`, `preprocessing.py`:
    - Abstractions for fetching, caching, streaming, and pre-processing time-series data.
  - `macro_data_integration.py`, `external_signals.py`:
    - Integrate macro and external signals into the feature space.
  - `news_fetcher.py`, `news_aggregator.py`:
    - Aggregation and normalization of news for Chat and Home pages.
  - `earnings_calendar.py`, `earnings_reaction.py`, `short_interest.py`, `insider_flow.py`, `watchlist.py`:
    - Fundamental events and flows used by Home/Alerts/Scanner.
- **Providers (`trading/data/providers`)**:
  - `alpha_vantage_provider.py`, `yfinance_provider.py`, `fallback_provider.py`, `base_provider.py`:
    - Pluggable data providers with a unified interface and failover logic.
- **Known status**:
  - Widely used across pages; failures are consistently caught at call sites, preventing UI crashes.

## Agents

- **Top-level `agents/`**:
  - `agent_controller.py`, `task_router.py`, `registry.py`, `orchestrator.py`:
    - High-level orchestration of LLM- and tool-based agents; route tasks between trading domain agents and infra services.
  - `agent_config.py`: global configuration of available agent types.
  - `llm/agent.py`:
    - Core LLM agent factory (`get_prompt_agent`), used in Home/Chat for narrative generation.
    - Tracks active LLM calls, model loader, and provider interfaces.
  - `llm_providers/anthropic_provider.py`, `local_provider.py`:
    - Provider-specific wrappers for Anthropic and local models.
  - `implementations/*`:
    - Concrete agents for model benchmarking, research fetching, etc.
  - `model_innovation_agent.py`: higher-level agent for exploring new model ideas.
- **`trading/agents`**:
  - Strategy/meta-strategy agents: `strategy_selector_agent`, `strategy_switcher`, `meta_strategy_agent`, `strategy_improver_agent`, `performance_critic_agent`.
  - Model lifecycle agents: `model_builder_agent`, `model_optimizer_agent`, `model_selector_agent`, `model_improver_agent`, `model_evaluator_agent`.
  - Execution/risk agents: `execution/trade_signals`, `execution/position_manager`, `execution/risk_controls`, `execution_risk_agent`, `execution/execution_providers`.
  - Operational agents: `agent_manager`, `agent_registry`, `agent_loop_manager`, `rl_trainer`, `data_quality_agent`, `regime_detection_agent`, `commentary_agent`, `upgrader/*`, `updater_agent`.
  - Dashboards: `task_dashboard.py`, `leaderboard_dashboard.py`.
- **`system/infra/agents`**:
  - Web dashboards (`web/dashboard.py`, `middleware.py`), auth (`auth/user_manager.py`, `session_manager.py`), notifications (`notifications/*`), automation services/APIs/logging/metrics.
- **Working status**:
  - Many of these agents are called indirectly by pages and background services; they are not UI-facing but are integral to orchestration.
  - Error handling at call sites in pages (`Home`, `Chat`, `Alerts`, etc.) ensures failures do not surface directly to users.

## Components (`components/`)

- **`watchlist_widget.py`**:
  - Provides `render_watchlist()` used on the Home page Watchlist section.
  - Likely supports adding/removing symbols, persistence via `config.user_store`, and integration with watchlist-based alerts.
- **`multi_timeframe_chart.py`**:
  - Component for synchronized multi-timeframe charts; may be used by Strategy Testing, Performance, or Scanner for detailed symbol views.
- **`news_candle_chart.py`**:
  - Combines candlestick price chart and aligned news markers; used for event-driven analysis (e.g., Home event feed or Scanner drilldown).
- **`onboarding.py`**:
  - Onboarding flow with educational content and/or first-run setup for preferences and watchlists.
- **`__init__.py`**:
  - Likely re-exports key components or sets package metadata.
- **Working status**:
  - All components are imported from pages with try/except guards to avoid hard failures when a component is missing.

---

**Overall status summary**

- **Pages**: All core pages (`0_Home` through `13_Scanner`) are implemented and wired to their respective backends. Only one intentional `NotImplementedError` (blocking arbitrary custom strategy code for security), and a single `TODO` to wire live benchmark data in Performance.
- **Models**: `trading/models` contains a mature, heterogeneous model library (statistical, ML, deep, and ensemble), centrally managed via `ModelRegistry` with metadata-driven filtering and clear fallbacks when optional dependencies are missing.
- **Analysis/Data/Agents/Components**: These directories house the supporting infrastructure for scanning, forecasting, memory, monitoring, and orchestration; pages depend on them heavily but guard against missing modules and external services to keep the UI resilient.
