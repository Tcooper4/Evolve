## Model Research Tool

- **Triggers in `pages/8_Model_Lab.py`**: There is **no explicit “Model Research Tool” UI** or button in `8_Model_Lab.py`. The large page defines 11 tabs:
  - `tab1` — **⚡ Quick Training**
  - `tab2` — **⚙️ Model Configuration**
  - `tab3` — **🎯 Hyperparameter Optimization**
  - `tab4` — **📊 Model Performance**
  - `tab5` — **🔍 Model Comparison**
  - `tab6` — **🧠 Explainability**
  - `tab7` — **📚 Model Registry**
  - `tab_monitoring` — **📊 Model Monitoring**
  - `tab_discovery` — **🤖 AI Model Discovery**
  - `tab_innovation` — **🧪 Model Innovation**
  - `tab_benchmark` — **📊 Benchmark Models**
- **Closest related features**:
  - **AI Model Discovery (tab_discovery)**:
    - UI: heading “🤖 AI-Powered Model Discovery”, descriptive bullet list, and a **“🔍 Discover Best Model”** primary button.
    - Backend: attempts to import `trading.agents.model_discovery_agent.ModelDiscoveryAgent`.
      - If import fails: shows “Model discovery is not available (agent rationalized).”
      - If available:
        - Instantiates and caches `st.session_state.model_discovery_agent`.
        - Expects `st.session_state.forecast_data` (loaded by the Forecasting page); otherwise errors “Please load data first in the Forecasting page”.
        - Config:
          - `max_models` (slider “Maximum models to try”)
          - `time_budget` (slider “Time budget (minutes)”)
          - `search_strategy` (select `Balanced` / `Accuracy-Focused` / `Speed-Focused`)
          - `include_ensembles` (checkbox)
        - On click:
          - If agent has `discover_best_model`, calls it with:
            - `data=forecast_data`, `target_column='close'`, and the above arguments.
          - Else if agent has `run_discovery`, calls it and adapts results into a best-model summary.
        - **Sources searched**: the discovery agent works entirely on **local model types and data**, not on research databases; it does not invoke any `ArxivResearchFetcher` or other web research sources.
        - **UI output**:
          - “🏆 Recommended Model” metrics (type, score, train time).
          - Parameters (if provided).
          - A table of all models evaluated (model name, score, train_time, predictions_per_sec) and a bar chart of scores.
          - Optional “💡 AI Insights” text.
          - A “💾 Save Best Model Config” button that stores `best_model` into `st.session_state.discovered_model_config` for later reuse.
  - **Model Innovation (tab_innovation)**:
    - UI: heading “🧪 AI Model Innovation Lab” and text “AI automatically designs novel model architectures tailored to your data”.
    - Backend: imports `agents.model_innovation_agent.ModelInnovationAgent` and `InnovationConfig`.
      - Stores one instance in `st.session_state.model_innovation_agent`.
      - Requires `st.session_state.training_data` from Quick Training; otherwise warns.
      - Config:
        - `base_architecture` select (`LSTM`, `Transformer`, `CNN`, `Hybrid`).
        - `innovation_level` slider (1–10).
      - Button: **“🚀 Generate Novel Architecture”**.
        - If agent has `analyze_data`, uses it; else builds simple `data_characteristics`.
        - If agent has `innovate_architecture`, calls it with `data`, `base_architecture`, `innovation_level`, and `data_characteristics`.
        - Else, if it has `discover_models`, uses results to synthesize a **pseudo-innovation result**.
        - **UI output**:
          - “🏗️ Architecture Design” text and optional image.
          - “💻 Implementation” code block of `implementation_code`.
          - “📊 Expected Performance” metrics (estimated accuracy, training time, complexity score).
          - “💡 Key Innovations” bullet list.
          - “🎯 Train This Model” button which simply stores `result` into `st.session_state.innovation_model` for future use.
  - **Model Benchmarking (tab_benchmark)**:
    - UI: heading “📊 Model Benchmarking”.
    - Backend:
      - Imports `agents.implementations.model_benchmarker.ModelBenchmarker`, `BenchmarkResult`, and `agents.implementations.implementation_generator.ModelCandidate`.
      - Requires `st.session_state.forecast_data` as `benchmark_data`.
      - Uses `trading.models.model_registry.get_registry().get_advanced_models()` to build a list of model names.
      - For each of up to 5 models:
        - Wraps it in a `ModelCandidate` (with placeholder implementation details) and calls `benchmarker.benchmark_model(candidate)`.
      - **UI output**:
        - Benchmark results table (per model: overall_score, r2_score, mse, mae, sharpe_ratio, max_drawdown, training_time, inference_time, memory_usage).
        - Bar charts for overall benchmark score and R² comparison.
        - “🏆 Best Performing Model” metrics.
- **Actual Model Research Tool implementation**:
  - **File**: `agents/implementations/research_fetcher.py`.
  - **Class**: `ArxivResearchFetcher`.
    - Sources: **arXiv API via HTTP** (`http://export.arxiv.org/api/query`).
    - Search terms (default): time series forecasting, financial prediction, machine learning trading, neural networks forecasting, quantitative finance, market prediction, deep learning time series, reinforcement learning trading.
    - Uses `aiohttp` for async HTTP and a local JSON cache `agents/research_cache.json` keyed by `search_term` and `days_back`.
    - Methods:
      - `fetch_papers_async(search_term)`: queries arXiv, parses XML by regex, returns a list of `ResearchPaper` dataclasses.
      - `_calculate_relevance_score(title, abstract, categories)`: keyword- and category-based scoring, normalized to \[0, 1\].
      - `_assess_implementation_complexity(...)`: classifies as `"low"`, `"medium"`, or `"high"` based on keywords like “transformer”, “reinforcement”, etc.
      - `_assess_potential_impact(...)`: infers `"low"`, `"medium"`, or `"high"` impact based on words like “state-of-the-art”, “novel”, etc., boosted by relevance score.
      - `fetch_recent_papers()`: concurrently fetches for all search terms, deduplicates by arXiv ID, sorts by relevance, and returns top `max_results`.
  - **Integration points**:
    - `agents/implementations/implementation_generator.py` imports `ResearchPaper` for use in generating model implementations from research.
    - `agents/implementations/__init__.py` re-exports `ArxivResearchFetcher` as part of the implementations package.
  - **Where it is called from the UI**:
    - A full-code search shows **no direct call from `pages/8_Model_Lab.py`** or any other Streamlit page to `ArxivResearchFetcher` or a “Model Research” tab/button.
    - The research fetcher is therefore an **infrastructure/tooling module** used by backend agents (for auto-evolutionary model generation) and is **not yet surfaced as a dedicated UI “Model Research Tool”**.
- **Session state and caching**:
  - In `research_fetcher.py`:
    - Maintains an in-memory `paper_cache` plus JSON file `agents/research_cache.json`.
    - Cache key: `{search_term}_{days_back}`; populated after successful requests and reused on later calls.
    - This cache is **independent of Streamlit session state**.
  - In Model Lab tabs:
    - **AI Model Discovery** and **Model Innovation** maintain their own agents in `st.session_state` (`model_discovery_agent`, `model_innovation_agent`) and reuse shared data (`forecast_data`, `training_data`), but **they do not connect to the ArxivResearchFetcher**.

**Conclusion**: The **Model Research Tool exists as `ArxivResearchFetcher` (arXiv client + scoring + caching)** but is **not wired directly into the Model Lab UI**. The closest visible features—AI Model Discovery, Model Innovation, and Model Benchmarking—use local agents and registries, not external research APIs. There is currently **no tab or button in `pages/8_Model_Lab.py` that exposes a research browser over arXiv/Papers With Code/web to the end user.**

## Page: 2_Forecasting.py — Complete Tab & Feature Inventory

_Not yet fully audited in this pass. Requires chunked reading of the 3,179-line file to enumerate all `st.tabs`, `st.expander`, `st.sidebar`, major workflows, external integrations, and stub features._

## Page: 3_Strategy_Testing.py — Complete Tab & Feature Inventory

_Not yet fully audited in this pass beyond confirming the intentional `NotImplementedError` that disables arbitrary custom code execution. A full tab/feature inventory will require chunked reads of the 3,655-line file._

## Page: 4_Trade_Execution.py — Complete Tab & Feature Inventory

_Pending full read; this section should enumerate all tabs, order-entry workflows, execution agents, risk hooks, and any stubbed broker integrations._

## Page: 5_Portfolio.py — Complete Tab & Feature Inventory

_Pending full read; will document all portfolio views, allocation charts, P&L breakdowns, and any integration with risk/reporting pages._

## Page: 6_Risk_Management.py — Complete Tab & Feature Inventory

_Pending full read; will cover VaR/CVaR, exposure limits, rule editors, and connections to execution risk agents._

## Page: 7_Performance.py — Complete Tab & Feature Inventory

_Pending full read; beyond the known benchmark TODO, needs a complete inventory of performance dashboards, attribution tools, and any stubbed features._

## Page: 8_Model_Lab.py — Complete Tab & Feature Inventory

**Tabs and high-level purpose**

- **⚡ Quick Training (tab1)**:
  - Data source selector: “Load from Market”, “Upload CSV File”, “Use Previous Data”.
  - Market loader uses `trading.data.data_loader.DataLoader` and `DataLoadRequest` (with `YFinanceProvider` or `ProviderManager`), storing results in `st.session_state.training_data`.
  - CSV upload parses various date-column names, sets index, validates presence of `Close`/`close`.
  - Reuses previously loaded data when selected.
  - Model selection: `LSTM`, `XGBoost`, `Prophet`, `ARIMA`.
  - Data preview: `st.expander("📋 Data Preview")` shows head and date range.
  - Training configuration:
    - Target column select (close/price/adj close).
    - Train/test split slider.
    - Forecast horizon.
    - Model name.
  - “🚀 Train Model” button:
    - Prepares target series, splits into train/test.
    - Instantiates the chosen model with default config:
      - LSTM: sequence_length=60, hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001.
      - XGBoost: basic tree params.
      - Prophet: date/target and Prophet params (changepoint/seasonality).
      - ARIMA: order=(5,1,0) with `use_auto_arima=True`.
    - Trains model (special cases for Prophet/ARIMA vs LSTM/XGBoost DataFrame inputs).
    - Evaluates on test set and computes MSE, MAE, RMSE, MAPE, R².
    - Stores results in `st.session_state.quick_training_results[model_name]`.
    - Optional logging into `st.session_state.model_log` and `st.session_state.perf_logger`.
    - UI:
      - Metric cards (RMSE, MAE, MAPE, R², MSE).
      - Plotly line chart of actual vs predicted on test set.
      - “💾 Save Model” (saves into `st.session_state.saved_models`).
      - “🔄 Train Another Model” reruns page.
- **⚙️ Model Configuration (tab2)**:
  - Reuses `st.session_state.training_data` when present; otherwise offers its own market data loader.
  - **Model architecture section**:
    - `model_type` select (LSTM/XGBoost/Prophet/ARIMA).
    - For each model:
      - LSTM:
        - Input/hidden dims, num_layers, sequence_length.
        - Dropout, learning_rate, batch_size, epochs.
        - Advanced expander “🔧 Advanced LSTM Options” with bidirectional, batch/layer norm, additional dropout.
      - XGBoost:
        - n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda.
      - Prophet:
        - changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale.
        - seasonality_mode, yearly/weekly/daily seasonality flags.
      - ARIMA:
        - p, d, q, use_auto_arima, seasonal flag, seasonal_periods.
  - **Feature Engineering Pipeline**:
    - Technical indicators toggle and multi-select (SMA, EMA, RSI, MACD, Bollinger, Volume).
    - MA windows, lag features (lag periods), rolling features (rolling windows), time features, normalization (Standard/MinMax/Robust).
  - **Data splitting & validation**:
    - Sliders for train%, val%, derived test%, optional cross-validation with `cv_folds`.
  - **Training configuration**:
    - Target column select, model name.
    - “🚀 Train Model with Configuration” button:
      - Applies feature engineering via `trading.data.preprocessing.FeatureEngineering` and `DataPreprocessor` when available; falls back gracefully.
      - Splits target into train/val/test by percentage.
      - Instantiates appropriate model and trains it (with simplified inputs for non-Prophet/ARIMA).
      - Recomputes metrics and persists to `st.session_state.configured_models[model_name]`.
      - Shows metrics, prediction-vs-actual chart, and “💾 Save Model” to `st.session_state.saved_models`.
- **🎯 Hyperparameter Optimization (tab3)**:
  - Reuses `st.session_state.training_data`.
  - Model selection and target column select.
  - Optimization method: **Grid Search**, **Random Search**, **Bayesian Optimization (Optuna)**, **Genetic Algorithm**.
  - Global config:
    - Optimization objective (RMSE/MAE/MSE/R²/MAPE).
    - n_trials, timeout, early stopping and patience.
  - **Search Space**:
    - LSTM: hidden_dim range, num_layers choices, dropout range, learning_rate choices.
    - XGBoost: n_estimators range, max_depth choices, learning_rate range, subsample range.
    - Prophet: changepoint_prior_scale range, seasonality_prior_scale range, seasonality_mode choices.
    - ARIMA: p range, d options, q range.
  - **“🚀 Start Optimization” button**:
    - Defines an `objective_function(params)` that:
      - Instantiates appropriate model with those params.
      - Trains and evaluates on an 80/20 train/test split.
      - Returns the chosen metric (negated for R² when “Maximize”).
    - Runs the chosen optimization algorithm:
      - Grid: all parameter combinations, limited to `n_trials`.
      - Random: draws random params per trial.
      - Bayesian (Optuna): uses `optuna.create_study` and `study.optimize`; handles missing Optuna gracefully.
      - Genetic: DEAP-based GA or fallback to random search if DEAP missing.
    - UI:
      - Live progress bar, status, best params JSON, and convergence history line chart.
      - Final summary: best score, trial count, optimization time; table of all trials; CSV download; “💾 Save Best Parameters” into `st.session_state.optimization_results`.
- **📊 Model Performance (tab4)**:
  - Aggregates models from `quick_training_results`, `configured_models`, and `saved_models`.
  - Model select; shows:
    - Info metrics (model type, train/test sizes).
    - Nested tabs:
      - **📈 Training Metrics**: metric cards; optional training/validation loss & accuracy curves; optional epoch-level table.
      - **✅ Validation Metrics**: prediction vs actual scatter, residual plots and distributions, residual stats.
      - **🧪 Test Set Evaluation**: test metrics and error analysis over time and distribution.
      - **📉 Performance Over Time**: metric histories (RMSE/MAE/R²) plus drift/degradation detection (RMSE change).
  - Export:
    - “📥 Export Performance Report” builds JSON report and exposes a download button.
    - “🔄 Re-evaluate Model” is currently a **stub** with only info text.
- **🔍 Model Comparison (tab5)**:
  - Requires at least two models from the same sources as above.
  - UI:
    - Multi-select of models to compare.
    - Comparison metrics table (RMSE/MAE/MAPE/R²/MSE/train/test size).
    - “🏆 Best Models by Metric” metrics (per metric best model).
    - Optional **Advanced Model Scoring** section:
      - Button “Calculate Comprehensive Scores” uses `trading.utils.metrics.scorer.ModelScorer`, if available.
      - For each model with predictions/actuals, computes overall and sub-scores (accuracy, reliability, robustness, efficiency).
      - UI: scores table, radar chart, and recommendation text with “Model Strengths” expander.
    - Performance charts overlay (bar/line charts for metrics across models).
    - Prediction overlay: actual vs each model’s predictions in one chart.
    - Statistical significance tests: pairwise t-tests on absolute errors via `scipy.stats.ttest_rel`.
    - Model selection recommendation: normalized metric-based scores and recommended model.
    - Ensemble creation:
      - Options: Average, Weighted Average, Voting.
      - Weighted method allows sliders per model and auto-normalises weights.
      - “🚀 Create Ensemble” builds ensemble predictions, computes metrics, stores result in `st.session_state.ensemble_models`, and overlays ensemble in comparison table.
- **🧠 Explainability (tab6)**:
  - Reuses models from `quick_training_results`, `configured_models`, and `saved_models`.
  - For a selected model, offers sub-tabs:
    - **🔢 Feature Importance**:
      - Importance method select (`Model-Specific`, `Permutation`, `SHAP`, `Correlation`).
      - “Calculate Feature Importance”:
        - For XGBoost: tries `feature_importances_` or booster gain; falls back to synthetic importances.
        - For other models: uses synthetic importances.
      - UI: horizontal bar chart and table of ranked features.
    - **📈 SHAP Values**:
      - Checks for `shap` library; otherwise warns.
      - Uses synthetic sample data and either TreeExplainer for XGBoost or random SHAP values.
      - SHAP plot types:
        - Summary plot (mean |SHAP| bar chart + distributions).
        - Dependence plot for chosen feature.
        - Waterfall plot for one sample (force-like with base + SHAP).
        - Force plot surrogate using bar chart of positive vs negative contributions.
    - **🍋 LIME Explanations**:
      - Checks for `lime` library; otherwise warns.
      - Generates synthetic `X_sample` and uses `LimeTabularExplainer` with a simple `predict_fn` wrapping the model.
      - Shows bar chart and table of LIME contributions for a selected instance.
    - **📉 Partial Dependence**:
      - Generates synthetic feature values and partial dependence (sine curve) for a selected feature.
    - **🎯 Individual Predictions**:
      - For models with `predictions` and `actuals`:
        - Selects an index and shows metrics (predicted, actual, error).
        - Generates synthetic feature contributions and displays bar chart plus explanation text.
  - Additionally, at bottom:
    - “🔬 Model Behavior Analysis” expander summarizing metrics, interpretability, complexity, feature sensitivity, and non-linearity qualitatively based on model_type.
- **📊 Model Monitoring (tab_monitoring)**:
  - Depends on `st.session_state.model_monitor` and optionally `st.session_state.model_log`.
  - Features:
    - “🔍 Model Drift Detection” calling `monitor.check_drift()` and showing drift score & recommendation.
    - “📈 Performance Trends”:
      - Uses `model_log.get_recent_logs` to plot R² over time and summarise metrics per `model_name`.
    - “🚨 Model Alerts” from `monitor.get_active_alerts()`.
  - When monitor/log are missing, quietly shows info messages.
- **📚 Model Registry (tab7)**:
  - Unified registry combining:
    - Quick/configured/saved/ensemble models and `model_registry_storage`.
  - Session state for `model_registry_storage`, `model_versions`, `model_deployment_status`.
  - Sub-tabs:
    - **📋 Model Library**:
      - Search, filter (by type), sort (Name/Date/Performance).
      - Table with model name, type, RMSE, R², trained date, deployment status.
      - Actions:
        - View Details (sets selection and reruns).
        - Load Model (into `current_model`).
        - Delete Model (removes from all origin stores).
        - Copy Metadata (shows JSON).
    - **📝 Model Details**:
      - Metadata (name, type, version, trained_at).
      - Performance metrics.
      - Model configuration JSON and parameter table.
      - Data info (train/test/val sizes).
      - Notes & tags (editable, saved in `model_notes`/`model_tags`).
      - Model lineage and version history.
    - **🚀 Deployment**:
      - Model select; shows metrics and current deployment status from `model_deployment_status`.
      - Buttons to deploy to Production/Staging, undeploy; updates status.
      - Deployment history table (if present); deployment configs stored in `deployment_configs`.
    - **📤 Export/Import**:
      - Export selected models as JSON; placeholders for Pickle/ONNX.
      - Import JSON of models and optionally add them to `imported_models`.
      - Model comparison history table from `optimization_results` (model/hyperopt combos).
- **🤖 AI Model Discovery (tab_discovery)**:
  - See **Model Research Tool** section above.
- **🧪 Model Innovation (tab_innovation)**:
  - See **Model Research Tool** section above.
- **📊 Benchmark Models (tab_benchmark)**:
  - See **Model Research Tool** section above.
- **Per-page assistant**:
  - At bottom, tries `ui.page_assistant.render_page_assistant("Model Lab")` with a try/except and ignores failures.

## Page: 11_Admin.py — Complete Tab & Feature Inventory

_Not yet fully audited in this pass, but grep confirms a **“⚙️ Configuration”** tab that explicitly mentions “feature flags”. A future audit should enumerate:_
- _All admin tabs (system health, agent monitoring, logs, configuration, feature flags, developer tools)._  
- _Every Streamlit section that toggles feature flags via `utils.feature_flags.FeatureFlags` (`get_feature_flags()`) and any gated/hidden features exposed only here._  

## Agent Inventory

_Pending: a full read of all files under `trading/agents/` and `agents/implementations/` to list each agent, its role, call graph, and whether it is invoked by any page. Initial inspection shows active implementation helpers:_
- `agents/implementations/research_fetcher.py` — **ArxivResearchFetcher**, described above.
- `agents/implementations/implementation_generator.py` — consumes `ResearchPaper` objects to propose concrete model implementations from literature.
- `agents/implementations/model_benchmarker.py` — used by Model Lab’s Benchmark tab to benchmark model candidates.

## Component Deep Read

_Pending: full read of `components/watchlist_widget.py`, `components/multi_timeframe_chart.py`, `components/news_candle_chart.py`, `components/onboarding.py`, and `components/__init__.py` to describe their render flows and persistence behaviors in detail._

## Hidden & Undocumented Features

- **Feature flags**:
  - `utils/feature_flags.py` defines a global `FeatureFlags` singleton with:
    - Env-driven default flags (`load_from_env`).
    - Runtime overrides via `set_flag(feature, enabled)` with logging.
    - `get_all_flags()` and `reload_flags()`.
  - `pages/11_Admin.py` (Config tab) refers to “feature flags”; full wiring to `FeatureFlags` needs to be enumerated in a follow-up pass.
- **AI Model Discovery / Innovation / Benchmarking**:
  - These advanced capabilities are exposed as separate tabs inside Model Lab and depend on background agents (`ModelDiscoveryAgent`, `ModelInnovationAgent`, `ModelBenchmarker`) that may or may not be present in a given deployment.
- **Background / monitoring tasks**:
  - Home page (`0_Home.py`) starts a one-per-session background thread to run `trading.services.monitoring_tools.check_model_degradation` and `check_strategy_degradation`, effectively a hidden health check.
  - Model Monitoring tab expects `model_monitor` and `model_log` objects in session state, which are likely created by non-UI background services.

## Stubs & Incomplete Features

- **Custom code execution in Strategy Testing**:
  - `pages/3_Strategy_Testing.py` intentionally raises a `NotImplementedError` for user-supplied arbitrary Python strategy code to prevent unsafe execution. Predefined strategies are implemented; user-uploaded code path is blocked by design.
- **Model Discovery Agent**:
  - `tab_discovery` degrades gracefully when `ModelDiscoveryAgent` is unavailable, showing a message and not exposing any controls.
- **Model Innovation Agent**:
  - `tab_innovation` falls back to synthetic results and plain text when the agent is missing or lacks expected methods.
- **Model Benchmarking**:
  - `tab_benchmark` relies on `ModelBenchmarker` and `ModelCandidate`; when imports fail, displays “Model Benchmarker not available” and does not attempt benchmarks.
- **Re-evaluate Model** (Performance tab):
  - “🔄 Re-evaluate Model” button is a **UI stub**; it only emits an info message and does not re-run the model.
- **Export formats beyond JSON** (Registry → Export/Import, Benchmarking → ONNX/Pickle):
  - Several UI paths mention Pickle/ONNX exports or non-JSON imports but explicitly state that **additional implementation is required**, acting as design placeholders.

