## PAGE: 2_Forecasting.py

### Tabs
- **Main Tabs**:
  - `🚀 Quick Forecast`
  - `⚙️ Advanced Forecasting`
  - `🤖 AI Model Selection`
  - `📊 Model Comparison`
  - `📈 Market Analysis`
  - `🔗 Multi-Asset (GNN)`
  - `🎲 Monte Carlo`
  - `🕵️ Insider Flow`
  - `📅 Earnings`

### Expanders
- **Global error boundary**:
  - `Developer details` (shown when page-level exception occurs).
- **Quick Forecast tab**:
  - `📊 Data Quality Metrics` (after successful data load; may appear twice depending on code path).
  - `📈 Multi-Timeframe Chart`
  - `📋 View Full Data`
  - `📊 Signal Breakdown` (inside AI Score section).
  - `⚙️ Forecast Postprocessing`
  - `📊 Detailed Analysis` (inside Natural Language Insights).
  - `📊 Detailed Analysis` (inside AI Commentary Service).
  - `📐 Accuracy & continuity`
  - `View Feature Importance & Explanations`
- **Advanced Forecasting tab**:
  - `📊 Model validation scores (walk-forward)`
  - `📊 View Feature Importance & Explanations`
- **AI Model Selection tab**:
  - `🧠 Why this model?`
  - `🔄 Alternative Models`
- **Model Comparison tab**:
  - No explicit expanders; uses charts and captions.
- **Market Analysis tab**:
  - No explicit expanders; uses error tracebacks if needed.
- **Multi-Asset (GNN) tab**:
  - `📋 View Forecast Data`
- **Earnings tab**:
  - `Historical earnings detail`

### Sidebar
- This page does **not** define any `st.sidebar` elements. All UI is rendered in the main page body.

### Major Features

#### Quick Forecast (Tab `🚀 Quick Forecast`)
- **Data loading workflow**:
  - **What user does**: Fills a form with `Ticker Symbol`, `Start Date`, `End Date` and clicks `📊 Load Data`.
  - **Backend calls**:
    - `DataLoader` and `DataLoadRequest` from `trading.data.data_loader` to fetch market data via `load_market_data`.
    - Optional data quality check using `src.utils.data_validation.DataValidator` (if import succeeds).
  - **Outputs**:
    - Stores cleaned data in `st.session_state["forecast_data"]` and symbol in `st.session_state["symbol"]`.
    - Displays success/error messages and data quality metrics.
    - Shows summary metrics (data points, current price, period return, annualized volatility).
    - Renders:
      - AI Score panel using `trading.analysis.ai_score.compute_ai_score`.
      - Multi-timeframe chart using `components.multi_timeframe_chart.render_multi_timeframe_chart`.
      - Candlestick or line chart using `utils.plotting_helper.create_candlestick_chart` (fallback to simple `go.Scatter` if import fails).
      - Data preview and full data table.
    - Shows earnings proximity warning using `trading.data.earnings_calendar.get_upcoming_earnings`.

- **Single-model quick forecast generation**:
  - **What user does**: Chooses a model (`ARIMA`, `XGBoost`, `Ridge`) and clicks `🚀 Generate Forecast`.
  - **Backend calls**:
    - `trading.models.forecast_router.ForecastRouter.get_forecast` for primary forecast (without walk-forward).
    - Fallback direct model training if router fails:
      - `trading.models.forecast_features.prepare_forecast_data`.
      - `trading.models.model_registry.get_registry` to obtain `ModelClass` for specific model types.
      - Direct model classes if registry fails: `LSTMForecaster`, `XGBoostModel`, `ProphetModel`, `ARIMAModel`.
      - `trading.forecasting.forecast_postprocessor.ForecastPostprocessor` for smoothing / bounds / outlier removal.
      - Optional logging via `st.session_state.model_log` and `st.session_state.perf_logger`.
      - `trading.memory.get_memory_store` / `trading.memory.memory_store.MemoryType` to log forecast into long-term memory.
    - Optional natural language explanation:
      - `nlp.natural_language_insights.NaturalLanguageInsights.generate_forecast_insights`.
    - Optional AI commentary service (if `st.session_state.commentary_service` exists) via its `generate_forecast_commentary` method.
    - Optional separate AI commentary via `agents.llm.agent.get_prompt_agent().process_prompt`.
    - Forecast display:
      - Preferred: `trading.ui.forecast_components.render_forecast_results` and `render_confidence_metrics`.
      - Fallback: `utils.plotting_helper.create_forecast_chart`.
    - Explainability:
      - `trading.models.forecast_explainability.ForecastExplainability.explain_forecast` to get feature importance and textual explanation.
  - **Outputs**:
    - Populates:
      - `st.session_state.current_forecast` (forecast DataFrame).
      - `st.session_state.current_model` (model name).
      - `st.session_state.current_model_instance` (trained model).
      - `st.session_state.current_forecast_result` (raw/processed forecast dict).
      - `st.session_state.forecast_explanation` (explanation object/dict).
    - Renders forecast charts, tables, accuracy/continuity metrics, postprocessing notes, feature importance plots, and narrative explanations.

- **Consensus forecast view**:
  - **What user does**: After a forecast exists, consensus is automatically computed (no extra button).
  - **Backend calls**:
    - `trading.models.forecast_router.ForecastRouter.get_consensus_forecast`.
  - **Outputs**:
    - `🎯 Model Consensus` expander with:
      - Consensus direction, conviction, consensus price delta vs current price.
      - Per-model price targets table.
      - List of models that failed.

#### Advanced Forecasting (Tab `⚙️ Advanced Forecasting`)
- **Advanced model configuration & training**:
  - **What user does**:
    - Selects model type from the registry (LSTM, XGBoost, Prophet, ARIMA, Ensemble, TCN, GARCH, Autoformer, CatBoost, Ridge).
    - Optionally runs `🔬 Auto-select best model`.
    - Configures model-specific hyperparameters via sliders/inputs.
    - Enables optional feature engineering: technical indicators, lag features, macro features, normalization.
    - Clicks `🚀 Train Model`.
  - **Backend calls**:
    - `trading.models.model_registry.get_registry` to fetch models, descriptions, and validation metrics.
    - `trading.models.forecast_router.ForecastRouter.select_best_model` for auto-selection.
    - Feature engineering via `FeatureEngineering` from `trading.data.preprocessing`:
      - `calculate_moving_averages`, `calculate_rsi`, `calculate_macd`, `calculate_bollinger_bands`.
    - Macro features via `trading.feature_engineering.macro_feature_engineering.MacroFeatureEngineer.enrich_trading_data`.
    - Normalization using `DataPreprocessor` from `trading.data.preprocessing`.
    - Model instances either from registry or direct classes (`LSTMForecaster`, `XGBoostModel`, `ProphetModel`, `ARIMAModel`).
    - Forecast generation via model’s `forecast_with_uncertainty`, `forecast`, or `predict`.
    - Postprocessing via `trading.forecasting.forecast_postprocessor.ForecastPostprocessor`.
    - Explainability via `trading.models.forecast_explainability.ForecastExplainability.explain_forecast`.
  - **Outputs**:
    - Populates:
      - `st.session_state.advanced_forecast` and `st.session_state.advanced_model`.
      - `st.session_state.current_forecast_result` and `st.session_state.current_model_instance`.
      - `st.session_state.forecast_explanation_tab2`.
      - `st.session_state.advanced_forecast_best_model` and `st.session_state.advanced_best_model_scores`.
    - Displays:
      - Validation / walk-forward scores table.
      - Advanced forecast chart and table.
      - Feature importance and explanation text.
      - Previous forecast preview.

#### AI Model Selection (Tab `🤖 AI Model Selection`)
- **Model recommendation using AI**:
  - **What user does**:
    - Clicks `🔍 Analyze Data & Recommend Model`.
  - **Backend calls**:
    - `trading.agents.model_selector_agent.ModelSelectorAgent` (if import succeeds).
      - `select_model` and `get_model_recommendations`.
    - Uses `trading.agents.model_selector_agent.ForecastingHorizon` and `MarketRegime` enums.
    - Fallback internal heuristic if agent unavailable.
  - **Outputs**:
    - `st.session_state.ai_recommendation` dict with:
      - Recommended model, confidence, reasoning, data characteristics, and alternative models.
    - UI shows:
      - Confidence metric and progress bar.
      - Reasoning and data characteristic bullets.
      - Alternative models with confidence scores.
    - Actions:
      - `✅ Use Recommended Model` sets `st.session_state.selected_model`.
      - `🔄 Choose Different Model` and override selectbox also write `st.session_state.selected_model` and toggle `st.session_state.show_override`.

- **Hybrid Model Selection helper**:
  - **What user does**: Clicks `Use Hybrid Selector`.
  - **Backend calls**:
    - `trading.forecasting.hybrid_model_selector.HybridModelSelector` (used only as context; selection done manually).
    - Optional regime detection via `trading.market.market_analyzer.MarketAnalyzer.detect_market_regime`.
  - **Outputs**:
    - Writes `st.session_state.hybrid_selected_models` and `st.session_state.hybrid_regime`.
    - Displays recommended configurations and guidance to use Model Comparison tab for ensemble training.

- **Model comparison table via router**:
  - **What user does**: No explicit button; when data exists it runs immediately in this section.
  - **Backend calls**:
    - `trading.models.model_registry.get_registry` to list models.
    - `trading.models.forecast_router.ForecastRouter.get_forecast` for each model.
  - **Outputs**:
    - DataFrame summarizing each model’s status, MAPE-like metric, 7d forecast, and notes.

#### Model Comparison (Tab `📊 Model Comparison`)
- **Visual comparison of multiple models**:
  - **What user does**:
    - Picks models from multiselect (arima, xgboost, lstm, prophet, catboost, ridge, tcn, ensemble).
    - Adjusts `Forecast horizon (days)` slider.
    - Clicks `Compare Models`.
  - **Backend calls**:
    - `trading.models.forecast_router.ForecastRouter.get_forecast` for each selected model.
  - **Outputs**:
    - Plotly chart with last ~20 days of historical prices plus forward forecasts for each model.
    - Skips models whose forecast is empty or outside 0.5x–2x of last price, adding caption notes.

#### Market Analysis (Tab `📈 Market Analysis`)
- **Correlation vs SPY and volatility regime**:
  - **What user does**: No additional inputs; uses existing `forecast_data`.
  - **Backend calls**:
    - `yfinance` to fetch SPY history.
    - Uses `numpy` and `plotly.graph_objects` for metrics and plots.
  - **Outputs**:
    - Rolling 60-day correlation chart and current value.
    - Volatility metrics (20d, 60d, 1y) and volatility regime classification.
    - Commentary about risk and mean reversion.

#### Multi-Asset Forecasting with GNN (Tab `🔗 Multi-Asset (GNN)`)
- **Multi-asset data loading**:
  - **What user does**:
    - Enters multiple tickers in text area.
    - Adjusts correlation threshold, forecast days, and training epochs.
    - Clicks `📥 Load Multi-Asset Data`.
  - **Backend calls**:
    - `trading.data.data_loader.DataLoader` and `DataLoadRequest` per ticker.
  - **Outputs**:
    - `st.session_state.gnn_data` and `st.session_state.gnn_tickers`.
    - Success/errors per ticker and combined overlapping dataset.

- **GNN-based multi-asset forecast**:
  - **What user does**:
    - After data load, selects primary asset.
    - Clicks `🔮 Train GNN & Generate Forecast`.
  - **Backend calls**:
    - `trading.models.advanced.gnn.gnn_model.GNNForecaster`:
      - `fit` on multi-asset price DataFrame.
      - `forecast` with horizon and target asset.
      - `get_relationship_matrix`.
  - **Outputs**:
    - Correlation heatmap of assets.
    - Relationship matrix heatmap.
    - GNN forecast chart for selected asset.
    - Metrics for horizon, assets used, average confidence, forecast return.
    - Forecast table with date, forecast, confidence.

#### Monte Carlo Simulation (Tab `🎲 Monte Carlo`)
- **Price path simulation**:
  - **What user does**:
    - Adjusts `Simulations` and `Horizon (days)` sliders.
    - Clicks `Run Monte Carlo`.
  - **Backend calls**:
    - Uses `numpy` and `plotly.graph_objects` only; no external trading modules.
  - **Outputs**:
    - Monte Carlo fan chart (5/25/50/75/95th percentile paths).
    - Metrics for each terminal percentile and probability that price ends above today’s price.

#### Insider Flow (Tab `🕵️ Insider Flow`)
- **Insider transactions summary**:
  - **What user does**: None beyond loading data and symbol in Quick Forecast.
  - **Backend calls**:
    - `trading.data.insider_flow.get_insider_flow`.
  - **Outputs**:
    - Metrics for insider buys/sells and a colored signal.
    - Table of recent insider transactions if available.

#### Earnings (Tab `📅 Earnings`)
- **Earnings reaction analysis**:
  - **What user does**: None beyond having a symbol in session state.
  - **Backend calls**:
    - `trading.data.earnings_reaction.get_earnings_reactions`.
  - **Outputs**:
    - Metrics for average move, EPS beat rate, positive reaction rate, typical move range.
    - Next earnings date info if available.
    - Bar chart of historical earnings day moves.
    - Detailed historical earnings DataFrame.

### Buttons with Actions
- **Quick Forecast tab**:
  - `📊 Load Data` (form submit):
    - Validates inputs and triggers data load via `DataLoader.load_market_data`.
    - Writes `st.session_state["forecast_data"]`, `st.session_state["symbol"]`.
  - `🚀 Generate Forecast`:
    - Triggers quick forecast pipeline (router + fallback models).
    - Writes `st.session_state.current_forecast`, `current_model`, `current_model_instance`, `current_forecast_result`.
    - May write to `trading.memory` store (no direct session_state keys there).
  - `Generate Plain English Explanation`:
    - Calls `NaturalLanguageInsights.generate_forecast_insights`.
  - `Generate Commentary` (AI Commentary Service section):
    - Calls `commentary_service.generate_forecast_commentary`.
  - `Generate AI Commentary`:
    - Calls `agents.llm.agent.get_prompt_agent().process_prompt`.
  - `Clear explanation` (Quick tab explainability):
    - Clears `st.session_state.forecast_explanation` and triggers `st.rerun()`.
  - `Generate Explanation` (Quick tab explainability):
    - Calls `ForecastExplainability.explain_forecast` and stores result in `st.session_state.forecast_explanation`.

- **Advanced Forecasting tab**:
  - `🔬 Auto-select best model`:
    - Calls `ForecastRouter.select_best_model` and writes:
      - `st.session_state["advanced_forecast_best_model"]`
      - `st.session_state["advanced_best_model_scores"]`
  - `🚀 Train Model`:
    - Executes advanced feature engineering, model training, forecasting, and explainability pipeline.
    - Writes:
      - `st.session_state.advanced_forecast`
      - `st.session_state.advanced_model`
      - `st.session_state.current_forecast_result`
      - `st.session_state.current_model_instance`
  - `Clear explanation` (Advanced tab):
    - Clears `st.session_state.forecast_explanation_tab2`.
  - `Generate Explanation` (Advanced tab):
    - Calls `ForecastExplainability.explain_forecast` and sets `st.session_state.forecast_explanation_tab2`.

- **AI Model Selection tab**:
  - `🔍 Analyze Data & Recommend Model`:
    - Computes horizon/regime, uses `ModelSelectorAgent` or fallback logic.
    - Writes `st.session_state.ai_recommendation`.
  - `✅ Use Recommended Model`:
    - Writes `st.session_state.selected_model`.
  - `🔄 Choose Different Model`:
    - Sets `st.session_state.show_override = True`.
  - `Confirm Override`:
    - Writes `st.session_state.selected_model` and resets `st.session_state.show_override`.
  - `Use Hybrid Selector`:
    - Optionally uses `MarketAnalyzer.detect_market_regime`.
    - Writes `st.session_state.hybrid_selected_models` and `st.session_state.hybrid_regime`.

- **Model Comparison tab**:
  - `Compare Models`:
    - For each selected model, calls `ForecastRouter.get_forecast` and renders comparison chart.

- **Multi-Asset (GNN) tab**:
  - `📥 Load Multi-Asset Data`:
    - Loads data per ticker, builds aligned DataFrame.
    - Writes `st.session_state.gnn_data` and `st.session_state.gnn_tickers`.
  - `🔮 Train GNN & Generate Forecast`:
    - Trains `GNNForecaster` and computes forecast.

- **Monte Carlo tab**:
  - `Run Monte Carlo`:
    - Runs simulation and renders chart and metrics.

### Session State Keys
- **Read and/or written**:
  - `forecasting_backend` (lazy-loaded backend dict).
  - `forecast_data` (loaded historical price data).
  - `selected_models` (list; not heavily used in this file).
  - `ai_recommendation` (AI model selection result).
  - `comparison_results` (not directly written in this file; placeholder for comparison).
  - `market_regime` (not updated here; used conceptually).
  - `symbol` (current ticker symbol).
  - `forecast_horizon` (global horizon setting for forecasts).
  - `forecast_horizon_slider` (widget key for slider).
  - `current_forecast` (quick forecast DataFrame).
  - `current_model` (quick forecast model name).
  - `current_model_instance` (quick forecast model object).
  - `current_forecast_result` (raw/processed forecast result dict or array).
  - `model_log` and `perf_logger` (checked before logging, not set here).
  - `evolve_session_id`, `evolve_force_onboarding`, `evolve_onboarding_done`, `evolve_show_form` are **not** used in this file (onboarding is in components).
  - `advanced_forecast` (advanced forecast DataFrame).
  - `advanced_model` (model type for advanced forecast).
  - `advanced_forecast_best_model` and `advanced_best_model_scores`.
  - `forecast_explanation` (quick tab explainability).
  - `forecast_explanation_tab2` (advanced tab explainability).
  - `ai_recommendation`, `selected_model`, `show_override`.
  - `hybrid_selected_models`, `hybrid_regime`.
  - `gnn_data`, `gnn_tickers`.
  - `mc_sims`, `mc_horizon` (widget keys for Monte Carlo).

### External Integrations (Imports under `trading/`, `agents/`, `components/`, `system/`)
- **Top-level or lazy-loaded imports**:
  - `from trading.data.data_loader import DataLoader, DataLoadRequest` (lazy via `_get_forecasting_backend` and again in GNN tab).
  - `from trading.data.providers.yfinance_provider import YFinanceProvider`.
  - `from trading.models.lstm_model import LSTMForecaster`.
  - `from trading.models.xgboost_model import XGBoostModel`.
  - `from trading.models.prophet_model import ProphetModel`.
  - `from trading.models.arima_model import ARIMAModel`.
  - `from trading.data.preprocessing import FeatureEngineering, DataPreprocessor`.
  - `from trading.agents.model_selector_agent import ModelSelectorAgent` (lazy backend and in AI Model Selection tab).
  - `from trading.market.market_analyzer import MarketAnalyzer`.
  - `from trading.analysis.ai_score import compute_ai_score`.
  - `from trading.models.model_registry import get_registry`.
  - `from trading.models.forecast_router import ForecastRouter`.
  - `from trading.models.forecast_features import prepare_forecast_data`.
  - `from trading.forecasting.forecast_postprocessor import ForecastPostprocessor`.
  - `from trading.memory import get_memory_store`.
  - `from trading.memory.memory_store import MemoryType`.
  - `from trading.models.forecast_explainability import ForecastExplainability`.
  - `from trading.feature_engineering.macro_feature_engineering import MacroFeatureEngineer`.
  - `from trading.forecasting.hybrid_model_selector import HybridModelSelector`.
  - `from trading.models.advanced.gnn.gnn_model import GNNForecaster`.
  - `from trading.data.earnings_calendar import get_upcoming_earnings`.
  - `from trading.data.insider_flow import get_insider_flow`.
  - `from trading.data.earnings_reaction import get_earnings_reactions`.
  - `from trading.ui.forecast_components import render_forecast_results, render_confidence_metrics`.
- **Agents / LLM**:
  - `from agents.llm.agent import get_prompt_agent`.
- **Components**:
  - `from components.multi_timeframe_chart import render_multi_timeframe_chart`.
- **Utilities / plotting**:
  - `from utils.plotting_helper import create_candlestick_chart`, `create_forecast_chart`.
  - `from src.utils.data_validation import DataValidator` (guarded by `try/except ImportError`).
- **Other external libraries**:
  - `yfinance` (within Market Analysis tab).

### Stubs, TODOs, and "Coming Soon"
- No functions in this file are pure stubs (`pass`) or marked `NotImplemented`.
- There are no explicit `TODO` or `FIXME` comments in the visible code.
- The GNN section includes explanatory text but does instantiate and use `GNNForecaster` when the button is pressed, so it is not a stub.
- The Hybrid Model Selector section notes that certain advanced behaviors are not available on the `HybridModelSelector` class and uses a simplified recommendation, but it is still functional rather than a stub.

