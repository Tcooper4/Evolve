## PAGE: 3_Strategy_Testing.py

### Tabs
- **Main Tabs**:
  - `🚀 Quick Backtest`
  - `🔧 Strategy Builder`
  - `💻 Advanced Editor`
  - `🎯 Strategy Combos`
  - `📊 Strategy Comparison`
  - `🔬 Advanced Analysis`
  - `🤖 RL Training`
  - `🤖 AI Strategy Research`

### Expanders
- **Global error boundary**:
  - `Developer details` (shown when page-level exception occurs).
- **Quick Backtest tab**:
  - No named expanders in the provided section (results are handled via `trading.ui.strategy_components` or direct layout).
- **Strategy Builder tab**:
  - `📝 Strategy Details`
  - `📈 Entry Conditions`
  - `📉 Exit Conditions`
  - `💰 Position Sizing`
  - `⚠️ Risk Management`
- **Advanced Editor tab**:
  - `📋 View Signals` (inside test results section; only shown if code execution path were enabled, but currently code raises `NotImplementedError`).
- **Strategy Combos tab**:
  - `⚙️ Advanced Settings`
- **Strategy Comparison tab**:
  - No explicit expanders; results shown via charts and tables.
- **Advanced Analysis tab**:
  - `📋 Detailed Results` (walk-forward analysis in Tab 1 Quick Backtest section).
- **AI Strategy Research tab**:
  - Individual strategy sections are each wrapped in an expander:
    - `Strategy {i}: {strategy_name}` (created in a loop for each discovered strategy).
- **RL Training tab**:
  - No explicit expanders; uses columns and charts.

### Sidebar
- This page does **not** define any `st.sidebar` elements. All UI components are in the main layout with `st.columns`, `st.expander`, and `st.tabs`.

### Major Features

#### Quick Backtest (Tab `🚀 Quick Backtest`)
- **Data loading for backtests**:
  - **What user does**:
    - Enters `Symbol`, `Start Date`, and `End Date` in a form and clicks `📊 Load Data`.
  - **Backend calls**:
    - `DataLoader` and `DataLoadRequest` from `trading.data.data_loader`; calls `loader.load_market_data(request)`.
  - **Outputs**:
    - Normalizes column names to lowercase and ensures `close`, `volume`, `open`, `high`, `low` exist (imputing defaults if missing).
    - Writes `st.session_state.loaded_data` (price data) and `st.session_state.backtest_symbol`.

- **Pre-built strategy selection and parametrization**:
  - **What user does**:
    - After data is loaded, chooses a strategy via `trading.ui.strategy_components.render_strategy_selector` (or fallback `st.selectbox`).
    - Adjusts per-strategy parameters through dynamically generated sliders based on `STRATEGY_REGISTRY`.
  - **Backend calls**:
    - `trading.ui.strategy_components.render_strategy_selector` (if available).
  - **Outputs**:
    - Determines `strategy_name` and `strategy_info` (class, config, params, description) used in backtest.

- **Sentiment-Based Strategy augmentation**:
  - **What user does**:
    - Toggles `Include sentiment signals`, configures `Sentiment Threshold`, and clicks `Generate Sentiment Signals`.
  - **Backend calls**:
    - `trading.signals.sentiment_signals.SentimentSignals`, using whichever method exists among:
      - `generate_signals`, `generate_sentiment_signals`, `get_signals`, `compute_signals`.
  - **Outputs**:
    - Computes summary metrics (`buy_count`, `sell_count`) and shows recent sentiment signals in a table.
    - Writes `st.session_state.sentiment_signals`.
    - These signals are prepared for potential use in strategy logic but direct integration into the pre-built strategies is not shown in this file.

- **Backtest execution and metrics**:
  - **What user does**:
    - Configures `Initial Capital` and `Commission` and clicks `🚀 Run Backtest`.
  - **Backend calls**:
    - Strategy class (from `STRATEGY_REGISTRY`):
      - `BollingerStrategy`, `MACDStrategy`, `RSIStrategy`, `SMAStrategy` (with or without config dataclasses).
    - `strategy.generate_signals(data)` to produce a signals DataFrame.
    - `trading.backtesting.backtester.Backtester.normalize_results` to standardize results format (if import succeeds).
    - Memory integration:
      - `_get_memory_store()` (wrapper around `trading.memory.get_memory_store`).
      - `trading.memory.memory_store.MemoryType` for namespaced writes.
  - **Outputs**:
    - Computes:
      - `returns`, `strategy_returns`, `cumulative_returns`, `strategy_cumulative_returns`.
      - Equity curve from initial capital.
      - Metrics: total return, Sharpe ratio, max drawdown, win rate.
    - Builds a `results` dict with:
      - `total_return`, `sharpe_ratio`, `max_drawdown`, `win_rate`.
      - `equity_curve` DataFrame.
      - `trades` list of normalized trade dicts (via `_normalize_trades` and optional `Backtester.normalize_results`).
    - Writes:
      - `st.session_state.backtest_results`.
      - `st.session_state.backtest_strategy`.
    - Persists summary metrics to long-term memory in two namespaces:
      - `StreamlitStrategyTesting` (backtest-level).
      - `backtests` (for Chat/agent context).

- **Backtest results visualization**:
  - **What user sees**:
    - If `trading.ui.strategy_components` is available:
      - `render_backtest_results` (charts).
      - `render_strategy_metrics` (key metrics).
      - `render_trade_list` (trade table + CSV download).
    - Fallback:
      - Metrics via `st.metric` (total return, Sharpe, max drawdown, win rate).
      - Equity curve chart via `plotly.graph_objects`.
      - Trade history table and CSV download.

- **Walk-Forward Analysis (within Quick Backtest tab)**:
  - **What user does**:
    - Checks `Enable walk-forward validation`.
    - Configures `Training window (days)`, `Test window (days)`, `Step size (days)`, `Number of iterations`.
    - Clicks `Run Walk-Forward Analysis`.
  - **Backend calls**:
    - `WalkForwardValidator` from `trading.models.walk_forward_validator`.
    - `_resolve_walk_forward_strategy` to map display name to an instantiated strategy using pre-built strategy classes and dataclass configs.
  - **Outputs**:
    - Runs `validator.walk_forward_test` with a progress callback.
    - Shows metrics: average return, consistency score, win rate, iterations.
    - Displays per-iteration returns in a bar chart and a detailed results DataFrame.

#### Strategy Builder (Tab `🔧 Strategy Builder`)
- **Visual, no-code strategy design**:
  - **What user does**:
    - Fills out:
      - Strategy metadata: name, description.
      - Entry and exit conditions with indicators, operators, and values (dynamic UI with user-specified counts).
      - Position sizing (fixed dollar, % of portfolio, ATR-based, Kelly).
      - Risk management (stop loss, take profit, max positions, max daily loss).
  - **Backend calls**:
    - Uses `CustomStrategyHandler` from `trading.strategies.custom_strategy_handler` (through `_get_strategy_backend`).
    - Generates Python code via helper function `generate_strategy_code`, which itself uses:
      - `generate_entry_conditions_code` and `generate_exit_conditions_code` to synthesize indicator logic.
  - **Outputs**:
    - Renders a human-readable preview of entry/exit rules and risk settings.
    - Validates that name and at least one entry/exit condition exist.
    - On `💾 Save Strategy`:
      - Builds a structured `strategy_config` dict.
      - Produces `strategy_code` string.
      - Calls `handler.create_strategy_from_code(name, code, parameters)`.
      - Stores `strategy_config` under `st.session_state.custom_strategies[strategy_name]`.
      - Calls `handler.load_strategies()` and reloads the page.
    - On `🧪 Test Strategy`:
      - Displays an informational message indicating that testing is available after saving and is performed via Quick Backtest tab.

- **Custom strategy management**:
  - **What user does**:
    - Uses the `Saved Custom Strategies` section to:
      - Select an existing strategy from `handler.strategies`.
      - Optionally `📂 Load Strategy` (currently only logs an info message; form population is not implemented).
      - `🗑️ Delete Strategy`, which calls `handler.delete_strategy` and reloads on success.

#### Advanced Editor (Tab `💻 Advanced Editor`)
- **Raw code editor for custom strategies**:
  - **What user does**:
    - Selects a base template (Empty, RSI, MA Crossover, Bollinger Bands, MACD) and optionally `📋 Load Template`.
    - Edits strategy code in a large text area.
    - Provides a strategy name.
    - Uses buttons:
      - `💾 Save Strategy`
      - `✓ Validate`
      - `🧪 Test`
  - **Backend calls**:
    - `CustomStrategyHandler` for saving and listing strategies.
    - Local helper `validate_strategy_code` for syntax and structure checks.
  - **Outputs**:
    - Validation:
      - Compiles code and checks for presence and signature of `generate_signals`.
      - Returns `valid`, `error`, and `warnings` (e.g., missing pandas import).
      - Displays success/warnings or error with code snippet.
    - Saving:
      - Calls `handler.create_strategy_from_code(name, code, parameters={})`.
      - On success, reloads `handler.strategies`.
    - Testing:
      - Attempts to run custom code but immediately raises `NotImplementedError("Custom code execution disabled for security...")`, so no actual execution occurs.
      - All subsequent test logic is unreachable due to this explicit guard.
    - Import/export:
      - `📥 Export Code` uses `st.download_button` to export the current code.
      - Import uses `st.file_uploader`, loads file contents to `st.session_state.editor_code`, and reloads.
    - Loading saved strategies:
      - Uses `handler.strategies` to populate a selectbox; `📂 Load` sets `st.session_state.editor_code` and `editor_strategy_name` and reloads.

#### Strategy Combos / Ensembles (Tab `🎯 Strategy Combos`)
- **Ensemble configuration and backtest**:
  - **What user does**:
    - Requires data loaded in Quick Backtest.
    - Selects 2–5 strategies (prebuilt and/or custom).
    - Chooses combination method: `weighted_average` or `voting`.
    - Sets per-strategy weights (normalized and visualized with progress bars).
    - Configures advanced ensemble settings (confidence and consensus thresholds).
    - Sets initial capital and clicks `🚀 Run Ensemble Backtest`.
  - **Backend calls**:
    - Uses pre-built strategies (`BollingerStrategy`, `MACDStrategy`, `RSIStrategy`, `SMAStrategy`) and custom strategies via `handler.execute_strategy`.
    - Ensemble implementation via:
      - `WeightedEnsembleStrategy` and `EnsembleConfig` from `trading.strategies.ensemble`.
  - **Outputs**:
    - Generates and stores per-strategy signal DataFrames.
    - Combines them with `WeightedEnsembleStrategy.combine_signals`.
    - Computes ensemble returns, equity curve, and metrics: total return, Sharpe, max drawdown, win rate.
    - Shows:
      - Metric summary for ensemble.
      - Equity curve comparison chart (ensemble vs individual strategies).
      - Performance comparison table with returns, Sharpe, drawdown, win rate, and weights.
    - Writes `st.session_state.ensemble_results` with signals, metrics, and equity curve.
    - Shows a brief summary of previous ensemble results if present.

#### Strategy Comparison (Tab `📊 Strategy Comparison`)
- **Parallel multi-strategy backtesting**:
  - **What user does**:
    - Requires data loaded in Quick Backtest.
    - Selects 2 or more strategies to compare (prebuilt + custom).
    - Configures initial capital and commission.
    - Clicks `🚀 Run Comparison`.
  - **Backend calls**:
    - For each strategy:
      - Prebuilt: instantiate class + config and call `generate_signals`.
      - Custom: run `handler.execute_strategy`.
  - **Outputs**:
    - For each strategy:
      - Equity curve and returns series.
      - Metrics: total return, Sharpe, max drawdown, win rate, total trades, profit factor.
    - Determines best performers by return, Sharpe, and drawdown.
    - Displays:
      - Best performer metrics.
      - Performance metrics table.
      - Equity curve comparison chart (best highlighted).
      - Drawdown comparison chart.
      - Returns distribution histogram overlay.
      - Statistical summary (mean, std, skewness, kurtosis, min, max daily returns).
    - Persists `st.session_state.comparison_results`.

#### Advanced Analysis (Tab `🔬 Advanced Analysis`)
- **Analysis type selection**:
  - **Options**:
    - Walk-Forward Analysis
    - Monte Carlo Simulation
    - Sensitivity Analysis
    - Parameter Optimization

- **Walk-Forward Analysis (Tab 6 version)**:
  - **What user does**:
    - Configures window sizes, step, and initial capital.
    - Selects a strategy from the prebuilt registry.
    - Clicks `🚀 Run Walk-Forward Analysis`.
  - **Backend calls**:
    - Uses prebuilt strategy classes.
    - `BacktestEvaluator` from `trading.backtesting.evaluator`.
    - Custom walk-forward loop implemented directly in this file.
    - Memory store via `_get_memory_store()` and `MemoryType.LONG_TERM` (namespace `StreamlitStrategyTesting`, category `walk_forward_analysis`).
  - **Outputs**:
    - Creates per-window metrics: return, Sharpe, max drawdown.
    - Summarizes and visualizes results and writes them to long-term memory.

- **Monte Carlo Simulation (Advanced Analysis tab)**:
  - **What user does**:
    - Configures number of simulations, initial capital, bootstrap method, and strategy.
    - Clicks `🚀 Run Monte Carlo Simulation`.
  - **Backend calls**:
    - Prebuilt strategy classes and config, to derive `strategy_returns`.
    - `MonteCarloSimulator` and `MonteCarloConfig` from `trading.backtesting.monte_carlo`.
  - **Outputs**:
    - Simulated equity paths and percentile curves via simulator.
    - Metrics: mean, median, 5th/95th percentile final values.
    - Chart with sample paths and percentile curves.

- **Sensitivity Analysis**:
  - **What user does**:
    - Selects a strategy and parameter range via slider.
    - Clicks `🚀 Run Sensitivity Analysis`.
  - **Backend calls**:
    - None beyond configuration; implementation is explicitly incomplete.
  - **Outputs**:
    - Displays an informational message:
      - “Sensitivity analysis implementation in progress...”

- **Parameter Optimization**:
  - **What user does**:
    - Selects strategy, optimization method (Grid/Random/Genetic/Bayesian), and objective.
    - Clicks `🚀 Run Optimization`.
  - **Backend calls**:
    - Not implemented in this file; no optimization algorithm is invoked.
  - **Outputs**:
    - Displays an informational message:
      - “Parameter optimization implementation in progress...”

#### AI Strategy Research (Tab `🤖 AI Strategy Research`)
- **AI-driven strategy discovery**:
  - **What user does**:
    - Requires `st.session_state.forecast_data` to be present.
    - Chooses:
      - `Strategy Focus` (Momentum, Mean Reversion, Breakout, Volatility, Multi-Factor).
      - `Current Market Regime` (including `Auto-Detect`).
      - `Number of strategies` and `Strategy Complexity`.
    - Clicks `🔬 Research Strategies`.
  - **Backend calls**:
    - `trading.agents.strategy_research_agent.StrategyResearchAgent`:
      - Prefer `research_strategies` if available.
      - Fallbacks:
        - `run_research_scan` to collect “discoveries”.
        - `run` to obtain `tested_strategies`.
  - **Outputs**:
    - Builds a `research_result` dict with:
      - `strategies`: list of strategy dicts.
      - `insights` summary.
      - `market_analysis` text.
    - For each strategy:
      - Shows metrics (expected return, Sharpe, win rate).
      - Displays description, entry rules, exit rules, parameters, optional backtest results, and implementation code.
      - `Test This Strategy` button sets `st.session_state.selected_strategy` to the chosen strategy.
    - Displays overall market analysis text.

#### RL Training (Tab `🤖 RL Training`)
- **Reinforcement learning strategy training**:
  - **What user does**:
    - Configures:
      - `Training Episodes`.
      - `Learning Rate`.
      - `Reward Function`.
      - `Discount Factor (γ)`.
    - Clicks `🚀 Train RL Agent`.
  - **Backend calls**:
    - `RLTrader` from `rl.rl_trader`:
      - All training occurs via:
        - `agent.train_episode(data)` inside a loop.
        - `agent.backtest(data)` when testing the trained agent.
  - **Outputs**:
    - Tracks `rewards_history` per episode.
    - Shows:
      - Final reward, average reward over last 100 episodes, improvement.
      - Rewards curve and moving average.
    - Stores:
      - `st.session_state.rl_agent`.
      - `st.session_state.rl_rewards_history`.
    - Backtest of RL agent produces metrics (total return, Sharpe, max drawdown) and an equity curve chart if dates and curve are present.

### Buttons with Actions (Non-Cosmetic)
- **Quick Backtest tab**:
  - `📊 Load Data`:
    - Loads historical data via `DataLoader.load_market_data` and writes session state keys.
  - `Generate Sentiment Signals`:
    - Computes sentiment-based signals via `SentimentSignals` and writes `st.session_state.sentiment_signals`.
  - `🚀 Run Backtest`:
    - Runs full backtest pipeline (signals → returns → metrics → normalization → memory persistence).
  - `Run Walk-Forward Analysis` (Quick Backtest sub-section):
    - Creates strategy with `_resolve_walk_forward_strategy` and runs `WalkForwardValidator.walk_forward_test`.

- **Strategy Builder tab**:
  - `💾 Save Strategy`:
    - Generates code via `generate_strategy_code`, calls `handler.create_strategy_from_code`, updates `st.session_state.custom_strategies` and reloads.
  - `🧪 Test Strategy`:
    - Shows an informational note that testing should be done in Quick Backtest after saving.
  - `📂 Load Strategy`:
    - Logs an info message indicating strategy loading (form population not implemented).
  - `🗑️ Delete Strategy`:
    - Deletes selected strategy via `handler.delete_strategy` and reloads.

- **Advanced Editor tab**:
  - `📋 Load Template`:
    - Loads template code into `st.session_state.editor_code` and reloads.
  - `💾 Save Strategy`:
    - Validates and saves user code via `handler.create_strategy_from_code`.
  - `✓ Validate`:
    - Validates code using `validate_strategy_code`.
  - `🧪 Test`:
    - Triggers a `NotImplementedError` indicating custom code execution is disabled.
  - `📥 Export Code`:
    - Exposes code as a downloadable `.py` file.
  - `Import Code` (file uploader):
    - Imports code into `st.session_state.editor_code` and reloads.
  - `📂 Load` (saved strategy selector):
    - Loads strategy’s code/name into session state and reloads.

- **Strategy Combos tab**:
  - `🚀 Run Ensemble Backtest`:
    - Runs ensemble generation and backtest pipeline (multi-strategy signals → ensemble combination → metrics and charts).

- **Strategy Comparison tab**:
  - `🚀 Run Comparison`:
    - Parallel backtesting for selected strategies and generation of comparison metrics/tables/charts.

- **Advanced Analysis tab**:
  - `🚀 Run Walk-Forward Analysis` (Tab 6 version):
    - Executes local walk-forward logic using prebuilt strategies and `BacktestEvaluator`.
  - `🚀 Run Monte Carlo Simulation`:
    - Configures and executes `MonteCarloSimulator` for strategy returns.
  - `🚀 Run Sensitivity Analysis`:
    - Currently only shows an “implementation in progress” message.
  - `🚀 Run Optimization`:
    - Currently only shows an “implementation in progress” message.

- **AI Strategy Research tab**:
  - `🔬 Research Strategies`:
    - Invokes appropriate method on `StrategyResearchAgent` to gather proposed strategies and analysis.
  - `Show implementation code` (per-strategy checkbox):
    - Reveals generated/returned code snippet for that strategy.
  - `Test This Strategy`:
    - Stores the strategy dict in `st.session_state.selected_strategy`.

- **RL Training tab**:
  - `🚀 Train RL Agent`:
    - Runs RL training loop with `RLTrader` on current data and stores the trained agent.
  - `Run Backtest with RL Agent`:
    - Calls `agent.backtest` and renders backtest metrics and equity curve.

### Session State Keys
- **Read and/or written**:
  - `strategy_backend` (lazy-loaded backend module map).
  - `loaded_data` (price data for backtesting).
  - `backtest_results`, `backtest_strategy`, `backtest_symbol`.
  - `custom_strategies` (dict of saved builder strategies).
  - `selected_strategy` (strategy name or, in AI tab, the selected strategy dict).
  - `strategy_combos` (dict for combos; not heavily used in visible code).
  - `sentiment_signals`.
  - `ensemble_results`.
  - `comparison_results`.
  - `forecast_data` (shared with Forecasting page; used for AI Strategy Research and RL tabs).
  - `strategy_handler` (CustomStrategyHandler instance).
  - `editor_code`, `editor_strategy_name`, `editor_code_area` (widget/session code buffers).
  - UI keys such as:
    - `use_sentiment_signals`, `mc_sims`, `mc_horizon`, `wf_train_window`, `wf_test_window`, `wf_step_size`, `wf_num_iterations`, `combo_capital`, `compare_capital`, `compare_commission`, etc.
  - `strategy_research_agent`.
  - `rl_agent`, `rl_rewards_history`.

### External Integrations (Imports under `trading/`, `agents/`, `components/`, `system/`)
- **Core trading/backtesting**:
  - `trading.data.data_loader.DataLoader`, `DataLoadRequest`.
  - `trading.strategies.bollinger_strategy.BollingerStrategy`, `BollingerConfig`.
  - `trading.strategies.macd_strategy.MACDStrategy`, `MACDConfig`.
  - `trading.strategies.rsi_strategy.RSIStrategy`.
  - `trading.strategies.sma_strategy.SMAStrategy`, `SMAConfig`.
  - `trading.strategies.custom_strategy_handler.CustomStrategyHandler`.
  - `trading.strategies.ensemble.WeightedEnsembleStrategy`, `EnsembleConfig`, `create_ensemble_strategy`.
  - `trading.backtesting.monte_carlo.MonteCarloSimulator`, `MonteCarloConfig`.
  - `trading.backtesting.evaluator.BacktestEvaluator`.
  - `trading.backtesting.backtester.Backtester`.
  - `trading.models.walk_forward_validator.WalkForwardValidator`.
- **Signals / research / RL**:
  - `trading.signals.sentiment_signals.SentimentSignals`.
  - `trading.agents.strategy_research_agent.StrategyResearchAgent`.
  - `rl.rl_trader.RLTrader`.
- **UI helpers**:
  - `ui.page_assistant.render_page_assistant` (called at end to attach page assistant for “Strategy Testing”).
  - `trading.ui.strategy_components.render_strategy_selector`, `render_backtest_results`, `render_strategy_metrics`, `render_trade_list`.
- **Memory / agents**:
  - `trading.memory.get_memory_store` (wrapped via `_get_memory_store`).
  - `trading.memory.memory_store.MemoryType`.

### Stubs & Incomplete Features
- **Custom code execution in Advanced Editor**:
  - The `test_code` path raises `NotImplementedError("Custom code execution disabled for security...")`, so advanced code cannot be run from within the app. All subsequent test-plot logic is unreachable.
- **Strategy Builder load functionality**:
  - `📂 Load Strategy` in Builder tab only displays an info message; it does not populate the form fields with the selected strategy’s configuration.
- **Sensitivity Analysis**:
  - When the user clicks `🚀 Run Sensitivity Analysis`, the app displays a message that implementation is in progress. No actual sensitivity computations are performed.
- **Parameter Optimization**:
  - When the user clicks `🚀 Run Optimization`, the app displays a message that implementation is in progress. No optimization routine is run.

