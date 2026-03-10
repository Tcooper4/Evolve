## PAGE: 5_Portfolio.py

### Tabs
- **Main Tabs**:
  - `📈 Overview`
  - `💼 Positions`
  - `📊 Performance`
  - `⚙️ Optimization`
  - `💰 Tax & Accounting`

### Expanders
- **Shared UI**:
  - `_empty_state(...)` is used in several contexts to render a styled placeholder (not a named tab).
- **Overview tab**:
  - `Sample Correlation Matrix (demo)` (expander text) – shown when there are no open positions but a correlation demo is rendered.
- **Positions tab**:
  - For option positions:
    - `📊 All Greeks` – shows theta, vega, rho when Greeks are successfully computed.
  - For each expandable position card:
    - The card itself is implemented as a `st.expander` with a header summarizing status, symbol, size, and P&L.
    - Within “⚡ Actions” for closing, a confirmation dialog uses `st.warning` and columns but not a separate expander.
- **Performance, Optimization, Tax tabs**:
  - Use headings and charts but no additional `st.expander` blocks beyond those noted above.

### Sidebar Elements
- **Sidebar title**:
  - `Controls`.
- **Portfolio State**:
  - Text input: `Portfolio File` (default `portfolio.json`).
  - Button: `Load Portfolio` (calls `load_portfolio_state(filename)`).
  - Button: `Save Portfolio` (calls `save_portfolio_state(filename)`).
- **Filters**:
  - Multiselect: `Strategy` – values from all open and closed positions’ `strategy`.
  - Multiselect: `Symbol` – values from all open and closed positions’ `symbol`.
- **Export Options**:
  - Selectbox: `Export Format` – `CSV` or `PDF`.
  - Button: `Export Trade Report` – builds a trade report DataFrame from closed positions and calls `utils.report_exporter.export_trade_report`.

### Major Features

#### Portfolio Initialization and Helpers
- **PortfolioManager**:
  - Initialized and stored in `st.session_state.portfolio_manager` if not present or `None`.
  - `portfolio = st.session_state.portfolio_manager` is used as the central access point.
  - If `portfolio` is missing a `state` attribute or `state` is `None`, the page displays an error and stops.
- **Helper functions**:
  - `calculate_option_greeks(...)`:
    - Uses Black–Scholes via `scipy.stats.norm` and `numpy` to compute delta, gamma, theta, vega, rho for a call or put.
    - Returns zeroed Greeks if `scipy` is unavailable or an exception occurs.
  - `calculate_position_beta(ticker, market_ticker='SPY', days=252)`:
    - Downloads historical data for a ticker and market index using `yfinance.download`.
    - Computes daily returns and uses `utils.math_utils.calculate_beta`.
    - Returns `1.0` if data is insufficient or if any error occurs.
  - `load_portfolio_state(filename)` / `save_portfolio_state(filename)`:
    - Delegate to `PortfolioManager.load/save` and show Streamlit success/error messages.
  - `plot_equity_curve(positions)`:
    - Builds a DataFrame of entry/exit events from `Position` objects, computes cumulative PnL, and returns a Plotly `Figure` with equity curve plus entry/exit markers.
  - `plot_rolling_metrics(positions, window)`:
    - Uses closed positions to compute return per trade and rolling Sharpe and win rate; returns a Plotly `Figure`, or `None` with logging if stats fail.
  - `plot_strategy_performance(positions)`:
    - Aggregates PnL and returns by `strategy`, computes per-strategy Sharpe, and returns a grouped bar chart of Sharpe vs total PnL.

#### Overview Tab (`📈 Overview`)
- **Performance summary**:
  - Fetches `summary = portfolio.get_performance_summary()`.
  - Displays metrics:
    - `Total PnL`, `Sharpe Ratio`, `Max Drawdown`, `Win Rate`.
- **Backtest integration**:
  - If no live positions but `st.session_state.backtest_results` exists:
    - Shows simulated equity (last value) and total return based on backtest result structure.
- **Portfolio analysis (allocation)**:
  - Calls `portfolio.get_portfolio_allocation()`:
    - Splits weights into sector and asset class allocation using a predefined `sector_mapping`.
    - Renders:
      - Sector allocation donut chart.
      - Asset class allocation donut chart.
- **Risk metrics**:
  - Computes approximate portfolio returns from closed positions and uses:
    - `calculate_volatility` to get annualized volatility.
    - `DataLoader` / `DataLoadRequest` and `calculate_beta` to estimate portfolio beta vs SPY.
  - Displays:
    - `Portfolio Beta`, `Portfolio Volatility`, `Max Position Size`, `Number of Positions`.
- **Correlation analysis**:
  - For open positions:
    - Uses `DataLoader` / `DataLoadRequest` to load historical close data for each symbol plus `SPY`, `QQQ`, `DIA`, `IWM`.
    - Builds a correlation matrix and renders a heatmap.
  - If no open positions:
    - Shows a “Sample Correlation Matrix (demo)” expander using `yfinance.download` on demo symbols (SPY, QQQ, AAPL, MSFT, NVDA).
- **Rebalance suggestions**:
  - Based on `portfolio_allocation` and `sector_allocation`, generates suggestions when:
    - Any position > 25% of portfolio.
    - Any sector > 40% of portfolio.
    - Cash > 20% or < 5%.
    - Fewer than 5 open positions.
  - Suggestions are shown as `st.warning`/`st.info` with a `💡 Suggestion` caption.
- **Position summary table**:
  - Fetches `positions_df = portfolio.get_position_summary()`.
  - Applies sidebar `Strategy` and `Symbol` filters.
  - Displays the table.
- **Position consolidation**:
  - On `Consolidate Positions`:
    - Imports `PositionConsolidator` from `trading.optimization.utils.consolidator`.
    - Converts open positions to a list of dicts including symbol, size, entry_price, direction, strategy, value.
    - Calls `consolidator.consolidate_positions` with `min_position_size=100` and `max_positions=20`.
    - Displays pre/post counts, eliminated and merged positions, and details via an expander.
    - `Apply Consolidation` button:
      - Writes `consolidated_positions` into `st.session_state` as a preview but does not modify the portfolio in-place.
- **Performance visualization**:
  - Uses internal helpers with tabs:
    - `Equity Curve` (closed positions).
    - `Rolling Metrics` (window slider).
    - `Strategy Performance`.
  - When no history, shows `_empty_state` text.

#### Positions Tab (`💼 Positions`)
- **Purpose**: Inspect and manage individual positions with metrics, notes, and actions.
- **What user does**:
  - Optional search filter by symbol/strategy, and toggle for showing closed positions.
  - For each position (open or closed, after filtering):
    - Expands a card to view detailed metrics, charts, notes, tags, and action controls.
- **Backend calls**:
  - Per position:
    - Current price:
      - `DataLoader.load_data` over a short lookback to get latest close (fallback to entry price).
    - Beta:
      - `calculate_position_beta` (internally uses `yfinance` and `calculate_beta`).
    - Position volatility:
      - Uses `DataLoader` and historical close returns.
    - P&L over time:
      - `DataLoader.load_data` from entry to current/exit date and constructs a line plot of P&L.
    - Option Greeks:
      - For symbols ending with `C` or `P`, uses `calculate_option_greeks` with heuristics for strike, time_to_expiry, volatility unless attributes exist on `Position`.
- **Outputs**:
  - Position detail section (three columns):
    - Position details (entry/current price, size, direction).
    - P&L metrics (unrealized P&L, position value, days held, take profit).
    - Risk management (stop loss, strategy name, timestamps).
  - Position metrics (beta, volatility, Greeks or generic risk/correlation placeholders).
  - Per-position P&L chart vs time.
  - Position history table (entry vs current/exit).
  - Notes and tags:
    - Managed via `st.session_state.position_notes` and `st.session_state.position_tags` keyed by `symbol_entry_time`.
  - Actions:
    - `🛑 Close Position`:
      - Triggers confirmation flow; on confirm, calls `portfolio.close_position(position, current_price)` and reloads.
    - Partial close:
      - Slider to pick partial percentage and button to “close” this portion, but currently only shows an info message; no actual size modification is applied.
    - Risk level update:
      - Inputs for `Stop Loss` and `Take Profit`, but update button only shows an info message; no modifications are applied to `Position` objects.

#### Performance Tab (`📊 Performance`)
- **Purpose**: Time-series performance, benchmark comparison, attribution, and risk-adjusted metrics.
- **What user does**:
  - Requires closed positions.
  - Views charts and selects benchmarks and rolling window parameters.
- **Backend calls**:
  - Uses `PerformanceMetrics` from `trading.utils.performance_metrics` for:
    - Sharpe, Sortino, cumulative returns, max drawdown.
  - Data loading:
    - `DataLoader`, `DataLoadRequest` for benchmark symbols.
- **Outputs**:
  - Portfolio value over time chart with entry/exit markers.
  - Returns by period (daily/monthly/yearly) metrics and distributions.
  - Benchmark comparison:
    - Normalized portfolio vs chosen ETF (SPY, QQQ, DIA, IWM, VTI).
    - Outperformance metric.
  - Performance attribution:
    - P&L by strategy and by symbol, with bar charts and tables.
  - Risk-adjusted metrics:
    - Sharpe, Sortino, Calmar, max drawdown, annualized volatility, win rate.
  - Rolling metrics:
    - Rolling Sharpe and volatility (dual-axis) and rolling returns, based on user-selected window.

#### Optimization Tab (`⚙️ Optimization`)
- **Purpose**: Optimize portfolio weights using mean–variance, Black–Litterman, or allocator-based strategies (Risk Parity, Maximum Sharpe, Minimum Variance).
- **What user does**:
  - Requires at least one open position and a working `PortfolioAllocator` import.
  - Selects:
    - `Optimization Method`.
    - Target return (for mean–variance/Black–Litterman).
    - Risk aversion and weight bounds.
    - If Black–Litterman:
      - Views and confidence per symbol.
  - Clicks `🚀 Run Optimization`.
- **Backend calls**:
  - Uses `PortfolioOptimizer` and `PortfolioAllocator`:
    - `mean_variance_optimization`.
    - `black_litterman_optimization`.
    - `PortfolioAllocator.allocate_portfolio` with an `AllocationStrategy` for non-MV/BL methods.
  - Returns historical data via `DataLoader` for each current symbol.
- **Outputs**:
  - Stores result in `st.session_state['optimization_result']`.
  - Displays:
    - Current vs optimal allocations (bar charts).
    - Rebalancing recommendations with buy/sell deltas.
    - Optimization summary metrics (expected return, volatility, Sharpe, max drawdown if provided).
  - Efficient frontier:
    - Computes a frontier via repeated mean–variance optimization and overlays current portfolio point.
  - What-if scenarios:
    - Accepts a symbol, hypothetical weight, and return; currently only logs an info message – actual portfolio impact calculation is not implemented.

#### Tax & Accounting Tab (`💰 Tax & Accounting`)
- **Purpose**: Tax lot construction, realized/unrealized gain analysis, dividends, tax-loss harvesting, wash sale detection, and export for tax filing.
- **What user does**:
  - Selects tax lot method (FIFO/LIFO/Specific ID) and tax year/export format/flags.
  - Views tax lot tables and gains/losses charts.
  - Views dividend history and “projected” dividend yields.
  - Reviews tax-loss harvesting opportunities and wash sale warnings.
  - Clicks `📥 Generate Tax Report` to build a tax report for a specific year and export CSV (or stub for PDF/Form 8949).
- **Backend calls**:
  - Dividend history:
    - `trading.data.providers.get_data_provider` (though the data provider object is not directly used; actual data is fetched via `yfinance.Ticker(...).dividends`).
  - No direct integration with tax filing APIs in this file; exports are local CSV or stubbed messages.
- **Outputs**:
  - Tax lot DataFrame with cost basis, proceeds, realized gains, and term (long/short).
  - Aggregated metrics for total, short-term, long-term gains.
  - Realized gains/losses by chosen period.
  - Unrealized gains/losses by symbol and totals.
  - Dividend history and random projected yields (projections are explicitly synthetic: `np.random.uniform`).
  - Tax loss harvesting table of positions with unrealized losses.
  - Wash sale warnings based on 30-day window between loss sale and subsequent buys.
  - Tax report DataFrame per year with export to CSV and summary metrics.

### Buttons with Actions (Non-Cosmetic)
- **Sidebar**:
  - `Load Portfolio` – triggers `load_portfolio_state`.
  - `Save Portfolio` – triggers `save_portfolio_state`.
  - `Export Trade Report` – builds a DataFrame from closed positions and calls `export_trade_report`.
- **Overview**:
  - `Consolidate Positions` – runs position consolidation algorithm and optionally saves preview to session state; no direct mutation of `PortfolioManager` is performed.
  - `Apply Consolidation` – saves consolidated positions preview to `st.session_state.consolidated_positions` and shows success message.
- **Positions**:
  - `🛑 Close Position` – initiates confirmation; on confirm, `portfolio.close_position(position, current_price)`.
  - `📉 Close {partial_close_pct}%` – logs info about partial closure but does not alter positions.
  - `💾 Update` (risk levels) – logs a message; does not change stop-loss/take-profit on the position object.
- **Performance**:
  - Benchmark selection and rolling window sliders affect derived charts and metrics; no backend mutating calls.
- **Optimization**:
  - `🚀 Run Optimization` – executes portfolio optimization logic and stores result.
- **Tax & Accounting**:
  - Realized and unrealized gains analysis is recalculated on UI events; `📥 Generate Tax Report` builds CSV and summary.

### Session State Keys
- **Core portfolio state**:
  - `portfolio_manager`.
- **Filters & exports**:
  - Sidebar: `filename`, `selected_strategy`, `selected_symbol`, `export_mode`.
- **Overview**:
  - None beyond standard keys; uses `portfolio_allocation`, etc. as local variables.
- **Positions**:
  - `position_notes`: dict keyed by `f"{symbol}_{entry_time.isoformat()}"`.
  - `position_tags`: dict keyed similarly.
  - `position_search`, `show_closed_pos`, plus per-position confirmation and risk slider/inputs.
- **Performance**:
  - Rolling window and benchmark selection keys (`rolling_window`, `benchmark_select`).
- **Optimization**:
  - `opt_method`, `target_return`, `risk_aversion`, `min_weight`, `max_weight`.
  - For Black–Litterman: `view_{symbol}`, `conf_{symbol}` keys.
  - `optimization_result` containing weights and metrics.
- **Tax & Accounting**:
  - `tax_lots` (unused placeholder), `tax_method`, `tax_method_select`.
  - Gains analysis: `gains_period`.
  - Tax export: `tax_year`, `export_format`, `export_dividends`, `export_wash_sales`, `export_summary`.

### External Integrations (Imports under `trading/`, `agents/`, `components/`, `system/`)
- **Portfolio & data**:
  - `trading.portfolio.portfolio_manager.PortfolioManager`, `Position`.
  - `trading.data.data_loader.DataLoader`, `DataLoadRequest`.
  - `trading.utils.performance_metrics.PerformanceMetrics`.
  - `trading.optimization.portfolio_optimizer.PortfolioOptimizer`.
  - `trading.optimization.utils.consolidator.PositionConsolidator`.
  - `portfolio.allocator.PortfolioAllocator`, `AllocationStrategy` (via direct or fallback import).
  - `trading.data.providers.get_data_provider` (used in dividend history).
- **Math & utilities**:
  - `utils.math_utils.calculate_volatility`, `calculate_beta`.
  - `utils.report_exporter.export_trade_report`.
- **UI**:
  - `ui.page_assistant.render_page_assistant("Portfolio")` at the end of the file.
- **Third-party**:
  - `yfinance` for correlation demos and dividend lookup.
  - `scipy.stats.norm` for option Greeks.
  - `numpy`, `pandas`, `plotly.graph_objects`.

### Stubs & Incomplete Features
- **Partial close actions**:
  - Partial close button only logs an informational message and does not adjust the position size or create a closing trade.
- **Risk level updates for individual positions**:
  - `Update Risk Levels` section collects new stop-loss/take-profit values but the `💾 Update` button does not change the `Position` object; it only shows an info message that this would be implemented.
- **What-If Scenarios in Optimization tab**:
  - The scenario analysis logs a textual description of the scenario but does not compute or present any impact on portfolio metrics.
- **PDF and Form 8949 exports**:
  - Tax report export supports CSV; `PDF` and `Form 8949 Format` paths are acknowledged with informational messages, but actual generation is not implemented.
- **Dividend projections**:
  - Projections use randomly generated yields via `np.random.uniform` for demo purposes, not real yield data.
- **PortfolioAllocator import failure**:
  - When `PortfolioAllocator` is not available, the Optimization tab displays an error and `st.stop()`, disabling optimization features for that environment.

