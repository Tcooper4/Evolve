## PAGE: 6_Risk_Management.py

### Tabs
- **Main Tabs**:
  - `📊 Risk Dashboard`
  - `📉 VaR Analysis`
  - `🎲 Monte Carlo`
  - `💥 Stress Testing`
  - `🔬 Advanced Analytics`

### Expanders
- **Global**:
  - `_empty_state(...)` helper renders a styled message; not tied to specific expanders but used across tabs when data is missing.
- **VaR Analysis tab**:
  - None; uses headings and charts only.
- **Monte Carlo tab**:
  - `🔧 Advanced Parameters` – shows random seed, block size (for block bootstrap), and path-visualization options.
- **Stress Testing tab**:
  - None; scenario builders use inline columns and controls.
- **Advanced Analytics tab**:
  - `📊 Detailed Risk Driver Analysis` – shows full table for identified risk drivers.

### Sidebar
- This page does **not** define any `st.sidebar` UI; all controls are in the main content area.

### Major Features

#### Initialization and Core State
- **Backends and availability**:
  - Attempts to import:
    - `trading.risk.risk_manager.RiskManager`.
    - `risk.advanced_risk.AdvancedRiskAnalyzer`.
    - `trading.backtesting.monte_carlo.MonteCarloSimulator`, `MonteCarloConfig`.
    - `trading.portfolio.portfolio_manager.PortfolioManager`.
    - `trading.data.data_loader.DataLoader`, `DataLoadRequest`.
  - Sets `RISK_MODULES_AVAILABLE` flag accordingly.
- **Session state keys**:
  - `risk_manager`: instance of `RiskManager` with a default risk configuration, or `None` if modules unavailable.
  - `advanced_risk_analyzer`: instance of `AdvancedRiskAnalyzer` or `None`.
  - `portfolio_manager`: `PortfolioManager` instance or `None`.
  - `risk_alerts`: list of risk violations and alerts.
  - `monte_carlo_results`: cached Monte Carlo outputs dict (or `None`).
  - `risk_history`: list of historical risk snapshots.
  - `risk_limits`: dict of risk limits:
    - `max_var`, `max_volatility`, `max_drawdown`, `max_beta`, `max_position_size`.

#### Shared Helper Functions
- **Risk Dashboard helpers**:
  - `calculate_risk_level(metrics, limits)`:
    - Produces `"Low"`, `"Medium"`, or `"High"` with a color based on VaR, volatility, drawdown, and beta vs limits.
  - `create_risk_gauge(risk_level, color)`:
    - Returns a Plotly gauge indicator figure for risk level.
  - `check_risk_limits(metrics, limits)`:
    - Compares metrics against limits and returns a list of violations with severity (`warning`/`critical`).
- **Portfolio data and risk metrics**:
  - `get_portfolio_data()`:
    - Uses `st.session_state.portfolio_manager.state.open_positions` to assemble a `positions` dict (`symbol` → `size`).
    - If no positions, returns `(None, None, True)` indicating an empty-state.
    - If `backtest_results` exists with an `equity_curve` containing `equity`, uses its percent-change series as `returns`.
  - `calculate_risk_metrics(returns)`:
    - Derives volatility, mean return, VaR (95%/99%), CVaR, max and average drawdown, placeholder beta, Sharpe, and Sortino.
  - `compute_risk_score(sharpe, max_dd, cvar, beta)`:
    - Returns `(score, label, color)` where score 0–100 is a composite risk score; label is `Low Risk`, `Moderate Risk`, or `High Risk`.
- **VaR helpers**:
  - `calculate_historical_var`, `calculate_parametric_var`, `calculate_monte_carlo_var`, `calculate_cvar`, `calculate_component_var`, `backtest_var`.
- **Monte Carlo helpers**:
  - `calculate_drawdown_distribution(simulated_paths)`:
    - Computes max drawdown per simulated path.
  - `calculate_probability_metrics(final_values, initial_capital)`:
    - Returns probabilities of profit, loss, big loss (>10%), and big profit (>10%).
- **Stress testing helpers**:
  - `get_historical_scenarios()`:
    - Returns definitions for:
      - `2008 Financial Crisis`, `2020 COVID Crash`, `1987 Black Monday`, `2000 Dot-com Bubble`.
  - `apply_stress_scenario(returns, scenario, positions)`:
    - Applies market/sector shocks to returns and compiles `portfolio_impact` per symbol.
  - `estimate_recovery_time(scenario, portfolio_impact_pct)`:
    - Adjusts `recovery_days` by severity of portfolio impact.
- **Advanced analytics helpers**:
  - `calculate_correlation_matrix(returns, positions, window)`:
    - Uses `DataLoader` and `YFinanceProvider` plus `trading.risk.risk_metrics.compute_correlation_matrix` if available; falls back to identity matrix on failure or insufficient data.
  - `calculate_factor_decomposition(returns)`:
    - Returns simple factor exposure and contributions (70% market, 30% idiosyncratic – placeholder).
  - `calculate_liquidity_risk(positions, portfolio_value)`:
    - Simulates liquidity scores, days to liquidate, and liquidity risk classification for each symbol (placeholder).
  - `calculate_concentration_risk(positions)`:
    - Computes HHI, top-5 concentration, effective number of positions, maximum concentration, and a qualitative risk level.
  - `calculate_greek_exposure(positions)`:
    - Returns random placeholder deltas/gammas/thetas/vegas/rhos for each symbol.
  - `calculate_rolling_risk_metrics(returns, window)` and `calculate_max_drawdown_simple(returns)`:
    - Compute rolling volatility, Sharpe, drawdown, skewness, kurtosis, and VaR for Advanced Analytics tab.

### Tab Details

#### Tab 1 – `📊 Risk Dashboard`
- **Purpose**: Real-time portfolio risk overview, including VaR, drawdown, volatility, beta, composite risk score, alerts, and correlation heatmap.
- **Data pipeline**:
  - Calls `get_portfolio_data()` to obtain `returns` and `positions`.
  - Uses `RiskManager` (if available) via `risk_manager.update_returns(returns)` and `risk_manager.current_metrics`; otherwise uses `calculate_risk_metrics`.
- **Outputs**:
  - **Risk gauge**:
    - Gauge plot and metrics for VaR (95%), CVaR (95%), Sharpe, Sortino, max drawdown, beta.
  - **Composite Risk Score**:
    - Score 0–100 and label (`Low Risk`/`Moderate Risk`/`High Risk`).
  - **Risk limits status**:
    - Violations from `check_risk_limits` displayed as `st.error` or `st.warning`.
    - Violations recorded in `st.session_state.risk_alerts`.
  - **Alerts feed**:
    - Last ten alerts in reverse chronological order.
  - **Correlation heatmap**:
    - Uses `calculate_correlation_matrix(returns, positions, window=60)` and `px.imshow`.
  - **Risk by position**:
    - Simple VaR and volatility allocations per symbol, derived from overall metrics scaled by weight.
  - **Short interest & squeeze risk**:
    - For a focus symbol in positions, fetches `get_short_interest(symbol)` and shows metrics:
      - Short float %, days to cover, squeeze score and signal.
  - **Historical risk trend**:
    - Uses `st.session_state.risk_history` (metrics over time) to plot subplots for VaR, volatility, and max drawdown with limit lines.
- **External integrations**:
  - `RiskManager`, `AdvancedRiskAnalyzer` (optional).
  - `trading.memory.get_memory_store`, `MemoryType.LONG_TERM`:
    - Periodically writes risk snapshots under namespace `risk`, category `snapshots`, at most once per hour and only if volatility and Sharpe are available.

#### Tab 2 – `📉 VaR Analysis`
- **Purpose**: Value at Risk calculation (Historical/Parametric/Monte Carlo), CVaR, component VaR, and VaR backtesting.
- **User controls**:
  - `VaR Calculation Method`: `Historical`, `Parametric`, or `Monte Carlo`.
  - `Confidence Level` slider \(90–99%\).
  - `Time Horizon` dropdown (1-day, 10-day, 1-month).
  - If Monte Carlo: `Number of Simulations` slider (1,000–50,000).
  - `Backtesting Window` slider for rolling VaR backtesting.
- **Outputs**:
  - VaR and CVaR as percentages and dollar-equivalents (assuming `portfolio_value` from session, default \$100k).
  - Histograms of returns with VaR and CVaR lines.
  - Comparison table of VaR for 90/95/99% confidence at the chosen horizon.
  - **Component VaR**:
    - Table and bar chart of VaR contributions per symbol based on a simplified proportional model.
  - **VaR backtesting**:
    - Rolling VaR predictions vs actual returns, counts of violations, violation rate vs expected rate, average exceedance, and a validation flag.
    - Time-series plot of returns, predicted VaR, and marked violations.
- **External integrations**:
  - `scipy.stats.norm` if available; otherwise uses hard-coded z-scores.

#### Tab 3 – `🎲 Monte Carlo`
- **Purpose**: Portfolio-level Monte Carlo simulation with configurable horizon, bootstrap method, and visualization.
- **User controls**:
  - Simulation size: 1,000–100,000 paths.
  - Time horizon unit (Days/Weeks/Months) and value.
  - Bootstrap method: `historical`, `block`, `parametric`.
  - Initial capital (pulls default from `portfolio_value` in session).
  - Advanced parameters: random seed, block size (for block bootstrap), whether to show individual paths and how many.
  - `🚀 Run Monte Carlo Simulation` button.
- **Backend calls**:
  - `MonteCarloConfig` and `MonteCarloSimulator` from `trading.backtesting.monte_carlo`.
  - Adjusts `returns` to the horizon by truncation or repeating.
- **Outputs**:
  - Stores simulation outputs in `st.session_state.monte_carlo_results`:
    - Simulator instance, simulated paths DataFrame, final values and returns, drawdown distribution, probability metrics, and config.
  - Summary metrics:
    - Mean/median final value and returns, P95 and P5 final value/returns, standard deviation and volatility.
  - Percentile outcomes table and probability metrics.
  - **Paths visualization**:
    - Optional sampled paths plus percentile bands (P10, P50, P90, Mean) based on `simulator.percentiles`.
  - Distribution charts:
    - Histograms for final returns and maximum drawdowns.
  - **Exports**:
    - CSV of final values/returns/drawdowns.
    - Text summary report with configuration and high-level results.

#### Tab 4 – `💥 Stress Testing`
- **Purpose**: Test portfolio behaviour under historical crises, factor shocks, and fully custom scenarios.
- **Modes**:
  - `Historical Scenarios`:
    - Lets the user pick from predefined crisis scenarios.
    - Summarizes description, market shock, volatility multiplier, duration, and recovery.
    - `🚀 Run Stress Test` applies scenario to `returns` and positions via `apply_stress_scenario`.
  - `Factor Stress Tests`:
    - Controls for:
      - Interest rate shock, volatility shock, correlation increase.
      - Sector-specific shocks: Technology, Financial, Energy, Consumer.
    - Builds a `custom_scenario` dict and runs `apply_stress_scenario`.
  - `Custom Scenario Builder`:
    - Allows naming and describing a scenario.
    - Configures market shock, volatility multiplier, duration/recovery days.
    - Defines per-symbol position shocks for each held symbol.
    - Runs `apply_stress_scenario` with user-defined parameters.
- **Outputs**:
  - `stress_test_result` in session state:
    - Scenario name and metadata, shocked returns, per-position impacts, portfolio-level impact percentage/dollar, estimated recovery time, and post-stress portfolio value.
  - Results section:
    - Portfolio impact metrics, estimated recovery time.
    - Table and bar chart of position-level impacts by symbol and shock.
  - **Scenario comparison**:
    - Maintains `stress_test_history` in session state to compare multiple scenarios in a table and bar chart.
  - `🗑️ Clear Results` button resets `stress_test_result`.

#### Tab 5 – `🔬 Advanced Analytics`
- **Purpose**: Deep-dive analysis: correlation, factor decomposition, tail risk, liquidity and concentration risk, Greek exposure, risk-adjusted performance, rolling metrics, and causal risk drivers.
- **Features**:
  - **Correlation Matrix**:
    - Uses `calculate_correlation_matrix` and shows a heatmap plus summary metrics (average, max, min correlation) with diversification guidance.
  - **Factor Decomposition**:
    - Displays market vs idiosyncratic exposures and returns using simplified placeholders.
  - **Tail Risk**:
    - Skewness, kurtosis, and qualitative interpretations.
    - Histogram of returns with 5th percentile marker.
  - **Liquidity Risk**:
    - Table of simulated liquidity scores, position values, days to liquidate, and risk classification.
    - Bar chart of days to liquidate per symbol.
  - **Concentration Risk**:
    - HHI, top-5 concentration, effective positions, max position size, and a qualitative concentration risk message.
    - Donut chart of position concentration.
  - **Greek Exposure**:
    - Table of placeholder Greek exposures for each symbol with a note clarifying they are not based on actual options data.
  - **Risk-Adjusted Metrics**:
    - Sharpe, Sortino, Calmar, volatility, and other metrics:
      - If `AdvancedRiskAnalyzer` is present, uses its values (availability of `omega_ratio`, `gain_loss_ratio`, `profit_factor`, `recovery_factor`).
      - Otherwise, computes or approximates from returns.
  - **Rolling Risk Metrics**:
    - With window slider, uses `calculate_rolling_risk_metrics` to plot volatility, Sharpe, max drawdown, skewness, kurtosis, and VaR(95%) in a 3×2 subplot grid.
  - **Risk Driver Analysis**:
    - On `Analyze Risk Drivers`, attempts to import `causal.driver_analysis.DriverAnalysis` and run `identify_risk_drivers` on `portfolio_data` (either from `st.session_state.portfolio_data` or constructed from `portfolio_manager.get_positions()` and returns).
    - Displays top risk drivers with VaR impact and risk contribution, plus optional pie chart and table.
    - Shows guidance if portfolio data is missing.

### Buttons with Actions (Non-Cosmetic)
- **Risk Dashboard**:
  - `🔄 Auto-refresh (30s)` – triggers rerun when toggled, but no explicit sleep/time loop is implemented; the risk metrics are recomputed on each render.
- **VaR Analysis**:
  - VaR configuration controls update calculated VaR/CVaR metrics and charts.
  - `Backtesting Window` slider drives VaR backtest calculations.
- **Monte Carlo**:
  - `🚀 Run Monte Carlo Simulation` – executes simulation and populates session-state results.
  - `📥 Download Results (CSV)` and `📄 Download Summary Report (TXT)` – export simulation outputs and a text summary.
- **Stress Testing**:
  - `🚀 Run Stress Test` (historical scenarios).
  - `🚀 Run Factor Stress Test`.
  - `🚀 Run Custom Stress Test`.
  - `🗑️ Clear Results` – clears stress test result and reloads.
- **Advanced Analytics**:
  - `Analyze Risk Drivers` – initiates causal risk driver analysis (if module is available).

### Session State Keys
- **Risk core**:
  - `risk_manager`, `advanced_risk_analyzer`, `portfolio_manager`, `risk_limits`, `risk_alerts`, `risk_history`.
- **Data and derived values**:
  - `monte_carlo_results` (simulator, paths, returns, drawdowns, config).
  - `portfolio_value` (used for VaR, Monte Carlo, stress tests, liquidity and impact calculations).
  - `stress_test_result`, `stress_test_history`.
- **UI and controls**:
  - Tab-specific keys for sliders, checkboxes, and selectboxes (e.g., `mc_simulations`, `backtest_window`, `auto_refresh`, `execution_mode` in shared session, etc.).

### External Integrations (trading/agents/components/system)
- `trading.risk.risk_manager.RiskManager`.
- `risk.advanced_risk.AdvancedRiskAnalyzer`.
- `trading.backtesting.monte_carlo.MonteCarloSimulator`, `MonteCarloConfig`.
- `trading.portfolio.portfolio_manager.PortfolioManager`.
- `trading.data.data_loader.DataLoader`, `DataLoadRequest`.
- `trading.data.providers.yfinance_provider.YFinanceProvider`.
- `trading.risk.risk_metrics.compute_correlation_matrix`.
- `trading.data.short_interest.get_short_interest`.
- `trading.memory.get_memory_store`, `trading.memory.memory_store.MemoryType`.
- `ui.page_assistant.render_page_assistant("Risk Management")`.
- `causal.driver_analysis.DriverAnalysis` (optional).

### Stubs & Incomplete Features
- **RiskManager and AdvancedRiskAnalyzer absence**:
  - When imports fail, the page falls back to simplified risk calculations and sets the corresponding session state entries to `None`. Advanced metrics like omega/gain_loss/profit_factor/recovery_factor may default to approximations or zeros.
- **Liquidity, Greek, and factor models**:
  - Liquidity scores, Greek exposures, and factor decomposition are explicitly **simplified placeholders** (random draws or fixed splits) and not derived from actual microstructure/option/factor models.
- **Risk driver analysis**:
  - Requires external `causal.driver_analysis` module; when missing or failing, the feature shows appropriate error messages and guidance but performs no analysis.

