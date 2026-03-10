## PAGE: 4_Trade_Execution.py

### Tabs
- This page does **not** use `st.tabs`; all functionality is organized as sequential sections and `st.expander` blocks on a single page.

### Expanders
- **Global error handling**:
  - No named expander; errors are shown via `st.error(traceback.format_exc())` at file end.
- **Advanced Orders section**:
  - `📦 Bracket Orders (Entry + Take Profit + Stop Loss)` (expanded by default)
  - `📈 Trailing Stop Orders`
  - `🔀 Conditional Orders (If-Then)`
  - `🔄 OCO Orders (One Cancels Other)`
  - `🦵 Multi-Leg Orders`
- **Automated Execution section**:
  - `⚙️ Execution Configuration` (per-selected strategy)
- **Other sections**:
  - Several analytics subsections use headings and charts but not `st.expander`.

### Sidebar
- This page does **not** define any `st.sidebar` elements. All UI is in the main layout using `st.columns`, `st.expander`, and section headers.

### Major Features

#### Quick Trade (Section: `🚀 Quick Trade`)
- **Purpose**: Single-ticket order entry for simple market/limit trades with integrated pre-trade risk checks and order summary.
- **What user does**:
  - Clicks `🔌 Connect / Load market data` to enable live pricing (sets `trade_page_loaded`).
  - Toggles trading mode between `Paper Trading` and `Live Trading`.
  - Enters:
    - `Symbol` (ticker).
    - `Side` (Buy/Sell).
    - `Order Type` (Market/Limit).
    - `Quantity`.
    - (If Limit) `Limit Price`, `Time in Force` (Day, GTC, IOC, FOK).
  - Optionally uses the **Position Size Calculator** to compute suggested quantity based on account value, risk per trade, and stop-loss percentage.
  - Reviews **Pre-Trade Risk Check** messages and **Order Summary**.
  - Checks `I confirm this trade` and clicks `🚀 Submit {Side} Order`.
- **Backend calls**:
  - Pricing:
    - `yfinance.Ticker` / `fast_info` or `.history` via cached helper `get_current_price` and `_current_price`.
  - Execution:
    - `ExecutionAgent` from `execution.execution_agent` (or fallback `execution` package).
    - Uses `ExecutionMode`, `OrderSide`, `OrderType`.
    - Asynchronous `agent.submit_order(...)` via an `asyncio` event loop.
  - Memory and audit:
    - `trading.memory.get_memory_store`, `MemoryType.LONG_TERM` to log accepted orders under `trades` namespace with `category="orders"`.
    - Optional commentary via `st.session_state.commentary_service.generate_trade_commentary`.
    - Optional WebSocket client (`utils.websocket_client.WebSocketClient`) to send a `trade_execution` event if `ws_client` is configured and connected.
    - Optional notifications via `system.infra.agents.notifications.notification_service.NotificationService`:
      - Sends email and/or Slack notifications based on `st.session_state.notification_settings`.
    - Optional audit logging via `st.session_state.audit_logger.log_strategy(...)`.
- **Outputs**:
  - Stores one or more dicts representing orders in:
    - `st.session_state.active_orders`.
    - `st.session_state.order_history`.
  - Risk-check list with:
    - `risk_checks` (validations passed).
    - `risk_warnings` (soft risk flags).
    - `risk_errors` (blocking issues).
  - Summary DataFrame of order details.
  - Recent orders table of the last 5 entries from order history.

#### Advanced Execution Settings & Algorithms (Section: `⚙️ Advanced Execution Settings`)
- **Purpose**: Configure and execute orders using advanced algorithms (TWAP, VWAP, Iceberg, AI “Smart” execution) through a dedicated execution engine.
- **What user does**:
  - Selects an execution algorithm:
    - `Market Order - Immediate execution`
    - `TWAP - Time-Weighted Average Price`
    - `VWAP - Volume-Weighted Average Price`
    - `Iceberg - Hide order size`
    - `Smart - AI-optimized execution`
  - For each algorithm, configures specific parameters:
    - TWAP: Duration (minutes), Number of slices.
    - VWAP: Participation rate (%).
    - Iceberg: Visible quantity (%).
    - Smart: Urgency (Low/Medium/High).
  - Enters order details in Advanced Engine `Order Entry`:
    - `Symbol`, `Action` (BUY/SELL), `Quantity`.
    - `Order Type` (Market/Limit/Stop/Stop-Limit) and appropriate `Limit Price`/`Stop Price`.
  - Clicks `🚀 Execute Order`.
- **Backend calls**:
  - `trading.execution.execution_engine.ExecutionEngine` (aliased as `AdvancedExecutionEngine`), instantiated once and stored in `st.session_state.execution_engine`.
  - `engine.execute_order(order_config)` with a config dict containing:
    - Symbol, action, quantity, order type, algorithm name, and algorithm-specific parameters, plus optional limit/stop prices.
- **Outputs**:
  - On success:
    - Displays metrics: average fill price, filled quantity, computed slippage vs expected price.
    - Optionally plots an execution timeline chart if `execution_timeline` is present in the result.
    - Appends an `order_info` dict (with algorithm, slippage, filled_quantity, etc.) into `active_orders` and `order_history`.
  - On failure:
    - Shows error message using `result['error']` or generic message.
  - If import of `AdvancedExecutionEngine` fails:
    - Renders an `_empty_state` with text “Advanced execution algorithms not available in this build.”

#### Advanced Orders (Section: `⚙️ Advanced Orders`)
- **Bracket Orders**:
  - **What user does**:
    - Configures entry (symbol, side, quantity, entry type and optional limit price).
    - Configures take-profit (percentage or fixed price) and stop-loss (percentage or fixed price), plus time in force.
    - Clicks `📦 Submit Bracket Order`.
  - **Backend calls**:
    - No broker or agent methods; this section is local logic only.
  - **Outputs**:
    - Calculates TP/SL absolute prices from percentages when needed, using current or entry price.
    - Shows confirmation messages for TP/SL.
    - Adds a `bracket_order` dict into `active_orders` and `order_history`.

- **Trailing Stop Orders**:
  - **What user does**:
    - Configures symbol, side, quantity, entry price, trailing type (percentage or fixed), trailing amount, activation price, and time in force.
    - Clicks `📈 Submit Trailing Stop Order`.
  - **Backend calls**:
    - No external execution call; logic is local.
  - **Outputs**:
    - Shows confirmation and trailing configuration summary.
    - Adds `trailing_order` dict into `active_orders` and `order_history`.

- **Conditional Orders (If-Then)**:
  - **What user does**:
    - Defines IF condition on symbol based on Price/Volume/Indicator (RSI, MACD, SMA) with operator and value.
    - Defines THEN action (symbol, side, quantity, order type, optional limit price).
    - Clicks `🔀 Submit Conditional Order`.
  - **Backend calls**:
    - No real-time evaluation engine here; conditions are stored but not wired to a scheduler in this file.
  - **Outputs**:
    - Shows success summary.
    - Adds `conditional_order` dict containing nested condition/action to `active_orders` and `order_history` with status `pending`.

- **OCO (One Cancels Other) Orders**:
  - **What user does**:
    - Configures two legs (side, quantity, type, price) for a symbol.
    - Clicks `🔄 Submit OCO Order`.
  - **Backend calls**:
    - Only local state updates; actual broker OCO wiring not shown.
  - **Outputs**:
    - Confirmation message.
    - Adds `oco_order` dict with both orders into session state (status `pending`).

- **Multi-Leg Orders**:
  - **What user does**:
    - Selects number of legs (2–4).
    - For each leg, inputs symbol, side, quantity, and price.
    - Clicks `🦵 Submit Multi-Leg Order`.
  - **Backend calls**:
    - None external; multi-leg behavior is conceptual here.
  - **Outputs**:
    - Confirmation message.
    - Adds `multileg_order` dict (list of legs) to active and historical orders.

#### Automated Execution (Section: `🤖 Automated Execution`)
- **Purpose**: Attach backtested strategies to live/paper execution with configurable safety constraints and monitoring.
- **What user does**:
  - Configures global:
    - `emergency_stop` toggle (via `🔴 EMERGENCY STOP` / `🟢 Resume Trading`).
  - Selects a strategy from combined built-in and custom registries (populated via `StrategyRegistry` and `CustomStrategyHandler` from `trading.strategies.registry` and `trading.strategies.custom_strategy_handler`).
  - Edits per-strategy execution config:
    - Symbols, max orders/day, max daily loss, max position size (%), minimum signal confidence, check interval (minutes).
  - Saves config (`💾 Save Configuration`).
  - Uses `▶️ Start Strategy` / `⏹️ Stop Strategy` to toggle strategy execution state.
- **Backend calls**:
  - Registries:
    - `get_strategy_registry()` and `get_custom_strategy_handler()` to retrieve available strategies.
  - Risk control:
    - `ExecutionRiskControlAgent` is imported but not directly used in this file.
  - Memory:
    - Execution logs are stored in `st.session_state.auto_execution_logs` but not persisted externally in this file.
- **Outputs**:
  - Session-managed execution config per strategy in `st.session_state.auto_execution_configs`.
  - Boolean active flags in `st.session_state.auto_execution_active`.
  - Emergency stop flag in `st.session_state.emergency_stop`.
  - Overview:
    - Active strategy list (each with an expander showing config and a stop button).
  - Real-time execution log:
    - Filterable by level, with clear-log functionality.

#### Order Management (Sections: `📋 Order Management`, `📊 Active Orders`, `📜 Order History`, `🔔 Fill Notifications`, `📈 Order Status Tracking`)
- **Purpose**: Monitor, modify, cancel, and audit both active and historical orders.
- **What user does**:
  - Enables/disables auto-refresh, adjusts refresh interval, or uses `🔄 Refresh Now`.
  - Views active orders aggregated from:
    - `ExecutionAgent.get_order_book()` and `get_order_status()`.
    - Session state `active_orders`.
  - Selects one or more orders via multiselect.
  - For a single selected order:
    - Modifies quantity and (for limit-type orders) price, then clicks `💾 Update Order`.
  - For multiple selected orders:
    - Uses batch operations:
      - `🗑️ Cancel All Selected` (currently updates state, not broker; placeholder comment).
      - `📊 Export Selected` to CSV.
  - Filters order history by symbol and status, adjusts the number of orders displayed, and exports history to CSV.
  - Manages fill notifications and clears them.
- **Backend calls**:
  - `ExecutionAgent.get_order_book()`, `ExecutionAgent.get_order_status(order_id)`.
  - Optional `ExecutionAgent.modify_order(...)` if method exists; fallback updates only session state.
  - No direct broker cancel API is wired; cancellation is simulated by updating `status` in session state.
- **Outputs**:
  - Active orders DataFrame (with price formatting and selection).
  - Order modification and cancellation updates to `st.session_state.active_orders`.
  - Order history DataFrame with filters and CSV export.
  - Fill notifications maintained in `st.session_state.fill_notifications`.
  - Status tracking summary metrics and bar chart derived from all orders (active + historical).

#### Execution Analytics (Section: `📊 Execution Analytics`)
- **Purpose**: Post-trade analysis of execution quality (slippage, price improvement, fill rates, execution time, VWAP/TWAP).
- **What user does**:
  - No inputs required; analytics auto-derive from orders with status `filled` in `order_history`.
- **Backend calls**:
  - Uses `st.session_state.execution_agent` (if present) to enhance analytics with:
    - `get_order_status(order_id)` for actual average price and timestamp.
  - Internal helper functions:
    - `calculate_vwap(prices, volumes)`.
    - `calculate_twap(prices, times)`.
- **Outputs**:
  - For each filled order, calculates:
    - `slippage_pct`, `slippage_abs`, `price_improvement`, `execution_time`.
  - Aggregates metrics:
    - Fill rate, average slippage, average execution time, total price improvement.
  - Visualizations:
    - Slippage distribution and slippage-by-order-type bar chart.
    - Fill rate over time and by symbol.
    - Price improvement distribution and by side.
    - Execution time distributions and by order type.
    - VWAP, TWAP, and average execution price, plus comparative chart.
  - Detailed analytics DataFrame with CSV export option.

### Buttons with Actions (Non-Cosmetic)
- **Top-level / Quick Trade**:
  - `🔌 Connect / Load market data`:
    - Sets `trade_page_loaded = True` and reruns app to enable price fetches.
  - `🔄 Refresh Orders`:
    - Triggers rerun to refresh order-related views.
  - `🚀 Submit {Side} Order`:
    - Submits single-ticket order via `ExecutionAgent`, updates order lists, writes to memory, optional commentary, WebSocket updates, notifications, and audit log.
  - `📋 Use Suggested Quantity`:
    - Writes `_suggested_quantity` in session state and reruns, updating `Quantity` default.

- **Advanced Execution Settings**:
  - `🚀 Execute Order`:
    - Executes algorithmic order via `AdvancedExecutionEngine.execute_order` and writes results/order info to session orders.

- **Advanced Orders**:
  - `📦 Submit Bracket Order`:
    - Creates bracket order dict and saves to orders lists; no broker call.
  - `📈 Submit Trailing Stop Order`:
    - Creates trailing stop order dict.
  - `🔀 Submit Conditional Order`:
    - Creates if-then conditional order dict.
  - `🔄 Submit OCO Order`:
    - Creates OCO order dict.
  - `🦵 Submit Multi-Leg Order`:
    - Creates multi-leg order dict.

- **Automated Execution**:
  - `🔴 EMERGENCY STOP` / `🟢 Resume Trading`:
    - Toggles `emergency_stop` and clears/updates active strategies; logs events.
  - `💾 Save Configuration`:
    - Writes per-strategy config to `auto_execution_configs`.
  - `▶️ Start Strategy` / `⏹️ Stop Strategy`:
    - Toggles `auto_execution_active[selected_strategy]` and adds log entries.
  - `⏹️ Stop {strategy_name}` in Active Strategies Overview:
    - Stops strategy from overview and logs event.
  - `🗑️ Clear Log`:
    - Clears `auto_execution_logs`.

- **Order Management**:
  - `🔄 Refresh Now`:
    - Reruns app to refresh order status.
  - `🗑️ Cancel Selected`:
    - Marks selected orders as cancelled in session state.
  - `💾 Update Order`:
    - Uses `ExecutionAgent.modify_order` if available, else updates order in session state.
  - Batch operations:
    - `🗑️ Cancel All Selected` (stub: currently informational, reruns after message).
    - `📊 Export Selected` and nested `📥 Download CSV`:
      - Exports selected orders to CSV.

- **Order History & Notifications**:
  - `📥 Export History to CSV`:
    - Exposes order history as downloadable CSV.
  - `🗑️ Clear Notifications`:
    - Clears `fill_notifications`.

- **Analytics**:
  - `📥 Export Analytics to CSV`:
    - Exports raw analytics data to CSV.

### Session State Keys
- **Core execution state**:
  - `execution_mode` (`"paper"` or `"live"`).
  - `broker_adapter`, `execution_agent`, `portfolio_manager`.
  - `active_orders`, `order_history`.
  - `trade_page_loaded`.
  - `_suggested_quantity`, `quick_symbol`, `quick_side`, `quick_order_type`, `quick_quantity`, `quick_limit_price`, `quick_tif`, etc.
- **Advanced execution**:
  - `execution_engine`.
  - Various algorithm parameter keys (e.g. `execution_algo`, `twap_duration`, `vwap_participation`, etc.).
- **Automated execution**:
  - `auto_execution_active` (per strategy bool).
  - `auto_execution_logs` (list of log dicts).
  - `auto_execution_configs` (per strategy config dict).
  - `emergency_stop`.
  - `strategy_registry`, `custom_strategy_handler`.
  - Selection and UI keys (e.g. `auto_exec_strategy_select`, `auto_exec_symbols_{strategy}`, safety sliders/inputs).
- **Order management**:
  - `order_refresh_interval`.
  - `selected_orders_for_cancel`.
  - `fill_notifications`.
  - `order_selection`, `order_multiselect`.
  - `previous_active_orders`.
- **Analytics**:
  - `execution_analytics_data`.

### External Integrations (Imports under `trading/`, `agents/`, `components/`, `system/`)
- **Execution & brokers**:
  - `execution.broker_adapter.BrokerAdapter`, `BrokerType` (with fallback import; may be `None`).
  - `execution.execution_agent.ExecutionAgent`, `ExecutionMode`, `OrderSide`, `OrderType` (with fallback import; may be `None`).
  - `trading.execution.execution_engine.ExecutionEngine` as `AdvancedExecutionEngine`.
- **Portfolio & risk**:
  - `trading.portfolio.portfolio_manager.PortfolioManager` (imported, not used in this file).
  - `trading.agents.execution_risk_control_agent.ExecutionRiskControlAgent` (imported, not used in this file).
- **Strategies & registries**:
  - `trading.strategies.bollinger_strategy.BollingerStrategy`, `BollingerConfig`.
  - `trading.strategies.macd_strategy.MACDStrategy`, `MACDConfig`.
  - `trading.strategies.rsi_strategy.RSIStrategy`.
  - `trading.strategies.sma_strategy.SMAStrategy`, `SMAConfig`.
  - `trading.strategies.custom_strategy_handler.CustomStrategyHandler`, `get_custom_strategy_handler`.
  - `trading.strategies.registry.StrategyRegistry`, `get_strategy_registry`.
- **Memory & notifications**:
  - `trading.memory.get_memory_store`, `trading.memory.memory_store.MemoryType`.
  - `system.infra.agents.notifications.notification_service.NotificationChannel`, `NotificationType`, `NotificationPriority`.
- **UI & infra components**:
  - `ui.page_assistant.render_page_assistant("Trade Execution")` (called in try/except at end).
  - `utils.websocket_client.WebSocketClient` (referenced when sending trade updates if `ws_client` in session).
- **Data and plotting**:
  - `yfinance` for pricing.
  - `pandas`, `numpy`, `plotly.graph_objects`.

### Stubs & Incomplete Features
- **Conditional, OCO, and Multi-Leg advanced orders**:
  - These orders are **stored in session state only** and not actually wired to the `ExecutionAgent` or broker; there is no background process shown here that monitors conditions or manages OCO linkage. They are functional at the UI/data level but not connected to back-end execution in this file.
- **Bracket and Trailing orders**:
  - Similarly, bracket and trailing configurations create local order records and show confirmations but do **not** submit to `ExecutionAgent` or a broker from this file.
- **Automated execution loop**:
  - Strategy selection, configuration, and start/stop flags are fully implemented in session state, but there is no loop/thread/async task in this file that periodically polls strategies and submits orders; actual signal-to-order logic is implemented elsewhere or not yet wired.
- **Batch operations for cancellation/export**:
  - `🗑️ Cancel All Selected` in the Batch Operations section only logs that cancellation will occur and reruns; actual per-order cancellation logic for this button is not implemented (though single-order cancellation is implemented above).
- **PortfolioManager and ExecutionRiskControlAgent**:
  - These are imported but not used in any function calls in this file; integration with actual portfolio and risk engine is not shown here.

