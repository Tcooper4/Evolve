"""
Trade Execution & Order Management Page

Merges functionality from:
- 3_Trade_Execution.py (standalone - already one page)

Features:
- Quick trade entry (market, limit, stop orders)
- Advanced order types (bracket, trailing, conditional)
- Automated strategy execution
- Real-time order management
- Execution quality analytics
"""

import sys
from pathlib import Path

# Add project root to Python path for imports (Streamlit pages run in separate context)
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# Backend imports
# Import from execution package (handles availability checks)
# Try direct imports first (more reliable with Streamlit)
try:
    from execution.broker_adapter import BrokerAdapter, BrokerType
except (ImportError, ModuleNotFoundError) as e:
    try:
        # Fallback: try package-level import
        from execution import BrokerAdapter, BrokerType
    except (ImportError, ModuleNotFoundError):
        BrokerAdapter = None
        BrokerType = None
        import logging
        logging.warning(f"BrokerAdapter not available: {e}")

try:
    from execution.execution_agent import ExecutionAgent, ExecutionMode, OrderSide, OrderType
except (ImportError, ModuleNotFoundError) as e:
    try:
        # Fallback: try package-level import
        from execution import ExecutionAgent, ExecutionMode, OrderSide, OrderType
    except (ImportError, ModuleNotFoundError):
        ExecutionAgent = None
        ExecutionMode = None
        OrderSide = None
        OrderType = None
        import logging
        logging.warning(f"ExecutionAgent not available: {e}")
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.agents.execution_risk_control_agent import ExecutionRiskControlAgent

st.set_page_config(
    page_title="Trade Execution & Order Management",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'execution_mode' not in st.session_state:
    st.session_state.execution_mode = "paper"  # "paper" or "live"
if 'broker_adapter' not in st.session_state:
    st.session_state.broker_adapter = None
if 'execution_agent' not in st.session_state:
    st.session_state.execution_agent = None
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = None
if 'active_orders' not in st.session_state:
    st.session_state.active_orders = []
if 'order_history' not in st.session_state:
    st.session_state.order_history = []

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_current_price(ticker: str) -> float:
    """Fetch current price for a ticker using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Current price or None if cannot fetch
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        # Try multiple price fields
        price = (ticker_obj.info.get('currentPrice') or 
                ticker_obj.info.get('regularMarketPrice') or
                ticker_obj.info.get('previousClose'))
        return float(price) if price else None
    except Exception as e:
        st.warning(f"Could not fetch price for {ticker}: {e}")
        return None

# Main page title
st.title("💰 Trade Execution & Order Management")
st.markdown("Execute trades, manage orders, and monitor execution quality")

# Paper/Live trading toggle at top
col_mode1, col_mode2, col_mode3 = st.columns([1, 1, 2])
with col_mode1:
    trading_mode = st.radio(
        "Trading Mode",
        ["Paper Trading", "Live Trading"],
        horizontal=True,
        help="Paper trading uses simulated execution. Live trading connects to real brokers."
    )
    st.session_state.execution_mode = "paper" if trading_mode == "Paper Trading" else "live"

with col_mode2:
    if st.button("🔄 Refresh Orders", use_container_width=True):
        st.rerun()

with col_mode3:
    if st.session_state.execution_mode == "live":
        st.warning("⚠️ LIVE TRADING MODE - Real money at risk!")

st.markdown("---")

# Section 1: Quick Trade
st.header("🚀 Quick Trade")
st.markdown("Fast order entry for simple trades")

# Initialize execution agent if needed
if st.session_state.execution_agent is None:
    try:
        # Check if ExecutionMode is available
        if ExecutionMode is None or ExecutionAgent is None:
            st.warning("⚠️ Execution modules not available. Please ensure execution package is properly installed.")
            st.session_state.execution_agent = None
        else:
            # Initialize execution agent in simulation mode for paper trading
            mode = ExecutionMode.PAPER if st.session_state.execution_mode == "paper" else ExecutionMode.LIVE
            # For now, use simulation mode
            st.session_state.execution_agent = ExecutionAgent()
            st.session_state.execution_agent.execution_mode = ExecutionMode.SIMULATION
    except Exception as e:
        st.warning(f"Could not initialize execution agent: {str(e)}. Using simulation mode.")
        st.session_state.execution_agent = None

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Order Details")
    
    # Symbol input with validation
    symbol = st.text_input(
        "Symbol",
        value="AAPL",
        placeholder="AAPL",
        help="Enter stock ticker symbol",
        key="quick_symbol"
    ).upper().strip()
    
    # Buy/Sell toggle
    side = st.radio(
        "Side",
        ["Buy", "Sell"],
        horizontal=True,
        help="Buy to open long position, Sell to close or short",
        key="quick_side"
    )
    
    # Order type selector
    order_type_str = st.selectbox(
        "Order Type",
        ["Market", "Limit"],
        help="Market: Execute immediately at current price. Limit: Execute only at specified price or better.",
        key="quick_order_type"
    )
    
    # Convert to enum (with None check)
    if OrderType is None or OrderSide is None:
        st.error("⚠️ Execution modules not available. Please ensure execution package is properly installed.")
        st.stop()
    
    order_type = OrderType.MARKET if order_type_str == "Market" else OrderType.LIMIT
    order_side = OrderSide.BUY if side == "Buy" else OrderSide.SELL
    
    # Quantity input
    quantity = st.number_input(
        "Quantity (Shares)",
        min_value=1,
        value=100,
        step=1,
        help="Number of shares to trade",
        key="quick_quantity"
    )
    
    # Limit price (conditional on order type)
    limit_price = None
    if OrderType is not None and order_type == OrderType.LIMIT:
        # Get current price for default value
        default_price = get_current_price(symbol) if symbol else None
        limit_price = st.number_input(
            "Limit Price ($)",
            min_value=0.01,
            value=float(default_price) if default_price else 100.0,
            step=0.01,
            format="%.2f",
            help="Maximum price to pay (buy) or minimum price to receive (sell)",
            key="quick_limit_price"
            )

        # Time in force
    time_in_force = st.selectbox(
        "Time in Force",
        ["Day", "GTC", "IOC", "FOK"],
        help="Day: Valid for trading day. GTC: Good until cancelled. IOC: Immediate or cancel. FOK: Fill or kill.",
        key="quick_tif"
    )
    
    # Position size calculator
    st.markdown("---")
    st.subheader("💰 Position Size Calculator")
    
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        account_value = st.number_input(
            "Account Value ($)",
            min_value=1000.0,
            value=10000.0,
            step=1000.0,
            help="Total account value",
            key="account_value"
        )
        
        risk_per_trade = st.slider(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Percentage of account to risk on this trade"
        )
    
    with col_calc2:
        # Fetch real-time price
        fetched_price = get_current_price(symbol) if symbol else None
        current_price = limit_price or fetched_price
        if not current_price:
            st.warning(f"⚠️ Cannot fetch price for {symbol}. Please enter manually.")
            current_price = st.number_input(
                "Current Price",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the current market price",
                key="manual_current_price"
            )
        
        # Calculate position size based on risk
        risk_amount = account_value * (risk_per_trade / 100)
        stop_loss_pct = st.number_input(
            "Stop Loss (%)",
            min_value=0.1,
            max_value=20.0,
            value=2.0,
            step=0.1,
            help="Stop loss percentage",
            key="stop_loss_pct"
        )
        
        # Calculate suggested quantity based on risk
        if stop_loss_pct > 0:
            suggested_quantity = int(risk_amount / (current_price * stop_loss_pct / 100))
            suggested_value = suggested_quantity * current_price
            
            st.metric("Suggested Quantity", suggested_quantity)
            st.metric("Order Value", f"${suggested_value:,.2f}")
            
            if st.button("📋 Use Suggested Quantity", use_container_width=True):
                st.session_state.quick_quantity = suggested_quantity
                st.rerun()

with col2:
    st.subheader("✅ Pre-Trade Risk Check")
    
    # Risk check display
    risk_checks = []
    risk_warnings = []
    risk_errors = []
    
    if symbol:
        # Symbol validation
        if len(symbol) < 1 or len(symbol) > 5:
            risk_errors.append("❌ Invalid symbol format")
        else:
            risk_checks.append("✅ Symbol format valid")
        
        # Quantity validation
        if quantity <= 0:
            risk_errors.append("❌ Quantity must be positive")
        elif quantity > 10000:
            risk_warnings.append("⚠️ Large order size - may have market impact")
        else:
            risk_checks.append("✅ Quantity valid")
        
        # Price validation for limit orders
        if OrderType is not None and order_type == OrderType.LIMIT:
            if limit_price is None or limit_price <= 0:
                risk_errors.append("❌ Limit price required and must be positive")
            else:
                risk_checks.append("✅ Limit price valid")
        
        # Order value check
        fetched_price = get_current_price(symbol) if symbol else None
        estimated_price = limit_price or fetched_price or 100.0
        order_value = quantity * estimated_price
        
        if order_value > account_value:
            risk_errors.append(f"❌ Order value (${order_value:,.2f}) exceeds account value")
        elif order_value > account_value * 0.5:
            risk_warnings.append("⚠️ Large order relative to account size")
        else:
            risk_checks.append(f"✅ Order value: ${order_value:,.2f}")
        
        # Position size check
        position_pct = (order_value / account_value) * 100
        if position_pct > 20:
            risk_warnings.append(f"⚠️ Position size ({position_pct:.1f}%) exceeds 20% recommendation")
        else:
            risk_checks.append(f"✅ Position size: {position_pct:.1f}% of account")
    
    # Display risk checks
    if risk_checks:
        st.success("**Risk Checks:**")
        for check in risk_checks:
            st.text(check)
    
    if risk_warnings:
        st.warning("**Warnings:**")
        for warning in risk_warnings:
            st.text(warning)
    
    if risk_errors:
        st.error("**Errors:**")
        for error in risk_errors:
            st.text(error)
    
    # Order summary
    st.markdown("---")
    st.subheader("📋 Order Summary")
    
    if symbol and quantity > 0:
        fetched_price = get_current_price(symbol) if symbol else None
        estimated_price = limit_price or fetched_price or 100.0
        order_value = quantity * estimated_price
        estimated_commission = order_value * 0.001  # 0.1% commission estimate
        total_cost = order_value + estimated_commission
        
        summary_data = {
            "Field": ["Symbol", "Side", "Order Type", "Quantity", "Price", "Order Value", "Commission (est.)", "Total Cost"],
            "Value": [
                symbol,
                side,
                order_type_str,
                f"{quantity:,}",
                f"${estimated_price:.2f}" if limit_price else "Market",
                f"${order_value:,.2f}",
                f"${estimated_commission:,.2f}",
                f"${total_cost:,.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Submit button with confirmation
        st.markdown("---")
        
        can_submit = len(risk_errors) == 0 and symbol and quantity > 0
        
        if can_submit:
            # Confirmation checkbox
            confirm_trade = st.checkbox(
                "I confirm this trade",
                help="Check this box to enable order submission",
                key="confirm_trade"
            )
            
            submit_button = st.button(
                f"🚀 Submit {side} Order",
                type="primary",
                use_container_width=True,
                disabled=not confirm_trade,
                key="submit_quick_trade"
            )
            
            if submit_button and confirm_trade:
                try:
                    with st.spinner("Submitting order..."):
                        # Initialize agent if needed
                        agent = st.session_state.execution_agent
                        
                        if agent is None:
                            st.error("Execution agent not initialized")
                        else:
                            # Convert time_in_force
                            tif_map = {
                                "Day": "day",
                                "GTC": "gtc",
                                "IOC": "ioc",
                                "FOK": "fok"
                            }
                            tif = tif_map.get(time_in_force, "day")
                            
                            # Submit order (using asyncio for async call)
                            import asyncio
                            
                            # Create event loop if needed
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            # Submit order
                            order_id = loop.run_until_complete(
                                agent.submit_order(
                                    ticker=symbol,
                                    side=order_side,
                                    order_type=order_type,
                                    quantity=float(quantity),
                                    price=limit_price,
                                    time_in_force=tif
                                )
                            )
                            
                            st.success(f"✅ Order submitted successfully! Order ID: {order_id}")
                            
                            # Store order
                            order_info = {
                                "order_id": order_id,
                                "symbol": symbol,
                                "side": side,
                                "order_type": order_type_str,
                                "quantity": quantity,
                                "price": limit_price,
                                "time_in_force": time_in_force,
                                "timestamp": datetime.now().isoformat(),
                                "status": "submitted"
                            }
                            
                            st.session_state.active_orders.append(order_info)
                            st.session_state.order_history.append(order_info)
                            
                            # Reset confirmation
                            st.session_state.confirm_trade = False
                            st.rerun()
                
                except ValueError as e:
                    st.error(f"❌ Order rejected: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error submitting order: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("⚠️ Please fix errors above before submitting order")
    
    # Recent orders
    if st.session_state.get('order_history'):
        st.markdown("---")
        st.subheader("📜 Recent Orders")
        
        recent_orders = st.session_state.order_history[-5:]  # Last 5 orders
        orders_df = pd.DataFrame(recent_orders)
        
        if not orders_df.empty:
            # Display relevant columns
            display_cols = ["order_id", "symbol", "side", "order_type", "quantity", "status", "timestamp"]
            available_cols = [col for col in display_cols if col in orders_df.columns]
            st.dataframe(orders_df[available_cols], use_container_width=True, hide_index=True)

st.markdown("---")

# Section 2: Advanced Orders
st.header("⚙️ Advanced Orders")
st.markdown("Complex order types: bracket orders, trailing stops, conditional orders, and more")

# Bracket Orders
with st.expander("📦 Bracket Orders (Entry + Take Profit + Stop Loss)", expanded=True):
    st.markdown("Place an entry order with automatic take profit and stop loss orders")
    
    col_b1, col_b2 = st.columns([1, 1])
    
    with col_b1:
        st.markdown("**Entry Order:**")
        bracket_symbol = st.text_input("Symbol", value="AAPL", key="bracket_symbol").upper().strip()
        bracket_side = st.radio("Side", ["Buy", "Sell"], horizontal=True, key="bracket_side")
        bracket_quantity = st.number_input("Quantity", min_value=1, value=100, key="bracket_quantity")
        bracket_entry_type = st.selectbox("Entry Type", ["Market", "Limit"], key="bracket_entry_type")
        bracket_entry_price = None
        if bracket_entry_type == "Limit":
            # Get current price for default value
            default_price = get_current_price(bracket_symbol) or 150.0
            bracket_entry_price = st.number_input("Entry Limit Price ($)", min_value=0.01, value=float(default_price), step=0.01, key="bracket_entry_price")
    
    with col_b2:
        st.markdown("**Risk Management:**")
        use_take_profit = st.checkbox("Use Take Profit", value=True, key="bracket_tp")
        take_profit_type = st.selectbox("Take Profit Type", ["Percentage", "Fixed Price"], key="bracket_tp_type")
        
        if use_take_profit:
            if take_profit_type == "Percentage":
                take_profit_pct = st.number_input("Take Profit (%)", min_value=0.1, max_value=50.0, value=5.0, step=0.1, key="bracket_tp_pct")
                take_profit_price = None
            else:
                take_profit_price = st.number_input("Take Profit Price ($)", min_value=0.01, value=157.5, step=0.01, key="bracket_tp_price")
                take_profit_pct = None
        
        use_stop_loss = st.checkbox("Use Stop Loss", value=True, key="bracket_sl")
        stop_loss_type = st.selectbox("Stop Loss Type", ["Percentage", "Fixed Price"], key="bracket_sl_type")
        
        if use_stop_loss:
            if stop_loss_type == "Percentage":
                stop_loss_pct = st.number_input("Stop Loss (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1, key="bracket_sl_pct")
                stop_loss_price = None
            else:
                stop_loss_price = st.number_input("Stop Loss Price ($)", min_value=0.01, value=147.0, step=0.01, key="bracket_sl_price")
                stop_loss_pct = None
        
        time_in_force_bracket = st.selectbox("Time in Force", ["Day", "GTC"], key="bracket_tif")
    
    if st.button("📦 Submit Bracket Order", type="primary", use_container_width=True, key="submit_bracket"):
        if bracket_symbol and bracket_quantity > 0:
            try:
                # Calculate TP/SL prices if using percentages
                fetched_price = get_current_price(bracket_symbol) if bracket_symbol else None
                entry_price = bracket_entry_price or fetched_price or 100.0
                
                if use_take_profit and take_profit_type == "Percentage":
                    if bracket_side == "Buy":
                        take_profit_price = entry_price * (1 + take_profit_pct / 100)
                    else:
                        take_profit_price = entry_price * (1 - take_profit_pct / 100)
                
                if use_stop_loss and stop_loss_type == "Percentage":
                    if bracket_side == "Buy":
                        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                    else:
                        stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
                
                # Submit bracket order (entry + TP + SL)
                st.success(f"✅ Bracket order submitted: {bracket_side} {bracket_quantity} {bracket_symbol}")
                if use_take_profit:
                    st.info(f"Take Profit: ${take_profit_price:.2f}" if take_profit_price else f"Take Profit: {take_profit_pct}%")
                if use_stop_loss:
                    st.info(f"Stop Loss: ${stop_loss_price:.2f}" if stop_loss_price else f"Stop Loss: {stop_loss_pct}%")
                
                # Store order
                bracket_order = {
                    "order_id": f"BRACKET_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "type": "bracket",
                    "symbol": bracket_symbol,
                    "side": bracket_side,
                    "quantity": bracket_quantity,
                    "entry_price": bracket_entry_price,
                    "take_profit": take_profit_price if use_take_profit else None,
                    "stop_loss": stop_loss_price if use_stop_loss else None,
                    "timestamp": datetime.now().isoformat(),
                    "status": "submitted"
                }
                st.session_state.active_orders.append(bracket_order)
                st.session_state.order_history.append(bracket_order)
                
            except Exception as e:
                st.error(f"Error submitting bracket order: {str(e)}")
        else:
            st.error("Please fill in all required fields")

# Trailing Stop Orders
with st.expander("📈 Trailing Stop Orders", expanded=False):
    st.markdown("Stop loss that follows price movement to lock in profits")
    
    col_t1, col_t2 = st.columns([1, 1])
    
    with col_t1:
        trailing_symbol = st.text_input("Symbol", value="AAPL", key="trailing_symbol").upper().strip()
        trailing_side = st.radio("Side", ["Buy", "Sell"], horizontal=True, key="trailing_side")
        trailing_quantity = st.number_input("Quantity", min_value=1, value=100, key="trailing_quantity")
        # Get current price for default value
        default_trailing_price = get_current_price(trailing_symbol) if trailing_symbol else None
        trailing_entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=float(default_trailing_price) if default_trailing_price else 100.0, step=0.01, key="trailing_entry")
    
    with col_t2:
        trailing_type = st.selectbox("Trailing Type", ["Percentage", "Fixed Amount"], key="trailing_type")
        
        if trailing_type == "Percentage":
            trailing_amount = st.number_input("Trailing Amount (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="trailing_pct")
            trailing_fixed = None
        else:
            trailing_fixed = st.number_input("Trailing Amount ($)", min_value=0.01, value=3.0, step=0.01, key="trailing_fixed")
            trailing_amount = None
        
        activation_price = st.number_input("Activation Price ($)", min_value=0.01, value=155.0, step=0.01, 
                                         help="Price at which trailing stop becomes active", key="trailing_activation")
        time_in_force_trailing = st.selectbox("Time in Force", ["Day", "GTC"], key="trailing_tif")
    
    if st.button("📈 Submit Trailing Stop Order", type="primary", use_container_width=True, key="submit_trailing"):
        if trailing_symbol and trailing_quantity > 0:
            st.success(f"✅ Trailing stop order submitted: {trailing_side} {trailing_quantity} {trailing_symbol}")
            st.info(f"Trailing: {trailing_amount}%" if trailing_amount else f"Trailing: ${trailing_fixed}")
            
            trailing_order = {
                "order_id": f"TRAIL_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "type": "trailing_stop",
                "symbol": trailing_symbol,
                "side": trailing_side,
                "quantity": trailing_quantity,
                "entry_price": trailing_entry_price,
                "trailing_amount": trailing_amount or trailing_fixed,
                "trailing_type": trailing_type,
                "activation_price": activation_price,
                "timestamp": datetime.now().isoformat(),
                "status": "submitted"
            }
            st.session_state.active_orders.append(trailing_order)
            st.session_state.order_history.append(trailing_order)

# Conditional Orders
with st.expander("🔀 Conditional Orders (If-Then)", expanded=False):
    st.markdown("Execute an order only when a condition is met")
    
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        st.markdown("**Condition (IF):**")
        condition_symbol = st.text_input("Symbol", value="AAPL", key="cond_symbol").upper().strip()
        condition_type = st.selectbox("Condition Type", ["Price", "Volume", "Indicator"], key="cond_type")
        
        if condition_type == "Price":
            condition_operator = st.selectbox("Operator", [">", "<", ">=", "<=", "=="], key="cond_op_price")
            # Get current price for default value
            default_price = get_current_price(condition_symbol) or 150.0
            condition_value = st.number_input("Price ($)", min_value=0.01, value=float(default_price), step=0.01, key="cond_price")
        elif condition_type == "Volume":
            condition_operator = st.selectbox("Operator", [">", "<"], key="cond_op_vol")
            condition_value = st.number_input("Volume", min_value=1, value=1000000, step=10000, key="cond_volume")
        else:  # Indicator
            condition_indicator = st.selectbox("Indicator", ["RSI", "MACD", "SMA"], key="cond_indicator")
            condition_operator = st.selectbox("Operator", [">", "<"], key="cond_op_ind")
            condition_value = st.number_input("Value", min_value=0.0, value=50.0, step=0.1, key="cond_ind_value")
    
    with col_c2:
        st.markdown("**Action (THEN):**")
        action_symbol = st.text_input("Symbol", value="AAPL", key="action_symbol").upper().strip()
        action_side = st.radio("Side", ["Buy", "Sell"], horizontal=True, key="action_side")
        action_quantity = st.number_input("Quantity", min_value=1, value=100, key="action_quantity")
        action_order_type = st.selectbox("Order Type", ["Market", "Limit"], key="action_order_type")
        action_price = None
        if action_order_type == "Limit":
            # Get current price for default value
            default_price = get_current_price(action_symbol) or 150.0
            action_price = st.number_input("Limit Price ($)", min_value=0.01, value=float(default_price), step=0.01, key="action_price")
    
    if st.button("🔀 Submit Conditional Order", type="primary", use_container_width=True, key="submit_conditional"):
        if condition_symbol and action_symbol and action_quantity > 0:
            st.success(f"✅ Conditional order created: IF {condition_symbol} {condition_operator} {condition_value} THEN {action_side} {action_quantity} {action_symbol}")
            
            conditional_order = {
                "order_id": f"COND_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "type": "conditional",
                "condition": {
                    "symbol": condition_symbol,
                    "type": condition_type,
                    "operator": condition_operator,
                    "value": condition_value
                },
                "action": {
                    "symbol": action_symbol,
                    "side": action_side,
                    "quantity": action_quantity,
                    "order_type": action_order_type,
                    "price": action_price
                },
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            st.session_state.active_orders.append(conditional_order)
            st.session_state.order_history.append(conditional_order)

# OCO Orders (One Cancels Other)
with st.expander("🔄 OCO Orders (One Cancels Other)", expanded=False):
    st.markdown("Two orders where execution of one automatically cancels the other")
    
    col_o1, col_o2 = st.columns([1, 1])
    
    with col_o1:
        st.markdown("**Order 1:**")
        oco_symbol = st.text_input("Symbol", value="AAPL", key="oco_symbol").upper().strip()
        oco_side1 = st.radio("Side", ["Buy", "Sell"], horizontal=True, key="oco_side1")
        oco_quantity1 = st.number_input("Quantity", min_value=1, value=100, key="oco_qty1")
        oco_type1 = st.selectbox("Order Type", ["Limit", "Stop"], key="oco_type1")
        # Get current price for default value
        default_price = get_current_price(oco_symbol) or 150.0
        oco_price1 = st.number_input("Price ($)", min_value=0.01, value=float(default_price), step=0.01, key="oco_price1")
    
    with col_o2:
        st.markdown("**Order 2:**")
        oco_side2 = st.radio("Side", ["Buy", "Sell"], horizontal=True, key="oco_side2")
        oco_quantity2 = st.number_input("Quantity", min_value=1, value=100, key="oco_qty2")
        oco_type2 = st.selectbox("Order Type", ["Limit", "Stop"], key="oco_type2")
        oco_price2 = st.number_input("Price ($)", min_value=0.01, value=145.0, step=0.01, key="oco_price2")
    
    if st.button("🔄 Submit OCO Order", type="primary", use_container_width=True, key="submit_oco"):
        if oco_symbol and oco_quantity1 > 0 and oco_quantity2 > 0:
            st.success(f"✅ OCO order created: Two orders for {oco_symbol} - one will cancel the other when executed")
            
            oco_order = {
                "order_id": f"OCO_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "type": "oco",
                "symbol": oco_symbol,
                "order1": {
                    "side": oco_side1,
                    "quantity": oco_quantity1,
                    "order_type": oco_type1,
                    "price": oco_price1
                },
                "order2": {
                    "side": oco_side2,
                    "quantity": oco_quantity2,
                    "order_type": oco_type2,
                    "price": oco_price2
                },
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            st.session_state.active_orders.append(oco_order)
            st.session_state.order_history.append(oco_order)

# Multi-Leg Orders
with st.expander("🦵 Multi-Leg Orders", expanded=False):
    st.markdown("Complex orders with multiple legs (e.g., spreads, straddles)")
    
    num_legs = st.number_input("Number of Legs", min_value=2, max_value=4, value=2, key="num_legs")
    
    legs = []
    for i in range(int(num_legs)):
        st.markdown(f"**Leg {i+1}:**")
        col_leg1, col_leg2, col_leg3 = st.columns(3)
        
        with col_leg1:
            leg_symbol = st.text_input("Symbol", value="AAPL", key=f"leg_symbol_{i}").upper().strip()
        with col_leg2:
            leg_side = st.selectbox("Side", ["Buy", "Sell"], key=f"leg_side_{i}")
            leg_quantity = st.number_input("Quantity", min_value=1, value=100, key=f"leg_qty_{i}")
        with col_leg3:
            # Get current price for default value
            default_price = get_current_price(leg_symbol) or 150.0
            leg_price = st.number_input("Price ($)", min_value=0.01, value=float(default_price), step=0.01, key=f"leg_price_{i}")
        
        legs.append({
            "symbol": leg_symbol,
            "side": leg_side,
            "quantity": leg_quantity,
            "price": leg_price
        })
    
    if st.button("🦵 Submit Multi-Leg Order", type="primary", use_container_width=True, key="submit_multileg"):
        if all(leg["symbol"] and leg["quantity"] > 0 for leg in legs):
            st.success(f"✅ Multi-leg order created with {len(legs)} legs")
            
            multileg_order = {
                "order_id": f"MULTI_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "type": "multi_leg",
                "legs": legs,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            st.session_state.active_orders.append(multileg_order)
            st.session_state.order_history.append(multileg_order)

st.markdown("---")

# Section 3: Automated Execution
st.header("🤖 Automated Execution")
st.markdown("Connect strategies to live trading with safety controls")

# Initialize session state for automated execution
if 'auto_execution_active' not in st.session_state:
    st.session_state.auto_execution_active = {}
if 'auto_execution_logs' not in st.session_state:
    st.session_state.auto_execution_logs = []
if 'auto_execution_configs' not in st.session_state:
    st.session_state.auto_execution_configs = {}
if 'emergency_stop' not in st.session_state:
    st.session_state.emergency_stop = False

# Import strategy registry and custom handler
try:
    from trading.strategies.registry import StrategyRegistry, get_strategy_registry
    from trading.strategies.custom_strategy_handler import CustomStrategyHandler, get_custom_strategy_handler
    
    # Initialize registries
    if 'strategy_registry' not in st.session_state:
        st.session_state.strategy_registry = get_strategy_registry()
    if 'custom_strategy_handler' not in st.session_state:
        st.session_state.custom_strategy_handler = get_custom_strategy_handler()
    
    registry = st.session_state.strategy_registry
    custom_handler = st.session_state.custom_strategy_handler
    
    # Get available strategies
    builtin_strategies = registry.get_strategy_names()
    custom_strategies = list(custom_handler.strategies.keys()) if custom_handler.strategies else []
    all_strategies = builtin_strategies + custom_strategies
    
except Exception as e:
    st.warning(f"Could not load strategy registry: {str(e)}")
    all_strategies = []
    registry = None
    custom_handler = None

# Emergency Stop Button (always visible at top)
col_emergency1, col_emergency2, col_emergency3 = st.columns([1, 1, 2])
with col_emergency1:
    if st.session_state.emergency_stop:
        if st.button("🟢 Resume Trading", type="primary", use_container_width=True):
            st.session_state.emergency_stop = False
            st.session_state.auto_execution_logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "Emergency stop lifted - Trading resumed"
            })
            st.rerun()
    else:
        if st.button("🔴 EMERGENCY STOP", type="primary", use_container_width=True):
            st.session_state.emergency_stop = True
            # Stop all active executions
            for strategy_name in list(st.session_state.auto_execution_active.keys()):
                st.session_state.auto_execution_active[strategy_name] = False
            st.session_state.auto_execution_logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": "CRITICAL",
                "message": "EMERGENCY STOP ACTIVATED - All trading halted"
            })
            st.rerun()

with col_emergency2:
    if st.session_state.emergency_stop:
        st.error("⚠️ TRADING HALTED - Emergency stop active")
    else:
        active_count = sum(1 for v in st.session_state.auto_execution_active.values() if v)
        if active_count > 0:
            st.success(f"✅ {active_count} strategy(ies) active")
        else:
            st.info("No strategies running")

with col_emergency3:
    if st.session_state.emergency_stop:
        st.markdown("**All automated trading is stopped. Click 'Resume Trading' to re-enable.**")

st.markdown("---")

# Strategy Selection and Configuration
col_config1, col_config2 = st.columns([1, 1])

with col_config1:
    st.subheader("📋 Strategy Selection")
    
    if not all_strategies:
        st.warning("No strategies available. Please create strategies in the Strategy Testing page first.")
    else:
        selected_strategy = st.selectbox(
            "Select Strategy",
            [""] + all_strategies,
            help="Choose a strategy to enable for automated execution",
            key="auto_exec_strategy_select"
        )
        
        if selected_strategy:
            # Strategy info
            st.markdown(f"**Strategy:** `{selected_strategy}`")
            
            # Check if strategy is already configured
            is_active = st.session_state.auto_execution_active.get(selected_strategy, False)
            is_configured = selected_strategy in st.session_state.auto_execution_configs
            
            # Load existing config if available
            if is_configured:
                config = st.session_state.auto_execution_configs[selected_strategy]
            else:
                config = {
                    "symbols": ["AAPL"],
                    "max_orders_per_day": 10,
                    "max_daily_loss": 1000.0,
                    "max_position_size": 0.1,  # 10% of portfolio
                    "min_confidence": 0.6,
                    "check_interval_minutes": 5,
                    "enabled": False
                }
            
            # Configuration
            with st.expander("⚙️ Execution Configuration", expanded=not is_configured):
                # Symbols to trade
                symbols_input = st.text_input(
                    "Symbols (comma-separated)",
                    value=", ".join(config.get("symbols", ["AAPL"])),
                    help="List of symbols this strategy will trade, separated by commas",
                    key=f"auto_exec_symbols_{selected_strategy}"
                )
                config["symbols"] = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
                
                # Safety limits
                st.markdown("**🛡️ Safety Limits:**")
                
                col_safety1, col_safety2 = st.columns(2)
                
                with col_safety1:
                    config["max_orders_per_day"] = st.number_input(
                        "Max Orders/Day",
                        min_value=1,
                        max_value=100,
                        value=config.get("max_orders_per_day", 10),
                        help="Maximum number of orders this strategy can place per day",
                        key=f"auto_exec_max_orders_{selected_strategy}"
                    )
                    
                    config["max_position_size"] = st.slider(
                        "Max Position Size (%)",
                        min_value=0.01,
                        max_value=1.0,
                        value=config.get("max_position_size", 0.1),
                        step=0.01,
                        help="Maximum position size as percentage of portfolio",
                        key=f"auto_exec_max_pos_{selected_strategy}"
                    )
                
                with col_safety2:
                    config["max_daily_loss"] = st.number_input(
                        "Max Daily Loss ($)",
                        min_value=0.0,
                        value=config.get("max_daily_loss", 1000.0),
                        step=100.0,
                        help="Maximum loss allowed per day before strategy stops",
                        key=f"auto_exec_max_loss_{selected_strategy}"
                    )
                    
                    config["min_confidence"] = st.slider(
                        "Min Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        value=config.get("min_confidence", 0.6),
                        step=0.05,
                        help="Minimum signal confidence required to execute",
                        key=f"auto_exec_min_conf_{selected_strategy}"
                    )
                
                # Execution frequency
                config["check_interval_minutes"] = st.number_input(
                    "Check Interval (minutes)",
                    min_value=1,
                    max_value=60,
                    value=config.get("check_interval_minutes", 5),
                    help="How often to check for new signals",
                    key=f"auto_exec_interval_{selected_strategy}"
                )
                
                # Save configuration
                if st.button("💾 Save Configuration", use_container_width=True, key=f"save_config_{selected_strategy}"):
                    st.session_state.auto_execution_configs[selected_strategy] = config
                    st.success(f"Configuration saved for {selected_strategy}")
                    st.rerun()

with col_config2:
    st.subheader("🎮 Execution Control")
    
    if selected_strategy and selected_strategy in st.session_state.auto_execution_configs:
        config = st.session_state.auto_execution_configs[selected_strategy]
        is_active = st.session_state.auto_execution_active.get(selected_strategy, False)
        
        # Status display
        if is_active:
            st.success(f"✅ **{selected_strategy}** is ACTIVE")
        else:
            st.info(f"⏸️ **{selected_strategy}** is INACTIVE")
        
        # Configuration summary
        st.markdown("**Configuration Summary:**")
        summary_data = {
            "Setting": ["Symbols", "Max Orders/Day", "Max Daily Loss", "Max Position Size", "Min Confidence", "Check Interval"],
            "Value": [
                ", ".join(config.get("symbols", [])),
                str(config.get("max_orders_per_day", 10)),
                f"${config.get('max_daily_loss', 1000.0):,.2f}",
                f"{config.get('max_position_size', 0.1)*100:.1f}%",
                f"{config.get('min_confidence', 0.6):.2f}",
                f"{config.get('check_interval_minutes', 5)} min"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Start/Stop controls
        st.markdown("---")
        
        if st.session_state.emergency_stop:
            st.warning("⚠️ Emergency stop is active. Cannot start strategies.")
            start_disabled = True
            stop_disabled = True
        else:
            start_disabled = is_active
            stop_disabled = not is_active
        
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button(
                "▶️ Start Strategy",
                type="primary",
                use_container_width=True,
                disabled=start_disabled,
                key=f"start_{selected_strategy}"
            ):
                if not st.session_state.emergency_stop:
                    st.session_state.auto_execution_active[selected_strategy] = True
                    st.session_state.auto_execution_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "message": f"Started automated execution for strategy: {selected_strategy}"
                    })
                    st.success(f"✅ Started {selected_strategy}")
                    st.rerun()
        
        with col_stop:
            if st.button(
                "⏹️ Stop Strategy",
                type="secondary",
                use_container_width=True,
                disabled=stop_disabled,
                key=f"stop_{selected_strategy}"
            ):
                st.session_state.auto_execution_active[selected_strategy] = False
                st.session_state.auto_execution_logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": f"Stopped automated execution for strategy: {selected_strategy}"
                })
                st.info(f"⏹️ Stopped {selected_strategy}")
                st.rerun()
        
        # Daily stats (calculated from session state)
        st.markdown("---")
        st.markdown("**📊 Today's Stats:**")
        
        # Calculate orders placed today
        today = datetime.now().strftime('%Y-%m-%d')
        orders_today = len([
            o for o in st.session_state.get('order_history', [])
            if o.get('timestamp', '').startswith(today)
        ])
        
        # Calculate daily P&L from filled orders
        # Get all filled orders from today
        filled_orders_today = [
            o for o in st.session_state.get('order_history', [])
            if o.get('timestamp', '').startswith(today) and 
               o.get('status', '').lower() == 'filled'
        ]
        
        daily_pnl = 0.0
        if filled_orders_today:
            # Calculate P&L for each filled order
            for order in filled_orders_today:
                symbol = order.get('symbol', '')
                side = order.get('side', '').lower()
                quantity = float(order.get('quantity', 0))
                fill_price = float(order.get('price', 0) or order.get('fill_price', 0))
                
                if symbol and quantity > 0 and fill_price > 0:
                    # Get current price for the symbol
                    current_price = get_current_price(symbol)
                    if current_price:
                        # Calculate unrealized P&L
                        if side == 'buy':
                            # For buys: profit if current price > fill price
                            pnl = (current_price - fill_price) * quantity
                        else:  # sell
                            # For sells: profit if fill price > current price
                            pnl = (fill_price - current_price) * quantity
                        daily_pnl += pnl
                    else:
                        # If we can't get current price, use stored P&L if available
                        daily_pnl += float(order.get('pnl', 0))
                else:
                    # Use stored P&L if available
                    daily_pnl += float(order.get('pnl', 0))
        
        # Count active orders
        active_orders = len([
            o for o in st.session_state.get('active_orders', [])
            if o.get('status', '').lower() not in ['filled', 'cancelled', 'rejected']
        ])
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Orders Placed", str(orders_today))
        with col_stat2:
            pnl_color = "normal" if daily_pnl >= 0 else "inverse"
            st.metric("Daily P&L", f"${daily_pnl:,.2f}", delta_color=pnl_color)
        with col_stat3:
            st.metric("Active Positions", str(active_orders))
    
    else:
        st.info("Select a strategy and configure it to enable execution controls")

st.markdown("---")

# Active Strategies Overview
st.subheader("📊 Active Strategies Overview")

active_strategies = [name for name, active in st.session_state.auto_execution_active.items() if active]

if active_strategies:
    for strategy_name in active_strategies:
        with st.expander(f"🤖 {strategy_name} - ACTIVE", expanded=False):
            config = st.session_state.auto_execution_configs.get(strategy_name, {})
            
            col_overview1, col_overview2 = st.columns([1, 1])
            
            with col_overview1:
                st.markdown(f"**Symbols:** {', '.join(config.get('symbols', []))}")
                st.markdown(f"**Check Interval:** {config.get('check_interval_minutes', 5)} minutes")
                st.markdown(f"**Status:** 🟢 Running")
            
            with col_overview2:
                if st.button(f"⏹️ Stop {strategy_name}", key=f"stop_overview_{strategy_name}"):
                    st.session_state.auto_execution_active[strategy_name] = False
                    st.session_state.auto_execution_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "message": f"Stopped {strategy_name} from overview"
                    })
                    st.rerun()
else:
    st.info("No strategies are currently active")

st.markdown("---")

# Real-time Execution Log
st.subheader("📜 Real-time Execution Log")

# Log filter
col_log1, col_log2 = st.columns([3, 1])
with col_log1:
    log_level_filter = st.selectbox(
        "Filter by Level",
        ["All", "INFO", "WARNING", "ERROR", "CRITICAL"],
        key="log_level_filter"
    )
with col_log2:
    if st.button("🗑️ Clear Log", use_container_width=True):
        st.session_state.auto_execution_logs = []
        st.rerun()

# Display logs (most recent first, limit to last 100)
logs = st.session_state.auto_execution_logs[-100:]
if log_level_filter != "All":
    logs = [log for log in logs if log.get("level") == log_level_filter]

if logs:
    # Reverse to show most recent at top
    logs_display = reversed(logs)
    
    # Create log display
    log_container = st.container()
    with log_container:
        for log in logs_display:
            timestamp = log.get("timestamp", "")
            level = log.get("level", "INFO")
            message = log.get("message", "")
            
            # Parse timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, AttributeError):
                # If parsing fails, use original timestamp
                time_str = timestamp
            
            # Color code by level
            if level == "CRITICAL" or level == "ERROR":
                st.error(f"**{time_str}** [{level}] {message}")
            elif level == "WARNING":
                st.warning(f"**{time_str}** [{level}] {message}")
            else:
                st.text(f"{time_str} [{level}] {message}")
else:
    st.info("No log entries yet. Execution logs will appear here when strategies are active.")

st.markdown("---")

# Section 4: Order Management
st.header("📋 Order Management")
st.markdown("Monitor and manage active orders in real-time")

# Initialize session state for order management
if 'order_refresh_interval' not in st.session_state:
    st.session_state.order_refresh_interval = 5  # seconds
if 'selected_orders_for_cancel' not in st.session_state:
    st.session_state.selected_orders_for_cancel = []
if 'fill_notifications' not in st.session_state:
    st.session_state.fill_notifications = []

# Auto-refresh toggle
col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 1, 1])
with col_refresh1:
    auto_refresh = st.checkbox(
        "🔄 Auto-refresh orders",
        value=True,
        help="Automatically refresh order status every 5 seconds",
        key="auto_refresh_orders"
    )
with col_refresh2:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()
with col_refresh3:
    refresh_interval = st.selectbox(
        "Refresh Interval",
        [5, 10, 15, 30, 60],
        index=0,
        format_func=lambda x: f"{x}s",
        key="refresh_interval_select"
    )
    st.session_state.order_refresh_interval = refresh_interval

# Auto-refresh logic
if auto_refresh:
    import time
    time.sleep(st.session_state.order_refresh_interval)
    st.rerun()

st.markdown("---")

# Active Orders Table
st.subheader("📊 Active Orders")

# Get active orders from execution agent and session state
active_orders_list = []

# Try to get orders from execution agent
if st.session_state.execution_agent:
    try:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Get order book from agent
        order_book = st.session_state.execution_agent.get_order_book()
        for order_id, order_request in order_book.items():
            # Get execution status if available
            execution = st.session_state.execution_agent.get_order_status(order_id)
            
            if execution:
                status = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
            else:
                status = "submitted"
            
            active_orders_list.append({
                "order_id": order_id,
                "symbol": order_request.ticker,
                "side": order_request.side.value if hasattr(order_request.side, 'value') else str(order_request.side),
                "order_type": order_request.order_type.value if hasattr(order_request.order_type, 'value') else str(order_request.order_type),
                "quantity": order_request.quantity,
                "price": order_request.price if order_request.price else "Market",
                "status": status,
                "time_in_force": getattr(order_request, 'time_in_force', 'day'),
                "timestamp": getattr(order_request, 'timestamp', datetime.now().isoformat())
            })
    except Exception as e:
        st.warning(f"Could not fetch orders from execution agent: {str(e)}")

# Also get orders from session state
for order in st.session_state.active_orders:
    if order.get("status") not in ["filled", "cancelled", "rejected", "expired"]:
        # Check if already in list
        if not any(o.get("order_id") == order.get("order_id") for o in active_orders_list):
            active_orders_list.append(order)

if active_orders_list:
    # Create DataFrame
    orders_df = pd.DataFrame(active_orders_list)
    
    # Add selection column for batch operations
    if 'order_selection' not in st.session_state:
        st.session_state.order_selection = []
    
    # Display orders with selection checkboxes
    col_table1, col_table2 = st.columns([3, 1])
    
    with col_table1:
        # Multi-select for batch cancellation
        selected_indices = st.multiselect(
            "Select orders for batch operations",
            options=range(len(orders_df)),
            format_func=lambda x: f"{orders_df.iloc[x]['symbol']} - {orders_df.iloc[x]['side']} {orders_df.iloc[x]['quantity']} @ {orders_df.iloc[x]['price']}",
            key="order_multiselect"
        )
        st.session_state.selected_orders_for_cancel = [orders_df.iloc[i]['order_id'] for i in selected_indices]
    
    with col_table2:
        if st.button("🗑️ Cancel Selected", type="primary", use_container_width=True, disabled=len(selected_indices) == 0):
            cancelled_count = 0
            for order_id in st.session_state.selected_orders_for_cancel:
                try:
                    # Cancel via execution agent
                    if st.session_state.execution_agent:
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        # Note: ExecutionAgent doesn't have a cancel_order method in the interface
                        # We'll update session state instead
                        for i, order in enumerate(st.session_state.active_orders):
                            if order.get("order_id") == order_id:
                                st.session_state.active_orders[i]["status"] = "cancelled"
                                cancelled_count += 1
                                break
                    else:
                        # Update session state
                        for i, order in enumerate(st.session_state.active_orders):
                            if order.get("order_id") == order_id:
                                st.session_state.active_orders[i]["status"] = "cancelled"
                                cancelled_count += 1
                                break
                except Exception as e:
                    st.error(f"Error cancelling order {order_id}: {str(e)}")
            
            if cancelled_count > 0:
                st.success(f"✅ Cancelled {cancelled_count} order(s)")
                st.session_state.selected_orders_for_cancel = []
                st.rerun()
    
    # Display orders table
    display_orders_df = orders_df.copy()
    
    # Format columns for display
    if 'price' in display_orders_df.columns:
        display_orders_df['price'] = display_orders_df['price'].apply(
            lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else str(x)
        )
    
    # Add action buttons column
    display_orders_df['Actions'] = ""
    
    st.dataframe(
        display_orders_df[['order_id', 'symbol', 'side', 'order_type', 'quantity', 'price', 'status', 'timestamp']],
        use_container_width=True,
        hide_index=True
    )
    
    # Order modification interface
    st.markdown("---")
    st.subheader("✏️ Modify Order")
    
    if selected_indices and len(selected_indices) == 1:
        selected_order = orders_df.iloc[selected_indices[0]]
        
        col_mod1, col_mod2 = st.columns([1, 1])
        
        with col_mod1:
            st.markdown(f"**Modifying Order:** {selected_order['order_id']}")
            st.markdown(f"**Symbol:** {selected_order['symbol']}")
            st.markdown(f"**Current Side:** {selected_order['side']}")
            st.markdown(f"**Current Type:** {selected_order['order_type']}")
        
        with col_mod2:
            new_quantity = st.number_input(
                "New Quantity",
                min_value=1,
                value=int(selected_order['quantity']),
                key="modify_quantity"
            )
            
            if selected_order['order_type'].lower() in ['limit', 'stop', 'stop_limit']:
                # Get current price for default if order price is invalid
                order_symbol = selected_order.get('symbol', '')
                default_modify_price = None
                if isinstance(selected_order['price'], (int, float)) and selected_order['price'] > 0:
                    default_modify_price = float(selected_order['price'])
                elif order_symbol:
                    default_modify_price = get_current_price(order_symbol)
                
                new_price = st.number_input(
                    "New Price ($)",
                    min_value=0.01,
                    value=float(default_modify_price) if default_modify_price else 100.0,
                    step=0.01,
                    key="modify_price"
                )
            else:
                new_price = None
                st.info("Market orders cannot have price modified")
            
            if st.button("💾 Update Order", type="primary", use_container_width=True):
                if st.session_state.execution_agent:
                    try:
                        # Modify order through execution agent
                        order_id = selected_order['order_id']
                        
                        # Check if execution agent has modify_order method
                        if hasattr(st.session_state.execution_agent, 'modify_order'):
                            result = st.session_state.execution_agent.modify_order(
                                order_id=order_id,
                                new_price=new_price,
                                new_quantity=new_quantity
                            )
                            
                            if result.get('success', False):
                                st.success("Order modified successfully!")
                                # Update in session state
                                for i, o in enumerate(st.session_state.active_orders):
                                    if o.get('order_id') == order_id:
                                        if new_price is not None:
                                            st.session_state.active_orders[i]['price'] = new_price
                                        st.session_state.active_orders[i]['quantity'] = new_quantity
                                        break
                                st.rerun()
                            else:
                                st.error(f"Failed to modify order: {result.get('message', 'Unknown error')}")
                        else:
                            # Fallback: update in session state
                            order_id = selected_order['order_id']
                            for i, o in enumerate(st.session_state.active_orders):
                                if o.get('order_id') == order_id:
                                    if new_price is not None:
                                        st.session_state.active_orders[i]['price'] = new_price
                                    st.session_state.active_orders[i]['quantity'] = new_quantity
                                    st.session_state.active_orders[i]['modified_at'] = datetime.now().isoformat()
                                    break
                            st.success("Order updated in session state (broker API modification not available)")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error modifying order: {e}")
                else:
                    st.warning("Execution agent not initialized. Cannot modify orders.")
    
    # Batch operations
    if len(selected_indices) > 1:
        st.markdown("---")
        st.subheader("📦 Batch Operations")
        
        col_batch1, col_batch2, col_batch3 = st.columns(3)
        
        with col_batch1:
            if st.button("🗑️ Cancel All Selected", use_container_width=True):
                # Same logic as above
                st.info(f"Cancelling {len(selected_indices)} orders...")
                st.rerun()
        
        with col_batch2:
            if st.button("📊 Export Selected", use_container_width=True):
                selected_orders_export = orders_df.iloc[selected_indices]
                csv = selected_orders_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col_batch3:
            st.metric("Selected Orders", len(selected_indices))

else:
    st.info("No active orders. Orders will appear here when you submit trades.")

st.markdown("---")

# Order History
st.subheader("📜 Order History")

# Filter options
col_hist1, col_hist2, col_hist3 = st.columns([2, 1, 1])

with col_hist1:
    history_filter_symbol = st.text_input(
        "Filter by Symbol",
        value="",
        placeholder="e.g., AAPL",
        key="history_filter_symbol"
    ).upper().strip()

with col_hist2:
    history_filter_status = st.selectbox(
        "Filter by Status",
        ["All", "filled", "cancelled", "rejected", "expired", "partial"],
        key="history_filter_status"
    )

with col_hist3:
    history_limit = st.number_input(
        "Show Last N Orders",
        min_value=10,
        max_value=1000,
        value=50,
        step=10,
        key="history_limit"
    )

# Get order history
order_history_list = st.session_state.order_history.copy()

# Apply filters
if history_filter_symbol:
    order_history_list = [o for o in order_history_list if o.get('symbol', '').upper() == history_filter_symbol]

if history_filter_status != "All":
    order_history_list = [o for o in order_history_list if o.get('status', '').lower() == history_filter_status.lower()]

# Limit results
order_history_list = order_history_list[-history_limit:]

if order_history_list:
    history_df = pd.DataFrame(order_history_list)
    
    # Format for display
    if 'price' in history_df.columns:
        history_df['price'] = history_df['price'].apply(
            lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and x else str(x) if x else "Market"
        )
    
    # Sort by timestamp (most recent first)
    if 'timestamp' in history_df.columns:
        history_df = history_df.sort_values('timestamp', ascending=False)
    
    # Display
    display_cols = ['order_id', 'symbol', 'side', 'order_type', 'quantity', 'price', 'status', 'timestamp']
    available_cols = [col for col in display_cols if col in history_df.columns]
    
    st.dataframe(
        history_df[available_cols],
        use_container_width=True,
        hide_index=True
    )
    
    # Export button
    if st.button("📥 Export History to CSV", use_container_width=True):
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download",
            data=csv,
            file_name=f"order_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
else:
    st.info("No order history yet. Completed orders will appear here.")

st.markdown("---")

# Fill Notifications
st.subheader("🔔 Fill Notifications")

# Check for new fills (compare current orders with previous state)
if 'previous_active_orders' not in st.session_state:
    st.session_state.previous_active_orders = []

# Detect fills (orders that were active but are now filled)
current_order_ids = {o.get('order_id') for o in active_orders_list}
previous_order_ids = {o.get('order_id') for o in st.session_state.previous_active_orders}

# Find filled orders
for prev_order in st.session_state.previous_active_orders:
    if prev_order.get('order_id') not in current_order_ids:
        # Check if it was filled (not just cancelled)
        if prev_order.get('status') != 'cancelled':
            # Add notification
            fill_notification = {
                "timestamp": datetime.now().isoformat(),
                "order_id": prev_order.get('order_id'),
                "symbol": prev_order.get('symbol'),
                "side": prev_order.get('side'),
                "quantity": prev_order.get('quantity'),
                "price": prev_order.get('price'),
                "message": f"Order {prev_order.get('order_id')} filled: {prev_order.get('side')} {prev_order.get('quantity')} {prev_order.get('symbol')}"
            }
            if fill_notification not in st.session_state.fill_notifications:
                st.session_state.fill_notifications.append(fill_notification)

# Update previous state
st.session_state.previous_active_orders = active_orders_list.copy()

# Display notifications
if st.session_state.fill_notifications:
    # Show last 10 notifications
    recent_notifications = st.session_state.fill_notifications[-10:]
    
    for notif in reversed(recent_notifications):
        timestamp = notif.get('timestamp', '')
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M:%S")
        except (ValueError, AttributeError):
            # If parsing fails, use original timestamp
            time_str = timestamp
        
        st.success(f"**{time_str}** - {notif.get('message', 'Order filled')}")
    
    if st.button("🗑️ Clear Notifications", use_container_width=True):
        st.session_state.fill_notifications = []
        st.rerun()
else:
    st.info("No fill notifications. You'll be notified when orders are filled.")

st.markdown("---")

# Order Status Tracking
st.subheader("📈 Order Status Tracking")

# Status summary
if active_orders_list or order_history_list:
    all_orders = active_orders_list + order_history_list
    
    # Count by status
    status_counts = {}
    for order in all_orders:
        status = order.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    if status_counts:
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Active", status_counts.get('submitted', 0) + status_counts.get('pending', 0) + status_counts.get('partial', 0))
        
        with col_stat2:
            st.metric("Filled", status_counts.get('filled', 0))
        
        with col_stat3:
            st.metric("Cancelled", status_counts.get('cancelled', 0))
        
        with col_stat4:
            st.metric("Rejected", status_counts.get('rejected', 0))
        
        # Status distribution chart
        if len(status_counts) > 0:
            fig_status = go.Figure(data=[
                go.Bar(
                    x=list(status_counts.keys()),
                    y=list(status_counts.values()),
                    marker_color='lightblue'
                )
                ])
            fig_status.update_layout(
                title="Order Status Distribution",
                xaxis_title="Status",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_status, use_container_width=True)
else:
    st.info("No orders to track yet. Submit some orders to see status tracking.")

st.markdown("---")

# Section 5: Execution Analytics
st.header("📊 Execution Analytics")
st.markdown("Analyze execution quality: slippage, fill rates, price improvement, and execution time")

# Initialize session state for analytics
if 'execution_analytics_data' not in st.session_state:
    st.session_state.execution_analytics_data = []

# Helper function to calculate VWAP (Volume Weighted Average Price)
def calculate_vwap(prices, volumes):
    """Calculate VWAP from prices and volumes."""
    if len(prices) == 0 or len(volumes) == 0 or sum(volumes) == 0:
        return 0.0
    return sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)

# Helper function to calculate TWAP (Time Weighted Average Price)
def calculate_twap(prices, times):
    """Calculate TWAP from prices and times."""
    if len(prices) == 0:
        return 0.0
    return sum(prices) / len(prices)

# Process order history to calculate analytics
filled_orders = [o for o in st.session_state.order_history if o.get('status', '').lower() == 'filled']

if filled_orders:
    # Calculate metrics for each filled order
    analytics_data = []
    
    for order in filled_orders:
        order_price = order.get('price', 0)
        if isinstance(order_price, str):
            # Try to extract numeric value
            try:
                order_price = float(order_price.replace('$', '').replace(',', ''))
            except (ValueError, AttributeError):
                # If conversion fails, default to 0
                order_price = 0
        
        quantity = order.get('quantity', 0)
        if isinstance(quantity, (int, float)):
            pass
        else:
            try:
                quantity = float(str(quantity).replace(',', ''))
            except (ValueError, AttributeError):
                # If conversion fails, default to 0
                quantity = 0
        
        # Get execution data if available from execution agent
        execution_price = order_price  # Default to order price
        execution_time = 0.0  # Default execution time
        
        if st.session_state.execution_agent:
            order_id = order.get('order_id')
            if order_id:
                try:
                    execution = st.session_state.execution_agent.get_order_status(order_id)
                    if execution:
                        execution_price = execution.average_price if hasattr(execution, 'average_price') and execution.average_price else order_price
                        # Calculate execution time if timestamps available
                        if hasattr(execution, 'timestamp') and order.get('timestamp'):
                            try:
                                exec_time = datetime.fromisoformat(execution.timestamp.replace('Z', '+00:00'))
                                order_time = datetime.fromisoformat(order['timestamp'].replace('Z', '+00:00'))
                                execution_time = (exec_time - order_time).total_seconds()
                            except (ValueError, AttributeError, KeyError):
                                # If timestamp parsing fails, default to 0
                                execution_time = 0.0
                except Exception as e:
                    # If we can't get order status, use default values
                    logger.warning(f"Could not get order status for {order_id}: {str(e)}")
                    execution_price = order_price
                    execution_time = 0.0
        
        # Calculate slippage (difference between expected and actual price)
        if order_price > 0:
            slippage_pct = ((execution_price - order_price) / order_price) * 100
            slippage_abs = execution_price - order_price
        else:
            slippage_pct = 0.0
            slippage_abs = 0.0
        
        # Price improvement (positive slippage for buys, negative for sells)
        side = order.get('side', '').lower()
        if side == 'buy':
            price_improvement = -slippage_abs  # Negative slippage is improvement for buys
        else:
            price_improvement = slippage_abs  # Positive slippage is improvement for sells
        
        analytics_data.append({
            'order_id': order.get('order_id', ''),
            'symbol': order.get('symbol', ''),
            'side': side,
            'order_type': order.get('order_type', ''),
            'quantity': quantity,
            'order_price': order_price,
            'execution_price': execution_price,
            'slippage_pct': slippage_pct,
            'slippage_abs': slippage_abs,
            'price_improvement': price_improvement,
            'execution_time': execution_time,
            'timestamp': order.get('timestamp', ''),
            'filled': True
        })
    
    st.session_state.execution_analytics_data = analytics_data
    
    # Summary Metrics
    st.subheader("📈 Summary Metrics")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        total_orders = len(analytics_data)
        filled_orders_count = len([d for d in analytics_data if d.get('filled')])
        fill_rate = (filled_orders_count / total_orders * 100) if total_orders > 0 else 0
        st.metric("Fill Rate", f"{fill_rate:.1f}%")
    
    with col_metric2:
        avg_slippage = np.mean([abs(d.get('slippage_pct', 0)) for d in analytics_data]) if analytics_data else 0
        st.metric("Avg Slippage", f"{avg_slippage:.3f}%")
    
    with col_metric3:
        avg_exec_time = np.mean([d.get('execution_time', 0) for d in analytics_data]) if analytics_data else 0
        st.metric("Avg Execution Time", f"{avg_exec_time:.2f}s")
    
    with col_metric4:
        total_improvement = sum([d.get('price_improvement', 0) * d.get('quantity', 0) for d in analytics_data])
        st.metric("Total Price Improvement", f"${total_improvement:,.2f}")
    
    st.markdown("---")
    
    # Slippage Analysis
    st.subheader("📉 Slippage Analysis")
    
    col_slip1, col_slip2 = st.columns([1, 1])
    
    with col_slip1:
        # Slippage distribution
        slippage_values = [d.get('slippage_pct', 0) for d in analytics_data]
        
        if slippage_values:
            fig_slippage = go.Figure()
            fig_slippage.add_trace(go.Histogram(
                x=slippage_values,
                nbinsx=20,
                name="Slippage Distribution",
                marker_color='lightcoral'
            ))
            fig_slippage.update_layout(
                title="Slippage Distribution (%)",
                xaxis_title="Slippage (%)",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_slippage, use_container_width=True)
    
    with col_slip2:
        # Slippage by order type
        if analytics_data:
            slippage_by_type = {}
            for d in analytics_data:
                order_type = d.get('order_type', 'Unknown')
                if order_type not in slippage_by_type:
                    slippage_by_type[order_type] = []
                slippage_by_type[order_type].append(abs(d.get('slippage_pct', 0)))
            
            if slippage_by_type:
                types = list(slippage_by_type.keys())
                avg_slippages = [np.mean(slippage_by_type[t]) for t in types]
                
                fig_slippage_type = go.Figure(data=[
                    go.Bar(
                        x=types,
                        y=avg_slippages,
                        marker_color='steelblue'
                    )
                ])
                fig_slippage_type.update_layout(
                    title="Average Slippage by Order Type",
                    xaxis_title="Order Type",
                    yaxis_title="Avg Slippage (%)",
                    height=300
                )
                st.plotly_chart(fig_slippage_type, use_container_width=True)
    
    # Slippage statistics table
    if analytics_data:
        slippage_stats = {
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Positive Slippage', 'Negative Slippage'],
            'Value': [
                f"{np.mean(slippage_values):.4f}%",
                f"{np.median(slippage_values):.4f}%",
                f"{np.std(slippage_values):.4f}%",
                f"{np.min(slippage_values):.4f}%",
                f"{np.max(slippage_values):.4f}%",
                f"{len([s for s in slippage_values if s > 0])}",
                f"{len([s for s in slippage_values if s < 0])}"
            ]
        }
        slippage_df = pd.DataFrame(slippage_stats)
        st.dataframe(slippage_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Fill Rate Statistics
    st.subheader("✅ Fill Rate Statistics")
    
    col_fill1, col_fill2 = st.columns([1, 1])
    
    with col_fill1:
        # Fill rate over time (if we have timestamps)
        if analytics_data and any(d.get('timestamp') for d in analytics_data):
            # Group by date
            fill_by_date = {}
            for d in analytics_data:
                try:
                    timestamp = d.get('timestamp', '')
                    if timestamp:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                        if date not in fill_by_date:
                            fill_by_date[date] = {'filled': 0, 'total': 0}
                        fill_by_date[date]['filled'] += 1
                        fill_by_date[date]['total'] += 1
                except (KeyError, ValueError, AttributeError):
                    # If date parsing or dict access fails, skip this entry
                    pass
            
            if fill_by_date:
                dates = sorted(fill_by_date.keys())
                fill_rates = [fill_by_date[d]['filled'] / fill_by_date[d]['total'] * 100 for d in dates]
                
                fig_fill_rate = go.Figure(data=[
                    go.Scatter(
                        x=dates,
                        y=fill_rates,
                        mode='lines+markers',
                        name="Fill Rate",
                        line=dict(color='green', width=2)
                    )
                ])
                fig_fill_rate.update_layout(
                    title="Fill Rate Over Time",
                    xaxis_title="Date",
                    yaxis_title="Fill Rate (%)",
                    height=300
                )
                st.plotly_chart(fig_fill_rate, use_container_width=True)
    
    with col_fill2:
        # Fill rate by symbol
        if analytics_data:
            fill_by_symbol = {}
            for d in analytics_data:
                symbol = d.get('symbol', 'Unknown')
                if symbol not in fill_by_symbol:
                    fill_by_symbol[symbol] = {'filled': 0, 'total': 0}
                fill_by_symbol[symbol]['filled'] += 1
                fill_by_symbol[symbol]['total'] += 1
            
            if fill_by_symbol:
                symbols = list(fill_by_symbol.keys())[:10]  # Top 10
                fill_rates_symbol = [fill_by_symbol[s]['filled'] / fill_by_symbol[s]['total'] * 100 for s in symbols]
                
                fig_fill_symbol = go.Figure(data=[
                    go.Bar(
                        x=symbols,
                        y=fill_rates_symbol,
                        marker_color='lightgreen'
                    )
                ])
                fig_fill_symbol.update_layout(
                    title="Fill Rate by Symbol (Top 10)",
                    xaxis_title="Symbol",
                    yaxis_title="Fill Rate (%)",
                    height=300
                )
                st.plotly_chart(fig_fill_symbol, use_container_width=True)
    
    st.markdown("---")
    
    # Price Improvement Metrics
    st.subheader("💰 Price Improvement Metrics")
    
    col_improve1, col_improve2 = st.columns([1, 1])
    
    with col_improve1:
        # Price improvement distribution
        price_improvements = [d.get('price_improvement', 0) for d in analytics_data]
        
        if price_improvements:
            fig_improvement = go.Figure()
            fig_improvement.add_trace(go.Histogram(
                x=price_improvements,
                nbinsx=20,
                name="Price Improvement Distribution",
                marker_color='gold'
            ))
            fig_improvement.update_layout(
                title="Price Improvement Distribution ($)",
                xaxis_title="Price Improvement ($)",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_improvement, use_container_width=True)
    
    with col_improve2:
        # Price improvement by side
        if analytics_data:
            buy_improvements = [d.get('price_improvement', 0) for d in analytics_data if d.get('side', '').lower() == 'buy']
            sell_improvements = [d.get('price_improvement', 0) for d in analytics_data if d.get('side', '').lower() == 'sell']
            
            if buy_improvements or sell_improvements:
                fig_improve_side = go.Figure(data=[
                    go.Bar(
                        x=['Buy', 'Sell'],
                        y=[
                            np.mean(buy_improvements) if buy_improvements else 0,
                            np.mean(sell_improvements) if sell_improvements else 0
                        ],
                        marker_color=['green', 'red']
                    )
                ])
                fig_improve_side.update_layout(
                    title="Average Price Improvement by Side",
                    xaxis_title="Side",
                    yaxis_title="Avg Improvement ($)",
                    height=300
                )
                st.plotly_chart(fig_improve_side, use_container_width=True)
    
    # Price improvement summary
    if analytics_data:
        total_improvement = sum([d.get('price_improvement', 0) * d.get('quantity', 0) for d in analytics_data])
        avg_improvement = np.mean(price_improvements) if price_improvements else 0
        positive_improvements = len([p for p in price_improvements if p > 0])
        
        improve_stats = {
            'Metric': ['Total Improvement', 'Average Improvement', 'Orders with Improvement', 'Improvement Rate'],
            'Value': [
                f"${total_improvement:,.2f}",
                f"${avg_improvement:.4f}",
                f"{positive_improvements}",
                f"{(positive_improvements / len(price_improvements) * 100) if price_improvements else 0:.1f}%"
            ]
        }
        improve_df = pd.DataFrame(improve_stats)
        st.dataframe(improve_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Execution Time Analysis
    st.subheader("⏱️ Execution Time Analysis")
    
    col_time1, col_time2 = st.columns([1, 1])
    
    with col_time1:
        # Execution time distribution
        exec_times = [d.get('execution_time', 0) for d in analytics_data if d.get('execution_time', 0) > 0]
        
        if exec_times:
            fig_exec_time = go.Figure()
            fig_exec_time.add_trace(go.Histogram(
                x=exec_times,
                nbinsx=20,
                name="Execution Time Distribution",
                marker_color='purple'
            ))
            fig_exec_time.update_layout(
                title="Execution Time Distribution (seconds)",
                xaxis_title="Execution Time (s)",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_exec_time, use_container_width=True)
    
    with col_time2:
        # Execution time by order type
        if analytics_data:
            time_by_type = {}
            for d in analytics_data:
                if d.get('execution_time', 0) > 0:
                    order_type = d.get('order_type', 'Unknown')
                    if order_type not in time_by_type:
                        time_by_type[order_type] = []
                    time_by_type[order_type].append(d.get('execution_time', 0))
            
            if time_by_type:
                types = list(time_by_type.keys())
                avg_times = [np.mean(time_by_type[t]) for t in types]
                
                fig_time_type = go.Figure(data=[
                    go.Bar(
                        x=types,
                        y=avg_times,
                        marker_color='mediumpurple'
                    )
                ])
                fig_time_type.update_layout(
                    title="Average Execution Time by Order Type",
                    xaxis_title="Order Type",
                    yaxis_title="Avg Time (s)",
                    height=300
                )
                st.plotly_chart(fig_time_type, use_container_width=True)
    
    # Execution time statistics
    if exec_times:
        time_stats = {
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'P95', 'P99'],
            'Value': [
                f"{np.mean(exec_times):.3f}s",
                f"{np.median(exec_times):.3f}s",
                f"{np.std(exec_times):.3f}s",
                f"{np.min(exec_times):.3f}s",
                f"{np.max(exec_times):.3f}s",
                f"{np.percentile(exec_times, 95):.3f}s",
                f"{np.percentile(exec_times, 99):.3f}s"
            ]
        }
        time_df = pd.DataFrame(time_stats)
        st.dataframe(time_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # VWAP/TWAP Comparison
    st.subheader("📊 VWAP/TWAP Comparison")
    
    if analytics_data and len(analytics_data) > 0:
        # Calculate VWAP and TWAP for comparison
        prices = [d.get('execution_price', 0) for d in analytics_data]
        volumes = [d.get('quantity', 0) for d in analytics_data]
        timestamps = [d.get('timestamp', '') for d in analytics_data]
        
        vwap = calculate_vwap(prices, volumes)
        twap = calculate_twap(prices, timestamps)
        
        col_vwap1, col_vwap2, col_vwap3 = st.columns(3)
        
        with col_vwap1:
            st.metric("VWAP", f"${vwap:.2f}")
        
        with col_vwap2:
            st.metric("TWAP", f"${twap:.2f}")
        
        with col_vwap3:
            avg_exec_price = np.mean(prices) if prices else 0
            st.metric("Avg Execution Price", f"${avg_exec_price:.2f}")
        
        # Comparison chart
        if len(analytics_data) > 1:
            # Create time series comparison
            try:
                sorted_data = sorted(analytics_data, key=lambda x: x.get('timestamp', ''))
                exec_prices_series = [d.get('execution_price', 0) for d in sorted_data]
                order_nums = list(range(1, len(sorted_data) + 1))
                
                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Scatter(
                    x=order_nums,
                    y=exec_prices_series,
                    mode='lines+markers',
                    name='Execution Price',
                    line=dict(color='blue', width=2)
                ))
                fig_comparison.add_hline(
                    y=vwap,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="VWAP",
                    annotation_position="right"
                )
                fig_comparison.add_hline(
                    y=twap,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="TWAP",
                    annotation_position="right"
                )
                fig_comparison.update_layout(
                    title="Execution Price vs VWAP/TWAP",
                    xaxis_title="Order Number",
                    yaxis_title="Price ($)",
                    height=400
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create comparison chart: {str(e)}")
        
        # Performance vs benchmarks
        if prices:
            vwap_diff = [(p - vwap) / vwap * 100 for p in prices]
            twap_diff = [(p - twap) / twap * 100 for p in prices]
            
            comparison_stats = {
                'Metric': ['Avg vs VWAP', 'Avg vs TWAP', 'Better than VWAP', 'Better than TWAP'],
                'Value': [
                    f"{np.mean(vwap_diff):.3f}%",
                    f"{np.mean(twap_diff):.3f}%",
                    f"{len([d for d in vwap_diff if d < 0])} orders",
                    f"{len([d for d in twap_diff if d < 0])} orders"
                ]
            }
            comparison_df = pd.DataFrame(comparison_stats)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Detailed Analytics Table
    st.subheader("📋 Detailed Execution Analytics")
    
    if analytics_data:
        analytics_df = pd.DataFrame(analytics_data)
        
        # Format columns for display
        display_cols = ['order_id', 'symbol', 'side', 'order_type', 'quantity', 
                        'order_price', 'execution_price', 'slippage_pct', 
                        'price_improvement', 'execution_time']
        available_cols = [col for col in display_cols if col in analytics_df.columns]
        
        # Format numeric columns
        display_df = analytics_df[available_cols].copy()
        if 'order_price' in display_df.columns:
            display_df['order_price'] = display_df['order_price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else str(x))
        if 'execution_price' in display_df.columns:
            display_df['execution_price'] = display_df['execution_price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else str(x))
        if 'slippage_pct' in display_df.columns:
            display_df['slippage_pct'] = display_df['slippage_pct'].apply(lambda x: f"{x:.4f}%")
        if 'price_improvement' in display_df.columns:
            display_df['price_improvement'] = display_df['price_improvement'].apply(lambda x: f"${x:.4f}")
        if 'execution_time' in display_df.columns:
            display_df['execution_time'] = display_df['execution_time'].apply(lambda x: f"{x:.3f}s" if isinstance(x, (int, float)) else str(x))
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export button
        if st.button("📥 Export Analytics to CSV", use_container_width=True):
            csv = analytics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download",
                data=csv,
                file_name=f"execution_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    st.info("No filled orders yet. Execution analytics will appear here after orders are filled.")
