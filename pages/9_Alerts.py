"""
Alerts & Notifications Page

Merges functionality from:
- 18_Alerts.py (standalone)

Features:
- Price alerts
- Strategy signal alerts
- Risk limit alerts
- Portfolio alerts
- Multiple notification channels (email, SMS, Telegram, Slack, webhook)
- Alert conditions builder
- Alert history
"""

import logging
import sys
from datetime import datetime, timedelta, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Backend imports
try:
    from trading.monitoring.health_check import HealthMonitor
    from system.infra.agents.alert_manager import AlertManager
    from trading.utils.notification_system import NotificationSystem
    
    ALERT_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some alert modules not available: {e}")
    ALERT_MODULES_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Alerts & Notifications",
    page_icon="🔔",
    layout="wide"
)

# Initialize session state
if 'alert_manager' not in st.session_state:
    try:
        st.session_state.alert_manager = AlertManager() if ALERT_MODULES_AVAILABLE else None
    except Exception as e:
        logger.warning(f"Could not initialize alert manager: {e}")
        st.session_state.alert_manager = None

if 'notification_system' not in st.session_state:
    try:
        st.session_state.notification_system = NotificationSystem() if ALERT_MODULES_AVAILABLE else None
    except Exception as e:
        logger.warning(f"Could not initialize notification system: {e}")
        st.session_state.notification_system = None

if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = {}

if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

if 'alert_templates' not in st.session_state:
    st.session_state.alert_templates = {}

# Main page title
st.title("🔔 Alerts & Notifications")
st.markdown("Configure and manage trading alerts with multi-channel notifications")

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Active Alerts",
    "➕ Create Alert",
    "📋 Alert Templates",
    "📜 Alert History",
    "⚙️ Notification Settings"
])

# TAB 1: Active Alerts
with tab1:
    st.header("📊 Active Alerts")
    st.markdown("Monitor and manage all your configured alerts in real-time.")
    
    # Quick Filters
    st.subheader("🔍 Quick Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox(
            "Filter by Type",
            ["All", "Price Alert", "Technical Indicator", "Strategy Signal", "Risk Limit", "Portfolio", "Custom"],
            help="Filter alerts by type"
        )
    
    with col2:
        filter_status = st.selectbox(
            "Filter by Status",
            ["All", "Active", "Paused"],
            help="Filter alerts by status"
        )
    
    with col3:
        search_alerts = st.text_input(
            "Search Alerts",
            placeholder="Search by name...",
            help="Search alerts by name"
        )
    
    # Filter alerts
    filtered_alerts = {}
    
    for name, alert_data in st.session_state.active_alerts.items():
        # Type filter
        if filter_type != "All":
            if alert_data.get('alert_type', 'Unknown') != filter_type:
                continue
        
        # Status filter
        if filter_status != "All":
            is_active = alert_data.get('enabled', True)
            if filter_status == "Active" and not is_active:
                continue
            if filter_status == "Paused" and is_active:
                continue
        
        # Search filter
        if search_alerts:
            if search_alerts.lower() not in name.lower():
                continue
        
        filtered_alerts[name] = alert_data
    
    st.markdown("---")
    
    # Alerts Table
    st.subheader("📋 Alerts Dashboard")
    
    if filtered_alerts:
        # Create alerts table
        alerts_data = []
        for name, alert_data in filtered_alerts.items():
            alert_type = alert_data.get('alert_type', 'Unknown')
            condition = alert_data.get('condition', 'N/A')
            enabled = alert_data.get('enabled', True)
            status = "🟢 Active" if enabled else "🔴 Paused"
            last_triggered = alert_data.get('last_triggered', 'Never')
            
            if last_triggered != 'Never' and isinstance(last_triggered, str):
                try:
                    dt = datetime.fromisoformat(last_triggered.replace('Z', '+00:00'))
                    last_triggered = dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, AttributeError) as e:
                    # Keep original string if parsing fails
                    pass
            
            alerts_data.append({
                "Alert Name": name,
                "Type": alert_type,
                "Condition": condition,
                "Status": status,
                "Last Triggered": last_triggered,
                "Trigger Count": alert_data.get('trigger_count', 0)
            })
        
        alerts_df = pd.DataFrame(alerts_data)
        st.dataframe(alerts_df, use_container_width=True, height=400)
        
        # Alert Actions
        st.markdown("---")
        st.subheader("⚙️ Alert Actions")
        
        selected_alert = st.selectbox(
            "Select Alert",
            options=list(filtered_alerts.keys()),
            help="Choose an alert to manage"
        )
        
        if selected_alert:
            alert_data = filtered_alerts[selected_alert]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                # Enable/Disable Toggle
                current_status = alert_data.get('enabled', True)
                new_status = st.checkbox(
                    "Enabled",
                    value=current_status,
                    key=f"enable_{selected_alert}",
                    help="Enable or disable this alert"
                )
                if new_status != current_status:
                    alert_data["enabled"] = new_status
                    st.session_state.active_alerts[selected_alert] = alert_data
                    st.rerun()
            
            with col2:
                # Test Alert
                if st.button("🧪 Test Alert", use_container_width=True, key=f"test_{selected_alert}"):
                    try:
                        # Simulate alert trigger
                        alert_data["last_triggered"] = datetime.now().isoformat()
                        alert_data["trigger_count"] = alert_data.get("trigger_count", 0) + 1
                        st.session_state.active_alerts[selected_alert] = alert_data
                        
                        # Add to history
                        st.session_state.alert_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "alert_name": selected_alert,
                            "alert_type": alert_data.get('alert_type', 'Unknown'),
                            "condition": alert_data.get('condition', 'N/A'),
                            "value": "Test Trigger",
                            "notification_sent": True,
                            "action_taken": None
                        })
                        
                        st.success(f"✅ Test alert triggered for '{selected_alert}'!")
                        st.info(f"📧 Notification would be sent via: {', '.join(alert_data.get('channels', ['Email']))}")
                    except Exception as e:
                        st.error(f"Error testing alert: {str(e)}")
            
            with col3:
                # Edit Alert
                if st.button("✏️ Edit", use_container_width=True, key=f"edit_{selected_alert}"):
                    st.session_state.editing_alert = selected_alert
                    st.info(f"Edit functionality for '{selected_alert}' - navigate to Tab 2 to edit")
            
            with col4:
                # View Details
                if st.button("👁️ View Details", use_container_width=True, key=f"view_{selected_alert}"):
                    st.session_state.viewing_alert = selected_alert
                    st.rerun()
            
            with col5:
                # Delete Alert
                if st.button("🗑️ Delete", use_container_width=True, key=f"delete_{selected_alert}"):
                    if selected_alert in st.session_state.active_alerts:
                        del st.session_state.active_alerts[selected_alert]
                        st.success(f"✅ Alert '{selected_alert}' deleted!")
                        st.rerun()
            
            # Alert Details (if viewing)
            if 'viewing_alert' in st.session_state and st.session_state.viewing_alert == selected_alert:
                st.markdown("---")
                st.subheader(f"📄 Alert Details: {selected_alert}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Alert Name:** {selected_alert}")
                    st.markdown(f"**Type:** {alert_data.get('alert_type', 'Unknown')}")
                    st.markdown(f"**Condition:** {alert_data.get('condition', 'N/A')}")
                    st.markdown(f"**Status:** {'🟢 Active' if alert_data.get('enabled', True) else '🔴 Paused'}")
                    st.markdown(f"**Frequency:** {alert_data.get('frequency', 'Once')}")
                
                with col2:
                    st.markdown(f"**Symbol:** {alert_data.get('symbol', 'N/A')}")
                    st.markdown(f"**Threshold:** {alert_data.get('threshold', 'N/A')}")
                    st.markdown(f"**Channels:** {', '.join(alert_data.get('channels', ['Email']))}")
                    st.markdown(f"**Priority:** {alert_data.get('priority', 'Medium')}")
                    st.markdown(f"**Trigger Count:** {alert_data.get('trigger_count', 0)}")
                
                last_triggered = alert_data.get('last_triggered', 'Never')
                if last_triggered != 'Never':
                    try:
                        dt = datetime.fromisoformat(last_triggered.replace('Z', '+00:00'))
                        st.markdown(f"**Last Triggered:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except (ValueError, AttributeError) as e:
                        st.markdown(f"**Last Triggered:** {last_triggered}")
                else:
                    st.markdown("**Last Triggered:** Never")
                
                if alert_data.get('description'):
                    st.markdown(f"**Description:** {alert_data.get('description')}")
        
        # Bulk Actions
        st.markdown("---")
        st.subheader("📦 Bulk Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Enable All", use_container_width=True):
                for name in filtered_alerts.keys():
                    if name in st.session_state.active_alerts:
                        st.session_state.active_alerts[name]["enabled"] = True
                st.success(f"✅ Enabled {len(filtered_alerts)} alert(s)!")
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause All", use_container_width=True):
                for name in filtered_alerts.keys():
                    if name in st.session_state.active_alerts:
                        st.session_state.active_alerts[name]["enabled"] = False
                st.success(f"✅ Paused {len(filtered_alerts)} alert(s)!")
                st.rerun()
        
        with col3:
            if st.button("🗑️ Delete All Filtered", use_container_width=True):
                deleted_count = 0
                for name in filtered_alerts.keys():
                    if name in st.session_state.active_alerts:
                        del st.session_state.active_alerts[name]
                        deleted_count += 1
                st.success(f"✅ Deleted {deleted_count} alert(s)!")
                st.rerun()
        
        # Recent Triggers Feed
        st.markdown("---")
        st.subheader("📢 Recent Triggers")
        
        # Get recent triggers from history
        recent_triggers = [
            trigger for trigger in st.session_state.alert_history
            if trigger.get('alert_name') in filtered_alerts.keys()
        ]
        
        # Sort by timestamp (newest first)
        recent_triggers.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        recent_triggers = recent_triggers[:10]  # Show last 10
        
        if recent_triggers:
            for trigger in recent_triggers:
                timestamp = trigger.get('timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, AttributeError) as e:
                        timestamp_str = timestamp
                else:
                    timestamp_str = timestamp
                
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{trigger.get('alert_name', 'Unknown')}**")
                        st.caption(f"Type: {trigger.get('alert_type', 'Unknown')}")
                    
                    with col2:
                        st.markdown(f"Condition: {trigger.get('condition', 'N/A')}")
                        st.caption(f"Value: {trigger.get('value', 'N/A')}")
                    
                    with col3:
                        st.caption(f"🕐 {timestamp_str}")
                        if trigger.get('notification_sent'):
                            st.success("✅ Sent")
                        else:
                            st.warning("⚠️ Failed")
        else:
            st.info("No recent triggers. Alerts will appear here when triggered.")
    else:
        st.info("No alerts found. Create alerts in Tab 2 to get started.")
        
        # Quick Stats
        st.markdown("---")
        st.subheader("📊 Alert Statistics")
        
        total_alerts = len(st.session_state.active_alerts)
        active_count = sum(1 for a in st.session_state.active_alerts.values() if a.get('enabled', True))
        paused_count = total_alerts - active_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", total_alerts)
        with col2:
            st.metric("Active", active_count)
        with col3:
            st.metric("Paused", paused_count)

# TAB 2: Create Alert
with tab2:
    st.header("➕ Create Alert")
    st.markdown("Create custom alerts with flexible conditions and multi-channel notifications.")
    
    # Check if editing existing alert or using template
    editing_alert_name = st.session_state.get('editing_alert', None)
    using_template = st.session_state.get('use_template', None)
    template_data = st.session_state.get('template_data', None)
    
    if using_template and template_data:
        st.success(f"📋 Using template: {using_template}")
        # Clear template state after showing message
        if 'use_template' in st.session_state:
            del st.session_state.use_template
        if 'template_data' in st.session_state:
            del st.session_state.template_data
        # Use template as existing alert for pre-population
        existing_alert = template_data.copy()
        existing_alert['alert_type'] = template_data.get('alert_type', 'Price Alert')
        editing_alert_name = None
    elif editing_alert_name and editing_alert_name in st.session_state.active_alerts:
        st.info(f"✏️ Editing alert: {editing_alert_name}")
        existing_alert = st.session_state.active_alerts[editing_alert_name]
    else:
        existing_alert = None
        editing_alert_name = None
    
    # Alert Type Selection
    st.subheader("📋 Alert Type")
    
    alert_type = st.selectbox(
        "Select Alert Type",
        [
            "Price Alert",
            "Technical Indicator Alert",
            "Strategy Signal Alert",
            "Risk Limit Alert",
            "Portfolio Alert",
            "Custom Condition"
        ],
        help="Choose the type of alert to create",
        index=0 if not existing_alert else [
            "Price Alert", "Technical Indicator Alert", "Strategy Signal Alert", 
            "Risk Limit Alert", "Portfolio Alert", "Custom Condition"
        ].index(existing_alert.get('alert_type', 'Price Alert')) if existing_alert.get('alert_type') in [
            "Price Alert", "Technical Indicator Alert", "Strategy Signal Alert", 
            "Risk Limit Alert", "Portfolio Alert", "Custom Condition"
        ] else 0
    )
    
    st.markdown("---")
    
    # Condition Builder (Dynamic based on type)
    st.subheader("🔧 Condition Builder")
    
    condition_config = {}
    
    if alert_type == "Price Alert":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input(
                "Symbol",
                value=existing_alert.get('symbol', 'AAPL') if existing_alert else 'AAPL',
                help="Stock symbol to monitor"
            ).upper()
        
        with col2:
            condition_operator = st.selectbox(
                "Condition",
                [">", "<", ">=", "<=", "=", "crosses above", "crosses below"],
                help="Price condition operator",
                index=0 if not existing_alert else [
                    ">", "<", ">=", "<=", "=", "crosses above", "crosses below"
                ].index(existing_alert.get('operator', '>')) if existing_alert.get('operator') in [
                    ">", "<", ">=", "<=", "=", "crosses above", "crosses below"
                ] else 0
            )
        
        with col3:
            threshold = st.number_input(
                "Threshold Price ($)",
                min_value=0.01,
                value=float(existing_alert.get('threshold', 150.0)) if existing_alert and existing_alert.get('threshold') else 150.0,
                step=0.01,
                help="Price threshold"
            )
        
        condition_config = {
            "symbol": symbol,
            "operator": condition_operator,
            "threshold": threshold
        }
        
        condition_str = f"{symbol} {condition_operator} ${threshold:.2f}"
    
    elif alert_type == "Technical Indicator Alert":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.text_input(
                "Symbol",
                value=existing_alert.get('symbol', 'AAPL') if existing_alert else 'AAPL',
                help="Stock symbol"
            ).upper()
        
        with col2:
            indicator = st.selectbox(
                "Indicator",
                ["RSI", "MACD", "Bollinger Bands", "SMA", "EMA", "Volume"],
                help="Technical indicator",
                index=0 if not existing_alert else [
                    "RSI", "MACD", "Bollinger Bands", "SMA", "EMA", "Volume"
                ].index(existing_alert.get('indicator', 'RSI')) if existing_alert.get('indicator') in [
                    "RSI", "MACD", "Bollinger Bands", "SMA", "EMA", "Volume"
                ] else 0
            )
        
        with col3:
            operator = st.selectbox(
                "Condition",
                [">", "<", ">=", "<=", "crosses above", "crosses below"],
                help="Condition operator"
            )
        
        with col4:
            if indicator == "RSI":
                threshold = st.number_input(
                    "RSI Value",
                    min_value=0,
                    max_value=100,
                    value=int(existing_alert.get('threshold', 70)) if existing_alert and existing_alert.get('threshold') else 70,
                    help="RSI threshold (0-100)"
                )
            else:
                threshold = st.number_input(
                    "Threshold",
                    min_value=0.0,
                    value=float(existing_alert.get('threshold', 0.0)) if existing_alert and existing_alert.get('threshold') else 0.0,
                    step=0.01
                )
        
        condition_config = {
            "symbol": symbol,
            "indicator": indicator,
            "operator": operator,
            "threshold": threshold
        }
        
        condition_str = f"{symbol} {indicator} {operator} {threshold}"
    
    elif alert_type == "Strategy Signal Alert":
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.selectbox(
                "Strategy",
                ["Bollinger Bands", "MACD", "RSI", "SMA Crossover", "Custom Strategy"],
                help="Trading strategy to monitor"
            )
        
        with col2:
            signal_type = st.selectbox(
                "Signal Type",
                ["Buy Signal", "Sell Signal", "Any Signal"],
                help="Type of signal to alert on"
            )
        
        condition_config = {
            "strategy": strategy,
            "signal_type": signal_type
        }
        
        condition_str = f"{strategy} - {signal_type}"
    
    elif alert_type == "Risk Limit Alert":
        col1, col2 = st.columns(2)
        
        with col1:
            risk_metric = st.selectbox(
                "Risk Metric",
                ["Daily Loss", "Drawdown", "VaR", "Position Size", "Leverage"],
                help="Risk metric to monitor"
            )
        
        with col2:
            operator = st.selectbox(
                "Condition",
                [">", "<", ">=", "<="],
                help="Condition operator"
            )
        
        threshold = st.number_input(
            "Threshold",
            min_value=0.0,
            value=float(existing_alert.get('threshold', 1000.0)) if existing_alert and existing_alert.get('threshold') else 1000.0,
            step=0.01,
            help="Risk threshold"
        )
        
        condition_config = {
            "risk_metric": risk_metric,
            "operator": operator,
            "threshold": threshold
        }
        
        condition_str = f"{risk_metric} {operator} {threshold}"
    
    elif alert_type == "Portfolio Alert":
        col1, col2 = st.columns(2)
        
        with col1:
            portfolio_metric = st.selectbox(
                "Portfolio Metric",
                ["Total Value", "Daily Return", "Total Return", "Cash Balance", "Number of Positions"],
                help="Portfolio metric to monitor"
            )
        
        with col2:
            operator = st.selectbox(
                "Condition",
                [">", "<", ">=", "<="],
                help="Condition operator"
            )
        
        threshold = st.number_input(
            "Threshold",
            min_value=0.0,
            value=float(existing_alert.get('threshold', 10000.0)) if existing_alert and existing_alert.get('threshold') else 10000.0,
            step=0.01,
            help="Portfolio threshold"
        )
        
        condition_config = {
            "portfolio_metric": portfolio_metric,
            "operator": operator,
            "threshold": threshold
        }
        
        condition_str = f"Portfolio {portfolio_metric} {operator} {threshold}"
    
    else:  # Custom Condition
        custom_condition = st.text_area(
            "Custom Condition",
            value=existing_alert.get('custom_condition', '') if existing_alert else '',
            height=100,
            help="Enter custom alert condition (Python expression)"
        )
        
        condition_config = {
            "custom_condition": custom_condition
        }
        
        condition_str = custom_condition if custom_condition else "Custom condition"
    
    st.markdown(f"**Condition Preview:** {condition_str}")
    
    st.markdown("---")
    
    # Alert Configuration
    st.subheader("⚙️ Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alert_name = st.text_input(
            "Alert Name",
            value=editing_alert_name if editing_alert_name else f"Alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Unique name for this alert"
        )
        
        frequency = st.selectbox(
            "Frequency",
            ["Once", "Daily", "Continuous"],
            help="How often to trigger this alert",
            index=0 if not existing_alert else [
                "Once", "Daily", "Continuous"
            ].index(existing_alert.get('frequency', 'Once')) if existing_alert.get('frequency') in [
                "Once", "Daily", "Continuous"
            ] else 0
        )
    
    with col2:
        priority = st.selectbox(
            "Priority",
            ["Low", "Medium", "High", "Critical"],
            help="Alert priority level",
            index=1 if not existing_alert else [
                "Low", "Medium", "High", "Critical"
            ].index(existing_alert.get('priority', 'Medium')) if existing_alert.get('priority') in [
                "Low", "Medium", "High", "Critical"
            ] else 1
        )
        
        description = st.text_area(
            "Description (Optional)",
            value=existing_alert.get('description', '') if existing_alert else '',
            height=100,
            help="Additional notes about this alert"
        )
    
    st.markdown("---")
    
    # Notification Settings
    st.subheader("📧 Notification Settings")
    
    st.markdown("**Notification Channels:**")
    
    channels = []
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        use_email = st.checkbox("Email", value=True, help="Send email notifications")
        if use_email:
            channels.append("Email")
    
    with col2:
        use_sms = st.checkbox("SMS", value=False, help="Send SMS notifications")
        if use_sms:
            channels.append("SMS")
    
    with col3:
        use_telegram = st.checkbox("Telegram", value=False, help="Send Telegram notifications")
        if use_telegram:
            channels.append("Telegram")
    
    with col4:
        use_slack = st.checkbox("Slack", value=False, help="Send Slack notifications")
        if use_slack:
            channels.append("Slack")
    
    with col5:
        use_webhook = st.checkbox("Webhook", value=False, help="Send webhook notifications")
        if use_webhook:
            channels.append("Webhook")
    
    if not channels:
        st.warning("⚠️ Please select at least one notification channel")
    
    # Message Template
    st.markdown("**Message Template:**")
    
    message_template = st.text_area(
        "Alert Message",
        value=existing_alert.get('message_template', 'Alert: {condition} triggered at {value}') if existing_alert else 'Alert: {condition} triggered at {value}',
        height=100,
        help="Custom message template. Use {condition}, {value}, {symbol} as placeholders."
    )
    
    # Recipients (for email)
    if use_email:
        email_recipients = st.text_area(
            "Email Recipients (one per line)",
            value='\n'.join(existing_alert.get('email_recipients', [])) if existing_alert and existing_alert.get('email_recipients') else '',
            height=80,
            help="Enter email addresses, one per line"
        )
        email_list = [email.strip() for email in email_recipients.split('\n') if email.strip()] if email_recipients else []
    else:
        email_list = []
    
    st.markdown("---")
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_button = st.button("🧪 Test Alert", use_container_width=True, help="Test the alert configuration")
    
    with col2:
        if editing_alert_name:
            create_button = st.button("💾 Update Alert", type="primary", use_container_width=True)
        else:
            create_button = st.button("✅ Create Alert", type="primary", use_container_width=True)
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()
    
    # Test Alert
    if test_button:
        if not alert_name:
            st.error("Please enter an alert name")
        elif not channels:
            st.error("Please select at least one notification channel")
        else:
            st.success(f"✅ Test alert would be sent via: {', '.join(channels)}")
            st.info(f"📧 Test message: {message_template.format(condition=condition_str, value='$150.00', symbol=symbol if 'symbol' in locals() else 'N/A')}")
    
    # Create/Update Alert
    if create_button:
        if not alert_name:
            st.error("Please enter an alert name")
        elif not channels:
            st.error("Please select at least one notification channel")
        else:
            try:
                # Create alert configuration
                alert_config = {
                    "alert_name": alert_name,
                    "alert_type": alert_type,
                    "condition": condition_str,
                    "condition_config": condition_config,
                    "frequency": frequency,
                    "priority": priority,
                    "description": description,
                    "channels": channels,
                    "message_template": message_template,
                    "email_recipients": email_list if use_email else [],
                    "enabled": True,
                    "created_at": datetime.now().isoformat(),
                    "last_triggered": None,
                    "trigger_count": 0
                }
                
                # Update or create
                if editing_alert_name:
                    # Preserve trigger count if updating
                    if editing_alert_name in st.session_state.active_alerts:
                        alert_config["trigger_count"] = st.session_state.active_alerts[editing_alert_name].get("trigger_count", 0)
                        alert_config["last_triggered"] = st.session_state.active_alerts[editing_alert_name].get("last_triggered")
                    
                    # Remove old name if renaming
                    if alert_name != editing_alert_name and editing_alert_name in st.session_state.active_alerts:
                        del st.session_state.active_alerts[editing_alert_name]
                    
                    st.session_state.active_alerts[alert_name] = alert_config
                    st.success(f"✅ Alert '{alert_name}' updated successfully!")
                    
                    # Clear editing state
                    if 'editing_alert' in st.session_state:
                        del st.session_state.editing_alert
                else:
                    # Check for duplicate name
                    if alert_name in st.session_state.active_alerts:
                        st.error(f"Alert '{alert_name}' already exists. Please choose a different name.")
                    else:
                        st.session_state.active_alerts[alert_name] = alert_config
                        st.success(f"✅ Alert '{alert_name}' created successfully!")
                
                st.rerun()
            
            except Exception as e:
                st.error(f"Error creating alert: {str(e)}")
                logger.exception("Alert creation error")

# TAB 3: Alert Templates
with tab3:
    st.header("📋 Alert Templates")
    st.markdown("Use pre-built alert templates to quickly create common alert configurations.")
    
    # Initialize templates if not exists
    if 'alert_templates' not in st.session_state:
        st.session_state.alert_templates = {
            "Price Breakout": {
                "alert_type": "Price Alert",
                "condition_config": {
                    "operator": "crosses above",
                    "threshold": 0
                },
                "frequency": "Once",
                "priority": "Medium",
                "description": "Alert when price breaks above a resistance level",
                "channels": ["Email"],
                "message_template": "Price breakout alert: {symbol} crossed above ${threshold:.2f}"
            },
            "Price Breakdown": {
                "alert_type": "Price Alert",
                "condition_config": {
                    "operator": "crosses below",
                    "threshold": 0
                },
                "frequency": "Once",
                "priority": "Medium",
                "description": "Alert when price breaks below a support level",
                "channels": ["Email"],
                "message_template": "Price breakdown alert: {symbol} crossed below ${threshold:.2f}"
            },
            "RSI Overbought": {
                "alert_type": "Technical Indicator Alert",
                "condition_config": {
                    "indicator": "RSI",
                    "operator": ">",
                    "threshold": 70
                },
                "frequency": "Daily",
                "priority": "Medium",
                "description": "Alert when RSI indicates overbought conditions",
                "channels": ["Email"],
                "message_template": "RSI Overbought: {symbol} RSI is {value}"
            },
            "RSI Oversold": {
                "alert_type": "Technical Indicator Alert",
                "condition_config": {
                    "indicator": "RSI",
                    "operator": "<",
                    "threshold": 30
                },
                "frequency": "Daily",
                "priority": "Medium",
                "description": "Alert when RSI indicates oversold conditions",
                "channels": ["Email"],
                "message_template": "RSI Oversold: {symbol} RSI is {value}"
            },
            "Daily Loss Limit": {
                "alert_type": "Risk Limit Alert",
                "condition_config": {
                    "risk_metric": "Daily Loss",
                    "operator": ">=",
                    "threshold": 1000
                },
                "frequency": "Daily",
                "priority": "High",
                "description": "Alert when daily loss exceeds threshold",
                "channels": ["Email", "SMS"],
                "message_template": "Risk Alert: Daily loss of ${value} exceeds limit of ${threshold:.2f}"
            },
            "Portfolio Value Drop": {
                "alert_type": "Portfolio Alert",
                "condition_config": {
                    "portfolio_metric": "Total Value",
                    "operator": "<",
                    "threshold": 0
                },
                "frequency": "Daily",
                "priority": "High",
                "description": "Alert when portfolio value drops below threshold",
                "channels": ["Email"],
                "message_template": "Portfolio Alert: Total value is ${value:.2f}"
            },
            "Buy Signal": {
                "alert_type": "Strategy Signal Alert",
                "condition_config": {
                    "strategy": "MACD",
                    "signal_type": "Buy Signal"
                },
                "frequency": "Continuous",
                "priority": "Medium",
                "description": "Alert on buy signals from trading strategies",
                "channels": ["Email"],
                "message_template": "Buy Signal: {strategy} generated a buy signal for {symbol}"
            },
            "Sell Signal": {
                "alert_type": "Strategy Signal Alert",
                "condition_config": {
                    "strategy": "MACD",
                    "signal_type": "Sell Signal"
                },
                "frequency": "Continuous",
                "priority": "Medium",
                "description": "Alert on sell signals from trading strategies",
                "channels": ["Email"],
                "message_template": "Sell Signal: {strategy} generated a sell signal for {symbol}"
            }
        }
    
    # Template Categories
    st.subheader("📚 Template Library")
    
    # Filter templates
    template_filter = st.selectbox(
        "Filter by Type",
        ["All", "Price Alert", "Technical Indicator Alert", "Strategy Signal Alert", "Risk Limit Alert", "Portfolio Alert"],
        help="Filter templates by alert type"
    )
    
    # Display templates
    templates = st.session_state.alert_templates
    
    filtered_templates = {}
    for name, template in templates.items():
        if template_filter == "All" or template.get("alert_type") == template_filter:
            filtered_templates[name] = template
    
    if filtered_templates:
        # Template Grid
        num_cols = 2
        template_names = list(filtered_templates.keys())
        
        for i in range(0, len(template_names), num_cols):
            cols = st.columns(num_cols)
            
            for j, col in enumerate(cols):
                if i + j < len(template_names):
                    template_name = template_names[i + j]
                    template = filtered_templates[template_name]
                    
                    with col:
                        with st.container(border=True):
                            st.markdown(f"### {template_name}")
                            
                            st.markdown(f"**Type:** {template.get('alert_type', 'Unknown')}")
                            
                            if template.get('description'):
                                st.caption(template.get('description'))
                            
                            # Template details
                            with st.expander("📋 Template Details"):
                                st.markdown(f"**Frequency:** {template.get('frequency', 'N/A')}")
                                st.markdown(f"**Priority:** {template.get('priority', 'N/A')}")
                                st.markdown(f"**Channels:** {', '.join(template.get('channels', []))}")
                                
                                if template.get('condition_config'):
                                    st.markdown("**Condition:**")
                                    config = template.get('condition_config')
                                    if 'operator' in config and 'threshold' in config:
                                        st.code(f"{config.get('operator')} {config.get('threshold')}")
                                    elif 'indicator' in config:
                                        st.code(f"{config.get('indicator')} {config.get('operator')} {config.get('threshold')}")
                                    elif 'risk_metric' in config:
                                        st.code(f"{config.get('risk_metric')} {config.get('operator')} {config.get('threshold')}")
                                    elif 'portfolio_metric' in config:
                                        st.code(f"{config.get('portfolio_metric')} {config.get('operator')} {config.get('threshold')}")
                                    elif 'strategy' in config:
                                        st.code(f"{config.get('strategy')} - {config.get('signal_type')}")
                            
                            # Use template button
                            if st.button(f"📝 Use Template", key=f"use_{template_name}", use_container_width=True):
                                st.session_state.use_template = template_name
                                st.session_state.template_data = template
                                st.info(f"Template '{template_name}' selected. Navigate to Tab 2 to configure and create the alert.")
                                st.rerun()
    else:
        st.info("No templates found for the selected filter.")
    
    st.markdown("---")
    
    # Create Custom Template
    st.subheader("➕ Create Custom Template")
    
    with st.expander("Create a new template from an existing alert", expanded=False):
        st.markdown("Save an existing alert configuration as a reusable template.")
        
        if st.session_state.active_alerts:
            template_source = st.selectbox(
                "Select Alert to Save as Template",
                [""] + list(st.session_state.active_alerts.keys()),
                help="Choose an existing alert to convert to a template"
            )
            
            if template_source:
                new_template_name = st.text_input(
                    "Template Name",
                    value=f"Template_{template_source}",
                    help="Name for the new template"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save as Template", use_container_width=True):
                        if new_template_name:
                            if new_template_name in st.session_state.alert_templates:
                                st.error(f"Template '{new_template_name}' already exists. Please choose a different name.")
                            else:
                                source_alert = st.session_state.active_alerts[template_source]
                                
                                # Create template from alert
                                template = {
                                    "alert_type": source_alert.get("alert_type", "Price Alert"),
                                    "condition_config": source_alert.get("condition_config", {}),
                                    "frequency": source_alert.get("frequency", "Once"),
                                    "priority": source_alert.get("priority", "Medium"),
                                    "description": source_alert.get("description", ""),
                                    "channels": source_alert.get("channels", ["Email"]),
                                    "message_template": source_alert.get("message_template", "Alert: {condition} triggered at {value}")
                                }
                                
                                st.session_state.alert_templates[new_template_name] = template
                                st.success(f"✅ Template '{new_template_name}' created successfully!")
                                st.rerun()
                        else:
                            st.error("Please enter a template name")
                
                with col2:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.rerun()
        else:
            st.info("No existing alerts to convert to templates. Create alerts in Tab 2 first.")
    
    # Manage Templates
    st.markdown("---")
    st.subheader("⚙️ Manage Templates")
    
    if st.session_state.alert_templates:
        template_to_manage = st.selectbox(
            "Select Template to Manage",
            [""] + list(st.session_state.alert_templates.keys()),
            help="Choose a template to edit or delete"
        )
        
        if template_to_manage:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Delete Template", use_container_width=True):
                    if template_to_manage in st.session_state.alert_templates:
                        del st.session_state.alert_templates[template_to_manage]
                        st.success(f"✅ Template '{template_to_manage}' deleted!")
                        st.rerun()
            
            with col2:
                if st.button("📋 View Template Details", use_container_width=True):
                    template = st.session_state.alert_templates[template_to_manage]
                    
                    st.markdown(f"### {template_to_manage}")
                    st.json(template)
    
    # Template Statistics
    st.markdown("---")
    st.subheader("📊 Template Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Templates", len(st.session_state.alert_templates))
    
    with col2:
        price_count = sum(1 for t in st.session_state.alert_templates.values() if t.get('alert_type') == 'Price Alert')
        st.metric("Price Alerts", price_count)
    
    with col3:
        indicator_count = sum(1 for t in st.session_state.alert_templates.values() if t.get('alert_type') == 'Technical Indicator Alert')
        st.metric("Indicator Alerts", indicator_count)

# TAB 4: Alert History
with tab4:
    st.header("📜 Alert History")
    st.markdown("View and analyze all alert triggers and notifications.")
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        history_date_from = st.date_input(
            "From Date",
            value=datetime.now().date() - timedelta(days=30),
            help="Start date for history"
        )
    
    with col2:
        history_date_to = st.date_input(
            "To Date",
            value=datetime.now().date(),
            help="End date for history"
        )
    
    with col3:
        history_alert_filter = st.selectbox(
            "Filter by Alert",
            ["All"] + list(st.session_state.active_alerts.keys()) if st.session_state.active_alerts else ["All"],
            help="Filter by specific alert"
        )
    
    with col4:
        history_status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Sent", "Failed"],
            help="Filter by notification status"
        )
    
    # Search
    history_search = st.text_input(
        "Search History",
        placeholder="Search by alert name, condition, or value...",
        help="Search alert history"
    )
    
    st.markdown("---")
    
    # Filter history
    filtered_history = []
    
    for trigger in st.session_state.alert_history:
        # Date filter
        try:
            trigger_date = datetime.fromisoformat(trigger.get('timestamp', '').replace('Z', '+00:00')).date()
            if trigger_date < history_date_from or trigger_date > history_date_to:
                continue
        except (ValueError, AttributeError, KeyError) as e:
            continue
        
        # Alert filter
        if history_alert_filter != "All":
            if trigger.get('alert_name') != history_alert_filter:
                continue
        
        # Status filter
        if history_status_filter != "All":
            notification_sent = trigger.get('notification_sent', False)
            if history_status_filter == "Sent" and not notification_sent:
                continue
            if history_status_filter == "Failed" and notification_sent:
                continue
        
        # Search filter
        if history_search:
            search_lower = history_search.lower()
            alert_name = trigger.get('alert_name', '').lower()
            condition = trigger.get('condition', '').lower()
            value = str(trigger.get('value', '')).lower()
            
            if search_lower not in alert_name and search_lower not in condition and search_lower not in value:
                continue
        
        filtered_history.append(trigger)
    
    # Sort by timestamp (newest first)
    filtered_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    st.subheader("📋 Trigger History")
    
    if filtered_history:
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Triggers", len(filtered_history))
        
        with col2:
            sent_count = sum(1 for t in filtered_history if t.get('notification_sent', False))
            st.metric("Notifications Sent", sent_count)
        
        with col3:
            failed_count = len(filtered_history) - sent_count
            st.metric("Failed", failed_count)
        
        with col4:
            unique_alerts = len(set(t.get('alert_name') for t in filtered_history))
            st.metric("Unique Alerts", unique_alerts)
        
        st.markdown("---")
        
        # History Table
        history_data = []
        
        for trigger in filtered_history:
            timestamp = trigger.get('timestamp', 'Unknown')
            if timestamp != 'Unknown':
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    date_str = dt.strftime('%Y-%m-%d')
                    time_str = dt.strftime('%H:%M:%S')
                except (ValueError, AttributeError) as e:
                    timestamp_str = timestamp
                    date_str = timestamp
                    time_str = ''
            else:
                timestamp_str = timestamp
                date_str = timestamp
                time_str = ''
            
            notification_sent = trigger.get('notification_sent', False)
            status = "✅ Sent" if notification_sent else "❌ Failed"
            
            history_data.append({
                "Timestamp": timestamp_str,
                "Date": date_str,
                "Time": time_str,
                "Alert Name": trigger.get('alert_name', 'Unknown'),
                "Type": trigger.get('alert_type', 'Unknown'),
                "Condition": trigger.get('condition', 'N/A'),
                "Value": trigger.get('value', 'N/A'),
                "Status": status,
                "Action": trigger.get('action_taken', 'N/A')
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Display table
        st.dataframe(
            history_df,
            use_container_width=True,
            height=400,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Alert Name": st.column_config.TextColumn("Alert Name", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Condition": st.column_config.TextColumn("Condition", width="large"),
                "Value": st.column_config.TextColumn("Value", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Action": st.column_config.TextColumn("Action", width="medium")
            }
        )
        
        # Export Options
        st.markdown("---")
        st.subheader("📤 Export History")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export to CSV", use_container_width=True):
                try:
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error exporting to CSV: {str(e)}")
        
        with col2:
            if st.button("📊 Export to Excel", use_container_width=True):
                try:
                    # Create Excel file
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        history_df.to_excel(writer, index=False, sheet_name='Alert History')
                    output.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Excel",
                        data=output,
                        file_name=f"alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error exporting to Excel: {str(e)}")
                    st.info("Note: openpyxl package required for Excel export")
        
        with col3:
            if st.button("🗑️ Clear History", use_container_width=True):
                if st.button("⚠️ Confirm Clear", key="confirm_clear"):
                    st.session_state.alert_history = []
                    st.success("✅ Alert history cleared!")
                    st.rerun()
        
        # Analytics
        st.markdown("---")
        st.subheader("📊 Analytics")
        
        # Trigger frequency chart
        if len(filtered_history) > 0:
            # Daily trigger count
            daily_counts = {}
            for trigger in filtered_history:
                try:
                    dt = datetime.fromisoformat(trigger.get('timestamp', '').replace('Z', '+00:00'))
                    date_key = dt.strftime('%Y-%m-%d')
                    daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
                except (ValueError, AttributeError, KeyError) as e:
                    continue
            
            if daily_counts:
                dates = sorted(daily_counts.keys())
                counts = [daily_counts[d] for d in dates]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=counts,
                    mode='lines+markers',
                    name='Triggers',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title="Daily Alert Triggers",
                    xaxis_title="Date",
                    yaxis_title="Number of Triggers",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Alert type distribution
            col1, col2 = st.columns(2)
            
            with col1:
                alert_type_counts = {}
                for trigger in filtered_history:
                    alert_type = trigger.get('alert_type', 'Unknown')
                    alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1
                
                if alert_type_counts:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(alert_type_counts.keys()),
                        values=list(alert_type_counts.values()),
                        hole=0.3
                    )])
                    fig.update_layout(
                        title="Triggers by Alert Type",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top triggered alerts
                alert_name_counts = {}
                for trigger in filtered_history:
                    alert_name = trigger.get('alert_name', 'Unknown')
                    alert_name_counts[alert_name] = alert_name_counts.get(alert_name, 0) + 1
                
                if alert_name_counts:
                    sorted_alerts = sorted(alert_name_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    fig = go.Figure(data=[go.Bar(
                        x=[count for _, count in sorted_alerts],
                        y=[name for name, _ in sorted_alerts],
                        orientation='h'
                    )])
                    fig.update_layout(
                        title="Top 10 Most Triggered Alerts",
                        xaxis_title="Trigger Count",
                        yaxis_title="Alert Name",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Notification success rate
            st.markdown("---")
            st.subheader("📧 Notification Analytics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_notifications = len(filtered_history)
                successful = sum(1 for t in filtered_history if t.get('notification_sent', False))
                success_rate = (successful / total_notifications * 100) if total_notifications > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col2:
                st.metric("Total Notifications", total_notifications)
            
            with col3:
                st.metric("Successful", successful)
    else:
        st.info("No alert history found. Alerts will appear here when triggered.")
        
        # Show empty state statistics
        st.markdown("---")
        st.subheader("📊 Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Triggers", 0)
        
        with col2:
            st.metric("Notifications Sent", 0)
        
        with col3:
            st.metric("Failed", 0)

# TAB 5: Notification Settings
with tab5:
    st.header("⚙️ Notification Settings")
    st.markdown("Configure notification channels and global notification rules.")
    
    # Initialize notification settings if not exists
    if 'notification_settings' not in st.session_state:
        st.session_state.notification_settings = {
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "smtp_username": "",
                "smtp_password": "",
                "from_address": "",
                "use_tls": True
            },
            "sms": {
                "enabled": False,
                "twilio_account_sid": "",
                "twilio_auth_token": "",
                "twilio_phone_number": "",
                "recipient_phone": ""
            },
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "chat_id": ""
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "#alerts"
            },
            "webhook": {
                "enabled": False,
                "webhook_urls": [],
                "payload_template": '{"alert": "{alert_name}", "condition": "{condition}", "value": "{value}"}'
            },
            "global_rules": {
                "quiet_hours_start": "22:00",
                "quiet_hours_end": "08:00",
                "max_notifications_per_hour": 50,
                "priority_filter": "Medium"
            }
        }
    
    settings = st.session_state.notification_settings
    
    # Email Settings
    with st.expander("📧 Email Settings", expanded=True):
        email_enabled = st.checkbox(
            "Enable Email Notifications",
            value=settings["email"]["enabled"],
            help="Enable email notification channel"
        )
        settings["email"]["enabled"] = email_enabled
        
        if email_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                settings["email"]["smtp_server"] = st.text_input(
                    "SMTP Server",
                    value=settings["email"]["smtp_server"],
                    help="SMTP server address (e.g., smtp.gmail.com)",
                    placeholder="smtp.gmail.com"
                )
                
                settings["email"]["smtp_port"] = st.number_input(
                    "SMTP Port",
                    min_value=1,
                    max_value=65535,
                    value=settings["email"]["smtp_port"],
                    help="SMTP port (usually 587 for TLS, 465 for SSL)"
                )
                
                settings["email"]["smtp_username"] = st.text_input(
                    "SMTP Username",
                    value=settings["email"]["smtp_username"],
                    help="SMTP username (usually your email address)",
                    type="default"
                )
                
                settings["email"]["smtp_password"] = st.text_input(
                    "SMTP Password",
                    value=settings["email"]["smtp_password"],
                    help="SMTP password or app-specific password",
                    type="password"
                )
            
            with col2:
                settings["email"]["from_address"] = st.text_input(
                    "From Address",
                    value=settings["email"]["from_address"],
                    help="Email address to send from",
                    placeholder="alerts@example.com"
                )
                
                settings["email"]["use_tls"] = st.checkbox(
                    "Use TLS",
                    value=settings["email"]["use_tls"],
                    help="Use TLS encryption for SMTP"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test Email", use_container_width=True):
                    if settings["email"]["smtp_server"] and settings["email"]["from_address"]:
                        st.success("✅ Test email sent! (Simulated)")
                        st.info("In production, this would send a test email to verify configuration.")
                    else:
                        st.error("Please configure SMTP server and from address first")
            
            with col2:
                if st.button("💾 Save Email Settings", use_container_width=True):
                    st.session_state.notification_settings["email"] = settings["email"]
                    st.success("✅ Email settings saved!")
    
    # SMS Settings
    with st.expander("📱 SMS Settings", expanded=False):
        sms_enabled = st.checkbox(
            "Enable SMS Notifications",
            value=settings["sms"]["enabled"],
            help="Enable SMS notification channel (requires Twilio)"
        )
        settings["sms"]["enabled"] = sms_enabled
        
        if sms_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                settings["sms"]["twilio_account_sid"] = st.text_input(
                    "Twilio Account SID",
                    value=settings["sms"]["twilio_account_sid"],
                    help="Your Twilio Account SID",
                    type="default"
                )
                
                settings["sms"]["twilio_auth_token"] = st.text_input(
                    "Twilio Auth Token",
                    value=settings["sms"]["twilio_auth_token"],
                    help="Your Twilio Auth Token",
                    type="password"
                )
            
            with col2:
                settings["sms"]["twilio_phone_number"] = st.text_input(
                    "Twilio Phone Number",
                    value=settings["sms"]["twilio_phone_number"],
                    help="Your Twilio phone number (e.g., +1234567890)",
                    placeholder="+1234567890"
                )
                
                settings["sms"]["recipient_phone"] = st.text_input(
                    "Recipient Phone Number",
                    value=settings["sms"]["recipient_phone"],
                    help="Phone number to receive SMS alerts",
                    placeholder="+1234567890"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test SMS", use_container_width=True):
                    if settings["sms"]["twilio_account_sid"] and settings["sms"]["recipient_phone"]:
                        st.success("✅ Test SMS sent! (Simulated)")
                        st.info("In production, this would send a test SMS via Twilio.")
                    else:
                        st.error("Please configure Twilio credentials and recipient phone number first")
            
            with col2:
                if st.button("💾 Save SMS Settings", use_container_width=True):
                    st.session_state.notification_settings["sms"] = settings["sms"]
                    st.success("✅ SMS settings saved!")
    
    # Telegram Settings
    with st.expander("💬 Telegram Settings", expanded=False):
        telegram_enabled = st.checkbox(
            "Enable Telegram Notifications",
            value=settings["telegram"]["enabled"],
            help="Enable Telegram notification channel"
        )
        settings["telegram"]["enabled"] = telegram_enabled
        
        if telegram_enabled:
            settings["telegram"]["bot_token"] = st.text_input(
                "Bot Token",
                value=settings["telegram"]["bot_token"],
                help="Your Telegram bot token (get from @BotFather)",
                type="password"
            )
            
            settings["telegram"]["chat_id"] = st.text_input(
                "Chat ID",
                value=settings["telegram"]["chat_id"],
                help="Telegram chat ID to send messages to",
                placeholder="123456789"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test Telegram", use_container_width=True):
                    if settings["telegram"]["bot_token"] and settings["telegram"]["chat_id"]:
                        st.success("✅ Test Telegram message sent! (Simulated)")
                        st.info("In production, this would send a test message via Telegram bot.")
                    else:
                        st.error("Please configure bot token and chat ID first")
            
            with col2:
                if st.button("💾 Save Telegram Settings", use_container_width=True):
                    st.session_state.notification_settings["telegram"] = settings["telegram"]
                    st.success("✅ Telegram settings saved!")
    
    # Slack Settings
    with st.expander("💼 Slack Settings", expanded=False):
        slack_enabled = st.checkbox(
            "Enable Slack Notifications",
            value=settings["slack"]["enabled"],
            help="Enable Slack notification channel"
        )
        settings["slack"]["enabled"] = slack_enabled
        
        if slack_enabled:
            settings["slack"]["webhook_url"] = st.text_input(
                "Webhook URL",
                value=settings["slack"]["webhook_url"],
                help="Slack webhook URL (get from Slack app settings)",
                type="default",
                placeholder="https://hooks.slack.com/services/..."
            )
            
            settings["slack"]["channel"] = st.text_input(
                "Channel",
                value=settings["slack"]["channel"],
                help="Slack channel to post to (e.g., #alerts)",
                placeholder="#alerts"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test Slack", use_container_width=True):
                    if settings["slack"]["webhook_url"]:
                        st.success("✅ Test Slack notification sent! (Simulated)")
                        st.info("In production, this would send a test message to Slack.")
                    else:
                        st.error("Please configure webhook URL first")
            
            with col2:
                if st.button("💾 Save Slack Settings", use_container_width=True):
                    st.session_state.notification_settings["slack"] = settings["slack"]
                    st.success("✅ Slack settings saved!")
    
    # Webhook Settings
    with st.expander("🔗 Webhook Settings", expanded=False):
        webhook_enabled = st.checkbox(
            "Enable Webhook Notifications",
            value=settings["webhook"]["enabled"],
            help="Enable custom webhook notifications"
        )
        settings["webhook"]["enabled"] = webhook_enabled
        
        if webhook_enabled:
            st.markdown("**Webhook URLs (one per line):**")
            webhook_urls_text = st.text_area(
                "Webhook URLs",
                value='\n'.join(settings["webhook"]["webhook_urls"]) if settings["webhook"]["webhook_urls"] else '',
                height=100,
                help="Enter webhook URLs, one per line"
            )
            settings["webhook"]["webhook_urls"] = [url.strip() for url in webhook_urls_text.split('\n') if url.strip()]
            
            settings["webhook"]["payload_template"] = st.text_area(
                "Payload Template (JSON)",
                value=settings["webhook"]["payload_template"],
                height=150,
                help="JSON template for webhook payload. Use {alert_name}, {condition}, {value} as placeholders."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test Webhook", use_container_width=True):
                    if settings["webhook"]["webhook_urls"]:
                        st.success("✅ Test webhook sent! (Simulated)")
                        st.info("In production, this would POST to the configured webhook URLs.")
                    else:
                        st.error("Please configure at least one webhook URL first")
            
            with col2:
                if st.button("💾 Save Webhook Settings", use_container_width=True):
                    st.session_state.notification_settings["webhook"] = settings["webhook"]
                    st.success("✅ Webhook settings saved!")
    
    st.markdown("---")
    
    # Global Notification Rules
    st.subheader("🌐 Global Notification Rules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Quiet Hours:**")
        quiet_start = st.time_input(
            "Start Time",
            value=datetime.strptime(settings["global_rules"]["quiet_hours_start"], "%H:%M").time(),
            help="Notifications will be suppressed during quiet hours"
        )
        settings["global_rules"]["quiet_hours_start"] = quiet_start.strftime("%H:%M")
        
        quiet_end = st.time_input(
            "End Time",
            value=datetime.strptime(settings["global_rules"]["quiet_hours_end"], "%H:%M").time(),
            help="End of quiet hours period"
        )
        settings["global_rules"]["quiet_hours_end"] = quiet_end.strftime("%H:%M")
    
    with col2:
        settings["global_rules"]["max_notifications_per_hour"] = st.number_input(
            "Max Notifications Per Hour",
            min_value=1,
            max_value=1000,
            value=settings["global_rules"]["max_notifications_per_hour"],
            help="Rate limit for notifications"
        )
        
        settings["global_rules"]["priority_filter"] = st.selectbox(
            "Minimum Priority",
            ["Low", "Medium", "High", "Critical"],
            index=["Low", "Medium", "High", "Critical"].index(settings["global_rules"]["priority_filter"]),
            help="Only send notifications for alerts at or above this priority"
        )
    
    if st.button("💾 Save Global Rules", use_container_width=True):
        st.session_state.notification_settings["global_rules"] = settings["global_rules"]
        st.success("✅ Global notification rules saved!")
    
    # Settings Summary
    st.markdown("---")
    st.subheader("📊 Settings Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Email", "✅ Enabled" if settings["email"]["enabled"] else "❌ Disabled")
    
    with col2:
        st.metric("SMS", "✅ Enabled" if settings["sms"]["enabled"] else "❌ Disabled")
    
    with col3:
        st.metric("Telegram", "✅ Enabled" if settings["telegram"]["enabled"] else "❌ Disabled")
    
    with col4:
        st.metric("Slack", "✅ Enabled" if settings["slack"]["enabled"] else "❌ Disabled")
    
    with col5:
        st.metric("Webhook", "✅ Enabled" if settings["webhook"]["enabled"] else "❌ Disabled")

