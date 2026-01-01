"""
Alerts Management Page

This page provides comprehensive alert management functionality including:
- Alert configuration
- Test preview of alert formatting
- Strategy-specific alert settings
- Multi-channel notification management
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the system directory to the path for imports
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "system", "infra", "agents")
)


def main():
    """Main function for the Alerts page."""
    st.set_page_config(page_title="Alerts Management", page_icon="🔔", layout="wide")

    st.title("🔔 Alerts Management")
    st.markdown("Configure and manage trading alerts and notifications")

    # Initialize session state
    if "alert_settings" not in st.session_state:
        st.session_state.alert_settings = load_alert_settings()

    # Sidebar for navigation
    with st.sidebar:
        st.header("Alert Configuration")

        alert_section = st.selectbox(
            "Alert Section",
            ["General Settings", "Strategy Alerts", "Test Preview", "Alert History"],
        )

    # Main content based on selection
    if alert_section == "General Settings":
        render_general_settings()
    elif alert_section == "Strategy Alerts":
        render_strategy_alerts()
    elif alert_section == "Test Preview":
        render_test_preview()
    elif alert_section == "Alert History":
        render_alert_history()


def load_alert_settings():
    """Load alert settings from configuration."""
    try:
        settings_path = Path("config/settings.json")
        if settings_path.exists():
            with open(settings_path, "r") as f:
                settings = json.load(f)
                return settings.get("alerts", {})
        else:
            return get_default_alert_settings()
    except Exception as e:
        st.error(f"Error loading alert settings: {e}")
        return get_default_alert_settings()


def get_default_alert_settings():
    """Get default alert settings."""
    return {
        "email": {
            "enabled": True,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "",
            "recipient_email": "",
            "use_tls": True,
        },
        "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
        "slack": {"enabled": False, "webhook_url": ""},
        "thresholds": {
            "model_performance": 0.8,
            "prediction_confidence": 0.7,
            "portfolio_drawdown": 0.1,
            "risk_level": 0.05,
        },
        "strategies": {
            "default": {
                "email_alerts": True,
                "telegram_alerts": True,
                "slack_alerts": False,
            }
        },
    }


def render_general_settings():
    """Render general alert settings."""
    st.header("📧 General Alert Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Email Configuration")

        email_enabled = st.checkbox(
            "Enable Email Alerts",
            value=st.session_state.alert_settings.get("email", {}).get("enabled", True),
        )

        if email_enabled:
            smtp_server = st.text_input(
                "SMTP Server",
                value=st.session_state.alert_settings.get("email", {}).get(
                    "smtp_server", "smtp.gmail.com"
                ),
            )

            smtp_port = st.number_input(
                "SMTP Port",
                value=st.session_state.alert_settings.get("email", {}).get(
                    "smtp_port", 587
                ),
                min_value=1,
                max_value=65535,
            )

            sender_email = st.text_input(
                "Sender Email",
                value=st.session_state.alert_settings.get("email", {}).get(
                    "sender_email", ""
                ),
                type="default",
            )

            recipient_email = st.text_input(
                "Recipient Email",
                value=st.session_state.alert_settings.get("email", {}).get(
                    "recipient_email", ""
                ),
                type="default",
            )

            use_tls = st.checkbox(
                "Use TLS",
                value=st.session_state.alert_settings.get("email", {}).get(
                    "use_tls", True
                ),
            )

            # Update settings
            if "email" not in st.session_state.alert_settings:
                st.session_state.alert_settings["email"] = {}

            st.session_state.alert_settings["email"].update(
                {
                    "enabled": email_enabled,
                    "smtp_server": smtp_server,
                    "smtp_port": smtp_port,
                    "sender_email": sender_email,
                    "recipient_email": recipient_email,
                    "use_tls": use_tls,
                }
            )

    with col2:
        st.subheader("Telegram Configuration")

        telegram_enabled = st.checkbox(
            "Enable Telegram Alerts",
            value=st.session_state.alert_settings.get("telegram", {}).get(
                "enabled", False
            ),
        )

        if telegram_enabled:
            bot_token = st.text_input(
                "Bot Token",
                value=st.session_state.alert_settings.get("telegram", {}).get(
                    "bot_token", ""
                ),
                type="password",
            )

            chat_id = st.text_input(
                "Chat ID",
                value=st.session_state.alert_settings.get("telegram", {}).get(
                    "chat_id", ""
                ),
                type="default",
            )

            # Update settings
            if "telegram" not in st.session_state.alert_settings:
                st.session_state.alert_settings["telegram"] = {}

            st.session_state.alert_settings["telegram"].update(
                {
                    "enabled": telegram_enabled,
                    "bot_token": bot_token,
                    "chat_id": chat_id,
                }
            )

    # Alert thresholds
    st.subheader("Alert Thresholds")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        model_performance = st.slider(
            "Model Performance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.alert_settings.get("thresholds", {}).get(
                "model_performance", 0.8
            ),
            step=0.05,
        )

    with col2:
        prediction_confidence = st.slider(
            "Prediction Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.alert_settings.get("thresholds", {}).get(
                "prediction_confidence", 0.7
            ),
            step=0.05,
        )

    with col3:
        portfolio_drawdown = st.slider(
            "Portfolio Drawdown Threshold",
            min_value=0.0,
            max_value=0.5,
            value=st.session_state.alert_settings.get("thresholds", {}).get(
                "portfolio_drawdown", 0.1
            ),
            step=0.01,
        )

    with col4:
        risk_level = st.slider(
            "Risk Level Threshold",
            min_value=0.0,
            max_value=0.2,
            value=st.session_state.alert_settings.get("thresholds", {}).get(
                "risk_level", 0.05
            ),
            step=0.01,
        )

    # Update thresholds
    if "thresholds" not in st.session_state.alert_settings:
        st.session_state.alert_settings["thresholds"] = {}

    st.session_state.alert_settings["thresholds"].update(
        {
            "model_performance": model_performance,
            "prediction_confidence": prediction_confidence,
            "portfolio_drawdown": portfolio_drawdown,
            "risk_level": risk_level,
        }
    )

    # Save button
    if st.button("💾 Save Settings"):
        save_alert_settings()
        st.success("Alert settings saved successfully!")


def render_strategy_alerts():
    """Render strategy-specific alert settings."""
    st.header("🎯 Strategy-Specific Alerts")

    # Get available strategies
    strategies = get_available_strategies()

    if not strategies:
        st.info("No strategies found. Create some strategies first.")
        return

    # Strategy selection
    selected_strategy = st.selectbox("Select Strategy", strategies)

    if selected_strategy:
        # Get current settings for selected strategy
        strategy_settings = st.session_state.alert_settings.get("strategies", {}).get(
            selected_strategy, {}
        )

        st.subheader(f"Alert Settings for {selected_strategy}")

        col1, col2, col3 = st.columns(3)

        with col1:
            email_alerts = st.checkbox(
                "Email Alerts", value=strategy_settings.get("email_alerts", True)
            )

        with col2:
            telegram_alerts = st.checkbox(
                "Telegram Alerts", value=strategy_settings.get("telegram_alerts", True)
            )

        with col3:
            slack_alerts = st.checkbox(
                "Slack Alerts", value=strategy_settings.get("slack_alerts", False)
            )

        # Update strategy settings
        if "strategies" not in st.session_state.alert_settings:
            st.session_state.alert_settings["strategies"] = {}

        st.session_state.alert_settings["strategies"][selected_strategy] = {
            "email_alerts": email_alerts,
            "telegram_alerts": telegram_alerts,
            "slack_alerts": slack_alerts,
        }

        # Save button
        if st.button("💾 Save Strategy Settings"):
            save_alert_settings()
            st.success(f"Alert settings for {selected_strategy} saved successfully!")


def render_test_preview():
    """Render test preview section."""
    st.header("🧪 Test Alert Preview")

    # Sample alert data
    sample_alerts = {
        "trading_signal": {
            "symbol": "AAPL",
            "action": "BUY",
            "price": 150.25,
            "quantity": 100,
            "confidence": 0.85,
            "timestamp": datetime.now(),
        },
        "performance_alert": {
            "metric": "Sharpe Ratio",
            "value": 1.75,
            "threshold": 1.5,
            "status": "above",
        },
        "system_alert": {
            "component": "Data Feed",
            "message": "Connection timeout detected",
            "alert_type": "warning",
        },
    }

    # Alert type selection
    alert_type = st.selectbox(
        "Select Alert Type to Preview", list(sample_alerts.keys())
    )

    if alert_type == "trading_signal":
        render_trading_signal_preview(sample_alerts["trading_signal"])
    elif alert_type == "performance_alert":
        render_performance_alert_preview(sample_alerts["performance_alert"])
    elif alert_type == "system_alert":
        render_system_alert_preview(sample_alerts["system_alert"])

    # Test send button
    st.subheader("Send Test Alert")

    col1, col2 = st.columns(2)

    with col1:
        test_channel = st.selectbox(
            "Test Channel", ["Email", "Telegram", "All Channels"]
        )

    with col2:
        if st.button("📤 Send Test Alert"):
            send_test_alert(alert_type, test_channel)


def render_trading_signal_preview(alert_data):
    """Render trading signal alert preview."""
    st.subheader("📊 Trading Signal Alert Preview")

    # Create preview card
    action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(alert_data["action"], "📊")

    preview_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px 0; background-color: #f9f9f9;">
        <h3>{action_emoji} Trading Signal: {alert_data['symbol']}</h3>
        <p><strong>Action:</strong> {alert_data['action']}</p>
        <p><strong>Price:</strong> ${alert_data['price']:.2f}</p>
        <p><strong>Quantity:</strong> {alert_data['quantity']}</p>
        <p><strong>Confidence:</strong> {alert_data['confidence']:.1%}</p>
        <p><strong>Time:</strong> {alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """

    st.markdown(preview_html, unsafe_allow_html=True)


def render_performance_alert_preview(alert_data):
    """Render performance alert preview."""
    st.subheader("📈 Performance Alert Preview")

    status_emoji = "🟢" if alert_data["status"] == "above" else "🔴"

    preview_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px 0; background-color: #f9f9f9;">
        <h3>{status_emoji} Performance Alert: {alert_data['metric']}</h3>
        <p><strong>Current Value:</strong> {alert_data['value']:.2f}</p>
        <p><strong>Threshold:</strong> {alert_data['threshold']:.2f}</p>
        <p><strong>Status:</strong> {alert_data['status'].title()} threshold</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """

    st.markdown(preview_html, unsafe_allow_html=True)


def render_system_alert_preview(alert_data):
    """Render system alert preview."""
    st.subheader("⚙️ System Alert Preview")

    alert_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}.get(
        alert_data["alert_type"], "📢"
    )

    preview_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px 0; background-color: #f9f9f9;">
        <h3>{alert_emoji} System Alert: {alert_data['component']}</h3>
        <p><strong>Message:</strong> {alert_data['message']}</p>
        <p><strong>Type:</strong> {alert_data['alert_type'].title()}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """

    st.markdown(preview_html, unsafe_allow_html=True)


def render_alert_history():
    """Render alert history."""
    st.header("📋 Alert History")

    # Mock alert history data
    alert_history = get_alert_history()

    if not alert_history:
        st.info("No alert history available.")
        return

    # Convert to DataFrame for display
    df = pd.DataFrame(alert_history)

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        alert_type_filter = st.selectbox(
            "Filter by Type", ["All"] + list(df["type"].unique())
        )

    with col2:
        status_filter = st.selectbox(
            "Filter by Status", ["All"] + list(df["status"].unique())
        )

    with col3:
        date_filter = st.date_input("Filter by Date", value=datetime.now().date())

    _unused_var = date_filter  # Placeholder, flake8 ignore: F841

    # Apply filters
    filtered_df = df.copy()

    if alert_type_filter != "All":
        filtered_df = filtered_df[filtered_df["type"] == alert_type_filter]

    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["status"] == status_filter]

    # Display filtered results
    st.dataframe(filtered_df, use_container_width=True)

    # Summary statistics
    st.subheader("Alert Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Alerts", len(df))

    with col2:
        st.metric("Successful", len(df[df["status"] == "sent"]))

    with col3:
        st.metric("Failed", len(df[df["status"] == "failed"]))

    with col4:
        success_rate = (
            len(df[df["status"] == "sent"]) / len(df) * 100 if len(df) > 0 else 0
        )
        st.metric("Success Rate", f"{success_rate:.1f}%")


def get_available_strategies():
    """Get list of available strategies."""
    # Mock data - in real implementation, this would load from strategy files
    return ["Momentum Strategy", "Mean Reversion", "Breakout Strategy", "Arbitrage"]


def get_alert_history():
    """Get alert history data."""
    # Mock data - in real implementation, this would load from database or log files
    return [
        {
            "timestamp": datetime.now() - timedelta(hours=1),
            "type": "trading_signal",
            "message": "BUY signal for AAPL",
            "status": "sent",
            "channel": "email",
        },
        {
            "timestamp": datetime.now() - timedelta(hours=2),
            "type": "performance_alert",
            "message": "Sharpe ratio below threshold",
            "status": "sent",
            "channel": "telegram",
        },
        {
            "timestamp": datetime.now() - timedelta(hours=3),
            "type": "system_alert",
            "message": "Data feed connection restored",
            "status": "failed",
            "channel": "email",
        },
    ]


def save_alert_settings():
    """Save alert settings to configuration file."""
    try:
        settings_path = Path("config/settings.json")

        # Load existing settings or create new
        if settings_path.exists():
            with open(settings_path, "r") as f:
                settings = json.load(f)
        else:
            settings = {}

        # Update with alert settings
        settings["alerts"] = st.session_state.alert_settings

        # Save to file
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

    except Exception as e:
        st.error(f"Error saving alert settings: {e}")


def send_test_alert(alert_type, channel):
    """Send a test alert."""
    try:
        # This would integrate with the actual alert manager
        st.success(f"Test {alert_type} alert sent via {channel}!")

        # In real implementation, this would call the alert manager
        # alert_manager = AlertManager()
        # alert_manager.send_test_alert(alert_type, channel)

    except Exception as e:
        st.error(f"Error sending test alert: {e}")


if __name__ == "__main__":
    main()
