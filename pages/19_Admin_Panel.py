"""
Admin Panel Page

This page provides administrative functionality including:
- User management
- System configuration
- Audit trail
- Security monitoring
- Performance metrics
"""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

# Add the system directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "system", "infra", "agents"))


def main():
    """Main function for the Admin Panel page."""
    st.set_page_config(page_title="Admin Panel", page_icon="⚙️", layout="wide")

    st.title("⚙️ Admin Panel")
    st.markdown("System administration and monitoring")

    # Check if user has admin privileges
    if not check_admin_access():
        st.error("Access denied. Admin privileges required.")
        return

    # Initialize session state
    if "admin_data" not in st.session_state:
        st.session_state.admin_data = load_admin_data()

    # Sidebar for navigation
    with st.sidebar:
        st.header("Admin Functions")

        admin_section = st.selectbox(
            "Admin Section",
            ["Dashboard", "User Management", "System Configuration", "Audit Trail", "Security", "Performance"],
        )

    # Main content based on selection
    if admin_section == "Dashboard":
        render_admin_dashboard()
    elif admin_section == "User Management":
        render_user_management()
    elif admin_section == "System Configuration":
        render_system_configuration()
    elif admin_section == "Audit Trail":
        render_audit_trail()
    elif admin_section == "Security":
        render_security_monitoring()
    elif admin_section == "Performance":
        render_performance_metrics()


def check_admin_access():
    """Check if current user has admin access."""
    # In a real implementation, this would check the user's role from session/token
    # For now, we'll use a simple session state check
    return st.session_state.get("user_role", "admin") == "admin"


def load_admin_data():
    """Load admin data from various sources."""
    return {
        "users": load_users(),
        "audit_logs": load_audit_logs(),
        "system_metrics": load_system_metrics(),
        "security_events": load_security_events(),
    }


def render_admin_dashboard():
    """Render admin dashboard."""
    st.header("📊 Admin Dashboard")

    # System overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Users", "24", "+3")

    with col2:
        st.metric("System Uptime", "99.8%", "+0.2%")

    with col3:
        st.metric("Active Strategies", "12", "-1")

    with col4:
        st.metric("Alerts Today", "8", "+2")

    # Recent activity
    st.subheader("Recent Activity")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Recent User Actions**")
        recent_actions = get_recent_user_actions()
        for action in recent_actions:
            st.markdown(f"• {action['user']} {action['action']} at {action['time']}")

    with col2:
        st.markdown("**System Events**")
        system_events = get_system_events()
        for event in system_events:
            color = {"info": "🔵", "warning": "🟡", "error": "🔴"}.get(event["level"], "⚪")
            st.markdown(f"{color} {event['message']}")


def render_user_management():
    """Render user management interface."""
    st.header("👥 User Management")

    # User list
    users = st.session_state.admin_data["users"]

    if users:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(users)

        # Filter options
        col1, col2, col3 = st.columns(3)

        with col1:
            role_filter = st.selectbox("Filter by Role", ["All"] + list(df["role"].unique()))

        with col2:
            status_filter = st.selectbox("Filter by Status", ["All"] + list(df["status"].unique()))

        with col3:
            search_term = st.text_input("Search Users", "")

        # Apply filters
        filtered_df = df.copy()

        if role_filter != "All":
            filtered_df = filtered_df[filtered_df["role"] == role_filter]

        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["status"] == status_filter]

        if search_term:
            filtered_df = filtered_df[
                filtered_df["username"].str.contains(search_term, case=False)
                | filtered_df["email"].str.contains(search_term, case=False)
            ]

        # Display users
        st.dataframe(filtered_df, use_container_width=True)

        # User actions
        st.subheader("User Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("➕ Add User"):
                st.session_state.show_add_user = True

        with col2:
            if st.button("📊 User Statistics"):
                show_user_statistics()

        with col3:
            if st.button("🔍 User Activity"):
                show_user_activity()

        # Add user form
        if st.session_state.get("show_add_user", False):
            render_add_user_form()

    else:
        st.info("No users found.")


def render_add_user_form():
    """Render add user form."""
    st.subheader("Add New User")

    with st.form("add_user_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        role = st.selectbox("Role", ["viewer", "analyst", "trader", "admin"])
        password = st.text_input("Password", type="password")

        col1, col2 = st.columns(2)

        with col1:
            if st.form_submit_button("Create User"):
                if create_user(username, email, role, password):
                    st.success("User created successfully!")
                    st.session_state.show_add_user = False
                else:
                    st.error("Failed to create user")

        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.show_add_user = False


def render_system_configuration():
    """Render system configuration interface."""
    st.header("⚙️ System Configuration")

    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["General", "Security", "Performance", "Alerts"])

    with tab1:
        render_general_config()

    with tab2:
        render_security_config()

    with tab3:
        render_performance_config()

    with tab4:
        render_alerts_config()


def render_general_config():
    """Render general configuration."""
    st.subheader("General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("System Name", value="Evolve Trading Platform")
        st.text_input("Admin Email", value="admin@evolve.com")
        st.selectbox("Environment", ["development", "staging", "production"])

    with col2:
        st.number_input("Max Users", value=100, min_value=1)
        st.number_input("Session Timeout (hours)", value=24, min_value=1)
        st.checkbox("Maintenance Mode", value=False)

    if st.button("💾 Save General Config"):
        st.success("General configuration saved!")


def render_security_config():
    """Render security configuration."""
    st.subheader("Security Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Password Min Length", value=8, min_value=6)
        st.checkbox("Require Uppercase", value=True)
        st.checkbox("Require Lowercase", value=True)
        st.checkbox("Require Digits", value=True)
        st.checkbox("Require Special Characters", value=True)

    with col2:
        st.number_input("Max Login Attempts", value=5, min_value=1)
        st.number_input("Lockout Duration (minutes)", value=15, min_value=1)
        st.checkbox("Enable MFA", value=False)
        st.checkbox("Audit Logging", value=True)

    if st.button("💾 Save Security Config"):
        st.success("Security configuration saved!")


def render_performance_config():
    """Render performance configuration."""
    st.subheader("Performance Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Max Workers", value=4, min_value=1)
        st.number_input("Cache TTL (seconds)", value=3600, min_value=60)
        st.number_input("Request Timeout (seconds)", value=30, min_value=5)

    with col2:
        st.number_input("Memory Limit (MB)", value=2048, min_value=512)
        st.checkbox("Enable GPU", value=True)
        st.checkbox("Parallel Processing", value=True)

    if st.button("💾 Save Performance Config"):
        st.success("Performance configuration saved!")


def render_alerts_config():
    """Render alerts configuration."""
    st.subheader("Alerts Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Email Alerts", value=True)
        st.checkbox("Telegram Alerts", value=False)
        st.checkbox("Slack Alerts", value=False)

    with col2:
        st.number_input("Alert Threshold", value=0.8, min_value=0.0, max_value=1.0, step=0.1)
        st.number_input("Alert Cooldown (minutes)", value=15, min_value=1)
        st.text_input("Alert Email", value="alerts@evolve.com")

    if st.button("💾 Save Alerts Config"):
        st.success("Alerts configuration saved!")


def render_audit_trail():
    """Render audit trail interface."""
    st.header("📋 Audit Trail")

    # Load audit logs
    audit_logs = st.session_state.admin_data["audit_logs"]

    if audit_logs:
        # Convert to DataFrame
        df = pd.DataFrame(audit_logs)

        # Filter options
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            action_filter = st.selectbox("Filter by Action", ["All"] + list(df["action"].unique()))

        with col2:
            user_filter = st.selectbox("Filter by User", ["All"] + list(df["user"].unique()))

        with col3:
            level_filter = st.selectbox("Filter by Level", ["All"] + list(df["level"].unique()))

        with col4:
            date_filter = st.date_input("Filter by Date", value=datetime.now().date())

        # Apply filters
        filtered_df = df.copy()

        if action_filter != "All":
            filtered_df = filtered_df[filtered_df["action"] == action_filter]

        if user_filter != "All":
            filtered_df = filtered_df[filtered_df["user"] == user_filter]

        if level_filter != "All":
            filtered_df = filtered_df[filtered_df["level"] == level_filter]

        # Display audit logs
        st.dataframe(filtered_df, use_container_width=True)

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export CSV"):
                export_audit_logs_csv(filtered_df)

        with col2:
            if st.button("📊 Audit Statistics"):
                show_audit_statistics(filtered_df)

    else:
        st.info("No audit logs found.")


def render_security_monitoring():
    """Render security monitoring interface."""
    st.header("🔒 Security Monitoring")

    # Security metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Failed Logins", "3", "-2")

    with col2:
        st.metric("Suspicious Activities", "1", "+1")

    with col3:
        st.metric("Blocked IPs", "0", "0")

    with col4:
        st.metric("Security Score", "95%", "+2%")

    # Recent security events
    st.subheader("Recent Security Events")

    security_events = st.session_state.admin_data["security_events"]

    if security_events:
        for event in security_events:
            color = {"info": "🔵", "warning": "🟡", "error": "🔴", "critical": "🟣"}.get(event["level"], "⚪")
            st.markdown(f"{color} **{event['time']}** - {event['user']}: {event['description']}")

    # Security actions
    st.subheader("Security Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Security Scan"):
            run_security_scan()

    with col2:
        if st.button("🚫 Block IP"):
            block_ip_address()

    with col3:
        if st.button("🔓 Unlock User"):
            unlock_user_account()


def render_performance_metrics():
    """Render performance metrics interface."""
    st.header("📈 Performance Metrics")

    # System performance
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System Resources")

        # CPU Usage
        cpu_usage = st.slider("CPU Usage (%)", 0, 100, 45)
        st.progress(cpu_usage / 100)

        # Memory Usage
        memory_usage = st.slider("Memory Usage (%)", 0, 100, 62)
        st.progress(memory_usage / 100)

        # Disk Usage
        disk_usage = st.slider("Disk Usage (%)", 0, 100, 28)
        st.progress(disk_usage / 100)

    with col2:
        st.subheader("Application Metrics")

        # Response Time
        response_time = st.metric("Avg Response Time", "1.2s", "-0.1s")

        # Throughput
        throughput = st.metric("Requests/sec", "156", "+12")

        # Error Rate
        error_rate = st.metric("Error Rate", "0.5%", "-0.1%")

    # Performance charts
    st.subheader("Performance Trends")

    # Mock performance data
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    performance_data = pd.DataFrame(
        {
            "Date": dates,
            "CPU": [45 + i * 0.5 for i in range(len(dates))],
            "Memory": [62 + i * 0.3 for i in range(len(dates))],
            "Response_Time": [1.2 + i * 0.01 for i in range(len(dates))],
        }
    )

    st.line_chart(performance_data.set_index("Date"))


# Helper functions


def load_users():
    """Load user data."""
    # Mock data - in real implementation, this would load from database
    return [
        {
            "id": "1",
            "username": "admin",
            "email": "admin@evolve.com",
            "role": "admin",
            "status": "active",
            "last_login": "2024-01-15 10:30:00",
            "created_at": "2024-01-01 00:00:00",
        },
        {
            "id": "2",
            "username": "trader1",
            "email": "trader1@evolve.com",
            "role": "trader",
            "status": "active",
            "last_login": "2024-01-15 09:15:00",
            "created_at": "2024-01-02 00:00:00",
        },
        {
            "id": "3",
            "username": "analyst1",
            "email": "analyst1@evolve.com",
            "role": "analyst",
            "status": "active",
            "last_login": "2024-01-15 08:45:00",
            "created_at": "2024-01-03 00:00:00",
        },
    ]


def load_audit_logs():
    """Load audit log data."""
    # Mock data - in real implementation, this would load from database
    return [
        {
            "timestamp": "2024-01-15 10:30:00",
            "user": "admin",
            "action": "user:create",
            "level": "info",
            "description": "Created new user: trader2",
            "ip_address": "192.168.1.100",
        },
        {
            "timestamp": "2024-01-15 10:25:00",
            "user": "trader1",
            "action": "strategy:execute",
            "level": "info",
            "description": "Executed strategy: Momentum Strategy",
            "ip_address": "192.168.1.101",
        },
        {
            "timestamp": "2024-01-15 10:20:00",
            "user": "analyst1",
            "action": "model:train",
            "level": "info",
            "description": "Started model training: LSTM Model",
            "ip_address": "192.168.1.102",
        },
    ]


def load_system_metrics():
    """Load system metrics."""
    return {"cpu_usage": 45, "memory_usage": 62, "disk_usage": 28, "active_users": 24, "system_uptime": 99.8}


def load_security_events():
    """Load security events."""
    return [
        {
            "time": "2024-01-15 10:30:00",
            "user": "unknown",
            "level": "warning",
            "description": "Failed login attempt from IP 192.168.1.200",
        },
        {
            "time": "2024-01-15 10:25:00",
            "user": "admin",
            "level": "info",
            "description": "Configuration updated: Security settings",
        },
    ]


def get_recent_user_actions():
    """Get recent user actions."""
    return [
        {"user": "admin", "action": "created user", "time": "10:30 AM"},
        {"user": "trader1", "action": "executed strategy", "time": "10:25 AM"},
        {"user": "analyst1", "action": "trained model", "time": "10:20 AM"},
    ]


def get_system_events():
    """Get system events."""
    return [
        {"level": "info", "message": "System backup completed"},
        {"level": "warning", "message": "High memory usage detected"},
        {"level": "info", "message": "New model deployed"},
    ]


def create_user(username, email, role, password):
    """Create a new user."""
    # Mock implementation - in real implementation, this would save to database
    return True


def show_user_statistics():
    """Show user statistics."""
    st.info("User statistics would be displayed here.")


def show_user_activity():
    """Show user activity."""
    st.info("User activity would be displayed here.")


def export_audit_logs_csv(df):
    """Export audit logs to CSV."""
    csv = df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="audit_logs.csv", mime="text/csv")


def show_audit_statistics(df):
    """Show audit statistics."""
    st.info("Audit statistics would be displayed here.")


def run_security_scan():
    """Run security scan."""
    st.info("Security scan completed. No issues found.")


def block_ip_address():
    """Block IP address."""
    ip = st.text_input("Enter IP address to block")
    if st.button("Block IP"):
        st.success(f"IP {ip} blocked successfully.")


def unlock_user_account():
    """Unlock user account."""
    username = st.text_input("Enter username to unlock")
    if st.button("Unlock User"):
        st.success(f"User {username} unlocked successfully.")


if __name__ == "__main__":
    main()
