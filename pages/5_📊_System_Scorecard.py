"""System scorecard page for displaying performance metrics."""

# Standard library imports
import datetime

# Third-party imports
import pandas as pd
import streamlit as st

# Local imports
from utils.system_status import get_system_scorecard

st.set_page_config(page_title="📊 System Scorecard", page_icon="📊", layout="wide")
st.title("📊 System Scorecard")

# Refresh button
def refresh():
    if "scorecard_refresh" not in st.session_state:
        st.session_state["scorecard_refresh"] = datetime.datetime.now().isoformat()
    else:
        st.session_state["scorecard_refresh"] = datetime.datetime.now().isoformat()

if st.button("🔄 Refresh Now"):
    refresh()

# Load metrics with default values
try:
    data = get_system_scorecard()
except Exception as e:
    st.error(f"Error loading system metrics: {str(e)}")
    data = {
        "sharpe_7d": 0.0,
        "sharpe_30d": 0.0,
        "win_rate": 0.0,
        "mse_avg": 0.0,
        "goal_status": {},
        "last_10_entries": pd.DataFrame(),
        "trades_per_day": pd.Series()
    }

# Metrics columns
col1, col2 = st.columns(2)
col1.metric("7d Sharpe", data.get("sharpe_7d", 0.0))
col2.metric("30d Sharpe", data.get("sharpe_30d", 0.0))
st.metric("Average MSE", data.get("mse_avg", 0.0))
st.metric("Win Rate (%)", data.get("win_rate", 0.0))

# Trades per day line chart
trades_per_day = data.get("trades_per_day", pd.Series())
if not trades_per_day.empty:
    st.line_chart(trades_per_day, use_container_width=True)
else:
    st.info("No trade data available for chart.")

# Load and display goal status
st.subheader("🎯 Goal Status")
goals = data.get("goal_status", {})
if goals:
    for key, value in goals.items():
        if value is True or (isinstance(value, str) and value.lower() == "pass"):
            st.markdown(f"✅ **{key}**: <span style='color:green'>Met</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"❌ **{key}**: <span style='color:red'>Missed</span>", unsafe_allow_html=True)
else:
    st.info("No goal status available.")

# Table of last 10 runs
st.subheader("📈 Last 10 Strategy/Model Runs")
last_10 = data.get("last_10_entries", pd.DataFrame())
if isinstance(last_10, pd.DataFrame) and not last_10.empty:
    st.dataframe(last_10, use_container_width=True)
else:
    st.info("No recent runs to display.") 