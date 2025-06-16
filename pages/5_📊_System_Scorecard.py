import streamlit as st
import pandas as pd
from utils.system_status import get_system_scorecard
import datetime

st.set_page_config(page_title="📊 System Scorecard", page_icon="📊", layout="wide")
st.title("📊 System Scorecard")

# Refresh button
def refresh():
    st.session_state["scorecard_refresh"] = datetime.datetime.now().isoformat()

if st.button("📥 Refresh Now"):
    refresh()

# Load metrics
data = get_system_scorecard()

# Metrics columns
col1, col2 = st.columns(2)
col1.metric("7d Sharpe", data["sharpe_7d"])
col2.metric("30d Sharpe", data["sharpe_30d"])
st.metric("Average MSE", data["mse_avg"])
st.metric("Win Rate (%)", data["win_rate"])

# Trades per day line chart
if not data["trades_per_day"].empty:
    st.line_chart(data["trades_per_day"], use_container_width=True)
else:
    st.info("No trade data available for chart.")

# Load and display goal status
st.subheader("🎯 Goal Status")
goals = data["goal_status"]
if goals:
    for key, value in goals.items():
        if value is True or (isinstance(value, str) and value.lower() == "pass"):
            st.markdown(f"✅ **{key}**: <span style='color:green'>Met</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"❌ **{key}**: <span style='color:red'>Missed</span>", unsafe_allow_html=True)
else:
    st.info("No goal status available.")

# Table of last 10 runs
st.subheader("📝 Last 10 Strategy/Model Runs")
last_10 = data["last_10_entries"]
if isinstance(last_10, pd.DataFrame) and not last_10.empty:
    st.dataframe(last_10, use_container_width=True)
else:
    st.info("No recent runs to display.") 