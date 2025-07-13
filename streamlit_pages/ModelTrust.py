"""
Model Trust Insights Dashboard

This Streamlit page provides insights into model trust levels, drift detection,
and strategy prioritization through interactive visualizations.
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from trading.memory.model_monitor import detect_drift, generate_strategy_priority


def load_weight_history():
    """Load weight history data if available."""
    if not os.path.exists("memory/weight_history.json"):
        return None
    with open("memory/weight_history.json", "r") as f:
        return json.load(f)


def main():
    st.title("🧠 Model Trust Insights")

    # Load available tickers
    history = load_weight_history()
    if history:
        tickers = sorted({t for v in history.values() for t in v})
    else:
        tickers = ["AAPL", "MSFT", "TSLA"]  # Default options

    # Ticker selection
    ticker = st.selectbox("Select a Ticker", options=tickers)

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Strategy Priorities", "Drift Detection", "Weight History"])

    with tab1:
        st.subheader("📊 Strategy Priorities")
        if st.button("Generate Strategy Priorities", key=os.getenv("KEY", "")):
            with st.spinner("Analyzing model priorities..."):
                priorities = generate_strategy_priority(ticker)
                if priorities:
                    # Display prioritized models
                    st.success("Strategy priorities generated successfully!")

                    # Create bar chart of weights
                    weights_df = pd.DataFrame(
                        {"Model": list(priorities["weights"].keys()), "Weight": list(priorities["weights"].values())}
                    )
                    fig = px.bar(
                        weights_df,
                        x="Model",
                        y="Weight",
                        title=f"{ticker} Model Weights",
                        color="Weight",
                        color_continuous_scale="RdYlGn",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show raw data
                    with st.expander("View Raw Priority Data"):
                        st.json(priorities)
                else:
                    st.warning("No priority data available for this ticker.")

    with tab2:
        st.subheader("🔍 Drift Detection")
        threshold = st.slider(
            "Drift Threshold", 0.01, 0.3, 0.2, help="Threshold for detecting significant weight changes"
        )

        if st.button("Detect Model Drift", key=os.getenv("KEY", "")):
            with st.spinner("Analyzing model drift..."):
                drift = detect_drift(ticker, threshold=threshold)
                if drift:
                    st.error("⚠️ Drift Detected")

                    # Create drift visualization
                    drift_df = pd.DataFrame(drift)
                    fig = px.bar(
                        drift_df,
                        x="model",
                        y="change",
                        color="change",
                        title="Model Weight Changes",
                        labels={"model": "Model", "change": "Weight Change"},
                        color_continuous_scale="RdBu",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show detailed drift information
                    with st.expander("View Detailed Drift Information"):
                        st.json(drift)
                else:
                    st.success("✅ No significant drift detected.")

    with tab3:
        st.subheader("📈 Weight History")
        if history and ticker in [t for v in history.values() for t in v]:
            records = {ts: entry[ticker] for ts, entry in history.items() if ticker in entry}
            df = pd.DataFrame.from_dict(records, orient="index").sort_index()
            df.index = pd.to_datetime(df.index)

            # Create line plot
            fig = px.line(df, title=f"{ticker} Model Weights Over Time", labels={"value": "Weight", "index": "Time"})
            fig.update_layout(xaxis_title="Time", yaxis_title="Weight", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Show current weights
            st.subheader("Current Weights")
            latest_weights = df.iloc[-1].sort_values(ascending=False)
            fig2 = px.bar(x=latest_weights.index, y=latest_weights.values, title="Latest Weight Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No weight history available for this ticker.")


if __name__ == "__main__":
    main()
