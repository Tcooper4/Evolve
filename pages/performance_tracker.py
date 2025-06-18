"""
Performance Tracker Dashboard

This page provides interactive visualizations of model performance metrics with enhanced filtering
and analysis capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from memory.performance_memory import PerformanceMemory
from llm.llm_summary import generate_strategy_commentary

def main():
    st.set_page_config(page_title="📊 Agentic Performance Tracker", layout="wide")
    st.title("📊 Performance Tracker")

    memory = PerformanceMemory()
    tickers = memory.get_all_tickers()
    if not tickers:
        st.warning("No performance records found.")
        return

    selected_ticker = st.selectbox("📈 Select Ticker", tickers)
    metrics = memory.get_metrics(selected_ticker)

    if not metrics:
        st.warning("No metrics found for this ticker.")
        return

    # ==== Structuring Data ====
    records = []
    for model, info in metrics.items():
        records.append({
            "Model": model,
            "MSE": info.get("mse"),
            "Sharpe Ratio": info.get("sharpe"),
            "Win Rate": info.get("win_rate"),
            "Timestamp": info.get("timestamp"),
            "Confidence Low": info.get("confidence_intervals", {}).get("low"),
            "Confidence High": info.get("confidence_intervals", {}).get("high"),
            "Data Size": info.get("dataset_size", 0),
            "Status": info.get("status", "Active")
        })
    df = pd.DataFrame(records)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # ==== Sidebar Filters ====
    st.sidebar.header("🔧 Filters")
    df = df.sort_values("Timestamp", ascending=True)

    date_range = st.sidebar.date_input("📅 Date Filter", [df["Timestamp"].min(), df["Timestamp"].max()])
    if len(date_range) == 2:
        df = df[(df["Timestamp"] >= pd.to_datetime(date_range[0])) & 
                (df["Timestamp"] <= pd.to_datetime(date_range[1]))]

    models = df["Model"].unique()
    selected_models = st.sidebar.multiselect("🎯 Filter by Model", models, default=list(models))
    df = df[df["Model"].isin(selected_models)]

    apply_rolling = st.sidebar.checkbox("📉 Rolling Average (window=2)", value=False)
    if apply_rolling:
        df["MSE"] = df.groupby("Model")["MSE"].transform(lambda x: x.rolling(2, min_periods=1).mean())

    # ==== Tabs ====
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📉 MSE", "📈 Sharpe", "🎯 Win Rate", "🔥 Heatmap", "🔍 Detail"])

    with tab1:
        st.subheader("📉 MSE Trend")
        fig1 = px.line(
            df, 
            x="Timestamp", 
            y="MSE", 
            color="Model", 
            markers=True,
            hover_data=["Sharpe Ratio", "Win Rate", "Confidence Low", "Confidence High"]
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader("📈 Sharpe Ratio by Model")
        fig2 = px.bar(
            df, 
            x="Model", 
            y="Sharpe Ratio", 
            color="Model", 
            barmode="group"
        )
        st.plotly_chart(fig2)

    with tab3:
        st.subheader("🎯 Win Rate by Model")
        fig3 = px.bar(
            df, 
            x="Model", 
            y="Win Rate", 
            color="Model", 
            barmode="group"
        )
        st.plotly_chart(fig3)

    with tab4:
        st.subheader("🔥 Model Comparison Heatmap")
        heat_df = df.pivot_table(index="Model", values=["MSE", "Sharpe Ratio", "Win Rate"])
        fig4, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heat_df, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig4)

    with tab5:
        st.subheader("🔍 Detailed Drilldown")
        selected_model = st.selectbox("🔎 Select Model", df["Model"].unique())
        ts = st.selectbox(
            "🕒 Select Timestamp", 
            df[df["Model"] == selected_model]["Timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S').unique()
        )
        row = df[(df["Model"] == selected_model) & 
                 (df["Timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S') == ts)]
        if not row.empty:
            st.code("\n".join([f"{k}: {v}" for k, v in row.iloc[0].to_dict().items()]), language="yaml")

    # ==== AI Commentary ====
    st.subheader("🧠 AI Strategy Commentary")
    st.markdown("```text\n" + generate_strategy_commentary(df) + "\n```")

    # ==== Retraining + Switch Suggestion ====
    st.subheader("🚦 Strategy Recommendations")
    bad_models = df[
        (df["MSE"] > 0.1) |
        (df["Sharpe Ratio"] < 1.0) |
        (df["Win Rate"] < 0.6)
    ]["Model"].unique()

    if bad_models.any():
        st.error(f"🚨 Suggest retraining or switching: {', '.join(bad_models)}")
    else:
        st.success("✅ All models meet performance criteria.")

    # ==== Auto-Deactivation ====
    st.subheader("📉 Auto-Deactivate Poor Performers")
    toggle = st.toggle("Enable Auto-Deactivation", value=False)
    if toggle:
        st.warning(f"The following models will be flagged as inactive: {', '.join(bad_models)}")
        # memory.mark_inactive_models(bad_models) # implement if desired
    else:
        st.caption("All models remain active.")

    # ==== Reactivation Toggle ====
    st.subheader("🔄 Manual Reactivation")
    if toggle and bad_models.any():
        reactivate = st.multiselect("Reactivate Models", bad_models)
        if st.button("✅ Apply Reactivation"):
            st.success(f"Reactivated: {', '.join(reactivate)}")
            # memory.reactivate_models(reactivate) # optional

    # ==== A/B Testing ====
    st.subheader("🧪 A/B Test Group Split")
    df["Group"] = df["Model"].apply(lambda x: "A" if "v1" in x.lower() else "B")
    ab_summary = df.groupby("Group")[["MSE", "Sharpe Ratio", "Win Rate"]].mean().round(3)
    st.dataframe(ab_summary)
    st.plotly_chart(
        px.bar(
            ab_summary.reset_index().melt(id_vars="Group"), 
            x="Group", 
            y="value", 
            color="variable", 
            barmode="group"
        )
    )

    # ==== Export ====
    st.subheader("📁 Export Report")
    st.download_button(
        "⬇ Export CSV", 
        df.to_csv(index=False).encode("utf-8"), 
        file_name=f"{selected_ticker}_performance.csv"
    )

if __name__ == "__main__":
    main() 