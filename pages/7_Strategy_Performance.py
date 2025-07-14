"""
Strategy Performance Visualization Page

This page provides interactive visualizations of strategy performance metrics,
including time series analysis, model comparisons, and performance breakdowns.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# Constants
PERF_LOG = "memory/performance_log.json"
PERF_ANALYSIS = "memory/performance_analysis.json"


def load_performance_data():
    """Load and process performance data from log file."""
    if not os.path.exists(PERF_LOG):
        return pd.DataFrame()

    with open(PERF_LOG, "r") as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Process timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract metrics into separate columns
    metrics_df = pd.json_normalize(df["metrics"])
    df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)

    # Add derived columns
    df["agentic_label"] = df["agentic"].map({True: "Agentic", False: "Manual"})
    df["date"] = df["timestamp"].dt.date

    return df


def plot_metric_timeseries(df, metric, ticker=None, model=None):
    """Create interactive time series plot for a specific metric."""
    # Filter data
    plot_df = df.copy()
    if ticker and ticker != "All" and "ticker" in plot_df.columns:
        plot_df = plot_df[plot_df["ticker"] == ticker]
    if model:
        plot_df = plot_df[plot_df["model"] == model]

    # Create figure
    fig = go.Figure()

    # Add traces for each model and agentic status
    for model_name in plot_df["model"].unique():
        for agentic in [True, False]:
            mask = (plot_df["model"] == model_name) & (plot_df["agentic"] == agentic)
            subset = plot_df[mask]

            if not subset.empty:
                fig.add_trace(
                    go.Scatter(
                        x=subset["timestamp"],
                        y=subset[metric],
                        name=f"{model_name} ({'Agentic' if agentic else 'Manual'})",
                        mode="lines+markers",
                        hovertemplate="<b>%{x}</b><br>"
                        + f"{metric.title()}: %{{y:.2f}}<br>"
                        + "<extra></extra>",
                    )
                )

    # Update layout
    fig.update_layout(
        title=f"{metric.title()} Over Time",
        xaxis_title="Date",
        yaxis_title=metric.title(),
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_metric_distribution(df, metric, ticker=None):
    """Create distribution plot for a specific metric."""
    # Filter data
    plot_df = df.copy()
    if ticker and ticker != "All" and "ticker" in plot_df.columns:
        plot_df = plot_df[plot_df["ticker"] == ticker]

    # Create figure
    fig = go.Figure()

    # Add box plots for each model
    for model_name in plot_df["model"].unique():
        model_data = plot_df[plot_df["model"] == model_name]

        fig.add_trace(
            go.Box(
                y=model_data[metric],
                name=model_name,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate="<b>%{x}</b><br>"
                + f"{metric.title()}: %{{y:.2f}}<br>"
                + "<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{metric.title()} Distribution by Model",
        yaxis_title=metric.title(),
        showlegend=False,
        boxmode="group",
    )

    return fig


def plot_metric_heatmap(df, metric, ticker=None):
    """Create heatmap of metric values by model and agentic status."""
    # Filter data
    plot_df = df.copy()
    if ticker and ticker != "All" and "ticker" in plot_df.columns:
        plot_df = plot_df[plot_df["ticker"] == ticker]

    # Calculate average metric by model and agentic status
    heatmap_data = plot_df.pivot_table(
        values=metric, index="model", columns="agentic_label", aggfunc="mean"
    )

    # Create figure
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="RdYlGn",
            text=[[f"{val:.2f}" for val in row] for row in heatmap_data.values],
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="<b>%{y}</b><br>"
            + "Type: %{x}<br>"
            + f"{metric.title()}: %{{z:.2f}}<br>"
            + "<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Average {metric.title()} by Model and Strategy Type",
        xaxis_title="Strategy Type",
        yaxis_title="Model",
        showlegend=False,
    )

    return fig


def create_snapshot_chart(df, metric, ticker=None):
    """Create a static snapshot chart for export."""
    # Filter data
    plot_df = df.copy()
    if ticker and ticker != "All" and "ticker" in plot_df.columns:
        plot_df = plot_df[plot_df["ticker"] == ticker]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot data
    sns.lineplot(
        data=plot_df,
        x="timestamp",
        y=metric,
        hue="agentic_label",
        style="model",
        markers=True,
        dashes=False,
        ax=ax,
    )

    # Customize plot
    ax.set_title(f"Snapshot: {metric.title()} Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric.title())
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def create_leaderboard(df, metric):
    """Create a leaderboard of model performance."""
    # Group by model and calculate metrics
    leaderboard_df = (
        df.groupby("model")
        .agg({metric: ["mean", "std", "count"], "agentic": ["sum", "count"]})
        .round(3)
    )

    # Flatten column names
    leaderboard_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in leaderboard_df.columns
    ]

    # Calculate additional metrics
    leaderboard_df["agentic_ratio"] = (
        leaderboard_df["agentic_sum"] / leaderboard_df["agentic_count"]
    ).round(3)

    # Sort by metric mean
    leaderboard_df = leaderboard_df.sort_values(f"{metric}_mean", ascending=False)

    return leaderboard_df


def get_top_models(df):
    """Get top performing models for each metric."""
    # Group by model and calculate mean metrics
    metrics_df = (
        df.groupby("model")
        .agg({"sharpe": "mean", "accuracy": "mean", "win_rate": "mean"})
        .round(3)
    )

    # Get top model for each metric
    top_models = {}
    for metric in ["sharpe", "accuracy", "win_rate"]:
        top = metrics_df.sort_values(metric, ascending=False).iloc[0]
        top_models[metric] = f"{top.name} ({top[metric]:.3f})"

    return top_models


def main():
    st.set_page_config(page_title="Strategy Performance", page_icon="📊", layout="wide")

    st.title("Strategy Performance Analysis")
    st.markdown(
        "Analyze and compare strategy performance metrics across different models and decision types."
    )

    # Load data
    df = load_performance_data()

    if df is None:
        st.info("No performance metrics have been logged yet.")

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        # Ticker selection - check if ticker column exists
        if "ticker" in df.columns and not df.empty:
            tickers = sorted(df["ticker"].unique())
            selected_ticker = st.selectbox("Select Ticker", ["All"] + list(tickers))
        else:
            selected_ticker = "All"
            st.info("No ticker data available")

        # Date range selection
        if not df.empty:
            min_date = df["date"].min()
            max_date = df["date"].max()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        else:
            date_range = (datetime.now().date(), datetime.now().date())

        # Model selection
        if "model" in df.columns and not df.empty:
            models = sorted(df["model"].unique())
            selected_models = st.multiselect(
                "Select Models", options=models, default=models
            )
        else:
            selected_models = []
            st.info("No model data available")

        # Metric selection
        metrics = ["sharpe", "accuracy", "win_rate"]
        selected_metric = st.selectbox("Select Metric", metrics)

    # Apply filters
    filtered_df = df.copy()
    if selected_ticker != "All" and "ticker" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["ticker"] == selected_ticker]
    if len(date_range) == 2 and not filtered_df.empty:
        filtered_df = filtered_df[
            (filtered_df["date"] >= date_range[0])
            & (filtered_df["date"] <= date_range[1])
        ]
    if selected_models and "model" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["model"].isin(selected_models)]

    # Main content
    if not filtered_df.empty:
        # Summary metrics
        st.subheader("Performance Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Decisions", len(filtered_df))
        with col2:
            st.metric("Agentic Decisions", len(filtered_df[filtered_df["agentic"]]))
        with col3:
            st.metric("Manual Overrides", len(filtered_df[~filtered_df["agentic"]]))
        with col4:
            st.metric(
                f"Avg {selected_metric.title()}",
                f"{filtered_df[selected_metric].mean():.2f}",
            )

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Time Series", "Distribution", "Heatmap"])

        with tab1:
            st.plotly_chart(
                plot_metric_timeseries(filtered_df, selected_metric, selected_ticker),
                use_container_width=True,
            )

        with tab2:
            st.plotly_chart(
                plot_metric_distribution(filtered_df, selected_metric, selected_ticker),
                use_container_width=True,
            )

        with tab3:
            st.plotly_chart(
                plot_metric_heatmap(filtered_df, selected_metric, selected_ticker),
                use_container_width=True,
            )

        # Export section
        st.subheader("📁 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            # Export filtered data
            if st.button("📤 Export Filtered Data (CSV)"):
                csv = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Filtered Data",
                    csv,
                    f"strategy_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                )

        with col2:
            # Export full log
            if os.path.exists(PERF_LOG):
                with open(PERF_LOG, "r") as f:
                    perf_data = json.load(f)
                perf_flat = pd.json_normalize(perf_data)
                perf_csv = perf_flat.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Full Performance Log",
                    perf_csv,
                    "performance_log_full.csv",
                    "text/csv",
                )

        # Snapshot chart export
        st.subheader("📸 Export Current Chart")

        # Create snapshot chart
        snapshot_fig = create_snapshot_chart(
            filtered_df, selected_metric, selected_ticker
        )
        st.pyplot(snapshot_fig)

        # Export chart as PNG
        if st.button("Save Chart as PNG"):
            # Create charts directory if it doesn't exist
            os.makedirs("charts", exist_ok=True)

            # Save chart
            filename = f"charts/snapshot_{selected_metric}_{datetime.now().strftime('%Y%m%d')}.png"
            snapshot_fig.savefig(filename, dpi=300, bbox_inches="tight")
            st.success(f"Chart saved as {filename}")

        # Strategy Leaderboard
        st.subheader("🏆 Strategy Leaderboard")

        # Create and display leaderboard
        leaderboard_df = create_leaderboard(filtered_df, selected_metric)

        # Format the leaderboard
        formatted_df = leaderboard_df.style.format(
            {
                f"{selected_metric}_mean": "{:.3f}",
                f"{selected_metric}_std": "{:.3f}",
                "agentic_ratio": "{:.1%}",
            }
        )

        # Display leaderboard with metrics explanation
        st.dataframe(formatted_df, use_container_width=True)

        # Add metrics explanation
        with st.expander("📊 Metrics Explanation"):
            st.markdown(
                """
            - **Mean**: Average performance for the metric
            - **Std**: Standard deviation of the metric
            - **Count**: Number of decisions made
            - **Agentic Ratio**: Proportion of agentic decisions
            """
            )

        # Top Models Summary
        st.subheader("🥇 Top Models by Metric (Current View)")
        top_models = get_top_models(filtered_df)
        st.markdown(
            f"""
        - **Best Sharpe**: {top_models['sharpe']}
        - **Best Accuracy**: {top_models['accuracy']}
        - **Best Win Rate**: {top_models['win_rate']}
        """
        )

        # Detailed metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(
            filtered_df.sort_values("timestamp", ascending=False)
            .drop("agentic_label", axis=1)
            .style.format(
                {"sharpe": "{:.2f}", "accuracy": "{:.2%}", "win_rate": "{:.2%}"}
            )
        )
    else:
        st.warning("No data available for the selected filters.")


if __name__ == "__main__":
    main()
