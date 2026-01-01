"""
Model Performance Dashboard

This page provides comprehensive model performance tracking and analysis,
showing best historical models per ticker and performance trends.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from memory.model_log import (
    clear_model_performance_log,
    get_available_models,
    get_available_tickers,
    get_model_performance_history,
    log_model_performance,
    render_best_models_summary,
    render_model_performance_dashboard,
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the model logging functionality


def main():
    """Main entry point for the Model Performance Dashboard."""
    st.set_page_config(
        page_title="ðŸ“Š Model Performance Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ðŸ“Š Model Performance Dashboard")
    st.markdown(
        "Track and analyze model performance metrics across different tickers and time periods."
    )

    # Sidebar for actions
    st.sidebar.header("ðŸ”§ Actions")

    # Add sample data button
    if st.sidebar.button(
        "ðŸ“Š Add Sample Data", help="Add sample model performance data for testing"
    ):
        add_sample_data()
        st.success("Sample data added successfully!")
        st.rerun()

    # Clear data button
    if st.sidebar.button(
        "ðŸ—‘ï¸ Clear All Data", help="Clear all model performance data"
    ):
        if st.sidebar.checkbox("Confirm deletion"):
            clear_model_performance_log()
            st.success("All data cleared successfully!")
            st.rerun()

    # Manual logging section
    st.sidebar.header("ðŸ“ Manual Logging")

    with st.sidebar.expander("Add New Performance Record"):
        add_manual_performance_record()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "ðŸ“ˆ Performance Dashboard",
            "ðŸ† Best Models Summary",
            "ðŸ“Š Quick Analytics",
        ]
    )

    with tab1:
        render_model_performance_dashboard()

    with tab2:
        render_best_models_summary()

    with tab3:
        render_quick_analytics()


def add_sample_data():
    """Add sample model performance data for demonstration."""

    # Sample data for different models and tickers
    sample_data = [
        # AAPL models
        {
            "model_name": "LSTM_v1",
            "ticker": "AAPL",
            "sharpe": 1.85,
            "mse": 0.0234,
            "drawdown": -0.12,
            "total_return": 0.25,
            "win_rate": 0.68,
            "accuracy": 0.72,
            "notes": "LSTM model with 50 epochs",
        },
        {
            "model_name": "XGBoost_v2",
            "ticker": "AAPL",
            "sharpe": 2.1,
            "mse": 0.0189,
            "drawdown": -0.08,
            "total_return": 0.31,
            "win_rate": 0.75,
            "accuracy": 0.78,
            "notes": "XGBoost with hyperparameter tuning",
        },
        {
            "model_name": "Transformer_v1",
            "ticker": "AAPL",
            "sharpe": 1.95,
            "mse": 0.0212,
            "drawdown": -0.15,
            "total_return": 0.28,
            "win_rate": 0.71,
            "accuracy": 0.74,
            "notes": "Transformer model with attention mechanism",
        },
        # TSLA models
        {
            "model_name": "LSTM_v1",
            "ticker": "TSLA",
            "sharpe": 1.45,
            "mse": 0.0456,
            "drawdown": -0.18,
            "total_return": 0.22,
            "win_rate": 0.62,
            "accuracy": 0.65,
            "notes": "LSTM model for TSLA",
        },
        {
            "model_name": "XGBoost_v2",
            "ticker": "TSLA",
            "sharpe": 1.78,
            "mse": 0.0389,
            "drawdown": -0.14,
            "total_return": 0.28,
            "win_rate": 0.69,
            "accuracy": 0.71,
            "notes": "XGBoost for TSLA",
        },
        # GOOGL models
        {
            "model_name": "LSTM_v1",
            "ticker": "GOOGL",
            "sharpe": 1.92,
            "mse": 0.0198,
            "drawdown": -0.11,
            "total_return": 0.26,
            "win_rate": 0.70,
            "accuracy": 0.73,
            "notes": "LSTM model for GOOGL",
        },
        {
            "model_name": "RandomForest_v1",
            "ticker": "GOOGL",
            "sharpe": 1.65,
            "mse": 0.0256,
            "drawdown": -0.13,
            "total_return": 0.24,
            "win_rate": 0.67,
            "accuracy": 0.69,
            "notes": "Random Forest for GOOGL",
        },
    ]

    # Log each sample record
    for data in sample_data:
        log_model_performance(**data)


def add_manual_performance_record():
    """Add a manual performance record through the UI."""

    col1, col2 = st.columns(2)

    with col1:
        model_name = st.text_input("Model Name", placeholder="e.g., LSTM_v1")
        ticker = st.text_input("Ticker", placeholder="e.g., AAPL").upper()
        sharpe = st.number_input("Sharpe Ratio", value=0.0, step=0.01)
        mse = st.number_input("MSE", value=0.0, step=0.0001, format="%.4f")
        drawdown = st.number_input("Drawdown", value=0.0, step=0.01, format="%.2f")

    with col2:
        total_return = st.number_input(
            "Total Return", value=0.0, step=0.01, format="%.2f"
        )
        win_rate = st.number_input("Win Rate", value=0.0, step=0.01, format="%.2f")
        accuracy = st.number_input("Accuracy", value=0.0, step=0.01, format="%.2f")
        notes = st.text_area("Notes", placeholder="Additional notes about the model")

    if st.button("ðŸ“ Log Performance"):
        if model_name and ticker:
            try:
                log_model_performance(
                    model_name=model_name,
                    ticker=ticker,
                    sharpe=sharpe if sharpe != 0 else None,
                    mse=mse if mse != 0 else None,
                    drawdown=drawdown if drawdown != 0 else None,
                    total_return=total_return if total_return != 0 else None,
                    win_rate=win_rate if win_rate != 0 else None,
                    accuracy=accuracy if accuracy != 0 else None,
                    notes=notes,
                )
                st.success("Performance logged successfully!")
            except Exception as e:
                st.error(f"Error logging performance: {str(e)}")
        else:
            st.warning("Please provide at least a model name and ticker.")


def render_quick_analytics():
    """Render quick analytics and insights."""
    st.header("ðŸ“Š Quick Analytics")

    # Get available data
    tickers = get_available_tickers()
    if not tickers:
        st.info("No data available for analytics.")
        return

    # Overall statistics
    st.subheader("ðŸ“ˆ Overall Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tickers", len(tickers))

    with col2:
        all_models = set()
        for ticker in tickers:
            models = get_available_models(ticker)
            all_models.update(models)
        st.metric("Total Models", len(all_models))

    with col3:
        df = get_model_performance_history()
        if not df.empty:
            avg_sharpe = df["sharpe"].mean()
            st.metric(
                "Avg Sharpe", f"{avg_sharpe:.3f}" if pd.notna(avg_sharpe) else "N/A"
            )
        else:
            st.metric("Avg Sharpe", "N/A")

    with col4:
        if not df.empty:
            avg_mse = df["mse"].mean()
            st.metric("Avg MSE", f"{avg_mse:.4f}" if pd.notna(avg_mse) else "N/A")
        else:
            st.metric("Avg MSE", "N/A")

    # Model performance comparison
    st.subheader("ðŸ† Top Performing Models")

    if not df.empty:
        # Find best models by different metrics
        best_models = {}

        for metric in ["sharpe", "mse", "total_return", "win_rate"]:
            if metric in df.columns and not df[metric].isna().all():
                if metric == "mse":
                    # For MSE, lower is better
                    best_idx = df[metric].idxmin()
                else:
                    # For other metrics, higher is better
                    best_idx = df[metric].idxmax()

                best_row = df.loc[best_idx]
                best_models[metric] = {
                    "model": best_row["model_name"],
                    "ticker": best_row["ticker"],
                    "value": best_row[metric],
                }

        # Display best models
        if best_models:
            cols = st.columns(len(best_models))
            for i, (metric, data) in enumerate(best_models.items()):
                with cols[i]:
                    metric_name = metric.replace("_", " ").title()
                    value = data["value"]

                    if metric == "mse":
                        formatted_value = f"{value:.4f}"
                    elif metric in ["total_return", "win_rate"]:
                        formatted_value = f"{value:.1%}"
                    else:
                        formatted_value = f"{value:.3f}"

                    st.metric(
                        f"Best {metric_name}",
                        formatted_value,
                        f"{data['model']} ({data['ticker']})",
                    )

    # Recent activity
    st.subheader("ðŸ•’ Recent Activity")

    if not df.empty:
        recent_data = df.head(10)[
            ["timestamp", "ticker", "model_name", "sharpe", "mse"]
        ]
        recent_data["timestamp"] = recent_data["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )

        # Format numeric columns
        recent_data["sharpe"] = recent_data["sharpe"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
        recent_data["mse"] = recent_data["mse"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )

        st.dataframe(recent_data, use_container_width=True)

    # Performance trends
    st.subheader("ðŸ“ˆ Performance Trends")

    if not df.empty and len(tickers) > 0:
        selected_ticker = st.selectbox("Select ticker for trend analysis", tickers)

        ticker_data = df[df["ticker"] == selected_ticker]
        if not ticker_data.empty:
            # Create trend chart
            import plotly.express as px

            # Prepare data for plotting
            plot_data = ticker_data[["timestamp", "model_name", "sharpe", "mse"]].copy()
            plot_data = plot_data.melt(
                id_vars=["timestamp", "model_name"],
                value_vars=["sharpe", "mse"],
                var_name="metric",
                value_name="value",
            )

            fig = px.line(
                plot_data,
                x="timestamp",
                y="value",
                color="model_name",
                facet_row="metric",
                title=f"Performance Trends for {selected_ticker}",
            )

            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
