"""
Model Performance Visualization Dashboard

This module provides a Streamlit interface for visualizing model performance metrics
stored in the PerformanceMemory system.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from trading.memory.performance_memory import PerformanceMemory

def main():
    st.title("📈 Agentic Model Performance Dashboard")
    
    # Initialize memory
    memory = PerformanceMemory()
    
    # Get available tickers
    tickers = memory.get_all_tickers()
    if not tickers:
        st.warning("No performance data available.")
        st.stop()
    
    # Ticker selection
    selected_ticker = st.selectbox("Select Ticker", tickers)
    all_metrics = memory.get_metrics(selected_ticker)
    
    if not all_metrics:
        st.warning("No model metrics found for this ticker.")
        st.stop()
    
    # Flatten metrics into DataFrame
    records = []
    for model, metrics in all_metrics.items():
        record = {
            "model": model,
            "mse": metrics.get("mse"),
            "sharpe": metrics.get("sharpe"),
            "win_rate": metrics.get("win_rate"),
            "timestamp": metrics.get("timestamp"),
            "confidence_low": metrics.get("confidence_intervals", {}).get("low"),
            "confidence_high": metrics.get("confidence_intervals", {}).get("high"),
            "dataset_size": metrics.get("dataset_size", 0)
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Display best model
    best_model = memory.get_best_model(selected_ticker, metric="mse")
    st.subheader(f"✅ Best Model by MSE: `{best_model}`")
    
    # Plot MSE vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        ax.plot(subset["timestamp"], subset["mse"], marker='o', label=model)
    ax.set_title("Model MSE Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Additional metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sharpe Ratio")
        fig_sharpe, ax_sharpe = plt.subplots(figsize=(8, 4))
        for model in df["model"].unique():
            subset = df[df["model"] == model]
            ax_sharpe.plot(subset["timestamp"], subset["sharpe"], marker='o', label=model)
        ax_sharpe.set_title("Sharpe Ratio Over Time")
        ax_sharpe.set_xlabel("Timestamp")
        ax_sharpe.set_ylabel("Sharpe Ratio")
        ax_sharpe.legend()
        ax_sharpe.grid(True)
        st.pyplot(fig_sharpe)
    
    with col2:
        st.subheader("Win Rate")
        fig_win, ax_win = plt.subplots(figsize=(8, 4))
        for model in df["model"].unique():
            subset = df[df["model"] == model]
            ax_win.plot(subset["timestamp"], subset["win_rate"], marker='o', label=model)
        ax_win.set_title("Win Rate Over Time")
        ax_win.set_xlabel("Timestamp")
        ax_win.set_ylabel("Win Rate")
        ax_win.legend()
        ax_win.grid(True)
        st.pyplot(fig_win)
    
    # Show raw metrics table
    st.subheader("📊 Raw Metrics")
    st.dataframe(
        df.style.format({
            'mse': '{:.4f}',
            'sharpe': '{:.2f}',
            'win_rate': '{:.2%}',
            'confidence_low': '{:.2f}',
            'confidence_high': '{:.2f}'
        })
    )
    
    # Show dataset statistics
    st.subheader("📈 Dataset Statistics")
    dataset_stats = df.groupby('model')['dataset_size'].agg(['mean', 'min', 'max']).reset_index()
    st.dataframe(
        dataset_stats.style.format({
            'mean': '{:.0f}',
            'min': '{:.0f}',
            'max': '{:.0f}'
        })
    )

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == "__main__":
    main() 