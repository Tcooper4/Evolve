"""
Model Performance Logging Module

This module provides functionality for logging model performance metrics
and displaying best historical models per ticker in the UI.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st

# Constants
LOG_DIR = Path("memory/logs")
MODEL_PERFORMANCE_FILE = LOG_DIR / "model_performance.json"
MODEL_PERFORMANCE_CSV = LOG_DIR / "model_performance.csv"
BEST_MODELS_FILE = LOG_DIR / "best_models.json"

# Required fields for CSV logging
CSV_FIELDS = [
    "timestamp",
    "ticker",
    "model_name", 
    "sharpe",
    "mse",
    "drawdown",
    "total_return",
    "win_rate",
    "accuracy",
    "notes"
]


def ensure_log_directory():
    """Ensure the log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_model_performance(
    model_name: str,
    ticker: str,
    sharpe: Optional[float] = None,
    mse: Optional[float] = None,
    drawdown: Optional[float] = None,
    total_return: Optional[float] = None,
    win_rate: Optional[float] = None,
    accuracy: Optional[float] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Log model performance metrics to both JSON and CSV formats.
    
    Args:
        model_name: Name of the model
        ticker: Stock ticker symbol
        sharpe: Sharpe ratio
        mse: Mean squared error
        drawdown: Maximum drawdown
        total_return: Total return percentage
        win_rate: Win rate percentage
        accuracy: Model accuracy
        notes: Additional notes
        
    Returns:
        Dict containing the logged performance data
    """
    ensure_log_directory()
    
    timestamp = datetime.now().isoformat()
    
    # Prepare performance data
    performance_data = {
        "timestamp": timestamp,
        "ticker": ticker,
        "model_name": model_name,
        "sharpe": sharpe,
        "mse": mse,
        "drawdown": drawdown,
        "total_return": total_return,
        "win_rate": win_rate,
        "accuracy": accuracy,
        "notes": notes
    }
    
    # Log to JSON
    _log_to_json(performance_data)
    
    # Log to CSV
    _log_to_csv(performance_data)
    
    # Update best models tracking
    _update_best_models(ticker, model_name, performance_data)
    
    return performance_data


def _log_to_json(performance_data: Dict[str, Any]):
    """Log performance data to JSON file."""
    if MODEL_PERFORMANCE_FILE.exists():
        with open(MODEL_PERFORMANCE_FILE, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []
    
    log_data.append(performance_data)
    
    with open(MODEL_PERFORMANCE_FILE, 'w') as f:
        json.dump(log_data, f, indent=2)


def _log_to_csv(performance_data: Dict[str, Any]):
    """Log performance data to CSV file."""
    file_exists = MODEL_PERFORMANCE_CSV.exists()
    
    with open(MODEL_PERFORMANCE_CSV, mode='a', newline='') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(performance_data)


def _update_best_models(ticker: str, model_name: str, performance_data: Dict[str, Any]):
    """Update best models tracking for each ticker."""
    if BEST_MODELS_FILE.exists():
        with open(BEST_MODELS_FILE, 'r') as f:
            best_models = json.load(f)
    else:
        best_models = {}
    
    if ticker not in best_models:
        best_models[ticker] = {
            "best_sharpe": {"model": None, "value": None, "timestamp": None},
            "best_mse": {"model": None, "value": None, "timestamp": None},
            "best_drawdown": {"model": None, "value": None, "timestamp": None},
            "best_total_return": {"model": None, "value": None, "timestamp": None},
            "best_win_rate": {"model": None, "value": None, "timestamp": None},
            "best_accuracy": {"model": None, "value": None, "timestamp": None}
        }
    
    # Update best models for each metric
    metrics_to_track = [
        ("sharpe", "best_sharpe", lambda x, y: x > y if x is not None and y is not None else x is not None),
        ("mse", "best_mse", lambda x, y: x < y if x is not None and y is not None else x is not None),
        ("drawdown", "best_drawdown", lambda x, y: x > y if x is not None and y is not None else x is not None),
        ("total_return", "best_total_return", lambda x, y: x > y if x is not None and y is not None else x is not None),
        ("win_rate", "best_win_rate", lambda x, y: x > y if x is not None and y is not None else x is not None),
        ("accuracy", "best_accuracy", lambda x, y: x > y if x is not None and y is not None else x is not None)
    ]
    
    for metric, best_key, comparison_func in metrics_to_track:
        current_value = performance_data.get(metric)
        best_value = best_models[ticker][best_key]["value"]
        
        if comparison_func(current_value, best_value):
            best_models[ticker][best_key] = {
                "model": model_name,
                "value": current_value,
                "timestamp": performance_data["timestamp"]
            }
    
    with open(BEST_MODELS_FILE, 'w') as f:
        json.dump(best_models, f, indent=2)


def get_model_performance_history(
    ticker: Optional[str] = None,
    model_name: Optional[str] = None,
    days_back: Optional[int] = None
) -> pd.DataFrame:
    """
    Get model performance history as a DataFrame.
    
    Args:
        ticker: Filter by ticker symbol
        model_name: Filter by model name
        days_back: Filter by number of days back
        
    Returns:
        DataFrame containing performance history
    """
    if not MODEL_PERFORMANCE_CSV.exists():
        return pd.DataFrame(columns=CSV_FIELDS)
    
    df = pd.read_csv(MODEL_PERFORMANCE_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Apply filters
    if ticker:
        df = df[df['ticker'] == ticker]
    
    if model_name:
        df = df[df['model_name'] == model_name]
    
    if days_back:
        cutoff_date = datetime.now() - pd.Timedelta(days=days_back)
        df = df[df['timestamp'] >= cutoff_date]
    
    return df.sort_values('timestamp', ascending=False)


def get_best_models(ticker: Optional[str] = None) -> Dict[str, Any]:
    """
    Get best models for each metric.
    
    Args:
        ticker: Filter by ticker symbol
        
    Returns:
        Dictionary containing best models for each metric
    """
    if not BEST_MODELS_FILE.exists():
        return {}
    
    with open(BEST_MODELS_FILE, 'r') as f:
        best_models = json.load(f)
    
    if ticker:
        return best_models.get(ticker, {})
    
    return best_models


def get_available_tickers() -> List[str]:
    """Get list of available tickers with performance data."""
    if not MODEL_PERFORMANCE_CSV.exists():
        return []
    
    df = pd.read_csv(MODEL_PERFORMANCE_CSV)
    return sorted(df['ticker'].unique().tolist())


def get_available_models(ticker: Optional[str] = None) -> List[str]:
    """Get list of available models with performance data."""
    if not MODEL_PERFORMANCE_CSV.exists():
        return []
    
    df = pd.read_csv(MODEL_PERFORMANCE_CSV)
    
    if ticker:
        df = df[df['ticker'] == ticker]
    
    return sorted(df['model_name'].unique().tolist())


def clear_model_performance_log():
    """Clear all model performance logs."""
    if MODEL_PERFORMANCE_FILE.exists():
        MODEL_PERFORMANCE_FILE.unlink()
    
    if MODEL_PERFORMANCE_CSV.exists():
        MODEL_PERFORMANCE_CSV.unlink()
    
    if BEST_MODELS_FILE.exists():
        BEST_MODELS_FILE.unlink()


# Streamlit UI Components
def render_model_performance_dashboard():
    """Render a comprehensive model performance dashboard in Streamlit."""
    st.header("📊 Model Performance Dashboard")
    
    # Get available data
    tickers = get_available_tickers()
    if not tickers:
        st.warning("No model performance data available. Run some models first to see performance metrics.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        tickers,
        index=0
    )
    
    models = get_available_models(selected_ticker)
    selected_models = st.sidebar.multiselect(
        "Select Models",
        models,
        default=models[:5] if len(models) > 5 else models
    )
    
    days_back = st.sidebar.slider(
        "Days Back",
        min_value=1,
        max_value=365,
        value=30
    )
    
    # Get performance data
    df = get_model_performance_history(
        ticker=selected_ticker,
        days_back=days_back
    )
    
    if selected_models:
        df = df[df['model_name'].isin(selected_models)]
    
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Display best models for this ticker
    st.subheader(f"🏆 Best Models for {selected_ticker}")
    best_models = get_best_models(selected_ticker)
    
    if best_models:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if best_models.get("best_sharpe", {}).get("model"):
                st.metric(
                    "Best Sharpe",
                    f"{best_models['best_sharpe']['value']:.3f}",
                    best_models['best_sharpe']['model']
                )
            
            if best_models.get("best_total_return", {}).get("model"):
                st.metric(
                    "Best Return",
                    f"{best_models['best_total_return']['value']:.1%}",
                    best_models['best_total_return']['model']
                )
        
        with col2:
            if best_models.get("best_mse", {}).get("model"):
                st.metric(
                    "Best MSE",
                    f"{best_models['best_mse']['value']:.4f}",
                    best_models['best_mse']['model']
                )
            
            if best_models.get("best_win_rate", {}).get("model"):
                st.metric(
                    "Best Win Rate",
                    f"{best_models['best_win_rate']['value']:.1%}",
                    best_models['best_win_rate']['model']
                )
        
        with col3:
            if best_models.get("best_drawdown", {}).get("model"):
                st.metric(
                    "Best Drawdown",
                    f"{best_models['best_drawdown']['value']:.1%}",
                    best_models['best_drawdown']['model']
                )
            
            if best_models.get("best_accuracy", {}).get("model"):
                st.metric(
                    "Best Accuracy",
                    f"{best_models['best_accuracy']['value']:.1%}",
                    best_models['best_accuracy']['model']
                )
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Trends", "📊 Metrics Comparison", "🏆 Leaderboard", "📋 Raw Data"])
    
    with tab1:
        st.subheader("Performance Trends Over Time")
        
        # Create time series plots
        metrics_to_plot = ['sharpe', 'mse', 'total_return', 'win_rate']
        
        for metric in metrics_to_plot:
            if metric in df.columns and not df[metric].isna().all():
                fig = st.line_chart(
                    df.set_index('timestamp')[['model_name', metric]].pivot(
                        columns='model_name', values=metric
                    ),
                    use_container_width=True
                )
                st.caption(f"{metric.replace('_', ' ').title()} over time")
    
    with tab2:
        st.subheader("Metrics Comparison")
        
        # Create comparison charts
        if not df.empty:
            # Average metrics by model
            avg_metrics = df.groupby('model_name')[['sharpe', 'mse', 'total_return', 'win_rate', 'accuracy']].mean()
            
            # Display as a heatmap
            st.dataframe(avg_metrics.style.format({
                'sharpe': '{:.3f}',
                'mse': '{:.4f}',
                'total_return': '{:.1%}',
                'win_rate': '{:.1%}',
                'accuracy': '{:.1%}'
            }))
    
    with tab3:
        st.subheader("Model Leaderboard")
        
        if not df.empty:
            # Calculate average performance by model
            leaderboard = df.groupby('model_name').agg({
                'sharpe': ['mean', 'std', 'count'],
                'mse': ['mean', 'std'],
                'total_return': ['mean', 'std'],
                'win_rate': ['mean', 'std'],
                'accuracy': ['mean', 'std']
            }).round(4)
            
            # Flatten column names
            leaderboard.columns = ['_'.join(col).strip() for col in leaderboard.columns]
            
            # Sort by average Sharpe ratio
            if 'sharpe_mean' in leaderboard.columns:
                leaderboard = leaderboard.sort_values('sharpe_mean', ascending=False)
            
            st.dataframe(leaderboard)
    
    with tab4:
        st.subheader("Raw Performance Data")
        
        # Display raw data with formatting
        display_df = df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format numeric columns
        numeric_columns = ['sharpe', 'mse', 'total_return', 'win_rate', 'accuracy', 'drawdown']
        for col in numeric_columns:
            if col in display_df.columns:
                if col in ['total_return', 'win_rate', 'accuracy', 'drawdown']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
                else:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Performance Data",
            data=csv,
            file_name=f"model_performance_{selected_ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def render_best_models_summary():
    """Render a summary of best models across all tickers."""
    st.header("🏆 Best Models Summary")
    
    best_models = get_best_models()
    if not best_models:
        st.info("No best models data available.")
        return
    
    # Create summary table
    summary_data = []
    
    for ticker, metrics in best_models.items():
        for metric_name, metric_data in metrics.items():
            if metric_data.get("model"):
                summary_data.append({
                    "Ticker": ticker,
                    "Metric": metric_name.replace("best_", "").replace("_", " ").title(),
                    "Model": metric_data["model"],
                    "Value": metric_data["value"],
                    "Date": metric_data["timestamp"][:10] if metric_data["timestamp"] else ""
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Format values
        for col in ["Value"]:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].apply(
                    lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
                )
        
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No best models data available.")


# Example usage function
def example_usage():
    """Example of how to use the model logging functionality."""
    
    # Log performance for different models
    log_model_performance(
        model_name="LSTM_v1",
        ticker="AAPL",
        sharpe=1.85,
        mse=0.0234,
        drawdown=-0.12,
        total_return=0.25,
        win_rate=0.68,
        accuracy=0.72,
        notes="LSTM model with 50 epochs"
    )
    
    log_model_performance(
        model_name="XGBoost_v2",
        ticker="AAPL",
        sharpe=2.1,
        mse=0.0189,
        drawdown=-0.08,
        total_return=0.31,
        win_rate=0.75,
        accuracy=0.78,
        notes="XGBoost with hyperparameter tuning"
    )
    
    log_model_performance(
        model_name="Transformer_v1",
        ticker="AAPL",
        sharpe=1.95,
        mse=0.0212,
        drawdown=-0.15,
        total_return=0.28,
        win_rate=0.71,
        accuracy=0.74,
        notes="Transformer model with attention mechanism"
    )
    
    print("Example performance data logged successfully!")
    print("Use render_model_performance_dashboard() to view in Streamlit")


if __name__ == "__main__":
    # Run example if executed directly
    example_usage() 