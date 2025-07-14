"""Performance logging utilities."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Try to import visualization libraries
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Performance data storage
performance_data = []


def log_strategy_performance(
    strategy_name: str,
    performance_metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log strategy performance metrics.

    Args:
        strategy_name: Name of the strategy
        performance_metrics: Dictionary of performance metrics
        metadata: Additional metadata
    """
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy_name,
        "metrics": performance_metrics,
        "metadata": metadata or {},
    }

    # Store in memory for trending
    performance_data.append(log_data)

    logger.info(f"Strategy Performance: {json.dumps(log_data)}")


def log_performance(
    ticker: str,
    model: str,
    agentic: bool,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log performance metrics for a specific ticker and model.

    Args:
        ticker: Stock ticker symbol
        model: Model name used
        agentic: Whether agentic selection was used
        metrics: Dictionary of performance metrics
        metadata: Additional metadata
    """
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "model": model,
        "agentic": agentic,
        "metrics": metrics,
        "metadata": metadata or {},
    }

    # Store in memory for trending
    performance_data.append(log_data)

    logger.info(f"Performance Log: {json.dumps(log_data)}")


def get_performance_data(
    strategy_name: Optional[str] = None,
    ticker: Optional[str] = None,
    model: Optional[str] = None,
    days_back: int = 30,
) -> List[Dict[str, Any]]:
    """Get performance data with optional filtering.

    Args:
        strategy_name: Filter by strategy name
        ticker: Filter by ticker
        model: Filter by model
        days_back: Number of days to look back

    Returns:
        List of performance records
    """
    cutoff_date = datetime.now() - timedelta(days=days_back)

    filtered_data = []
    for record in performance_data:
        record_date = datetime.fromisoformat(record["timestamp"])
        if record_date < cutoff_date:
            continue

        if strategy_name and record.get("strategy") != strategy_name:
            continue

        if ticker and record.get("ticker") != ticker:
            continue

        if model and record.get("model") != model:
            continue

        filtered_data.append(record)

    return filtered_data


def create_performance_trend_chart(
    metric_name: str,
    strategy_name: Optional[str] = None,
    ticker: Optional[str] = None,
    model: Optional[str] = None,
    days_back: int = 30,
    chart_type: str = "line",
) -> Optional[str]:
    """Create a performance trend chart using matplotlib.

    Args:
        metric_name: Name of the metric to plot
        strategy_name: Filter by strategy name
        ticker: Filter by ticker
        model: Filter by model
        days_back: Number of days to look back
        chart_type: Type of chart ('line', 'bar', 'scatter')

    Returns:
        Path to saved chart file or None if failed
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available for chart creation")
        return None

    try:
        # Get filtered data
        data = get_performance_data(strategy_name, ticker, model, days_back)

        if not data:
            logger.warning("No performance data available for chart")
            return None

        # Extract timestamps and metric values
        timestamps = []
        values = []

        for record in data:
            try:
                timestamp = datetime.fromisoformat(record["timestamp"])
                value = record["metrics"].get(metric_name)

                if value is not None:
                    timestamps.append(timestamp)
                    values.append(float(value))
            except (ValueError, TypeError):
                continue

        if not values:
            logger.warning(f"No valid data for metric: {metric_name}")
            return None

        # Create chart
        plt.figure(figsize=(12, 6))

        if chart_type == "line":
            plt.plot(timestamps, values, marker="o", linewidth=2, markersize=4)
        elif chart_type == "bar":
            plt.bar(timestamps, values, alpha=0.7)
        elif chart_type == "scatter":
            plt.scatter(timestamps, values, alpha=0.6)

        # Format chart
        plt.title(f"{metric_name} Performance Trend")
        plt.xlabel("Time")
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)

        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(
            mdates.DayLocator(interval=max(1, days_back // 7))
        )
        plt.xticks(rotation=45)

        # Add trend line
        if len(values) > 1:
            z = np.polyfit(range(len(values)), values, 1)
            p = np.poly1d(z)
            plt.plot(timestamps, p(range(len(values))), "r--", alpha=0.8, label="Trend")
            plt.legend()

        plt.tight_layout()

        # Save chart
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)

        filename = f"performance_trend_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = charts_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance trend chart saved: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error creating performance trend chart: {e}")
        return None


def create_streamlit_performance_dashboard():
    """Create a Streamlit dashboard for performance visualization."""
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit not available for dashboard creation")
        return

    try:
        st.title("Performance Trending Dashboard")

        # Sidebar filters
        st.sidebar.header("Filters")

        # Get unique values for filters
        strategies = list(
            set(
                record.get("strategy")
                for record in performance_data
                if record.get("strategy")
            )
        )
        tickers = list(
            set(
                record.get("ticker")
                for record in performance_data
                if record.get("ticker")
            )
        )
        models = list(
            set(
                record.get("model")
                for record in performance_data
                if record.get("model")
            )
        )

        selected_strategy = st.sidebar.selectbox("Strategy", ["All"] + strategies)
        selected_ticker = st.sidebar.selectbox("Ticker", ["All"] + tickers)
        selected_model = st.sidebar.selectbox("Model", ["All"] + models)
        days_back = st.sidebar.slider("Days Back", 7, 90, 30)

        # Get filtered data
        strategy_filter = selected_strategy if selected_strategy != "All" else None
        ticker_filter = selected_ticker if selected_ticker != "All" else None
        model_filter = selected_model if selected_model != "All" else None

        data = get_performance_data(
            strategy_filter, ticker_filter, model_filter, days_back
        )

        if not data:
            st.warning("No performance data available for selected filters")
            return

        # Convert to DataFrame for easier manipulation
        df_data = []
        for record in data:
            for metric_name, value in record["metrics"].items():
                df_data.append(
                    {
                        "timestamp": datetime.fromisoformat(record["timestamp"]),
                        "metric": metric_name,
                        "value": float(value),
                        "strategy": record.get("strategy", ""),
                        "ticker": record.get("ticker", ""),
                        "model": record.get("model", ""),
                    }
                )

        df = pd.DataFrame(df_data)

        # Performance overview
        st.header("Performance Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            if not df.empty:
                avg_value = df["value"].mean()
                st.metric("Average Performance", f"{avg_value:.4f}")

        with col2:
            if not df.empty:
                latest_value = df.groupby("metric")["value"].last().mean()
                st.metric("Latest Performance", f"{latest_value:.4f}")

        with col3:
            if not df.empty:
                trend = (
                    df.groupby("metric")["value"]
                    .apply(lambda x: x.iloc[-1] - x.iloc[0])
                    .mean()
                )
                st.metric("Trend", f"{trend:+.4f}")

        # Performance trends by metric
        st.header("Performance Trends by Metric")

        metrics = df["metric"].unique()
        selected_metric = st.selectbox("Select Metric", metrics)

        metric_data = df[df["metric"] == selected_metric]

        if not metric_data.empty:
            # Create trend chart
            fig, ax = plt.subplots(figsize=(10, 6))

            for strategy in metric_data["strategy"].unique():
                if strategy:
                    strategy_data = metric_data[metric_data["strategy"] == strategy]
                    ax.plot(
                        strategy_data["timestamp"],
                        strategy_data["value"],
                        marker="o",
                        label=strategy,
                        alpha=0.7,
                    )

            ax.set_title(f"{selected_metric} Performance Trend")
            ax.set_xlabel("Time")
            ax.set_ylabel(selected_metric)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Performance statistics
            st.subheader("Performance Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Summary Statistics**")
                stats = metric_data["value"].describe()
                st.write(stats)

            with col2:
                st.write("**Recent Performance**")
                recent_data = metric_data.tail(10)
                st.dataframe(
                    recent_data[["timestamp", "value", "strategy"]].set_index(
                        "timestamp"
                    )
                )

        # Performance comparison
        st.header("Performance Comparison")

        if len(metrics) > 1:
            comparison_metric = st.selectbox(
                "Select Metric for Comparison", metrics, key="comparison"
            )

            comparison_data = df[df["metric"] == comparison_metric]

            if not comparison_data.empty:
                # Box plot
                fig, ax = plt.subplots(figsize=(10, 6))
                comparison_data.boxplot(column="value", by="strategy", ax=ax)
                ax.set_title(f"{comparison_metric} Distribution by Strategy")
                ax.set_xlabel("Strategy")
                ax.set_ylabel(comparison_metric)
                plt.xticks(rotation=45)
                st.pyplot(fig)

    except Exception as e:
        logger.error(f"Error creating Streamlit dashboard: {e}")
        st.error(f"Error creating dashboard: {e}")


def export_performance_data(
    filename: str = None,
    format: str = "csv",
    strategy_name: Optional[str] = None,
    ticker: Optional[str] = None,
    model: Optional[str] = None,
    days_back: int = 30,
) -> Optional[str]:
    """Export performance data to file.

    Args:
        filename: Output filename (auto-generated if None)
        format: Export format ('csv', 'json', 'excel')
        strategy_name: Filter by strategy name
        ticker: Filter by ticker
        model: Filter by model
        days_back: Number of days to look back

    Returns:
        Path to exported file or None if failed
    """
    try:
        data = get_performance_data(strategy_name, ticker, model, days_back)

        if not data:
            logger.warning("No performance data to export")
            return None

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_export_{timestamp}.{format}"

        # Create export directory
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)

        filepath = export_dir / filename

        if format == "csv":
            # Convert to DataFrame and export
            df_data = []
            for record in data:
                for metric_name, value in record["metrics"].items():
                    df_data.append(
                        {
                            "timestamp": record["timestamp"],
                            "metric": metric_name,
                            "value": float(value),
                            "strategy": record.get("strategy", ""),
                            "ticker": record.get("ticker", ""),
                            "model": record.get("model", ""),
                            "metadata": json.dumps(record.get("metadata", {})),
                        }
                    )

            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)

        elif format == "json":
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "excel":
            # Convert to DataFrame and export
            df_data = []
            for record in data:
                for metric_name, value in record["metrics"].items():
                    df_data.append(
                        {
                            "timestamp": record["timestamp"],
                            "metric": metric_name,
                            "value": float(value),
                            "strategy": record.get("strategy", ""),
                            "ticker": record.get("ticker", ""),
                            "model": record.get("model", ""),
                            "metadata": json.dumps(record.get("metadata", {})),
                        }
                    )

            df = pd.DataFrame(df_data)
            df.to_excel(filepath, index=False)

        logger.info(f"Performance data exported to: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error exporting performance data: {e}")
        return None
