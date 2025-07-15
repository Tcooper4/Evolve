"""Performance logging utilities with persistent score tracking."""

import json
import logging
import pickle
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


class ModelScoreTracker:
    """Model score tracker with persistent storage."""
    
    def __init__(self, storage_path: str = "data/model_scores"):
        """Initialize the model score tracker.
        
        Args:
            storage_path: Path to store score data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.scores_file = self.storage_path / "model_scores.json"
        self.history_file = self.storage_path / "score_history.pkl"
        self.model_scores: Dict[str, Dict[str, Any]] = {}
        self.score_history: List[Dict[str, Any]] = []
        
        # Load existing data
        self.load()
        
    def update_score(
        self, 
        model_name: str, 
        metric_name: str, 
        score: float, 
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update a model's score for a specific metric.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the metric
            score: Score value
            timestamp: Timestamp for the score
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Initialize model if not exists
        if model_name not in self.model_scores:
            self.model_scores[model_name] = {
                "metrics": {},
                "last_updated": timestamp.isoformat(),
                "total_updates": 0
            }
            
        # Update score
        if metric_name not in self.model_scores[model_name]["metrics"]:
            self.model_scores[model_name]["metrics"][metric_name] = {
                "current_score": score,
                "best_score": score,
                "worst_score": score,
                "avg_score": score,
                "score_history": [],
                "last_updated": timestamp.isoformat()
            }
        else:
            metric_data = self.model_scores[model_name]["metrics"][metric_name]
            metric_data["current_score"] = score
            metric_data["best_score"] = max(metric_data["best_score"], score)
            metric_data["worst_score"] = min(metric_data["worst_score"], score)
            metric_data["score_history"].append({
                "score": score,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {}
            })
            
            # Calculate running average
            scores = [h["score"] for h in metric_data["score_history"]]
            metric_data["avg_score"] = np.mean(scores)
            metric_data["last_updated"] = timestamp.isoformat()
            
        # Update model metadata
        self.model_scores[model_name]["last_updated"] = timestamp.isoformat()
        self.model_scores[model_name]["total_updates"] += 1
        
        # Record in history
        history_entry = {
            "timestamp": timestamp.isoformat(),
            "model_name": model_name,
            "metric_name": metric_name,
            "score": score,
            "metadata": metadata or {}
        }
        self.score_history.append(history_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self.score_history) > 1000:
            self.score_history = self.score_history[-1000:]
            
        logger.debug(f"Updated score for {model_name}.{metric_name}: {score}")
        
    def get_score(self, model_name: str, metric_name: str) -> Optional[float]:
        """Get current score for a model and metric.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the metric
            
        Returns:
            Current score or None if not found
        """
        if (model_name in self.model_scores and 
            metric_name in self.model_scores[model_name]["metrics"]):
            return self.model_scores[model_name]["metrics"][metric_name]["current_score"]
        return None
        
    def get_model_scores(self, model_name: str) -> Dict[str, Any]:
        """Get all scores for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with all metric scores
        """
        return self.model_scores.get(model_name, {})
        
    def get_best_model(self, metric_name: str) -> Optional[str]:
        """Get the best performing model for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Name of the best model or None
        """
        best_model = None
        best_score = float('-inf')
        
        for model_name, model_data in self.model_scores.items():
            if metric_name in model_data["metrics"]:
                score = model_data["metrics"][metric_name]["current_score"]
                if score > best_score:
                    best_score = score
                    best_model = model_name
                    
        return best_model
        
    def get_score_summary(self) -> Dict[str, Any]:
        """Get summary of all model scores.
        
        Returns:
            Dictionary with score summary
        """
        summary = {
            "total_models": len(self.model_scores),
            "total_history_entries": len(self.score_history),
            "models": {}
        }
        
        for model_name, model_data in self.model_scores.items():
            summary["models"][model_name] = {
                "metrics_count": len(model_data["metrics"]),
                "total_updates": model_data["total_updates"],
                "last_updated": model_data["last_updated"],
                "metrics": {}
            }
            
            for metric_name, metric_data in model_data["metrics"].items():
                summary["models"][model_name]["metrics"][metric_name] = {
                    "current_score": metric_data["current_score"],
                    "best_score": metric_data["best_score"],
                    "worst_score": metric_data["worst_score"],
                    "avg_score": metric_data["avg_score"],
                    "history_count": len(metric_data["score_history"])
                }
                
        return summary
        
    def save(self, filepath: Optional[str] = None):
        """Save model scores to file.
        
        Args:
            filepath: Optional custom filepath
        """
        try:
            if filepath is None:
                filepath = self.scores_file
                
            # Save current scores
            with open(filepath, 'w') as f:
                json.dump(self.model_scores, f, indent=2)
                
            # Save history
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.score_history, f)
                
            logger.info(f"Saved model scores to {filepath}")
            logger.info(f"Saved score history to {self.history_file}")
            
        except Exception as e:
            logger.error(f"Could not save model scores: {e}")
            
    def load(self, filepath: Optional[str] = None):
        """Load model scores from file.
        
        Args:
            filepath: Optional custom filepath
        """
        try:
            if filepath is None:
                filepath = self.scores_file
                
            # Load current scores
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    self.model_scores = json.load(f)
                    
            # Load history
            if self.history_file.exists():
                with open(self.history_file, 'rb') as f:
                    self.score_history = pickle.load(f)
                    
            logger.info(f"Loaded model scores from {filepath}")
            logger.info(f"Loaded {len(self.score_history)} history entries")
            
        except Exception as e:
            logger.warning(f"Could not load model scores: {e}")
            self.model_scores = {}
            self.score_history = []
            
    def export_scores(self, filepath: str, format: str = "csv"):
        """Export scores to file.
        
        Args:
            filepath: Output filepath
            format: Export format ('csv', 'json', 'excel')
        """
        try:
            if format == "csv":
                # Create DataFrame for export
                export_data = []
                for model_name, model_data in self.model_scores.items():
                    for metric_name, metric_data in model_data["metrics"].items():
                        export_data.append({
                            "model_name": model_name,
                            "metric_name": metric_name,
                            "current_score": metric_data["current_score"],
                            "best_score": metric_data["best_score"],
                            "worst_score": metric_data["worst_score"],
                            "avg_score": metric_data["avg_score"],
                            "last_updated": metric_data["last_updated"]
                        })
                        
                df = pd.DataFrame(export_data)
                df.to_csv(filepath, index=False)
                
            elif format == "json":
                with open(filepath, 'w') as f:
                    json.dump(self.model_scores, f, indent=2)
                    
            elif format == "excel":
                # Create DataFrame for export
                export_data = []
                for model_name, model_data in self.model_scores.items():
                    for metric_name, metric_data in model_data["metrics"].items():
                        export_data.append({
                            "model_name": model_name,
                            "metric_name": metric_name,
                            "current_score": metric_data["current_score"],
                            "best_score": metric_data["best_score"],
                            "worst_score": metric_data["worst_score"],
                            "avg_score": metric_data["avg_score"],
                            "last_updated": metric_data["last_updated"]
                        })
                        
                df = pd.DataFrame(export_data)
                df.to_excel(filepath, index=False)
                
            logger.info(f"Exported scores to {filepath}")
            
        except Exception as e:
            logger.error(f"Could not export scores: {e}")


# Global instance
_score_tracker = ModelScoreTracker()


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
    
    # Update score tracker
    for metric_name, score in performance_metrics.items():
        _score_tracker.update_score(strategy_name, metric_name, score, metadata=metadata)

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
    
    # Update score tracker
    for metric_name, score in metrics.items():
        _score_tracker.update_score(model, metric_name, score, metadata=metadata)

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

        # Save chart
        chart_path = f"charts/performance_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        Path(chart_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance chart saved to {chart_path}")
        return chart_path

    except Exception as e:
        logger.error(f"Error creating performance chart: {e}")
        return None


def create_streamlit_performance_dashboard():
    """Create a Streamlit performance dashboard."""
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit not available for dashboard creation")
        return

    try:
        st.title("Performance Dashboard")

        # Sidebar filters
        st.sidebar.header("Filters")
        days_back = st.sidebar.slider("Days Back", 1, 365, 30)
        metric_filter = st.sidebar.selectbox(
            "Metric",
            ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
        )

        # Get performance data
        data = get_performance_data(days_back=days_back)

        if not data:
            st.warning("No performance data available")
            return

        # Create DataFrame
        df_data = []
        for record in data:
            if metric_filter in record["metrics"]:
                df_data.append({
                    "Timestamp": datetime.fromisoformat(record["timestamp"]),
                    "Strategy": record.get("strategy", "Unknown"),
                    "Ticker": record.get("ticker", "Unknown"),
                    "Model": record.get("model", "Unknown"),
                    "Metric": record["metrics"][metric_filter]
                })

        if not df_data:
            st.warning(f"No data for metric: {metric_filter}")
            return

        df = pd.DataFrame(df_data)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{df['Metric'].mean():.4f}")
        with col2:
            st.metric("Best", f"{df['Metric'].max():.4f}")
        with col3:
            st.metric("Worst", f"{df['Metric'].min():.4f}")

        # Performance chart
        st.subheader(f"{metric_filter.replace('_', ' ').title()} Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Timestamp"], df["Metric"], marker="o", alpha=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric_filter.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Data table
        st.subheader("Performance Data")
        st.dataframe(df)

    except Exception as e:
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
        filename: Output filename
        format: Export format ('csv', 'json', 'excel')
        strategy_name: Filter by strategy name
        ticker: Filter by ticker
        model: Filter by model
        days_back: Number of days to look back

    Returns:
        Path to exported file or None if failed
    """
    try:
        # Get filtered data
        data = get_performance_data(strategy_name, ticker, model, days_back)

        if not data:
            logger.warning("No performance data to export")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_export_{timestamp}.{format}"

        # Create export directory
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        filepath = export_dir / filename

        # Export based on format
        if format == "csv":
            # Flatten data for CSV export
            flat_data = []
            for record in data:
                base_record = {
                    "timestamp": record["timestamp"],
                    "strategy": record.get("strategy", ""),
                    "ticker": record.get("ticker", ""),
                    "model": record.get("model", ""),
                    "agentic": record.get("agentic", False),
                }

                # Add metrics
                for metric_name, metric_value in record["metrics"].items():
                    base_record[f"metric_{metric_name}"] = metric_value

                flat_data.append(base_record)

            df = pd.DataFrame(flat_data)
            df.to_csv(filepath, index=False)

        elif format == "json":
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "excel":
            # Flatten data for Excel export
            flat_data = []
            for record in data:
                base_record = {
                    "timestamp": record["timestamp"],
                    "strategy": record.get("strategy", ""),
                    "ticker": record.get("ticker", ""),
                    "model": record.get("model", ""),
                    "agentic": record.get("agentic", False),
                }

                # Add metrics
                for metric_name, metric_value in record["metrics"].items():
                    base_record[f"metric_{metric_name}"] = metric_value

                flat_data.append(base_record)

            df = pd.DataFrame(flat_data)
            df.to_excel(filepath, index=False)

        logger.info(f"Performance data exported to {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error exporting performance data: {e}")
        return None


# Convenience functions for score tracker
def save_model_scores(filepath: Optional[str] = None):
    """Save model scores (convenience function)."""
    _score_tracker.save(filepath)


def load_model_scores(filepath: Optional[str] = None):
    """Load model scores (convenience function)."""
    _score_tracker.load(filepath)


def get_model_score(model_name: str, metric_name: str) -> Optional[float]:
    """Get model score (convenience function)."""
    return _score_tracker.get_score(model_name, metric_name)


def get_score_summary() -> Dict[str, Any]:
    """Get score summary (convenience function)."""
    return _score_tracker.get_score_summary()
