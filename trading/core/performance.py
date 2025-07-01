"""Core performance tracking and evaluation module."""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Configurable global for classification tasks ---
CLASSIFICATION = False  # Set True to enable precision/recall metrics

# Configure logging
log_file = Path("memory/logs/performance.log")
logger = logging.getLogger("performance")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# --- Dynamic Target Loading ---
def load_targets() -> Dict[str, float]:
    """Load performance targets from JSON, fallback to defaults.
    
    Returns:
        Dictionary containing performance targets for various metrics.
    """
    default_targets = {
        "sharpe": 1.3,
        "drawdown": 0.25,
        "mse": 0.05,
        "r2": 0.5,
        "precision": 0.7,
        "recall": 0.7
    }
    target_path = Path("memory/goals/targets.json")
    if target_path.exists():
        try:
            with open(target_path, "r") as f:
                targets = json.load(f)
            logger.info("Loaded targets from targets.json")
            return {'success': True, 'result': {**default_targets, **targets}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error loading targets.json: {e}")
    return default_targets.copy()

# --- Target Override ---
def update_targets(new_targets: Dict[str, float]) -> None:
    """Update and save targets to JSON.
    
    Args:
        new_targets: Dictionary containing new performance targets to save.
    """
    target_path = Path("memory/goals/targets.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target_path, "w") as f:
            json.dump(new_targets, f, indent=4)
        logger.info("Updated targets.json with new targets.")
    except Exception as e:
        logger.error(f"Error updating targets.json: {e}")

# --- Default Metrics ---
DEFAULT_METRICS = {
    "sharpe": 0.0,
    "drawdown": 0.0,
    "mse": 0.0,
    "accuracy": 0.0,
    "r2": 0.0,
    "precision": 0.0,
    "recall": 0.0
}

def log_performance(
    ticker: str,
    model: str,
    strategy: str,
    sharpe: Optional[float] = None,
    drawdown: Optional[float] = None,
    mse: Optional[float] = None,
    accuracy: Optional[float] = None,
    r2: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    notes: Optional[str] = None

) -> None:
    """Log performance metrics to CSV file.
    
    Args:
        ticker: Stock ticker symbol
        model: Model name/type
        strategy: Strategy name/type
        sharpe: Sharpe ratio
        drawdown: Maximum drawdown
        mse: Mean squared error
        accuracy: Prediction accuracy
        r2: R-squared value
        precision: Precision score
        recall: Recall score
        notes: Additional notes
    """
    try:
        # Create log directory if it doesn't exist
        log_dir = Path("memory/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "model": model,
            "strategy": strategy,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "mse": mse,
            "accuracy": accuracy,
            "r2": r2,
            "precision": precision,
            "recall": recall,
            "notes": notes
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([entry])
        
        # Append to CSV
        log_path = log_dir / "performance_log.csv"
        df.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
        
    except Exception as e:
        logger.error(f"Error logging performance: {str(e)}")

def calculate_rolling_metrics(df: pd.DataFrame, window: int = 7) -> Dict[str, float]:
    """Calculate rolling averages for key metrics.
    
    Args:
        df: Performance log DataFrame
        window: Rolling window size in days
        
    Returns:
        Dictionary of rolling averages for each metric
    """
    try:
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Get last N days of data
        recent_data = df.tail(window)
        
        # Calculate metrics with error handling
        metrics = {}
        for metric in DEFAULT_METRICS.keys():
            try:
                metrics[metric] = recent_data[metric].mean()
            except (KeyError, TypeError):
                metrics[metric] = DEFAULT_METRICS[metric]
                logger.warning(f"Could not calculate {metric}, using default value")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating rolling metrics: {str(e)}")
        return DEFAULT_METRICS.copy()

def evaluate_performance(classification: Optional[bool] = None) -> Dict[str, Any]:
    """Evaluate current performance against goals.
    
    Args:
        classification: Whether to include classification metrics in evaluation.
            If None, uses the global CLASSIFICATION setting.
    
    Returns:
        Dictionary containing goal status and metrics
    """
    try:
        targets = load_targets()
        # Load performance log
        log_path = Path("memory/logs/performance_log.csv")
        if not log_path.exists():
            logger.warning("Performance log not found")
            return {
                "status": "No Data",
                "message": "Performance log not found",
                "timestamp": datetime.now().isoformat(),
                **{f"current_avg_{k}": v for k, v in DEFAULT_METRICS.items()}
            }
        
        try:
            df = pd.read_csv(log_path)
        except Exception as e:
            logger.error(f"Error reading performance log: {str(e)}")
            return {
                "status": "Error",
                "message": f"Error reading performance log: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                **{f"current_avg_{k}": v for k, v in DEFAULT_METRICS.items()}
            }
            
        if df.empty:
            logger.warning("Performance log is empty")
            return {
                "status": "No Data",
                "message": "Performance log is empty",
                "timestamp": datetime.now().isoformat(),
                **{f"current_avg_{k}": v for k, v in DEFAULT_METRICS.items()}
            }
        
        # Calculate rolling metrics
        metrics = calculate_rolling_metrics(df)
        
        # Evaluate against targets
        status = "Performing"
        issues = []
        
        if metrics["sharpe"] < targets["sharpe"]:
            issues.append(f"Sharpe ratio {metrics['sharpe']:.2f} below target {targets['sharpe']}")
            status = "Underperforming"
            
        if metrics["drawdown"] > targets["drawdown"]:
            issues.append(f"Drawdown {metrics['drawdown']:.2f} above target {targets['drawdown']}")
            status = "Underperforming"
            
        if metrics["mse"] > targets["mse"]:
            issues.append(f"MSE {metrics['mse']:.2f} above target {targets['mse']}")
            status = "Underperforming"
        
        if "r2" in metrics and metrics["r2"] < targets.get("r2", 0.0):
            issues.append(f"R2 {metrics['r2']:.2f} below target {targets.get('r2', 0.0)}")
            status = "Underperforming"
        
        # Classification metrics
        use_classification = classification if classification is not None else CLASSIFICATION
        if use_classification:
            if "precision" in metrics and metrics["precision"] < targets.get("precision", 0.0):
                issues.append(f"Precision {metrics['precision']:.2f} below target {targets.get('precision', 0.0)}")
                status = "Underperforming"
            if "recall" in metrics and metrics["recall"] < targets.get("recall", 0.0):
                issues.append(f"Recall {metrics['recall']:.2f} below target {targets.get('recall', 0.0)}")
                status = "Underperforming"
        
        # Prepare status report
        status_report = {
            "targets": targets,
            **{f"current_avg_{k}": round(metrics[k], 4) for k in DEFAULT_METRICS.keys()},
            "goal_status": status,
            "issues": issues,
            "last_evaluated": datetime.now().isoformat()
        }
        
        # Log status
        if status == "Underperforming":
            logger.warning(f"Performance issues detected: {', '.join(issues)}")
            try:
                from core.agents import handle_underperformance
                handle_underperformance(status_report)
            except Exception as e:
                logger.error(f"Error calling handle_underperformance: {e}")
        else:
            logger.info("All performance targets met")
        
        # Save status to JSON
        goals_dir = Path("memory/goals")
        goals_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(goals_dir / "status.json", "w") as f:
                json.dump(status_report, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving status report: {str(e)}")
        
        return status_report
        
    except Exception as e:
        error_msg = f"Error evaluating performance: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "Error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
            **{f"current_avg_{k}": v for k, v in DEFAULT_METRICS.items()}
        } 

def plot_performance_trends(log_path: str = "memory/logs/performance_log.csv") -> None:
    """Plot Sharpe, Drawdown, and MSE over time and save as PNG.
    
    Args:
        log_path: Path to the performance log CSV file
    """
    try:
        if not os.path.exists(log_path):
            logger.warning(f"Performance log not found at {log_path}")
            return
        df = pd.read_csv(log_path)
        if df.empty:
            logger.warning("Performance log is empty")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        fig = go.Figure()
        for metric in ["sharpe", "drawdown", "mse"]:
            if metric in df.columns:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[metric], mode='lines+markers', name=metric.capitalize()))
        fig.update_layout(title="Performance Trends", xaxis_title="Time", yaxis_title="Metric Value")
        out_path = "memory/goals/performance_plot.png"
        pio.write_image(fig, out_path)
        logger.info(f"Saved performance plot to {out_path}")
    except Exception as e:
        logger.error(f"Error plotting performance trends: {e}") 