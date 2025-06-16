"""Core performance tracking and evaluation module."""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional

# Configure logging
log_file = Path("memory/logs/performance.log")
logger = logging.getLogger("performance")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Performance targets
TARGETS = {
    "sharpe": 1.3,
    "drawdown": 0.25,
    "mse": 0.05
}

# Default metrics for when data is missing
DEFAULT_METRICS = {
    "sharpe": 0.0,
    "drawdown": 0.0,
    "mse": 0.0,
    "accuracy": 0.0
}

def log_performance(
    ticker: str,
    model: str,
    strategy: str,
    sharpe: Optional[float] = None,
    drawdown: Optional[float] = None,
    mse: Optional[float] = None,
    accuracy: Optional[float] = None,
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
        Dictionary of rolling averages
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
        for metric in ['sharpe', 'drawdown', 'mse', 'accuracy']:
            try:
                metrics[metric] = recent_data[metric].mean()
            except (KeyError, TypeError):
                metrics[metric] = DEFAULT_METRICS[metric]
                logger.warning(f"Could not calculate {metric}, using default value")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating rolling metrics: {str(e)}")
        return DEFAULT_METRICS.copy()

def evaluate_performance() -> Dict[str, Any]:
    """Evaluate current performance against goals.
    
    Returns:
        Dictionary containing goal status and metrics
    """
    try:
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
        
        if metrics["sharpe"] < TARGETS["sharpe"]:
            issues.append(f"Sharpe ratio {metrics['sharpe']:.2f} below target {TARGETS['sharpe']}")
            status = "Underperforming"
            
        if metrics["drawdown"] > TARGETS["drawdown"]:
            issues.append(f"Drawdown {metrics['drawdown']:.2f} above target {TARGETS['drawdown']}")
            status = "Underperforming"
            
        if metrics["mse"] > TARGETS["mse"]:
            issues.append(f"MSE {metrics['mse']:.2f} above target {TARGETS['mse']}")
            status = "Underperforming"
        
        # Prepare status report
        status_report = {
            "target_sharpe": TARGETS["sharpe"],
            "current_avg_sharpe": round(metrics["sharpe"], 2),
            "target_drawdown": TARGETS["drawdown"],
            "current_avg_drawdown": round(metrics["drawdown"], 2),
            "target_mse": TARGETS["mse"],
            "current_avg_mse": round(metrics["mse"], 2),
            "current_avg_accuracy": round(metrics["accuracy"], 2),
            "goal_status": status,
            "issues": issues,
            "last_evaluated": datetime.now().isoformat()
        }
        
        # Log status
        if status == "Underperforming":
            logger.warning(f"Performance issues detected: {', '.join(issues)}")
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