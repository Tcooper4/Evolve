"""
Core Performance Tracking and Evaluation Module

This module provides comprehensive performance tracking, evaluation,
and visualization capabilities for the trading system. It includes metrics calculation,
goal evaluation, and performance trend analysis.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import os

# Configure logging
logger = logging.getLogger(__name__)

# --- Configurable Settings ---
CLASSIFICATION = False  # Set True to enable precision/recall metrics

# --- Data Models ---
@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    sharpe: float = 0.0
    drawdown: float = 0.0
    mse: float = 0.0
    accuracy: float = 0.0
    r2: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

@dataclass
class PerformanceEntry:
    """Data class for performance log entries."""
    timestamp: str
    ticker: str
    model: str
    strategy: str
    sharpe: Optional[float] = None
    drawdown: Optional[float] = None
    mse: Optional[float] = None
    accuracy: Optional[float] = None
    r2: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    notes: Optional[str] = None

@dataclass
class PerformanceStatus:
    """Data class for performance status evaluation."""
    status: str
    message: str
    timestamp: str
    targets: Dict[str, float]
    current_metrics: Dict[str, float]
    issues: List[str]
    last_evaluated: str

# --- Default Values ---
DEFAULT_METRICS = PerformanceMetrics()
DEFAULT_TARGETS = {
    "sharpe": 1.3,
    "drawdown": 0.25,
    "mse": 0.05,
    "r2": 0.5,
    "precision": 0.7,
    "recall": 0.7
}

# --- File Paths ---
class PerformancePaths:
    """Centralized file path management for performance tracking."""
    
    @staticmethod
    def get_log_dir() -> Path:
        """Get the performance log directory."""
        return Path("memory/logs")
    
    @staticmethod
    def get_goals_dir() -> Path:
        """Get the goals directory."""
        return Path("memory/goals")
    
    @staticmethod
    def get_performance_log() -> Path:
        """Get the performance log file path."""
        return PerformancePaths.get_log_dir() / "performance_log.csv"
    
    @staticmethod
    def get_targets_file() -> Path:
        """Get the targets file path."""
        return PerformancePaths.get_goals_dir() / "targets.json"
    
    @staticmethod
    def get_status_file() -> Path:
        """Get the status file path."""
        return PerformancePaths.get_goals_dir() / "status.json"
    
    @staticmethod
    def get_performance_plot() -> Path:
        """Get the performance plot file path."""
        return PerformancePaths.get_goals_dir() / "performance_plot.png"

# --- Target Management ---
class TargetManager:
    """Manages performance targets and their persistence."""
    
    @staticmethod
    def load_targets() -> Dict[str, float]:
        """Load performance targets from JSON, fallback to defaults.
        
        Returns:
            Dictionary containing performance targets for various metrics.
        """
        target_path = PerformancePaths.get_targets_file()
        if target_path.exists():
            try:
                with open(target_path, "r") as f:
                    targets = json.load(f)
                logger.info("Loaded targets from targets.json")
                return {**DEFAULT_TARGETS, **targets}
            except Exception as e:
                logger.error(f"Error loading targets.json: {e}")
        
        return DEFAULT_TARGETS.copy()
    
    @staticmethod
    def update_targets(new_targets: Dict[str, float]) -> bool:
        """Update and save targets to JSON.
        
        Args:
            new_targets: Dictionary containing new performance targets to save.
            
        Returns:
            True if successful, False otherwise.
        """
        target_path = PerformancePaths.get_targets_file()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(target_path, "w") as f:
                json.dump(new_targets, f, indent=4)
            logger.info("Updated targets.json with new targets.")
            return True
        except Exception as e:
            logger.error(f"Error updating targets.json: {e}")
            return False

# --- Performance Logging ---
class PerformanceLogger:
    """Handles performance metric logging and persistence."""
    
    @staticmethod
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
    ) -> bool:
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
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create log directory if it doesn't exist
            log_dir = PerformancePaths.get_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare log entry
            entry = PerformanceEntry(
                timestamp=datetime.now().isoformat(),
                ticker=ticker,
                model=model,
                strategy=strategy,
                sharpe=sharpe,
                drawdown=drawdown,
                mse=mse,
                accuracy=accuracy,
                r2=r2,
                precision=precision,
                recall=recall,
                notes=notes
            )
            
            # Convert to DataFrame
            df = pd.DataFrame([asdict(entry)])
            
            # Append to CSV
            log_path = PerformancePaths.get_performance_log()
            df.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
            
            logger.info(f"Logged performance for {ticker} - {model} - {strategy}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging performance: {str(e)}")
            return False
    
    @staticmethod
    def load_performance_data() -> Optional[pd.DataFrame]:
        """Load performance data from CSV file.
        
        Returns:
            DataFrame containing performance data or None if error.
        """
        log_path = PerformancePaths.get_performance_log()
        if not log_path.exists():
            logger.warning("Performance log not found")
            return None
        
        try:
            df = pd.read_csv(log_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            return df
        except Exception as e:
            logger.error(f"Error reading performance log: {str(e)}")
            return None

# --- Metrics Calculation ---
class MetricsCalculator:
    """Handles calculation of various performance metrics."""
    
    @staticmethod
    def calculate_rolling_metrics(df: pd.DataFrame, window: int = 7) -> Dict[str, float]:
        """Calculate rolling averages for key metrics.
        
        Args:
            df: Performance log DataFrame
            window: Rolling window size in days
            
        Returns:
            Dictionary of rolling averages for each metric
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided for rolling metrics calculation")
                return asdict(DEFAULT_METRICS)
            
            # Get last N days of data
            recent_data = df.tail(window)
            
            # Calculate metrics with error handling
            metrics = {}
            for metric in asdict(DEFAULT_METRICS).keys():
                try:
                    if metric in recent_data.columns:
                        metrics[metric] = recent_data[metric].mean()
                    else:
                        metrics[metric] = getattr(DEFAULT_METRICS, metric)
                except (KeyError, TypeError) as e:
                    metrics[metric] = getattr(DEFAULT_METRICS, metric)
                    logger.warning(f"Could not calculate {metric}, using default value: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {str(e)}")
            return asdict(DEFAULT_METRICS)
    
    @staticmethod
    def evaluate_against_targets(metrics: Dict[str, float], targets: Dict[str, float], 
                               use_classification: bool = False) -> tuple[str, List[str]]:
        """Evaluate metrics against targets.
        
        Args:
            metrics: Current performance metrics
            targets: Target performance metrics
            use_classification: Whether to include classification metrics
            
        Returns:
            Tuple of (status, issues_list)
        """
        status = "Performing"
        issues = []
        
        # Core metrics evaluation
        if metrics.get("sharpe", 0) < targets.get("sharpe", 0):
            issues.append(f"Sharpe ratio {metrics.get('sharpe', 0):.2f} below target {targets.get('sharpe', 0)}")
            status = "Underperforming"
            
        if metrics.get("drawdown", 0) > targets.get("drawdown", 0):
            issues.append(f"Drawdown {metrics.get('drawdown', 0):.2f} above target {targets.get('drawdown', 0)}")
            status = "Underperforming"
            
        if metrics.get("mse", 0) > targets.get("mse", 0):
            issues.append(f"MSE {metrics.get('mse', 0):.2f} above target {targets.get('mse', 0)}")
            status = "Underperforming"
        
        if metrics.get("r2", 0) < targets.get("r2", 0):
            issues.append(f"R2 {metrics.get('r2', 0):.2f} below target {targets.get('r2', 0)}")
            status = "Underperforming"
        
        # Classification metrics
        if use_classification:
            if metrics.get("precision", 0) < targets.get("precision", 0):
                issues.append(f"Precision {metrics.get('precision', 0):.2f} below target {targets.get('precision', 0)}")
                status = "Underperforming"
            if metrics.get("recall", 0) < targets.get("recall", 0):
                issues.append(f"Recall {metrics.get('recall', 0):.2f} below target {targets.get('recall', 0)}")
                status = "Underperforming"
        
        return status, issues

# --- Performance Evaluation ---
class PerformanceEvaluator:
    """Handles comprehensive performance evaluation and status reporting."""
    
    @staticmethod
    def evaluate_performance(classification: Optional[bool] = None) -> Dict[str, Any]:
        """Evaluate current performance against goals.
        
        Args:
            classification: Whether to include classification metrics in evaluation.
                If None, uses the global CLASSIFICATION setting.
        
        Returns:
            Dictionary containing goal status and metrics
        """
        try:
            targets = TargetManager.load_targets()
            df = PerformanceLogger.load_performance_data()
            
            if df is None or df.empty:
                logger.warning("No performance data available for evaluation")
                return PerformanceEvaluator._create_no_data_response(targets)
            
            # Calculate rolling metrics
            metrics = MetricsCalculator.calculate_rolling_metrics(df)
            
            # Evaluate against targets
            use_classification = classification if classification is not None else CLASSIFICATION
            status, issues = MetricsCalculator.evaluate_against_targets(
                metrics, targets, use_classification
            )
            
            # Prepare status report
            status_report = {
                "targets": targets,
                **{f"current_avg_{k}": round(v, 4) for k, v in metrics.items()},
                "goal_status": status,
                "issues": issues,
                "last_evaluated": datetime.now().isoformat()
            }
            
            # Log status
            if status == "Underperforming":
                logger.warning(f"Performance issues detected: {', '.join(issues)}")
                PerformanceEvaluator._handle_underperformance(status_report)
            else:
                logger.info("All performance targets met")
            
            # Save status to JSON
            PerformanceEvaluator._save_status_report(status_report)
            
            return status_report
            
        except Exception as e:
            error_msg = f"Error evaluating performance: {str(e)}"
            logger.error(error_msg)
            return PerformanceEvaluator._create_error_response(error_msg)
    
    @staticmethod
    def _create_no_data_response(targets: Dict[str, float]) -> Dict[str, Any]:
        """Create response for no data scenario."""
        return {
            "status": "No Data",
            "message": "Performance log not found or empty",
            "timestamp": datetime.now().isoformat(),
            **{f"current_avg_{k}": v for k, v in asdict(DEFAULT_METRICS).items()}
        }
    
    @staticmethod
    def _create_error_response(error_msg: str) -> Dict[str, Any]:
        """Create response for error scenario."""
        return {
            "status": "Error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
            **{f"current_avg_{k}": v for k, v in asdict(DEFAULT_METRICS).items()}
        }
    
    @staticmethod
    def _handle_underperformance(status_report: Dict[str, Any]) -> None:
        """Handle underperformance scenarios."""
        try:
            from core.agents import handle_underperformance
            handle_underperformance(status_report)
        except ImportError:
            logger.warning("handle_underperformance function not available")
        except Exception as e:
            logger.error(f"Error calling handle_underperformance: {e}")
    
    @staticmethod
    def _save_status_report(status_report: Dict[str, Any]) -> None:
        """Save status report to JSON file."""
        goals_dir = PerformancePaths.get_goals_dir()
        goals_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(PerformancePaths.get_status_file(), "w") as f:
                json.dump(status_report, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving status report: {str(e)}")

# --- Visualization ---
class PerformanceVisualizer:
    """Handles performance visualization and plotting."""
    
    @staticmethod
    def plot_performance_trends(log_path: Optional[str] = None) -> bool:
        """Plot Sharpe, Drawdown, and MSE over time and save as PNG.
        
        Args:
            log_path: Path to the performance log CSV file. If None, uses default.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if log_path is None:
                log_path = str(PerformancePaths.get_performance_log())
            
            if not os.path.exists(log_path):
                logger.warning(f"Performance log not found at {log_path}")
                return False
                
            df = pd.read_csv(log_path)
            if df.empty:
                logger.warning("Performance log is empty")
                return False

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            fig = go.Figure()
            
            for metric in ["sharpe", "drawdown", "mse"]:
                if metric in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'], 
                        y=df[metric], 
                        mode='lines+markers', 
                        name=metric.capitalize()
                    ))
            
            fig.update_layout(
                title="Performance Trends", 
                xaxis_title="Time", 
                yaxis_title="Metric Value"
            )
            
            out_path = PerformancePaths.get_performance_plot()
            pio.write_image(fig, out_path)
            logger.info(f"Saved performance plot to {out_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error plotting performance trends: {e}")
            return False

# --- Main Performance Tracker Class ---
class PerformanceTracker:
    """Main class for performance tracking operations."""
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.target_manager = TargetManager()
        self.logger = PerformanceLogger()
        self.calculator = MetricsCalculator()
        self.evaluator = PerformanceEvaluator()
        self.visualizer = PerformanceVisualizer()
    
    def log_metrics(self, **kwargs) -> bool:
        """Log performance metrics with simplified interface.
        
        Args:
            **kwargs: Performance metrics to log
            
        Returns:
            True if successful, False otherwise.
        """
        return self.logger.log_performance(**kwargs)
    
    def evaluate(self, classification: Optional[bool] = None) -> Dict[str, Any]:
        """Evaluate current performance.
        
        Args:
            classification: Whether to include classification metrics
            
        Returns:
            Performance evaluation results
        """
        return self.evaluator.evaluate_performance(classification)
    
    def plot_trends(self) -> bool:
        """Plot performance trends.
        
        Returns:
            True if successful, False otherwise.
        """
        return self.visualizer.plot_performance_trends()
    
    def update_targets(self, new_targets: Dict[str, float]) -> bool:
        """Update performance targets.
        
        Args:
            new_targets: New target values
            
        Returns:
            True if successful, False otherwise.
        """
        return self.target_manager.update_targets(new_targets)
    
    def get_targets(self) -> Dict[str, float]:
        """Get current performance targets.
        
        Returns:
            Current target values
        """
        return self.target_manager.load_targets()

# --- Legacy Function Compatibility ---
def load_targets() -> Dict[str, float]:
    """Legacy function for loading targets."""
    return TargetManager.load_targets()

def update_targets(new_targets: Dict[str, float]) -> None:
    """Legacy function for updating targets."""
    TargetManager.update_targets(new_targets)

def log_performance(**kwargs) -> None:
    """Legacy function for logging performance."""
    PerformanceLogger.log_performance(**kwargs)

def calculate_rolling_metrics(df: pd.DataFrame, window: int = 7) -> Dict[str, float]:
    """Legacy function for calculating rolling metrics."""
    return MetricsCalculator.calculate_rolling_metrics(df, window)

def evaluate_performance(classification: Optional[bool] = None) -> Dict[str, Any]:
    """Legacy function for evaluating performance."""
    return PerformanceEvaluator.evaluate_performance(classification)

def plot_performance_trends(log_path: str = "memory/logs/performance_log.csv") -> None:
    """Legacy function for plotting performance trends."""
    PerformanceVisualizer.plot_performance_trends(log_path) 