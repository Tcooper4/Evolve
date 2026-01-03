"""
Core Performance Tracking and Evaluation Module

This module provides comprehensive performance tracking, evaluation,
and visualization capabilities for the trading system. It includes metrics calculation,
goal evaluation, and performance trend analysis.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from trading.utils.safe_math import safe_drawdown

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
    drawdown_duration: int = 0  # Duration in days
    max_time_underwater: int = 0  # Maximum consecutive days underwater
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


@dataclass
class ModelLeaderboardEntry:
    """Data class for model leaderboard entries."""

    model_name: str
    ticker: str
    strategy: str
    metric_value: float
    metric_name: str
    timestamp: str
    rank: int
    percentile: float
    trend: str  # 'improving', 'declining', 'stable'
    confidence: float  # Confidence in the metric (0-1)


@dataclass
class LeaderboardConfig:
    """Configuration for leaderboard queries."""

    metric_name: str
    top_n: int = 10
    min_confidence: float = 0.5
    time_window_days: Optional[int] = None
    include_trends: bool = True
    group_by: Optional[str] = None  # 'model', 'ticker', 'strategy'
    sort_ascending: bool = (
        False  # True for metrics where lower is better (e.g., drawdown)
    )


# --- Default Values ---
DEFAULT_METRICS = PerformanceMetrics()
DEFAULT_TARGETS = {
    "sharpe": 1.3,
    "drawdown": 0.25,
    "mse": 0.05,
    "r2": 0.5,
    "precision": 0.7,
    "recall": 0.7,
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

    @staticmethod
    def get_leaderboard_cache() -> Path:
        """Get the leaderboard cache file path."""
        return PerformancePaths.get_goals_dir() / "leaderboard_cache.json"


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
        notes: Optional[str] = None,
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
                notes=notes,
            )

            # Convert to DataFrame and append to CSV
            df_entry = pd.DataFrame([asdict(entry)])
            log_path = PerformancePaths.get_performance_log()

            if log_path.exists():
                df_entry.to_csv(log_path, mode="a", header=False, index=False)
            else:
                df_entry.to_csv(log_path, index=False)

            logger.info(f"Logged performance for {ticker} - {model} - {strategy}")
            return True

        except Exception as e:
            logger.error(f"Error logging performance: {e}")
            return False

    @staticmethod
    def load_performance_data() -> Optional[pd.DataFrame]:
        """Load performance data from CSV file.

        Returns:
            DataFrame containing performance data or None if error.
        """
        try:
            log_path = PerformancePaths.get_performance_log()
            if log_path.exists():
                df = pd.read_csv(log_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            else:
                logger.warning("Performance log file not found")
                return None

        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return None


# --- Metrics Calculator ---


class MetricsCalculator:
    """Calculates various performance metrics."""

    @staticmethod
    def calculate_rolling_metrics(
        df: pd.DataFrame, window: int = 7
    ) -> Dict[str, float]:
        """Calculate rolling performance metrics.

        Args:
            df: DataFrame with performance data
            window: Rolling window size in days

        Returns:
            Dictionary containing rolling metrics.
        """
        try:
            if df.empty:
                return {}

            # Sort by timestamp
            df.sort_values("timestamp")

            # Calculate rolling metrics
            rolling_metrics = {}

            # Rolling Sharpe (assuming we have returns data)
            if "sharpe" in df.columns and not df["sharpe"].isna().all():
                rolling_metrics["rolling_sharpe"] = (
                    df["sharpe"].rolling(window=window).mean().iloc[-1]
                )

            # Rolling drawdown
            if "drawdown" in df.columns and not df["drawdown"].isna().all():
                rolling_metrics["rolling_drawdown"] = (
                    df["drawdown"].rolling(window=window).max().iloc[-1]
                )

            # Rolling accuracy
            if "accuracy" in df.columns and not df["accuracy"].isna().all():
                rolling_metrics["rolling_accuracy"] = (
                    df["accuracy"].rolling(window=window).mean().iloc[-1]
                )

            # Rolling MSE
            if "mse" in df.columns and not df["mse"].isna().all():
                rolling_metrics["rolling_mse"] = (
                    df["mse"].rolling(window=window).mean().iloc[-1]
                )

            # Rolling R2
            if "r2" in df.columns and not df["r2"].isna().all():
                rolling_metrics["rolling_r2"] = (
                    df["r2"].rolling(window=window).mean().iloc[-1]
                )

            return rolling_metrics

        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return {}

    @staticmethod
    def _calculate_drawdown_duration(returns: pd.Series) -> int:
        """Calculate the duration of the maximum drawdown.

        Args:
            returns: Series of returns

        Returns:
            Duration in days
        """
        try:
            cumulative = (1 + returns).cumprod()
            drawdown = safe_drawdown(cumulative)

            # Find the maximum drawdown period
            max_dd_idx = drawdown.idxmin()
            recovery_idx = drawdown[max_dd_idx:].idxmax()

            if pd.isna(recovery_idx):
                return len(returns)

            return (recovery_idx - max_dd_idx).days

        except Exception as e:
            logger.error(f"Error calculating drawdown duration: {e}")
            return 0

    @staticmethod
    def _calculate_max_time_underwater(returns: pd.Series) -> int:
        """Calculate the maximum consecutive days underwater.

        Args:
            returns: Series of returns

        Returns:
            Maximum consecutive days underwater
        """
        try:
            cumulative = (1 + returns).cumprod()
            underwater = cumulative < 1

            max_consecutive = 0
            current_consecutive = 0

            for is_underwater in underwater:
                if is_underwater:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            return max_consecutive

        except Exception as e:
            logger.error(f"Error calculating max time underwater: {e}")
            return 0

    @staticmethod
    def evaluate_against_targets(
        metrics: Dict[str, float],
        targets: Dict[str, float],
        use_classification: bool = False,
    ) -> tuple[str, List[str]]:
        """Evaluate performance metrics against targets.

        Args:
            metrics: Current performance metrics
            targets: Target performance metrics
            use_classification: Whether to use classification metrics

        Returns:
            Tuple of (status, list of issues)
        """
        issues = []

        # Check Sharpe ratio
        if "sharpe" in metrics and "sharpe" in targets:
            if metrics["sharpe"] < targets["sharpe"]:
                issues.append(
                    f"Sharpe ratio {metrics['sharpe']:.3f} below target {targets['sharpe']:.3f}"
                )

        # Check drawdown
        if "drawdown" in metrics and "drawdown" in targets:
            if abs(metrics["drawdown"]) > targets["drawdown"]:
                issues.append(
                    f"Drawdown {abs(metrics['drawdown']):.3f} above target {targets['drawdown']:.3f}"
                )

        # Check MSE
        if "mse" in metrics and "mse" in targets:
            if metrics["mse"] > targets["mse"]:
                issues.append(
                    f"MSE {metrics['mse']:.3f} above target {targets['mse']:.3f}"
                )

        # Check R2
        if "r2" in metrics and "r2" in targets:
            if metrics["r2"] < targets["r2"]:
                issues.append(
                    f"R² {metrics['r2']:.3f} below target {targets['r2']:.3f}"
                )

        # Check classification metrics if enabled
        if use_classification:
            if "precision" in metrics and "precision" in targets:
                if metrics["precision"] < targets["precision"]:
                    issues.append(
                        f"Precision {metrics['precision']:.3f} below target {targets['precision']:.3f}"
                    )

            if "recall" in metrics and "recall" in targets:
                if metrics["recall"] < targets["recall"]:
                    issues.append(
                        f"Recall {metrics['recall']:.3f} below target {targets['recall']:.3f}"
                    )

        # Determine overall status
        if len(issues) == 0:
            status = "excellent"
        elif len(issues) <= 2:
            status = "good"
        elif len(issues) <= 4:
            status = "fair"
        else:
            status = "poor"

        return status, issues


# --- Performance Evaluator ---


class PerformanceEvaluator:
    """Evaluates overall performance and generates status reports."""

    @staticmethod
    def evaluate_performance(classification: Optional[bool] = None) -> Dict[str, Any]:
        """Evaluate overall performance and generate status report.

        Args:
            classification: Whether to use classification metrics (defaults to global setting)

        Returns:
            Dictionary containing performance evaluation results.
        """
        try:
            # Load data and targets
            df = PerformanceLogger.load_performance_data()
            targets = TargetManager.load_targets()

            if df is None or df.empty:
                return PerformanceEvaluator._create_no_data_response(targets)

            # Use global classification setting if not specified
            if classification is None:
                classification = CLASSIFICATION

            # Calculate current metrics
            current_metrics = {}

            # Get latest metrics for each model/ticker combination
            latest_data = df.groupby(["model", "ticker"]).last()

            # Calculate aggregate metrics
            for metric in ["sharpe", "drawdown", "mse", "accuracy", "r2"]:
                if metric in latest_data.columns:
                    values = latest_data[metric].dropna()
                    if not values.empty:
                        current_metrics[f"avg_{metric}"] = values.mean()
                        current_metrics[f"best_{metric}"] = (
                            values.max() if metric != "drawdown" else values.min()
                        )
                        current_metrics[f"worst_{metric}"] = (
                            values.min() if metric != "drawdown" else values.max()
                        )

            # Add classification metrics if enabled
            if classification:
                for metric in ["precision", "recall"]:
                    if metric in latest_data.columns:
                        values = latest_data[metric].dropna()
                        if not values.empty:
                            current_metrics[f"avg_{metric}"] = values.mean()
                            current_metrics[f"best_{metric}"] = values.max()
                            current_metrics[f"worst_{metric}"] = values.min()

            # Evaluate against targets
            status, issues = MetricsCalculator.evaluate_against_targets(
                current_metrics, targets, classification
            )

            # Create status report
            status_report = {
                "status": status,
                "message": f"Performance status: {status}",
                "timestamp": datetime.now().isoformat(),
                "targets": targets,
                "current_metrics": current_metrics,
                "issues": issues,
                "last_evaluated": datetime.now().isoformat(),
                "data_points": len(df),
                "models_evaluated": df["model"].nunique(),
                "tickers_evaluated": df["ticker"].nunique(),
            }

            # Save status report
            PerformanceEvaluator._save_status_report(status_report)

            # Handle underperformance
            if status in ["fair", "poor"]:
                PerformanceEvaluator._handle_underperformance(status_report)

            return status_report

        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return PerformanceEvaluator._create_error_response(str(e))

    @staticmethod
    def _create_no_data_response(targets: Dict[str, float]) -> Dict[str, Any]:
        """Create response when no performance data is available."""
        return {
            "status": "no_data",
            "message": "No performance data available for evaluation",
            "timestamp": datetime.now().isoformat(),
            "targets": targets,
            "current_metrics": {},
            "issues": ["No performance data available"],
            "last_evaluated": datetime.now().isoformat(),
        }

    @staticmethod
    def _create_error_response(error_msg: str) -> Dict[str, Any]:
        """Create response when an error occurs during evaluation."""
        return {
            "status": "error",
            "message": f"Error during performance evaluation: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "targets": {},
            "current_metrics": {},
            "issues": [error_msg],
            "last_evaluated": datetime.now().isoformat(),
        }

    @staticmethod
    def _handle_underperformance(status_report: Dict[str, Any]) -> None:
        """Handle underperformance by logging and potentially triggering alerts."""
        logger.warning(
            f"Performance underperformance detected: {status_report['issues']}"
        )

        # Alerting mechanism implementation
        try:
            # Send email alert if configured
            if os.environ.get("ALERT_EMAIL_ENABLED", "false").lower() == "true":
                alert_email = os.environ.get("ALERT_EMAIL")
                if alert_email:
                    PerformanceEvaluator._send_email_alert(alert_email, status_report)

            # Send Slack notification if configured
            if os.environ.get("SLACK_WEBHOOK_URL"):
                PerformanceEvaluator._send_slack_alert(status_report)

            # Log to external monitoring service if configured
            if os.environ.get("MONITORING_ENDPOINT"):
                PerformanceEvaluator._send_monitoring_alert(status_report)

        except Exception as e:
            logger.error(f"Error sending performance alerts: {e}")

    @staticmethod
    def _send_email_alert(email: str, status_report: Dict[str, Any]) -> None:
        """Send email alert for performance issues."""
        try:
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart()
            msg["From"] = os.environ.get("SMTP_FROM_EMAIL", "alerts@trading-system.com")
            msg["To"] = email
            msg["Subject"] = f"Performance Alert: {status_report['status']}"

            body = f"""
            Performance Alert Detected

            Status: {status_report['status']}
            Message: {status_report['message']}
            Issues: {', '.join(status_report['issues'])}
            Timestamp: {status_report['timestamp']}

            Current Metrics:
            {json.dumps(status_report['current_metrics'], indent=2)}
            """

            msg.attach(MIMEText(body, "plain"))

            # Send email (configure SMTP settings in environment)
            smtp_server = os.environ.get("SMTP_SERVER", "localhost")
            smtp_port = int(os.environ.get("SMTP_PORT", "587"))
            smtp_user = os.environ.get("SMTP_USER")
            smtp_password = os.environ.get("SMTP_PASSWORD")

            if smtp_user and smtp_password:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
                server.quit()
                logger.info(f"Performance alert email sent to {email}")

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")

    @staticmethod
    def _send_slack_alert(status_report: Dict[str, Any]) -> None:
        """Send Slack alert for performance issues."""
        try:
            import requests

            webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
            if not webhook_url:
                return

            message = {
                "text": f"🚨 Performance Alert: {status_report['status']}",
                "attachments": [
                    {
                        "color": (
                            "danger"
                            if status_report["status"] == "error"
                            else "warning"
                        ),
                        "fields": [
                            {
                                "title": "Message",
                                "value": status_report["message"],
                                "short": False,
                            },
                            {
                                "title": "Issues",
                                "value": ", ".join(status_report["issues"]),
                                "short": False,
                            },
                            {
                                "title": "Timestamp",
                                "value": status_report["timestamp"],
                                "short": True,
                            },
                        ],
                    }
                ],
            }

            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                logger.info("Performance alert sent to Slack")
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    @staticmethod
    def _send_monitoring_alert(status_report: Dict[str, Any]) -> None:
        """Send alert to external monitoring service."""
        try:
            import requests

            endpoint = os.environ.get("MONITORING_ENDPOINT")
            api_key = os.environ.get("MONITORING_API_KEY")

            if not endpoint or not api_key:
                return

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "alert_type": "performance_underperformance",
                "severity": (
                    "warning" if status_report["status"] == "warning" else "critical"
                ),
                "message": status_report["message"],
                "details": status_report,
                "timestamp": datetime.now().isoformat(),
            }

            response = requests.post(endpoint, json=payload, headers=headers)
            if response.status_code in [200, 201]:
                logger.info("Performance alert sent to monitoring service")
            else:
                logger.error(f"Failed to send monitoring alert: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending monitoring alert: {e}")

    @staticmethod
    def _save_status_report(status_report: Dict[str, Any]) -> None:
        """Save status report to file."""
        try:
            status_path = PerformancePaths.get_status_file()
            status_path.parent.mkdir(parents=True, exist_ok=True)

            with open(status_path, "w") as f:
                json.dump(status_report, f, indent=4)

            logger.info("Saved performance status report")

        except Exception as e:
            logger.error(f"Error saving status report: {e}")


# --- Performance Visualizer ---


class PerformanceVisualizer:
    """Handles performance visualization and plotting."""

    @staticmethod
    def plot_performance_trends(log_path: Optional[str] = None) -> bool:
        """Plot performance trends over time.

        Args:
            log_path: Optional path to performance log file

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Load data
            if log_path:
                df = pd.read_csv(log_path)
            else:
                df = PerformanceLogger.load_performance_data()

            if df is None or df.empty:
                logger.warning("No data available for plotting")
                return False

            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Performance Trends Over Time", fontsize=16)

            # Plot 1: Sharpe Ratio
            if "sharpe" in df.columns:
                df.groupby("timestamp")["sharpe"].mean().plot(
                    ax=axes[0, 0], title="Average Sharpe Ratio"
                )
                axes[0, 0].set_ylabel("Sharpe Ratio")

            # Plot 2: Drawdown
            if "drawdown" in df.columns:
                df.groupby("timestamp")["drawdown"].mean().plot(
                    ax=axes[0, 1], title="Average Drawdown"
                )
                axes[0, 1].set_ylabel("Drawdown")

            # Plot 3: Accuracy
            if "accuracy" in df.columns:
                df.groupby("timestamp")["accuracy"].mean().plot(
                    ax=axes[1, 0], title="Average Accuracy"
                )
                axes[1, 0].set_ylabel("Accuracy")

            # Plot 4: MSE
            if "mse" in df.columns:
                df.groupby("timestamp")["mse"].mean().plot(
                    ax=axes[1, 1], title="Average MSE"
                )
                axes[1, 1].set_ylabel("MSE")

            plt.tight_layout()

            # Save plot
            plot_path = PerformancePaths.get_performance_plot()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Performance plot saved to {plot_path}")
            return True

        except Exception as e:
            logger.error(f"Error plotting performance trends: {e}")
            return False


# --- Leaderboard Management ---


class LeaderboardManager:
    """Manages model leaderboards and rankings."""

    @staticmethod
    def get_top_n_models_by_metric(
        metric_name: str,
        top_n: int = 10,
        time_window_days: Optional[int] = None,
        min_confidence: float = 0.5,
        include_trends: bool = True,
        group_by: Optional[str] = None,
        sort_ascending: bool = False,
    ) -> List[ModelLeaderboardEntry]:
        """Get top N models by a specific metric.

        Args:
            metric_name: Name of the metric to rank by
            top_n: Number of top models to return
            time_window_days: Optional time window in days to filter data
            min_confidence: Minimum confidence threshold
            include_trends: Whether to include trend analysis
            group_by: Optional grouping ('model', 'ticker', 'strategy')
            sort_ascending: Whether to sort in ascending order (for metrics where lower is better)

        Returns:
            List of ModelLeaderboardEntry objects
        """
        try:
            # Load performance data
            df = PerformanceLogger.load_performance_data()
            if df is None or df.empty:
                logger.warning("No performance data available for leaderboard")
                return []

            # Filter by time window if specified
            if time_window_days:
                cutoff_date = datetime.now() - timedelta(days=time_window_days)
                df = df[df["timestamp"] >= cutoff_date]

            # Check if metric exists
            if metric_name not in df.columns:
                logger.error(f"Metric '{metric_name}' not found in performance data")
                return []

            # Remove rows with missing metric values
            df_clean = df.dropna(subset=[metric_name])
            if df_clean.empty:
                logger.warning(f"No data available for metric '{metric_name}'")
                return []

            # Group data if specified
            if group_by and group_by in df_clean.columns:
                # Calculate aggregate metrics for each group
                grouped_data = (
                    df_clean.groupby(group_by)[metric_name]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                grouped_data = grouped_data[
                    grouped_data["count"] >= 3
                ]  # Minimum 3 data points for confidence

                # Calculate confidence based on standard deviation and sample size
                grouped_data["confidence"] = 1 / (
                    1 + grouped_data["std"] / grouped_data["mean"].abs()
                )
                grouped_data["confidence"] = grouped_data["confidence"].clip(0, 1)

                # Filter by confidence
                grouped_data = grouped_data[
                    grouped_data["confidence"] >= min_confidence
                ]

                # Sort by metric value
                grouped_data = grouped_data.sort_values(
                    "mean", ascending=sort_ascending
                )

                # Get top N
                top_data = grouped_data.head(top_n)

                # Create leaderboard entries
                entries = []
                for idx, row in top_data.iterrows():
                    # Calculate trend if requested
                    trend = "stable"
                    if include_trends:
                        group_data = df_clean[df_clean[group_by] == row[group_by]]
                        if len(group_data) >= 2:
                            recent_avg = group_data.tail(3)[metric_name].mean()
                            older_avg = group_data.head(3)[metric_name].mean()
                            if recent_avg > older_avg * 1.05:
                                trend = "improving"
                            elif recent_avg < older_avg * 0.95:
                                trend = "declining"

                    entry = ModelLeaderboardEntry(
                        model_name=str(row[group_by]),
                        ticker="N/A",  # Not applicable for grouped data
                        strategy="N/A",
                        metric_value=float(row["mean"]),
                        metric_name=metric_name,
                        timestamp=datetime.now().isoformat(),
                        rank=idx + 1,
                        percentile=float(
                            (len(grouped_data) - idx) / len(grouped_data) * 100
                        ),
                        trend=trend,
                        confidence=float(row["confidence"]),
                    )
                    entries.append(entry)

                return entries

            else:
                # Individual entries ranking
                # Get latest entry for each model/ticker/strategy combination
                latest_data = (
                    df_clean.groupby(["model", "ticker", "strategy"])
                    .last()
                    .reset_index()
                )

                # Calculate confidence based on data quality
                latest_data[
                    "confidence"
                ] = 0.8  # Default confidence for individual entries

                # Filter by confidence
                latest_data = latest_data[latest_data["confidence"] >= min_confidence]

                # Sort by metric value
                latest_data = latest_data.sort_values(
                    metric_name, ascending=sort_ascending
                )

                # Get top N
                top_data = latest_data.head(top_n)

                # Create leaderboard entries
                entries = []
                for idx, row in top_data.iterrows():
                    # Calculate trend if requested
                    trend = "stable"
                    if include_trends:
                        model_data = df_clean[
                            (df_clean["model"] == row["model"])
                            & (df_clean["ticker"] == row["ticker"])
                            & (df_clean["strategy"] == row["strategy"])
                        ]
                        if len(model_data) >= 2:
                            recent_avg = model_data.tail(3)[metric_name].mean()
                            older_avg = model_data.head(3)[metric_name].mean()
                            if recent_avg > older_avg * 1.05:
                                trend = "improving"
                            elif recent_avg < older_avg * 0.95:
                                trend = "declining"

                    entry = ModelLeaderboardEntry(
                        model_name=row["model"],
                        ticker=row["ticker"],
                        strategy=row["strategy"],
                        metric_value=float(row[metric_name]),
                        metric_name=metric_name,
                        timestamp=(
                            row["timestamp"].isoformat()
                            if hasattr(row["timestamp"], "isoformat")
                            else str(row["timestamp"])
                        ),
                        rank=idx + 1,
                        percentile=float(
                            (len(latest_data) - idx) / len(latest_data) * 100
                        ),
                        trend=trend,
                        confidence=float(row["confidence"]),
                    )
                    entries.append(entry)

                return entries

        except Exception as e:
            logger.error(f"Error getting top N models by metric: {e}")
            return []

    @staticmethod
    def get_leaderboard_summary() -> Dict[str, Any]:
        """Get a summary of all leaderboards.

        Returns:
            Dictionary containing leaderboard summaries
        """
        try:
            df = PerformanceLogger.load_performance_data()
            if df is None or df.empty:
                return {}

            summary = {
                "total_models": df["model"].nunique(),
                "total_tickers": df["ticker"].nunique(),
                "total_strategies": df["strategy"].nunique(),
                "total_entries": len(df),
                "date_range": {
                    "start": (
                        df["timestamp"].min().isoformat()
                        if hasattr(df["timestamp"].min(), "isoformat")
                        else str(df["timestamp"].min())
                    ),
                    "end": (
                        df["timestamp"].max().isoformat()
                        if hasattr(df["timestamp"].max(), "isoformat")
                        else str(df["timestamp"].max())
                    ),
                },
                "available_metrics": [
                    col
                    for col in df.columns
                    if col not in ["timestamp", "ticker", "model", "strategy", "notes"]
                ],
                "top_models_by_metric": {},
            }

            # Get top models for each metric
            for metric in summary["available_metrics"]:
                try:
                    top_models = LeaderboardManager.get_top_n_models_by_metric(
                        metric, top_n=5
                    )
                    summary["top_models_by_metric"][metric] = [
                        {
                            "model": entry.model_name,
                            "value": entry.metric_value,
                            "rank": entry.rank,
                        }
                        for entry in top_models
                    ]
                except Exception as e:
                    logger.warning(f"Error getting top models for metric {metric}: {e}")

            return summary

        except Exception as e:
            logger.error(f"Error getting leaderboard summary: {e}")
            return {}

    @staticmethod
    def cache_leaderboard_data() -> bool:
        """Cache leaderboard data for faster access.

        Returns:
            True if successful, False otherwise
        """
        try:
            summary = LeaderboardManager.get_leaderboard_summary()

            cache_path = PerformancePaths.get_leaderboard_cache()
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            cache_data = {
                "summary": summary,
                "cached_at": datetime.now().isoformat(),
                "cache_version": "1.0",
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=4)

            logger.info("Cached leaderboard data")
            return True

        except Exception as e:
            logger.error(f"Error caching leaderboard data: {e}")
            return False


# --- Performance Tracker ---


class PerformanceTracker:
    """Main performance tracking class that coordinates all functionality."""

    def __init__(self):
        """Initialize the performance tracker."""
        self.logger = PerformanceLogger()
        self.evaluator = PerformanceEvaluator()
        self.visualizer = PerformanceVisualizer()
        self.leaderboard_manager = LeaderboardManager()

    def log_metrics(self, **kwargs) -> bool:
        """Log performance metrics.

        Args:
            **kwargs: Performance metrics to log

        Returns:
            True if successful, False otherwise
        """
        return self.logger.log_performance(**kwargs)

    def evaluate(self, classification: Optional[bool] = None) -> Dict[str, Any]:
        """Evaluate performance.

        Args:
            classification: Whether to use classification metrics

        Returns:
            Performance evaluation results
        """
        return self.evaluator.evaluate_performance(classification)

    def plot_trends(self) -> bool:
        """Plot performance trends.

        Returns:
            True if successful, False otherwise
        """
        return self.visualizer.plot_performance_trends()

    def update_targets(self, new_targets: Dict[str, float]) -> bool:
        """Update performance targets.

        Args:
            new_targets: New target values

        Returns:
            True if successful, False otherwise
        """
        return TargetManager.update_targets(new_targets)

    def get_targets(self) -> Dict[str, float]:
        """Get current performance targets.

        Returns:
            Dictionary of current targets
        """
        return TargetManager.load_targets()

    def get_top_n_models_by_metric(
        self,
        metric_name: str,
        top_n: int = 10,
        time_window_days: Optional[int] = None,
        min_confidence: float = 0.5,
        include_trends: bool = True,
        group_by: Optional[str] = None,
        sort_ascending: bool = False,
    ) -> List[ModelLeaderboardEntry]:
        """Get top N models by metric.

        Args:
            metric_name: Name of the metric to rank by
            top_n: Number of top models to return
            time_window_days: Optional time window in days
            min_confidence: Minimum confidence threshold
            include_trends: Whether to include trend analysis
            group_by: Optional grouping
            sort_ascending: Whether to sort in ascending order

        Returns:
            List of ModelLeaderboardEntry objects
        """
        return self.leaderboard_manager.get_top_n_models_by_metric(
            metric_name,
            top_n,
            time_window_days,
            min_confidence,
            include_trends,
            group_by,
            sort_ascending,
        )

    def get_leaderboard_summary(self) -> Dict[str, Any]:
        """Get leaderboard summary.

        Returns:
            Dictionary containing leaderboard summaries
        """
        return self.leaderboard_manager.get_leaderboard_summary()

    def cache_leaderboard_data(self) -> bool:
        """Cache leaderboard data.

        Returns:
            True if successful, False otherwise
        """
        return self.leaderboard_manager.cache_leaderboard_data()


# --- Convenience Functions ---


def load_targets() -> Dict[str, float]:
    """Load performance targets."""
    return TargetManager.load_targets()


def update_targets(new_targets: Dict[str, float]) -> None:
    """Update performance targets."""
    TargetManager.update_targets(new_targets)


def log_performance(**kwargs) -> None:
    """Log performance metrics."""
    PerformanceLogger.log_performance(**kwargs)


def calculate_rolling_metrics(df: pd.DataFrame, window: int = 7) -> Dict[str, float]:
    """Calculate rolling performance metrics."""
    return MetricsCalculator.calculate_rolling_metrics(df, window)


def evaluate_performance(classification: Optional[bool] = None) -> Dict[str, Any]:
    """Evaluate overall performance."""
    return PerformanceEvaluator.evaluate_performance(classification)


def plot_performance_trends(log_path: str = "memory/logs/performance_log.csv") -> None:
    """Plot performance trends."""
    PerformanceVisualizer.plot_performance_trends(log_path)


def get_top_n_models_by_metric(
    metric_name: str,
    top_n: int = 10,
    time_window_days: Optional[int] = None,
    min_confidence: float = 0.5,
    include_trends: bool = True,
    group_by: Optional[str] = None,
    sort_ascending: bool = False,
) -> List[ModelLeaderboardEntry]:
    """Get top N models by metric."""
    return LeaderboardManager.get_top_n_models_by_metric(
        metric_name,
        top_n,
        time_window_days,
        min_confidence,
        include_trends,
        group_by,
        sort_ascending,
    )
