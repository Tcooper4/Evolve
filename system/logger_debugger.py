"""
Logger Debugger Module

This module provides logging utilities and debugging tools for the trading system.
Separated from environment test code to provide clean, focused logging functionality.
"""

import json
import logging
import logging.config
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import yaml


class LoggerDebugger:
    """Logger debugger for the trading platform."""

    def __init__(self, config_path: str = "config/app_config.yaml") -> None:
        """Initialize the logger debugger.

        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.debug_dir = Path("debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("reports/debug")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load application configuration.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Configuration dictionary.

        Raises:
            SystemExit: If configuration file is not found.
        """
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self) -> None:
        """Initialize logging configuration.

        Raises:
            SystemExit: If logging configuration file is not found.
        """
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)

        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)

        logging.config.dictConfig(log_config)

    def analyze_errors(self, log_files: List[str]) -> Dict[str, Any]:
        """Analyze error logs.

        Args:
            log_files: List of log file paths to analyze.

        Returns:
            Dictionary containing error analysis results.

        Raises:
            Exception: If error analysis fails.
        """
        self.logger.info("Analyzing error logs")

        try:
            # Load error logs
            errors = []
            for file in log_files:
                with open(file) as f:
                    for line in f:
                        if "ERROR" in line:
                            errors.append(
                                {
                                    "timestamp": line.split()[0],
                                    "message": line.split("ERROR")[-1].strip(),
                                    "file": file,
                                }
                            )

            # Analyze errors
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "total_errors": len(errors),
                "error_types": {},
                "error_frequency": {},
                "error_timeline": [],
            }

            # Count error types
            for error in errors:
                error_type = error["message"].split(":")[0]
                if error_type not in analysis["error_types"]:
                    analysis["error_types"][error_type] = 0
                analysis["error_types"][error_type] += 1

            # Calculate error frequency
            for error in errors:
                hour = error["timestamp"].split(":")[0]
                if hour not in analysis["error_frequency"]:
                    analysis["error_frequency"][hour] = 0
                analysis["error_frequency"][hour] += 1

            # Create error timeline
            analysis["error_timeline"] = sorted(errors, key=lambda x: x["timestamp"])

            # Save analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = self.reports_dir / f"error_analysis_{timestamp}.json"

            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)

            # Generate visualizations
            self._generate_error_plots(analysis)

            self.logger.info(f"Error analysis saved to {analysis_file}")

            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze errors: {e}")
            raise

    def monitor_errors(self, duration: int = 300) -> List[Dict[str, Any]]:
        """Monitor errors in real-time.

        Args:
            duration: Duration to monitor in seconds.

        Returns:
            List of errors detected during monitoring.

        Raises:
            Exception: If error monitoring fails.
        """
        self.logger.info(f"Monitoring errors for {duration} seconds")

        try:
            # Set up error monitoring
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=duration)

            errors = []
            while datetime.now() < end_time:
                # Check for new errors in log files
                for log_file in Path("logs").glob("*.log"):
                    with open(log_file) as f:
                        for line in f:
                            if "ERROR" in line:
                                errors.append(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "message": line.split("ERROR")[-1].strip(),
                                        "file": str(log_file),
                                    }
                                )

                import time
                time.sleep(1)  # Wait 1 second between checks

            # Save monitoring results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            monitoring_file = self.debug_dir / f"error_monitoring_{timestamp}.json"

            with open(monitoring_file, "w") as f:
                json.dump(
                    {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "errors": errors,
                    },
                    f,
                    indent=2,
                )

            # Print monitoring results
            self._print_monitoring_results(errors)

            return errors
        except Exception as e:
            self.logger.error(f"Failed to monitor errors: {e}")
            raise

    def fix_errors(self, error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fix suggestions for errors.

        Args:
            error_analysis: Error analysis results.

        Returns:
            List of fix suggestions.

        Raises:
            Exception: If fix generation fails.
        """
        self.logger.info("Generating fix suggestions")

        try:
            suggestions = []

            # Generate suggestions based on error types
            for error_type, count in error_analysis["error_types"].items():
                if "ImportError" in error_type:
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "count": count,
                            "suggestion": "Check if all required packages are installed",
                            "action": "pip install -r requirements.txt",
                            "priority": "high" if count > 10 else "medium",
                        }
                    )
                elif "ValidationError" in error_type:
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "count": count,
                            "suggestion": "Validate input data before processing",
                            "action": "Add data validation checks",
                            "priority": "high" if count > 5 else "medium",
                        }
                    )
                elif "ConnectionError" in error_type:
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "count": count,
                            "suggestion": "Check network connectivity and API endpoints",
                            "action": "Verify API keys and network settings",
                            "priority": "high" if count > 3 else "medium",
                        }
                    )
                else:
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "count": count,
                            "suggestion": "Review error logs for specific patterns",
                            "action": "Add error handling and logging",
                            "priority": "medium",
                        }
                    )

            # Save suggestions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suggestions_file = self.reports_dir / f"fix_suggestions_{timestamp}.json"

            with open(suggestions_file, "w") as f:
                json.dump(suggestions, f, indent=2)

            # Print suggestions
            self._print_fix_suggestions(suggestions)

            self.logger.info(f"Fix suggestions saved to {suggestions_file}")

            return suggestions
        except Exception as e:
            self.logger.error(f"Failed to generate fix suggestions: {e}")
            raise

    def _print_monitoring_results(self, errors: List[Dict[str, Any]]) -> None:
        """Print monitoring results.

        Args:
            errors: List of errors detected during monitoring.
        """
        print(f"\nError Monitoring Results:")
        print(f"Total errors detected: {len(errors)}")
        print(f"Monitoring period: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if errors:
            print("\nRecent errors:")
            for error in errors[-5:]:  # Show last 5 errors
                print(f"  - {error['timestamp']}: {error['message']}")
        else:
            print("No errors detected during monitoring period.")

    def _print_fix_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """Print fix suggestions.

        Args:
            suggestions: List of fix suggestions.
        """
        print(f"\nFix Suggestions:")
        print(f"Total suggestions: {len(suggestions)}")

        for suggestion in suggestions:
            priority_color = {
                "high": "\033[91m",  # Red
                "medium": "\033[93m",  # Yellow
                "low": "\033[92m",  # Green
            }.get(suggestion["priority"], "\033[0m")

            print(f"\n{priority_color}Priority: {suggestion['priority'].upper()}\033[0m")
            print(f"Error Type: {suggestion['error_type']}")
            print(f"Count: {suggestion['count']}")
            print(f"Suggestion: {suggestion['suggestion']}")
            print(f"Action: {suggestion['action']}")

    def _generate_error_plots(self, analysis: Dict[str, Any]) -> None:
        """Generate error visualization plots.

        Args:
            analysis: Error analysis results.
        """
        try:
            # Create error type distribution plot
            plt.figure(figsize=(12, 8))

            # Error type distribution
            plt.subplot(2, 2, 1)
            error_types = list(analysis["error_types"].keys())
            error_counts = list(analysis["error_types"].values())
            plt.bar(error_types, error_counts)
            plt.title("Error Type Distribution")
            plt.xticks(rotation=45)
            plt.ylabel("Count")

            # Error frequency over time
            plt.subplot(2, 2, 2)
            hours = list(analysis["error_frequency"].keys())
            frequencies = list(analysis["error_frequency"].values())
            plt.plot(hours, frequencies, marker='o')
            plt.title("Error Frequency Over Time")
            plt.xlabel("Hour")
            plt.ylabel("Error Count")

            # Error timeline
            plt.subplot(2, 2, 3)
            timeline = analysis["error_timeline"]
            if timeline:
                timestamps = [error["timestamp"] for error in timeline]
                plt.plot(range(len(timestamps)), timestamps, marker='o')
                plt.title("Error Timeline")
                plt.xlabel("Error Index")
                plt.ylabel("Timestamp")

            # Summary statistics
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, f"Total Errors: {analysis['total_errors']}", fontsize=12)
            plt.text(0.1, 0.6, f"Error Types: {len(analysis['error_types'])}", fontsize=12)
            plt.text(0.1, 0.4, f"Analysis Time: {analysis['timestamp']}", fontsize=10)
            plt.axis('off')
            plt.title("Summary Statistics")

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.reports_dir / f"error_analysis_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Error plots saved to {plot_file}")

        except Exception as e:
            self.logger.error(f"Failed to generate error plots: {e}")

    def get_logger_status(self) -> Dict[str, Any]:
        """Get current logger status.

        Returns:
            Dictionary containing logger status information.
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "loggers": {},
            "handlers": {},
            "log_files": [],
        }

        # Get logger information
        for name in logging.root.manager.loggerDict:
            logger_obj = logging.getLogger(name)
            status["loggers"][name] = {
                "level": logging.getLevelName(logger_obj.level),
                "handlers": len(logger_obj.handlers),
                "propagate": logger_obj.propagate,
            }

        # Get handler information
        for handler in logging.root.handlers:
            handler_name = handler.__class__.__name__
            if handler_name not in status["handlers"]:
                status["handlers"][handler_name] = 0
            status["handlers"][handler_name] += 1

        # Get log file information
        log_dir = Path("logs")
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                status["log_files"].append({
                    "name": log_file.name,
                    "size": log_file.stat().st_size,
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                })

        return status

    def clear_logs(self, days_to_keep: int = 7) -> int:
        """Clear old log files.

        Args:
            days_to_keep: Number of days of logs to keep.

        Returns:
            Number of files deleted.
        """
        self.logger.info(f"Clearing logs older than {days_to_keep} days")

        try:
            log_dir = Path("logs")
            if not log_dir.exists():
                return 0

            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0

            for log_file in log_dir.glob("*.log"):
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_date < cutoff_date:
                    log_file.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted old log file: {log_file.name}")

            self.logger.info(f"Deleted {deleted_count} old log files")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to clear logs: {e}")
            return 0


# Convenience functions for backward compatibility
def analyze_errors(log_files: List[str]) -> Dict[str, Any]:
    """Analyze error logs using LoggerDebugger."""
    debugger = LoggerDebugger()
    return debugger.analyze_errors(log_files)


def monitor_errors(duration: int = 300) -> List[Dict[str, Any]]:
    """Monitor errors using LoggerDebugger."""
    debugger = LoggerDebugger()
    return debugger.monitor_errors(duration)


def get_logger_status() -> Dict[str, Any]:
    """Get logger status using LoggerDebugger."""
    debugger = LoggerDebugger()
    return debugger.get_logger_status()
