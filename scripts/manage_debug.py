#!/usr/bin/env python3
"""
Debug management script.
Provides commands for debugging the application, including running in debug mode, collecting debug info, and exporting logs.

This script supports:
- Running the application in debug mode
- Collecting debug information
- Exporting debug logs

Usage:
    python manage_debug.py <command> [options]

Commands:
    run         Run application in debug mode
    collect     Collect debug information
    export      Export debug logs

Examples:
    # Run in debug mode
    python manage_debug.py run

    # Collect debug information
    python manage_debug.py collect --output debug_info.json

    # Export debug logs
    python manage_debug.py export --output debug_logs.zip
"""

import argparse
import ast
import json
import logging
import logging.config
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List

import ipdb
import matplotlib.pyplot as plt
import yaml


class DebugManager:
    """Manages debugging operations for the trading platform."""

    def __init__(self, config_path: str = "config/app_config.yaml") -> None:
        """Initialize the debug manager.

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

    def debug_function(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Debug a function with interactive debugging.

        Args:
            func: Function to debug.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Result of the function execution.

        Raises:
            Exception: If debugging fails.
        """
        self.logger.info(f"Debugging function: {func.__name__}")

        try:
            # Set up debugger
            ipdb.set_trace

            # Run function with debugger
            result = func(*args, **kwargs)

            return result
        except Exception as e:
            self.logger.error(f"Failed to debug function: {e}")
            raise

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
        """Suggest fixes for common errors.

        Args:
            error_analysis: Error analysis results.

        Returns:
            List of fix suggestions.

        Raises:
            Exception: If fix suggestions fail.
        """
        self.logger.info("Suggesting error fixes")

        try:
            # Generate fix suggestions
            suggestions = []
            for error_type, count in error_analysis["error_types"].items():
                if error_type == "ValueError":
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "description": "Check input values and type conversions",
                            "example": "Ensure numeric inputs are valid numbers",
                            "severity": "high" if count > 10 else "medium",
                        }
                    )
                elif error_type == "KeyError":
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "description": "Check dictionary key existence",
                            "example": "Use dict.get() with default values",
                            "severity": "high" if count > 10 else "medium",
                        }
                    )
                elif error_type == "TypeError":
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "description": "Check function argument types",
                            "example": "Add type hints and validation",
                            "severity": "high" if count > 10 else "medium",
                        }
                    )
                elif error_type == "AttributeError":
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "description": "Check object attribute existence",
                            "example": "Use hasattr() before accessing attributes",
                            "severity": "high" if count > 10 else "medium",
                        }
                    )
                elif error_type == "ImportError":
                    suggestions.append(
                        {
                            "error_type": error_type,
                            "description": "Check module imports and dependencies",
                            "example": "Verify package installation and paths",
                            "severity": "high" if count > 10 else "medium",
                        }
                    )

            # Save suggestions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suggestions_file = self.reports_dir / f"error_fixes_{timestamp}.json"

            with open(suggestions_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "suggestions": suggestions,
                    },
                    f,
                    indent=2,
                )

            # Print suggestions
            self._print_fix_suggestions(suggestions)

            return suggestions
        except Exception as e:
            self.logger.error(f"Failed to suggest error fixes: {e}")
            raise

    def _print_monitoring_results(self, errors: List[Dict[str, Any]]) -> None:
        """Print error monitoring results.

        Args:
            errors: List of errors to display.
        """
        print("\nError Monitoring Results:")
        print(f"\nTotal Errors: {len(errors)}")

        if errors:
            print("\nRecent Errors:")
            for error in errors[-5:]:  # Show last 5 errors
                print(f"\nTimestamp: {error['timestamp']}")
                print(f"Message: {error['message']}")
                print(f"File: {error['file']}")

    def _print_fix_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """Print error fix suggestions.

        Args:
            suggestions: List of fix suggestions to display.
        """
        print("\nError Fix Suggestions:")

        for suggestion in suggestions:
            print(f"\nError Type: {suggestion['error_type']}")
            print(f"Description: {suggestion['description']}")
            print(f"Example: {suggestion['example']}")
            print(f"Severity: {suggestion['severity']}")

    def _generate_error_plots(self, analysis: Dict[str, Any]) -> None:
        """Generate error visualization plots.

        Args:
            analysis: Error analysis results.

        Raises:
            Exception: If plot generation fails.
        """
        try:
            # Set style
            plt.style.use("seaborn")

            # Create plots directory
            plots_dir = self.reports_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Error types distribution
            plt.figure(figsize=(10, 6))
            plt.pie(
                analysis["error_types"].values(),
                labels=analysis["error_types"].keys(),
                autopct="%1.1f%%",
            )
            plt.title("Error Types Distribution")
            plt.tight_layout()
            plt.savefig(plots_dir / "error_types_distribution.png")
            plt.close()

            # Error frequency over time
            plt.figure(figsize=(12, 6))
            hours = sorted(analysis["error_frequency"].keys())
            frequencies = [analysis["error_frequency"][hour] for hour in hours]
            plt.plot(hours, frequencies, marker="o")
            plt.title("Error Frequency Over Time")
            plt.xlabel("Hour")
            plt.ylabel("Number of Errors")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "error_frequency.png")
            plt.close()

            self.logger.info(f"Error plots saved to {plots_dir}")
        except Exception as e:
            self.logger.error(f"Failed to generate error plots: {e}")
            raise

    def _safe_parse_function(self, function_name: str) -> Callable:
        """Safely parse function name to callable object.

        Args:
            function_name: Name of the function to resolve.

        Returns:
            Callable function object.

        Raises:
            ValueError: If function name is invalid or function not found.
        """
        # Safe function mapping - only allow specific functions
        safe_functions = {
            "test_strategy": self._test_strategy_function,
            "analyze_data": self._analyze_data_function,
            "validate_model": self._validate_model_function,
        }

        if function_name not in safe_functions:
            raise ValueError(
                f"Function '{function_name}' is not in the safe functions list"
            )

        return safe_functions[function_name]

    def _safe_parse_arguments(self, args_str: str) -> List[Any]:
        """Safely parse arguments string to list of values.

        Args:
            args_str: String representation of arguments.

        Returns:
            List of parsed arguments.

        Raises:
            ValueError: If arguments cannot be safely parsed.
        """
        try:
            # Use ast.literal_eval for safe parsing of literals
            return ast.literal_eval(args_str)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Cannot safely parse arguments: {e}")

    def _test_strategy_function(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Test strategy function for debugging.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Test results dictionary.
        """
        return {
            "success": True,
            "result": {
                "status": "success",
                "function": "test_strategy",
                "args": args,
                "kwargs": kwargs,
            },
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _analyze_data_function(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Analyze data function for debugging.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Analysis results dictionary.
        """
        return {
            "success": True,
            "result": {
                "status": "success",
                "function": "analyze_data",
                "args": args,
                "kwargs": kwargs,
            },
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_model_function(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Validate model function for debugging.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Validation results dictionary.
        """
        return {
            "success": True,
            "result": {
                "status": "success",
                "function": "validate_model",
                "args": args,
                "kwargs": kwargs,
            },
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }


def main() -> None:
    """Main function for the debug manager script."""
    parser = argparse.ArgumentParser(description="Debug Manager")
    parser.add_argument(
        "command",
        choices=["debug", "analyze", "monitor", "fix"],
        help="Command to execute",
    )
    parser.add_argument(
        "--function", help="Function to debug (must be in safe functions list)"
    )
    parser.add_argument("--log-files", nargs="+", help="Log files to analyze")
    parser.add_argument(
        "--duration", type=int, default=300, help="Duration for monitoring in seconds"
    )
    parser.add_argument("--args", help="Arguments for the function (as string)")
    parser.add_argument(
        "--kwargs", type=json.loads, help="Keyword arguments for the function"
    )

    args = parser.parse_args()
    manager = DebugManager()

    try:
        if args.command == "debug":
            if not args.function:
                print("Error: --function is required for debug command")
                sys.exit(1)

            # Safely parse function and arguments
            func = manager._safe_parse_function(args.function)
            func_args = manager._safe_parse_arguments(args.args) if args.args else []
            func_kwargs = args.kwargs or {}

            result = manager.debug_function(func, *func_args, **func_kwargs)
            print(f"Debug result: {result}")

        elif args.command == "analyze":
            if not args.log_files:
                print("Error: --log-files is required for analyze command")
                sys.exit(1)

            result = manager.analyze_errors(args.log_files)
            print(f"Analysis complete: {result['total_errors']} errors found")

        elif args.command == "monitor":
            result = manager.monitor_errors(args.duration)
            print(f"Monitoring complete: {len(result)} errors detected")

        elif args.command == "fix":
            if not args.log_files:
                print("Error: --log-files is required for fix command")
                sys.exit(1)

            analysis = manager.analyze_errors(args.log_files)
            result = manager.fix_errors(analysis)
            print(f"Fix suggestions generated: {len(result)} suggestions")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
