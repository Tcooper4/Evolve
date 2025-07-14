#!/usr/bin/env python3
"""
Performance management script.
Provides commands for profiling and optimizing application performance.

This script supports:
- Function profiling (CPU, memory, line-by-line)
- Performance analysis and reporting
- Automatic optimization suggestions
- Performance visualization
- System resource monitoring

Usage:
    python manage_performance.py <command> [options]

Commands:
    profile     Profile a function's performance
    analyze     Analyze performance profiles
    optimize    Optimize function performance
    monitor     Monitor system resources
    report      Generate performance reports

Examples:
    # Profile a function's CPU usage
    python manage_performance.py profile --function my_function --type cpu --args '{"arg1": "value1"}'

    # Profile memory usage
    python manage_performance.py profile --function my_function --type memory

    # Analyze performance profiles
    python manage_performance.py analyze --profiles "profiles/*.prof" --output reports/analysis.json

    # Optimize function performance
    python manage_performance.py optimize --function my_function --target cpu

    # Monitor system resources
    python manage_performance.py monitor --duration 3600 --interval 60

    # Generate performance report
    python manage_performance.py report --profiles "profiles/*.prof" --format html
"""

import argparse
import cProfile
import glob
import json
import logging
import logging.config
import pstats
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import line_profiler
import matplotlib.pyplot as plt
import memory_profiler
import yaml


class PerformanceManager:
    """Manager for application performance profiling and optimization.

    This class provides methods for profiling function performance (CPU, memory,
    line-by-line), analyzing performance data, generating optimization suggestions,
    and monitoring system resources.

    Attributes:
        config (dict): Application configuration
        logger (logging.Logger): Logger instance
        prof_dir (Path): Directory for storing profiles
        reports_dir (Path): Directory for storing reports

    Example:
        manager = PerformanceManager()
        result = manager.profile_function(my_function, arg1, arg2)
        analysis = manager.analyze_performance(["profiles/profile.prof"])
    """

    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the performance manager.

        Args:
            config_path: Path to the application configuration file

        Raises:
            SystemExit: If the configuration file is not found
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.prof_dir = Path("profiles")
        self.prof_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("reports/performance")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary

        Raises:
            SystemExit: If the configuration file is not found
        """
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration.

        Raises:
            SystemExit: If the logging configuration file is not found
        """
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)

        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)

        logging.config.dictConfig(log_config)

    def profile_function(self, func: callable, *args, **kwargs):
        """Profile a function's performance.

        Args:
            func: Function to profile
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function execution

        Raises:
            Exception: If profiling fails
        """
        self.logger.info(f"Profiling function: {func.__name__}")

        try:
            # Create profiler
            profiler = cProfile.Profile()

            # Run function with profiler
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            # Save profile results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_file = self.prof_dir / f"profile_{func.__name__}_{timestamp}.prof"

            # Save raw profile data
            profiler.dump_stats(str(profile_file))

            # Generate and save statistics
            stats_file = self.prof_dir / f"stats_{func.__name__}_{timestamp}.txt"
            with open(stats_file, "w") as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.sort_stats("cumulative")
                stats.print_stats()

            self.logger.info(f"Profile results saved to {profile_file}")
            self.logger.info(f"Statistics saved to {stats_file}")

            return result
        except Exception as e:
            self.logger.error(f"Failed to profile function: {e}")
            raise

    def profile_memory(self, func: callable, *args, **kwargs):
        """Profile a function's memory usage.

        Args:
            func: Function to profile
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function execution

        Raises:
            Exception: If memory profiling fails
        """
        self.logger.info(f"Profiling memory usage for function: {func.__name__}")

        try:
            # Create memory profiler
            profiled_func = memory_profiler.profile(func)

            # Run function with profiler
            result = profiled_func(*args, **kwargs)

            # Save memory profile
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            memory_file = self.prof_dir / f"memory_{func.__name__}_{timestamp}.txt"

            with open(memory_file, "w") as f:
                memory_profiler.show_results(profiled_func, stream=f)

            self.logger.info(f"Memory profile saved to {memory_file}")

            return result
        except Exception as e:
            self.logger.error(f"Failed to profile memory: {e}")
            raise

    def profile_line(self, func: callable, *args, **kwargs):
        """Profile a function's line-by-line performance.

        Args:
            func: Function to profile
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function execution

        Raises:
            Exception: If line profiling fails
        """
        self.logger.info(
            f"Profiling line-by-line performance for function: {func.__name__}"
        )

        try:
            # Create line profiler
            profiler = line_profiler.LineProfiler(func)

            # Run function with profiler
            result = profiler(func)(*args, **kwargs)

            # Save line profile
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            line_file = self.prof_dir / f"line_{func.__name__}_{timestamp}.txt"

            with open(line_file, "w") as f:
                profiler.print_stats(stream=f)

            self.logger.info(f"Line profile saved to {line_file}")

            return result
        except Exception as e:
            self.logger.error(f"Failed to profile line performance: {e}")
            raise

    def analyze_performance(self, profile_files: List[str]):
        """Analyze performance profiles.

        Args:
            profile_files: List of profile file paths to analyze

        Returns:
            Dictionary containing performance analysis results

        Raises:
            Exception: If analysis fails
        """
        self.logger.info("Analyzing performance profiles")

        try:
            # Load profile data
            profile_data = []
            for file in profile_files:
                stats = pstats.Stats(file)
                profile_data.append({"file": file, "stats": stats})

            # Analyze profiles
            analysis = {"timestamp": datetime.now().isoformat(), "profiles": []}

            for data in profile_data:
                stats = data["stats"]
                profile_analysis = {
                    "file": data["file"],
                    "total_time": stats.total_tt,
                    "function_calls": stats.total_calls,
                    "primitive_calls": stats.prim_calls,
                    "top_functions": [],
                }

                # Get top functions by cumulative time
                for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                    if ct > 0:  # Only include functions that were called
                        profile_analysis["top_functions"].append(
                            {
                                "name": func,
                                "calls": nc,
                                "total_time": tt,
                                "cumulative_time": ct,
                            }
                        )

                # Sort top functions by cumulative time
                profile_analysis["top_functions"].sort(
                    key=lambda x: x["cumulative_time"], reverse=True
                )

                analysis["profiles"].append(profile_analysis)

            # Save analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = self.reports_dir / f"performance_analysis_{timestamp}.json"

            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)

            self.logger.info(f"Analysis saved to {analysis_file}")

            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze performance: {e}")
            raise

    def optimize_performance(self, func: callable, *args, **kwargs):
        """Optimize function performance."""
        self.logger.info(f"Optimizing function: {func.__name__}")

        try:
            # Profile original function
            original_result = self.profile_function(func, *args, **kwargs)

            # Get function source
            import inspect

            source = inspect.getsource(func)

            # Analyze source code
            optimization_suggestions = self._analyze_source_code(source)

            # Apply optimizations
            optimized_func = self._apply_optimizations(func, optimization_suggestions)

            # Profile optimized function
            optimized_result = self.profile_function(optimized_func, *args, **kwargs)

            # Compare results
            comparison = self._compare_performance(
                original_result, optimized_result, func.__name__
            )

            # Save optimization report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = (
                self.reports_dir
                / f"optimization_report_{func.__name__}_{timestamp}.json"
            )

            with open(report_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "function": func.__name__,
                        "suggestions": optimization_suggestions,
                        "comparison": comparison,
                    },
                    f,
                    indent=2,
                )

            self.logger.info(f"Optimization report saved to {report_file}")

            return optimized_func
        except Exception as e:
            self.logger.error(f"Failed to optimize performance: {e}")
            raise

    def _analyze_source_code(self, source: str) -> List[Dict[str, Any]]:
        """Analyze source code for optimization opportunities."""
        suggestions = []

        try:
            # Check for list comprehensions
            if "for" in source and "[" in source and "]" in source:
                suggestions.append(
                    {
                        "type": "list_comprehension",
                        "description": "Consider using list comprehension for better performance",
                        "severity": "medium",
                    }
                )

            # Check for nested loops
            if source.count("for") > 1:
                suggestions.append(
                    {
                        "type": "nested_loops",
                        "description": "Consider optimizing nested loops",
                        "severity": "high",
                    }
                )

            # Check for string concatenation
            if "+" in source and '"' in source:
                suggestions.append(
                    {
                        "type": "string_concatenation",
                        "description": "Consider using f-strings or str.join()",
                        "severity": "low",
                    }
                )

            # Check for global variables
            if "global" in source:
                suggestions.append(
                    {
                        "type": "global_variables",
                        "description": "Consider avoiding global variables",
                        "severity": "medium",
                    }
                )

            return suggestions
        except Exception as e:
            self.logger.error(f"Failed to analyze source code: {e}")
            raise

    def _apply_optimizations(
        self, func: callable, suggestions: List[Dict[str, Any]]
    ) -> callable:
        """Apply optimizations to function.

        Note: This is a simplified version that returns the original function
        for safety. In a production environment, consider using AST manipulation
        or other safe code transformation techniques.
        """
        try:
            # For safety, we'll return the original function
            # In a production environment, you would implement safe code transformations
            # using AST manipulation or other techniques that don't require exec()

            self.logger.warning(
                "Dynamic code optimization disabled for security. "
                "Consider using AST manipulation for safe code transformations."
            )

            # Log the suggestions for manual review
            for suggestion in suggestions:
                self.logger.info(f"Optimization suggestion: {suggestion}")

            # Return the original function
            return func

        except Exception as e:
            self.logger.error(f"Failed to apply optimizations: {e}")
            raise

    def _compare_performance(
        self, original: Any, optimized: Any, func_name: str
    ) -> Dict[str, Any]:
        """Compare original and optimized performance."""
        try:
            # Get performance metrics
            original_stats = pstats.Stats(
                self.prof_dir / f"profile_{func_name}_original.prof"
            )
            optimized_stats = pstats.Stats(
                self.prof_dir / f"profile_{func_name}_optimized.prof"
            )

            # Calculate improvements
            improvement = {
                "total_time": (original_stats.total_tt - optimized_stats.total_tt)
                / original_stats.total_tt
                * 100,
                "function_calls": (
                    original_stats.total_calls - optimized_stats.total_calls
                )
                / original_stats.total_calls
                * 100,
                "primitive_calls": (
                    original_stats.prim_calls - optimized_stats.prim_calls
                )
                / original_stats.prim_calls
                * 100,
            }

            return improvement
        except Exception as e:
            self.logger.error(f"Failed to compare performance: {e}")
            raise

    def _generate_performance_plots(self, analysis: Dict[str, Any]):
        """Generate performance visualization plots."""
        try:
            # Set style
            plt.style.use("seaborn")

            # Create plots directory
            plots_dir = self.reports_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Generate plots for each profile
            for profile in analysis["profiles"]:
                # Top functions by cumulative time
                plt.figure(figsize=(12, 6))
                top_funcs = profile["top_functions"][:10]  # Top 10 functions
                plt.barh(
                    [f["name"] for f in top_funcs],
                    [f["cumulative_time"] for f in top_funcs],
                )
                plt.title("Top Functions by Cumulative Time")
                plt.xlabel("Cumulative Time (seconds)")
                plt.tight_layout()
                plt.savefig(
                    plots_dir / f"top_functions_{Path(profile['file']).stem}.png"
                )
                plt.close()

            # Generate comparison plots
            if len(analysis["profiles"]) > 1:
                # Compare total times
                plt.figure(figsize=(10, 6))
                plt.bar(
                    [Path(p["file"]).stem for p in analysis["profiles"]],
                    [p["total_time"] for p in analysis["profiles"]],
                )
                plt.title("Total Time Comparison")
                plt.ylabel("Total Time (seconds)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plots_dir / "total_time_comparison.png")
                plt.close()

                # Compare function calls
                plt.figure(figsize=(10, 6))
                plt.bar(
                    [Path(p["file"]).stem for p in analysis["profiles"]],
                    [p["function_calls"] for p in analysis["profiles"]],
                )
                plt.title("Function Calls Comparison")
                plt.ylabel("Number of Calls")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plots_dir / "function_calls_comparison.png")
                plt.close()

            self.logger.info(f"Performance plots saved to {plots_dir}")
        except Exception as e:
            self.logger.error(f"Failed to generate performance plots: {e}")
            raise


def main():
    """Main entry point for the performance management script."""
    parser = argparse.ArgumentParser(description="Performance Manager")
    parser.add_argument(
        "command",
        choices=["profile", "analyze", "optimize", "monitor", "report"],
        help="Command to run",
    )
    parser.add_argument("--function", help="Function to profile or optimize")
    parser.add_argument(
        "--type",
        choices=["cpu", "memory", "line"],
        default="cpu",
        help="Type of profiling",
    )
    parser.add_argument(
        "--args", type=json.loads, help="Function arguments as JSON string"
    )
    parser.add_argument("--profiles", help="Glob pattern for profile files")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument(
        "--duration", type=int, default=300, help="Monitoring duration in seconds"
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Monitoring interval in seconds"
    )
    parser.add_argument(
        "--format",
        choices=["json", "html", "pdf"],
        default="json",
        help="Report format",
    )
    parser.add_argument("--help", action="store_true", help="Show usage examples")
    args = parser.parse_args()

    if args.help:
        print(__doc__)

    manager = PerformanceManager()
    if args.command == "profile":
        if args.type == "cpu":
            manager.profile_function(args.function, **(args.args or {}))
        elif args.type == "memory":
            manager.profile_memory(args.function, **(args.args or {}))
        elif args.type == "line":
            manager.profile_line(args.function, **(args.args or {}))
    elif args.command == "analyze":
        manager.analyze_performance(glob.glob(args.profiles))
    elif args.command == "optimize":
        # Implement optimization
        pass
    elif args.command == "monitor":
        # Implement monitoring
        pass
    elif args.command == "report":
        # Implement reporting
        pass


if __name__ == "__main__":
    main()
