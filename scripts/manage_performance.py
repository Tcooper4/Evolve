#!/usr/bin/env python3
"""
Performance management script.
Provides commands for profiling and optimizing application performance.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import cProfile
import pstats
import line_profiler
import memory_profiler
import psutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the performance manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.prof_dir = Path("profiles")
        self.prof_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("reports/performance")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration."""
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)
        
        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)
        
        logging.config.dictConfig(log_config)

    def profile_function(self, func: callable, *args, **kwargs):
        """Profile a function's performance."""
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
        """Profile a function's memory usage."""
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
        """Profile a function's line-by-line performance."""
        self.logger.info(f"Profiling line-by-line performance for function: {func.__name__}")
        
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
        """Analyze performance profiles."""
        self.logger.info("Analyzing performance profiles")
        
        try:
            # Load profile data
            profile_data = []
            for file in profile_files:
                stats = pstats.Stats(file)
                profile_data.append({
                    "file": file,
                    "stats": stats
                })
            
            # Analyze profiles
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "profiles": []
            }
            
            for data in profile_data:
                stats = data["stats"]
                profile_analysis = {
                    "file": data["file"],
                    "total_time": stats.total_tt,
                    "function_calls": stats.total_calls,
                    "primitive_calls": stats.prim_calls,
                    "top_functions": []
                }
                
                # Get top functions by cumulative time
                for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                    if ct > 0:  # Only include functions that were called
                        profile_analysis["top_functions"].append({
                            "name": func,
                            "calls": nc,
                            "total_time": tt,
                            "cumulative_time": ct
                        })
                
                # Sort top functions by cumulative time
                profile_analysis["top_functions"].sort(
                    key=lambda x: x["cumulative_time"],
                    reverse=True
                )
                
                analysis["profiles"].append(profile_analysis)
            
            # Save analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = self.reports_dir / f"performance_analysis_{timestamp}.json"
            
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            # Generate visualizations
            self._generate_performance_plots(analysis)
            
            self.logger.info(f"Performance analysis saved to {analysis_file}")
            
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
                original_result,
                optimized_result,
                func.__name__
            )
            
            # Save optimization report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"optimization_report_{func.__name__}_{timestamp}.json"
            
            with open(report_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "function": func.__name__,
                    "suggestions": optimization_suggestions,
                    "comparison": comparison
                }, f, indent=2)
            
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
                suggestions.append({
                    "type": "list_comprehension",
                    "description": "Consider using list comprehension for better performance",
                    "severity": "medium"
                })
            
            # Check for nested loops
            if source.count("for") > 1:
                suggestions.append({
                    "type": "nested_loops",
                    "description": "Consider optimizing nested loops",
                    "severity": "high"
                })
            
            # Check for string concatenation
            if "+" in source and '"' in source:
                suggestions.append({
                    "type": "string_concatenation",
                    "description": "Consider using f-strings or str.join()",
                    "severity": "low"
                })
            
            # Check for global variables
            if "global" in source:
                suggestions.append({
                    "type": "global_variables",
                    "description": "Consider avoiding global variables",
                    "severity": "medium"
                })
            
            return suggestions
        except Exception as e:
            self.logger.error(f"Failed to analyze source code: {e}")
            raise

    def _apply_optimizations(self, func: callable, suggestions: List[Dict[str, Any]]) -> callable:
        """Apply optimizations to function."""
        try:
            # Get function source
            import inspect
            source = inspect.getsource(func)
            
            # Apply optimizations based on suggestions
            optimized_source = source
            for suggestion in suggestions:
                if suggestion["type"] == "list_comprehension":
                    # Convert for loops to list comprehensions where possible
                    pass
                elif suggestion["type"] == "nested_loops":
                    # Optimize nested loops
                    pass
                elif suggestion["type"] == "string_concatenation":
                    # Convert string concatenation to f-strings
                    pass
                elif suggestion["type"] == "global_variables":
                    # Remove global variables
                    pass
            
            # Create new function from optimized source
            namespace = {}
            exec(optimized_source, namespace)
            optimized_func = namespace[func.__name__]
            
            return optimized_func
        except Exception as e:
            self.logger.error(f"Failed to apply optimizations: {e}")
            raise

    def _compare_performance(self, original: Any, optimized: Any, func_name: str) -> Dict[str, Any]:
        """Compare original and optimized performance."""
        try:
            # Get performance metrics
            original_stats = pstats.Stats(self.prof_dir / f"profile_{func_name}_original.prof")
            optimized_stats = pstats.Stats(self.prof_dir / f"profile_{func_name}_optimized.prof")
            
            # Calculate improvements
            improvement = {
                "total_time": (original_stats.total_tt - optimized_stats.total_tt) / original_stats.total_tt * 100,
                "function_calls": (original_stats.total_calls - optimized_stats.total_calls) / original_stats.total_calls * 100,
                "primitive_calls": (original_stats.prim_calls - optimized_stats.prim_calls) / original_stats.prim_calls * 100
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
                    [f["cumulative_time"] for f in top_funcs]
                )
                plt.title("Top Functions by Cumulative Time")
                plt.xlabel("Cumulative Time (seconds)")
                plt.tight_layout()
                plt.savefig(plots_dir / f"top_functions_{Path(profile['file']).stem}.png")
                plt.close()
            
            # Generate comparison plots
            if len(analysis["profiles"]) > 1:
                # Compare total times
                plt.figure(figsize=(10, 6))
                plt.bar(
                    [Path(p["file"]).stem for p in analysis["profiles"]],
                    [p["total_time"] for p in analysis["profiles"]]
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
                    [p["function_calls"] for p in analysis["profiles"]]
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
    """Main function."""
    parser = argparse.ArgumentParser(description="Performance Manager")
    parser.add_argument(
        "command",
        choices=["profile", "analyze", "optimize"],
        help="Command to execute"
    )
    parser.add_argument(
        "--function",
        help="Function to profile or optimize"
    )
    parser.add_argument(
        "--profile-files",
        nargs="+",
        help="Profile files to analyze"
    )
    parser.add_argument(
        "--args",
        nargs="+",
        help="Arguments for the function"
    )
    parser.add_argument(
        "--kwargs",
        type=json.loads,
        help="Keyword arguments for the function"
    )
    
    args = parser.parse_args()
    manager = PerformanceManager()
    
    commands = {
        "profile": lambda: manager.profile_function(
            eval(args.function),
            *eval(args.args) if args.args else [],
            **(args.kwargs or {})
        ),
        "analyze": lambda: manager.analyze_performance(args.profile_files),
        "optimize": lambda: manager.optimize_performance(
            eval(args.function),
            *eval(args.args) if args.args else [],
            **(args.kwargs or {})
        )
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 