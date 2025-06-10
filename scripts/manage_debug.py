#!/usr/bin/env python3
"""
Debug management script.
Provides commands for managing error handling and debugging.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import traceback
import pdb
import ipdb
import logging
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

class DebugManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the debug manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.debug_dir = Path("debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("reports/debug")
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

    def debug_function(self, func: callable, *args, **kwargs):
        """Debug a function with interactive debugging."""
        self.logger.info(f"Debugging function: {func.__name__}")
        
        try:
            # Set up debugger
            debugger = ipdb.set_trace
            
            # Run function with debugger
            result = func(*args, **kwargs)
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to debug function: {e}")
            raise

    def analyze_errors(self, log_files: List[str]):
        """Analyze error logs."""
        self.logger.info("Analyzing error logs")
        
        try:
            # Load error logs
            errors = []
            for file in log_files:
                with open(file) as f:
                    for line in f:
                        if "ERROR" in line:
                            errors.append({
                                "timestamp": line.split()[0],
                                "message": line.split("ERROR")[-1].strip(),
                                "file": file
                            })
            
            # Analyze errors
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "total_errors": len(errors),
                "error_types": {},
                "error_frequency": {},
                "error_timeline": []
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
            analysis["error_timeline"] = sorted(
                errors,
                key=lambda x: x["timestamp"]
            )
            
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

    def monitor_errors(self, duration: int = 300):
        """Monitor errors in real-time."""
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
                                errors.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "message": line.split("ERROR")[-1].strip(),
                                    "file": str(log_file)
                                })
                
                time.sleep(1)  # Wait 1 second between checks
            
            # Save monitoring results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            monitoring_file = self.debug_dir / f"error_monitoring_{timestamp}.json"
            
            with open(monitoring_file, "w") as f:
                json.dump({
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "errors": errors
                }, f, indent=2)
            
            # Print monitoring results
            self._print_monitoring_results(errors)
            
            return errors
        except Exception as e:
            self.logger.error(f"Failed to monitor errors: {e}")
            raise

    def fix_errors(self, error_analysis: Dict[str, Any]):
        """Suggest fixes for common errors."""
        self.logger.info("Suggesting error fixes")
        
        try:
            # Generate fix suggestions
            suggestions = []
            for error_type, count in error_analysis["error_types"].items():
                if error_type == "ValueError":
                    suggestions.append({
                        "error_type": error_type,
                        "description": "Check input values and type conversions",
                        "example": "Ensure numeric inputs are valid numbers",
                        "severity": "high" if count > 10 else "medium"
                    })
                elif error_type == "KeyError":
                    suggestions.append({
                        "error_type": error_type,
                        "description": "Check dictionary key existence",
                        "example": "Use dict.get() with default values",
                        "severity": "high" if count > 10 else "medium"
                    })
                elif error_type == "TypeError":
                    suggestions.append({
                        "error_type": error_type,
                        "description": "Check function argument types",
                        "example": "Add type hints and validation",
                        "severity": "high" if count > 10 else "medium"
                    })
                elif error_type == "AttributeError":
                    suggestions.append({
                        "error_type": error_type,
                        "description": "Check object attribute existence",
                        "example": "Use hasattr() before accessing attributes",
                        "severity": "high" if count > 10 else "medium"
                    })
                elif error_type == "ImportError":
                    suggestions.append({
                        "error_type": error_type,
                        "description": "Check module imports and dependencies",
                        "example": "Verify package installation and paths",
                        "severity": "high" if count > 10 else "medium"
                    })
            
            # Save suggestions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suggestions_file = self.reports_dir / f"error_fixes_{timestamp}.json"
            
            with open(suggestions_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "suggestions": suggestions
                }, f, indent=2)
            
            # Print suggestions
            self._print_fix_suggestions(suggestions)
            
            return suggestions
        except Exception as e:
            self.logger.error(f"Failed to suggest error fixes: {e}")
            raise

    def _print_monitoring_results(self, errors: List[Dict[str, Any]]):
        """Print error monitoring results."""
        print("\nError Monitoring Results:")
        print(f"\nTotal Errors: {len(errors)}")
        
        if errors:
            print("\nRecent Errors:")
            for error in errors[-5:]:  # Show last 5 errors
                print(f"\nTimestamp: {error['timestamp']}")
                print(f"Message: {error['message']}")
                print(f"File: {error['file']}")

    def _print_fix_suggestions(self, suggestions: List[Dict[str, Any]]):
        """Print error fix suggestions."""
        print("\nError Fix Suggestions:")
        
        for suggestion in suggestions:
            print(f"\nError Type: {suggestion['error_type']}")
            print(f"Description: {suggestion['description']}")
            print(f"Example: {suggestion['example']}")
            print(f"Severity: {suggestion['severity']}")

    def _generate_error_plots(self, analysis: Dict[str, Any]):
        """Generate error visualization plots."""
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
                autopct="%1.1f%%"
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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Debug Manager")
    parser.add_argument(
        "command",
        choices=["debug", "analyze", "monitor", "fix"],
        help="Command to execute"
    )
    parser.add_argument(
        "--function",
        help="Function to debug"
    )
    parser.add_argument(
        "--log-files",
        nargs="+",
        help="Log files to analyze"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration for monitoring in seconds"
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
    manager = DebugManager()
    
    commands = {
        "debug": lambda: manager.debug_function(
            eval(args.function),
            *eval(args.args) if args.args else [],
            **(args.kwargs or {})
        ),
        "analyze": lambda: manager.analyze_errors(args.log_files),
        "monitor": lambda: manager.monitor_errors(args.duration),
        "fix": lambda: manager.fix_errors(
            manager.analyze_errors(args.log_files)
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