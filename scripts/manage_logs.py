#!/usr/bin/env python3
"""
Log management script.
Provides commands for managing application logs, including viewing, archiving, and cleaning logs.

This script supports:
- Viewing logs
- Archiving logs
- Cleaning old logs

Usage:
    python manage_logs.py <command> [options]

Commands:
    view        View logs
    archive     Archive logs
    clean       Clean old logs

Examples:
    # View logs
    python manage_logs.py view --log-file logs/app.log

    # Archive logs
    python manage_logs.py archive --log-dir logs/ --output archive/logs.zip

    # Clean old logs
    python manage_logs.py clean --log-dir logs/ --days 30
"""

import argparse
import gzip
import logging
import logging.config
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class LogManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the log manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.logs_dir = Path("logs")
        self.archive_dir = Path("logs/archive")

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

    def rotate_logs(self, days: int = 7):
        """Rotate log files."""
        self.logger.info(f"Rotating logs older than {days} days...")

        if not self.logs_dir.exists():
            self.logger.error("Logs directory not found")
            return False

        # Create archive directory if it doesn't exist
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        cutoff_date = datetime.now() - timedelta(days=days)
        rotated_count = 0

        try:
            # Rotate log files
            for log_file in self.logs_dir.glob("*.log"):
                if log_file.is_file():
                    file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        # Create archive name
                        archive_name = (
                            f"{log_file.stem}_{file_date.strftime('%Y%m%d')}.log.gz"
                        )
                        archive_path = self.archive_dir / archive_name

                        # Compress and move file
                        with open(log_file, "rb") as f_in:
                            with gzip.open(archive_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        # Remove original file
                        log_file.unlink()
                        rotated_count += 1

            self.logger.info(f"Rotated {rotated_count} log files")
            return True
        except Exception as e:
            self.logger.error(f"Failed to rotate logs: {e}")
            return False

    def clean_logs(self, days: int = 30):
        """Clean old log files."""
        self.logger.info(f"Cleaning logs older than {days} days...")

        if not self.logs_dir.exists():
            self.logger.error("Logs directory not found")
            return False

        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0

        try:
            # Clean current logs
            for log_file in self.logs_dir.glob("*.log"):
                if log_file.is_file():
                    file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        log_file.unlink()
                        cleaned_count += 1

            # Clean archived logs
            for archive_file in self.archive_dir.glob("*.log.gz"):
                if archive_file.is_file():
                    file_date = datetime.fromtimestamp(archive_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        archive_file.unlink()
                        cleaned_count += 1

            self.logger.info(f"Cleaned {cleaned_count} log files")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clean logs: {e}")
            return False

    def analyze_logs(self, pattern: Optional[str] = None, days: int = 7):
        """Analyze log files."""
        self.logger.info("Analyzing log files...")

        if not self.logs_dir.exists():
            self.logger.error("Logs directory not found")
            return False

        cutoff_date = datetime.now() - timedelta(days=days)
        analysis_results = {
            "error_count": 0,
            "warning_count": 0,
            "error_patterns": {},
            "warning_patterns": {},
        }

        try:
            # Analyze current logs
            for log_file in self.logs_dir.glob("*.log"):
                if log_file.is_file():
                    file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_date >= cutoff_date:
                        self._analyze_log_file(log_file, pattern, analysis_results)

            # Analyze archived logs
            for archive_file in self.archive_dir.glob("*.log.gz"):
                if archive_file.is_file():
                    file_date = datetime.fromtimestamp(archive_file.stat().st_mtime)
                    if file_date >= cutoff_date:
                        with gzip.open(archive_file, "rt") as f:
                            self._analyze_log_content(f, pattern, analysis_results)

            # Print results
            print("\nLog Analysis Results:")
            print(f"Total Errors: {analysis_results['error_count']}")
            print(f"Total Warnings: {analysis_results['warning_count']}")

            if analysis_results["error_patterns"]:
                print("\nError Patterns:")
                for pattern, count in analysis_results["error_patterns"].items():
                    print(f"  {pattern}: {count}")

            if analysis_results["warning_patterns"]:
                print("\nWarning Patterns:")
                for pattern, count in analysis_results["warning_patterns"].items():
                    print(f"  {pattern}: {count}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to analyze logs: {e}")
            return False

    def _analyze_log_file(
        self, log_file: Path, pattern: Optional[str], results: Dict[str, Any]
    ):
        """Analyze a single log file."""
        try:
            with open(log_file) as f:
                self._analyze_log_content(f, pattern, results)
        except Exception as e:
            self.logger.error(f"Failed to analyze {log_file}: {e}")

    def _analyze_log_content(
        self, file_obj, pattern: Optional[str], results: Dict[str, Any]
    ):
        """Analyze log content."""
        for line in file_obj:
            if pattern and pattern not in line:
                continue

            if "ERROR" in line:
                results["error_count"] += 1
                self._update_pattern_count(results["error_patterns"], line)
            elif "WARNING" in line:
                results["warning_count"] += 1
                self._update_pattern_count(results["warning_patterns"], line)

    def _update_pattern_count(self, patterns: Dict[str, int], line: str):
        """Update pattern count in analysis results."""
        # Extract the main message part
        try:
            message = line.split(" - ")[-1].strip()
            patterns[message] = patterns.get(message, 0) + 1
        except Exception as e:
            self.logger.warning(f"Failed to update pattern count: {e}")

    def export_logs(self, output_dir: str, days: int = 7):
        """Export logs to a directory."""
        self.logger.info(f"Exporting logs from the last {days} days...")

        if not self.logs_dir.exists():
            self.logger.error("Logs directory not found")
            return False

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cutoff_date = datetime.now() - timedelta(days=days)
        exported_count = 0

        try:
            # Export current logs
            for log_file in self.logs_dir.glob("*.log"):
                if log_file.is_file():
                    file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_date >= cutoff_date:
                        shutil.copy2(log_file, output_path / log_file.name)
                        exported_count += 1

            # Export archived logs
            for archive_file in self.archive_dir.glob("*.log.gz"):
                if archive_file.is_file():
                    file_date = datetime.fromtimestamp(archive_file.stat().st_mtime)
                    if file_date >= cutoff_date:
                        shutil.copy2(archive_file, output_path / archive_file.name)
                        exported_count += 1

            self.logger.info(f"Exported {exported_count} log files to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Log Manager")
    parser.add_argument(
        "command",
        choices=["rotate", "clean", "analyze", "export"],
        help="Command to execute",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days for log operations"
    )
    parser.add_argument("--pattern", help="Pattern to search for in logs")
    parser.add_argument("--output-dir", help="Output directory for log export")

    args = parser.parse_args()
    manager = LogManager()

    commands = {
        "rotate": lambda: manager.rotate_logs(args.days),
        "clean": lambda: manager.clean_logs(args.days),
        "analyze": lambda: manager.analyze_logs(args.pattern, args.days),
        "export": lambda: manager.export_logs(args.output_dir, args.days)
        if args.output_dir
        else False,
    }

    if args.command in commands:
        if args.command == "export" and not args.output_dir:
            parser.error("export requires --output-dir")

        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
