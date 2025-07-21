#!/usr/bin/env python3
"""
Data management script.
Provides commands for managing datasets, including import, export, validation, and cleaning.

This script supports:
- Importing datasets
- Exporting datasets
- Validating data quality
- Cleaning data

Usage:
    python manage_data.py <command> [options]

Commands:
    import      Import a dataset
    export      Export a dataset
    validate    Validate data quality
    clean       Clean data

Examples:
    # Import a dataset
    python manage_data.py import --file data/input.csv

    # Export a dataset
    python manage_data.py export --file data/output.csv

    # Validate data quality
    python manage_data.py validate --file data/input.csv

    # Clean data
    python manage_data.py clean --file data/input.csv --output data/cleaned.csv
"""

import argparse
import json
import logging
import logging.config
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from utils.launch_utils import setup_logging


class DataManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the data manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Set up logging for the service."""
        return setup_logging(service_name="service")

    def backup_data(self, backup_name: Optional[str] = None):
        """Backup data to a specified location."""
        self.logger.info(f"Backing up data to {backup_name}")

        try:
            # Create backup directory
            backup_dir = self.data_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Generate backup name if not provided
            if not backup_name:
                backup_name = f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            backup_path = backup_dir / f"{backup_name}.zip"

            # Archive data directory
            shutil.make_archive(
                str(backup_path).replace(".zip", ""), "zip", str(self.data_dir)
            )

            self.logger.info(f"Data backup created: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup data: {e}")
            return False

    def restore_data(self, backup_name: str):
        """Restore data from backup."""
        self.logger.info(f"Restoring data from {backup_name}...")

        backup_path = self.backup_dir / f"{backup_name}.zip"
        if not backup_path.exists():
            self.logger.error(f"Backup not found: {backup_path}")
            return False

        try:
            # Restore backup
            shutil.unpack_archive(str(backup_path), self.data_dir, "zip")

            self.logger.info("Data restored successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore data: {e}")
            return False

    def clean_data(self, days: int = 30):
        """Clean old data files."""
        self.logger.info(f"Cleaning data older than {days} days...")

        if not self.data_dir.exists():
            self.logger.error("Data directory not found")
            return False

        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0

        try:
            # Clean data files
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file():
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_date < cutoff_date:
                        file_path.unlink()
                        cleaned_count += 1

            # Clean old backups
            for backup_path in self.backup_dir.glob("*.zip"):
                backup_date = datetime.fromtimestamp(backup_path.stat().st_mtime)
                if backup_date < cutoff_date:
                    backup_path.unlink()
                    cleaned_count += 1

            self.logger.info(f"Cleaned {cleaned_count} files")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clean data: {e}")
            return False

    def validate_data(self):
        """Validate data files."""
        self.logger.info("Validating data files...")

        if not self.data_dir.exists():
            self.logger.error("Data directory not found")
            return False

        validation_results = {"valid": [], "invalid": []}

        try:
            # Validate CSV files
            for csv_path in self.data_dir.rglob("*.csv"):
                try:
                    pd.read_csv(csv_path)
                    validation_results["valid"].append(str(csv_path))
                except Exception as e:
                    validation_results["invalid"].append((str(csv_path), str(e)))

            # Validate JSON files
            for json_path in self.data_dir.rglob("*.json"):
                try:
                    with open(json_path) as f:
                        json.load(f)
                    validation_results["valid"].append(str(json_path))
                except Exception as e:
                    validation_results["invalid"].append((str(json_path), str(e)))

            # Log results
            if validation_results["valid"]:
                self.logger.info("Valid files:")
                for file in validation_results["valid"]:
                    self.logger.info(f"  {file}")

            if validation_results["invalid"]:
                self.logger.error("Invalid files:")
                for file, error in validation_results["invalid"]:
                    self.logger.error(f"  {file}: {error}")

            return len(validation_results["invalid"]) == 0
        except Exception as e:
            self.logger.error(f"Failed to validate data: {e}")
            return False

    def optimize_data(self):
        """Optimize data files."""
        self.logger.info("Optimizing data files...")

        if not self.data_dir.exists():
            self.logger.error("Data directory not found")
            return False

        try:
            # Optimize CSV files
            for csv_path in self.data_dir.rglob("*.csv"):
                try:
                    # Read CSV
                    df = pd.read_csv(csv_path)

                    # Optimize data types
                    for col in df.columns:
                        if df[col].dtype == "object":
                            # Try to convert to numeric
                            try:
                                df[col] = pd.to_numeric(df[col], errors="ignore")
                            except Exception as e:
                                self.logger.warning(
                                    f"⚠️ Could not convert column {col} to numeric: {e}"
                                )
                                continue

                    # Save optimized CSV
                    df.to_csv(csv_path, index=False)
                    self.logger.info(f"Optimized {csv_path}")
                except Exception as e:
                    self.logger.error(f"Failed to optimize {csv_path}: {e}")

            self.logger.info("Data optimization completed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to optimize data: {e}")
            return False

    def list_data(self, pattern: Optional[str] = None):
        """List data files."""
        self.logger.info("Listing data files...")

        if not self.data_dir.exists():
            self.logger.error("Data directory not found")
            return False

        try:
            # List files
            if pattern:
                files = list(self.data_dir.rglob(pattern))
            else:
                files = list(self.data_dir.rglob("*"))

            # Print file information
            for file_path in files:
                if file_path.is_file():
                    size = file_path.stat().st_size
                    modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    print(f"{file_path} ({size:,} bytes, modified: {modified})")

            return True
        except Exception as e:
            self.logger.error(f"Failed to list data: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Data Manager")
    parser.add_argument(
        "command",
        choices=["backup", "restore", "clean", "validate", "optimize", "list"],
        help="Command to execute",
    )
    parser.add_argument("--backup-name", help="Name of backup to create or restore")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days for cleaning old data"
    )
    parser.add_argument("--pattern", help="File pattern for listing data")

    args = parser.parse_args()
    manager = DataManager()

    commands = {
        "backup": lambda: manager.backup_data(args.backup_name),
        "restore": lambda: (
            manager.restore_data(args.backup_name) if args.backup_name else False
        ),
        "clean": lambda: manager.clean_data(args.days),
        "validate": manager.validate_data,
        "optimize": manager.optimize_data,
        "list": lambda: manager.list_data(args.pattern),
    }

    if args.command in commands:
        if args.command == "restore" and not args.backup_name:
            parser.error("restore requires --backup-name")

        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
