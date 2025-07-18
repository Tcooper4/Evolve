#!/usr/bin/env python3
"""
Application Management Script

Provides utilities for managing the trading application including:
- Dependency installation
- Testing
- Code formatting and linting
- Docker operations
- Cleanup tasks
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from utils.service_utils import setup_service_logging


class ApplicationManager:
    """Manages application operations and maintenance tasks."""

    def __init__(self, config_path: str = "config/app_config.yaml"):
        """
        Initialize the application manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = setup_service_logging("application_manager")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "server": {"port": 8000, "host": "localhost"},
            "database": {"url": "sqlite:///trading.db"},
            "logging": {"level": "INFO", "file": "logs/app.log"},
        }

    def run_command(self, command: List[str], cwd: Optional[str] = None) -> int:
        """Run a shell command and return its exit code.

        Args:
            command: List of command arguments
            cwd: Working directory for the command

        Returns:
            Exit code of the command
        """
        try:
            process = subprocess.run(
                command, cwd=cwd, check=True, capture_output=True, text=True
            )
            return process.returncode
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            self.logger.error(f"Output: {e.output}")
            return e.returncode

    def install_dependencies(self):
        """Install application dependencies.

        Returns:
            True if installation is successful, False otherwise
        """
        self.logger.info("Installing dependencies...")

        # Install main dependencies
        if self.run_command(["pip", "install", "-r", "requirements.txt"]) != 0:
            self.logger.error("Failed to install main dependencies")
            return False

        # Install development dependencies
        if self.run_command(["pip", "install", "-r", "requirements-dev.txt"]) != 0:
            self.logger.error("Failed to install development dependencies")
            return False

        self.logger.info("Dependencies installed successfully")
        return True

    def run_tests(self):
        """Run application tests.

        Returns:
            True if tests pass, False otherwise
        """
        self.logger.info("Running tests...")

        if self.run_command(["pytest", "tests/", "-v"]) != 0:
            self.logger.error("Tests failed")
            return False

        self.logger.info("Tests completed successfully")
        return True

    def run_linting(self):
        """Run code linting.

        Returns:
            True if linting passes, False otherwise
        """
        self.logger.info("Running linting...")

        # Run flake8
        if self.run_command(["flake8", "trading", "tests"]) != 0:
            self.logger.error("Flake8 checks failed")
            return False

        # Run black
        if self.run_command(["black", "--check", "trading", "tests"]) != 0:
            self.logger.error("Black checks failed")
            return False

        # Run isort
        if self.run_command(["isort", "--check-only", "trading", "tests"]) != 0:
            self.logger.error("isort checks failed")
            return False

        # Run mypy
        if self.run_command(["mypy", "trading", "tests"]) != 0:
            self.logger.error("Type checking failed")
            return False

        self.logger.info("Linting completed successfully")
        return True

    def format_code(self):
        """Format code using black and isort.

        Returns:
            True if formatting is successful, False otherwise
        """
        self.logger.info("Formatting code...")

        # Run black
        if self.run_command(["black", "trading", "tests"]) != 0:
            self.logger.error("Black formatting failed")
            return False

        # Run isort
        if self.run_command(["isort", "trading", "tests"]) != 0:
            self.logger.error("isort formatting failed")
            return False

        self.logger.info("Code formatting completed successfully")
        return True

    def build_docker(self):
        """Build Docker image.

        Returns:
            True if build is successful, False otherwise
        """
        self.logger.info("Building Docker image...")

        if self.run_command(["docker", "build", "-t", "trading-app", "."]) != 0:
            self.logger.error("Docker build failed")
            return False

        self.logger.info("Docker image built successfully")
        return True

    def run_docker(self):
        """Run Docker container.

        Returns:
            True if container runs successfully, False otherwise
        """
        self.logger.info("Running Docker container...")

        if (
            self.run_command(
                [
                    "docker",
                    "run",
                    "-d",
                    "-p",
                    f"{self.config['server']['port']}:{self.config['server']['port']}",
                    "--name",
                    "trading-app",
                    "trading-app",
                ]
            )
            != 0
        ):
            self.logger.error("Failed to run Docker container")
            return False

        self.logger.info("Docker container running successfully")
        return True

    def stop_docker(self):
        """Stop Docker container.

        Returns:
            True if container stops successfully, False otherwise
        """
        self.logger.info("Stopping Docker container...")

        if self.run_command(["docker", "stop", "trading-app"]) != 0:
            self.logger.error("Failed to stop Docker container")
            return False

        self.logger.info("Docker container stopped successfully")
        return True

    def clean(self):
        """Clean up temporary files and caches.

        Returns:
            True if cleanup is successful, False otherwise
        """
        self.logger.info("Cleaning up...")

        # Remove Python cache files
        if (
            self.run_command(
                [
                    "find",
                    ".",
                    "-type",
                    "d",
                    "-name",
                    "__pycache__",
                    "-exec",
                    "rm",
                    "-r",
                    "{}",
                    "+",
                ]
            )
            != 0
        ):
            self.logger.error("Failed to remove Python cache files")
            return False

        # Remove .pyc files
        if (
            self.run_command(
                [
                    "find",
                    ".",
                    "-name",
                    "*.pyc",
                    "-delete",
                ]
            )
            != 0
        ):
            self.logger.error("Failed to remove .pyc files")
            return False

        # Remove test cache
        if (
            self.run_command(
                [
                    "find",
                    ".",
                    "-type",
                    "d",
                    "-name",
                    ".pytest_cache",
                    "-exec",
                    "rm",
                    "-r",
                    "{}",
                    "+",
                ]
            )
            != 0
        ):
            self.logger.error("Failed to remove pytest cache")
            return False

        self.logger.info("Cleanup completed successfully")
        return True

    def check_health(self):
        """Check application health.

        Returns:
            True if application is healthy, False otherwise
        """
        self.logger.info("Checking application health...")

        # Check if main application files exist
        required_files = [
            "app.py",
            "requirements.txt",
            "config/app_config.yaml",
        ]

        for file_path in required_files:
            if not Path(file_path).exists():
                self.logger.error(f"Required file {file_path} not found")
                return False

        # Check if logs directory exists
        if not Path("logs").exists():
            self.logger.warning("Logs directory not found")

        self.logger.info("Application health check passed")
        return True

    def backup_data(self):
        """Backup application data.

        Returns:
            True if backup is successful, False otherwise
        """
        self.logger.info("Backing up application data...")

        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)

        # Backup database
        if Path("trading.db").exists():
            if (
                self.run_command(
                    ["cp", "trading.db", f"backups/trading.db.backup"]
                )
                != 0
            ):
                self.logger.error("Failed to backup database")
                return False

        # Backup logs
        if Path("logs").exists():
            if (
                self.run_command(
                    ["tar", "-czf", "backups/logs.tar.gz", "logs/"]
                )
                != 0
            ):
                self.logger.error("Failed to backup logs")
                return False

        self.logger.info("Data backup completed successfully")
        return True

    def restore_data(self):
        """Restore application data.

        Returns:
            True if restore is successful, False otherwise
        """
        self.logger.info("Restoring application data...")

        # Restore database
        if Path("backups/trading.db.backup").exists():
            if (
                self.run_command(
                    ["cp", "backups/trading.db.backup", "trading.db"]
                )
                != 0
            ):
                self.logger.error("Failed to restore database")
                return False

        # Restore logs
        if Path("backups/logs.tar.gz").exists():
            if (
                self.run_command(
                    ["tar", "-xzf", "backups/logs.tar.gz"]
                )
                != 0
            ):
                self.logger.error("Failed to restore logs")
                return False

        self.logger.info("Data restore completed successfully")
        return True


def main():
    """Main function to run the application manager."""
    parser = argparse.ArgumentParser(description="Trading Application Manager")
    parser.add_argument(
        "command",
        choices=[
            "install",
            "test",
            "lint",
            "format",
            "build",
            "run",
            "stop",
            "clean",
            "health",
            "backup",
            "restore",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        default="config/app_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    manager = ApplicationManager(args.config)

    commands = {
        "install": manager.install_dependencies,
        "test": manager.run_tests,
        "lint": manager.run_linting,
        "format": manager.format_code,
        "build": manager.build_docker,
        "run": manager.run_docker,
        "stop": manager.stop_docker,
        "clean": manager.clean,
        "health": manager.check_health,
        "backup": manager.backup_data,
        "restore": manager.restore_data,
    }

    success = commands[args.command]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
