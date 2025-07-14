#!/usr/bin/env python3
"""
Application management script.
Provides commands for managing the application lifecycle, including installation, testing, linting, formatting, Docker builds, and cleaning.

This script supports:
- Installing dependencies
- Running tests
- Linting and formatting code
- Building and running Docker containers
- Cleaning build artifacts

Usage:
    python manage.py <command> [options]

Commands:
    install     Install dependencies
    test        Run tests
    lint        Run code linting
    format      Format code
    docker      Build and run Docker containers
    clean       Clean build artifacts

Examples:
    # Install dependencies
    python manage.py install

    # Run tests
    python manage.py test

    # Lint code
    python manage.py lint

    # Format code
    python manage.py format

    # Build Docker image
    python manage.py docker --build

    # Clean build artifacts
    python manage.py clean
"""

import argparse
import logging
import logging.config
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml


class ApplicationManager:
    """Application manager for handling lifecycle commands.

    This class provides methods for installing dependencies, running tests,
    linting, formatting, building Docker images, and cleaning up temporary files.

    Example:
        manager = ApplicationManager()
        manager.install_dependencies()
        manager.run_tests()
    """

    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the application manager.

        Args:
            config_path: Path to the application configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")

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

        # Remove test cache
        if self.run_command(["rm", "-rf", ".pytest_cache"]) != 0:
            self.logger.error("Failed to remove test cache")
            return False

        # Remove coverage files
        if self.run_command(["rm", "-rf", "htmlcov", ".coverage"]) != 0:
            self.logger.error("Failed to remove coverage files")
            return False

        self.logger.info("Cleanup completed successfully")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Trading Application Manager")
    parser.add_argument(
        "command",
        choices=["install", "test", "lint", "format", "docker", "clean"],
        help="Command to run",
    )
    parser.add_argument("--help", action="store_true", help="Show usage examples")
    args = parser.parse_args()

    if args.help:
        print(__doc__)
        return {
            "success": True,
            "result": {"status": "help_displayed", "command": "help"},
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    manager = ApplicationManager()
    result = {"status": "unknown", "command": args.command}

    if args.command == "install":
        success = manager.install_dependencies()
        result["status"] = "success" if success else "failed"
    elif args.command == "test":
        success = manager.run_tests()
        result["status"] = "success" if success else "failed"
    elif args.command == "lint":
        success = manager.run_linting()
        result["status"] = "success" if success else "failed"
    elif args.command == "format":
        success = manager.format_code()
        result["status"] = "success" if success else "failed"
    elif args.command == "docker":
        build_success = manager.build_docker()
        run_success = manager.run_docker() if build_success else False
        result["status"] = "success" if build_success and run_success else "failed"
    elif args.command == "clean":
        success = manager.clean()
        result["status"] = "success" if success else "failed"

    return result


if __name__ == "__main__":
    main()
