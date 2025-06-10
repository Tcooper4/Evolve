#!/usr/bin/env python3
"""
Application management script.
Provides commands for managing the application lifecycle.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import subprocess
from pathlib import Path
from typing import List, Optional

class ApplicationManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the application manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")

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

    def run_command(self, command: List[str], cwd: Optional[str] = None) -> int:
        """Run a shell command and return its exit code."""
        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True
            )
            return process.returncode
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            self.logger.error(f"Output: {e.output}")
            return e.returncode

    def install_dependencies(self):
        """Install application dependencies."""
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
        """Run application tests."""
        self.logger.info("Running tests...")
        
        if self.run_command(["pytest", "tests/", "-v"]) != 0:
            self.logger.error("Tests failed")
            return False
        
        self.logger.info("Tests completed successfully")
        return True

    def run_linting(self):
        """Run code linting."""
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
        """Format code using black and isort."""
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
        """Build Docker image."""
        self.logger.info("Building Docker image...")
        
        if self.run_command(["docker", "build", "-t", "trading-app", "."]) != 0:
            self.logger.error("Docker build failed")
            return False
        
        self.logger.info("Docker image built successfully")
        return True

    def run_docker(self):
        """Run Docker container."""
        self.logger.info("Running Docker container...")
        
        if self.run_command([
            "docker", "run", "-d",
            "-p", f"{self.config['server']['port']}:{self.config['server']['port']}",
            "--name", "trading-app",
            "trading-app"
        ]) != 0:
            self.logger.error("Failed to run Docker container")
            return False
        
        self.logger.info("Docker container running successfully")
        return True

    def stop_docker(self):
        """Stop Docker container."""
        self.logger.info("Stopping Docker container...")
        
        if self.run_command(["docker", "stop", "trading-app"]) != 0:
            self.logger.error("Failed to stop Docker container")
            return False
        
        self.logger.info("Docker container stopped successfully")
        return True

    def clean(self):
        """Clean up temporary files and caches."""
        self.logger.info("Cleaning up...")
        
        # Remove Python cache files
        if self.run_command(["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-r", "{}", "+"]) != 0:
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
        choices=[
            "install",
            "test",
            "lint",
            "format",
            "build-docker",
            "run-docker",
            "stop-docker",
            "clean"
        ],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    manager = ApplicationManager()
    
    commands = {
        "install": manager.install_dependencies,
        "test": manager.run_tests,
        "lint": manager.run_linting,
        "format": manager.format_code,
        "build-docker": manager.build_docker,
        "run-docker": manager.run_docker,
        "stop-docker": manager.stop_docker,
        "clean": manager.clean
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 