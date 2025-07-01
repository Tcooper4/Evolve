#!/usr/bin/env python3
"""
Dependency management script.
Provides commands for managing project dependencies, including installation, updating, and listing packages.

This script supports:
- Installing dependencies
- Updating dependencies
- Listing installed packages

Usage:
    python manage_deps.py <command> [options]

Commands:
    install     Install dependencies
    update      Update dependencies
    list        List installed packages

Examples:
    # Install dependencies
    python manage_deps.py install

    # Update dependencies
    python manage_deps.py update

    # List installed packages
    python manage_deps.py list
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

class DependencyManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the dependency manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.requirements_file = Path("requirements.txt")
        self.dev_requirements_file = Path("requirements-dev.txt")

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

    def install_dependencies(self, dev: bool = False):
        """Install dependencies from requirements file."""
        self.logger.info("Installing dependencies...")
        
        requirements_file = self.dev_requirements_file if dev else self.requirements_file
        if not requirements_file.exists():
            self.logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )
            self.logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False

    def update_dependencies(self, dev: bool = False):
        """Update dependencies to their latest versions."""
        self.logger.info("Updating dependencies...")
        
        requirements_file = self.dev_requirements_file if dev else self.requirements_file
        if not requirements_file.exists():
            self.logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            # Update pip
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=True
            )
            
            # Update dependencies
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "-r", str(requirements_file)],
                check=True
            )
            
            self.logger.info("Dependencies updated successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to update dependencies: {e}")
            return False

    def freeze_dependencies(self, dev: bool = False):
        """Freeze current dependencies to requirements file."""
        self.logger.info("Freezing dependencies...")
        
        requirements_file = self.dev_requirements_file if dev else self.requirements_file
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            
            with open(requirements_file, "w") as f:
                f.write(result.stdout)
            
            self.logger.info(f"Dependencies frozen to {requirements_file}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to freeze dependencies: {e}")
            return False

    def check_dependencies(self):
        """Check for outdated dependencies."""
        self.logger.info("Checking for outdated dependencies...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                self.logger.info("Outdated dependencies found:")
                print(result.stdout)
            else:
                self.logger.info("All dependencies are up to date")
            
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to check dependencies: {e}")
            return False

    def clean_dependencies(self):
        """Clean up unused dependencies."""
        self.logger.info("Cleaning up dependencies...")
        
        try:
            # Uninstall unused packages
            subprocess.run(
                [sys.executable, "-m", "pip", "autoremove"],
                check=True
            )
            
            # Clean pip cache
            subprocess.run(
                [sys.executable, "-m", "pip", "cache", "purge"],
                check=True
            )
            
            self.logger.info("Dependencies cleaned successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clean dependencies: {e}")
            return False

    def verify_dependencies(self):
        """Verify all dependencies are installed correctly."""
        self.logger.info("Verifying dependencies...")
        
        try:
            # Check both requirements files
            for requirements_file in [self.requirements_file, self.dev_requirements_file]:
                if requirements_file.exists():
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "check", "-r", str(requirements_file)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        self.logger.error(f"Dependency conflicts found in {requirements_file}:")
                        print(result.stdout)
                        return False
            
            self.logger.info("All dependencies verified successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to verify dependencies: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Dependency Manager")
    parser.add_argument(
        "command",
        choices=["install", "update", "freeze", "check", "clean", "verify"],
        help="Command to execute"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use development requirements"
    )
    
    args = parser.parse_args()
    manager = DependencyManager()
    
    commands = {
        "install": lambda: manager.install_dependencies(args.dev),
        "update": lambda: manager.update_dependencies(args.dev),
        "freeze": lambda: manager.freeze_dependencies(args.dev),
        "check": manager.check_dependencies,
        "clean": manager.clean_dependencies,
        "verify": manager.verify_dependencies
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 