#!/usr/bin/env python3
"""
Documentation management script.
Provides commands for generating, building, and serving project documentation.

This script supports:
- Generating documentation
- Building documentation
- Serving documentation locally

Usage:
    python manage_docs.py <command> [options]

Commands:
    generate    Generate documentation
    build       Build documentation
    serve       Serve documentation locally

Examples:
    # Generate documentation
    python manage_docs.py generate

    # Build documentation
    python manage_docs.py build

    # Serve documentation locally
    python manage_docs.py serve --port 8000
"""

import argparse
import logging
import logging.config
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


class DocumentationManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the documentation manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.docs_dir = Path("docs")
        self.api_dir = self.docs_dir / "api"
        self.examples_dir = self.docs_dir / "examples"
        self.diagrams_dir = self.docs_dir / "diagrams"

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    from utils.launch_utils import setup_logging

def setup_logging():
    """Set up logging for the service."""
    return setup_logging(service_name="service")def generate_api_docs(self):
        """Generate API documentation."""
        self.logger.info("Generating API documentation...")

        try:
            # Create API directory
            self.api_dir.mkdir(parents=True, exist_ok=True)

            # Generate documentation using pdoc
            result = subprocess.run(
                ["pdoc", "--html", "--output-dir", str(self.api_dir), "trading"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                self.logger.info(f"API documentation generated in {self.api_dir}")
                return True
            else:
                self.logger.error("Failed to generate API documentation")
                print(result.stderr)
                return False
        except Exception as e:
            self.logger.error(f"Failed to generate API documentation: {e}")
            return False

    def generate_user_guides(self):
        """Generate user guides."""
        self.logger.info("Generating user guides...")

        try:
            # Create docs directory
            self.docs_dir.mkdir(parents=True, exist_ok=True)

            # Generate documentation using MkDocs
            result = subprocess.run(["mkdocs", "build"], capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info("User guides generated successfully")
                return True
            else:
                self.logger.error("Failed to generate user guides")
                print(result.stderr)
                return False
        except Exception as e:
            self.logger.error(f"Failed to generate user guides: {e}")
            return False

    def generate_examples(self):
        """Generate example notebooks."""
        self.logger.info("Generating example notebooks...")

        try:
            # Create examples directory
            self.examples_dir.mkdir(parents=True, exist_ok=True)

            # Generate example notebooks
            examples = [
                "basic_usage.ipynb",
                "advanced_features.ipynb",
                "custom_strategies.ipynb",
                "performance_analysis.ipynb",
            ]

            for example in examples:
                # Create notebook
                subprocess.run(
                    [
                        "jupyter",
                        "nbconvert",
                        "--to",
                        "notebook",
                        "--execute",
                        f"examples/{example}",
                    ],
                    capture_output=True,
                )

            self.logger.info(f"Example notebooks generated in {self.examples_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate example notebooks: {e}")
            return False

    def generate_diagrams(self):
        """Generate system architecture diagrams."""
        self.logger.info("Generating system architecture diagrams...")

        try:
            # Create diagrams directory
            self.diagrams_dir.mkdir(parents=True, exist_ok=True)

            # Generate diagrams using Graphviz
            diagrams = [
                "system_architecture.dot",
                "data_flow.dot",
                "component_interaction.dot",
            ]

            for diagram in diagrams:
                # Generate diagram
                subprocess.run(
                    [
                        "dot",
                        "-Tpng",
                        f"docs/diagrams/{diagram}",
                        "-o",
                        f"docs/diagrams/{diagram.replace('.dot', '.png')}",
                    ],
                    capture_output=True,
                )

            self.logger.info(f"Diagrams generated in {self.diagrams_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate diagrams: {e}")
            return False

    def serve_docs(self, port: int = 8000):
        """Serve documentation locally."""
        self.logger.info(f"Serving documentation on port {port}...")

        try:
            # Start MkDocs server
            subprocess.run(["mkdocs", "serve", "--port", str(port)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to serve documentation: {e}")
            return False

    def clean_docs(self):
        """Clean generated documentation."""
        self.logger.info("Cleaning documentation...")

        try:
            # Clean API documentation
            if self.api_dir.exists():
                shutil.rmtree(self.api_dir)

            # Clean example notebooks
            if self.examples_dir.exists():
                shutil.rmtree(self.examples_dir)

            # Clean diagrams
            if self.diagrams_dir.exists():
                shutil.rmtree(self.diagrams_dir)

            # Clean site directory
            site_dir = self.docs_dir / "site"
            if site_dir.exists():
                shutil.rmtree(site_dir)

            self.logger.info("Documentation cleaned successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clean documentation: {e}")
            return False

    def validate_docs(self):
        """Validate documentation."""
        self.logger.info("Validating documentation...")

        try:
            # Check for broken links
            result = subprocess.run(
                ["mkdocs", "build", "--strict"], capture_output=True, text=True
            )

            if result.returncode == 0:
                self.logger.info("Documentation validation passed")
                return True
            else:
                self.logger.error("Documentation validation failed")
                print(result.stderr)
                return False
        except Exception as e:
            self.logger.error(f"Failed to validate documentation: {e}")
            return False

    def update_docs(self):
        """Update documentation."""
        self.logger.info("Updating documentation...")

        try:
            # Clean existing documentation
            self.clean_docs()

            # Generate new documentation
            if not self.generate_api_docs():
                return False
            if not self.generate_user_guides():
                return False
            if not self.generate_examples():
                return False
            if not self.generate_diagrams():
                return False

            # Validate documentation
            if not self.validate_docs():
                return False

            self.logger.info("Documentation updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update documentation: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Documentation Manager")
    parser.add_argument(
        "command",
        choices=[
            "api",
            "guides",
            "examples",
            "diagrams",
            "serve",
            "clean",
            "validate",
            "update",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for serving documentation"
    )

    args = parser.parse_args()
    manager = DocumentationManager()

    commands = {
        "api": manager.generate_api_docs,
        "guides": manager.generate_user_guides,
        "examples": manager.generate_examples,
        "diagrams": manager.generate_diagrams,
        "serve": lambda: manager.serve_docs(args.port),
        "clean": manager.clean_docs,
        "validate": manager.validate_docs,
        "update": manager.update_docs,
    }

    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
