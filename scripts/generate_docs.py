#!/usr/bin/env python3
"""
Documentation generation script.
Provides commands for generating and exporting project documentation.

This script supports:
- Generating documentation from source code
- Exporting documentation in various formats

Usage:
    python generate_docs.py <command> [options]

Commands:
    generate    Generate documentation
    export      Export documentation

Examples:
    # Generate documentation
    python generate_docs.py generate

    # Export documentation
    python generate_docs.py export --format pdf --output docs/project_docs.pdf
"""

import os
import sys
import logging
import logging.config
import yaml
import subprocess
from pathlib import Path
from typing import List, Optional

class DocumentationGenerator:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the documentation generator."""
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

    def generate_api_docs(self):
        """Generate API documentation using pdoc."""
        self.logger.info("Generating API documentation...")
        
        if self.run_command([
            "pdoc",
            "--html",
            "--output-dir", "docs/api",
            "trading"
        ]) != 0:
            self.logger.error("Failed to generate API documentation")
            return False
        
        self.logger.info("API documentation generated successfully")
        return True

    def generate_user_guide(self):
        """Generate user guide using MkDocs."""
        self.logger.info("Generating user guide...")
        
        # Build MkDocs site
        if self.run_command(["mkdocs", "build"]) != 0:
            self.logger.error("Failed to build MkDocs site")
            return False
        
        self.logger.info("User guide generated successfully")
        return True

    def generate_examples(self):
        """Generate example notebooks."""
        self.logger.info("Generating example notebooks...")
        
        examples_dir = Path("docs/examples")
        examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate example notebooks
        examples = [
            "basic_usage.ipynb",
            "advanced_features.ipynb",
            "custom_strategies.ipynb",
            "risk_management.ipynb",
            "portfolio_optimization.ipynb"
        ]
        
        for example in examples:
            if self.run_command([
                "jupyter",
                "nbconvert",
                "--to", "notebook",
                "--execute",
                f"docs/examples/{example}"
            ]) != 0:
                self.logger.error(f"Failed to generate example: {example}")
                return False
        
        self.logger.info("Example notebooks generated successfully")
        return True

    def generate_diagrams(self):
        """Generate system architecture diagrams."""
        self.logger.info("Generating system diagrams...")
        
        diagrams_dir = Path("docs/diagrams")
        diagrams_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate diagrams using Graphviz
        diagrams = [
            "system_architecture.dot",
            "data_flow.dot",
            "component_interaction.dot"
        ]
        
        for diagram in diagrams:
            if self.run_command([
                "dot",
                "-Tpng",
                f"docs/diagrams/{diagram}",
                "-o", f"docs/diagrams/{diagram.replace('.dot', '.png')}"
            ]) != 0:
                self.logger.error(f"Failed to generate diagram: {diagram}")
                return False
        
        self.logger.info("System diagrams generated successfully")
        return True

    def generate_all(self):
        """Generate all documentation."""
        self.logger.info("Generating all documentation...")
        
        # Create docs directory
        Path("docs").mkdir(exist_ok=True)
        
        # Generate documentation
        if not all([
            self.generate_api_docs(),
            self.generate_user_guide(),
            self.generate_examples(),
            self.generate_diagrams()
        ]):
            self.logger.error("Documentation generation failed")
            return False
        
        self.logger.info("All documentation generated successfully")
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Documentation Generator")
    parser.add_argument(
        "command",
        choices=["api", "guide", "examples", "diagrams", "all"],
        help="Documentation to generate"
    )
    
    args = parser.parse_args()
    generator = DocumentationGenerator()
    
    commands = {
        "api": generator.generate_api_docs,
        "guide": generator.generate_user_guide,
        "examples": generator.generate_examples,
        "diagrams": generator.generate_diagrams,
        "all": generator.generate_all
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 