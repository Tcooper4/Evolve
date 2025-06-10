#!/usr/bin/env python3
"""
Environment management script.
Provides commands for managing the application's environment variables and configuration.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class EnvironmentManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the environment manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.env_file = Path(".env")

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

    def create_env_file(self, template_path: Optional[str] = None):
        """Create .env file from template or default values."""
        self.logger.info("Creating .env file...")
        
        if template_path and Path(template_path).exists():
            # Copy from template
            with open(template_path) as f:
                env_content = f.read()
        else:
            # Create default .env file
            env_content = """# Application Settings
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Server Settings
HOST=0.0.0.0
PORT=8501

# Database Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false

# API Keys (replace with your keys)
ALPHA_VANTAGE_API_KEY=
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Security Settings
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Monitoring Settings
ENABLE_MONITORING=true
ALERT_EMAIL=alerts@example.com
SLACK_WEBHOOK_URL=

# Feature Flags
ENABLE_NLP=true
ENABLE_FORECASTING=true
ENABLE_TRADING=true
"""
        
        try:
            with open(self.env_file, "w") as f:
                f.write(env_content)
            self.logger.info(".env file created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create .env file: {e}")
            return False

    def update_env_file(self, updates: Dict[str, str]):
        """Update values in .env file."""
        self.logger.info("Updating .env file...")
        
        if not self.env_file.exists():
            self.logger.error(".env file not found")
            return False
        
        try:
            # Read current .env file
            with open(self.env_file) as f:
                lines = f.readlines()
            
            # Update values
            for i, line in enumerate(lines):
                if "=" in line:
                    key = line.split("=")[0].strip()
                    if key in updates:
                        lines[i] = f"{key}={updates[key]}\n"
            
            # Write updated content
            with open(self.env_file, "w") as f:
                f.writelines(lines)
            
            self.logger.info(".env file updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update .env file: {e}")
            return False

    def validate_env_file(self):
        """Validate .env file values."""
        self.logger.info("Validating .env file...")
        
        if not self.env_file.exists():
            self.logger.error(".env file not found")
            return False
        
        try:
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check required variables
            required_vars = [
                "APP_ENV",
                "DEBUG",
                "LOG_LEVEL",
                "HOST",
                "PORT",
                "REDIS_HOST",
                "REDIS_PORT"
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                self.logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                return False
            
            self.logger.info("Environment variables validated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False

    def export_env_vars(self):
        """Export environment variables to shell."""
        self.logger.info("Exporting environment variables...")
        
        if not self.env_file.exists():
            self.logger.error(".env file not found")
            return False
        
        try:
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            # Export variables
            for key, value in os.environ.items():
                if key.startswith(("APP_", "REDIS_", "API_", "SECRET_")):
                    print(f"export {key}={value}")
            
            self.logger.info("Environment variables exported successfully")
            return True
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False

    def check_env_health(self):
        """Check environment health."""
        self.logger.info("Checking environment health...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
                self.logger.error(f"Python version {python_version.major}.{python_version.minor} is not supported")
                return False
            
            # Check required directories
            required_dirs = ["logs", "data", "config"]
            for directory in required_dirs:
                if not Path(directory).exists():
                    self.logger.error(f"Required directory not found: {directory}")
                    return False
            
            # Check required files
            required_files = [
                "config/app_config.yaml",
                "config/logging_config.yaml",
                ".env"
            ]
            for file in required_files:
                if not Path(file).exists():
                    self.logger.error(f"Required file not found: {file}")
                    return False
            
            self.logger.info("Environment health check completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Environment Manager")
    parser.add_argument(
        "command",
        choices=["create", "update", "validate", "export", "health"],
        help="Command to execute"
    )
    parser.add_argument(
        "--template",
        help="Path to .env template file"
    )
    parser.add_argument(
        "--updates",
        help="JSON string of environment variable updates"
    )
    
    args = parser.parse_args()
    manager = EnvironmentManager()
    
    commands = {
        "create": lambda: manager.create_env_file(args.template),
        "update": lambda: manager.update_env_file(json.loads(args.updates)) if args.updates else False,
        "validate": manager.validate_env_file,
        "export": manager.export_env_vars,
        "health": manager.check_env_health
    }
    
    if args.command in commands:
        if args.command == "update" and not args.updates:
            parser.error("update requires --updates")
        
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 