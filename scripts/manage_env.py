#!/usr/bin/env python3
"""
Environment management script.
Provides commands for managing environment variables and settings.

This script supports:
- Setting environment variables
- Viewing environment variables
- Loading environment variables from files

Usage:
    python manage_env.py <command> [options]

Commands:
    set         Set an environment variable
    view        View environment variables
    load        Load environment variables from a file

Examples:
    # Set an environment variable
    python manage_env.py set --key API_KEY --value myapikey

    # View environment variables
    python manage_env.py view

    # Load environment variables from a file
    python manage_env.py load --file .env
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

from trading.utils.env_manager import EnvironmentManager

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Environment Manager")
    parser.add_argument(
        "command",
        choices=["create-template", "validate", "rotate-secret", "check-health"],
        help="Command to execute"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file"
    )
    parser.add_argument(
        "--key",
        help="Secret key to rotate"
    )
    parser.add_argument(
        "--value",
        help="New secret value"
    )
    
    args = parser.parse_args()
    manager = EnvironmentManager(args.env_file)
    
    try:
        if args.command == "create-template":
            manager.create_template()
            print("Template created successfully")
            
        elif args.command == "validate":
            if manager.validate():
                print("Environment validation successful")
            else:
                print("Environment validation failed")
                sys.exit(1)
                
        elif args.command == "rotate-secret":
            if not args.key or not args.value:
                print("Error: --key and --value are required for rotate-secret")
                sys.exit(1)
            manager.rotate_secret(args.key, args.value)
            print(f"Secret {args.key} rotated successfully")
            
        elif args.command == "check-health":
            # Check Python version
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
                print(f"Error: Python version {python_version.major}.{python_version.minor} is not supported")
                sys.exit(1)
            
            # Check required directories
            required_dirs = ["logs", "data", "config"]
            for directory in required_dirs:
                if not Path(directory).exists():
                    print(f"Error: Required directory not found: {directory}")
                    sys.exit(1)
            
            print("Environment health check completed successfully")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == "__main__":
    main() 