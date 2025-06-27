#!/usr/bin/env python3
"""
Security management script.
Provides commands for managing application security, including user management, permissions, and security audits.

This script supports:
- Managing users and permissions
- Running security audits
- Exporting security reports

Usage:
    python manage_security.py <command> [options]

Commands:
    user        Manage users
    audit       Run security audit
    report      Export security report

Examples:
    # Manage users
    python manage_security.py user --action add --username alice

    # Run security audit
    python manage_security.py audit

    # Export security report
    python manage_security.py report --output security_report.json
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import secrets
import string
import hashlib
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurityManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the security manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.keys_dir = Path("config/keys")
        self.keys_dir.mkdir(parents=True, exist_ok=True)

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

    def generate_key(self, key_type: str, length: int = 32) -> str:
        """Generate a secure key."""
        if key_type == "api":
            # Generate API key
            alphabet = string.ascii_letters + string.digits
            return ''.join(secrets.choice(alphabet) for _ in range(length))
        elif key_type == "jwt":
            # Generate JWT secret
            return secrets.token_hex(length)
        elif key_type == "encryption":
            # Generate encryption key
            return Fernet.generate_key().decode()
        else:
            raise ValueError(f"Unknown key type: {key_type}")

    def rotate_keys(self, key_type: Optional[str] = None):
        """Rotate security keys."""
        self.logger.info("Rotating security keys...")
        
        key_types = [key_type] if key_type else ["api", "jwt", "encryption"]
        rotated_keys = {}
        
        try:
            for kt in key_types:
                # Generate new key
                new_key = self.generate_key(kt)
                
                # Save old key to backup
                key_file = self.keys_dir / f"{kt}_key.txt"
                if key_file.exists():
                    backup_file = self.keys_dir / f"{kt}_key.backup"
                    key_file.rename(backup_file)
                
                # Save new key
                with open(key_file, "w") as f:
                    f.write(new_key)
                
                rotated_keys[kt] = new_key
                self.logger.info(f"Rotated {kt} key")
            
            return rotated_keys
        except Exception as e:
            self.logger.error(f"Failed to rotate keys: {e}")
            return None

    def check_security(self):
        """Check security configuration."""
        self.logger.info("Checking security configuration...")
        
        checks = {
            "keys": self._check_keys(),
            "permissions": self._check_permissions(),
            "config": self._check_config(),
            "dependencies": self._check_dependencies()
        }
        
        # Print results
        print("\nSecurity Check Results:")
        for check, result in checks.items():
            status = "✓" if result["status"] else "✗"
            print(f"\n{status} {check.upper()}")
            if result["issues"]:
                print("  Issues:")
                for issue in result["issues"]:
                    print(f"    - {issue}")
            if result["recommendations"]:
                print("  Recommendations:")
                for rec in result["recommendations"]:
                    print(f"    - {rec}")
        
        return all(check["status"] for check in checks.values())

    def _check_keys(self) -> Dict[str, Any]:
        """Check security keys."""
        issues = []
        recommendations = []
        
        # Check if keys exist
        required_keys = ["api", "jwt", "encryption"]
        for key in required_keys:
            key_file = self.keys_dir / f"{key}_key.txt"
            if not key_file.exists():
                issues.append(f"Missing {key} key")
                recommendations.append(f"Generate {key} key using: python scripts/manage_security.py rotate --key-type {key}")
        
        # Check key permissions
        for key_file in self.keys_dir.glob("*_key.txt"):
            if key_file.stat().st_mode & 0o777 != 0o600:
                issues.append(f"Insecure permissions on {key_file.name}")
                recommendations.append(f"Fix permissions: chmod 600 {key_file}")
        
        return {
            "status": not issues,
            "issues": issues,
            "recommendations": recommendations
        }

    def _check_permissions(self) -> Dict[str, Any]:
        """Check file permissions."""
        issues = []
        recommendations = []
        
        # Check sensitive directories
        sensitive_dirs = ["config", "logs", "data"]
        for dir_name in sensitive_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                if dir_path.stat().st_mode & 0o777 != 0o700:
                    issues.append(f"Insecure permissions on {dir_name} directory")
                    recommendations.append(f"Fix permissions: chmod 700 {dir_name}")
        
        return {
            "status": not issues,
            "issues": issues,
            "recommendations": recommendations
        }

    def _check_config(self) -> Dict[str, Any]:
        """Check security configuration."""
        issues = []
        recommendations = []
        
        # Check SSL configuration
        if not self.config.get("server", {}).get("ssl", {}).get("enabled", False):
            issues.append("SSL not enabled")
            recommendations.append("Enable SSL in app_config.yaml")
        
        # Check password policy
        if not self.config.get("security", {}).get("password_policy", {}).get("enabled", False):
            issues.append("Password policy not enabled")
            recommendations.append("Enable password policy in app_config.yaml")
        
        return {
            "status": not issues,
            "issues": issues,
            "recommendations": recommendations
        }

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check security dependencies."""
        issues = []
        recommendations = []
        
        # Check for known vulnerable packages
        try:
            from importlib.metadata import distributions, version
            for dist in distributions():
                package_name = dist.metadata['Name']
                if package_name in ["cryptography", "pyjwt"]:
                    current_version = version(package_name)
                    # Add version checking logic here if needed
                    pass
        except Exception as e:
            self.logger.warning(f"Could not check dependencies: {e}")
        
        return {
            "status": not issues,
            "issues": issues,
            "recommendations": recommendations
        }

    def encrypt_file(self, file_path: str, output_path: Optional[str] = None):
        """Encrypt a file."""
        self.logger.info(f"Encrypting file: {file_path}")
        
        try:
            # Load encryption key
            key_file = self.keys_dir / "encryption_key.txt"
            if not key_file.exists():
                self.logger.error("Encryption key not found")
                return False
            
            with open(key_file) as f:
                key = f.read().encode()
            
            # Initialize Fernet
            f = Fernet(key)
            
            # Read and encrypt file
            with open(file_path, "rb") as file:
                data = file.read()
            
            encrypted_data = f.encrypt(data)
            
            # Write encrypted data
            if output_path:
                output_file = Path(output_path)
            else:
                output_file = Path(file_path).with_suffix(".enc")
            
            with open(output_file, "wb") as file:
                file.write(encrypted_data)
            
            self.logger.info(f"File encrypted: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to encrypt file: {e}")
            return False

    def decrypt_file(self, file_path: str, output_path: Optional[str] = None):
        """Decrypt a file."""
        self.logger.info(f"Decrypting file: {file_path}")
        
        try:
            # Load encryption key
            key_file = self.keys_dir / "encryption_key.txt"
            if not key_file.exists():
                self.logger.error("Encryption key not found")
                return False
            
            with open(key_file) as f:
                key = f.read().encode()
            
            # Initialize Fernet
            f = Fernet(key)
            
            # Read and decrypt file
            with open(file_path, "rb") as file:
                encrypted_data = file.read()
            
            decrypted_data = f.decrypt(encrypted_data)
            
            # Write decrypted data
            if output_path:
                output_file = Path(output_path)
            else:
                output_file = Path(file_path).with_suffix(".dec")
            
            with open(output_file, "wb") as file:
                file.write(decrypted_data)
            
            self.logger.info(f"File decrypted: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to decrypt file: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Security Manager")
    parser.add_argument(
        "command",
        choices=["rotate", "check", "encrypt", "decrypt"],
        help="Command to execute"
    )
    parser.add_argument(
        "--key-type",
        choices=["api", "jwt", "encryption"],
        help="Type of key to rotate"
    )
    parser.add_argument(
        "--file",
        help="File to encrypt/decrypt"
    )
    parser.add_argument(
        "--output",
        help="Output file path for encryption/decryption"
    )
    
    args = parser.parse_args()
    manager = SecurityManager()
    
    commands = {
        "rotate": lambda: manager.rotate_keys(args.key_type),
        "check": lambda: manager.check_security(),
        "encrypt": lambda: manager.encrypt_file(args.file, args.output) if args.file else False,
        "decrypt": lambda: manager.decrypt_file(args.file, args.output) if args.file else False
    }
    
    if args.command in commands:
        if args.command in ["encrypt", "decrypt"] and not args.file:
            parser.error(f"{args.command} requires --file")
        
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 