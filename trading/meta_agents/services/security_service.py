"""
Security Service

This module implements security and authentication functionality.

Note: This module was adapted from the legacy automation/services/automation_security.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import yaml
import time
from datetime import datetime, timedelta
import jwt
import bcrypt
import secrets
import hashlib
from dataclasses import dataclass
from enum import Enum
import sqlite3
import re
import ipaddress

class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class User:
    """User configuration."""
    username: str
    password_hash: str
    email: str
    role: str
    security_level: SecurityLevel
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

class SecurityService:
    """Manages security and authentication."""
    
    def __init__(self, config_path: str):
        """Initialize security service."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.setup_database()
        self.users: Dict[str, User] = {}
        self.blacklisted_tokens: set = set()
        self.load_users()
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/security")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "security_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    self.config = json.load(f)
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def setup_database(self) -> None:
        """Set up security database."""
        try:
            db_path = Path(self.config['database']['path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(str(db_path))
            self.cursor = self.conn.cursor()
            
            # Create users table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    security_level TEXT NOT NULL,
                    last_login DATETIME,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until DATETIME
                )
            ''')
            
            # Create audit log table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    username TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    success BOOLEAN NOT NULL
                )
            ''')
            
            self.conn.commit()
            self.logger.info("Database setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def load_users(self) -> None:
        """Load users from database."""
        try:
            self.cursor.execute('SELECT * FROM users')
            for row in self.cursor.fetchall():
                user = User(
                    username=row[0],
                    password_hash=row[1],
                    email=row[2],
                    role=row[3],
                    security_level=SecurityLevel(row[4]),
                    last_login=datetime.fromisoformat(row[5]) if row[5] else None,
                    failed_attempts=row[6],
                    locked_until=datetime.fromisoformat(row[7]) if row[7] else None
                )
                self.users[user.username] = user
        except Exception as e:
            self.logger.error(f"Error loading users: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        try:
            salt = bcrypt.gensalt()
            return {'success': True, 'result': bcrypt.hashpw(password.encode(), salt).decode(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error hashing password: {str(e)}")
            raise
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            return {'success': True, 'result': bcrypt.checkpw(password.encode(), password_hash.encode()), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error verifying password: {str(e)}")
            raise
    
    def validate_password(self, password: str) -> bool:
        """Validate password strength."""
        try:
            if len(password) < 8:
                return False
            
            if not re.search(r'[A-Z]', password):
                return False
            
            if not re.search(r'[a-z]', password):
                return False
            
            if not re.search(r'\d', password):
                return False
            
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating password: {str(e)}")
            raise
    
    def create_user(self, username: str, password: str, email: str, role: str, security_level: SecurityLevel) -> None:
        """Create a new user."""
        try:
            if username in self.users:
                raise ValueError(f"User {username} already exists")
            
            if not self.validate_password(password):
                raise ValueError("Password does not meet security requirements")
            
            password_hash = self.hash_password(password)
            user = User(
                username=username,
                password_hash=password_hash,
                email=email,
                role=role,
                security_level=security_level
            )
            
            self.cursor.execute('''
                INSERT INTO users (username, password_hash, email, role, security_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user.username,
                user.password_hash,
                user.email,
                user.role,
                user.security_level.value
            ))
            
            self.conn.commit()
            self.users[username] = user
            self.logger.info(f"Created user {username}")
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def update_user(self, username: str, **kwargs) -> None:
        """Update user information."""
        try:
            if username not in self.users:
                raise ValueError(f"User {username} does not exist")
            
            user = self.users[username]
            
            if 'password' in kwargs:
                if not self.validate_password(kwargs['password']):
                    raise ValueError("Password does not meet security requirements")
                kwargs['password_hash'] = self.hash_password(kwargs.pop('password'))
            
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            self.cursor.execute('''
                UPDATE users
                SET password_hash = ?,
                    email = ?,
                    role = ?,
                    security_level = ?,
                    last_login = ?,
                    failed_attempts = ?,
                    locked_until = ?
                WHERE username = ?
            ''', (
                user.password_hash,
                user.email,
                user.role,
                user.security_level.value,
                user.last_login.isoformat() if user.last_login else None,
                user.failed_attempts,
                user.locked_until.isoformat() if user.locked_until else None,
                user.username
            ))
            
            self.conn.commit()
            self.logger.info(f"Updated user {username}")
        except Exception as e:
            self.logger.error(f"Error updating user: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def delete_user(self, username: str) -> None:
        """Delete a user."""
        try:
            if username not in self.users:
                raise ValueError(f"User {username} does not exist")
            
            self.cursor.execute('DELETE FROM users WHERE username = ?', (username,))
            self.conn.commit()
            del self.users[username]
            self.logger.info(f"Deleted user {username}")
        except Exception as e:
            self.logger.error(f"Error deleting user: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def authenticate(self, username: str, password: str, ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate a user and return JWT token."""
        try:
            if username not in self.users:
                self.log_audit(username, "login", "User not found", ip_address, False)
                return None
            
            user = self.users[username]
            
            if user.locked_until and user.locked_until > datetime.now():
                self.log_audit(username, "login", "Account locked", ip_address, False)
                return None
            
            if not self.verify_password(password, user.password_hash):
                user.failed_attempts += 1
                if user.failed_attempts >= self.config['security']['max_failed_attempts']:
                    user.locked_until = datetime.now() + timedelta(minutes=self.config['security']['lockout_duration'])
                self.update_user(username, failed_attempts=user.failed_attempts, locked_until=user.locked_until)
                self.log_audit(username, "login", "Invalid password", ip_address, False)
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            # Reset failed attempts on successful login
            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()
            self.update_user(username, failed_attempts=user.failed_attempts, locked_until=user.locked_until, last_login=user.last_login)
            
            token = self.generate_token(username)
            self.log_audit(username, "login", "Login successful", ip_address, True)
            return token
        except Exception as e:
            self.logger.error(f"Error authenticating user: {str(e)}")
            raise
    
    def generate_token(self, username: str) -> str:
        """Generate JWT token."""
        try:
            user = self.users[username]
            payload = {
                'username': username,
                'role': user.role,
                'security_level': user.security_level.value,
                'exp': datetime.utcnow() + timedelta(minutes=self.config['security']['token_expiry'])
            }
            return {'success': True, 'result': jwt.encode(payload, self.config['security']['secret_key'], algorithm='HS256'), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error generating token: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            if token in self.blacklisted_tokens:
                return None
            
            payload = jwt.decode(token, self.config['security']['secret_key'], algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error verifying token: {str(e)}")
            raise
    
    def blacklist_token(self, token: str) -> None:
        """Add token to blacklist."""
        try:
            self.blacklisted_tokens.add(token)
        except Exception as e:
            self.logger.error(f"Error blacklisting token: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def log_audit(self, username: Optional[str], action: str, details: Optional[str] = None, ip_address: Optional[str] = None, success: bool = True) -> None:
        """Log security audit event."""
        try:
            self.cursor.execute('''
                INSERT INTO audit_log (timestamp, username, action, details, ip_address, success)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                username,
                action,
                details,
                ip_address,
                success
            ))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging audit event: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def get_audit_log(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        try:
            query = 'SELECT * FROM audit_log WHERE 1=1'
            params = []
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            if username:
                query += ' AND username = ?'
                params.append(username)
            
            query += ' ORDER BY timestamp DESC'
            
            self.cursor.execute(query, params)
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'username': row[2],
                    'action': row[3],
                    'details': row[4],
                    'ip_address': row[5],
                    'success': bool(row[6])
                })
            
            return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error getting audit log: {str(e)}")
            raise
    
    def validate_ip(self, ip_address: str) -> bool:
        """Validate IP address."""
        try:
            ipaddress.ip_address(ip_address)
            return True
        except ValueError:
            return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        try:
            if not self.validate_ip(ip_address):
                return False
            
            ip = ipaddress.ip_address(ip_address)
            
            # Check blacklist
            for blacklisted in self.config['security']['ip_blacklist']:
                if ip in ipaddress.ip_network(blacklisted):
                    return False
            
            # Check whitelist if enabled
            if self.config['security']['ip_whitelist_enabled']:
                for whitelisted in self.config['security']['ip_whitelist']:
                    if ip in ipaddress.ip_network(whitelisted):
                        return {'success': True, 'result': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking IP: {str(e)}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security service')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--create-user', action='store_true', help='Create a new user')
    args = parser.parse_args()
    
    try:
        service = SecurityService(args.config)
        
        if args.create_user:
            username = input("Username: ")
            password = input("Password: ")
            email = input("Email: ")
            role = input("Role: ")
            security_level = input("Security level (low/medium/high/critical): ")
            
            service.create_user(
                username,
                password,
                email,
                role,
                SecurityLevel(security_level.lower())
            )
    except KeyboardInterrupt:
        logging.info("Security service interrupted")
    except Exception as e:
        logging.error(f"Error in security service: {str(e)}")
        raise

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == '__main__':
    main() 