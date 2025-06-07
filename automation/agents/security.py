import jwt
import bcrypt
import logging
from typing import Dict, Optional, List, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
from functools import wraps
import hashlib
import secrets
import re
import ssl
import certifi
import aiohttp
import ipaddress
import socket
import platform
import psutil
import hmac
import base64
from dataclasses import dataclass

@dataclass
class SecurityEvent:
    id: str
    type: str
    severity: str
    timestamp: str
    source: str
    details: Dict
    resolved: bool = False

class SecuritySystem:
    def __init__(self, config: Dict):
        self.config = config
        self.security_config = config.get('security', {})
        self.setup_logging()
        self.users = self._load_users()
        self.sessions = {}
        self.failed_attempts = {}
        self.blocked_ips = {}
        self.jwt_secret = self.security_config.get('jwt_secret', secrets.token_hex(32))
        self.security_events: List[SecurityEvent] = []
        self.rate_limits = self.security_config.get('rate_limits', {})
        self.ip_whitelist = self.security_config.get('ip_whitelist', [])
        self.ip_blacklist = self.security_config.get('ip_blacklist', [])

    def setup_logging(self):
        """Configure logging for the security system."""
        log_path = Path("automation/logs/security")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "security.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_users(self) -> Dict:
        """Load users from configuration file."""
        users_path = Path("automation/config/users.json")
        if users_path.exists():
            with open(users_path) as f:
                return json.load(f)
        return {}

    def _save_users(self):
        """Save users to configuration file."""
        users_path = Path("automation/config/users.json")
        with open(users_path, 'w') as f:
            json.dump(self.users, f, indent=2)

    def create_user(self, 
                   username: str, 
                   password: str, 
                   roles: List[str],
                   email: Optional[str] = None) -> bool:
        """
        Create a new user.
        
        Args:
            username: User's username
            password: User's password
            roles: List of user roles
            email: User's email (optional)
        
        Returns:
            bool: True if user was created successfully
        """
        if username in self.users:
            return False
        
        # Hash password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode(), salt)
        
        self.users[username] = {
            'password_hash': hashed_password.decode(),
            'roles': roles,
            'email': email,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'failed_attempts': 0
        }
        
        self._save_users()
        return True

    def authenticate(self, username: str, password: str, ip_address: str) -> Optional[str]:
        """
        Authenticate a user.
        
        Args:
            username: User's username
            password: User's password
            ip_address: User's IP address
        
        Returns:
            Optional[str]: JWT token if authentication successful, None otherwise
        """
        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            if datetime.now() < self.blocked_ips[ip_address]['until']:
                self.logger.warning(f"Blocked login attempt from {ip_address}")
                return None
            else:
                del self.blocked_ips[ip_address]
        
        # Check if user exists
        if username not in self.users:
            self._record_failed_attempt(ip_address)
            return None
        
        user = self.users[username]
        
        # Check if account is locked
        if user['failed_attempts'] >= self.security_config.get('max_failed_attempts', 5):
            if datetime.now() < user.get('lock_until', datetime.now()):
                self.logger.warning(f"Account {username} is locked")
                return None
            else:
                user['failed_attempts'] = 0
                user.pop('lock_until', None)
        
        # Verify password
        if not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
            user['failed_attempts'] += 1
            if user['failed_attempts'] >= self.security_config.get('max_failed_attempts', 5):
                user['lock_until'] = datetime.now() + timedelta(minutes=30)
                self.logger.warning(f"Account {username} locked due to too many failed attempts")
            self._save_users()
            self._record_failed_attempt(ip_address)
            return None
        
        # Reset failed attempts
        user['failed_attempts'] = 0
        user['last_login'] = datetime.now().isoformat()
        self._save_users()
        
        # Generate JWT token
        token = self._generate_token(username, user['roles'])
        return token

    def _generate_token(self, username: str, roles: List[str]) -> str:
        """Generate a JWT token."""
        payload = {
            'username': username,
            'roles': roles,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify a JWT token."""
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            self.logger.warning("Expired token")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None

    def _record_failed_attempt(self, ip_address: str):
        """Record a failed login attempt."""
        if ip_address not in self.failed_attempts:
            self.failed_attempts[ip_address] = {
                'count': 0,
                'first_attempt': datetime.now()
            }
        
        self.failed_attempts[ip_address]['count'] += 1
        
        # Check if IP should be blocked
        if self.failed_attempts[ip_address]['count'] >= self.security_config.get('max_failed_attempts', 5):
            block_duration = timedelta(minutes=30)
            self.blocked_ips[ip_address] = {
                'until': datetime.now() + block_duration
            }
            self.logger.warning(f"IP {ip_address} blocked for {block_duration}")

    def has_permission(self, token: str, required_role: str) -> bool:
        """Check if user has required role."""
        payload = self.verify_token(token)
        if not payload:
            return False
        
        return required_role in payload['roles']

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user's password."""
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        # Verify old password
        if not bcrypt.checkpw(old_password.encode(), user['password_hash'].encode()):
            return False
        
        # Hash new password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(new_password.encode(), salt)
        
        user['password_hash'] = hashed_password.decode()
        self._save_users()
        return True

    def reset_password(self, username: str) -> Optional[str]:
        """Reset user's password and return new password."""
        if username not in self.users:
            return None
        
        # Generate new password
        new_password = secrets.token_urlsafe(12)
        
        # Hash new password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(new_password.encode(), salt)
        
        self.users[username]['password_hash'] = hashed_password.decode()
        self._save_users()
        return new_password

    def update_user_roles(self, username: str, roles: List[str]) -> bool:
        """Update user's roles."""
        if username not in self.users:
            return False
        
        self.users[username]['roles'] = roles
        self._save_users()
        return True

    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        if username not in self.users:
            return False
        
        del self.users[username]
        self._save_users()
        return True

    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information."""
        if username not in self.users:
            return None
        
        user = self.users[username].copy()
        user.pop('password_hash', None)
        return user

    def get_all_users(self) -> List[Dict]:
        """Get information for all users."""
        return [
            {
                'username': username,
                'roles': user['roles'],
                'email': user.get('email'),
                'created_at': user['created_at'],
                'last_login': user.get('last_login')
            }
            for username, user in self.users.items()
        ]

    def generate_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Generate a JWT token for authentication."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode(), hashed.encode())

    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)

    def verify_api_key(self, api_key: str, stored_key: str) -> bool:
        """Verify an API key using constant-time comparison."""
        return hmac.compare_digest(api_key.encode(), stored_key.encode())

    def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if a rate limit has been exceeded."""
        now = datetime.now()
        if key not in self.failed_attempts:
            self.failed_attempts[key] = 1
            return True
        
        attempts = self.failed_attempts[key]
        if attempts >= limit:
            return False
        
        self.failed_attempts[key] += 1
        return True

    def is_ip_allowed(self, ip: str) -> bool:
        """Check if an IP address is allowed."""
        # Check if IP is blocked
        if ip in self.blocked_ips:
            if datetime.now() < self.blocked_ips[ip]:
                return False
            else:
                del self.blocked_ips[ip]
        
        # Check whitelist
        if self.ip_whitelist:
            return ip in self.ip_whitelist
        
        # Check blacklist
        if ip in self.ip_blacklist:
            return False
        
        return True

    def block_ip(self, ip: str, duration: int = 3600):
        """Block an IP address for a specified duration."""
        self.blocked_ips[ip] = datetime.now() + timedelta(seconds=duration)
        self.logger.warning(f"Blocked IP address: {ip} for {duration} seconds")

    def record_security_event(
        self,
        type: str,
        severity: str,
        source: str,
        details: Dict
    ):
        """Record a security event."""
        event = SecurityEvent(
            id=str(len(self.security_events) + 1),
            type=type,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            source=source,
            details=details
        )
        
        self.security_events.append(event)
        self.logger.warning(f"Security event recorded: {event.id} - {type}")

    def get_security_events(
        self,
        type: Optional[str] = None,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[SecurityEvent]:
        """Get security events with optional filtering."""
        events = self.security_events
        
        if type:
            events = [e for e in events if e.type == type]
        if severity:
            events = [e for e in events if e.severity == severity]
        if source:
            events = [e for e in events if e.source == source]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            events = events[:limit]
        
        return events

    def mark_event_resolved(self, event_id: str):
        """Mark a security event as resolved."""
        for event in self.security_events:
            if event.id == event_id:
                event.resolved = True
                break

    async def scan_system(self):
        """Perform a security scan of the system."""
        scan_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'network_info': await self._get_network_info(),
            'process_info': self._get_process_info(),
            'security_issues': []
        }
        
        # Check for security issues
        issues = await self._check_security_issues()
        scan_results['security_issues'] = issues
        
        # Record scan results
        self.record_security_event(
            type='system_scan',
            severity='info',
            source='security_system',
            details=scan_results
        )
        
        return scan_results

    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free
            }
        }

    async def _get_network_info(self) -> Dict:
        """Get network information."""
        network_info = {
            'interfaces': {},
            'connections': []
        }
        
        # Get network interfaces
        for interface, addrs in psutil.net_if_addrs().items():
            network_info['interfaces'][interface] = [
                {
                    'family': str(addr.family),
                    'address': addr.address,
                    'netmask': addr.netmask
                }
                for addr in addrs
            ]
        
        # Get network connections
        for conn in psutil.net_connections():
            network_info['connections'].append({
                'local_address': f"{conn.laddr.ip}:{conn.laddr.port}",
                'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                'status': conn.status,
                'pid': conn.pid
            })
        
        return network_info

    def _get_process_info(self) -> List[Dict]:
        """Get information about running processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes

    async def _check_security_issues(self) -> List[Dict]:
        """Check for security issues."""
        issues = []
        
        # Check for open ports
        open_ports = await self._check_open_ports()
        if open_ports:
            issues.append({
                'type': 'open_ports',
                'severity': 'warning',
                'details': open_ports
            })
        
        # Check for suspicious processes
        suspicious_processes = self._check_suspicious_processes()
        if suspicious_processes:
            issues.append({
                'type': 'suspicious_processes',
                'severity': 'warning',
                'details': suspicious_processes
            })
        
        # Check for weak permissions
        weak_permissions = self._check_weak_permissions()
        if weak_permissions:
            issues.append({
                'type': 'weak_permissions',
                'severity': 'warning',
                'details': weak_permissions
            })
        
        return issues

    async def _check_open_ports(self) -> List[Dict]:
        """Check for open ports."""
        open_ports = []
        common_ports = [21, 22, 23, 25, 53, 80, 443, 3306, 5432, 27017]
        
        for port in common_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    open_ports.append({
                        'port': port,
                        'service': self._get_service_name(port)
                    })
                sock.close()
            except:
                pass
        
        return open_ports

    def _get_service_name(self, port: int) -> str:
        """Get service name for a port."""
        services = {
            21: 'FTP',
            22: 'SSH',
            23: 'Telnet',
            25: 'SMTP',
            53: 'DNS',
            80: 'HTTP',
            443: 'HTTPS',
            3306: 'MySQL',
            5432: 'PostgreSQL',
            27017: 'MongoDB'
        }
        return services.get(port, 'Unknown')

    def _check_suspicious_processes(self) -> List[Dict]:
        """Check for suspicious processes."""
        suspicious = []
        suspicious_patterns = [
            r'nc\.exe',
            r'nmap',
            r'wireshark',
            r'netcat',
            r'backdoor',
            r'rootkit'
        ]
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for pattern in suspicious_patterns:
                    if re.search(pattern, proc.info['name'], re.I):
                        suspicious.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': proc.info['cmdline']
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return suspicious

    def _check_weak_permissions(self) -> List[Dict]:
        """Check for weak file permissions."""
        weak_permissions = []
        sensitive_paths = [
            'automation/config',
            'automation/logs',
            'automation/agents'
        ]
        
        for path in sensitive_paths:
            try:
                stat = Path(path).stat()
                if stat.st_mode & 0o777 > 0o750:  # More permissive than 750
                    weak_permissions.append({
                        'path': path,
                        'mode': oct(stat.st_mode & 0o777)
                    })
            except:
                pass
        
        return weak_permissions

    def clear_security_events(self, before: Optional[datetime] = None):
        """Clear old security events."""
        if before:
            self.security_events = [
                e for e in self.security_events
                if datetime.fromisoformat(e.timestamp) > before
            ]
        else:
            self.security_events = []
        
        self.logger.info(f"Cleared security events before {before}")

def require_auth(required_role: Optional[str] = None):
    """Decorator for requiring authentication."""
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            # Get token from request
            token = kwargs.get('token')
            if not token:
                return {'error': 'No token provided'}, 401
            
            # Verify token
            security_system = kwargs.get('security_system')
            if not security_system:
                return {'error': 'Security system not available'}, 500
            
            payload = security_system.verify_token(token)
            if not payload:
                return {'error': 'Invalid token'}, 401
            
            # Check role if required
            if required_role and not security_system.has_permission(token, required_role):
                return {'error': 'Insufficient permissions'}, 403
            
            # Add user info to kwargs
            kwargs['user'] = payload
            return await f(*args, **kwargs)
        
        return decorated_function
    return decorator 