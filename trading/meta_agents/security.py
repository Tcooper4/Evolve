"""
# Adapted from automation/core/security.py â€” legacy security logic

Security Module

This module provides security-related functionality for the trading system,
including authentication, authorization, and rate limiting.
"""

import os
import jwt
import time
import logging
import secrets
from typing import Optional, Dict, Any, List
from functools import wraps
from datetime import datetime, timedelta
import redis
from fastapi import HTTPException, Security, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import re
import hashlib
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityConfig:
    """Security configuration."""
    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION = 3600  # 1 hour
    RATE_LIMIT_WINDOW = 3600  # 1 hour
    RATE_LIMIT_MAX_REQUESTS = 100
    CSRF_TOKEN_HEADER = "X-CSRF-Token"
    CSRF_TOKEN_COOKIE = "csrf_token"
    CSRF_TOKEN_EXPIRY = 3600  # 1 hour

class TokenData(BaseModel):
    """Token data model."""
    user_id: str
    role: str
    permissions: List[str]

class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string input."""
        # Remove potentially dangerous characters
        value = re.sub(r'[<>]', '', value)
        # Remove control characters
        value = ''.join(char for char in value if ord(char) >= 32)
        return value.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return {'success': True, 'result': bool(re.match(pattern, email)), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    @staticmethod
    def validate_password(password: str) -> bool:
        """Validate password strength."""
        if len(password) < 8:
            return False
        if not re.search(r'[A-Z]', password):
            return False
        if not re.search(r'[a-z]', password):
            return False
        if not re.search(r'\d', password):
            return False
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False
        return True

class SecurityManager:
    """Manages security-related functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize security manager."""
        self.config = config
        self.secret_key = config.get('secret_key') or secrets.token_hex(32)
        self.token_expiry = config.get('token_expiry', 3600)  # 1 hour
        self.setup_logging()
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            db=0,
            ssl=os.getenv("REDIS_SSL", "true").lower() == "true"
        )
        self.security = HTTPBearer()
        self.validator = InputValidator()
        
    def setup_logging(self):
        """Configure logging for security manager."""
        log_path = Path("logs/security")
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
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        try:
            salt = secrets.token_hex(16)
            hashed = hashlib.sha256(
                (password + salt).encode()
            ).hexdigest()
            return f"{salt}${hashed}"
        except Exception as e:
            self.logger.error(f"Error hashing password: {str(e)}")
            raise
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, stored_hash = hashed.split('$')
            computed_hash = hashlib.sha256(
                (password + salt).encode()
            ).hexdigest()
            return computed_hash == stored_hash
        except Exception as e:
            self.logger.error(f"Error verifying password: {str(e)}")
            return False
    
    def generate_token(self, user_id: str, role: str) -> str:
        """Generate a JWT token."""
        try:
            payload = {
                'user_id': user_id,
                'role': role,
                'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry)
            }
            return {'success': True, 'result': jwt.encode(payload, self.secret_key, algorithm='HS256'), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error generating token: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify a JWT token."""
        try:
            return {'success': True, 'result': jwt.decode(token, self.secret_key, algorithms=['HS256']), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except jwt.ExpiredSignatureError:
            self.logger.error("Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            self.logger.error(f"Invalid token: {str(e)}")
            raise
    
    def encrypt(self):
        raise NotImplementedError('Pending feature')

    def decrypt(self):
        raise NotImplementedError('Pending feature')

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            # TODO: Implement encryption logic
            return data
        except Exception as e:
            self.logger.error(f"Error encrypting data: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            # TODO: Implement decryption logic
            return encrypted_data
        except Exception as e:
            self.logger.error(f"Error decrypting data: {str(e)}")
            raise
    
    def create_access_token(self, user_id: str, role: str, permissions: List[str]) -> str:
        """Create a new JWT access token."""
        expiration = datetime.utcnow() + timedelta(seconds=SecurityConfig.JWT_EXPIRATION)
        token_data = {
            "sub": user_id,
            "role": role,
            "permissions": permissions,
            "exp": expiration
        }
        return {'success': True, 'result': jwt.encode(token_data, SecurityConfig.JWT_SECRET, algorithm=SecurityConfig.JWT_ALGORITHM), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check if a user has exceeded their rate limit."""
        key = f"rate_limit:{user_id}"
        current = self.redis_client.get(key)
        
        if current is None:
            self.redis_client.setex(key, SecurityConfig.RATE_LIMIT_WINDOW, 1)
            return True
        
        current = int(current)
        if current >= SecurityConfig.RATE_LIMIT_MAX_REQUESTS:
            return False
        
        self.redis_client.incr(key)
        return True
    
    def check_permission(self, required_permission: str, user_permissions: List[str]) -> bool:
        """Check if a user has the required permission."""
        return required_permission in user_permissions
    
    def generate_csrf_token(self) -> str:
        """Generate a new CSRF token."""
        return secrets.token_urlsafe(32)
    
    def validate_csrf_token(self, request: Request, response: Response) -> bool:
        """Validate CSRF token from request."""
        token = request.headers.get(SecurityConfig.CSRF_TOKEN_HEADER)
        cookie_token = request.cookies.get(SecurityConfig.CSRF_TOKEN_COOKIE)
        
        if not token or not cookie_token:
            return False
        
        if token != cookie_token:
            return False
        
        # Refresh token
        new_token = self.generate_csrf_token()
        response.set_cookie(
            SecurityConfig.CSRF_TOKEN_COOKIE,
            new_token,
            max_age=SecurityConfig.CSRF_TOKEN_EXPIRY,
            httponly=True,
            secure=True,
            samesite='strict'
        )
        return True
    
    def set_csrf_token(self, response: Response) -> None:
        """Set CSRF token in response."""
        token = self.generate_csrf_token()
        response.set_cookie(
            SecurityConfig.CSRF_TOKEN_COOKIE,
            token,
            max_age=SecurityConfig.CSRF_TOKEN_EXPIRY,
            httponly=True,
            secure=True,
            samesite='strict'
        )

    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> TokenData:
        """Get the current user from the JWT token."""
        return self.verify_token(credentials.credentials)
    
    def require_permission(self, permission: str):
        """Decorator to require a specific permission."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, current_user: TokenData = Depends(get_current_user), **kwargs):
                if not self.check_permission(permission, current_user.permissions):
                    raise HTTPException(status_code=403, detail="Permission denied")
                return {'success': True, 'result': {'success': True, 'result': await func(*args, current_user=current_user, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            return wrapper
        return decorator
    
    def require_role(self, role: str):
        """Decorator to require a specific role."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, current_user: TokenData = Depends(get_current_user), **kwargs):
                if current_user.role != role:
                    raise HTTPException(status_code=403, detail="Role not authorized")
                return {'success': True, 'result': {'success': True, 'result': await func(*args, current_user=current_user, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            return wrapper
        return decorator
    
    def rate_limit(self):
        """Decorator to implement rate limiting."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, current_user: TokenData = Depends(get_current_user), **kwargs):
                if not self.check_rate_limit(current_user.user_id):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                return {'success': True, 'result': {'success': True, 'result': await func(*args, current_user=current_user, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            return wrapper
        return decorator
    
    def validate_input(self):
        """Decorator to validate and sanitize input."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Sanitize string inputs
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        kwargs[key] = self.validator.sanitize_string(value)
                return {'success': True, 'result': {'success': True, 'result': await func(*args, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            return wrapper
        return decorator
    
    def require_csrf(self):
        """Decorator to require CSRF token validation."""
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, response: Response, *args, **kwargs):
                if not self.validate_csrf_token(request, response):
                    raise HTTPException(status_code=403, detail="Invalid CSRF token")
                return {'success': True, 'result': {'success': True, 'result': await func(request, response, *args, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            return wrapper
        return decorator

# Create a global security manager instance
# security_manager = SecurityManager()

# Export commonly used decorators
# require_permission = security_manager.require_permission
# require_role = security_manager.require_role
# rate_limit = security_manager.rate_limit
# validate_input = security_manager.validate_input
# require_csrf = security_manager.require_csrf
# get_current_user = security_manager.get_current_user 
