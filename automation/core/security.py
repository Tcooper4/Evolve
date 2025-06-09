"""
Security Module

This module provides security-related functionality for the automation system,
including authentication, authorization, and rate limiting.
"""

import os
import jwt
import time
import logging
from typing import Optional, Dict, Any, List
from functools import wraps
from datetime import datetime, timedelta
import redis
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

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

class TokenData(BaseModel):
    """Token data model."""
    user_id: str
    role: str
    permissions: List[str]

class SecurityManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            db=0,
            ssl=os.getenv("REDIS_SSL", "true").lower() == "true"
        )
        self.security = HTTPBearer()
    
    def create_access_token(self, user_id: str, role: str, permissions: List[str]) -> str:
        """Create a new JWT access token."""
        expiration = datetime.utcnow() + timedelta(seconds=SecurityConfig.JWT_EXPIRATION)
        token_data = {
            "sub": user_id,
            "role": role,
            "permissions": permissions,
            "exp": expiration
        }
        return jwt.encode(token_data, SecurityConfig.JWT_SECRET, algorithm=SecurityConfig.JWT_ALGORITHM)
    
    def verify_token(self, token: str) -> TokenData:
        """Verify a JWT token."""
        try:
            payload = jwt.decode(token, SecurityConfig.JWT_SECRET, algorithms=[SecurityConfig.JWT_ALGORITHM])
            return TokenData(
                user_id=payload["sub"],
                role=payload["role"],
                permissions=payload["permissions"]
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
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
                return await func(*args, current_user=current_user, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, role: str):
        """Decorator to require a specific role."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, current_user: TokenData = Depends(get_current_user), **kwargs):
                if current_user.role != role:
                    raise HTTPException(status_code=403, detail="Role not authorized")
                return await func(*args, current_user=current_user, **kwargs)
            return wrapper
        return decorator
    
    def rate_limit(self):
        """Decorator to implement rate limiting."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, current_user: TokenData = Depends(get_current_user), **kwargs):
                if not self.check_rate_limit(current_user.user_id):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                return await func(*args, current_user=current_user, **kwargs)
            return wrapper
        return decorator

# Create a global security manager instance
security_manager = SecurityManager()

# Export commonly used decorators
require_permission = security_manager.require_permission
require_role = security_manager.require_role
rate_limit = security_manager.rate_limit
get_current_user = security_manager.get_current_user 