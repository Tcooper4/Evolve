import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import redis
import json
import logging
from pydantic import BaseModel, EmailStr
import uuid
from dataclasses import field
import hmac
import hashlib

logger = logging.getLogger(__name__)

class User(BaseModel):
    """User model."""
    id: str
    email: EmailStr
    username: str
    password_hash: str
    role: str
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)

class UserManager:
    def __init__(self, redis_client: redis.Redis, secret_key: str):
        self.redis = redis_client
        self.secret_key = secret_key
        self.logger = logging.getLogger(__name__)
        
    async def create_user(
        self,
        email: str,
        username: str,
        password: str,
        role: str = "user",
        metadata: Optional[Dict] = None
    ) -> User:
        """Create a new user."""
        try:
            # Check if user exists
            if await self.get_user_by_email(email):
                raise ValueError("User with this email already exists")
                
            if await self.get_user_by_username(username):
                raise ValueError("User with this username already exists")
                
            # Hash password
            password_hash = bcrypt.hashpw(
                password.encode(),
                bcrypt.gensalt()
            ).decode()
            
            # Create user
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                username=username,
                password_hash=password_hash,
                role=role,
                created_at=datetime.now(),
                metadata=metadata or {}
            )
            
            # Store in Redis
            await self.redis.set(
                f"user:{user.id}",
                user.json()
            )
            
            # Create indexes
            await self.redis.set(
                f"user:email:{email}",
                user.id
            )
            await self.redis.set(
                f"user:username:{username}",
                user.id
            )
            
            return user
            
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            raise

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            user_data = await self.redis.get(f"user:{user_id}")
            if not user_data:
                return None
                
            return User.parse_raw(user_data)
            
        except Exception as e:
            self.logger.error(f"Error getting user: {str(e)}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            user_id = await self.redis.get(f"user:email:{email}")
            if not user_id:
                return None
                
            return await self.get_user(user_id)
            
        except Exception as e:
            self.logger.error(f"Error getting user by email: {str(e)}")
            return None

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            user_id = await self.redis.get(f"user:username:{username}")
            if not user_id:
                return None
                
            return await self.get_user(user_id)
            
        except Exception as e:
            self.logger.error(f"Error getting user by username: {str(e)}")
            return None

    async def update_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[User]:
        """Update user."""
        try:
            # Get user
            user = await self.get_user(user_id)
            if not user:
                return None
                
            # Update fields
            if email is not None:
                # Check if email is taken
                existing_user = await self.get_user_by_email(email)
                if existing_user and existing_user.id != user_id:
                    raise ValueError("Email already taken")
                user.email = email
                
            if username is not None:
                # Check if username is taken
                existing_user = await self.get_user_by_username(username)
                if existing_user and existing_user.id != user_id:
                    raise ValueError("Username already taken")
                user.username = username
                
            if password is not None:
                user.password_hash = bcrypt.hashpw(
                    password.encode(),
                    bcrypt.gensalt()
                ).decode()
                
            if role is not None:
                user.role = role
                
            if is_active is not None:
                user.is_active = is_active
                
            if metadata is not None:
                user.metadata.update(metadata)
                
            # Store in Redis
            await self.redis.set(
                f"user:{user.id}",
                user.json()
            )
            
            # Update indexes
            if email is not None:
                await self.redis.set(
                    f"user:email:{email}",
                    user.id
                )
                
            if username is not None:
                await self.redis.set(
                    f"user:username:{username}",
                    user.id
                )
                
            return user
            
        except Exception as e:
            self.logger.error(f"Error updating user: {str(e)}")
            raise

    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        try:
            # Get user
            user = await self.get_user(user_id)
            if not user:
                return False
                
            # Delete from Redis
            await self.redis.delete(f"user:{user_id}")
            await self.redis.delete(f"user:email:{user.email}")
            await self.redis.delete(f"user:username:{user.username}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting user: {str(e)}")
            return False

    async def verify_password(self, user_id: str, password: str) -> bool:
        """Verify user password."""
        try:
            # Get user
            user = await self.get_user(user_id)
            if not user:
                return False
                
            # Verify password
            return bcrypt.checkpw(
                password.encode(),
                user.password_hash.encode()
            )
            
        except Exception as e:
            self.logger.error(f"Error verifying password: {str(e)}")
            return False

    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        try:
            # Get user
            user = await self.get_user(user_id)
            if not user:
                return
                
            # Update last login
            user.last_login = datetime.now()
            
            # Store in Redis
            await self.redis.set(
                f"user:{user.id}",
                user.json()
            )
            
        except Exception as e:
            self.logger.error(f"Error updating last login: {str(e)}")

    async def list_users(
        self,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[User]:
        """List users with optional filtering."""
        try:
            # Get all user IDs
            user_ids = await self.redis.keys("user:*")
            user_ids = [uid.decode().split(":")[1] for uid in user_ids]
            
            # Get users
            users = []
            for user_id in user_ids[offset:offset + limit]:
                user = await self.get_user(user_id)
                if not user:
                    continue
                    
                # Apply filters
                if role and user.role != role:
                    continue
                if is_active is not None and user.is_active != is_active:
                    continue
                    
                users.append(user)
                
            return users
            
        except Exception as e:
            self.logger.error(f"Error listing users: {str(e)}")
            return []
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return user data with timing attack protection."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password with timing attack protection."""
        try:
            # Use hmac.compare_digest for constant-time comparison
            return hmac.compare_digest(
                hashlib.sha256(password.encode()).hexdigest(),
                hashed_password
            )
        except Exception:
            return False
    
    def verify_api_key(self, provided_key: str, stored_key: str) -> bool:
        """Verify API key with timing attack protection."""
        try:
            # Use hmac.compare_digest for constant-time comparison
            return hmac.compare_digest(provided_key, stored_key)
        except Exception:
            return False