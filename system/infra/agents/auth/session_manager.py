"""
Session Manager for handling user sessions and tokens.

This module provides functionality for:
- Session creation and management
- Token generation and validation
- Session expiration and cleanup
- Rate limiting
"""

import logging
import uuid
from dataclasses import field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import jwt
import redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Session(BaseModel):
    """Session model."""

    id: str
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    metadata: Dict = field(default_factory=dict)


class SessionManager:
    def __init__(
        self,
        redis_client: redis.Redis,
        secret_key: str,
        token_expiry: int = 3600,  # 1 hour
        session_expiry: int = 86400,  # 24 hours
        max_sessions_per_user: int = 5,
    ):
        self.redis = redis_client
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.session_expiry = session_expiry
        self.max_sessions_per_user = max_sessions_per_user
        self.logger = logging.getLogger(__name__)

    async def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        metadata: Optional[Dict] = None,
    ) -> Session:
        """Create a new session for a user."""
        try:
            # Check session limit
            active_sessions = await self.get_user_sessions(user_id)
            if len(active_sessions) >= self.max_sessions_per_user:
                # Remove oldest session
                oldest_session = min(active_sessions, key=lambda s: s.created_at)
                await self.delete_session(oldest_session.id)

            # Generate session ID and token
            session_id = str(uuid.uuid4())
            token = self._generate_token(user_id)

            # Create session
            now = datetime.now()
            session = Session(
                id=session_id,
                user_id=user_id,
                token=token,
                created_at=now,
                expires_at=now + timedelta(seconds=self.session_expiry),
                last_activity=now,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata or {},
            )

            # Store in Redis
            await self.redis.set(
                f"session:{session_id}", session.json(), ex=self.session_expiry
            )

            # Add to user's sessions
            await self.redis.sadd(f"user:sessions:{user_id}", session_id)

            return session

        except Exception as e:
            self.logger.error(f"Error creating session: {str(e)}")
            raise

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        try:
            session_data = await self.redis.get(f"session:{session_id}")
            if not session_data:
                return None

            return Session.parse_raw(session_data)

        except Exception as e:
            self.logger.error(f"Error getting session: {str(e)}")
            return None

    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        try:
            session_ids = await self.redis.smembers(f"user:sessions:{user_id}")
            sessions = []

            for session_id in session_ids:
                session = await self.get_session(session_id)
                if session:
                    sessions.append(session)

            return sessions

        except Exception as e:
            self.logger.error(f"Error getting user sessions: {str(e)}")
            return []

    async def validate_session(self, session_id: str) -> bool:
        """Validate a session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False

            # Check expiration
            if datetime.now() > session.expires_at:
                await self.delete_session(session_id)
                return False

            # Update last activity
            session.last_activity = datetime.now()
            await self.redis.set(
                f"session:{session_id}", session.json(), ex=self.session_expiry
            )

            return True

        except Exception as e:
            self.logger.error(f"Error validating session: {str(e)}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False

            # Remove from Redis
            await self.redis.delete(f"session:{session_id}")
            await self.redis.srem(f"user:sessions:{session.user_id}", session_id)

            return True

        except Exception as e:
            self.logger.error(f"Error deleting session: {str(e)}")
            return False

    async def delete_user_sessions(self, user_id: str) -> bool:
        """Delete all sessions for a user."""
        try:
            session_ids = await self.redis.smembers(f"user:sessions:{user_id}")
            for session_id in session_ids:
                await self.delete_session(session_id)

            return True

        except Exception as e:
            self.logger.error(f"Error deleting user sessions: {str(e)}")
            return False

    def _generate_token(self, user_id: str) -> str:
        """Generate a JWT token."""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify a JWT token."""
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return None
