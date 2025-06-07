import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import re
from datetime import datetime
import json
import yaml
import asyncio
import aiohttp
from dataclasses import dataclass
import jwt
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import bcrypt
import redis
import websockets
import socketio

@dataclass
class User:
    id: str
    username: str
    email: str
    role: str
    permissions: List[str]
    last_active: str

@dataclass
class Comment:
    id: str
    user_id: str
    content: str
    timestamp: str
    file_path: str
    line_number: int
    resolved: bool

@dataclass
class Review:
    id: str
    user_id: str
    status: str
    comments: List[Comment]
    timestamp: str
    file_path: str

class DocumentationCollaboration:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.collab_config = config.get('documentation', {}).get('collaboration', {})
        self.setup_authentication()
        self.setup_websocket()
        self.setup_redis()
        self.users: Dict[str, User] = {}
        self.comments: Dict[str, Comment] = {}
        self.reviews: Dict[str, Review] = {}

    def setup_logging(self):
        """Configure logging for the documentation collaboration system."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "collaboration.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_authentication(self):
        """Setup authentication system."""
        self.jwt_secret = self.collab_config.get('jwt_secret', '')
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
        # Load initial users
        users_config = self.collab_config.get('users', {})
        for user_id, user_data in users_config.items():
            self.users[user_id] = User(
                id=user_id,
                username=user_data['username'],
                email=user_data['email'],
                role=user_data['role'],
                permissions=user_data['permissions'],
                last_active=datetime.now().isoformat()
            )

    def setup_websocket(self):
        """Setup WebSocket server for real-time collaboration."""
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins='*'
        )
        self.app = FastAPI()
        self.sio.attach(self.app)
        
        @self.sio.event
        async def connect(sid, environ):
            self.logger.info(f"Client connected: {sid}")
        
        @self.sio.event
        async def disconnect(sid):
            self.logger.info(f"Client disconnected: {sid}")
        
        @self.sio.event
        async def join_document(sid, data):
            document_id = data.get('document_id')
            if document_id:
                self.sio.enter_room(sid, document_id)
                self.logger.info(f"Client {sid} joined document: {document_id}")
        
        @self.sio.event
        async def leave_document(sid, data):
            document_id = data.get('document_id')
            if document_id:
                self.sio.leave_room(sid, document_id)
                self.logger.info(f"Client {sid} left document: {document_id}")
        
        @self.sio.event
        async def edit_document(sid, data):
            document_id = data.get('document_id')
            changes = data.get('changes')
            if document_id and changes:
                await self.sio.emit(
                    'document_updated',
                    {'changes': changes},
                    room=document_id,
                    skip_sid=sid
                )
                self.logger.info(f"Document {document_id} updated by {sid}")

    def setup_redis(self):
        """Setup Redis for real-time collaboration state."""
        redis_config = self.collab_config.get('redis', {})
        self.redis = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            decode_responses=True
        )

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str = 'user',
        permissions: Optional[List[str]] = None
    ) -> User:
        """Create a new user."""
        try:
            # Generate user ID
            user_id = f"user_{len(self.users) + 1}"
            
            # Hash password
            password_hash = bcrypt.hashpw(
                password.encode('utf-8'),
                bcrypt.gensalt()
            ).decode('utf-8')
            
            # Create user
            user = User(
                id=user_id,
                username=username,
                email=email,
                role=role,
                permissions=permissions or [],
                last_active=datetime.now().isoformat()
            )
            
            # Store user
            self.users[user_id] = user
            await self.redis.hset(
                f"user:{user_id}",
                mapping={
                    'username': username,
                    'email': email,
                    'role': role,
                    'permissions': json.dumps(permissions or []),
                    'password_hash': password_hash
                }
            )
            
            self.logger.info(f"Created user: {user_id}")
            return user
            
        except Exception as e:
            self.logger.error(f"Failed to create user: {str(e)}")
            raise

    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[str]:
        """Authenticate user and return JWT token."""
        try:
            # Find user
            user = next(
                (u for u in self.users.values() if u.username == username),
                None
            )
            if not user:
                return None
            
            # Get password hash
            password_hash = await self.redis.hget(f"user:{user.id}", 'password_hash')
            if not password_hash:
                return None
            
            # Verify password
            if not bcrypt.checkpw(
                password.encode('utf-8'),
                password_hash.encode('utf-8')
            ):
                return None
            
            # Generate token
            token = jwt.encode(
                {
                    'sub': user.id,
                    'username': user.username,
                    'role': user.role,
                    'exp': datetime.utcnow().timestamp() + 3600
                },
                self.jwt_secret,
                algorithm='HS256'
            )
            
            # Update last active
            user.last_active = datetime.now().isoformat()
            await self.redis.hset(
                f"user:{user.id}",
                'last_active',
                user.last_active
            )
            
            self.logger.info(f"User authenticated: {user.id}")
            return token
            
        except Exception as e:
            self.logger.error(f"Failed to authenticate user: {str(e)}")
            raise

    async def add_comment(
        self,
        user_id: str,
        content: str,
        file_path: str,
        line_number: int
    ) -> Comment:
        """Add a comment to documentation."""
        try:
            # Generate comment ID
            comment_id = f"comment_{len(self.comments) + 1}"
            
            # Create comment
            comment = Comment(
                id=comment_id,
                user_id=user_id,
                content=content,
                timestamp=datetime.now().isoformat(),
                file_path=file_path,
                line_number=line_number,
                resolved=False
            )
            
            # Store comment
            self.comments[comment_id] = comment
            await self.redis.hset(
                f"comment:{comment_id}",
                mapping={
                    'user_id': user_id,
                    'content': content,
                    'timestamp': comment.timestamp,
                    'file_path': file_path,
                    'line_number': line_number,
                    'resolved': 'false'
                }
            )
            
            # Notify users
            await self.sio.emit(
                'new_comment',
                {
                    'comment_id': comment_id,
                    'file_path': file_path,
                    'line_number': line_number
                },
                room=file_path
            )
            
            self.logger.info(f"Added comment: {comment_id}")
            return comment
            
        except Exception as e:
            self.logger.error(f"Failed to add comment: {str(e)}")
            raise

    async def resolve_comment(self, comment_id: str):
        """Resolve a comment."""
        try:
            # Get comment
            comment = self.comments.get(comment_id)
            if not comment:
                raise ValueError(f"Comment not found: {comment_id}")
            
            # Update comment
            comment.resolved = True
            await self.redis.hset(
                f"comment:{comment_id}",
                'resolved',
                'true'
            )
            
            # Notify users
            await self.sio.emit(
                'comment_resolved',
                {'comment_id': comment_id},
                room=comment.file_path
            )
            
            self.logger.info(f"Resolved comment: {comment_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to resolve comment: {str(e)}")
            raise

    async def create_review(
        self,
        user_id: str,
        file_path: str
    ) -> Review:
        """Create a review for documentation."""
        try:
            # Generate review ID
            review_id = f"review_{len(self.reviews) + 1}"
            
            # Create review
            review = Review(
                id=review_id,
                user_id=user_id,
                status='pending',
                comments=[],
                timestamp=datetime.now().isoformat(),
                file_path=file_path
            )
            
            # Store review
            self.reviews[review_id] = review
            await self.redis.hset(
                f"review:{review_id}",
                mapping={
                    'user_id': user_id,
                    'status': 'pending',
                    'comments': '[]',
                    'timestamp': review.timestamp,
                    'file_path': file_path
                }
            )
            
            # Notify users
            await self.sio.emit(
                'new_review',
                {'review_id': review_id, 'file_path': file_path},
                room=file_path
            )
            
            self.logger.info(f"Created review: {review_id}")
            return review
            
        except Exception as e:
            self.logger.error(f"Failed to create review: {str(e)}")
            raise

    async def update_review_status(
        self,
        review_id: str,
        status: str
    ):
        """Update review status."""
        try:
            # Get review
            review = self.reviews.get(review_id)
            if not review:
                raise ValueError(f"Review not found: {review_id}")
            
            # Update review
            review.status = status
            await self.redis.hset(
                f"review:{review_id}",
                'status',
                status
            )
            
            # Notify users
            await self.sio.emit(
                'review_updated',
                {
                    'review_id': review_id,
                    'status': status
                },
                room=review.file_path
            )
            
            self.logger.info(f"Updated review status: {review_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update review status: {str(e)}")
            raise

    async def add_comment_to_review(
        self,
        review_id: str,
        comment_id: str
    ):
        """Add a comment to a review."""
        try:
            # Get review and comment
            review = self.reviews.get(review_id)
            comment = self.comments.get(comment_id)
            
            if not review:
                raise ValueError(f"Review not found: {review_id}")
            if not comment:
                raise ValueError(f"Comment not found: {comment_id}")
            
            # Add comment to review
            review.comments.append(comment)
            comments_json = json.dumps([c.id for c in review.comments])
            await self.redis.hset(
                f"review:{review_id}",
                'comments',
                comments_json
            )
            
            # Notify users
            await self.sio.emit(
                'review_comment_added',
                {
                    'review_id': review_id,
                    'comment_id': comment_id
                },
                room=review.file_path
            )
            
            self.logger.info(f"Added comment to review: {review_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add comment to review: {str(e)}")
            raise

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_users(
        self,
        role: Optional[str] = None,
        active_only: bool = False
    ) -> List[User]:
        """Get users with optional filtering."""
        users = list(self.users.values())
        
        if role:
            users = [u for u in users if u.role == role]
        if active_only:
            users = [u for u in users if (
                datetime.now() - datetime.fromisoformat(u.last_active)
            ).total_seconds() < 3600]
        
        return users

    def get_comments(
        self,
        file_path: Optional[str] = None,
        user_id: Optional[str] = None,
        resolved: Optional[bool] = None
    ) -> List[Comment]:
        """Get comments with optional filtering."""
        comments = list(self.comments.values())
        
        if file_path:
            comments = [c for c in comments if c.file_path == file_path]
        if user_id:
            comments = [c for c in comments if c.user_id == user_id]
        if resolved is not None:
            comments = [c for c in comments if c.resolved == resolved]
        
        return comments

    def get_reviews(
        self,
        file_path: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Review]:
        """Get reviews with optional filtering."""
        reviews = list(self.reviews.values())
        
        if file_path:
            reviews = [r for r in reviews if r.file_path == file_path]
        if user_id:
            reviews = [r for r in reviews if r.user_id == user_id]
        if status:
            reviews = [r for r in reviews if r.status == status]
        
        return reviews

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the collaboration server."""
        try:
            import uvicorn
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {str(e)}")
            raise 