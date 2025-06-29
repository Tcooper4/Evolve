"""
API Service

Implements HTTP API endpoints and authentication functionality.
Adapted from legacy automation/services/automation_api.py.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import jwt
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class Token(BaseModel):
    """Token model."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None

class APIService:
    """Manages API endpoints and authentication."""
    
    def __init__(self, config_path: str = "config/api.json"):
        """Initialize API service."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.setup_app()
        self.setup_routes()
        self.setup_middleware()
        
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/api")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "api_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
    
    def setup_app(self) -> None:
        """Set up FastAPI application."""
        self.app = FastAPI(
            title="Meta-Agents API",
            description="API for Meta-Agents system",
            version="1.0.0"
        )
        
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    def setup_middleware(self) -> None:
        """Set up middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self) -> None:
        """Set up API routes."""
        
        @self.app.post("/token", response_model=Token)
        async def login(form_data: OAuth2PasswordRequestForm = Depends()):
            """Login endpoint."""
            try:
                user = await self.verify_credentials(form_data.username, form_data.password)
                if not user:
                    raise HTTPException(
                        status_code=401,
                        detail="Incorrect username or password",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                access_token = self.generate_token(user)
                return {"access_token": access_token, "token_type": "bearer"}
            except Exception as e:
                self.logger.error(f"Login error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/users/me", response_model=User)
        async def read_users_me(token: str = Depends(self.oauth2_scheme)):
            """Get current user."""
            try:
                user = await self.get_user(token)
                if not user:
                    raise HTTPException(status_code=401, detail="Invalid token")
                return user
            except Exception as e:
                self.logger.error(f"Get user error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    def verify_credentials(self, username: str, password: str) -> Optional[User]:
        """Verify user credentials."""
        raise NotImplementedError('Pending feature')
    
    def generate_token(self, user: User) -> str:
        """Generate JWT token."""
        try:
            expiration = datetime.utcnow() + timedelta(minutes=15)
            data = {
                "sub": user.username,
                "exp": expiration
            }
            return jwt.encode(data, self.config["secret_key"], algorithm="HS256")
        except Exception as e:
            self.logger.error(f"Token generation error: {str(e)}")
            raise
    
    def get_user(self, token: str) -> Optional[User]:
        """Get user from token."""
        raise NotImplementedError('Pending feature')
    
    async def start(self) -> None:
        """Start API service."""
        try:
            config = uvicorn.Config(
                self.app,
                host=self.config["host"],
                port=self.config["port"],
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            self.logger.error(f"Error starting API service: {str(e)}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='API service')
    parser.add_argument('--config', default="config/api.json", help='Path to config file')
    args = parser.parse_args()
    
    try:
        service = APIService(args.config)
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logging.info("API service interrupted")
    except Exception as e:
        logging.error(f"Error in API service: {str(e)}")
        raise

if __name__ == '__main__':
    main() 