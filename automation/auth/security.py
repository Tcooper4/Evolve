import logging
import jwt
import time
import pyotp
import qrcode
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from fastapi import HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
import redis
from .user_manager import UserManager
import json

logger = logging.getLogger(__name__)

class SecurityConfig(BaseModel):
    """Security configuration model."""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30
    mfa_issuer: str = "Automation System"
    oauth2_clients: Dict[str, Dict[str, str]] = {}
    role_permissions: Dict[str, List[str]] = {
        "admin": ["*"],
        "manager": ["read:*", "write:*", "delete:own"],
        "user": ["read:own", "write:own"]
    }

class MFASecret(BaseModel):
    """MFA secret model."""
    user_id: str
    secret: str
    created_at: datetime
    last_used: Optional[datetime] = None
    backup_codes: List[str] = []

class OAuth2Client(BaseModel):
    """OAuth2 client model."""
    client_id: str
    client_secret: str
    redirect_uris: List[str]
    grant_types: List[str]
    scopes: List[str]
    user_id: Optional[str] = None

class SecurityManager:
    def __init__(self, redis_client: redis.Redis, config: SecurityConfig):
        """Initialize security manager."""
        self.redis = redis_client
        self.config = config
        self.user_manager = UserManager(redis_client)
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.logger = logging.getLogger(__name__)

    async def create_mfa_secret(self, user_id: str) -> Dict:
        """Create MFA secret for a user."""
        try:
            # Generate secret
            secret = pyotp.random_base32()
            
            # Generate backup codes
            backup_codes = [pyotp.random_base32()[:8] for _ in range(8)]
            
            # Create MFA secret
            mfa_secret = MFASecret(
                user_id=user_id,
                secret=secret,
                created_at=datetime.now(),
                backup_codes=backup_codes
            )
            
            # Store in Redis
            await self.redis.set(
                f"mfa:{user_id}",
                mfa_secret.json(),
                ex=3600  # Expire after 1 hour
            )
            
            # Generate QR code
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                user_id,
                issuer_name=self.config.mfa_issuer
            )
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            qr_image = qr.make_image(fill_color="black", back_color="white")
            
            return {
                "secret": secret,
                "backup_codes": backup_codes,
                "qr_code": qr_image,
                "provisioning_uri": provisioning_uri
            }
            
        except Exception as e:
            self.logger.error(f"Error creating MFA secret: {str(e)}")
            raise

    async def verify_mfa(self, user_id: str, token: str) -> bool:
        """Verify MFA token."""
        try:
            # Get MFA secret
            mfa_data = await self.redis.get(f"mfa:{user_id}")
            if not mfa_data:
                return False
                
            mfa_secret = MFASecret.parse_raw(mfa_data)
            
            # Verify token
            totp = pyotp.TOTP(mfa_secret.secret)
            if totp.verify(token):
                # Update last used
                mfa_secret.last_used = datetime.now()
                await self.redis.set(
                    f"mfa:{user_id}",
                    mfa_secret.json(),
                    ex=3600
                )
                return True
                
            # Check backup codes
            if token in mfa_secret.backup_codes:
                mfa_secret.backup_codes.remove(token)
                mfa_secret.last_used = datetime.now()
                await self.redis.set(
                    f"mfa:{user_id}",
                    mfa_secret.json(),
                    ex=3600
                )
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error verifying MFA: {str(e)}")
            return False

    async def create_oauth2_client(
        self,
        client_id: str,
        redirect_uris: List[str],
        grant_types: List[str],
        scopes: List[str],
        user_id: Optional[str] = None
    ) -> OAuth2Client:
        """Create OAuth2 client."""
        try:
            # Generate client secret
            client_secret = pyotp.random_base32()
            
            # Create client
            client = OAuth2Client(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uris=redirect_uris,
                grant_types=grant_types,
                scopes=scopes,
                user_id=user_id
            )
            
            # Store in Redis
            await self.redis.set(
                f"oauth2:client:{client_id}",
                client.json()
            )
            
            return client
            
        except Exception as e:
            self.logger.error(f"Error creating OAuth2 client: {str(e)}")
            raise

    async def verify_oauth2_token(self, token: str) -> Dict:
        """Verify OAuth2 token."""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check expiry
            if payload["exp"] < time.time():
                raise HTTPException(
                    status_code=401,
                    detail="Token has expired"
                )
                
            return payload
            
        except jwt.InvalidTokenError as e:
            self.logger.error(f"Invalid OAuth2 token: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )

    async def create_access_token(
        self,
        user_id: str,
        scopes: List[str],
        client_id: Optional[str] = None
    ) -> str:
        """Create access token."""
        try:
            # Create payload
            payload = {
                "sub": user_id,
                "scopes": scopes,
                "client_id": client_id,
                "exp": time.time() + (self.config.jwt_expiry_minutes * 60)
            }
            
            # Create token
            token = jwt.encode(
                payload,
                self.config.jwt_secret,
                algorithm=self.config.jwt_algorithm
            )
            
            return token
            
        except Exception as e:
            self.logger.error(f"Error creating access token: {str(e)}")
            raise

    async def check_permission(
        self,
        user_id: str,
        permission: str,
        resource_id: Optional[str] = None
    ) -> bool:
        """Check if user has permission."""
        try:
            # Get user
            user = await self.user_manager.get_user(user_id)
            if not user:
                return False
                
            # Get role permissions
            role_permissions = self.config.role_permissions.get(user.role, [])
            
            # Check wildcard permission
            if "*" in role_permissions:
                return True
                
            # Check specific permission
            if permission in role_permissions:
                return True
                
            # Check resource-specific permission
            if resource_id and f"{permission}:{resource_id}" in role_permissions:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking permission: {str(e)}")
            return False

    async def audit_log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict] = None
    ) -> None:
        """Create audit log entry."""
        try:
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": details or {}
            }
            
            # Store in Redis
            await self.redis.lpush(
                "audit_logs",
                json.dumps(log_entry)
            )
            
            # Trim logs to last 1000 entries
            await self.redis.ltrim("audit_logs", 0, 999)
            
        except Exception as e:
            self.logger.error(f"Error creating audit log: {str(e)}")

    async def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get audit logs with optional filtering."""
        try:
            # Get logs
            logs = await self.redis.lrange("audit_logs", offset, offset + limit - 1)
            
            # Parse and filter logs
            filtered_logs = []
            for log in logs:
                log_entry = json.loads(log)
                
                # Apply filters
                if user_id and log_entry["user_id"] != user_id:
                    continue
                if action and log_entry["action"] != action:
                    continue
                if resource_type and log_entry["resource_type"] != resource_type:
                    continue
                if resource_id and log_entry["resource_id"] != resource_id:
                    continue
                    
                filtered_logs.append(log_entry)
                
            return filtered_logs
            
        except Exception as e:
            self.logger.error(f"Error getting audit logs: {str(e)}")
            return [] 