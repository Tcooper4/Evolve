import asyncio
import base64
import ipaddress
import json
import logging
import re
import secrets
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import jwt
from cachetools import TTLCache
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator
from ratelimit import limits, sleep_and_retry

logger = logging.getLogger(__name__)


class SecurityConfig(BaseModel):
    """Configuration for security."""

    jwt_secret: str
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=30)
    password_min_length: int = Field(default=8)
    password_require_uppercase: bool = Field(default=True)
    password_require_lowercase: bool = Field(default=True)
    password_require_digits: bool = Field(default=True)
    password_require_special: bool = Field(default=True)
    max_login_attempts: int = Field(default=5)
    lockout_duration_minutes: int = Field(default=15)
    allowed_ips: List[str] = []
    rate_limit_calls: int = Field(default=100)
    rate_limit_period: int = Field(default=60)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)

    @validator("jwt_secret")
    def validate_jwt_secret(cls, v):
        if not v:
            raise ValueError("JWT secret is required")
        return v

    @validator("allowed_ips")
    def validate_ips(cls, v):
        for ip in v:
            try:
                ipaddress.ip_network(ip)
            except ValueError:
                raise ValueError(f"Invalid IP address: {ip}")
        return v


class User(BaseModel):
    """User model."""

    username: str
    email: str
    full_name: str
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    scopes: List[str] = []
    failed_login_attempts: int = 0
    last_login_attempt: Optional[datetime] = None
    locked_until: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Token(BaseModel):
    """Token model."""

    access_token: str
    token_type: str
    expires_at: datetime


class TokenData(BaseModel):
    """Token data model."""

    username: Optional[str] = None
    scopes: List[str] = []


class AutomationSecurity:
    """Security functionality."""

    def __init__(self, config_path: str = "automation/config/security.json"):
        """Initialize security system."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_crypto()
        self.setup_cache()
        self.users: Dict[str, User] = {}
        self.lock = asyncio.Lock()

    def _load_config(self, config_path: str) -> SecurityConfig:
        """Load security configuration."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return SecurityConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load security config: {str(e)}")
            raise

    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_path / "security.log"), logging.StreamHandler()],
        )

    def setup_crypto(self):
        """Setup cryptographic components."""
        try:
            # Setup password hashing
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

            # Generate RSA key pair
            self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            self.public_key = self.private_key.public_key()

        except Exception as e:
            logger.error(f"Failed to setup crypto: {str(e)}")
            raise

    def setup_cache(self):
        """Setup security caching."""
        self.cache = TTLCache(maxsize=self.config.cache_size, ttl=self.config.cache_ttl)

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def create_user(
        self,
        username: str,
        email: str,
        full_name: str,
        password: str,
        is_superuser: bool = False,
        scopes: List[str] = [],
    ) -> User:
        """Create a new user."""
        try:
            # Validate password
            self._validate_password(password)

            # Check if user exists
            if username in self.users:
                raise ValueError(f"User {username} already exists")

            # Create user
            user = User(
                username=username,
                email=email,
                full_name=full_name,
                hashed_password=self._hash_password(password),
                is_superuser=is_superuser,
                scopes=scopes,
            )

            self.users[username] = user
            return user

        except Exception as e:
            logger.error(f"Failed to create user: {str(e)}")
            raise

    def _validate_password(self, password: str):
        """Validate password strength."""
        if len(password) < self.config.password_min_length:
            raise ValueError(f"Password must be at least {self.config.password_min_length} characters")

        if self.config.password_require_uppercase and not re.search(r"[A-Z]", password):
            raise ValueError("Password must contain uppercase letters")

        if self.config.password_require_lowercase and not re.search(r"[a-z]", password):
            raise ValueError("Password must contain lowercase letters")

        if self.config.password_require_digits and not re.search(r"\d", password):
            raise ValueError("Password must contain digits")

        if self.config.password_require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValueError("Password must contain special characters")

    def _hash_password(self, password: str) -> str:
        """Hash password."""
        return self.pwd_context.hash(password)

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password."""
        return self.pwd_context.verify(plain_password, hashed_password)

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[User]:
        """Authenticate user."""
        try:
            # Check IP
            if not self._is_ip_allowed(ip_address):
                raise ValueError(f"IP address {ip_address} not allowed")

            # Get user
            user = self.users.get(username)
            if not user:
                return None

            # Check if locked
            if user.locked_until and user.locked_until > datetime.now():
                raise ValueError(f"Account locked until {user.locked_until}")

            # Verify password
            if not self._verify_password(password, user.hashed_password):
                # Update failed attempts
                user.failed_login_attempts += 1
                user.last_login_attempt = datetime.now()

                # Check if should lock
                if user.failed_login_attempts >= self.config.max_login_attempts:
                    user.locked_until = datetime.now() + timedelta(minutes=self.config.lockout_duration_minutes)

                return None

            # Reset failed attempts
            user.failed_login_attempts = 0
            user.last_login_attempt = None
            user.locked_until = None

            return user

        except Exception as e:
            logger.error(f"Failed to authenticate user: {str(e)}")
            raise

    def _is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP is allowed."""
        if not self.config.allowed_ips:
            return True

        try:
            ip = ipaddress.ip_address(ip_address)
            return any(ip in ipaddress.ip_network(allowed_ip) for allowed_ip in self.config.allowed_ips)
        except ValueError:
            return False

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def create_token(self, user: User, expires_delta: Optional[timedelta] = None) -> Token:
        """Create access token."""
        try:
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=self.config.jwt_expire_minutes)

            to_encode = {"sub": user.username, "scopes": user.scopes, "exp": expire}

            encoded_jwt = jwt.encode(to_encode, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)

            return Token(access_token=encoded_jwt, token_type="bearer", expires_at=expire)

        except Exception as e:
            logger.error(f"Failed to create token: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def verify_token(self, token: str) -> TokenData:
        """Verify access token."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])

            username: str = payload.get("sub")
            if username is None:
                raise ValueError("Invalid token")

            scopes: List[str] = payload.get("scopes", [])

            return TokenData(username=username, scopes=scopes)

        except Exception as e:
            logger.error(f"Failed to verify token: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def encrypt_data(self, data: str) -> str:
        """Encrypt data using RSA."""
        try:
            encrypted = self.public_key.encrypt(
                data.encode(),
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
            )
            return base64.b64encode(encrypted).decode()

        except Exception as e:
            logger.error(f"Failed to encrypt data: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using RSA."""
        try:
            encrypted = base64.b64decode(encrypted_data)
            decrypted = self.private_key.decrypt(
                encrypted,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
            )
            return decrypted.decode()

        except Exception as e:
            logger.error(f"Failed to decrypt data: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def generate_secret(
        self,
        length: int = 32,
        include_uppercase: bool = True,
        include_lowercase: bool = True,
        include_digits: bool = True,
        include_special: bool = True,
    ) -> str:
        """Generate secure secret."""
        try:
            chars = ""
            if include_uppercase:
                chars += string.ascii_uppercase
            if include_lowercase:
                chars += string.ascii_lowercase
            if include_digits:
                chars += string.digits
            if include_special:
                chars += string.punctuation

            if not chars:
                raise ValueError("No character sets selected")

            return "".join(secrets.choice(chars) for _ in range(length))

        except Exception as e:
            logger.error(f"Failed to generate secret: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear caches
            self.cache.clear()
            self.users.clear()

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
