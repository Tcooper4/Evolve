import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from pathlib import Path
import json
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from jose import JWTError, jwt
from passlib.context import CryptContext
from ratelimit import limits, sleep_and_retry
import uvicorn

from system.infra.agents.core.models.task import Task, TaskStatus, TaskPriority, TaskType
from trading.automation_core import AutomationCore
from trading.automation_tasks import AutomationTasks
from trading.automation_workflows import AutomationWorkflows
from trading.automation_monitoring import AutomationMonitoring
from trading.automation_notification import AutomationNotification
from trading.automation_scheduler import AutomationScheduler

logger = logging.getLogger(__name__)

class APIConfig(BaseModel):
    """Configuration for API."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    jwt_secret: str
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=30)
    cors_origins: List[str] = Field(default=["*"])
    cors_methods: List[str] = Field(default=["*"])
    cors_headers: List[str] = Field(default=["*"])
    rate_limit_calls: int = Field(default=100)
    rate_limit_period: int = Field(default=60)
    
    @validator('jwt_secret')
    def validate_jwt_secret(cls, v):
        if not v:
            raise ValueError("JWT secret is required")
        return v

class User(BaseModel):
    """User model."""
    username: str
    email: str
    full_name: str
    disabled: bool = False
    scopes: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class UserInDB(User):
    """User model with password."""
    hashed_password: str

class Token(BaseModel):
    """Token model."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: List[str] = []

class AutomationAPI:
    """API functionality."""
    
    def __init__(
        self,
        core: AutomationCore,
        tasks: AutomationTasks,
        workflows: AutomationWorkflows,
        monitoring: AutomationMonitoring,
        notification: AutomationNotification,
        scheduler: AutomationScheduler,
        config_path: str = "automation/config/api.json"
    ):
        """Initialize API."""
        self.core = core
        self.tasks = tasks
        self.workflows = workflows
        self.monitoring = monitoring
        self.notification = notification
        self.scheduler = scheduler
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_security()
        self.setup_app()
        self.users: Dict[str, UserInDB] = {}
        self.lock = asyncio.Lock()
        
    def _load_config(self, config_path: str) -> APIConfig:
        """Load API configuration."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return APIConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load API config: {str(e)}")
            raise
            
    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "api.log"),
                logging.StreamHandler()
            ]
        )
        
    def setup_security(self):
        """Setup security components."""
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
    def setup_app(self):
        """Setup FastAPI application."""
        self.app = FastAPI(
            title="Automation API",
            description="API for automation system",
            version="1.0.0",
            docs_url=None,
            redoc_url=None
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=self.config.cors_methods,
            allow_headers=self.config.cors_headers
        )
        
        # Add routes
        self._add_routes()
        
    def _add_routes(self):
        """Add API routes."""
        
        @self.app.post("/token", response_model=Token)
        async def login(form_data: OAuth2PasswordRequestForm = Depends()):
            """Login endpoint."""
            user = await self._authenticate_user(
                form_data.username,
                form_data.password
            )
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"}
                )
                
            access_token = self._create_access_token(
                data={"sub": user.username, "scopes": user.scopes}
            )
            return {"access_token": access_token, "token_type": "bearer"}
            
        @self.app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            """Custom Swagger UI."""
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title="Automation API - Swagger UI",
                oauth2_redirect_url="/docs/oauth2-redirect",
                swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-bundle.js",
                swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css"
            )
            
        @self.app.get("/openapi.json", include_in_schema=False)
        async def get_openapi_endpoint():
            """OpenAPI schema."""
            return get_openapi(
                title="Automation API",
                version="1.0.0",
                description="API for automation system",
                routes=self.app.routes
            )
            
        @self.app.get("/tasks", response_model=List[Task])
        async def get_tasks(
            status: Optional[TaskStatus] = None,
            task_type: Optional[TaskType] = None,
            current_user: User = Depends(self._get_current_user)
        ):
            """Get tasks endpoint."""
            if "tasks:read" not in current_user.scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions"
                )
                
            return await self.tasks.list_tasks(status, task_type)
            
        @self.app.post("/tasks", response_model=Task)
        async def create_task(
            task: Task,
            current_user: User = Depends(self._get_current_user)
        ):
            """Create task endpoint."""
            if "tasks:write" not in current_user.scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions"
                )
                
            return await self.tasks.schedule_task(
                name=task.name,
                description=task.description,
                task_type=task.type,
                parameters=task.parameters,
                priority=task.priority
            )
            
        @self.app.get("/workflows")
        async def get_workflows(
            status: Optional[str] = None,
            current_user: User = Depends(self._get_current_user)
        ):
            """Get workflows endpoint."""
            if "workflows:read" not in current_user.scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions"
                )
                
            return await self.workflows.get_workflows(status)
            
        @self.app.post("/workflows")
        async def create_workflow(
            workflow: dict,
            current_user: User = Depends(self._get_current_user)
        ):
            """Create workflow endpoint."""
            if "workflows:write" not in current_user.scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions"
                )
                
            return await self.workflows.create_workflow(
                name=workflow["name"],
                description=workflow["description"],
                steps=workflow["steps"]
            )
            
        @self.app.get("/metrics")
        async def get_metrics(
            current_user: User = Depends(self._get_current_user)
        ):
            """Get metrics endpoint."""
            if "metrics:read" not in current_user.scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions"
                )
                
            return await self.monitoring.get_metrics()
            
        @self.app.get("/schedules")
        async def get_schedules(
            enabled: Optional[bool] = None,
            task_type: Optional[TaskType] = None,
            current_user: User = Depends(self._get_current_user)
        ):
            """Get schedules endpoint."""
            if "schedules:read" not in current_user.scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions"
                )
                
            return await self.scheduler.get_schedules(enabled, task_type)
            
        @self.app.post("/schedules")
        async def create_schedule(
            schedule: dict,
            current_user: User = Depends(self._get_current_user)
        ):
            """Create schedule endpoint."""
            if "schedules:write" not in current_user.scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions"
                )
                
            return await self.scheduler.create_schedule(
                name=schedule["name"],
                description=schedule["description"],
                task_type=schedule["task_type"],
                parameters=schedule["parameters"],
                cron_expression=schedule.get("cron_expression"),
                interval_seconds=schedule.get("interval_seconds"),
                start_date=schedule.get("start_date"),
                end_date=schedule.get("end_date"),
                timezone=schedule.get("timezone", "UTC")
            )
            
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """Global exception handler."""
            logger.error(f"Unhandled exception: {str(exc)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
            
    async def _authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[UserInDB]:
        """Authenticate user."""
        user = self.users.get(username)
        if not user:
            return None
            
        if not self._verify_password(password, user.hashed_password):
            return None
            
        return user
        
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password."""
        return self.pwd_context.verify(plain_password, hashed_password)
        
    def _get_password_hash(self, password: str) -> str:
        """Get password hash."""
        return self.pwd_context.hash(password)
        
    def _create_access_token(self, data: dict) -> str:
        """Create access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(
            minutes=self.config.jwt_expire_minutes
        )
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )
        
    async def _get_current_user(
        self,
        token: str = Depends(oauth2_scheme)
    ) -> User:
        """Get current user from token."""
        credentials_exception = HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
                
            token_data = TokenData(
                username=username,
                scopes=payload.get("scopes", [])
            )
            
        except JWTError:
            raise credentials_exception
            
        user = self.users.get(token_data.username)
        if user is None:
            raise credentials_exception
            
        return user
        
    async def start(self):
        """Start the API server."""
        try:
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug
            )
        except Exception as e:
            logger.error(f"Failed to start API server: {str(e)}")
            raise
            
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear users
            self.users.clear()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise 